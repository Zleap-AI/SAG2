"""
事项保存器 - 负责持久化

职责：
- 保存事项到数据库（MySQL）
- 同步到向量库（Elasticsearch）
- 批量处理优化
"""

import time
import struct
from typing import Any, Dict, List

from sqlalchemy import select, text, update
from sqlalchemy.orm import selectinload

from pipeline.core.ai.factory import get_embedding_client
from pipeline.core.config import get_settings
from pipeline.core.storage.elasticsearch import get_es_client
from pipeline.core.storage.repositories.entity_repository import EntityVectorRepository
from pipeline.core.storage.repositories.event_repository import EventVectorRepository
from pipeline.db import Entity, EventEntity, EventEntityEmbedding, SourceEvent, get_session_factory
from pipeline.modules.extract.config import ExtractConfig
from pipeline.utils import get_logger, is_retryable_error

logger = get_logger("extract.saver")


EVENT_ENTITY_EMBEDDING_TABLE_NAME = "event_entity_embedding"
EVENT_ENTITY_EMBEDDING_TRUNCATE_DIMS = 128


def to_event_entity_embedding_vec_bytes(embedding: List[float]) -> bytes:
    """
    将 embedding 截断为 128 维并转换为 float32 bytes（512字节）
    """
    if len(embedding) < EVENT_ENTITY_EMBEDDING_TRUNCATE_DIMS:
        raise ValueError(
            f"embedding dims too short: got={len(embedding)} "
            f"need>={EVENT_ENTITY_EMBEDDING_TRUNCATE_DIMS}"
        )
    truncated = [float(x) for x in embedding[:EVENT_ENTITY_EMBEDDING_TRUNCATE_DIMS]]
    return struct.pack(f"<{EVENT_ENTITY_EMBEDDING_TRUNCATE_DIMS}f", *truncated)


class EventSaver:
    """事项保存器 - 负责持久化"""

    def __init__(self):
        """初始化保存器"""
        self.session_factory = get_session_factory()
        self.logger = get_logger("extract.saver")
        self.settings = get_settings()

        # 向量库相关（延迟初始化）
        self._es_client = None
        self._event_repo = None
        self._entity_repo = None

    async def commit(self, events: List[SourceEvent], config: ExtractConfig) -> List[SourceEvent]:
        """
        提交事项（保存到DB + 同步向量库）

        Args:
            events: 事项列表（已完成解析，包含实体关联）
            config: 提取配置

        Returns:
            保存后的事项列表（包含完整关系数据）
        """
        if not events:
            self.logger.info("没有事项需要提交")
            return []

        self.logger.info(f"开始提交 {len(events)} 个事项")

        # 1. 保存到 MySQL
        event_ids = await self._save_to_database(events)

        # 2. 重新加载（带完整关系数据）
        fresh_events = await self._load_events_with_relations(event_ids)

        # 3. SAAS 环境写入 event_entity_embedding
        if self._should_sync_event_entity_embedding():
            await self._sync_event_entity_embeddings_to_db(fresh_events, config)
        else:
            self.logger.info(
                f"跳过 {EVENT_ENTITY_EMBEDDING_TABLE_NAME} 写入: "
                f"SERVER_TYPE={self.settings.server_type}"
            )

        # 4. 同步到向量库（如果启用）
        if config.enable_event_vector_sync:
            await self._sync_to_vector_store(fresh_events, config)

        self.logger.info(f"提交完成: {len(fresh_events)} 个事项")
        return fresh_events

    def _should_sync_event_entity_embedding(self) -> bool:
        """仅 SAAS 环境写入事项实体向量表"""
        return self.settings.server_type == "SAAS"

    def _collect_event_entities(self, events: List[SourceEvent]) -> List[EventEntity]:
        """收集事项中的所有事件-实体关联"""
        event_entities: List[EventEntity] = []
        for event in events:
            if hasattr(event, "event_associations") and event.event_associations:
                event_entities.extend(event.event_associations)
        return event_entities

    async def _sync_event_entity_embeddings_to_db(
        self, events: List[SourceEvent], config: ExtractConfig
    ) -> Dict[str, Any]:
        """
        同步事件-实体关联 embedding 到 MySQL 表 event_entity_embedding（仅 SAAS）
        """
        if not events:
            return {"total": 0, "upserted": 0}

        event_entities = self._collect_event_entities(events)
        if not event_entities:
            self.logger.info(f"没有事件-实体关联需要写入 {EVENT_ENTITY_EMBEDDING_TABLE_NAME}")
            return {"total": 0, "upserted": 0}

        start_time = time.perf_counter()
        embedding_client = await get_embedding_client(scenario="general")

        rows_to_upsert: List[Dict[str, Any]] = []
        embedding_failed = 0
        assoc_by_id = {assoc.id: assoc for assoc in event_entities}

        for i in range(0, len(event_entities), config.embedding_batch_size):
            batch = event_entities[i : i + config.embedding_batch_size]
            texts = [assoc.description or f"{assoc.event_id}-{assoc.entity_id}" for assoc in batch]

            try:
                vectors = await embedding_client.batch_generate(texts)
                expected_count = len(batch)
                actual_count = len(vectors)
                if actual_count != expected_count:
                    missing_count = max(0, expected_count - actual_count)
                    extra_count = max(0, actual_count - expected_count)
                    self.logger.error(
                        "批量事件实体向量数量不匹配: "
                        f"expected={expected_count}, actual={actual_count}, "
                        f"missing={missing_count}, extra={extra_count}; "
                        "将降级为单条重试"
                    )
                    raise ValueError(
                        "batch_generate returned mismatched vector count: "
                        f"expected={expected_count}, actual={actual_count}"
                    )
                for assoc, vector in zip(batch, vectors):
                    try:
                        rows_to_upsert.append(
                            {"id": assoc.id, "vec": to_event_entity_embedding_vec_bytes(vector)}
                        )
                    except Exception as pack_error:
                        self.logger.error(
                            f"向量转换失败 {assoc.id}: {pack_error}"
                        )
                        embedding_failed += 1

            except Exception as batch_error:
                self.logger.warning(f"批量生成事件实体向量失败，降级重试: {batch_error}")
                for assoc in batch:
                    try:
                        text_for_vec = assoc.description or f"{assoc.event_id}-{assoc.entity_id}"
                        vector = await embedding_client.generate(text_for_vec)
                        rows_to_upsert.append(
                            {"id": assoc.id, "vec": to_event_entity_embedding_vec_bytes(vector)}
                        )
                    except Exception as retry_error:
                        self.logger.error(f"单条事件实体向量生成失败 {assoc.id}: {retry_error}")
                        embedding_failed += 1

        if not rows_to_upsert:
            stats = {
                "total": len(event_entities),
                "upserted": 0,
                "embedding_failed": embedding_failed,
                "db_failed": 0,
                "time": f"{(time.perf_counter() - start_time):.2f}s",
            }
            self.logger.info(f"{EVENT_ENTITY_EMBEDDING_TABLE_NAME} 写入为空: {stats}")
            return stats

        upsert_sql = text(
            f"""
            INSERT INTO {EVENT_ENTITY_EMBEDDING_TABLE_NAME} (id, vec)
            VALUES (:id, :vec)
            ON DUPLICATE KEY UPDATE vec = VALUES(vec)
            """
        )

        upserted = 0
        db_failed = 0
        for i in range(0, len(rows_to_upsert), config.index_batch_size):
            batch = rows_to_upsert[i : i + config.index_batch_size]
            try:
                async with self.session_factory() as session:
                    await session.execute(upsert_sql, batch)
                    await session.commit()
                upserted += len(batch)
            except Exception as batch_db_error:
                self.logger.error(f"批量写入 {EVENT_ENTITY_EMBEDDING_TABLE_NAME} 失败，降级重试: {batch_db_error}")
                for row in batch:
                    try:
                        async with self.session_factory() as session:
                            await session.execute(upsert_sql, row)
                            await session.commit()
                        upserted += 1
                    except Exception as row_db_error:
                        self.logger.error(
                            f"单条写入 {EVENT_ENTITY_EMBEDDING_TABLE_NAME} 失败 {row['id']}: {row_db_error}"
                        )
                        db_failed += 1

        # 最终一致性兜底：校验当前批次是否都已落库，缺失则逐条补偿一次
        expected_ids = set(assoc_by_id.keys())
        existing_ids = set()
        expected_id_list = list(expected_ids)
        for i in range(0, len(expected_id_list), config.index_batch_size):
            id_batch = expected_id_list[i : i + config.index_batch_size]
            async with self.session_factory() as session:
                result = await session.execute(
                    select(EventEntityEmbedding.id).where(EventEntityEmbedding.id.in_(id_batch))
                )
                existing_ids.update(result.scalars().all())

        missing_ids = expected_ids - existing_ids
        recovered = 0
        if missing_ids:
            self.logger.warning(
                f"{EVENT_ENTITY_EMBEDDING_TABLE_NAME} 检测到缺失 {len(missing_ids)} 条，开始补偿重试"
            )
            for missing_id in missing_ids:
                assoc = assoc_by_id.get(missing_id)
                if assoc is None:
                    continue
                try:
                    text_for_vec = assoc.description or f"{assoc.event_id}-{assoc.entity_id}"
                    vector = await embedding_client.generate(text_for_vec)
                    row = {"id": assoc.id, "vec": to_event_entity_embedding_vec_bytes(vector)}
                    async with self.session_factory() as session:
                        await session.execute(upsert_sql, row)
                        await session.commit()
                    recovered += 1
                except Exception as recover_error:
                    self.logger.error(
                        f"补偿写入 {EVENT_ENTITY_EMBEDDING_TABLE_NAME} 失败 {missing_id}: {recover_error}"
                    )

            # 补偿后再次校验，仍缺失则抛错避免静默丢失
            async with self.session_factory() as session:
                result = await session.execute(
                    select(EventEntityEmbedding.id).where(
                        EventEntityEmbedding.id.in_(list(expected_ids))
                    )
                )
                existing_after_recover = set(result.scalars().all())
            final_missing_ids = expected_ids - existing_after_recover
            if final_missing_ids:
                raise RuntimeError(
                    f"{EVENT_ENTITY_EMBEDDING_TABLE_NAME} 写入不完整: "
                    f"expected={len(expected_ids)}, actual={len(existing_after_recover)}, "
                    f"missing={len(final_missing_ids)}"
                )
        else:
            final_missing_ids = set()

        total_time = time.perf_counter() - start_time
        stats = {
            "total": len(event_entities),
            "upserted": upserted,
            "embedding_failed": embedding_failed,
            "db_failed": db_failed,
            "recovered": recovered,
            "missing_after_verify": len(final_missing_ids),
            "time": f"{total_time:.2f}s",
        }
        if embedding_failed > 0 or db_failed > 0:
            self.logger.warning(f"{EVENT_ENTITY_EMBEDDING_TABLE_NAME} 写入部分失败: {stats}")
        else:
            self.logger.info(
                f"{EVENT_ENTITY_EMBEDDING_TABLE_NAME} 写入成功: "
                f"{upserted}/{len(event_entities)} 条, 耗时{total_time:.2f}s"
            )
        return stats

    async def _save_to_database(self, events: List[SourceEvent]) -> List[str]:
        """
        保存事项到数据库

        Args:
            events: 事项列表

        Returns:
            事项ID列表
        """
        event_ids = []

        # 收集需要软删除的 article_id 集合（去重）
        article_ids = {e.article_id for e in events if e.article_id}

        async with self.session_factory() as session:
            # 先软删除旧事项，再写入新事项，在同一个事务中保证一致性
            # 防止服务重启导致平行调用多次提取时，旧事项未被清理
            for aid in article_ids:
                await session.execute(
                    update(SourceEvent)
                    .where(
                        SourceEvent.article_id == aid,
                        SourceEvent.not_deleted(),
                    )
                    .values(status="DELETED")
                )

            for event in events:
                event_ids.append(event.id)
                session.add(event)

                # 添加实体关联
                if hasattr(event, "event_associations") and event.event_associations:
                    for assoc in event.event_associations:
                        session.add(assoc)

            await session.commit()

        self.logger.info(f"已保存 {len(event_ids)} 个事项到数据库")
        return event_ids

    async def _load_events_with_relations(self, event_ids: List[str]) -> List[SourceEvent]:
        """
        从数据库加载事项（预加载关系数据）

        Args:
            event_ids: 事项ID列表

        Returns:
            事项列表（包含完整关系数据）
        """
        if not event_ids:
            return []

        async with self.session_factory() as session:
            result = await session.execute(
                select(SourceEvent)
                .where(SourceEvent.id.in_(event_ids))
                .options(
                    selectinload(SourceEvent.event_associations).selectinload(EventEntity.entity)
                )
            )
            events = list(result.scalars().all())

            # 确保数据在 session 外可访问
            session.expire_on_commit = False

            # 触发关系数据加载
            for event in events:
                _ = event.created_time
                _ = event.updated_time
                if hasattr(event, "event_associations"):
                    for assoc in event.event_associations:
                        _ = assoc.id
                        if assoc.entity:
                            _ = assoc.entity.name
                            _ = assoc.entity.type

            return events

    async def _sync_to_vector_store(self, events: List[SourceEvent], config: ExtractConfig) -> None:
        """
        同步事项和实体到向量库

        Args:
            events: 事项列表
            config: 提取配置
        """
        self.logger.info(f"开始同步 {len(events)} 个事项到向量库")

        # 初始化向量库客户端（延迟初始化）
        if self._es_client is None:
            self._es_client = get_es_client()
            self._event_repo = EventVectorRepository(self._es_client.client)
            self._entity_repo = EntityVectorRepository(self._es_client.client)

        # 检查连接
        if not await self._es_client.ping():
            self.logger.error("向量库连接失败，跳过同步")
            return

        # 收集所有唯一的实体
        unique_entities = await self._collect_unique_entities(events)

        # 1. 同步实体（如果启用）
        if unique_entities and config.enable_entity_vector_sync:
            await self._sync_entities(list(unique_entities.values()), config)
        elif unique_entities:
            self.logger.info(
                f"跳过 {len(unique_entities)} 个实体的向量同步 (enable_entity_vector_sync=False)"
            )

        # 2. 同步事项
        await self._sync_events(events, config)

        # 3. 同步事件-实体关联（如果启用）
        if config.enable_event_entity_vector_sync:
            await self._sync_event_entities(events, config)

        # 统计状态
        entity_status = (
            f"{len(unique_entities)} 个实体"
            if config.enable_entity_vector_sync
            else "实体同步已禁用"
        )

        # 统计事件-实体关联数量
        event_entity_count = sum(
            len(e.event_associations) for e in events if hasattr(e, "event_associations")
        )
        event_entity_status = (
            f"{event_entity_count} 个关联"
            if config.enable_event_entity_vector_sync
            else "事件-实体关联同步已禁用"
        )

        self.logger.info(
            f"向量库同步完成: {len(events)} 个事项, {entity_status}, {event_entity_status}"
        )

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出 - 确保资源清理"""
        if self._es_client is not None:
            try:
                await self._es_client.client.close()
                self._es_client = None
                self._event_repo = None
                self._entity_repo = None
            except Exception as e:
                self.logger.warning(f"关闭ES客户端失败: {e}")
        return False

    async def _collect_unique_entities(self, events: List[SourceEvent]) -> Dict[str, Entity]:
        """收集所有唯一的实体"""
        unique_entities = {}

        for event in events:
            if hasattr(event, "event_associations") and event.event_associations:
                for assoc in event.event_associations:
                    entity_id = assoc.entity_id
                    if entity_id not in unique_entities:
                        entity = await self._load_entity_by_id(entity_id)
                        if entity:
                            unique_entities[entity_id] = entity

        return unique_entities

    async def _load_entity_by_id(self, entity_id: str) -> Entity:
        """从数据库加载实体"""
        async with self.session_factory() as session:
            result = await session.execute(select(Entity).where(Entity.id == entity_id))
            return result.scalar_one_or_none()

    async def _sync_entities(self, entities: List[Entity], config: ExtractConfig) -> Dict[str, Any]:
        """
        同步实体到向量库（批量处理）

        Args:
            entities: 实体列表
            config: 提取配置

        Returns:
            统计信息
        """
        if not entities:
            return {"total": 0, "indexed": 0}

        # 过滤不需要索引的实体类型
        original_count = len(entities)
        # entities = [e for e in entities if should_index_entity_to_vector_store(e.type)]

        if len(entities) < original_count:
            skipped = original_count - len(entities)
            self.logger.info(f"过滤掉 {skipped} 个不需要索引的实体")

        if not entities:
            self.logger.info("所有实体都被过滤，跳过向量同步")
            return {"total": 0, "indexed": 0}

        return await self._batch_sync_entities(entities, config)

    async def _batch_sync_entities(
        self, entities: List[Entity], config: ExtractConfig
    ) -> Dict[str, Any]:
        """
        批量同步实体到向量库（使用批量处理工具）

        Args:
            entities: 实体列表
            config: 提取配置

        Returns:
            统计信息
        """
        from pipeline.utils.batch import batch_generate_embeddings, batch_index_to_es

        start_time = time.perf_counter()
        embedding_client = await get_embedding_client(scenario="general")
        es_client = get_es_client()

        # 阶段1: 批量生成向量
        def build_document(entity: Entity, vector: List[float]) -> Dict[str, Any]:
            return {
                "id": entity.id,
                "entity_id": entity.id,
                "source_config_id": entity.source_config_id,
                "type": entity.type,
                "name": entity.name,
                "vector": vector,
                "normalized_name": entity.normalized_name or "",
                "description": entity.description or "",
                "created_time": (
                    entity.created_time.isoformat() if entity.created_time else None
                ),
            }

        embedding_result = await batch_generate_embeddings(
            items=entities,
            text_extractor=lambda e: e.name,
            embedding_client=embedding_client,
            batch_size=config.embedding_batch_size,
            on_success=build_document,
        )

        documents = embedding_result["results"]
        embedding_failed = embedding_result["failed"]

        # 阶段2: 批量索引
        index_result = await batch_index_to_es(
            documents=documents,
            es_client=es_client,
            index_name=self._entity_repo.INDEX_NAME,
            batch_size=config.index_batch_size,
            routing=config.source_config_id,
        )

        indexed = index_result["indexed"]
        es_failed = index_result["failed"]

        total_time = time.perf_counter() - start_time

        stats = {
            "total": len(entities),
            "indexed": indexed,
            "embedding_failed": embedding_failed,
            "es_failed": es_failed,
            "time": f"{total_time:.2f}s",
        }

        if es_failed > 0 or embedding_failed > 0:
            self.logger.warning(f"实体同步部分失败: {stats}")
        else:
            self.logger.info(f"实体同步成功: {indexed}/{len(entities)} 条, 耗时{total_time:.2f}s")

        return stats

    async def _sync_events(
        self, events: List[SourceEvent], config: ExtractConfig
    ) -> Dict[str, Any]:
        """
        同步事项到向量库（批量处理）

        Args:
            events: 事项列表
            config: 提取配置

        Returns:
            统计信息
        """
        if not events:
            return {"total": 0, "indexed": 0}

        return await self._batch_sync_events(events, config)

    async def _batch_sync_events(
        self, events: List[SourceEvent], config: ExtractConfig
    ) -> Dict[str, Any]:
        """
        批量同步事项到向量库（两阶段处理）

        Args:
            events: 事项列表
            config: 提取配置

        Returns:
            统计信息
        """
        start_time = time.perf_counter()

        embedding_client = await get_embedding_client(scenario="general")
        es_client = get_es_client()

        documents = []
        embedding_failed = 0

        # 阶段1: 批量生成向量
        for i in range(0, len(events), config.embedding_batch_size):
            batch = events[i : i + config.embedding_batch_size]

            try:
                # 准备文本（每个事项需要2个向量，使用 embedding 专用长度限制）
                title_texts = [event.title for event in batch]
                content_texts = [
                    f"{event.title}\n\n{event.content[:config.embedding_max_length]}"
                    for event in batch
                ]

                # 批量生成向量
                title_vectors = await embedding_client.batch_generate(title_texts)
                content_vectors = await embedding_client.batch_generate(content_texts)

                # 构建文档
                for event, title_vec, content_vec in zip(batch, title_vectors, content_vectors):
                    doc = self._build_event_document(event, title_vec, content_vec)
                    documents.append(doc)

            except Exception as e:
                self.logger.warning(f"批量生成向量失败，降级重试: {e}")
                for event in batch:
                    try:
                        title_vec = await embedding_client.generate(event.title)
                        content_for_vec = (
                            f"{event.title}\n\n{event.content[:config.embedding_max_length]}"
                        )
                        content_vec = await embedding_client.generate(content_for_vec)
                        doc = self._build_event_document(event, title_vec, content_vec)
                        documents.append(doc)
                    except Exception as retry_e:
                        self.logger.error(f"单条生成向量失败 {event.id}: {retry_e}")
                        embedding_failed += 1

        # 阶段2: 批量索引（使用工具）
        from pipeline.utils.batch import batch_index_to_es

        index_result = await batch_index_to_es(
            documents=documents,
            es_client=es_client,
            index_name=self._event_repo.INDEX_NAME,
            batch_size=config.index_batch_size,
            routing=config.source_config_id,
        )

        indexed = index_result["indexed"]
        es_failed = index_result["failed"]
        total_time = time.perf_counter() - start_time

        stats = {
            "total": len(events),
            "indexed": indexed,
            "embedding_failed": embedding_failed,
            "es_failed": es_failed,
            "time": f"{total_time:.2f}s",
        }

        if es_failed > 0 or embedding_failed > 0:
            self.logger.warning(f"事项同步部分失败: {stats}")
        else:
            self.logger.info(f"事项同步成功: {indexed}/{len(events)} 条, 耗时{total_time:.2f}s")

        return stats

    async def _sync_event_entities(
        self, events: List[SourceEvent], config: ExtractConfig
    ) -> Dict[str, Any]:
        """
        同步事件-实体关联到向量库（批量处理）

        Args:
            events: 事项列表
            config: 提取配置

        Returns:
            统计信息
        """
        if not events:
            return {"total": 0, "indexed": 0}

        start_time = time.perf_counter()

        embedding_client = await get_embedding_client(scenario="general")
        es_client = get_es_client()

        # 收集所有 EventEntity 关联
        event_entities = self._collect_event_entities(events)

        if not event_entities:
            self.logger.info("没有事件-实体关联需要同步")
            return {"total": 0, "indexed": 0}

        # 阶段1: 批量生成向量（使用工具）
        from pipeline.utils.batch import batch_generate_embeddings, batch_index_to_es

        def build_document(assoc: EventEntity, vector: List[float]) -> Dict[str, Any]:
            return {
                "id": assoc.id,
                "event_id": assoc.event_id,
                "entity_id": assoc.entity_id,
                "source_config_id": config.source_config_id,
                "description": assoc.description or "",
                "vector": vector,
                "created_time": (
                    assoc.created_time.isoformat() if assoc.created_time else None
                ),
                "is_delete": False,
            }

        embedding_result = await batch_generate_embeddings(
            items=event_entities,
            text_extractor=lambda a: a.description or f"{a.event_id}-{a.entity_id}",
            embedding_client=embedding_client,
            batch_size=config.embedding_batch_size,
            on_success=build_document,
        )

        documents = embedding_result["results"]
        embedding_failed = embedding_result["failed"]

        # 阶段2: 批量索引（使用工具）
        index_result = await batch_index_to_es(
            documents=documents,
            es_client=es_client,
            index_name="event_entity_vectors",
            batch_size=config.index_batch_size,
            routing=config.source_config_id,
        )

        indexed = index_result["indexed"]
        es_failed = index_result["failed"]
        total_time = time.perf_counter() - start_time

        stats = {
            "total": len(event_entities),
            "indexed": indexed,
            "embedding_failed": embedding_failed,
            "es_failed": es_failed,
            "time": f"{total_time:.2f}s",
        }

        if es_failed > 0 or embedding_failed > 0:
            self.logger.warning(f"事件-实体关联同步部分失败: {stats}")
        else:
            self.logger.info(
                f"事件-实体关联同步成功: {indexed}/{len(event_entities)} 条, 耗时{total_time:.2f}s"
            )

        return stats

    def _build_event_document(
        self, event: SourceEvent, title_vec: List[float], content_vec: List[float]
    ) -> Dict[str, Any]:
        """构建事项文档（用于向量库索引）"""
        # 提取关联实体ID
        entity_ids = []
        if hasattr(event, "event_associations") and event.event_associations:
            entity_ids = [assoc.entity_id for assoc in event.event_associations]

        # 准备额外字段
        extra_fields = {}
        if event.extra_data and "tags" in event.extra_data:
            extra_fields["tags"] = event.extra_data["tags"]
        if event.category:
            extra_fields["category"] = event.category
        if event.keywords:
            extra_fields["keywords"] = event.keywords

        return {
            "id": event.id,
            "event_id": event.id,
            "source_config_id": event.source_config_id,
            "source_type": event.source_type,
            "source_id": event.source_id,
            "title": event.title,
            "summary": event.summary or "",
            "content": event.content,
            "title_vector": title_vec,
            "content_vector": content_vec,
            "entity_ids": entity_ids,
            "start_time": event.start_time.isoformat() if event.start_time else None,
            "end_time": event.end_time.isoformat() if event.end_time else None,
            "created_time": (event.created_time.isoformat() if event.created_time else None),
            **extra_fields,
        }
