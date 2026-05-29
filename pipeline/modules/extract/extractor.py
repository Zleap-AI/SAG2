"""
事项提取器

主控制器 - 协调提取流程

流程: chunks → processor(LLM) → filter(过滤) → parser(解析) → saver(保存)
"""

import asyncio
import inspect
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Awaitable, Callable, Dict, List, Optional, Union

from sqlalchemy import or_, select
from sqlalchemy.orm import selectinload

from pipeline.core.ai.base import BaseLLMClient
from pipeline.core.prompt.manager import PromptManager
from pipeline.db import (
    ArticleParseStatus,
    SourceChunk,
    EntityType as DBEntityType,
    EventEntity,
    SourceEvent,
    get_session_factory,
)
from pipeline.exceptions import ExtractError
from pipeline.modules.extract.config import ExtractConfig
from pipeline.modules.extract.processor import EventProcessor
from pipeline.modules.extract.parser import ResultParser, ParseContext
from pipeline.modules.extract.saver import EventSaver
from pipeline.utils import get_logger, get_utc_now

logger = get_logger("extract.extractor")


class EventExtractor:
    """
    事项提取器（主控制器）

    流程: chunks → processor(LLM) → filter(过滤) → parser(解析) → saver(保存)
    """

    def __init__(
        self,
        prompt_manager: PromptManager,
        model_config: Optional[Dict] = None,
        on_progress: Optional[Callable[[int, int], Union[Awaitable[None], None]]] = None,
    ):
        """
        初始化事项提取器

        Args:
            prompt_manager: 提示词管理器
            model_config: LLM配置字典（可选）
            on_progress: 进度回调 (completed, total)
        """
        self.prompt_manager = prompt_manager
        self.model_config = model_config
        self._on_progress = on_progress
        self._llm_client = None  # 延迟初始化
        self.session_factory = get_session_factory()
        self.logger = get_logger("extract.extractor")

        # 组件（延迟初始化）
        self._saver = None
        self._parser = None

    async def _get_llm_client(self) -> BaseLLMClient:
        """获取LLM客户端（懒加载）"""
        if self._llm_client is None:
            from pipeline.core.ai.factory import create_llm_client

            self._llm_client = await create_llm_client(
                scenario="extract", model_config=self.model_config
            )

        return self._llm_client

    async def extract(self, config: ExtractConfig) -> List[SourceEvent]:
        """
        提取事项（统一入口 - 新架构）

        工作流程：
        1. 加载所有chunks
        2. 按max_concurrency并发处理（Semaphore控制）
        3. 每个chunk由一个ExtractorAgent处理
        4. 合并所有结果
        5. 保存到数据库 + Elasticsearch
        6. 更新源状态为已完成

        Args:
            config: 提取配置

        Returns:
            所有chunks提取的事项列表

        Example:
            config = ExtractConfig(
                source_config_id="source-uuid",
                chunk_ids=["chunk-1", "chunk-2", "chunk-3"],
                max_concurrency=3
            )
            events = await extractor.extract(config)
        """
        self.logger.info(
            f"开始批量提取: chunks={len(config.chunk_ids)}, " f"并发数={config.max_concurrency}"
        )

        sync_date_value = datetime.now(timezone.utc).replace(tzinfo=None)

        try:
            # 1. 加载所有chunks
            chunks = await self._load_chunks(config.chunk_ids)

            if not chunks:
                self.logger.info("没有找到可用的chunks")
                return []

            # 2. 标记运行中
            await self._update_source_status(chunks, status="EXTRACTING")

            # 3. 并发处理chunks（每个chunk一个Agent）
            all_events = await self._process_chunks_with_agents(chunks, config)

            self.logger.info(f"批量提取完成: chunks={len(chunks)}, events={len(all_events)}")

            # 4. 按原文顺序重新排序并分配全局 rank
            if all_events:
                # 创建 chunk_id -> chunk.rank 的映射
                chunk_rank_map = {chunk.id: chunk.rank for chunk in chunks}

                # 排序规则：
                # 1. 先按 chunk.rank（保证 chunk 之间的顺序）
                # 2. 再按事项的时间（会话）或 chunk 内 rank（文档）
                def sort_key(event):
                    chunk_order = chunk_rank_map.get(event.chunk_id, 9999)
                    event_order = event.rank or 0
                    return (chunk_order, event_order)

                all_events.sort(key=sort_key)

                # 重新分配全局连续 rank
                for i, event in enumerate(all_events):
                    event.rank = i

                self.logger.info(
                    f"事项已按原文顺序排序: chunks={len(chunks)}, events={len(all_events)}"
                )

            # 5. 保存到数据库（包括ES）
            if all_events:
                await self._save_events(all_events, config)

                # 6. 重新从数据库加载事项（带完整关系数据）
                event_ids = [e.id for e in all_events]
                all_events = await self._reload_events_with_relations(event_ids)
            else:
                self.logger.info("没有提取到任何事项，跳过保存")

            # 7. 更新源状态为已完成，写入 sync_date
            if chunks:
                await self._update_source_status(
                    chunks, status="COMPLETED", sync_date=sync_date_value
                )

            return all_events

        except Exception as e:
            # 提取失败时，更新状态为失败，写入 sync_date
            self.logger.error(f"提取失败: {e}", exc_info=True)
            try:
                chunks = await self._load_chunks(config.chunk_ids)
                if chunks:
                    await self._update_source_status(
                        chunks, status="FAILED", error=str(e), sync_date=sync_date_value
                    )
            except Exception as update_error:
                self.logger.error(f"更新失败状态时出错: {update_error}")

            raise ExtractError(f"提取失败: {e}") from e
        finally:
            # 确保资源清理
            if self._saver is not None and hasattr(self._saver, '_es_client'):
                if self._saver._es_client is not None:
                    try:
                        await self._saver._es_client.client.close()
                        self._saver._es_client = None
                        self._saver._event_repo = None
                        self._saver._entity_repo = None
                    except Exception as cleanup_err:
                        self.logger.warning(f"清理ES客户端失败: {cleanup_err}")

    async def _load_chunks(self, chunk_ids: List[str]) -> List[SourceChunk]:
        """批量加载chunks（按rank排序）"""
        async with self.session_factory() as session:
            result = await session.execute(
                select(SourceChunk)
                .where(SourceChunk.id.in_(chunk_ids))
                .order_by(SourceChunk.rank)  # 🆕 按 rank 排序
            )
            chunks = list(result.scalars().all())

            if len(chunks) != len(chunk_ids):
                missing = set(chunk_ids) - {c.id for c in chunks}
                self.logger.info(f"部分chunk不存在: {missing}")

            return chunks

    async def _process_chunks_with_agents(
        self, chunks: List[SourceChunk], config: ExtractConfig
    ) -> List[SourceEvent]:
        """
        并发处理chunks（每个chunk一个Agent）

        使用asyncio.Semaphore控制并发数量：
        - 同时最多有max_concurrency个Agent在运行
        - 超出的chunk会自动排队等待
        - 一个chunk完成后，立即启动下一个

        Args:
            chunks: chunk列表
            config: 提取配置

        Returns:
            合并后的所有事项
        """
        semaphore = asyncio.Semaphore(config.max_concurrency)

        # 进度跟踪
        completed = 0
        success_count = 0
        failed_count = 0
        total = len(chunks)
        lock = asyncio.Lock()

        async def process_single_chunk(chunk: SourceChunk, index: int) -> List[SourceEvent]:
            """处理单个chunk（带并发控制和进度统计）"""
            nonlocal completed, success_count, failed_count

            async with semaphore:  # 🔒 获取并发槽位（没有就等待）
                is_success = False
                events = []
                error_msg = None

                try:
                    self.logger.info(
                        f"[{index+1}/{total}] 开始处理: chunk_id={chunk.id}, "
                        f"type={chunk.source_type}"
                    )

                    # 调用chunk级提取（使用ExtractorAgent）
                    events = await self.extract_from_chunk(chunk, config)
                    is_success = True

                except Exception as e:
                    error_msg = str(e)
                    self.logger.error(
                        f"❌ [{index+1}/{total}] 失败: "
                        f"chunk_id={chunk.id}, error={e}",
                        exc_info=True,
                    )

                # 更新进度（锁内只做计数，锁外做回调）
                async with lock:
                    completed += 1
                    if is_success:
                        success_count += 1
                    else:
                        failed_count += 1
                    progress = completed * 100 // total
                    should_report = self._on_progress and (completed % 10 == 0 or completed == total)
                    snap_completed, snap_total = completed, total

                if is_success:
                    self.logger.info(
                        f"✅ [{index+1}/{total}] 完成 ({progress}%): "
                        f"chunk_id={chunk.id}, events={len(events)}"
                    )
                else:
                    self.logger.error(
                        f"❌ [{index+1}/{total}] 失败 ({progress}%): "
                        f"chunk_id={chunk.id}"
                    )

                if should_report:
                    try:
                        result = self._on_progress(snap_completed, snap_total)
                        if inspect.isawaitable(result):
                            await result
                    except Exception as e:
                        self.logger.warning(f"进度回调失败: {e}")

                return events if is_success else []
                # 🔓 离开时自动释放槽位

        # 并发执行所有chunk
        self.logger.info(f"🚀 启动并发提取: total={total}, concurrency={config.max_concurrency}")

        tasks = [process_single_chunk(chunk, i) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # 合并结果
        all_events = []
        for events in results:
            all_events.extend(events)

        # 最终统计
        self.logger.info(
            f"📊 批量提取统计: 总数={total}, 成功={success_count}, "
            f"失败={failed_count}, 事项={len(all_events)}"
        )

        return all_events

    async def _save_events(
        self, events: List[SourceEvent], config: ExtractConfig
    ) -> List[SourceEvent]:
        """
        保存事项（使用 EventSaver）

        Args:
            events: 事项列表
            config: 提取配置

        Returns:
            保存后的事项列表（包含完整关系数据）
        """
        if self._saver is None:
            self._saver = EventSaver()

        return await self._saver.commit(events, config)

    async def extract_from_chunk(
        self, chunk: SourceChunk, config: ExtractConfig
    ) -> List[SourceEvent]:
        """
        从chunk提取事项

        Args:
            chunk: 来源片段对象
            config: 提取配置

        Returns:
            提取的事项列表
        """
        try:
            self.logger.info(f"开始从chunk提取: chunk_id={chunk.id}, type={chunk.source_type}")

            # 0. 内容长度过滤
            content_length = chunk.chunk_length or len(chunk.content or "")
            if config.chunk_min_length > 0 and content_length < config.chunk_min_length:
                self.logger.info(f"Chunk {chunk.id} 内容太短({content_length}字符)，跳过")
                return []

            # 1. 加载内容和元数据
            content_items, raw_metadata = await self._load_chunk_content(chunk, config)
            if not content_items:
                self.logger.info(f"Chunk {chunk.id} 无内容（可能全为图片已被过滤）")
                return []

            # 2. 加载实体类型
            entity_types = await self._load_entity_types_for_chunk(config)

            # 3. 创建 EventProcessor
            llm_client = await self._get_llm_client()
            processor = EventProcessor(
                llm_client=llm_client,
                prompt_manager=self.prompt_manager,
                config=config,
            )
            await processor.initialize(entity_types)

            # 4. 构建元数据
            metadata = {
                "document_title": raw_metadata.get("title", ""),
                "document_summary": raw_metadata.get("summary", ""),  # 全文摘要（全局视角）
                "chunk_title": chunk.heading or f"片段{chunk.rank + 1}",
                "previous_context": self._format_previous_context(
                    raw_metadata.get("previous_chunk")
                ),
            }

            # 5. 调用 LLM 提取
            raw_result = await processor.process(
                items=content_items,
                metadata=metadata,
                source_type=chunk.source_type,
            )

            # 6. 解析结果（Dict -> SourceEvent）
            if self._parser is None:
                self._parser = ResultParser(config)

            # 构建解析上下文
            context = ParseContext(
                source_config_id=config.source_config_id,
                source_type=chunk.source_type,
                source_id=chunk.source_id,
                chunk_id=chunk.id,
                source_created_time=await processor.get_source_created_time(
                    content_items, chunk.source_type
                ),
            )

            # 解析事项
            raw_items = raw_result.get("data", {}).get("items", [])
            events = self._parser.parse_events(raw_items, content_items, context)

            # 7. 处理实体关联
            events = await self._parser.process_entity_associations(events, entity_types)

            self.logger.info(f"Chunk提取完成: chunk_id={chunk.id}, events={len(events)}")
            return events

        except Exception as e:
            self.logger.error(f"Chunk提取失败: {e}", exc_info=True)
            raise ExtractError(f"Chunk提取失败: {e}") from e

    def _format_previous_context(self, previous_chunk) -> str:
        """格式化前文上下文"""
        if not previous_chunk:
            return ""

        title = previous_chunk.get("heading") or previous_chunk.get("title") or "前文"
        content = previous_chunk.get("content", "")

        if len(content) > 300:
            content = content[:300] + "..."

        return f"**{title}**\n{content}"

    async def _load_chunk_content(self, chunk: SourceChunk, config: ExtractConfig):
        """加载chunk的内容和元数据"""
        if chunk.source_type == "ARTICLE":
            return await self._load_article_content(chunk, config)
        else:
            raise ExtractError(f"不支持的类型: {chunk.source_type}")

    async def _load_article_content(self, chunk: SourceChunk, config: ExtractConfig):
        """加载文章片段 + 上文chunk内容作为背景"""
        from pipeline.db import Article, ArticleSection

        async with self.session_factory() as session:
            # 1. 加载文章（源背景）
            article = await session.get(Article, chunk.source_id)
            if not article:
                raise ExtractError(f"文章不存在: {chunk.source_id}")

            # 2. 加载当前chunk的sections（待处理内容）
            section_ids = chunk.references if chunk.references else []

            if section_ids:
                query = select(ArticleSection).where(ArticleSection.id.in_(section_ids))
                # 过滤图片类型的 section（避免噪音）
                # 注意：type 可能为 NULL，需要用 or_ 处理
                if config.filter_image_sections:
                    query = query.where(
                        or_(ArticleSection.type.is_(None), ArticleSection.type != "IMAGE")
                    )
                sections_result = await session.execute(query.order_by(ArticleSection.rank))
            else:
                query = select(ArticleSection).where(ArticleSection.article_id == chunk.source_id)
                # 过滤图片类型的 section（避免噪音）
                # 注意：type 可能为 NULL，需要用 or_ 处理
                if config.filter_image_sections:
                    query = query.where(
                        or_(ArticleSection.type.is_(None), ArticleSection.type != "IMAGE")
                    )
                sections_result = await session.execute(query.order_by(ArticleSection.rank))

            sections = list(sections_result.scalars().all())

            # 3. 加载上一个chunk的内容（上文背景）
            previous_chunk = None
            if chunk.rank > 0:
                prev_result = await session.execute(
                    select(SourceChunk)
                    .where(SourceChunk.source_id == chunk.source_id)
                    .where(SourceChunk.source_type == "ARTICLE")
                    .where(SourceChunk.rank == chunk.rank - 1)
                )
                previous_chunk = prev_result.scalars().first()  # 使用 first() 避免多行报错

            return sections, {
                # Article 表字段（源背景）
                "title": article.title,
                "summary": article.summary,
                # 当前 Chunk 信息
                "chunk_rank": chunk.rank,
                "chunk_heading": chunk.heading,
                # 上文 Chunk（提供上下文）
                "previous_chunk": (
                    {
                        "heading": previous_chunk.heading,
                        "content": (
                            previous_chunk.content[:800]
                            if len(previous_chunk.content or "") > 800
                            else previous_chunk.content
                        ),
                    }
                    if previous_chunk and previous_chunk.content
                    else None
                ),
            }

    async def _load_entity_types_for_chunk(self, config: ExtractConfig) -> List[DBEntityType]:
        """
        Load entity type definitions (always returns non-empty list)

        Priority:
        1. Default global types (is_default=True)
        2. Source-level custom types
        3. Runtime types in config (converted to DBEntityType objects)

        Returns:
            List of DBEntityType objects
        """
        entity_types: List[DBEntityType] = []

        async with self.session_factory() as session:
            # 加载默认类型
            default_result = await session.execute(
                select(DBEntityType)
                .where(DBEntityType.is_default == True)
                .where(DBEntityType.is_active == True)
            )
            default_types = default_result.scalars().all()
            entity_types.extend(default_types)

            # 加载source级别类型
            if config.source_config_id:
                custom_result = await session.execute(
                    select(DBEntityType)
                    .where(DBEntityType.source_config_id == config.source_config_id)
                    .where(DBEntityType.is_active == True)
                )
                custom_types = custom_result.scalars().all()
                entity_types.extend(custom_types)

        # 运行时类型（最高优先级）- 需要转换为 DBEntityType 对象
        if config.custom_entity_types:
            async with self.session_factory() as session:
                for custom_et in config.custom_entity_types:
                    existing_result = await session.execute(
                        select(DBEntityType)
                        .where(DBEntityType.type == custom_et.type)
                        .where(
                            (DBEntityType.source_config_id == config.source_config_id)
                            | (DBEntityType.is_default == True)
                        )
                        .where(DBEntityType.is_active == True)
                    )
                    existing = existing_result.scalar_one_or_none()

                    if existing:
                        entity_types.append(existing)
                    else:
                        value_constraints = getattr(custom_et, "value_constraints", None)
                        if not value_constraints:
                            validation_rule = getattr(custom_et, "validation_rule", None)
                            if validation_rule:
                                value_constraints = validation_rule

                        temp_et = DBEntityType(
                            id=str(uuid.uuid4()),
                            source_config_id=config.source_config_id,
                            scope="source",
                            type=custom_et.type,
                            name=custom_et.name,
                            description=custom_et.description or "",
                            weight=Decimal(str(custom_et.weight)),
                            similarity_threshold=Decimal("0.800"),
                            value_constraints=value_constraints,
                            extra_data=(
                                {
                                    "extraction_prompt": getattr(
                                        custom_et, "extraction_prompt", None
                                    ),
                                    "extraction_examples": getattr(
                                        custom_et, "extraction_examples", None
                                    ),
                                }
                                if (
                                    hasattr(custom_et, "extraction_prompt")
                                    or hasattr(custom_et, "extraction_examples")
                                )
                                else None
                            ),
                            is_default=False,
                            is_active=True,
                        )
                        entity_types.append(temp_et)

        # 按 type 去重，只保留首次出现的实体类型
        seen: set = set()
        deduped: List[DBEntityType] = []
        for et in entity_types:
            if et.type not in seen:
                seen.add(et.type)
                deduped.append(et)
        entity_types = deduped

        # 记录最终加载的实体类型（用于调试）
        entity_type_names = [et.type for et in entity_types]
        self.logger.info(
            f"实体类型加载完成: 共 {len(entity_types)} 个类型 - {entity_type_names}"
        )

        return entity_types

    async def _reload_events_with_relations(self, event_ids: List[str]) -> List[SourceEvent]:
        """
        Reload events from database with relations preloaded

        Solves cross-session issue: re-query after save to ensure all relations are loaded correctly

        Args:
            event_ids: List of event IDs

        Returns:
            List of events with complete relations
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

            # 设置 expire_on_commit=False，确保数据在 session 外可访问
            session.expire_on_commit = False

            # 触发关系数据加载（确保所有字段在 session 外可访问）
            for event in events:
                # 触发事项字段加载
                _ = event.title
                _ = event.created_time

                # 触发关联和实体加载
                if hasattr(event, "event_associations"):
                    for assoc in event.event_associations:
                        _ = assoc.id
                        if assoc.entity:
                            _ = assoc.entity.name
                            _ = assoc.entity.type

            self.logger.info(f"重新加载了 {len(events)} 个事项（包含完整关系）")
            return events

    async def _update_source_status(
        self,
        chunks: List[SourceChunk],
        status: str,
        error: Optional[str] = None,
        sync_date: Optional[datetime] = None,
    ) -> None:
        """
        Update source status (Article)

        Args:
            chunks: chunk list (used to determine source type and ID)
            status: internal status value (EXTRACTING/COMPLETED/FAILED)
            error: error message (optional, provided on failure)
            sync_date: sync time (UTC time pre-fetched before extraction starts, written on completion/failure)
        """
        if not chunks:
            return

        from pipeline.db import Article

        # 确定源类型和ID（同一批chunks应该来自同一个源）
        source_type = chunks[0].source_type
        source_id = chunks[0].source_id

        # 验证所有chunks来自同一个源
        if not all(c.source_type == source_type and c.source_id == source_id for c in chunks):
            self.logger.info("Chunks来自不同的源，无法统一更新状态")
            return

        async with self.session_factory() as session:
            try:
                if source_type == "ARTICLE":
                    result = await session.execute(select(Article).where(Article.id == source_id))
                    source = result.scalar_one_or_none()

                    if source:
                        source.status = status
                        if status == "EXTRACTING":
                            source.parse_status = ArticleParseStatus.EXTRACTING.value
                        elif status == "COMPLETED":
                            source.parse_status = ArticleParseStatus.COMPLETED.value
                            source.sync_date = get_utc_now().replace(tzinfo=None)
                        elif status == "FAILED":
                            source.parse_status = ArticleParseStatus.EXTRACTION_FAILED.value
                        if error:
                            source.error = error
                        await session.commit()
                        self.logger.info(f"✅ 已更新文章状态: {source_id} -> {status}")
                    else:
                        self.logger.info(f"文章不存在: {source_id}")

                else:
                    self.logger.info(f"不支持的源类型: {source_type}")

            except Exception as e:
                self.logger.error(f"更新源状态失败: {e}", exc_info=True)
