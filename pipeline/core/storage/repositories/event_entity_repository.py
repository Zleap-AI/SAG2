"""
实体-事件关系向量 Repository

提供实体-事件关系向量的业务查询方法
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from elasticsearch_dsl import Q, Search

from pipeline.core.storage.repositories.base import BaseRepository


class EventEntityRepository(BaseRepository):
    """事件-实体关系向量 Repository"""

    INDEX_NAME = "event_entity_vectors"

    async def index_event_entity(
        self,
        association_id: str,
        event_id: str,
        entity_id: str,
        source_config_id: str,
        description: str,
        vector: List[float],
        is_delete: bool = False,
        **kwargs,
    ) -> str:
        """
        索引单个事件-实体关联

        Args:
            association_id: 关联ID
            event_id: 事件ID
            entity_id: 实体ID
            source_config_id: 信息源ID
            description: 关联描述
            vector: 向量
            is_delete: 是否删除
            **kwargs: 其他字段（created_time等）

        Returns:
            文档ID
        """
        document = {
            "event_id": event_id,
            "entity_id": entity_id,
            "source_config_id": source_config_id,
            "description": description,
            "vector": vector,
            "is_delete": is_delete,
            **kwargs,
        }

        # 使用 source_config_id 作为路由键，确保同一信息源的数据在同一分片
        return await self.index_document(
            self.INDEX_NAME, association_id, document, routing=source_config_id
        )

    def _is_valid_vector(self, vector: List[float]) -> bool:
        """
        验证向量是否有效（不包含NaN或Inf值）

        Args:
            vector: 待验证的向量

        Returns:
            bool: 向量是否有效
        """
        if not vector:
            return False

        try:
            np_array = np.array(vector, dtype=np.float32)
            return not (np.isnan(np_array).any() or np.isinf(np_array).any())
        except (ValueError, TypeError):
            return False

    async def search_similar_by_description(
        self,
        query_vector: List[float],
        k: int = 10,
        source_config_id: Optional[str] = None,
        source_config_ids: Optional[List[str]] = None,
        event_id: Optional[str] = None,
        event_ids: Optional[List[str]] = None,
        entity_id: Optional[str] = None,
        entity_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        通过描述向量搜索相似的事件-实体关联

        Args:
            query_vector: 查询向量
            k: 返回数量
            source_config_id: 信息源ID（单个，向后兼容）
            source_config_ids: 信息源ID列表（支持多源搜索）
            event_id: 事件ID过滤（单个，向后兼容）
            event_ids: 事件ID列表（支持多个事件）
            entity_id: 实体ID过滤（单个，向后兼容）
            entity_ids: 实体ID列表（支持多个实体）

        Returns:
            相似关联列表
        """
        if not self._is_valid_vector(query_vector):
            raise ValueError("查询向量包含无效值（NaN或Inf）")

        # 参数兼容处理：优先使用列表，如果没有则使用单个值
        if not source_config_ids and source_config_id:
            source_config_ids = [source_config_id]

        if not event_ids and event_id:
            event_ids = [event_id]

        if not entity_ids and entity_id:
            entity_ids = [entity_id]

        # 添加过滤条件
        filters = []
        if source_config_ids:
            # 单源使用 term 查询，多源使用 terms 查询
            if len(source_config_ids) == 1:
                filters.append(Q("term", source_config_id=source_config_ids[0]))
            else:
                filters.append(Q("terms", source_config_id=source_config_ids))

        if event_ids:
            # 单个使用 term 查询，多个使用 terms 查询
            if len(event_ids) == 1:
                filters.append(Q("term", event_id=event_ids[0]))
            else:
                filters.append(Q("terms", event_id=event_ids))

        if entity_ids:
            # 单个使用 term 查询，多个使用 terms 查询
            if len(entity_ids) == 1:
                filters.append(Q("term", entity_id=entity_ids[0]))
            else:
                filters.append(Q("terms", entity_id=entity_ids))

        # 只查询未删除的关联
        filters.append(Q("term", is_delete=False))

        # 构建filter
        filter_query = None
        if filters:
            filter_query = Q("bool", must=filters).to_dict()

        # 使用 source_config_id 作为路由键优化查询性能
        # 仅在单源时使用 routing，多源时禁用以支持跨分片查询
        routing = source_config_ids[0] if source_config_ids and len(source_config_ids) == 1 else None

        # 使用vector_search方法
        return await self.es_client.vector_search(
            index=self.INDEX_NAME,
            field="vector",
            vector=query_vector,
            size=k,
            filter_query=filter_query,
            routing=routing,
        )

    async def get_by_event(
        self, event_id: str, source_config_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取事件的所有实体关联

        Args:
            event_id: 事件ID
            source_config_id: 信息源ID（可选）

        Returns:
            关联列表
        """
        s = Search(using=self.es_client, index=self.INDEX_NAME)

        s = s.filter("term", event_id=event_id)
        s = s.filter("term", is_delete=False)

        if source_config_id:
            s = s.filter("term", source_config_id=source_config_id)

        s = s[:100]

        # 转换为字典并执行
        search_dict = s.to_dict()
        # 如果指定了 source_config_id，使用 routing 优化查询性能
        routing = source_config_id if source_config_id else None
        response = await self.es_client.search(
            index=self.INDEX_NAME,
            query=search_dict.get("query", {}),
            size=search_dict.get("size", 10),
            routing=routing
        )

        # 处理两种返回格式：list（默认）或 dict（完整响应）
        associations = []

        if isinstance(response, list):
            # 格式1：直接是文档列表（ES client 的默认返回格式）
            for assoc_data in response:
                if isinstance(assoc_data, dict):
                    associations.append(assoc_data)

        elif isinstance(response, dict) and "hits" in response:
            # 格式2：完整响应格式
            hits = response["hits"].get("hits", [])
            for hit in hits:
                if isinstance(hit, dict) and "_source" in hit:
                    associations.append(hit["_source"])

        return associations

    async def get_by_entity(
        self, entity_id: str, source_config_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取实体的所有事件关联

        Args:
            entity_id: 实体ID
            source_config_id: 信息源ID（可选）

        Returns:
            关联列表
        """
        s = Search(using=self.es_client, index=self.INDEX_NAME)

        s = s.filter("term", entity_id=entity_id)
        s = s.filter("term", is_delete=False)

        if source_config_id:
            s = s.filter("term", source_config_id=source_config_id)

        s = s[:100]

        # 转换为字典并执行
        search_dict = s.to_dict()
        # 如果指定了 source_config_id，使用 routing 优化查询性能
        routing = source_config_id if source_config_id else None
        response = await self.es_client.search(
            index=self.INDEX_NAME,
            query=search_dict.get("query", {}),
            size=search_dict.get("size", 10),
            routing=routing
        )

        # 处理两种返回格式：list（默认）或 dict（完整响应）
        associations = []

        if isinstance(response, list):
            # 格式1：直接是文档列表（ES client 的默认返回格式）
            for assoc_data in response:
                if isinstance(assoc_data, dict):
                    associations.append(assoc_data)

        elif isinstance(response, dict) and "hits" in response:
            # 格式2：完整响应格式
            hits = response["hits"].get("hits", [])
            for hit in hits:
                if isinstance(hit, dict) and "_source" in hit:
                    associations.append(hit["_source"])

        return associations

    async def get_associations_by_ids(
        self, association_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        根据关联ID列表批量获取

        Args:
            association_ids: 关联ID列表

        Returns:
            关联详细信息列表
        """
        if not association_ids:
            return []

        # 分批处理，避免超过 ES 的 max_result_window 限制
        BATCH_SIZE = 3000
        results = []

        for i in range(0, len(association_ids), BATCH_SIZE):
            batch_ids = association_ids[i:i + BATCH_SIZE]

            query = {
                "terms": {
                    "_id": batch_ids
                }
            }

            response = await self.es_client.search(
                index=self.INDEX_NAME,
                query=query,
                size=len(batch_ids),
                return_full_response=True,
            )

            # return_full_response=True 时返回格式：{total, max_score, hits: [{id, score, source, index}]}
            for hit in response.get("hits", []):
                # source 字段包含实际的文档数据
                source_data = hit.get("source", {})
                if source_data:
                    results.append(source_data.copy())

        return results

    async def batch_search_similar_by_event_entity_pairs(
        self,
        query_vector: List[float],
        event_entity_pairs: List[Tuple[str, str]],
        source_config_ids: Optional[List[str]] = None,
        include_source: bool = False,
    ) -> Dict[Tuple[str, str], float]:
        """
        批量查询 (event_id, entity_id) 组合的描述向量相似度

        注意：同一实体在不同事项中的 describe 是不同的，必须按组合查询

        Args:
            query_vector: 查询向量
            event_entity_pairs: (event_id, entity_id) 组合列表
            source_config_ids: 信息源ID列表
            include_source: 是否返回完整的 source 数据（包含 description 等）

        Returns:
            include_source=False: {(event_id, entity_id): similarity_score}
            include_source=True: {(event_id, entity_id): {"score": float, "description": str, ...}}
        """
        if not event_entity_pairs:
            return {}

        if not self._is_valid_vector(query_vector):
            return {}

        # 分批处理，避免超过 ES 的 max_result_window 限制
        BATCH_SIZE = 2500
        all_results = {}

        for i in range(0, len(event_entity_pairs), BATCH_SIZE):
            batch_pairs = event_entity_pairs[i:i + BATCH_SIZE]

            # 分离 event_ids 和 entity_ids 用于 terms 查询
            batch_event_ids = [p[0] for p in batch_pairs]
            batch_entity_ids = [p[1] for p in batch_pairs]

            # 构建过滤条件：同时匹配 event_id 和 entity_id
            filters = [
                {"terms": {"event_id": batch_event_ids}},
                {"terms": {"entity_id": batch_entity_ids}},
                {"term": {"is_delete": False}}
            ]

            # 处理信息源过滤：单源用 term + routing，多源用 terms
            routing = None
            if source_config_ids:
                if len(source_config_ids) == 1:
                    # 单源：使用 term 查询 + routing 优化
                    filters.append({"term": {"source_config_id": source_config_ids[0]}})
                    routing = source_config_ids[0]
                else:
                    # 多源：使用 terms 查询，无 routing
                    filters.append({"terms": {"source_config_id": source_config_ids}})

            filter_query = {"bool": {"must": filters}}

            # 使用 vector_search，单源时传入 routing 优化查询性能
            results = await self.es_client.vector_search(
                index=self.INDEX_NAME,
                field="vector",
                vector=query_vector,
                size=len(batch_pairs) * 2,
                filter_query=filter_query,
                routing=routing,
            )

            # 按 (event_id, entity_id) 组织结果
            for hit in results:
                event_id = hit.get("event_id")
                entity_id = hit.get("entity_id")
                score = hit.get("_score", 0.0)
                if event_id and entity_id:
                    if include_source:
                        # 返回完整信息
                        all_results[(event_id, entity_id)] = {
                            "score": float(score),
                            "description": hit.get("description", ""),
                            "source_config_id": hit.get("source_config_id"),
                            "is_delete": hit.get("is_delete", False),
                        }
                    else:
                        # 只返回分数
                        all_results[(event_id, entity_id)] = float(score)

        return all_results