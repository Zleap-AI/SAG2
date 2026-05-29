"""
实体向量 Repository

提供实体向量的业务查询方法
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from elasticsearch import AsyncElasticsearch
from elasticsearch_dsl import Q, Search

from pipeline.core.storage.repositories.base import BaseRepository
from pipeline.db import get_session_factory


class EntityVectorRepository(BaseRepository):
    """实体向量 Repository"""

    INDEX_NAME = "entity_vectors"
    
    # 类级别缓存：类型阈值 (缓存 key -> (thresholds_dict, timestamp))
    _type_thresholds_cache: Dict[str, Tuple[Dict[str, float], float]] = {}
    _CACHE_TTL_SECONDS = 300  # 缓存 5 分钟

    def __init__(self, es_client: AsyncElasticsearch):
        """
        初始化 Repository

        Args:
            es_client: Elasticsearch 异步客户端
        """
        super().__init__(es_client)
        self.session_factory = get_session_factory()

    async def index_entity(
        self,
        entity_id: str,
        source_config_id: str,
        entity_type: str,
        name: str,
        vector: List[float],
        **kwargs,
    ) -> str:
        """
        索引单个实体

        Args:
            entity_id: 实体ID
            source_config_id: 信息源ID
            entity_type: 实体类型
            name: 实体名称
            vector: 向量
            **kwargs: 其他字段（created_time等）

        Returns:
            文档ID
        """
        document = {
            "entity_id": entity_id,
            "source_config_id": source_config_id,
            "type": entity_type,
            "name": name,
            "vector": vector,
            **kwargs,
        }

        # 使用 source_config_id 作为路由键，确保同一信息源的数据在同一分片
        return await self.index_document(
            self.INDEX_NAME, entity_id, document, routing=source_config_id
        )

    async def search_by_name(
        self, name: str, source_config_id: Optional[str] = None, size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        按名称搜索实体

        Args:
            name: 实体名称（模糊匹配）
            source_config_id: 信息源ID（可选）
            size: 返回数量

        Returns:
            实体列表
        """
        s = Search(using=self.es_client, index=self.INDEX_NAME)

        # 构建查询
        s = s.query("match", name=name)

        if source_config_id:
            s = s.filter("term", source_config_id=source_config_id)

        s = s[:size]

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
        return [hit["_source"] for hit in response["hits"]["hits"]]

    async def _get_entity_type_info(
        self, source_config_ids: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        获取实体类型的阈值和权重信息（带内存缓存，TTL=5分钟）

        Args:
            source_config_ids: 信息源ID列表（可选，支持多源）

        Returns:
            实体类型信息: {type: {"threshold": 0.8, "weight": 1.5}}
        """
        # 生成缓存 key
        cache_key = ",".join(sorted(source_config_ids)) if source_config_ids else "__global__"
        
        # 检查缓存
        now = time.time()
        if cache_key in self._type_thresholds_cache:
            cached_info, cached_time = self._type_thresholds_cache[cache_key]
            if now - cached_time < self._CACHE_TTL_SECONDS:
                return {k: v.copy() for k, v in cached_info.items()}  # 深拷贝
        
        # 缓存未命中，查询数据库
        from sqlalchemy import select
        from pipeline.db import EntityType

        type_info = {}

        async with self.session_factory() as session:
            query = select(EntityType.type, EntityType.similarity_threshold, EntityType.weight)

            if source_config_ids:
                # 查找信息源特定的实体类型，如果没有则使用系统默认类型
                query = query.where(
                    (EntityType.source_config_id.in_(source_config_ids)) | (EntityType.source_config_id.is_(None))
                )
            else:
                # 只查找系统默认类型
                query = query.where(EntityType.source_config_id.is_(None))

            query = query.where(EntityType.is_active == True)

            result = await session.execute(query)
            for entity_type, threshold, weight in result.fetchall():
                # 如果同一个类型有多个定义，使用更具体的（信息源特定的）
                if entity_type not in type_info:
                    type_info[entity_type] = {
                        "threshold": float(threshold),
                        "weight": float(weight) if weight else 1.0
                    }

        # 更新缓存
        self._type_thresholds_cache[cache_key] = (type_info, now)
        
        return type_info

    async def search_similar(
        self,
        query_vector: List[float],
        k: int = 10,
        source_config_id: Optional[str] = None,
        source_config_ids: Optional[List[str]] = None,
        entity_type: Optional[str] = None,
        include_type_threshold: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        向量相似度搜索

        Args:
            query_vector: 查询向量
            k: 返回数量
            source_config_id: 信息源ID（单个，向后兼容）
            source_config_ids: 信息源ID列表（支持多源搜索）
            entity_type: 实体类型（可选）
            include_type_threshold: 是否包含实体类型相似度阈值信息

        Returns:
            相似实体列表，如果 include_type_threshold=True 则包含 type_threshold 字段
        """
        # 参数兼容处理：优先使用 source_config_ids，如果没有则使用 source_config_id
        if not source_config_ids and source_config_id:
            source_config_ids = [source_config_id]



        # 添加过滤条件
        filter_query = None
        filters = []
        if source_config_ids:
            # 单源使用 term 查询，多源使用 terms 查询
            if len(source_config_ids) == 1:
                filters.append(Q("term", source_config_id=source_config_ids[0]))
            else:
                filters.append(Q("terms", source_config_id=source_config_ids))
        if entity_type:
            filters.append(Q("term", type=entity_type))

        if filters:
            filter_query = Q("bool", must=filters).to_dict()

        # 使用 source_config_id 作为路由键优化查询性能
        # 仅在单源时使用 routing，多源时禁用以支持跨分片查询
        routing = source_config_ids[0] if source_config_ids and len(source_config_ids) == 1 else None

        # 使用vector_search方法
        search_results = await self.es_client.vector_search(
            index=self.INDEX_NAME,
            field="vector",
            vector=query_vector,
            size=k,
            filter_query=filter_query,
            routing=routing,
        )

        results = []
        if include_type_threshold:
            # 获取实体类型信息（包含阈值和权重）
            type_info = await self._get_entity_type_info(source_config_ids)

            for hit in search_results:
                entity_data = hit.copy()
                # 添加实体类型的阈值和权重
                entity_type_name = entity_data.get("type")
                if entity_type_name and entity_type_name in type_info:
                    entity_data["type_threshold"] = type_info[entity_type_name]["threshold"]
                    entity_data["type_weight"] = type_info[entity_type_name]["weight"]
                else:
                    # 使用默认值
                    entity_data["type_threshold"] = 0.800
                    entity_data["type_weight"] = 1.0
                results.append(entity_data)
        else:
            results = search_results

        return results

    async def search_by_names_exact(
        self,
        names: List[str],
        source_config_ids: List[str],
        entity_types: Optional[List[str]] = None,
        size_per_name: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        按名称精确匹配搜索实体（批量）

        Args:
            names: 实体名称列表（精确匹配）
            source_config_ids: 信息源ID列表
            entity_types: 实体类型列表（可选，用于过滤）
            size_per_name: 每个名称最多返回的数量

        Returns:
            实体列表，包含 _score 字段
        """
        if not names or not source_config_ids:
            return []

        # 构建 bool 查询：names 使用 should（OR），source_config_ids 使用 filter
        should_clauses = [{"term": {"name.keyword": name}} for name in names]

        filter_clauses = []
        # 添加信息源过滤
        if len(source_config_ids) == 1:
            filter_clauses.append({"term": {"source_config_id": source_config_ids[0]}})
        else:
            filter_clauses.append({"terms": {"source_config_id": source_config_ids}})

        # 添加实体类型过滤（如果指定）
        if entity_types:
            if len(entity_types) == 1:
                filter_clauses.append({"term": {"type": entity_types[0]}})
            else:
                filter_clauses.append({"terms": {"type": entity_types}})

        query = {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1,
                "filter": filter_clauses
            }
        }

        # 计算总数量限制
        total_size = min(len(names) * size_per_name, 10000)

        # 使用 return_full_response=True 获取完整响应
        response = await self.es_client.search(
            index=self.INDEX_NAME,
            query=query,
            size=total_size,
            return_full_response=True,
        )

        results = []
        for hit in response.get("hits", []):
            entity_data = hit.get("source", {}).copy()
            entity_data["_score"] = hit.get("score", 1.0)
            results.append(entity_data)

        return results

    async def get_by_source(
        self, source_config_id: str, entity_type: Optional[str] = None, size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        获取信息源的所有实体

        Args:
            source_config_id: 信息源ID
            entity_type: 实体类型（可选）
            size: 返回数量

        Returns:
            实体列表
        """
        s = Search(using=self.es_client, index=self.INDEX_NAME)

        s = s.filter("term", source_config_id=source_config_id)

        if entity_type:
            s = s.filter("term", type=entity_type)

        s = s[:size]

        # 转换为字典并执行
        search_dict = s.to_dict()
        # 使用 routing 优化查询性能
        response = await self.es_client.search(
            index=self.INDEX_NAME,
            query=search_dict.get("query", {}),
            size=search_dict.get("size", 10),
            routing=source_config_id
        )
        return [hit["_source"] for hit in response["hits"]["hits"]]

    async def get_entities_by_ids(
        self, entity_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        根据实体ID列表批量获取实体信息（包含向量）

        Args:
            entity_ids: 实体ID列表

        Returns:
            实体详细信息列表，包含 entity_id, name, type, vector 字段
        """
        if not entity_ids:
            return []

        # Elasticsearch 默认的 max_result_window 是 10,000
        # 分批请求以避免超过这个限制
        BATCH_SIZE = 5000  # 使用保守的批次大小
        results = []

        for i in range(0, len(entity_ids), BATCH_SIZE):
            batch_ids = entity_ids[i:i + BATCH_SIZE]

            # 构建ES查询
            query = {
                "terms": {
                    "entity_id": batch_ids
                }
            }

            response = await self.es_client.search(
                index=self.INDEX_NAME,
                query=query,
                size=len(batch_ids),
                return_full_response=True,
            )

            for hit in response.get("hits", []):
                entity_data = hit.get("source", {}).copy()
                results.append(entity_data)

        return results

    async def batch_search_similar_by_ids(
        self,
        query_vector: List[float],
        entity_ids: List[str],
        source_config_ids: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        批量查询指定实体的向量相似度

        Args:
            query_vector: 查询向量
            entity_ids: 实体ID列表
            source_config_ids: 信息源ID列表

        Returns:
            {entity_id: similarity_score}
        """
        if not entity_ids:
            return {}

        # 分批处理，避免超过 ES 的 max_result_window 限制
        BATCH_SIZE = 5000
        all_scores = {}

        for i in range(0, len(entity_ids), BATCH_SIZE):
            batch_ids = entity_ids[i:i + BATCH_SIZE]

            # 构建过滤条件
            filters = [{"terms": {"entity_id": batch_ids}}]
            if source_config_ids:
                filters.append({"terms": {"source_config_id": source_config_ids}})

            filter_query = {"bool": {"must": filters}} if filters else None

            # 使用 vector_search
            results = await self.es_client.vector_search(
                index=self.INDEX_NAME,
                field="vector",
                vector=query_vector,
                size=len(batch_ids),
                filter_query=filter_query,
            )

            for hit in results:
                entity_id = hit.get("entity_id")
                score = hit.get("_score", 0.0)
                if entity_id:
                    all_scores[entity_id] = float(score)

        return all_scores
