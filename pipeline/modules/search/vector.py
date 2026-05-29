"""
向量检索器

独立于三阶段的向量检索器，直接使用 Query 向量检索 Event/Chunk，
支持 title/heading 和 content 向量的混合搜索。

使用示例：
    from pipeline.modules.search import VectorSearcher, VectorConfig

    config = VectorConfig(
        return_type="event",
        top_k=20,
        title_weight=0.3,
        content_weight=0.7,
        similarity_threshold=0.4
    )

    searcher = VectorSearcher()
    events = await searcher.search(
        query="人工智能技术发展",
        source_config_ids=["source_1", "source_2"],
        config=config
    )
"""

import time
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload

from pipeline.core.storage.elasticsearch import get_es_client
from pipeline.db import SourceEvent, SourceChunk, get_session_factory
from pipeline.modules.load.processor import DocumentProcessor
from pipeline.modules.search.config import VectorConfig
from pipeline.utils import get_logger

logger = get_logger("search.vector")


class VectorSearcher:
    """
    向量检索器

    支持段落(SourceChunk)和事项(SourceEvent)的混合向量搜索。
    使用 ES script_score 查询实现 title/heading + content 的混合相似度计算。
    """

    INDEX_EVENTS = "event_vectors"
    INDEX_CHUNKS = "source_chunks"

    def __init__(self):
        """初始化向量检索器"""
        self.es_client = get_es_client()
        self.session_factory = get_session_factory()
        self.processor = DocumentProcessor()

    async def search_chunks_for_rerank(
        self,
        query: str,
        source_config_ids: List[str],
        query_vector: Optional[List[float]] = None,
        config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        执行向量搜索，返回格式与 reranker 一致

        专门用于接入 SAGSearcher 的 _rerank 流程。
        使用 ES kNN 向量搜索（避免 script_score 的编译限制）。

        Args:
            query: 查询文本
            source_config_ids: 信息源ID列表
            query_vector: 可选的预计算向量（避免重复计算）
            config: SearchConfig 对象

        Returns:
            {
                "sections": [...],  # 段落列表，按相似度降序
                "_timings": {...}   # 耗时统计
            }
        """
        start_time = time.perf_counter()

        # 从 SearchConfig 获取参数
        top_k = 20
        min_score = 0.0

        if config:
            if hasattr(config, 'rerank') and config.rerank:
                top_k = getattr(config.rerank, 'max_results', top_k)
                min_score = getattr(config.rerank, 'score_threshold', min_score)

        logger.info("=" * 60)
        logger.info(f"【向量检索-Rerank】Query: '{query}'")
        logger.info(f"  top_k={top_k}, min_score={min_score}")
        logger.info("=" * 60)

        # Step 1: 生成查询向量
        vector_time = 0.0
        if query_vector is None:
            vector_start = time.perf_counter()
            query_vector = await self.processor.generate_embedding(query)
            vector_time = time.perf_counter() - vector_start
            logger.info(f"✓ 向量生成完成，维度={len(query_vector)}，耗时={vector_time:.3f}s")
        else:
            logger.info(f"✓ 使用预计算向量，维度={len(query_vector)}")

        # Step 2: 使用 SourceChunkRepository 的 kNN 搜索（避免 script_score 编译限制）
        from pipeline.core.storage.repositories.source_chunk_repository import SourceChunkRepository
        from pipeline.core.storage.elasticsearch import get_es_client

        es_client = get_es_client()
        chunk_repo = SourceChunkRepository(es_client)

        es_start = time.perf_counter()
        es_results = await chunk_repo.search_similar_by_content(
            query_vector=query_vector,
            k=top_k,
            source_config_ids=source_config_ids,
        )
        es_time = time.perf_counter() - es_start
        logger.info(f"✓ ES kNN 搜索完成，命中 {len(es_results)} 个段落，耗时={es_time:.3f}s")

        if not es_results:
            logger.info("【向量检索-Rerank】未找到匹配段落")
            total_time = time.perf_counter() - start_time
            return {
                "sections": [],
                "_timings": {
                    "vector_gen": vector_time,
                    "es_search": es_time,
                    "total": total_time,
                }
            }

        # Step 3: 格式化结果（ES kNN 直接返回完整文档）
        sections = []
        for result in es_results:
            score = result.get("_score", 0.0)
            if score < min_score:
                continue

            sections.append({
                "chunk_id": result.get("chunk_id"),
                "source_id": result.get("source_id"),
                "source_config_id": result.get("source_config_id"),
                "heading": result.get("heading"),
                "content": result.get("content"),
                "rank": result.get("rank"),
                "score": score,
                "weight": score,
            })

        # 按分数排序并截取 top_k
        sections = sorted(sections, key=lambda x: x["score"], reverse=True)[:top_k]

        total_time = time.perf_counter() - start_time

        logger.info("=" * 60)
        logger.info(f"【向量检索-Rerank】完成，返回 {len(sections)} 个段落，总耗时={total_time:.3f}s")
        logger.info("=" * 60)

        # Top-5 日志
        for i, sec in enumerate(sections[:5]):
            heading = sec.get("heading", "")[:40] if sec.get("heading") else "无标题"
            logger.info(f"  Top-{i+1}: score={sec['score']:.4f} | {heading}...")

        return {
            "sections": sections,
            "_timings": {
                "vector_gen": vector_time,
                "es_search": es_time,
                "total": total_time,
            }
        }


__all__ = ["VectorSearcher"]
