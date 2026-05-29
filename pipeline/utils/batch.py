"""
批量处理工具模块

提供向量生成和ES索引的批量处理逻辑
"""

import time
from typing import Any, Callable, Dict, List, Optional, TypeVar

from pipeline.utils import get_logger, is_retryable_error

logger = get_logger("utils.batch")

T = TypeVar('T')


class BatchProcessor:
    """批量处理器基类"""

    def __init__(
        self,
        batch_size: int = 10,
        logger_name: Optional[str] = None,
    ):
        """
        初始化批量处理器

        Args:
            batch_size: 批量大小
            logger_name: 日志记录器名称
        """
        self.batch_size = batch_size
        self.logger = get_logger(logger_name or "utils.batch")


class EmbeddingBatchProcessor(BatchProcessor):
    """向量生成批量处理器"""

    async def process(
        self,
        items: List[T],
        text_extractor: Callable[[T], str],
        embedding_client,
        on_success: Optional[Callable[[T, List[float]], Any]] = None,
    ) -> Dict[str, Any]:
        """
        批量生成向量

        Args:
            items: 待处理项列表
            text_extractor: 文本提取函数 (item -> text)
            embedding_client: 向量客户端
            on_success: 成功回调 (item, vector) -> result

        Returns:
            统计信息字典 {total, success, failed, results}
        """
        start_time = time.perf_counter()
        results = []
        failed_count = 0

        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]

            try:
                texts = [text_extractor(item) for item in batch]
                vectors = await embedding_client.batch_generate(texts)

                # 验证返回数量
                if len(vectors) != len(batch):
                    raise ValueError(
                        f"batch_generate returned mismatched vector count: "
                        f"expected={len(batch)}, actual={len(vectors)}"
                    )

                # 处理成功项
                for item, vector in zip(batch, vectors):
                    if on_success:
                        result = on_success(item, vector)
                        results.append(result)

            except Exception as e:
                self.logger.warning(f"批量生成向量失败，降级重试: {e}")
                # 降级：逐个重试
                for item in batch:
                    try:
                        text = text_extractor(item)
                        vector = await embedding_client.generate(text)
                        if on_success:
                            result = on_success(item, vector)
                            results.append(result)
                    except Exception as retry_e:
                        self.logger.error(f"单条生成向量失败: {retry_e}")
                        failed_count += 1
                        # 记录是否可重试
                        if is_retryable_error(retry_e):
                            self.logger.warning(f"向量生成失败（可重试）")
                        else:
                            self.logger.error(f"向量生成失败（不可重试）")

        total_time = time.perf_counter() - start_time

        return {
            "total": len(items),
            "success": len(results),
            "failed": failed_count,
            "results": results,
            "time": f"{total_time:.2f}s",
        }


class ESBulkIndexProcessor(BatchProcessor):
    """ES批量索引���理器"""

    async def process(
        self,
        documents: List[Dict[str, Any]],
        es_client,
        index_name: str,
        routing: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        批量索引到ES

        Args:
            documents: 文档列表
            es_client: ES客户端
            index_name: 索引名称
            routing: 路由值

        Returns:
            统计信息字典 {total, indexed, failed}
        """
        start_time = time.perf_counter()
        indexed = 0
        failed = 0

        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]

            try:
                result = await es_client.bulk_index(
                    index=index_name,
                    documents=batch,
                    return_details=True,
                    routing=routing,
                )

                indexed += result["success_count"]

                if result["error_count"] > 0:
                    failed_ids = {err["id"] for err in result["errors"]}
                    for doc in batch:
                        if doc["id"] in failed_ids:
                            try:
                                await es_client.index_document(
                                    index=index_name,
                                    document=doc,
                                    doc_id=doc["id"],
                                    routing=routing,
                                )
                                indexed += 1
                            except Exception as retry_e:
                                self.logger.error(f"重试索引失败: {doc['id']}: {retry_e}")
                                failed += 1

            except Exception as e:
                self.logger.error(f"批量索引失败，降级重试: {e}")
                # 降级：整批逐个重试
                for doc in batch:
                    try:
                        await es_client.index_document(
                            index=index_name,
                            document=doc,
                            doc_id=doc["id"],
                            routing=routing,
                        )
                        indexed += 1
                    except Exception as retry_e:
                        self.logger.error(f"降级索引失败: {doc['id']}: {retry_e}")
                        failed += 1
                        # 记录是否可重试
                        if is_retryable_error(retry_e):
                            self.logger.warning(f"ES索引失败（可重试）: {doc['id']}")
                        else:
                            self.logger.error(f"ES索引失败（不可重试）: {doc['id']}")

        total_time = time.perf_counter() - start_time

        return {
            "total": len(documents),
            "indexed": indexed,
            "failed": failed,
            "time": f"{total_time:.2f}s",
        }


async def batch_generate_embeddings(
    items: List[T],
    text_extractor: Callable[[T], str],
    embedding_client,
    batch_size: int = 10,
    on_success: Optional[Callable[[T, List[float]], Any]] = None,
) -> Dict[str, Any]:
    """
    批量生成向量（便捷函数）

    Args:
        items: 待处理项列表
        text_extractor: 文本提取函数 (item -> text)
        embedding_client: 向量客户端
        batch_size: 批量大小
        on_success: 成功回调 (item, vector) -> result

    Returns:
        统计信息字典
    """
    processor = EmbeddingBatchProcessor(batch_size=batch_size)
    return await processor.process(items, text_extractor, embedding_client, on_success)


async def batch_index_to_es(
    documents: List[Dict[str, Any]],
    es_client,
    index_name: str,
    batch_size: int = 50,
    routing: Optional[str] = None,
) -> Dict[str, Any]:
    """
    批量索引到ES（便捷函数）

    Args:
        documents: 文档列表
        es_client: ES客户端
        index_name: 索引名称
        batch_size: 批量大小
        routing: 路由值

    Returns:
        统计信息字典
    """
    processor = ESBulkIndexProcessor(batch_size=batch_size)
    return await processor.process(documents, es_client, index_name, routing)
