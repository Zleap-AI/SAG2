"""
文档加载器

负责加载文档、调用解析器和处理器、保存到数据库
"""

from abc import ABC, abstractmethod
import asyncio
import inspect
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from sqlalchemy import delete, insert, select, update
from sqlalchemy.exc import OperationalError

from pipeline.db import (
    Article,
    ArticleParseStatus,
    ArticleSection,
    SourceEvent,
    SourceChunk,
    SourceConfig,
    get_session_factory,
)
from pipeline.exceptions import LoadError
from pipeline.modules.load.config import DocumentLoadConfig, LoadResult
from pipeline.modules.load.chunking import ChunkingResult
from pipeline.modules.load.parser import MarkdownParser
from pipeline.modules.load.processor import DocumentProcessor
from pipeline.utils import estimate_tokens, get_logger, normalize_heading_text, is_retryable_error
import uuid

logger = get_logger("modules.load.loader")


class BaseLoader(ABC):
    """加载器基类"""

    def __init__(
        self,
        processor: Optional[DocumentProcessor] = None,
        progress_callback: Optional[Callable[[str], Optional[Awaitable[None]]]] = None,
    ) -> None:
        """
        初始化加载器

        Args:
            processor: 文档处理器（如果不提供，使用默认）
        """
        self.processor = processor or DocumentProcessor()
        self._progress_callback = progress_callback
        self.session_factory = get_session_factory()
        logger.info(f"{self.__class__.__name__} 初始化完成")

    @abstractmethod
    async def load(self, config) -> LoadResult:
        """
        加载数据（主入口方法）

        Args:
            config: 加载配置对象

        Returns:
            LoadResult（包含source_id和chunk_ids）
        """
        pass

    @abstractmethod
    async def _save_to_database(self, *args, **kwargs) -> tuple[str, List[str]]:
        """
        保存到数据库

        Returns:
            (source_id, chunk_ids)
        """
        pass

    async def _generate_embedding(self, text: str):
        """
        生成向量（委托给 processor）

        Args:
            text: 文本内容

        Returns:
            向量数组
        """
        return await self.processor.generate_embedding(text)

    async def _notify_progress(self, message: str):
        """通知外层当前阶段进度消息"""
        if not self._progress_callback:
            return
        try:
            result = self._progress_callback(message)
            if inspect.isawaitable(result):
                await result
        except Exception as e:  # noqa: BLE001
            logger.warning(f"进度通知失败: {e}")

    async def _index_source_chunks_to_es(
        self, source_id: str, source_type: str
    ) -> None:
        """
        索引 SourceChunk 到 Elasticsearch（通用方法）

        Args:
            source_id: 源ID (UUID)
            source_type: 源类型 ("ARTICLE" 或 "CHAT")
        """
        try:
            from pipeline.core.storage import SourceChunkRepository, ElasticsearchClient

            # 创建 ES 客户端
            es_client_wrapper = ElasticsearchClient()
            repo = SourceChunkRepository(es_client_wrapper.client)

            async with self.session_factory() as session:
                # 获取所有 SourceChunk
                stmt = (
                    select(SourceChunk)
                    .where(
                        SourceChunk.source_id == source_id,
                        SourceChunk.source_type == source_type,
                    )
                    .order_by(SourceChunk.rank)
                )
                result = await session.execute(stmt)
                chunks = result.scalars().all()

                if not chunks:
                    logger.warning(
                        f"源没有 SourceChunk: {source_id} (type={source_type})"
                    )
                    return

                # 批量处理配置
                embedding_batch_size = getattr(self, '_embedding_batch_size', 10)
                es_bulk_size = getattr(self, '_es_bulk_index_size', 50)

                stats = await self._batch_index_chunks(
                    chunks=chunks,
                    repo=repo,
                    es_client=es_client_wrapper,
                    embedding_batch_size=embedding_batch_size,
                    es_bulk_size=es_bulk_size,
                    source_config_id=chunks[0].source_config_id
                )
                logger.info(
                    f"SourceChunk 批量索引完成: {source_id} (type={source_type})",
                    extra=stats
                )

        except Exception as e:
            logger.error(f"索引失败: {source_id}: {e}", exc_info=True)
        finally:
            # 确保 ES 客户端被关闭
            try:
                await es_client_wrapper.client.close()
            except Exception as close_err:
                logger.warning(f"关闭ES客户端失败: {close_err}")

    async def _batch_index_chunks(
        self,
        chunks: List,
        repo,
        es_client,
        embedding_batch_size: int,
        es_bulk_size: int,
        source_config_id: str
    ) -> Dict[str, Any]:
        """
        批量处理 chunks 的向量生成和ES索引

        Args:
            chunks: SourceChunk 列表
            repo: SourceChunkRepository 实例
            es_client: ElasticsearchClient 包装实例
            embedding_batch_size: 向量生成批量大小
            es_bulk_size: ES索引批量大小
            source_config_id: 信息源配置ID（用于路由）

        Returns:
            统计信息字典
        """
        import time
        from pipeline.core.ai.factory import get_embedding_client

        start_time = time.perf_counter()
        embedding_client = await get_embedding_client(scenario='general')

        documents = []
        embedding_failed = 0

        # === 阶段1: 批量生成向量 ===
        for i in range(0, len(chunks), embedding_batch_size):
            batch_chunks = chunks[i:i+embedding_batch_size]

            try:
                # 准备文本
                heading_texts = [c.heading for c in batch_chunks if c.heading]
                content_texts = [
                    f"{c.heading}\n\n{c.content[:1024]}"
                    for c in batch_chunks
                ]

                # 批量生成向量
                heading_vectors = []
                if heading_texts:
                    heading_vectors = await embedding_client.batch_generate(heading_texts)

                content_vectors = await embedding_client.batch_generate(content_texts)

                # 构建文档列表
                heading_idx = 0
                for j, chunk in enumerate(batch_chunks):
                    heading_vec = None
                    if chunk.heading and heading_idx < len(heading_vectors):
                        heading_vec = heading_vectors[heading_idx]
                        heading_idx += 1

                    doc = {
                        "id": chunk.id,
                        "chunk_id": chunk.id,
                        "source_id": chunk.source_id,
                        "source_config_id": chunk.source_config_id,
                        "rank": chunk.rank,
                        "heading": chunk.heading,
                        "content": chunk.content,
                        "heading_vector": heading_vec,
                        "content_vector": content_vectors[j],
                        "references": chunk.references,
                        "chunk_type": "TEXT",
                        "content_length": chunk.chunk_length,
                    }
                    documents.append(doc)

            except Exception as e:
                logger.warning(f"批量生成向量失败，降级重试: {e}")
                # 降级：逐个重试
                for chunk in batch_chunks:
                    try:
                        heading_vec = None
                        if chunk.heading:
                            heading_vec = await self._generate_embedding(chunk.heading)

                        content_vec = await self._generate_embedding(
                            f"{chunk.heading}\n\n{chunk.content[:1024]}"
                        )

                        doc = {
                            "id": chunk.id,
                            "chunk_id": chunk.id,
                            "source_id": chunk.source_id,
                            "source_config_id": chunk.source_config_id,
                            "rank": chunk.rank,
                            "heading": chunk.heading,
                            "content": chunk.content,
                            "heading_vector": heading_vec,
                            "content_vector": content_vec,
                            "references": chunk.references,
                            "chunk_type": "TEXT",
                            "content_length": chunk.chunk_length,
                        }
                        documents.append(doc)
                    except Exception as retry_e:
                        logger.error(f"单条生成向量失败: {chunk.id}: {retry_e}")
                        embedding_failed += 1
                        # 记录是否可重试
                        if is_retryable_error(retry_e):
                            logger.warning(f"向量生成失败（可重试）: {chunk.id}")
                        else:
                            logger.error(f"向量生成失败（不可重试）: {chunk.id}")

        # === 阶段2: 批量索引到ES ===
        indexed = 0
        es_failed = 0

        for i in range(0, len(documents), es_bulk_size):
            batch = documents[i:i+es_bulk_size]

            try:
                # 批量索引
                result = await es_client.bulk_index(
                    index=repo.INDEX_NAME,
                    documents=batch,
                    return_details=True,
                    routing=source_config_id
                )

                indexed += result["success_count"]

                # 处理失败项：逐个重试
                if result["error_count"] > 0:
                    failed_ids = {err["id"] for err in result["errors"]}
                    for doc in batch:
                        if doc["id"] in failed_ids:
                            try:
                                await es_client.index_document(
                                    index=repo.INDEX_NAME,
                                    document=doc,
                                    doc_id=doc["id"],
                                    routing=source_config_id
                                )
                                indexed += 1
                            except Exception as retry_e:
                                logger.error(f"重试索引失败: {doc['id']}: {retry_e}")
                                es_failed += 1

            except Exception as e:
                logger.error(f"批量索引失败，降级重试: {e}")
                # 降级：整批逐个重试
                for doc in batch:
                    try:
                        await es_client.index_document(
                            index=repo.INDEX_NAME,
                            document=doc,
                            doc_id=doc["id"],
                            routing=source_config_id
                        )
                        indexed += 1
                    except Exception as retry_e:
                        logger.error(f"降级索引失败: {doc['id']}: {retry_e}")
                        es_failed += 1
                        # 记录是否可重试
                        if is_retryable_error(retry_e):
                            logger.warning(f"ES索引失败（可重试）: {doc['id']}")
                        else:
                            logger.error(f"ES索引失败（不可重试）: {doc['id']}")

        total_time = time.perf_counter() - start_time

        return {
            "total_chunks": len(chunks),
            "indexed_count": indexed,
            "embedding_failed": embedding_failed,
            "es_failed": es_failed,
            "embedding_batches": (len(chunks) + embedding_batch_size - 1) // embedding_batch_size,
            "es_batches": (len(documents) + es_bulk_size - 1) // es_bulk_size,
            "total_time": f"{total_time:.2f}s",
            "avg_time": f"{total_time/len(chunks):.3f}s/chunk"
        }


class DocumentLoader(BaseLoader):
    """文档加载器"""

    def __init__(
        self,
        parser: Optional[MarkdownParser] = None,
        processor: Optional[DocumentProcessor] = None,
        max_tokens: Optional[int] = None,
        min_content_length: int = 100,
        merge_short_sections: bool = True,
        chunk_mode: str = "standard",
        progress_callback: Optional[Callable[[str], Optional[Awaitable[None]]]] = None,
    ) -> None:
        """
        初始化文档加载器

        Args:
            parser: 文档解析器（如果不提供，使用默认）
            processor: 文档处理器（如果不提供，使用默认）
            max_tokens: 最大token数（用于创建默认parser）
            min_content_length: 最小内容长度（用于创建默认parser）
            merge_short_sections: 是否启用短片段合并（用于创建默认parser）
            chunk_mode: 切块模式（用于创建默认parser）
        """
        # 调用父类初始化
        super().__init__(processor=processor, progress_callback=progress_callback)

        # 创建 parser（如果未提供）
        if parser is not None:
            self.parser = parser
        else:
            parser_params = {}
            if max_tokens is not None:
                parser_params["max_tokens"] = max_tokens
            self.parser = MarkdownParser(**parser_params)

    async def _mark_article_parse_failed(self, article_id: Optional[str], error: str) -> None:
        """将文章 parse_status 标记为 EXTRACTION_FAILED（best effort）。"""
        if not article_id:
            return

        try:
            async with self.session_factory() as session:
                article = await session.get(Article, article_id)
                if not article:
                    return
                article.parse_status = ArticleParseStatus.EXTRACTION_FAILED.value
                article.error = error
                await session.commit()
        except Exception as update_err:  # noqa: BLE001
            logger.warning(
                "更新文章解析失败状态失败: article_id=%s, error=%s",
                article_id,
                update_err,
            )

    async def load(self, config: DocumentLoadConfig) -> LoadResult:
        """
        加载文档（主入口方法）

        Args:
            config: DocumentLoadConfig配置对象

        Returns:
            LoadResult（包含article_id和chunk_ids）

        Example:
            >>> config = DocumentLoadConfig(
            ...     source_config_id="source-uuid",
            ...     path="doc.md",
            ...     background="技术文档"
            ... )
            >>> result = await loader.load(config)
            >>> # result.source_id, result.chunk_ids
        """
        # 保存批量处理配置到实例变量
        self._enable_batch_indexing = config.enable_batch_indexing
        self._embedding_batch_size = config.embedding_batch_size
        self._es_bulk_index_size = config.es_bulk_index_size

        if not config.path:
            raise LoadError("文件加载模式必须提供 path")

        path = config.path if isinstance(config.path, Path) else Path(config.path)

        if not path.is_file():
            raise LoadError(f"不是文件: {path}")

        return await self.load_file(
            file_path=path,
            source_config_id=config.source_config_id,
            background=config.background or "",
            auto_vector=config.auto_vector,
            max_tokens=config.max_tokens,
            min_content_length=config.min_content_length,
            merge_short_sections=config.merge_short_sections,
            chunk_mode=config.chunk_mode,
        )

    async def load_file(
        self,
        file_path: Path,
        source_config_id: str,
        background: str = "",
        auto_vector: bool = True,
        max_tokens: Optional[int] = None,
        min_content_length: Optional[int] = None,
        merge_short_sections: Optional[bool] = None,
        chunk_mode: Optional[str] = None,
    ) -> LoadResult:
        """
        加载文档文件

        Args:
            file_path: 文件路径
            source_config_id: 信息源ID
            background: 背景信息
            auto_vector: 是否自动索引到Elasticsearch
            max_tokens: 每个片段的最大token数
            min_content_length: 最小内容长度
            merge_short_sections: 是否合并短片段
            chunk_mode: 切块模式

        Returns:
            LoadResult（包含article_id和chunk_ids）

        Raises:
            LoadError: 加载失败
        """
        article_id = None
        try:
            logger.info(f"开始加载文档: {file_path}")

            # 1. 检查文件
            if not file_path.exists():
                raise LoadError(f"文件不存在: {file_path}")

            if not file_path.is_file():
                raise LoadError(f"不是文件: {file_path}")

            # 预创建 Article 记录
            article_id = str(uuid.uuid4())
            async with self.session_factory() as session:
                article_orm = Article(
                    id=article_id,
                    source_config_id=source_config_id,
                    title=normalize_heading_text(file_path.stem) or "Untitled",
                    status="PENDING",
                )
                session.add(article_orm)
                await session.commit()

            # 2. 解析文档（根据配置参数）
            await self._notify_progress("正在切块")
            parser_params = {}
            if max_tokens is not None and max_tokens != 8000:
                parser_params["max_tokens"] = max_tokens
            if chunk_mode is not None:
                parser_params["chunk_mode"] = chunk_mode

            chunking_result: Optional[ChunkingResult] = None
            if parser_params:
                # 创建临时 parser 使用指定参数
                parser = MarkdownParser(**parser_params)
                content, section_count = await parser.parse_file_async(file_path)
                chunking_result = parser.get_last_chunking_result()
            else:
                content, section_count = await self.parser.parse_file_async(file_path)
                chunking_result = self.parser.get_last_chunking_result()

            logger.info(f"文档解析完成，共{section_count}个章节")

            # 3. 提取标题
            title = self.parser.extract_title(content)

            # 4. 保存到数据库
            article_id, chunk_ids = await self._save_to_database(
                title=title,
                content=content,
                source_config_id=source_config_id,
                article_id=article_id,
                chunking_result=chunking_result,
            )

            logger.info(
                f"文档加载完成: {title}",
                extra={
                    "article_id": article_id,
                    "chunk_count": len(chunk_ids),
                    "file_path": str(file_path),
                },
            )

            # 5. 索引到Elasticsearch（可选）
            if auto_vector:
                await self._index_to_elasticsearch(article_id)

            # 6. 返回LoadResult
            return LoadResult(
                source_id=article_id,
                source_type="ARTICLE",
                chunk_ids=chunk_ids,
                source_config_id=source_config_id,
                title=title,
                chunk_count=len(chunk_ids),
                extra={
                    "file_path": str(file_path),
                    "section_count": section_count,
                }
            )

        except Exception as e:
            if article_id:
                await self._mark_article_parse_failed(article_id, str(e))
            logger.error(f"文档加载失败: {file_path}: {e}", exc_info=True)

            # 区分可重试和不可重试错误
            if is_retryable_error(e):
                logger.warning(f"文档加载失败（可重试）: {file_path}")
            else:
                logger.error(f"文档加载失败（不可重试）: {file_path}")

            if isinstance(e, LoadError):
                raise
            raise LoadError(f"文档加载失败: {e}") from e

    async def _save_to_database(
        self,
        title: str,
        content: str,
        source_config_id: str,
        article_id: str,
        chunking_result: ChunkingResult,
        document_id_for_binding: Optional[str] = None,
    ) -> tuple[str, List[str]]:
        """
        保存文章、SourceChunk 和 ArticleSection 到数据库

        Args:
            title: 文章标题
            content: 文章正文（完整 markdown）
            source_config_id: 信息源ID
            article_id: 文章ID（必须已存在的记录）
            chunking_result: 切片框架结果
            document_id_for_binding: 可选文档ID（绑定到 Article.source_id）

        Returns:
            (article_id, chunk_ids)
        """
        max_retries = 3  # 死锁重试次数
        batch_size = 100  # 批量插入大小

        # ── 事务外预计算：构建所有待插入数据，避免持锁期间做 CPU 密集操作 ──
        all_section_data = []
        section_id_by_order: Dict[int, str] = {}

        for section_draft in chunking_result.article_sections:
            section_id = str(uuid.uuid4())
            section_id_by_order[section_draft.order_index] = section_id
            image_url = None
            if section_draft.section_type == "IMAGE":
                image_url = (section_draft.metadata or {}).get("image_src")
            section_extra_data = dict(section_draft.metadata or {})
            section_extra_data["token_count"] = max(
                0, estimate_tokens(section_draft.content or "")
            )
            all_section_data.append(
                {
                    "id": section_id,
                    "article_id": article_id,
                    "order_index": section_draft.order_index,
                    "render_group_index": section_draft.render_group_index,
                    "type": section_draft.section_type,
                    "rank": section_draft.order_index,
                    "heading": normalize_heading_text(section_draft.heading),
                    "content": section_draft.content or "",
                    "raw_content": section_draft.raw_content,
                    "image_url": image_url,
                    "length": len(section_draft.content or ""),
                    "extra_data": section_extra_data,
                }
            )

        chunk_ids = []
        all_chunk_data = []
        for chunk_draft in chunking_result.source_chunks:
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)
            references = [
                section_id_by_order[idx]
                for idx in chunk_draft.section_order_indices
                if idx in section_id_by_order
            ]
            all_chunk_data.append(
                {
                    "id": chunk_id,
                    "source_type": "ARTICLE",
                    "source_id": article_id,
                    "source_config_id": source_config_id,
                    "article_id": article_id,
                    "heading": normalize_heading_text(chunk_draft.heading),
                    "content": chunk_draft.content,
                    "raw_content": chunk_draft.raw_content,
                    "rank": chunk_draft.rank,
                    "chunk_length": len(chunk_draft.content or ""),
                    "references": references,
                    "extra_data": chunk_draft.metadata or {},
                }
            )

        total_sentences = len(all_section_data)

        # ── 事务内：仅做 DB 操作，最小化持锁时间 ──
        for attempt in range(max_retries):
            try:
                async with self.session_factory() as session:
                    # 检查信息源是否存在
                    source = await session.get(SourceConfig, source_config_id)
                    if not source:
                        raise LoadError(f"信息源不存在: {source_config_id}")

                    if not article_id:
                        raise LoadError("article_id 不能为空：当前仅支持更新已存在 Article")

                    article = await session.get(Article, article_id)
                    if not article:
                        raise LoadError(f"文章不存在，禁止新建: {article_id}")

                    # 更新现有 Article
                    article.title = normalize_heading_text(title) or "Untitled"
                    article.summary = None
                    article.content = content
                    article.category = None
                    article.tags = None
                    article.error = None
                    if document_id_for_binding:
                        article.source_id = document_id_for_binding

                    # 删除旧的 SourceChunk 和 ArticleSection，并软删除关联的 SourceEvent
                    stmt_chunk = delete(SourceChunk).where(
                        SourceChunk.source_id == article_id,
                        SourceChunk.source_type == "ARTICLE"
                    )
                    await session.execute(stmt_chunk)

                    stmt_section = delete(ArticleSection).where(
                        ArticleSection.article_id == article_id)
                    await session.execute(stmt_section)

                    # 软删除旧事项（与 ArticleSection 物理删除保持同步）
                    await session.execute(
                        update(SourceEvent)
                        .where(
                            SourceEvent.article_id == article_id,
                            SourceEvent.not_deleted(),
                        )
                        .values(status="DELETED")
                    )

                    # 批量插入预计算好的 section 和 chunk 数据
                    if all_section_data:
                        for i in range(0, len(all_section_data), batch_size):
                            batch = all_section_data[i:i + batch_size]
                            stmt = insert(ArticleSection).values(batch)
                            await session.execute(stmt)

                    if all_chunk_data:
                        for i in range(0, len(all_chunk_data), batch_size):
                            batch = all_chunk_data[i:i + batch_size]
                            stmt = insert(SourceChunk).values(batch)
                            await session.execute(stmt)

                    await session.commit()

                    logger.info(
                        f"文章保存成功",
                        extra={
                            "article_id": article.id,
                            "chunk_count": len(chunk_ids),
                            "total_sentences": total_sentences,
                        },
                    )

                    return article.id, chunk_ids

            except OperationalError as e:
                err_msg = str(e)
                is_retryable = "Deadlock" in err_msg or "Lock wait timeout" in err_msg
                if is_retryable and attempt < max_retries - 1:
                    wait_time = 1.0 * (2 ** attempt)  # 指数退避: 1s, 2s, 4s
                    logger.warning(
                        f"数据库锁冲突（可重试），{wait_time}s 后重试 (attempt {attempt + 1}/{max_retries}): {err_msg}"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                # 不可重试的数据库错误
                logger.error(f"数据库操作失败（不可重试）: {err_msg}")
                raise

    async def _index_to_elasticsearch(self, article_id: str) -> None:
        """
        索引文章 SourceChunk 到 Elasticsearch

        Args:
            article_id: 文章ID (UUID)
        """
        # 调用父类的通用索引方法
        await self._index_source_chunks_to_es(article_id, "ARTICLE")
