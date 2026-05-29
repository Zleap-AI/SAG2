"""
文档解析器

基于智能切片器，提供 Token 级精确控制
仅支持 Markdown 格式文档加载
"""

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.exceptions import LoadError
from pipeline.modules.load.chunking import (
    ChunkingResult,
    MarkdownArticleSectionBuilder,
    MarkdownBlockParser,
    MarkdownInputNormalizer,
    MarkdownSourceChunkAssembler,
    RAGChunkingPipeline,
)
from pipeline.modules.load.chunking.types import (
    BlockType,
    ChunkDraft,
    InputDocument,
    SectionDraft,
    StructuredBlock,
)
from pipeline.utils import (
    TokenEstimator,
    get_logger,
    normalize_heading_text,
)

logger = get_logger("modules.load.parser")


class MarkdownParser:
    """Markdown 文档解析器（支持 standard / heading_strict 切块模式）"""

    def __init__(
        self,
        max_tokens: int = 1000,
        model_type: str = "generic",
        section_max_tokens: Optional[int] = None,
        chunk_mode: str = "standard",
    ) -> None:
        """
        初始化解析器

        Args:
            max_tokens: 每个 chunk 的最大 token 数量
            model_type: 用于 token 估算的模型类型
            section_max_tokens: ArticleSection token 软上限（默认自动推导）
            chunk_mode: 切块模式（standard / heading_strict / overlap）
                - standard: 贪婪聚合，跨标题合并小块
                - heading_strict: 遇到新标题强制断开，每个标题下独立成 chunk
                - overlap: 同 standard（暂未实现 overlap，等同 standard）
        """
        self.max_tokens = max_tokens
        self.chunk_mode = chunk_mode
        self.token_estimator = TokenEstimator(model_type)
        self.section_max_tokens = section_max_tokens or max(128, min(512, max_tokens // 4))
        self._last_chunking_result: Optional[ChunkingResult] = None
        heading_strict = (chunk_mode == "heading_strict")
        self.chunking_pipeline = RAGChunkingPipeline(
            input_normalizer=MarkdownInputNormalizer(),
            block_parser=MarkdownBlockParser(),
            section_builder=MarkdownArticleSectionBuilder(
                section_max_tokens=self.section_max_tokens,
                model_type=model_type,
            ),
            chunk_assembler=MarkdownSourceChunkAssembler(
                source_chunk_max_tokens=max_tokens,
                model_type=model_type,
                heading_strict=heading_strict,
            ),
        )

        # 标题正则表达式
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

        logger.info(
            "文档解析器初始化完成",
            extra={
                "max_tokens": max_tokens,
                "model_type": model_type,
                "section_max_tokens": self.section_max_tokens,
                "chunk_mode": chunk_mode,
            }
        )

    def parse_file(self, file_path: Path) -> tuple[str, int]:
        """
        解析 Markdown 文件

        Args:
            file_path: Markdown 文件路径（.md / .markdown）

        Returns:
            (完整内容, chunk 数量)

        Raises:
            LoadError: 文件读取失败
        """
        content, result = asyncio.run(self.parse_file_with_plan_async(file_path))
        return content, len(result.source_chunks)

    def parse_file_with_plan(self, file_path: Path) -> tuple[str, ChunkingResult]:
        """同步包装：解析文件并返回双层切片结果"""
        return asyncio.run(self.parse_file_with_plan_async(file_path))

    async def parse_file_async(self, file_path: Path) -> tuple[str, int]:
        """异步解析 Markdown 文件，返回 (content, chunk_count)"""
        content, result = await self.parse_file_with_plan_async(file_path)
        return content, len(result.source_chunks)

    async def parse_file_with_plan_async(self, file_path: Path) -> tuple[str, ChunkingResult]:
        """
        解析文件并返回双层切片结果（SectionDraft + SourceChunk）
        """
        try:
            logger.info(f"开始解析文件: {file_path.name} ({file_path.suffix})")

            if not file_path.exists():
                raise LoadError(f"文件不存在: {file_path}")

            file_suffix = file_path.suffix.lower()
            if file_suffix not in {'.md', '.markdown'}:
                raise LoadError(
                    f"不支持的文件格式: {file_path.suffix}，仅支持 .md / .markdown"
                )

            content = file_path.read_text(encoding="utf-8")

            result = await self.parse_content_with_plan_async(
                content,
                source_path=file_path.parent,
            )

            logger.info(
                f"文件解析完成: {file_path.name}",
                extra={
                    "article_sections": len(result.article_sections),
                    "source_chunks": len(result.source_chunks),
                },
            )
            return content, result

        except Exception as e:
            logger.error(f"文件解析失败: {file_path}: {e}", exc_info=True)
            raise LoadError(f"文件解析失败: {e}") from e

    def parse_content_with_plan(
        self,
        content: str,
        source_path: Optional[Path] = None,
    ) -> ChunkingResult:
        """
        同步包装：返回包含 SectionDraft + SourceChunk 的双层切片结果
        """
        return asyncio.run(
            self.parse_content_with_plan_async(
                content=content,
                source_path=source_path,
            )
        )

    async def parse_content_with_plan_async(
        self,
        content: str,
        source_path: Optional[Path] = None,
    ) -> ChunkingResult:
        """返回包含 SectionDraft + SourceChunk 的双层切片结果（异步）"""
        if self.chunk_mode == "heading_strict":
            result = self._parse_content_heading_strict(content, source_path)
        else:
            result = await self.chunking_pipeline.run_async(content, source_path=source_path)
        self._last_chunking_result = result
        return result

    def _parse_content_heading_strict(
        self,
        content: str,
        source_path: Optional[Path] = None,
    ) -> ChunkingResult:
        """
        旧版 heading_strict 切片：按标题边界切分，每个标题块整体作为一个 chunk，
        不对内容做句子级切分，保留原始空格连接的文本格式。
        """
        sections = self._extract_sections_heading_strict(content)
        source_chunks: List[ChunkDraft] = []
        section_drafts: List[SectionDraft] = []

        for idx, section in enumerate(sections):
            headings = section["headings"]
            content_lines = section["content_lines"]

            if headings:
                min_level = min(h[0] for h in headings)
                main_heading = next(h[1] for h in headings if h[0] == min_level)
                heading_content = "\n".join(h[2] for h in headings)
                content_text = "\n".join(content_lines).strip()
                if content_text:
                    full_content = heading_content + "\n" + content_text
                else:
                    full_content = heading_content
                normalized_heading = normalize_heading_text(main_heading)
            else:
                normalized_heading = ""
                full_content = "\n".join(content_lines).strip()

            if not full_content:
                continue

            section_drafts.append(
                SectionDraft(
                    order_index=idx,
                    render_group_index=idx,
                    heading=normalized_heading,
                    content=full_content,
                    raw_content=full_content,
                    section_type="TEXT",
                    metadata={"legacy_mode": "heading_strict"},
                )
            )
            source_chunks.append(
                ChunkDraft(
                    rank=idx,
                    heading=normalized_heading,
                    content=full_content,
                    raw_content=full_content,
                    chunk_type="TEXT",
                    section_order_indices=[idx],
                    metadata={"legacy_mode": "heading_strict"},
                )
            )

        doc = InputDocument(
            content=content,
            source_path=source_path,
            is_markdown=True,
            metadata={"legacy_mode": "heading_strict"},
        )
        blocks = [
            StructuredBlock(
                block_id="legacy-0",
                block_type=BlockType.TEXT,
                raw_content=content,
                heading="",
                start_index=0,
                end_index=len(content),
                metadata={"legacy_mode": "heading_strict"},
            )
        ]
        return ChunkingResult(
            input_doc=doc,
            blocks=blocks,
            article_sections=section_drafts,
            source_chunks=source_chunks,
        )

    def _extract_sections_heading_strict(self, content: str) -> List[Dict]:
        """
        Heading 严格模式下按标题切分章节（旧版 legacy 实现）。

        每遇到新标题就开始新章节，每个标题+其后内容整体保留，
        不做句子级分割，保持原始空格连接的文本格式。

        Returns:
            章节列表，每个章节格式：
            {"headings": [(level, title, heading_line), ...], "content_lines": [...]}
        """
        lines = content.split("\n")
        sections = []
        current_section: Dict = {"headings": [], "content_lines": []}

        for line in lines:
            heading_match = self.heading_pattern.match(line)
            if heading_match:
                if current_section["headings"] or current_section["content_lines"]:
                    sections.append(current_section)
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                current_section = {
                    "headings": [(level, title, line)],
                    "content_lines": [],
                }
            else:
                current_section["content_lines"].append(line)

        if current_section["headings"] or current_section["content_lines"]:
            sections.append(current_section)

        logger.debug(f"[Heading严格模式] 提取到 {len(sections)} 个章节")
        return sections

    def get_last_chunking_result(self) -> Optional[ChunkingResult]:
        """获取最近一次 parse 的双层切片结果"""
        return self._last_chunking_result

    def extract_title(self, content: str) -> str:
        """
        从 Markdown 内容中提取标题（第一个一级标题）

        Args:
            content: Markdown 文本

        Returns:
            标题，如果没有则返回 "Untitled"

        Example:
            >>> parser = MarkdownParser()
            >>> title = parser.extract_title("# My Title\\n\\nContent")
            >>> print(title)  # "My Title"
        """
        match = self.heading_pattern.search(content)
        if match:
            return normalize_heading_text(match.group(2))
        return "Untitled"

    def _normalize_heading(self, heading: Optional[str]) -> str:
        """规范化标题，避免原始 Markdown 标题行过长触发模型校验失败。"""
        return normalize_heading_text(heading)
