"""Markdown Block -> Section 编排（纯文本简化版）。"""

from __future__ import annotations

from typing import List

from pipeline.modules.load.chunking.base import BaseArticleSectionBuilder
from pipeline.modules.load.chunking.chunker.text import MarkdownTextChunker
from pipeline.modules.load.chunking.types import InputDocument, SectionDraft, StructuredBlock


class MarkdownArticleSectionBuilder(BaseArticleSectionBuilder):
    """纯文本 md：直接用 MarkdownTextChunker 处理所有 blocks。"""

    def __init__(
        self,
        section_max_tokens: int,
        model_type: str = "generic",
    ) -> None:
        self.section_max_tokens = max(32, section_max_tokens)
        self.chunker = MarkdownTextChunker(
            section_max_tokens=self.section_max_tokens,
            model_type=model_type,
        )

    async def build_sections(
        self,
        doc: InputDocument,
        blocks: List[StructuredBlock],
    ) -> List[SectionDraft]:
        _ = doc
        sections: List[SectionDraft] = []
        order_index = 0

        for block in blocks:
            block_sections = await self.chunker.build_sections(
                block=block,
                order_start=order_index,
                render_group_index=order_index + 1,
            )
            sections.extend(block_sections)
            order_index += len(block_sections)

        sections = self._merge_whitespace_only_text_sections(sections)
        for idx, section in enumerate(sections):
            section.order_index = idx
        return sections

    @classmethod
    def _merge_whitespace_only_text_sections(
        cls,
        sections: List[SectionDraft],
    ) -> List[SectionDraft]:
        """把"仅空白的 TEXT section"吸附到相邻 section，避免产生无语义 section。"""
        if len(sections) <= 1:
            return sections

        merged: List[SectionDraft] = []
        leading_whitespace = ""

        for section in sections:
            if cls._is_whitespace_only_text_section(section):
                ws = section.raw_content if section.raw_content is not None else section.content
                if merged:
                    cls._append_trailing_whitespace(merged[-1], ws)
                else:
                    leading_whitespace += ws
                continue

            if leading_whitespace:
                cls._prepend_leading_whitespace(section, leading_whitespace)
                leading_whitespace = ""
            merged.append(section)

        if leading_whitespace:
            if merged:
                cls._append_trailing_whitespace(merged[-1], leading_whitespace)
            else:
                return sections

        return merged

    @staticmethod
    def _is_whitespace_only_text_section(section: SectionDraft) -> bool:
        raw = section.raw_content if section.raw_content is not None else section.content
        return raw.strip() == ""

    @staticmethod
    def _append_trailing_whitespace(section: SectionDraft, whitespace: str) -> None:
        if not whitespace:
            return
        section.raw_content = f"{section.raw_content}{whitespace}"
        section.content = f"{section.content}{whitespace}"

    @staticmethod
    def _prepend_leading_whitespace(section: SectionDraft, whitespace: str) -> None:
        if not whitespace:
            return
        section.raw_content = f"{whitespace}{section.raw_content}"
        section.content = f"{whitespace}{section.content}"
