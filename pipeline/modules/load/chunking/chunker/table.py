"""Markdown 表格块切分。"""

from __future__ import annotations

from typing import List

from pipeline.modules.load.chunking.chunker.base import BaseBlockChunker
from pipeline.modules.load.chunking.types import BlockType, SectionDraft, StructuredBlock


class MarkdownTableChunker(BaseBlockChunker):
    """TABLE block 切分器。"""

    async def build_sections(
        self,
        block: StructuredBlock,
        order_start: int,
        render_group_index: int,
    ) -> List[SectionDraft]:
        table_format = str(block.metadata.get("table_format", "")).lower()
        if table_format == "html":
            return [self._build_table_raw_section(block, order_start, render_group_index, "html_table")]

        lines = block.raw_content.splitlines(keepends=True)
        if len(lines) < 2:
            return [self._build_table_raw_section(block, order_start, render_group_index, "markdown_table")]

        sections: List[SectionDraft] = []
        header_raw = "".join(lines[:2])
        sections.append(
            SectionDraft(
                order_index=order_start,
                render_group_index=render_group_index,
                heading=block.heading,
                content=header_raw,
                raw_content=header_raw,
                section_type=BlockType.TABLE.value,
                metadata={
                    "block_type": BlockType.TABLE.value,
                    "render_format": "markdown_table",
                    "role": "table_header",
                    "row_index": 0,
                    "block_id": block.block_id,
                    "assemble_policy": "table_with_header",
                },
            )
        )
        for idx, row in enumerate(lines[2:], start=1):
            sections.append(
                SectionDraft(
                    order_index=order_start + idx,
                    render_group_index=render_group_index,
                    heading=block.heading,
                    content=row,
                    raw_content=row,
                    section_type=BlockType.TABLE.value,
                    metadata={
                        "block_type": BlockType.TABLE.value,
                        "render_format": "markdown_table",
                        "role": "table_row",
                        "row_index": idx,
                        "block_id": block.block_id,
                        "assemble_policy": "table_with_header",
                    },
                )
            )
        return sections

    @staticmethod
    def _build_table_raw_section(
        block: StructuredBlock,
        order_start: int,
        render_group_index: int,
        render_format: str,
    ) -> SectionDraft:
        return SectionDraft(
            order_index=order_start,
            render_group_index=render_group_index,
            heading=block.heading,
            content=block.raw_content,
            raw_content=block.raw_content,
            section_type=BlockType.TABLE.value,
            metadata={
                "block_type": BlockType.TABLE.value,
                "render_format": render_format,
                "role": "table_raw",
                "block_id": block.block_id,
                "assemble_policy": "table_with_header",
                "table_format": block.metadata.get("table_format", "markdown"),
            },
        )
