"""Markdown 代码块切分。"""

from __future__ import annotations

from typing import List

from pipeline.modules.load.chunking.chunker.base import BaseBlockChunker
from pipeline.modules.load.chunking.types import BlockType, SectionDraft, StructuredBlock


class MarkdownCodseChunker(BaseBlockChunker):
    """CODE block 切分器。"""

    async def build_sections(
        self,
        block: StructuredBlock,
        order_start: int,
        render_group_index: int,
    ) -> List[SectionDraft]:
        return [
            SectionDraft(
                order_index=order_start,
                render_group_index=render_group_index,
                heading=block.heading,
                content=block.raw_content,
                raw_content=block.raw_content,
                section_type=BlockType.CODE.value,
                metadata={
                    "block_type": BlockType.CODE.value,
                    "render_format": "markdown_code",
                    "block_id": block.block_id,
                    "language": block.metadata.get("language", ""),
                    "split_strategy": "single_block_placeholder",
                    "assemble_policy": "generic",
                },
            )
        ]
