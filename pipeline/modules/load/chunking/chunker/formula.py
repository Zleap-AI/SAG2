"""Markdown 公式块切分。"""

from __future__ import annotations

from typing import List

from pipeline.modules.load.chunking.chunker.base import BaseBlockChunker
from pipeline.modules.load.chunking.types import BlockType, SectionDraft, StructuredBlock


class MarkdownFormulaChunker(BaseBlockChunker):
    """FORMULA block 切分器，默认保持整块不拆分。"""

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
                section_type=BlockType.FORMULA.value,
                metadata={
                    "block_type": BlockType.FORMULA.value,
                    "render_format": "markdown_formula",
                    "formula_type": block.metadata.get("formula_type", ""),
                    "block_id": block.block_id,
                    "split_strategy": "single_formula_single_section",
                    "assemble_policy": "generic",
                },
            )
        ]
