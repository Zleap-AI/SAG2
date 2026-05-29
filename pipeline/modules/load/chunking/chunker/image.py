"""Markdown 图片块切分（纯文本数据集存根，不执行图片解析）。"""

from __future__ import annotations

from typing import List

from pipeline.modules.load.chunking.chunker.base import BaseBlockChunker
from pipeline.modules.load.chunking.types import BlockType, SectionDraft, StructuredBlock
from pipeline.utils import get_logger

logger = get_logger("modules.load.chunking.chunker.image")


class MarkdownImageChusnker(BaseBlockChunker):
    """IMAGE block 切分器（存根：数据集为纯文本，不执行图片解析）。"""

    async def build_sections(
        self,
        block: StructuredBlock,
        order_start: int,
        render_group_index: int,
    ) -> List[SectionDraft]:
        image_src = (block.metadata.get("image_src") or "").strip()
        alt_text = (block.metadata.get("alt_text") or "").strip()
        content = f"[IMAGE] alt: {alt_text}" if alt_text else "[IMAGE]"
        return [
            SectionDraft(
                order_index=order_start,
                render_group_index=render_group_index,
                heading=block.heading,
                content=content,
                raw_content=block.raw_content,
                section_type=BlockType.IMAGE.value,
                metadata={
                    "block_type": BlockType.IMAGE.value,
                    "render_format": "markdown_image",
                    "block_id": block.block_id,
                    "alt_text": alt_text,
                    "image_src": image_src,
                    "split_strategy": "single_image_single_section",
                    "assemble_policy": "generic",
                },
            )
        ]
