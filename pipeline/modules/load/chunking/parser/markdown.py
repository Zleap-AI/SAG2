"""
Markdown 输入标准化与结构解析。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

from pipeline.modules.load.chunking.base import BaseBlockParser, BaseInputNormalizer
from pipeline.modules.load.chunking.types import BlockType, InputDocument, StructuredBlock
from pipeline.utils import normalize_heading_text


class MarkdownInputNormalizer(BaseInputNormalizer):
    """Markdown 输入标准化"""

    def normalize(self, content: str, source_path: Optional[Path] = None) -> InputDocument:
        return InputDocument(
            content=content or "",
            source_path=source_path,
            is_markdown=True,
            metadata={},
        )


class MarkdownBlockParser(BaseBlockParser):
    """Markdown 结构识别（纯文本 md：仅识别标题）"""

    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def parse_blocks(self, doc: InputDocument) -> List[StructuredBlock]:
        """纯文本 md：按标题边界拆分为 TEXT blocks"""
        text = doc.content
        headings = self._heading_positions(text, [])
        blocks, _ = self._build_text_blocks(
            text=text,
            start=0,
            end=len(text),
            headings=headings,
            counter=0,
        )
        return [b for b in blocks if b.raw_content != ""]

    def _heading_positions(
        self,
        text: str,
        occupied: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Tuple[int, str]]:
        headings: List[Tuple[int, str]] = []
        for match in self.HEADING_PATTERN.finditer(text):
            headings.append((match.start(), normalize_heading_text(match.group(2))))
        return headings

    @staticmethod
    def _resolve_heading(headings: List[Tuple[int, str]], position: int) -> str:
        if not headings:
            return ""
        candidate = ""
        for pos, title in headings:
            if pos <= position:
                candidate = title
            else:
                break
        return candidate

    def _build_text_blocks(
        self,
        text: str,
        start: int,
        end: int,
        headings: List[Tuple[int, str]],
        counter: int,
    ) -> Tuple[List[StructuredBlock], int]:
        """把文本段按标题边界进一步拆分，保证 Section 可拿到就近标题。"""
        if start >= end:
            return [], counter

        split_points = [pos for pos, _ in headings if start < pos < end]
        boundaries = [start, *split_points, end]
        blocks: List[StructuredBlock] = []

        for idx in range(len(boundaries) - 1):
            seg_start = boundaries[idx]
            seg_end = boundaries[idx + 1]
            raw = text[seg_start:seg_end]
            if raw == "":
                continue
            blocks.append(
                StructuredBlock(
                    block_id=f"text-{counter}",
                    block_type=BlockType.TEXT,
                    raw_content=raw,
                    heading=self._resolve_heading(headings, seg_start),
                    start_index=seg_start,
                    end_index=seg_end,
                )
            )
            counter += 1

        return blocks, counter

