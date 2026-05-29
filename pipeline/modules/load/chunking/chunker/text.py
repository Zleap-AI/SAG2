"""Markdown 文本块切分（纯文本简化版）。"""

from __future__ import annotations

from typing import List

from pipeline.modules.load.chunking.chunker.base import BaseBlockChunker
from pipeline.modules.load.chunking.tokenizer import TokenizerTokenEstimator
from pipeline.modules.load.chunking.types import BlockType, SectionDraft, StructuredBlock


class MarkdownTextChunker(BaseBlockChunker):
    """TEXT block 切分器（纯文本：无公式/链接保护）。"""

    PRIMARY_DELIMITERS = (
        "\r\n\r\n",
        "\n\n",
        "\r\n",
        "\n",
        "\r",
        "。",
        "！",
        "？",
        "!",
        "?",
        ". ",
    )
    SECONDARY_DELIMITERS = (
        "，",
        ",",
        "、",
        "；",
        ";",
    )

    def __init__(self, section_max_tokens: int, model_type: str = "generic") -> None:
        self.section_max_tokens = max(32, section_max_tokens)
        self.token_estimator = TokenizerTokenEstimator(model_type)

    async def build_sections(
        self,
        block: StructuredBlock,
        order_start: int,
        render_group_index: int,
    ) -> List[SectionDraft]:
        text_parts = self._split_text(block.raw_content)
        sections: List[SectionDraft] = []
        for idx, part in enumerate(text_parts):
            sections.append(
                SectionDraft(
                    order_index=order_start + idx,
                    render_group_index=render_group_index,
                    heading=block.heading,
                    content=part,
                    raw_content=part,
                    section_type=BlockType.TEXT.value,
                    metadata={
                        "block_type": BlockType.TEXT.value,
                        "render_format": "markdown_text",
                        "block_id": block.block_id,
                        "assemble_policy": "generic",
                    },
                )
            )
        return sections

    def _split_text(self, text: str) -> List[str]:
        if text == "":
            return [""]
        first_pass = self._split_by_delimiters(text, self.PRIMARY_DELIMITERS)
        if not first_pass:
            first_pass = [text]

        result: List[str] = []
        for part in first_pass:
            tokens = self.token_estimator.estimate_tokens(part)
            if tokens <= self.section_max_tokens:
                result.append(part)
                continue

            second_pass = self._split_by_delimiters(part, self.SECONDARY_DELIMITERS)
            if len(second_pass) <= 1:
                result.extend(self._force_split(part))
                continue

            for frag in second_pass:
                frag_tokens = self.token_estimator.estimate_tokens(frag)
                if frag_tokens <= self.section_max_tokens:
                    result.append(frag)
                else:
                    result.extend(self._force_split(frag))
        return result

    def _split_by_delimiters(self, text: str, delimiters: tuple[str, ...]) -> List[str]:
        parts: List[str] = []
        start = 0
        idx = 0
        while idx < len(text):
            matched = None
            for delimiter in delimiters:
                if text.startswith(delimiter, idx):
                    matched = delimiter
                    break

            if matched:
                end = idx + len(matched)
                parts.append(text[start:end])
                start = end
                idx = end
            else:
                idx += 1

        if start < len(text):
            parts.append(text[start:])
        normalized_parts = self._merge_whitespace_only_parts(parts if parts else [text])
        return normalized_parts if normalized_parts else [text]

    def _force_split(self, text: str) -> List[str]:
        chunks: List[str] = []
        current_pos = 0
        while current_pos < len(text):
            remaining = text[current_pos:]
            cut = self._find_prefix_len_by_tokens(remaining, self.section_max_tokens)
            if cut <= 0:
                cut = 1
            chunks.append(remaining[:cut])
            current_pos += cut
        return chunks

    def _find_prefix_len_by_tokens(self, text: str, max_tokens: int) -> int:
        if not text:
            return 0
        low, high = 0, len(text)
        best = 0
        while low <= high:
            mid = (low + high) // 2
            prefix = text[:mid]
            tokens = self.token_estimator.estimate_tokens(prefix)
            if tokens <= max_tokens:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        return best

    @staticmethod
    def _merge_whitespace_only_parts(parts: List[str]) -> List[str]:
        if len(parts) <= 1:
            return parts

        merged: List[str] = []
        leading_whitespace = ""
        for part in parts:
            if part.strip() == "":
                if merged:
                    merged[-1] += part
                else:
                    leading_whitespace += part
                continue

            if leading_whitespace:
                part = f"{leading_whitespace}{part}"
                leading_whitespace = ""
            merged.append(part)

        if leading_whitespace:
            if merged:
                merged[-1] += leading_whitespace
            else:
                merged.append(leading_whitespace)
        return merged
