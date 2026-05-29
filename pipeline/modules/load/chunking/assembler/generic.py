"""统一 SourceChunk 组装器（纯文本简化版）。"""

from __future__ import annotations

from typing import List

from pipeline.modules.load.chunking.base import BaseSourceChunkAssembler
from pipeline.modules.load.chunking.tokenizer import TokenizerTokenEstimator
from pipeline.modules.load.chunking.types import ChunkDraft, InputDocument, SectionDraft


class PolicyBasedSourceChunkAssembler(BaseSourceChunkAssembler):
    """按顺序跨 block 组装 chunk（纯文本：无表格特殊处理）。"""

    def __init__(
        self,
        source_chunk_max_tokens: int,
        model_type: str = "generic",
        standalone_block_max_tokens: int = 600,
        heading_strict: bool = False,
    ) -> None:
        self.source_chunk_max_tokens = max(64, min(source_chunk_max_tokens, 1000))
        self.standalone_block_max_tokens = max(1, standalone_block_max_tokens)
        self.token_estimator = TokenizerTokenEstimator(model_type)
        self.heading_strict = heading_strict

    def assemble_chunks(
        self,
        doc: InputDocument,
        sections: List[SectionDraft],
    ) -> List[ChunkDraft]:
        _ = doc
        chunks: List[ChunkDraft] = []
        current: List[SectionDraft] = []
        current_tokens = 0

        for section in sections:
            section_tokens = self._section_tokens(section)

            # heading_strict 模式：遇到新标题强制断开
            if self.heading_strict and current and section.heading != current[0].heading:
                chunks.append(self._build_chunk(current))
                current = []
                current_tokens = 0

            # 大块（>standalone_block_max_tokens）独立成 chunk
            if section_tokens > self.standalone_block_max_tokens:
                if current:
                    chunks.append(self._build_chunk(current))
                    current = []
                    current_tokens = 0

                # 大块内部按 source_chunk_max_tokens 切分
                for unit in self._split_large_section(section):
                    chunks.append(self._build_chunk([unit]))
                continue

            # 小块贪婪聚合
            if not current:
                current = [section]
                current_tokens = section_tokens
            elif current_tokens + section_tokens <= self.source_chunk_max_tokens:
                current.append(section)
                current_tokens += section_tokens
            else:
                chunks.append(self._build_chunk(current))
                current = [section]
                current_tokens = section_tokens

        if current:
            chunks.append(self._build_chunk(current))

        for idx, chunk in enumerate(chunks):
            chunk.rank = idx
        return chunks

    def _split_large_section(self, section: SectionDraft) -> List[SectionDraft]:
        """大 section 按 token 上限切分为多个子 section。"""
        content = section.content.strip()
        if not content:
            return [section]

        # 简单按段落切分
        paragraphs = content.split("\n\n")
        units: List[SectionDraft] = []
        current_parts: List[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.token_estimator.estimate_tokens(para)
            if not current_parts:
                current_parts = [para]
                current_tokens = para_tokens
            elif current_tokens + para_tokens <= self.source_chunk_max_tokens:
                current_parts.append(para)
                current_tokens += para_tokens
            else:
                units.append(self._clone_section(section, "\n\n".join(current_parts)))
                current_parts = [para]
                current_tokens = para_tokens

        if current_parts:
            units.append(self._clone_section(section, "\n\n".join(current_parts)))

        return units if units else [section]

    @staticmethod
    def _clone_section(original: SectionDraft, new_content: str) -> SectionDraft:
        """克隆 section 并替换 content。"""
        return SectionDraft(
            order_index=original.order_index,
            render_group_index=original.render_group_index,
            heading=original.heading,
            content=new_content,
            raw_content=new_content,
            section_type=original.section_type,
            metadata=original.metadata.copy(),
        )

    def _build_chunk(self, sections: List[SectionDraft]) -> ChunkDraft:
        if not sections:
            raise ValueError("Cannot build chunk from empty sections")

        heading = ""
        for sec in sections:
            if sec.heading:
                heading = sec.heading
                break

        content = "\n".join(s.content.strip() for s in sections).strip()
        raw_content = "".join(s.raw_content for s in sections)
        section_order_indices = [s.order_index for s in sections]
        chunk_type = sections[0].section_type
        render_group_indices = sorted({s.render_group_index for s in sections})

        return ChunkDraft(
            rank=0,
            heading=heading,
            content=content if content else raw_content,
            raw_content=raw_content,
            chunk_type=chunk_type,
            section_order_indices=section_order_indices,
            metadata={
                "section_order_indices": section_order_indices,
                "render_group_index": render_group_indices[0],
                "render_group_indices": render_group_indices,
                "chunk_type": chunk_type,
            },
        )

    def _section_tokens(self, section: SectionDraft) -> int:
        return self.token_estimator.estimate_tokens(section.content.strip())


class MarkdownSourceChunkAssembler(PolicyBasedSourceChunkAssembler):
    """兼容原类名：Markdown 使用统一策略组装器。"""

    pass
