"""
RAG 切片框架核心类型定义
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class BlockType(str, Enum):
    """结构块类型"""

    TEXT = "TEXT"
    FORMULA = "FORMULA"
    TABLE = "TABLE"
    CODE = "CODE"
    IMAGE = "IMAGE"


@dataclass
class InputDocument:
    """输入层标准化后的文档"""

    content: str
    source_path: Optional[Path] = None
    is_markdown: bool = True
    metadata: Dict = field(default_factory=dict)


@dataclass
class StructuredBlock:
    """结构识别层产物"""

    block_id: str
    block_type: BlockType
    raw_content: str
    heading: str = ""
    start_index: int = 0
    end_index: int = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class SectionDraft:
    """ArticleSection 草稿"""

    order_index: int
    render_group_index: int
    heading: str
    content: str
    raw_content: str
    section_type: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class ChunkDraft:
    """SourceChunk 草稿"""

    rank: int
    heading: str
    content: str
    raw_content: str
    chunk_type: str
    section_order_indices: List[int]
    metadata: Dict = field(default_factory=dict)


@dataclass
class ChunkingResult:
    """整条切片链路结果"""

    input_doc: InputDocument
    blocks: List[StructuredBlock]
    article_sections: List[SectionDraft]
    source_chunks: List[ChunkDraft]
