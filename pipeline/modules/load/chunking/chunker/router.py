"""按 BlockType 路由到对应 chunker。"""

from __future__ import annotations

from typing import Dict, Optional

from pipeline.modules.load.chunking.chunker.base import BaseBlockChunker
from pipeline.modules.load.chunking.types import BlockType, StructuredBlock


class BlockChunkerRouter:
    """BlockType -> BaseBlockChunker 的路由器。"""

    def __init__(
        self,
        chunker_by_type: Dict[BlockType, BaseBlockChunker],
        default_chunker: Optional[BaseBlockChunker] = None,
    ) -> None:
        self.chunker_by_type = dict(chunker_by_type)
        self.default_chunker = default_chunker

    def resolve(self, block: StructuredBlock) -> BaseBlockChunker:
        chunker = self.chunker_by_type.get(block.block_type)
        if chunker is not None:
            return chunker
        if self.default_chunker is not None:
            return self.default_chunker
        raise ValueError(f"未找到 block_type={block.block_type} 的 chunker")
