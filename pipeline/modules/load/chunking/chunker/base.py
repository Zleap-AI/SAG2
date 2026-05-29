"""Block -> Section 的切分基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from pipeline.modules.load.chunking.types import SectionDraft, StructuredBlock


class BaseBlockChunker(ABC):
    """按块类型生成 SectionDraft 的最小单元。"""

    @abstractmethod
    async def build_sections(
        self,
        block: StructuredBlock,
        order_start: int,
        render_group_index: int,
    ) -> List[SectionDraft]:
        pass
