"""
搜索模块

提供SAG搜索引擎，支持五种策略：VECTOR / ATOMIC / MULTI / MULTI1 / HOPLLM

架构：
- SAGSearcher/EventSearcher: 统一搜索入口（推荐使用）
- VectorSearcher: 纯向量检索器
- AtomicSearcher: 原子事项检索器
- MultiSearcher: 多元事项检索器（支持 multi/multi1/hopllm 三种策略）
"""

from pipeline.modules.search.config import (
    SearchConfig,
    SearchBaseConfig,
    RerankConfig,
    VectorConfig,
    AtomicConfig,
    MultiConfig,
    RerankStrategy,
)
from pipeline.modules.search.searcher import (
    SAGSearcher,
    EventSearcher,
)
from pipeline.modules.search.vector import VectorSearcher
from pipeline.modules.search.atomic import AtomicSearcher
from pipeline.modules.search.multi import MultiSearcher

__all__ = [
    # 配置
    "SearchConfig",
    "SearchBaseConfig",
    "RerankConfig",
    "VectorConfig",
    "AtomicConfig",
    "MultiConfig",
    "RerankStrategy",
    # 搜索器（推荐）
    "SAGSearcher",
    "EventSearcher",
    "VectorSearcher",
    "AtomicSearcher",
    "MultiSearcher",
]
