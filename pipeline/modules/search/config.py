"""
搜索模块配置

支持三种搜索策略：
- VECTOR: 纯向量搜索
- ATOMIC: 原子事项检索
- MULTI: 多元事项检索（通过 MultiConfig.strategy 参数支持 multi/multi1/hopllm 三种子策略）
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator

from pipeline.models.base import pipelineBaseModel


class RerankStrategy(str, Enum):
    """
    搜索策略（三种）

    - VECTOR:  纯向量匹配（跳过 Recall/Expand，直接向量搜索段落）
    - ATOMIC:  原子事项检索（三元组 + LLM 精选）
    - MULTI:   多元事项检索（多实体 + 多跳扩展 + LLM 精选）
               通过 MultiConfig.strategy 参数支持三种子策略：
               * "multi": 固定跳数扩展
               * "multi1": 双阶段扩跳（全量种子）
               * "hopllm": 双阶段扩跳（粗排种子）
    """
    VECTOR = "vector"
    ATOMIC = "atomic"
    MULTI = "multi"

    def __str__(self) -> str:
        return self.value


class ReturnType(str, Enum):
    """
    返回类型

    - EVENT: 事项（默认）
    - PARAGRAPH: 段落
    """
    EVENT = "event"
    PARAGRAPH = "paragraph"

    def __str__(self) -> str:
        return self.value


class RerankConfig(pipelineBaseModel):
    """
    搜索策略配置

    支持三种搜索方式：
    - VECTOR: 纯向量搜索（跳过 Recall/Expand）
    - ATOMIC: 原子事项检索（三元组 + LLM 精选）
    - MULTI: 多元事项检索（多实体 + 多跳扩展 + LLM 精选，默认）
            通过 MultiConfig.strategy 参数可选择 multi/multi1/hopllm 子策略
    """

    # 排序策略
    strategy: RerankStrategy = Field(
        default=RerankStrategy.MULTI,
        description="搜索策略（VECTOR/ATOMIC/MULTI）"
    )


class VectorConfig(pipelineBaseModel):
    """
    向量检索器配置

    独立于三阶段搜索，直接使用 Query 向量检索 Event/Chunk，
    支持 title/heading 和 content 向量的混合搜索。

    示例：
        config = VectorConfig(
            return_type="event",
            top_k=20,
            title_weight=0.3,
            content_weight=0.7,
            similarity_threshold=0.4
        )
    """

    # 返回数量
    top_k: int = Field(
        default=20,
        ge=1, le=1000,
        description="返回的最大数量"
    )

    # 返回类型
    return_type: Literal["chunk", "event"] = Field(
        default="event",
        description="返回类型：chunk=段落(SourceChunk)，event=事项(SourceEvent)"
    )

    # 向量权重（需满足 title_weight + content_weight = 1.0）
    title_weight: float = Field(
        default=0.3,
        ge=0.0, le=1.0,
        description="标题向量权重（事件用 title_vector，段落用 heading_vector）"
    )

    content_weight: float = Field(
        default=0.7,
        ge=0.0, le=1.0,
        description="内容向量权重（content_vector）"
    )

    # 相似度阈值（直接用作召回）
    similarity_threshold: float = Field(
        default=0.4,
        ge=0.0, le=1.0,
        description="相似度阈值，低于此分数的结果会被过滤"
    )


class QueryNormalizationConfig(pipelineBaseModel):
    """
    Query 标准化处理配置

    用于对查询进行预处理，提取关键词：
    - 文本清洗（lowercase、标点规范化）
    - jieba 分词
    - 停用词过滤（使用 goto456/stopwords 中文停用词表）
    """

    # 唯一开关
    enabled: bool = Field(
        default=False,
        description="是否启用 Query 标准化处理（启用后会进行分词+停用词过滤）"
    )


class SearchBaseConfig(pipelineBaseModel):
    """
    搜索基础配置

    用于引擎层统一配置，包含基础参数 + 算法配置
    """

    # 基础参数（引擎需要）
    query: str = Field(..., description="搜索查询")
    original_query: str = Field(default="", description="原始查询")
    start_time: Optional[datetime] = Field(
        default=None,
        description="时间范围开始（可选，UTC；用于 ES 时间过滤）",
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="时间范围结束（可选；用于时间过滤）",
    )
    source_ids: Optional[List[str]] = Field(
        default=None,
        description="事项来源ID列表（Article/Conversation ID），可选，用于精确过滤",
    )

    @field_validator("start_time", "end_time", mode="after")
    @classmethod
    def _strip_timezone(cls, v):
        """
        数据库存储 UTC 时间（MySQL connection SET time_zone='+00:00'）。
        前端传入 ISO 8601 带时区偏移（本地时间），需转为 naive UTC 匹配 DB。
        """
        if v is not None and v.tzinfo is not None:
            return v.astimezone(timezone.utc).replace(tzinfo=None)
        return v

    # 功能开关
    enable_query_rewrite: bool = Field(
        default=True,
        description="启用query重写（将口语化表述整理为更适合查询的问题）"
    )

    # Query 标准化配置
    query_normalization: QueryNormalizationConfig = Field(
        default_factory=QueryNormalizationConfig,
        description="Query 标准化处理配置"
    )

    # 实体类型过滤（Recall 和 Expand 阶段都使用）
    exclude_entity_types: List[str] = Field(
        default=["start_time", "end_time"],
        description="[黑名单] 需要排除的实体类型"
    )

    # 返回类型控制
    return_type: ReturnType = Field(
        default=ReturnType.EVENT,
        description="返回类型：事项(event) 或 段落(paragraph)，默认是事项"
    )

    # 线索数量控制（统一控制 Expand 和 Rerank 阶段）
    max_clues_per_event: int = Field(
        default=3,
        ge=1, le=5,
        description="Expand阶段：限制事项-实体双向线索；Rerank阶段：限制事项的实体线索"
    )

    # 段落返回控制
    return_chunks: bool = Field(
        default=False,
        description="是否返回段落信息（从事项的chunk_id获取，按事项排序去重）"
    )

    # 重排配置
    rerank: RerankConfig = Field(
        default_factory=RerankConfig, description="重排配置")

    # 策略专属配置（透传给 SearchConfig.strategy_config）
    strategy_config: Optional[Any] = Field(
        default=None,
        description="策略专属配置实例（MultiConfig/AtomicConfig/VectorConfig）"
    )


class SearchConfig(SearchBaseConfig):
    """
    搜索完整配置（基础配置 + 运行时上下文）

    继承SearchBaseConfig，添加运行时必需的上下文信息

    示例：
        # 单源搜索（向后兼容）
        config = SearchConfig(
            query="人工智能",
            source_config_id="source_123",
            recall=RecallConfig(max_entities=30),
            expand=ExpandConfig(max_hops=3),
            rerank=RerankConfig(strategy=RerankStrategy.MULTI)
        )

        # 多源搜索（新增功能）
        config = SearchConfig(
            query="人工智能",
            source_config_ids=["source_001", "source_002", "source_003"],
            recall=RecallConfig(max_entities=30),
            expand=ExpandConfig(max_hops=3),
            rerank=RerankConfig(strategy=RerankStrategy.MULTI)
        )
    """

    # === 运行时上下文 ===
    source_config_id: Optional[str] = Field(None, description="数据源ID（单个，向后兼容）")
    source_config_ids: Optional[List[str]] = Field(
        None, description="数据源ID列表（支持多源搜索）")
    article_id: Optional[str] = Field(None, description="文章ID")
    background: Optional[str] = Field(None, description="背景信息")

    # === 策略专属配置（可选，传入后由 SAGSearcher 透传给对应子 searcher）===
    strategy_config: Optional[Any] = Field(
        default=None,
        description="策略专属配置实例（MultiConfig/AtomicConfig/VectorConfig），"
                    "传入后 SAGSearcher 会将其透传给对应子 searcher，覆盖子 searcher 的默认值"
    )

    def model_post_init(self, __context):
        """初始化后验证和处理 source_config_id/source_config_ids"""
        # 验证：至少提供一个
        if not self.source_config_id and not self.source_config_ids:
            raise ValueError("必须提供 source_config_id 或 source_config_ids 参数")

        # 统一处理：如果只提供 source_config_id，转换为 source_config_ids
        if self.source_config_id and not self.source_config_ids:
            self.source_config_ids = [self.source_config_id]
        elif self.source_config_ids and not self.source_config_id:
            # 多源场景，source_config_id 设为第一个（向后兼容）
            self.source_config_id = self.source_config_ids[0]

    def get_source_config_ids(self) -> List[str]:
        """
        获取统一的 source_config_ids 列表

        Returns:
            source_config_ids 列表（至少包含一个元素）
        """
        return self.source_config_ids or []

    def is_multi_source(self) -> bool:
        """是否为多源搜索"""
        return len(self.get_source_config_ids()) > 1


class AtomicConfig(pipelineBaseModel):
    """
    原子事项检索器配置

    检索恰好包含 2 个实体的原子化三元组事项。

    示例：
        config = AtomicConfig(
            top_k=20,
            similarity_threshold=0.4
        )
    """

    entity_top_k: int = Field(
        default=20,
        ge=1, le=1000,
        description="返回的最大实体数量"
    )
    atomic_top_k: int = Field(
        default=20,
        ge=1, le=1000,
        description="原子化事项最大数量"
    )

    key_similarity_threshold: float = Field(
        default=0.9,
        ge=0.0, le=1.0,
        description="相似度阈值，低于此分数的结果会被过滤"
    )

    similarity_threshold: float = Field(
        default=0.4,
        ge=0.0, le=1.0,
        description="事项向量检索相似度阈值"
    )

    max_hops: int = Field(
        default=1,
        ge=0, le=10,
        description="多跳扩展次数（0=不扩展，1=扩展1轮）"
    )

    max_events: int = Field(
        default=1000,
        ge=1, le=5000,
        description="粗排序最大返回事项数量"
    )

    rerank_top_k: int = Field(
        default=5,
        ge=1, le=20,
        description="LLM 精选返回数量"
    )

    max_sections: int = Field(
        default=10,
        ge=1, le=50,
        description="最终返回段落最大数量（chunk_id 去重后截断）"
    )


class MultiConfig(pipelineBaseModel):
    """
    多元事项检索器统一配置

    支持三种扩展策略：
    - "multi":   单阶段固定跳数扩展（使用 max_hops 参数）
    - "multi1":  双阶段扩展，阶段B以 hop1 全量事项实体为种子（广度优先）
    - "hopllm":  双阶段扩展，阶段B以粗排后事项实体为种子（质量优先）

    示例：
        # 单阶段策略
        config = MultiConfig(strategy="multi", max_hops=2, max_events=100)

        # 双阶段策略（multi1 或 hopllm）
        config = MultiConfig(
            strategy="hopllm",
            max_events_a=100,
            max_events_b=50,
            max_hop_retries=3
        )
    """

    # ========== 策略选择 ==========
    strategy: str = Field(
        default="multi",
        description="扩展策略：multi=单阶段固定跳数, multi1=双阶段全量种子, hopllm=双阶段粗排种子"
    )

    # ========== 通用参数 ==========
    entity_top_k: int = Field(
        default=20,
        ge=1, le=1000,
        description="返回的最大实体数量"
    )
    multi_top_k: int = Field(
        default=20,
        ge=1, le=1000,
        description="多元事项最大数量"
    )
    key_similarity_threshold: float = Field(
        default=0.9,
        ge=0.0, le=1.0,
        description="相似度阈值，低于此分数的结果会被过滤"
    )
    similarity_threshold: float = Field(
        default=0.4,
        ge=0.0, le=1.0,
        description="事项向量检索相似度阈值"
    )
    rerank_top_k: int = Field(
        default=10,
        ge=1, le=20,
        description="LLM 精选返回数量"
    )
    max_sections: int = Field(
        default=10,
        ge=1, le=50,
        description="最终返回段落最大数量（chunk_id 去重后截断）"
    )

    # ========== multi 策略专用参数 ==========
    max_hops: int = Field(
        default=1,
        ge=0, le=10,
        description="[multi] 多跳扩展次数（0=不扩展，1=扩展1轮）"
    )
    max_events: int = Field(
        default=100,
        ge=1, le=5000,
        description="[multi] 粗排序最大返回事项数量"
    )

    # ========== multi1/hopllm 策略专用参数 ==========
    max_events_a: int = Field(
        default=100,
        ge=1, le=5000,
        description="[multi1/hopllm] eventset（hop0+hop1）Step6 粗排最大候选数量"
    )
    max_events_b: int = Field(
        default=0,
        ge=0, le=5000,
        description="[multi1/hopllm] eventset1（hop2+）扩跳目标数量和粗排最大候选数量"
    )
    max_hop_retries: int = Field(
        default=3,
        ge=1, le=10,
        description="[multi1/hopllm] 阶段B最大重试跳数"
    )


__all__ = [
    # 配置
    "SearchConfig",
    "SearchBaseConfig",
    "RerankConfig",
    "VectorConfig",
    "AtomicConfig",
    "MultiConfig",
    "QueryNormalizationConfig",
    "RerankStrategy",
    "ReturnType",
]
