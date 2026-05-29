"""
Extract模块配置类

定义事项提取的配置选项
"""

from typing import List, Optional

from pydantic import Field, model_validator

from pipeline.models.base import pipelineBaseModel
from pipeline.models.entity import CustomEntityType


class ExtractBaseConfig(pipelineBaseModel):
    """
    提取配置基类 - 基础配置

    包含提取行为的基础参数，可在Engine中预设
    """

    # ==================== 并发控制 ====================
    max_concurrency: int = Field(
        default=5, ge=1, le=100, description="最大并发数（Agent并发处理chunk数量）"
    )

    # ==================== 向量同步配置 ====================
    enable_event_vector_sync: bool = Field(default=True, description="是否同步事项向量到向量库")
    enable_entity_vector_sync: bool = Field(
        default=False, description="是否同步实体向量到向量库（实验性功能）"
    )
    enable_event_entity_vector_sync: bool = Field(
        default=False, description="是否同步事件-实体关联描述向量到向量库（实验性功能）"
    )
    embedding_batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="向量生成批量大小（每批调用embedding API的文本数量，受API限制）",
    )
    index_batch_size: int = Field(
        default=50,
        ge=1,
        le=500,
        description="索引批量大小（每批写入向量库的文档数量）",
    )
    embedding_max_length: int = Field(
        default=500,
        ge=100,
        le=1000,
        description="Embedding输入最大字符长度（受模型token限制，中文约1.5token/字符，建议500-800）",
    )

    # ==================== 上下文配置 ====================
    prev_chunk_count: int = Field(
        default=1,
        ge=0,
        le=5,
        description="加载前文chunk数量（用于提取时的上下文背景，0表示不加载）",
    )
    max_content_length: int = Field(
        default=3000, ge=500, description="内容最大长度（兜底截断，防止异常长文本影响性能）"
    )

    # ==================== 质量过滤配置 ====================
    chunk_min_length: int = Field(
        default=20, ge=0, description="Chunk最小内容长度，低于此值跳过提取（0表示不过滤）"
    )
    event_min_length: int = Field(default=15, ge=0, description="事项正文最小长度（过滤标题党）")
    text_min_length: int = Field(default=10, ge=0, description="纯文本最小长度（过滤纯链接内容）")
    filter_image_sections: bool = Field(
        default=False,
        description="是否过滤图片类型的section（不参与事项提取，避免噪音；如果过滤后无正文则跳过该chunk）",
    )
    filter_keywords: List[str] = Field(
        default_factory=lambda: ["扫码", "加群", "优惠券", "点击领取", "限时免费"],
        description="过滤关键词黑名单（匹配时计入低质量信号）",
    )

    @model_validator(mode='after')
    def set_local_defaults(self):
        """LOCAL 模式下自动开启向量同步（因为没有 MySQL embedding 表）"""
        from pipeline.core.config import get_settings

        settings = get_settings()
        if settings.server_type == "LOCAL":
            # LOCAL 模式必须开启 entity 和 event_entity 向量同步
            if not self.enable_entity_vector_sync:
                self.enable_entity_vector_sync = True
            if not self.enable_event_entity_vector_sync:
                self.enable_event_entity_vector_sync = True
        return self

    # ==================== 实体配置 ====================
    custom_entity_types: List[CustomEntityType] = Field(
        default_factory=list, description="自定义实体类型列表（运行时优先级最高）"
    )

    # ==================== 历史召回配置 ====================
    enable_related_events: bool = Field(
        default=True, description="是否召回历史事项作为LLM提取的背景参考"
    )
    related_events_top_k: int = Field(default=3, ge=1, le=10, description="召回历史事项数量")
    related_events_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="历史事项相似度阈值（低于此值的不召回）"
    )

    # ==================== 提示词注入 ====================
    timezone: str = Field(default="Asia/Shanghai", description="时区（用于提示词中的时间显示）")
    custom_background: str = Field(
        default="", description="自定义背景信息（追加到提示词Background段落）"
    )
    custom_requirements: str = Field(
        default="", description="自定义提取要求（追加到提示词Requirements段落）"
    )
    enable_strict_filtering: bool = Field(
        default=True, description="是否启用严格的内容过滤（会传入严格过滤到custom_requirements）"
    )
    test_mode: bool = Field(
        default=False, description="测试模式：读取test_extract.yaml而非extract.yaml，读取entity_types_test列表而非entity_types表"
    )


class ExtractConfig(ExtractBaseConfig):
    """
    事项提取配置 - 完整配置（基础+运行时上下文）

    运行时上下文可由以下方式提供：
    1. 直接传入chunk_ids（单独调用）
    2. Engine处理后设置chunk_ids（链式调用）
    3. Engine从上下文自动读取chunk_ids（自动模式）
    """

    # ==================== 运行时上下文 ====================
    source_config_id: str = Field(..., description="信息源ID")
    article_id: Optional[str] = Field(
        default=None, description="文档ID（用于文档级别的实体类型配置）"
    )
    chunk_ids: List[str] = Field(..., min_length=1, description="Chunk ID列表")
    # 注意：enable_strict_filtering 和 test_mode 继承自 ExtractBaseConfig，不需要重复定义
