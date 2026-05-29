"""
Load模块配置类和返回结果

定义文档加载的配置选项和返回格式
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import Field, field_validator, model_validator

from pipeline.models.base import pipelineBaseModel


# ============ 返回结果模型 ============

class LoadResult(pipelineBaseModel):
    """
    加载结果（统一返回格式）

    用于衔接Load → Extract流程
    """

    # === 核心数据 ===
    source_id: str = Field(..., description="源ID（article_id）")
    source_type: str = Field(..., description="源类型（ARTICLE）")
    chunk_ids: List[str] = Field(..., description="生成的Chunk ID列表")

    # === 元数据 ===
    source_config_id: str = Field(..., description="信息源配置ID")
    title: Optional[str] = Field(default=None, description="标题")
    chunk_count: int = Field(..., description="Chunk数量")

    # === 扩展数据 ===
    extra: Dict = Field(default_factory=dict, description="额外信息")


# ============ 配置模型 ============


class LoadBaseConfig(pipelineBaseModel):
    """
    加载配置基类 - 基础配置

    包含所有数据源通用的配置参数
    """

    # === 通用配置 ===
    max_tokens: int = Field(
        default=1000,
        ge=100,
        le=100000,
        description="每个chunk的最大token数"
    )

    # === 存储配置 ===
    auto_vector: bool = Field(
        default=True,
        description="是否自动索引到Elasticsearch"
    )

    # === 提示词增强 ===
    background: Optional[str] = Field(
        default=None,
        description="背景信息（补充元数据生成上下文）"
    )

    # === 信息源 ===
    source_config_id: Optional[str] = Field(default=None, description="信息源ID")

    @field_validator('source_config_id')
    @classmethod
    def validate_source_config_id(cls, v):
        """验证source_config_id不能为空字符串"""
        if v is not None and (not v or not v.strip()):
            raise ValueError("source_config_id 不能为空字符串")
        return v.strip() if v else v

    # === 批量处理配置 ===
    enable_batch_indexing: bool = Field(
        default=True,
        description="是否启用批量索引优化"
    )

    embedding_batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="向量生成批量大小（每批处理的文本数量）"
    )

    es_bulk_index_size: int = Field(
        default=50,
        ge=1,
        le=200,
        description="ES批量索引大小（每批索引的文档数量）"
    )


class DocumentLoadConfig(LoadBaseConfig):
    """文档加载配置 - 从文件路径加载 Markdown 文档"""

    # === 数据来源 ===
    path: Optional[Union[str, Path]] = Field(
        default=None,
        description="文件路径"
    )

    # === 文档处理配置 ===
    min_content_length: int = Field(
        default=100,
        ge=10,
        description="最小内容长度（字符数）"
    )
    merge_short_sections: bool = Field(
        default=False,
        description="是否启用短片段合并"
    )
    chunk_mode: str = Field(
        default="standard",
        description=(
            "切块模式：standard（智能切分+可合并短片段）、"
            "heading_strict（严格按标题切分、不合并短片段）、"
            "overlap（固定1200 token切块+100 token重叠）"
        )
    )

    @model_validator(mode='after')
    def check_path(self) -> 'DocumentLoadConfig':
        """验证文件路径"""
        if not self.path:
            raise ValueError("必须提供 path")
        return self

    @model_validator(mode='after')
    def normalize_chunk_mode(self) -> 'DocumentLoadConfig':
        allowed_modes = {"standard", "heading_strict", "overlap"}
        if self.chunk_mode not in allowed_modes:
            raise ValueError(f"不支持的chunk_mode: {self.chunk_mode}")
        return self
