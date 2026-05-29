"""
SQLAlchemy ORM模型定义

所有数据库表的定义 
"""

# pylint: disable=not-callable
# SQLAlchemy's func.now() is callable at runtime but Pylint doesn't recognize it

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional

from sqlalchemy import (
    CHAR,
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    VARBINARY,
    or_,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.mysql import LONGTEXT, MEDIUMTEXT, VARCHAR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pipeline.db.base import Base


class SourceConfig(Base):
    """信息源配置表"""

    __tablename__ = "source_config"

    # 主键：UUID
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True)

    # 信息源基本信息
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(255))


    target_config: Mapped[Optional[dict]] = mapped_column(JSON)

    # 时间戳
    created_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_time: Mapped[Optional[datetime]] = mapped_column(DateTime, onupdate=func.now())

    # 关系
    articles: Mapped[List["Article"]] = relationship(
        "Article",
        back_populates="source",
        cascade="all, delete-orphan",
    )
    source_events: Mapped[List["SourceEvent"]] = relationship(
        "SourceEvent",
        back_populates="source",
        cascade="all, delete-orphan",
    )
    entity_types: Mapped[List["EntityType"]] = relationship(
        "EntityType",
        back_populates="source",
        cascade="all, delete-orphan",
    )
    entities: Mapped[List["Entity"]] = relationship(
        "Entity",
        back_populates="source",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<SourceConfig(id={self.id}, name={self.name})>"


class KBDocument(Base):
    """知识库文档表（兼容主系统 kb_document 结构，当前不绑定外键）"""

    __tablename__ = "kb_document"
    __table_args__ = (
        Index("idx_kb_document_kb_source", "knowledge_base_id", "source_id"),
        Index("idx_kb_document_source_id", "source_id"),
        Index("idx_kb_document_knowledge_base_id", "knowledge_base_id"),
        Index("idx_kb_document_uploader_id", "uploader_id"),
        Index("idx_kb_document_created_time", "created_time"),
        {"comment": "文档表"},
    )

    id: Mapped[str] = mapped_column(String(191), primary_key=True)
    name: Mapped[str] = mapped_column(String(191), nullable=False, comment="文档名称")
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, comment="文件大小")
    file_type: Mapped[str] = mapped_column(String(50), nullable=False, comment="文件类型")
    knowledge_base_id: Mapped[str] = mapped_column(String(191), nullable=False, comment="知识库ID")
    uploader_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, comment="上传者ID")
    source_id: Mapped[Optional[str]] = mapped_column(
        String(191),
        nullable=True,
        comment="外部来源ID",
    )
    parse_status: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True, comment="文档解析状态VAR36"
    )
    parse_task_id: Mapped[Optional[str]] = mapped_column(
        String(191), nullable=True, comment="解析任务ID"
    )
    source_file_url: Mapped[Optional[str]] = mapped_column(
        String(2000), nullable=True, comment="源文件地址"
    )
    pdf_url: Mapped[Optional[str]] = mapped_column(
        String(2000), nullable=True, comment="pdf file url"
    )
    md_file_url: Mapped[Optional[str]] = mapped_column(
        String(2000), nullable=True, comment="markdown file url"
    )
    parse_result_url: Mapped[Optional[str]] = mapped_column(
        String(2000), nullable=True, comment="解析结果地址"
    )
    document_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, comment="元数据")
    copied_from: Mapped[Optional[str]] = mapped_column(
        String(191), nullable=True, comment="复制来源记录ID，记录此记录是从哪个document复制而来"
    )
    created_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False, comment="创建时间"
    )
    updated_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False, comment="更新时间"
    )

    def __repr__(self) -> str:
        return f"<KBDocument(id={self.id}, name={self.name[:30]})>"


class ArticleParseStatus(str, Enum):
    """文章解析状态"""

    PENDING = "PENDING"
    PARSING = "PARSING"
    PARSED = "PARSED"
    EXTRACTING = "EXTRACTING"
    COMPLETED = "COMPLETED"
    PARSE_FAILED = "PARSE_FAILED"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    FAILED = "FAILED"
    PENDING_RETRY_V2 = "PENDING_RETRY_V2"


class Article(Base):
    """文章表"""

    __tablename__ = "article"

    # 主键：UUID
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True)

    # 信息源配置ID：UUID（外键）
    source_config_id: Mapped[str] = mapped_column(
        CHAR(36),
        ForeignKey("source_config.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
        index=True,
    )

    # 基本信息
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    source_id: Mapped[Optional[str]] = mapped_column(
        VARCHAR(100), nullable=True, comment="来源文章ID"
    )
    summary: Mapped[Optional[str]] = mapped_column(Text)
    content: Mapped[Optional[str]] = mapped_column(LONGTEXT)

    # 分类和标签
    category: Mapped[Optional[str]] = mapped_column(String(50), index=True)
    tags: Mapped[Optional[dict]] = mapped_column(JSON)  # List[str]

    # 状态：PENDING, COMPLETED, FAILED, PROCESSING
    status: Mapped[str] = mapped_column(String(20), default="PENDING", nullable=False)
    parse_status: Mapped["ArticleParseStatus"] = mapped_column(
        VARCHAR(66), default=ArticleParseStatus.PENDING, nullable=False, comment="解析状态"
    )

    sync_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, comment="事项同步完成时间"
    )

    # 处理错误信息（失败时记录）
    error: Mapped[Optional[str]] = mapped_column(Text)

    # 扩展数据：{"url": "", "headings": []}
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON)

    # 时间戳
    created_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_time: Mapped[Optional[datetime]] = mapped_column(DateTime, onupdate=func.now())
    sync_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, comment="同步时间（提取事项）"
    )

    # 关系
    source: Mapped["SourceConfig"] = relationship(
        "SourceConfig",
        back_populates="articles",
    )
    sections: Mapped[List["ArticleSection"]] = relationship(
        "ArticleSection",
        back_populates="article",
        cascade="all, delete-orphan",
    )
    source_events: Mapped[List["SourceEvent"]] = relationship(
        "SourceEvent",
        back_populates="article",
        cascade="all, delete-orphan",
    )
    entity_types: Mapped[List["EntityType"]] = relationship(
        "EntityType",
        back_populates="article",
        cascade="all, delete-orphan",
    )

    # 索引
    __table_args__ = (
        Index("idx_source_config_id", "source_config_id"),
        Index("idx_source_config_status", "source_config_id", "status"),
        Index("idx_article_source_id", "source_id"),
        Index("idx_category", "category"),
    )

    def __repr__(self) -> str:
        return f"<Article(id={self.id}, title={self.title[:30]})>"


class ArticleSection(Base):
    """文章片段表"""

    __tablename__ = "article_section"

    # 主键：UUID
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True)

    # 文章ID：UUID（外键）
    article_id: Mapped[str] = mapped_column(
        CHAR(36),
        ForeignKey("article.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
        index=True,
    )

    # 片段信息
    order_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="排序索引")
    render_group_index: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="渲染分组索引"
    )
    type: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, comment="片段类型：TEXT/IMAGE/CODE/TABLE等"
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    heading: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(LONGTEXT, nullable=False, comment="处理后的内容（纯文本）")
    raw_content: Mapped[Optional[str]] = mapped_column(
        LONGTEXT, nullable=True, comment="原始内容（可能包含markdown/html）"
    )
    image_url: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True, comment="图片URL（仅图片类型）"
    )
    length: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="内容长度")

    # 扩展数据：{"type": "TEXT|IMAGE|CODE", "length": 0}
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON)

    # 时间戳
    created_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # 关系
    article: Mapped["Article"] = relationship(
        "Article",
        back_populates="sections",
    )

    # 索引
    __table_args__ = (
        Index("idx_article_id", "article_id"),
        Index("idx_article_rank", "article_id", "rank"),
        Index("idx_article_order", "article_id", "order_index"),
    )

    def __repr__(self) -> str:
        return f"<ArticleSection(id={self.id}, heading={self.heading[:30]})>"


class EntityType(Base):
    """实体类型定义表"""

    __tablename__ = "entity_type"

    # 主键：UUID
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True)

    # 应用范围：global/source/article
    scope: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default="global",
        index=True,
        comment="应用范围：global/source/article",
    )

    # 信息源配置ID：NULL表示系统默认类型（外键）
    source_config_id: Mapped[Optional[str]] = mapped_column(
        CHAR(36),
        ForeignKey("source_config.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=True,
        index=True,
    )

    # 文档ID：仅 scope=article 时有值（外键）
    article_id: Mapped[Optional[str]] = mapped_column(
        CHAR(36),
        ForeignKey("article.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=True,
        index=True,
        comment="文档ID（仅 scope=article 时有值）",
    )

    # 类型标识符：time, location, person等
    type: Mapped[str] = mapped_column(String(50), nullable=False)

    # 类型名称（显示名称）
    name: Mapped[str] = mapped_column(String(100), nullable=False)

    # 类型描述
    description: Mapped[Optional[str]] = mapped_column(Text)

    # 默认权重（0.00-9.99）
    weight: Mapped[Decimal] = mapped_column(Numeric(3, 2), default=Decimal("1.00"), nullable=False)

    # 相似度匹配阈值（0.000-1.000）- 用于实体向量搜索和去重时的最低相似度要求
    similarity_threshold: Mapped[Decimal] = mapped_column(
        Numeric(4, 3), default=Decimal("0.800"), nullable=False
    )

    # 是否启用
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # 是否为系统默认类型
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # 值格式模板（如 "{number}{unit}"）
    value_format: Mapped[Optional[str]] = mapped_column(String(100))

    # 值约束（JSON 格式，存储枚举列表、数值范围等）
    value_constraints: Mapped[Optional[dict]] = mapped_column(JSON)

    # 扩展数据：{"extraction_prompt": "", "validation_rule": {}}
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON)

    # 时间戳
    created_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # 关系
    source: Mapped[Optional["SourceConfig"]] = relationship(
        "SourceConfig",
        back_populates="entity_types",
    )
    article: Mapped[Optional["Article"]] = relationship(
        "Article",
        back_populates="entity_types",
    )
    entities: Mapped[List["Entity"]] = relationship(
        "Entity",
        back_populates="entity_type",
    )

    # 唯一约束和索引
    __table_args__ = (
        Index(
            "uk_scope_source_config_article_type",
            "scope",
            "source_config_id",
            "article_id",
            "type",
            unique=True,
        ),
        Index("idx_default_active", "is_default", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<EntityType(id={self.id}, type={self.type}, name={self.name})>"


class Entity(Base):
    """实体表（多对多关系：通过 event_entity 关联表与事项关联）"""

    __tablename__ = "entity"

    # 主键：UUID
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True)

    # 信息源配置ID：UUID（外键）
    source_config_id: Mapped[str] = mapped_column(
        CHAR(36),
        ForeignKey("source_config.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
        index=True,
    )

    # 实体类型ID：UUID（外键）
    entity_type_id: Mapped[str] = mapped_column(
        CHAR(36),
        ForeignKey("entity_type.id", ondelete="RESTRICT", onupdate="CASCADE"),
        nullable=False,
        index=True,
    )

    # 类型标识符（冗余字段，便于查询）
    type: Mapped[str] = mapped_column(String(50), nullable=False)

    # 实体信息
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(500), nullable=False, index=True)

    # 描述
    description: Mapped[Optional[str]] = mapped_column(Text)

    # ========== 类型化值字段（用于统计分析） ==========

    # 值类型标识（int/float/datetime/bool/enum/text）
    value_type: Mapped[Optional[str]] = mapped_column(String(20), index=True)

    # 原始提取文本（保留原始值，如 "199元"）
    value_raw: Mapped[Optional[str]] = mapped_column(Text)

    # 整数值字段
    int_value: Mapped[Optional[int]] = mapped_column(BigInteger, index=True)

    # 浮点数值字段（使用 DECIMAL 保证精度）
    float_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 4), index=True)

    # 日期时间值字段
    datetime_value: Mapped[Optional[datetime]] = mapped_column(DateTime, index=True)

    # 布尔值字段
    bool_value: Mapped[Optional[bool]] = mapped_column(Boolean)

    # 枚举值字段
    enum_value: Mapped[Optional[str]] = mapped_column(String(100), index=True)

    # 单位字段（如 "元", "美元", "公斤"）
    value_unit: Mapped[Optional[str]] = mapped_column(String(50))

    # 解析置信度（0-1）
    value_confidence: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))

    # 扩展数据：{"synonyms": [], "weight": 1.0, "confidence": 1.0}
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON)

    # 时间戳
    created_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # 关系
    source: Mapped["SourceConfig"] = relationship(
        "SourceConfig",
        back_populates="entities",
    )
    entity_type: Mapped["EntityType"] = relationship(
        "EntityType",
        back_populates="entities",
    )
    # 多对多关系：通过 event_entity 关联表
    event_associations: Mapped[List["EventEntity"]] = relationship(
        "EventEntity",
        back_populates="entity",
        cascade="all, delete-orphan",
    )

    # 唯一约束和索引
    __table_args__ = (
        Index(
            "uk_source_config_type_name", "source_config_id", "type", "normalized_name", unique=True
        ),
        Index("idx_source_config_id", "source_config_id"),
        Index("idx_entity_type_id", "entity_type_id"),
        Index("idx_normalized_name", "normalized_name"),
        Index("idx_source_config_type", "source_config_id", "type"),
        # 类型化值复合索引（用于统计查询）
        Index("ix_entity_type_value_type", "type", "value_type"),
        Index("ix_entity_source_config_value_type", "source_config_id", "value_type"),
    )

    def __repr__(self) -> str:
        return f"<Entity(id={self.id}, name={self.name}, type={self.type})>"


class EventEntity(Base):
    """事项-实体关联表（多对多关系）"""

    __tablename__ = "event_entity"
    __table_args__ = (
        Index("uk_event_entity", "event_id", "entity_id", unique=True),
        Index("idx_event_id", "event_id"),
        Index("idx_entity_id", "entity_id"),
    )

    # 主键：UUID
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True)

    # 事项ID：UUID（外键）
    event_id: Mapped[str] = mapped_column(
        CHAR(36),
        ForeignKey("source_event.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
        index=True,
    )

    # 实体ID：UUID（外键）
    entity_id: Mapped[str] = mapped_column(
        CHAR(36),
        ForeignKey("entity.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
        index=True,
    )

    # 该实体在此事项中的权重（0.00-9.99）
    weight: Mapped[Decimal] = mapped_column(Numeric(3, 2), default=Decimal("1.00"))

    # 该实体在此事项中的描述/角色（如："某公司CEO"、"天使投资人"）
    description: Mapped[Optional[str]] = mapped_column(Text)

    # 扩展数据：{"confidence": 0.95, "context": ""}
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON)

    # 时间戳
    created_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    # 关系
    event: Mapped["SourceEvent"] = relationship(
        "SourceEvent",
        back_populates="event_associations",
        lazy="noload",  # 防止延迟加载错误
    )
    entity: Mapped["Entity"] = relationship(
        "Entity",
        back_populates="event_associations",
        lazy="noload",  # 防止延迟加载错误
    )
    embedding: Mapped[Optional["EventEntityEmbedding"]] = relationship(
        "EventEntityEmbedding",
        back_populates="event_entity",
        lazy="noload",
        uselist=False,
    )

    def __repr__(self) -> str:
        return f"<EventEntity(event_id={self.event_id}, entity_id={self.entity_id}, weight={self.weight})>"


class EventEntityEmbedding(Base):
    """事项实体向量表（一对一）"""

    __tablename__ = "event_entity_embedding"
    __table_args__ = (
        ForeignKeyConstraint(
            ["id"],
            ["event_entity.id"],
            ondelete="CASCADE",
            onupdate="CASCADE",
            name="fk_event_entity_embedding_event_entity",
        ),
        Index("idx_event_entity_embedding_updated_time", "updated_time"),
        {"comment": "事项实体向量表（一对一）"},
    )

    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True, comment="关联 event_entity.id")
    vec: Mapped[bytes] = mapped_column(
        VARBINARY(512),
        nullable=False,
        comment="128-dim float32 embedding bytes",
    )
    created_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False
    )
    updated_time: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"),
        nullable=False,
    )

    event_entity: Mapped["EventEntity"] = relationship(
        "EventEntity",
        back_populates="embedding",
        lazy="noload",
        uselist=False,
    )


class SourceEvent(Base):
    """源事件表"""

    __tablename__ = "source_event"

    # 主键：UUID
    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True)

    # 信息源配置ID：UUID（外键）
    source_config_id: Mapped[str] = mapped_column(
        CHAR(36),
        ForeignKey("source_config.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
        index=True,
    )

    # 来源标识（多态字段，统一接口）
    source_type: Mapped[str] = mapped_column(
        String(20), nullable=False, index=True, comment="来源类型：ARTICLE/CHAT"
    )
    source_id: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True, comment="来源ID"
    )

    # 文章ID：UUID（外键）
    article_id: Mapped[Optional[str]] = mapped_column(
        CHAR(36),
        ForeignKey("article.id", ondelete="RESTRICT", onupdate="CASCADE"),
        nullable=True,
        index=True,
    )

    # 事件信息
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    summary: Mapped[str] = mapped_column(MEDIUMTEXT, nullable=False)
    content: Mapped[str] = mapped_column(LONGTEXT, nullable=False)

    # 事项分类（如：技术、产品、市场、研究、管理等）
    category: Mapped[Optional[str]] = mapped_column(String(50), default="")

    # 关键词列表
    keywords: Mapped[Optional[dict]] = mapped_column(JSON, comment="关键词列表")

    # 业务字段（兼容主系统）
    type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    priority: Mapped[Optional[str]] = mapped_column(String(50), default="")
    status: Mapped[Optional[str]] = mapped_column(String(50), default="")

    # 排序序号
    rank: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # 层级结构字段
    level: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False, comment="层级深度（0=顶层）"
    )
    parent_id: Mapped[Optional[str]] = mapped_column(
        CHAR(36),
        ForeignKey("source_event.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=True,
        index=True,
        comment="父事项ID（自引用）",
    )

    # 时间范围
    start_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # 原始片段引用
    references: Mapped[Optional[dict]] = mapped_column(JSON)

    # 来源片段ID：UUID（指向 SourceChunk）
    chunk_id: Mapped[Optional[str]] = mapped_column(CHAR(36), index=True)

    # 扩展数据：{"keywords": [], "category": "", "priority": "", "status": ""}
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON)

    # 时间戳
    created_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_time: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # 关系
    source: Mapped["SourceConfig"] = relationship(
        "SourceConfig",
        back_populates="source_events",
    )
    article: Mapped[Optional["Article"]] = relationship(
        "Article",
        back_populates="source_events",
    )
    # 多对多关系：通过 event_entity 关联表
    event_associations: Mapped[List["EventEntity"]] = relationship(
        "EventEntity",
        back_populates="event",
        cascade="all, delete-orphan",
    )
    # 层级关系：父子事项（自引用）
    parent: Mapped[Optional["SourceEvent"]] = relationship(
        "SourceEvent",
        remote_side="SourceEvent.id",
        back_populates="children",
    )
    children: Mapped[List["SourceEvent"]] = relationship(
        "SourceEvent",
        back_populates="parent",
        cascade="all, delete-orphan",
    )

    @property
    def entities(self) -> List["Entity"]:
        """通过关联表访问实体列表"""
        return [assoc.entity for assoc in self.event_associations]

    # 索引
    # 注意：MySQL 不支持在有外键动作的列上使用 CHECK 约束，数据完整性由应用层保证
    __table_args__ = (
        Index("idx_source_config_id", "source_config_id"),
        Index("idx_source", "source_type", "source_id"),
        Index("idx_source_rank", "source_type", "source_id", "rank"),
        Index("idx_article_id", "article_id"),
        Index("idx_article_rank", "article_id", "rank"),
        Index("idx_chunk_id", "chunk_id"),
        Index("idx_parent_id", "parent_id"),
        Index("idx_level", "level"),
        Index("idx_parent_level", "parent_id", "level"),
        Index("idx_start_time", "start_time"),
        Index("idx_end_time", "end_time"),
    )

    def __repr__(self) -> str:
        return f"<SourceEvent(id={self.id}, title={self.title[:30]})>"

    @classmethod
    def not_deleted(cls):
        return or_(cls.status.is_(None), cls.status != "DELETED")



class SourceChunk(Base):
    """
    来源片段聚合表 - 聚合ArticleSection句子或ChatMessage句子为片段
    """

    __tablename__ = "source_chunk"

    # 有默认值的字段 - 主键
    id: Mapped[str] = mapped_column(
        CHAR(36),
        primary_key=True,
        default=lambda: str(__import__("uuid").uuid4()),
    )

    # 信息源配置ID（必填，外键）
    source_config_id: Mapped[str] = mapped_column(
        CHAR(36),
        ForeignKey("source_config.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
        index=True,
    )

    # 来源标识（多态字段，主要使用）
    source_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    source_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    # 外键字段（级联删除）
    article_id: Mapped[Optional[str]] = mapped_column(
        CHAR(36),
        ForeignKey("article.id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=True,
        index=True,
    )

    # 可选字段（无默认值但可为空）
    content: Mapped[Optional[str]] = mapped_column(LONGTEXT, nullable=True)
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    references: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    heading: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    raw_content: Mapped[Optional[str]] = mapped_column(LONGTEXT, nullable=True)

    # 有默认值的字段
    rank: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_length: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_time: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_time: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # 关系
    source_config: Mapped["SourceConfig"] = relationship("SourceConfig")
    article: Mapped[Optional["Article"]] = relationship("Article")

    # 索引
    __table_args__ = (
        Index("idx_source", "source_type", "source_id", "rank"),
        Index("idx_source_config_id", "source_config_id"),
        Index("idx_article_id", "article_id"),
        Index("idx_created", "created_time"),
        {"comment": "来源片段聚合表 - 聚合ArticleSection为片段"},
    )

    def __repr__(self) -> str:
        return f"<SourceChunk(id={self.id}, source_type={self.source_type}, source_id={self.source_id})>"


__all__ = [
    "SourceConfig",
    "KBDocument",
    "ArticleParseStatus",
    "Article",
    "ArticleSection",
    "EntityType",
    "Entity",
    "EventEntity",
    "EventEntityEmbedding",
    "SourceEvent",
    "SourceChunk",
]
