"""
数据库模块

提供SQLAlchemy ORM模型和数据库操作
"""

from pipeline.db.base import Base, get_engine, get_session_factory, init_database, close_database, reset_engine
from pipeline.db.models import (
    Article,
    ArticleParseStatus,
    ArticleSection,
    Entity,
    EntityType,
    EventEntity,
    EventEntityEmbedding,
    KBDocument,
    SourceChunk,
    SourceConfig,
    SourceEvent,
)

__all__ = [
    # Base
    "Base",
    "get_engine",
    "get_session_factory",
    "init_database",
    "close_database",
    "reset_engine",
    # Models
    "SourceConfig",
    "Article",
    "ArticleParseStatus",
    "ArticleSection",
    "EntityType",
    "Entity",
    "EventEntity",
    "EventEntityEmbedding",
    "SourceEvent",
    "SourceChunk",
    "KBDocument",
]
