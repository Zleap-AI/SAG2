"""
存储模块

提供 Elasticsearch 客户端及 Repository 访问层
"""

from pipeline.core.storage.documents import (
    SourceChunkDocument,
    EntityVectorDocument,
    EventVectorDocument,
    REGISTERED_DOCUMENTS,
)
from pipeline.core.storage.elasticsearch import (
    ESConfig,
    ElasticsearchClient,
    close_es_client,
    get_es_client,
)
from pipeline.core.storage.repositories import (
    BaseRepository,
    EntityVectorRepository,
    EventVectorRepository,
    SourceChunkRepository,
)

__all__ = [
    # Elasticsearch
    "ESConfig",
    "ElasticsearchClient",
    "get_es_client",
    "close_es_client",
    # ES Documents
    "EntityVectorDocument",
    "EventVectorDocument",
    "SourceChunkDocument",
    "REGISTERED_DOCUMENTS",
    # ES Repositories
    "BaseRepository",
    "EntityVectorRepository",
    "EventVectorRepository",
    "SourceChunkRepository",
]
