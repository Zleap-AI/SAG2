"""
工具函数模块
"""

from pipeline.utils.logger import get_logger, logger, setup_logging
from pipeline.utils.text import (
    clean_whitespace,
    compute_text_hash,
    count_chinese_characters,
    estimate_tokens,
    extract_markdown_headings,
    normalize_heading_text,
    normalize_entity_name,
    normalize_text,
    split_text_by_paragraphs,
    truncate_text,
)
from pipeline.utils.time import (
    calculate_time_decay,
    format_datetime,
    get_time_ago,
    get_utc_now,
    parse_iso_datetime,
)
from pipeline.utils.text import TokenEstimator
from pipeline.utils.retry import (
    is_retryable_error,
    is_retryable_db_error,
    is_retryable_network_error,
    retry_async,
)
from pipeline.utils.batch import (
    batch_generate_embeddings,
    batch_index_to_es,
    EmbeddingBatchProcessor,
    ESBulkIndexProcessor,
)

__all__ = [
    # Logger
    "setup_logging",
    "get_logger",
    "logger",
    # Text
    "normalize_text",
    "normalize_heading_text",
    "normalize_entity_name",
    "extract_markdown_headings",
    "compute_text_hash",
    "truncate_text",
    "split_text_by_paragraphs",
    "estimate_tokens",
    "clean_whitespace",
    "count_chinese_characters",
    # Time
    "get_utc_now",
    "parse_iso_datetime",
    "format_datetime",
    "get_time_ago",
    "calculate_time_decay",
    # Token
    "TokenEstimator",
    # Retry
    "is_retryable_error",
    "is_retryable_db_error",
    "is_retryable_network_error",
    "retry_async",
    # Batch
    "batch_generate_embeddings",
    "batch_index_to_es",
    "EmbeddingBatchProcessor",
    "ESBulkIndexProcessor",
]
