"""
LLM客户端工厂

根据配置创建相应的LLM客户端，支持场景化配置
"""

import hashlib
import json
from typing import Any, Dict, Optional

from pipeline.core.ai.base import BaseLLMClient, LLMRetryClient
from pipeline.core.ai.models import ModelConfig, LLMProvider
from pipeline.core.ai.llm import OpenAIClient
from pipeline.core.config import get_settings
from pipeline.exceptions import ConfigError
from pipeline.utils import get_logger

logger = get_logger("ai.factory")

# 全局客户端单例
_embedding_client = None
_embedding_config_fingerprint: Optional[str] = None


def _get_client_fingerprint(config: Dict[str, Any]) -> str:
    """
    生成客户端配置指纹（通用函数）

    只包含影响客户端实例的核心参数：
    - model: 模型名称
    - api_key: API密钥
    - base_url: API地址

    其他参数（temperature, dimensions, timeout等）不影响客户端实例本身

    Args:
        config: 配置字典

    Returns:
        配置指纹（MD5 hash）
    """
    key_params = {
        "model": config.get("model"),
        "api_key": config.get("api_key"),
        "base_url": config.get("base_url"),
    }
    # 生成配置的hash值
    config_str = json.dumps(key_params, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


async def _load_db_config(type: str = "llm", scenario: str = "general") -> Optional[Dict[str, Any]]:
    """
    从数据库加载模型配置（通用函数）- 已禁用，默认使用环境变量

    降级策略（针对 LLM）：
    1. 查询 type + scenario 的专用配置
    2. 降级到 type + 'general'
    3. 返回 None（使用环境变量兜底）

    对于 Embedding 等：
    - 直接查 type + scenario（通常是 general）

    Args:
        type: 模型类型 (llm/embedding)
        scenario: 使用场景

    Returns:
        配置字典或None
    """
    # 默认使用环境变量配置，不再从数据库加载
    logger.debug(f"使用环境变量配置: type={type}, scenario={scenario}")
    return None




async def create_llm_client(
    scenario: str = "general",
    model_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> BaseLLMClient | LLMRetryClient:
    """
    创建LLM客户端（统一入口，支持场景化配置）

    配置优先级（从高到低）：
    1. model_config 显式传入
    2. 环境变量配置 (兜底)

    Args:
        scenario: 场景标识，默认 'general'
            - 'extract' : 事项提取
            - 'search'  : 搜索
            - 'chat'    : 对话
            - 'summary' : 摘要
            - 'system'  : 系统（Agent创建等）
            - 'general' : 通用（默认）

        model_config: LLM配置字典（可选）
            {
                'model': 'gpt-4',
                'api_key': 'sk-xxx',
                'base_url': 'https://api.302.ai',
                'temperature': 0.7,
                'max_tokens': 8000,
                ...
            }
            - 如果传入：直接使用（最高优先级）
            - 如果不传：自动从配置管理器获取

        **kwargs: 零散参数（向后兼容）

    Returns:
        LLM客户端实例

    Raises:
        ConfigError: 无法获取有效配置时抛出

    Examples:
        # 方式1：只传场景，自动获取配置（推荐）
        >>> client = await create_llm_client(scenario='extract')

        # 方式2：显式传入配置
        >>> client = await create_llm_client(
        ...     scenario='extract',
        ...     model_config={'model': 'gpt-4', 'temperature': 0.1}
        ... )

        # 方式3：使用默认通用场景
        >>> client = await create_llm_client()

    说明：
    - 统一使用 OpenAIClient（兼容 OpenAI 官方 + 302.AI 中转）
    - 通过 base_url 区分不同服务商
    """
    settings = get_settings()

    # ============ 配置合并（三层优先级）============

    # Layer 3: 环境变量兜底
    config = {
        "model": settings.llm_model,
        "api_key": settings.llm_api_key,
        "base_url": settings.llm_base_url,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "top_p": settings.llm_top_p,
        "frequency_penalty": settings.llm_frequency_penalty,
        "presence_penalty": settings.llm_presence_penalty,
        "timeout": settings.llm_timeout,
        "max_retries": settings.llm_max_retries,
    }

    # Layer 2: 数据库配置已移除，直接使用环境变量

    # Layer 1: 显式配置（最高优先级）
    if model_config:
        config.update(model_config)
        logger.debug(f"🎯 使用显式配置: scenario={scenario}")

    # 兼容零散参数（向后兼容）
    if kwargs:
        config.update(kwargs)

    # ============ 验证必需参数 ============
    if not config.get("api_key"):
        raise ConfigError(
            f"❌ LLM配置错误：缺少 API Key！\n"
            f"场景: {scenario}\n"
            f"请检查环境变量 LLM_API_KEY"
        )

    if not config.get("model"):
        raise ConfigError(f"❌ LLM配置错误：缺少模型名称！场景: {scenario}")

    # ============ 构建配置对象 ============
    model_config_obj = ModelConfig(
        provider=LLMProvider.OPENAI,  # 统一使用 OPENAI（兼容所有中转服务）
        model=config["model"],
        api_key=config["api_key"],
        base_url=config.get("base_url"),
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        top_p=config["top_p"],
        frequency_penalty=config["frequency_penalty"],
        presence_penalty=config["presence_penalty"],
        timeout=config["timeout"],
        max_retries=config["max_retries"],
    )

    # ============ 创建客户端（统一使用OpenAIClient）============
    # OpenAIClient 兼容：OpenAI 官方 + 302.AI 中转 + 其他兼容服务
    base_client = OpenAIClient(model_config_obj)

    # 包装重试机制
    with_retry = config.get("with_retry", True)
    if with_retry:
        logger.debug(
            f"✅ 创建LLM客户端（带重试）: scenario={scenario}",
            extra={
                "scenario": scenario,
                "model": config["model"],
                "base_url": config.get("base_url") or "OpenAI官方",
                "max_retries": config["max_retries"],
            },
        )
        return LLMRetryClient(base_client)

    logger.debug(
        f"✅ 创建LLM客户端: scenario={scenario}",
        extra={
            "scenario": scenario,
            "model": config["model"],
        },
    )
    return base_client


# ============================================================
# 说明：
# - LLM 客户端：每次创建新实例，各模块自行管理（extractor, searcher 等）
# - Embedding 客户端：全局单例，配置变更自动替换
# ============================================================


# ============ Embedding 客户端工厂 ============


async def create_embedding_client(
    scenario: str = "general",
    embedding_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> "EmbeddingClient":
    """
    创建Embedding客户端（统一入口，支持分层配置）

    配置优先级（从高到低）：
    1. embedding_config 显式传入
    2. 数据库配置 (if USE_DB_CONFIG=true, model_type='embedding')
    3. 环境变量配置 (兜底)

    Args:
        scenario: 使用场景，默认 'general'（当前 embedding 只用 general，未来可扩展）
        embedding_config: Embedding配置字典（可选）
            {
                'model': 'Qwen/Qwen3-Embedding-0.6B',
                'api_key': 'sk-xxx',
                'base_url': 'https://api.302.ai',
                'dimensions': 1536,
                ...
            }
        **kwargs: 零散参数（向后兼容）

    Returns:
        EmbeddingClient实例

    Raises:
        ConfigError: 无法获取有效配置时抛出

    Examples:
        # 方式1：自动获取配置（推荐）
        >>> client = await create_embedding_client()

        # 方式2：显式传入配置
        >>> client = await create_embedding_client(
        ...     embedding_config={'model': 'text-embedding-3-large'}
        ... )
    """
    settings = get_settings()

    # ============ 配置合并（三层优先级）============

    # Layer 3: 环境变量兜底
    config = {
        "model": settings.embedding_model_name,
        "api_key": settings.embedding_api_key or settings.llm_api_key,
        "base_url": settings.embedding_base_url or settings.llm_base_url,
        "dimensions": settings.embedding_dimensions,
        "timeout": 60,
        "max_retries": 3,
    }

    # Layer 2: 数据库配置（指定 type='embedding'）
    if settings.use_db_config:
        db_config = await _load_db_config(type="embedding", scenario=scenario)
        if db_config:
            # 提取 dimensions（可能在 extra_data 中）
            if "extra_data" in db_config and db_config["extra_data"]:
                if "dimensions" in db_config["extra_data"]:
                    db_config["dimensions"] = db_config["extra_data"]["dimensions"]
            config.update(db_config)
            logger.info(f"📊 使用数据库Embedding配置: model={db_config.get('model')}")
        else:
            logger.debug("数据库无Embedding配置，使用环境变量")

    # Layer 1: 显式配置（最高优先级）
    if embedding_config:
        config.update(embedding_config)
        logger.info("🎯 使用显式Embedding配置")

    # 兼容零散参数
    if kwargs:
        config.update(kwargs)

    # ============ 验证必需参数 ============
    if not config.get("api_key"):
        raise ConfigError(
            "❌ Embedding配置错误：缺少 API Key！\n"
            f"场景: {scenario}\n"
            "请检查：数据库配置 或 环境变量 EMBEDDING_API_KEY/LLM_API_KEY"
        )

    if not config.get("model"):
        raise ConfigError(f"❌ Embedding配置错误：缺少模型名称！场景: {scenario}")

    # ============ 创建客户端 ============
    from pipeline.core.ai.embedding import EmbeddingClient

    # ✅ 提取参数创建客户端（包含 api_key，确保数据库配置生效）
    client = EmbeddingClient(
        model=config["model"], base_url=config.get("base_url"), api_key=config.get("api_key")
    )

    # 如果有 dimensions 参数，需要在生成时传递
    # TODO: 更新 EmbeddingClient.generate() 支持 dimensions 参数

    logger.info(
        "✅ 创建Embedding客户端",
        extra={
            "scenario": scenario,
            "model": config["model"],
            "base_url": config.get("base_url") or "OpenAI官方",
            "dimensions": config.get("dimensions") or "默认",
        },
    )
    return client


# 全局 Embedding 客户端单例（配置变更时自动替换）
_embedding_client: Optional["EmbeddingClient"] = None
_embedding_config_fingerprint: Optional[str] = None


async def get_embedding_client(scenario: str = "general") -> "EmbeddingClient":
    """
    获取Embedding客户端（单例，配置自动更新）

    工作原理：
    - 维护全局唯一实例
    - 每次调用检测配置是否变化（基于指纹）
    - 配置变化时自动替换为新实例
    - 配置未变时复用现有实例

    指纹参数：model, api_key, base_url（通用三要素）

    Args:
        scenario: 使用场景，默认 'general'

    Returns:
        EmbeddingClient实例
    """
    global _embedding_client, _embedding_config_fingerprint

    # 1. 获取完整配置（合并环境变量、数据库配置等）
    settings = get_settings()
    config = {
        "model": settings.embedding_model_name,
        "api_key": settings.embedding_api_key or settings.llm_api_key,
        "base_url": settings.embedding_base_url or settings.llm_base_url,
        "dimensions": settings.embedding_dimensions,
        "timeout": 60,
        "max_retries": 3,
    }

    # 2. 尝试从数据库加载配置
    if settings.use_db_config:
        db_config = await _load_db_config(type="embedding", scenario=scenario)
        if db_config:
            # 提取 dimensions（可能在 extra_data 中）
            if "extra_data" in db_config and db_config["extra_data"]:
                if "dimensions" in db_config["extra_data"]:
                    db_config["dimensions"] = db_config["extra_data"]["dimensions"]
            config.update(db_config)
            logger.debug(f"使用数据库Embedding配置: model={db_config.get('model')}")

    # 3. 生成配置指纹（基于关键参数：model, api_key, base_url）
    current_fingerprint = _get_client_fingerprint(config)

    # 4. 检查配置是否变化
    if _embedding_client is None or current_fingerprint != _embedding_config_fingerprint:
        # 配置变化或首次创建
        action = "更新" if _embedding_client else "创建"

        from pipeline.core.ai.embedding import EmbeddingClient

        _embedding_client = EmbeddingClient(
            model=config["model"], base_url=config.get("base_url"), api_key=config.get("api_key")
        )
        _embedding_config_fingerprint = current_fingerprint

        logger.info(
            f"🔄 {action}Embedding客户端: model={config['model']}, "
            f"base_url={config.get('base_url') or '默认'}, "
            f"fingerprint={current_fingerprint[:8]}..."
        )
    else:
        logger.debug(f"♻️ 复用Embedding客户端（配置未变）: {config['model']}")

    return _embedding_client


def reset_embedding_client() -> None:
    """重置Embedding客户端单例"""
    global _embedding_client, _embedding_config_fingerprint
    _embedding_client = None
    _embedding_config_fingerprint = None
    logger.info("已重置Embedding客户端")


async def close_all_clients() -> None:
    """关闭所有全局客户端，释放资源"""
    global _embedding_client, _embedding_config_fingerprint

    if _embedding_client:
        try:
            # EmbeddingClient 可能没有 close 方法，先检查
            if hasattr(_embedding_client, 'close'):
                await _embedding_client.close()
            logger.info("已关闭Embedding客户端")
        except Exception as e:
            logger.warning(f"关闭Embedding客户端时出错: {e}")
        finally:
            _embedding_client = None
            _embedding_config_fingerprint = None

