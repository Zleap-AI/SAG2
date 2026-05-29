"""
LLM客户端基类

定义LLM客户端的统一接口
"""

import asyncio
import random
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

from pipeline.core.ai.models import ModelConfig, LLMMessage, LLMResponse, LLMRole
from pipeline.exceptions import LLMError, LLMTimeoutError
from pipeline.utils import get_logger

logger = get_logger("ai.llm")


class BaseLLMClient(ABC):
    """LLM客户端基类"""

    def __init__(self, config: ModelConfig) -> None:
        """
        初始化LLM客户端

        Args:
            config: LLM配置
        """
        self.config = config
        logger.info(
            "初始化%s客户端",
            config.provider.value,
            extra={"model": config.model},
        )

    @abstractmethod
    async def chat(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        聊天补全

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大输出token数
            **kwargs: 其他参数

        Returns:
            LLM响应

        Raises:
            LLMError: LLM调用失败
            LLMTimeoutError: 调用超时
        """
        ...

    @abstractmethod
    def chat_stream(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        include_reasoning: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[str, Optional[str]]]:
        """
        流式聊天补全

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大输出token数
            include_reasoning: 是否返回推理内容（reasoning_content）
            **kwargs: 其他参数

        Yields:
            元组 (content, reasoning) - content为内容片段，reasoning为推理片段（如果有）

        Raises:
            LLMError: LLM调用失败
            LLMTimeoutError: 调用超时
        """
        ...

    async def chat_with_schema(
        self,
        messages: List[LLMMessage],
        response_schema: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        结构化输出（JSON Schema）

        注意：不自动注入提示词，调用方需在 messages 中自行定义输出格式要求。
        本方法只负责：1. 调用 LLM  2. 解析 JSON  3. Schema 校验（如果提供）

        Args:
            messages: 消息列表（应包含 SYSTEM 定义的输出格式）
            response_schema: JSON Schema定义（用于校验，可选）
            temperature: 温度参数
            max_tokens: 最大输出token数
            **kwargs: 其他参数

        Returns:
            解析后的JSON对象

        Raises:
            LLMError: LLM调用失败或JSON格式无效
            ValidationError: 响应不符合Schema（仅当提供schema时）
        """
        import json

        # 直接调用 LLM，不注入额外提示词
        response = await self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # 解析JSON响应
        try:
            import re

            # 提取JSON内容（可能被markdown代码块包裹）
            content = response.content.strip()

            # 使用正则表达式提取 ```json 或 ``` 代码块
            json_block_match = re.search(
                r"```(?:json)?\s*\n(.*?)\n```",
                content,
                re.DOTALL | re.IGNORECASE,
            )
            if json_block_match:
                content = json_block_match.group(1).strip()
                logger.debug("从 markdown 代码块中提取 JSON")
            else:
                logger.debug("直接解析 JSON（无代码块）")

            # 解析：先 json.loads，失败则用 json_repair 兜底（尾逗号、多余文字等）
            try:
                result = json.loads(content)
            except json.JSONDecodeError as parse_err:
                try:
                    import json_repair

                    result = json_repair.loads(content)
                    logger.info("LLM 返回的 JSON 经 json_repair 修复后解析成功")
                except Exception:
                    raise parse_err

            # 某些模型会返回“JSON字符串包裹JSON对象”，这里做一次解包。
            if isinstance(result, str):
                nested_content = result.strip()
                if nested_content.startswith(("{", "[")):
                    try:
                        result = json.loads(nested_content)
                    except json.JSONDecodeError:
                        try:
                            import json_repair

                            result = json_repair.loads(nested_content)
                            logger.info("LLM 返回了嵌套 JSON 字符串，已二次解析成功")
                        except Exception:
                            pass

            # 如果提供了schema，进行验证
            if response_schema:
                expected_type = response_schema.get("type")
                if expected_type == "object" and not isinstance(result, dict):
                    raise LLMError(
                        f"响应类型不符合Schema: 期望 object，实际 {type(result).__name__}"
                    )
                if expected_type == "array" and not isinstance(result, list):
                    raise LLMError(
                        f"响应类型不符合Schema: 期望 array，实际 {type(result).__name__}"
                    )

                # 尝试使用jsonschema进行严格验证
                try:
                    import jsonschema

                    jsonschema.validate(instance=result, schema=response_schema)
                    logger.debug("JSON schema validation passed")
                except ImportError:
                    # jsonschema未安装，使用简单验证（仅当根为 dict 时检查 required）
                    if isinstance(result, dict) and "properties" in response_schema:
                        required = response_schema.get("required", [])
                        for field in required:
                            if field not in result:
                                raise ValueError(f"缺少必需字段: {field}")
                    logger.debug("JSON simple validation passed")
                except Exception as e:
                    # jsonschema验证失败
                    if type(e).__name__ == "ValidationError":
                        logger.error(
                            "JSON schema validation failed: %s\n响应内容: %s",
                            e,
                            str(result)[:500],
                        )
                        raise LLMError(f"响应不符合Schema: {e}") from e
                    raise
            else:
                # 没有schema，只验证JSON格式（已通过json.loads）
                logger.debug("JSON format validation passed (no schema provided)")

            return result

        except json.JSONDecodeError as e:
            logger.error("JSON解析失败: %s\n内容: %s", e, response.content)
            raise LLMError(f"LLM返回的不是有效的JSON: {e}") from e
        except ValueError as e:
            logger.error("Schema验证失败: %s", e)
            raise LLMError(f"响应不符合Schema: {e}") from e

    def _prepare_messages(
        self,
        messages: List[LLMMessage],
    ) -> List[Dict[str, str]]:
        """
        准备消息列表（转换为API格式）

        Args:
            messages: 消息列表

        Returns:
            API格式的消息列表
        """
        return [msg.to_dict() for msg in messages]

    async def close(self) -> None:
        """
        关闭客户端，释放资源

        子类如果有需要关闭的资源（如HTTP连接），应该重写此方法
        """
        pass


class LLMRetryClient:
    """带重试机制的LLM客户端包装器"""

    def __init__(
        self,
        client: BaseLLMClient,
        max_retries: Optional[int] = None,
        retry_delay: float = 4.0,
        backoff_factor: float = 2.0,
    ) -> None:
        """
        初始化重试客户端

        Args:
            client: 基础LLM客户端
            max_retries: 最大重试次数（None则使用client配置）
            retry_delay: 初始重试延迟（秒）
            backoff_factor: 退避因子
        """
        self.client = client
        self.max_retries = max_retries or client.config.max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor

    def _should_retry(self, error: Exception) -> bool:
        """
        判断错误是否应该重试

        Args:
            error: 异常对象

        Returns:
            True表示应该重试，False表示不应该重试
        """
        # 超时错误不重试（网络问题，重试可能继续超时）
        if isinstance(error, LLMTimeoutError):
            return False

        # 速率限制错误应该重试
        from pipeline.exceptions import LLMRateLimitError

        if isinstance(error, LLMRateLimitError):
            return True

        # 其他LLM错误可以重试
        if isinstance(error, LLMError):
            return True

        # 未知错误默认不重试
        return False

    def _compute_delay(self, attempt: int) -> float:
        """
        计算指数退避延迟（含随机抖动）

        delay = retry_delay × backoff_factor^attempt × (0.5 ~ 1.0 jitter)

        示例 (retry_delay=4, backoff_factor=2):
          attempt 0: 4 × 1  × jitter = 2.0~4.0s
          attempt 1: 4 × 2  × jitter = 4.0~8.0s
          attempt 2: 4 × 4  × jitter = 8.0~16.0s
          attempt 3: 4 × 8  × jitter = 16.0~32.0s
          attempt 4: 4 × 16 × jitter = 32.0~64.0s
        """
        base_delay = self.retry_delay * (self.backoff_factor ** attempt)
        jitter = 0.5 + random.random() * 0.5
        return base_delay * jitter

    async def chat(
        self,
        messages: List[LLMMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        带重试的聊天补全

        实现指数退避重试策略（含随机抖动，避免多 worker 同时重试）

        根据错误类型智能决定是否重试：
        - 超时错误：不重试
        - 速率限制：重试
        - 其他LLM错误：重试
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                return await self.client.chat(messages, **kwargs)
            except Exception as e:
                last_error = e

                # 判断是否应该重试
                if not self._should_retry(e):
                    logger.error("遇到不可重试错误: %s", e)
                    raise

                if attempt < self.max_retries:
                    delay = self._compute_delay(attempt)
                    logger.warning(
                        "LLM调用失败，%.1fs后重试 (尝试 %d/%d)",
                        delay,
                        attempt + 1,
                        self.max_retries,
                        extra={"error": str(e), "error_type": type(e).__name__},
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "LLM调用失败，已重试%d次",
                        self.max_retries,
                        exc_info=True,
                    )

        raise LLMError(f"LLM调用失败，已重试{self.max_retries}次") from last_error

    async def chat_stream(
        self,
        messages: List[LLMMessage],
        **kwargs: Any,
    ) -> AsyncIterator[tuple[str, Optional[str]]]:
        """
        流式调用（不重试）

        流式调用失败时无法重试，直接抛出异常

        Yields:
            元组 (content, reasoning) - content为内容片段，reasoning为推理片段（如果有）
        """
        async for chunk in self.client.chat_stream(messages, **kwargs):
            yield chunk

    async def chat_with_schema(
        self,
        messages: List[LLMMessage],
        response_schema: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        带重试的结构化输出

        根据错误类型智能决定是否重试
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                return await self.client.chat_with_schema(
                    messages,
                    response_schema,
                    **kwargs,
                )
            except Exception as e:
                last_error = e

                # 判断是否应该重试
                if not self._should_retry(e):
                    logger.error("遇到不可重试错误: %s", e)
                    raise

                if attempt < self.max_retries:
                    delay = self._compute_delay(attempt)
                    logger.warning(
                        "结构化输出失败，%.1fs后重试 (尝试 %d/%d)",
                        delay,
                        attempt + 1,
                        self.max_retries,
                        extra={"error": str(e), "error_type": type(e).__name__},
                    )
                    await asyncio.sleep(delay)

        raise LLMError(
            f"结构化输出失败，已重试{self.max_retries}次, 最后一次错误: {last_error}"
        ) from last_error
