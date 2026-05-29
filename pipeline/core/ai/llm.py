"""
OpenAI LLM客户端实现

注意:
- 支持标准OpenAI模型 (sophnet/Qwen3-30B-A3B-Thinking-2507, gpt-3.5-turbo等)
- 支持思考模型 (Thinking Models): 某些模型(如Qwen3-30B-A3B-Thinking)会将推理过程
  放在reasoning_content字段中而不是content字段。本实现会自动检测并处理这种情况。
"""

from typing import Any, AsyncIterator, Iterable, List, Optional, cast

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from openai.types.chat import ChatCompletionMessageParam

from pipeline.core.ai.base import BaseLLMClient
from pipeline.core.ai.models import ModelConfig, LLMMessage, LLMProvider, LLMResponse, LLMUsage
from pipeline.exceptions import LLMError, LLMRateLimitError, LLMTimeoutError
from pipeline.core.config.settings import get_settings 
from pipeline.utils import get_logger

logger = get_logger("ai.openai")


class OpenAIClient(BaseLLMClient):
    """OpenAI客户端实现"""

    def __init__(self, config: ModelConfig) -> None:
        """
        初始化OpenAI客户端

        Args:
            config: LLM配置
        """
        super().__init__(config)

        # 构建默认 headers（用于控制内容过滤等）
        default_headers = self._build_default_headers()

        # 创建AsyncOpenAI客户端
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            default_headers=default_headers if default_headers else None,
        )

    def _build_default_headers(self) -> dict:
        """
        构建默认请求头

        Returns:
            默认请求头字典
        """


        settings = get_settings()
        headers = {}

        # 如果禁用内容过滤（绿网），添加 DashScope header
        if not settings.llm_data_inspection:
            headers["X-DashScope-DataInspection"] = '{"input": "disable", "output": "disable"}'

        return headers

    async def chat(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        OpenAI聊天补全

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大输出token数
            **kwargs: 其他参数

        Returns:
            LLM响应

        Raises:
            LLMError: 调用失败
            LLMTimeoutError: 调用超时
            LLMRateLimitError: 速率限制
        """
        try:
            # 准备消息
            api_messages = self._prepare_messages(messages)
            # 获取配置
            settings = get_settings()
            # 记录使用的模型信息
            logger.info(
                "🤖 调用 LLM - 模型: %s, base_url: %s, temperature: %.2f, max_tokens: %s, timeout: %s, enable_think: %s",
                self.config.model,
                self.config.base_url,
                temperature or self.config.temperature,
                max_tokens or self.config.max_tokens or "未设置",
                self.config.timeout,
                settings.llm_enable_think
            )

            # 记录消息内容（调试用）
            logger.debug(
                "📤 LLM 请求消息 (%d条): %s",
                len(messages),
                [
                    {
                        "role": m.role,
                        "content": (
                            m.content[:10000] + "..." if len(m.content) > 10000 else m.content
                        ),
                    }
                    for m in messages
                ],
            )

            # 调用API（使用 cast 显式类型转换）
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=cast(Iterable[ChatCompletionMessageParam], api_messages),
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,  # 从配置读取，不硬编码
                extra_body = {
                "chat_template_kwargs": {"enable_thinking": settings.llm_enable_think},
                },
                **kwargs,
            )

            # 解析响应
            choice = response.choices[0]
            usage = response.usage

            # 处理响应内容
            content = choice.message.content
            reasoning = getattr(choice.message, "reasoning_content", None) or getattr(choice.message, "reasoning", None)  

            logger.debug(
                "OpenAI响应: content=%s, reasoning_content=%s, finish_reason=%s",
                choice.message.content,
                reasoning,
                choice.finish_reason,
            )
            #  添加总 token 数
            logger.info(
                f"Token usage | prompt: {usage.prompt_tokens}, "
                f"completion: {usage.completion_tokens}, "
                f"total: {usage.prompt_tokens + usage.completion_tokens}"
            )


            return LLMResponse(
                content=content or "",
                model=response.model,
                usage=LLMUsage(
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                ),
                finish_reason=choice.finish_reason or "stop",
            )

        except APITimeoutError as e:
            logger.error(
                "❌ OpenAI调用超时 - 模型: %s, base_url: %s, timeout: %s, 错误: %s",
                self.config.model,
                self.config.base_url,
                self.config.timeout,
                e,
            )
            raise LLMTimeoutError(f"OpenAI调用超时: {e}") from e
        except RateLimitError as e:
            logger.error(
                "❌ OpenAI速率限制 - 模型: %s, 错误: %s",
                self.config.model,
                e,
            )
            raise LLMRateLimitError(f"OpenAI速率限制: {e}") from e
        except (APIError, APIConnectionError) as e:
            logger.error(
                "❌ OpenAI调用失败 - 模型: %s, base_url: %s, 错误: %s",
                self.config.model,
                self.config.base_url,
                e,
                exc_info=True,
            )
            raise LLMError(f"OpenAI调用失败: {e}") from e
        except Exception as e:
            logger.error("未知错误: %s", e, exc_info=True)
            raise LLMError(f"OpenAI调用失败: {e}") from e

    async def chat_stream(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        include_reasoning: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[str, Optional[str]]]:
        """
        OpenAI流式聊天补全

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大输出token数
            include_reasoning: 是否返回推理内容（reasoning_content）
            **kwargs: 其他参数

        Yields:
            元组 (content, reasoning) - content为内容片段，reasoning为推理片段（如果有）

        Raises:
            LLMError: 调用失败
        """
        try:
            # 记录使用的模型信息（添加max_tokens）
            settings = get_settings()
            logger.info(
                "🤖 调用流式LLM - 模型: %s, base_url: %s, temperature: %.2f, max_tokens: %s, timeout: %s, enable_think: %s",
                self.config.model,
                self.config.base_url,
                temperature or self.config.temperature,
                max_tokens or self.config.max_tokens or "未设置",
                self.config.timeout,
                settings.llm_enable_think
            )

            # 打印输入消息（调试用）
            for i, msg in enumerate(messages):
                content_preview = msg.content[:5000] if len(msg.content) > 5000 else msg.content
                logger.info(f"📝 消息[{i}] role={msg.role.value}: {content_preview}")

            # 准备消息
            api_messages = self._prepare_messages(messages)

            # 调用流式API（使用 cast 显式类型转换）
            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=cast(Iterable[ChatCompletionMessageParam], api_messages),
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                stream=True,
                extra_body = {
                "chat_template_kwargs": {"enable_thinking": settings.llm_enable_think},
                },
                **kwargs,
            )

            # 逐个生成内容片段
            async for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    content = delta.content if delta.content else None
                    reasoning = None

                    # 如果需要推理内容，尝试获取reasoning_content
                    if include_reasoning:
                        reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None) 

                    # 只在有内容或推理时yield
                    if content or reasoning:
                        yield (content or "", reasoning)

        except APITimeoutError as e:
            logger.error("OpenAI流式调用超时: %s", e)
            raise LLMTimeoutError(f"OpenAI流式调用超时: {e}") from e
        except (APIError, APIConnectionError) as e:
            logger.error("OpenAI流式调用失败: %s", e, exc_info=True)
            raise LLMError(f"OpenAI流式调用失败: {e}") from e
        except Exception as e:
            logger.error("未知错误: %s", e, exc_info=True)
            raise LLMError(f"OpenAI流式调用失败: {e}") from e

    async def close(self) -> None:
        """关闭OpenAI客户端，释放HTTP连接"""
        try:
            await self.client.close()
            logger.debug("OpenAI客户端已关闭")
        except Exception as e:
            logger.warning(f"关闭OpenAI客户端时出错: {e}")


async def create_openai_client(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    api_key: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
) -> OpenAIClient:
    """
    创建OpenAI客户端（从环境变量读取默认值）

    Args:
        api_key: API密钥
        model: 模型名称（可选，默认从环境变量读取）
        base_url: 基础URL（可选，默认从环境变量读取）
        temperature: 温度参数（可选，默认从环境变量读取）
        max_tokens: 最大输出token数（可选，默认从环境变量读取）
        timeout: 超时时间（秒）（可选，默认从环境变量读取）
        max_retries: 最大重试次数（可选，默认从环境变量读取）

    Returns:
        OpenAI客户端实例
    """

    settings = get_settings()

    config = ModelConfig(
        provider=LLMProvider.OPENAI,
        model=model or settings.llm_model,
        api_key=api_key,
        base_url=base_url or settings.llm_base_url,
        temperature=temperature or settings.llm_temperature,
        max_tokens=max_tokens or settings.llm_max_tokens,
        top_p=settings.llm_top_p,
        frequency_penalty=settings.llm_frequency_penalty,
        presence_penalty=settings.llm_presence_penalty,
        timeout=timeout or settings.llm_timeout,
        max_retries=max_retries or settings.llm_max_retries,
    )

    return OpenAIClient(config)
