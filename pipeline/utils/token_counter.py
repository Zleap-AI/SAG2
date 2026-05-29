"""
Token 消耗统计工具

用于统计所有 LLM 调用的 token 消耗量，支持：
- 自动记录每次 LLM 调用的 input/output/total tokens
- 按场景分类统计（NER、Rerank、Query Rewrite 等）
- 保存日志到 output 文件夹
- 提供装饰器和上下文管理器两种使用方式

使用示例：
    # 方式1：装饰器
    @track_tokens(scenario="ner")
    async def extract_entities(query: str):
        response = await llm_client.chat_with_schema(...)
        return response

    # 方式2：上下文管理器
    async with TokenTracker(scenario="rerank") as tracker:
        response = await llm_client.chat_with_schema(...)
        tracker.record(response)

    # 方式3：手动记录
    tracker = TokenCounter()
    tracker.add_record(
        scenario="ner",
        model="[REDACTED]",
        input_tokens=100,
        output_tokens=50,
        total_tokens=150
    )
    tracker.save_to_file("output/tokens.json")
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from functools import wraps

from pipeline.utils import get_logger

logger = get_logger("utils.token_counter")


class TokenCounter:
    """Token 消耗统计器"""

    def __init__(self):
        self.records: List[Dict[str, Any]] = []
        self.summary: Dict[str, Dict[str, int]] = {}

    def add_record(
        self,
        scenario: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        添加一条 token 消耗记录

        Args:
            scenario: 场景名称（如 "ner", "rerank", "query_rewrite"）
            model: 模型名称（如 "[REDACTED]"）
            input_tokens: 输入 token 数
            output_tokens: 输出 token 数
            total_tokens: 总 token 数
            metadata: 额外元数据（如 query, response 等）
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "scenario": scenario,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "metadata": metadata or {},
        }
        self.records.append(record)

        # 更新汇总统计
        if scenario not in self.summary:
            self.summary[scenario] = {
                "count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }
        self.summary[scenario]["count"] += 1
        self.summary[scenario]["input_tokens"] += input_tokens
        self.summary[scenario]["output_tokens"] += output_tokens
        self.summary[scenario]["total_tokens"] += total_tokens

        logger.debug(
            f"[Token记录] {scenario} | model={model} | "
            f"input={input_tokens}, output={output_tokens}, total={total_tokens}"
        )

    def get_summary(self) -> Dict[str, Any]:
        """
        获取汇总统计

        Returns:
            {
                "total": {"input_tokens": int, "output_tokens": int, "total_tokens": int},
                "by_scenario": {scenario: {...}, ...}
            }
        """
        total_input = sum(s["input_tokens"] for s in self.summary.values())
        total_output = sum(s["output_tokens"] for s in self.summary.values())
        total_tokens = sum(s["total_tokens"] for s in self.summary.values())

        return {
            "total": {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "total_tokens": total_tokens,
            },
            "by_scenario": self.summary,
        }

    def save_to_file(self, output_path: str):
        """
        保存 token 日志到文件

        Args:
            output_path: 输出文件路径（JSON 格式）
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "summary": self.get_summary(),
            "records": self.records,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"✓ Token日志已保存: {output_file}")

    def reset(self):
        """重置统计数据"""
        self.records.clear()
        self.summary.clear()


# 全局单例
_global_counter = TokenCounter()


def get_global_counter() -> TokenCounter:
    """获取全局 token 计数器"""
    return _global_counter


def reset_global_counter():
    """重置全局 token 计数器"""
    _global_counter.reset()


class TokenTracker:
    """
    Token 追踪上下文管理器

    使用示例：
        async with TokenTracker(scenario="ner") as tracker:
            response = await llm_client.chat_with_schema(...)
            tracker.record(response)
    """

    def __init__(
        self,
        scenario: str,
        counter: Optional[TokenCounter] = None,
    ):
        self.scenario = scenario
        self.counter = counter or _global_counter
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.warning(f"[Token追踪] {self.scenario} 执行出错: {exc_val}")
        return False

    async def __aenter__(self):
        self.start_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.warning(f"[Token追踪] {self.scenario} 执行出错: {exc_val}")
        return False

    def record(
        self,
        response: Dict[str, Any],
        model: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        记录 LLM 响应的 token 消耗

        Args:
            response: LLM 响应（需包含 usage 字段）
            model: 模型名称
            metadata: 额外元数据
        """
        usage = response.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens

        elapsed = time.perf_counter() - self.start_time if self.start_time else 0
        meta = metadata or {}
        meta["elapsed_seconds"] = round(elapsed, 3)

        self.counter.add_record(
            scenario=self.scenario,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            metadata=meta,
        )


def track_tokens(scenario: str, counter: Optional[TokenCounter] = None):
    """
    Token 追踪装饰器

    使用示例：
        @track_tokens(scenario="ner")
        async def extract_entities(query: str):
            response = await llm_client.chat_with_schema(...)
            return response
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with TokenTracker(scenario=scenario, counter=counter) as tracker:
                result = await func(*args, **kwargs)
                # 假设返回值是 LLM response 或包含 response 的字典
                if isinstance(result, dict) and "usage" in result:
                    tracker.record(result)
                return result
        return wrapper
    return decorator


__all__ = [
    "TokenCounter",
    "TokenTracker",
    "track_tokens",
    "get_global_counter",
    "reset_global_counter",
]
