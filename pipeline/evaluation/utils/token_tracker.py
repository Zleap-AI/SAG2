"""
LLM Token 追踪器模块

提供 LLMTokenTracker 类和 enable_llm_tracking() 函数，
用于并发安全地统计 LLM 调用的 token 消耗，并按调用阶段分组记录。
"""

import asyncio
from typing import Dict, Any


class LLMTokenTracker:
    """追踪 LLM 调用的 token 消耗（支持并发安全）"""

    def __init__(self):
        self.total = {"prompt": 0, "completion": 0, "total": 0}
        self.stages = {}
        self._lock = asyncio.Lock()  # 并发保护锁

    async def record(self, stage: str, usage):
        """记录 token（并发安全）

        Args:
            stage: 调用阶段标识，如 "SEARCH"、"LOAD"、"EXTRACT"、"QA"
            usage: LLM 返回的 usage 对象，需含 prompt_tokens / completion_tokens / total_tokens
        """
        if not usage:
            return
        prompt = getattr(usage, 'prompt_tokens', 0)
        completion = getattr(usage, 'completion_tokens', 0)
        total = getattr(usage, 'total_tokens', 0)

        async with self._lock:
            self.total["prompt"] += prompt
            self.total["completion"] += completion
            self.total["total"] += total

            if stage not in self.stages:
                self.stages[stage] = {"calls": 0,
                                      "prompt": 0, "completion": 0, "total": 0}

            self.stages[stage]["calls"] += 1
            self.stages[stage]["prompt"] += prompt
            self.stages[stage]["completion"] += completion
            self.stages[stage]["total"] += total

    def get_summary(self) -> Dict[str, Any]:
        """获取 token 消耗统计汇总

        Returns:
            包含 total_prompt、total_completion、total_tokens、total_calls 和按阶段明细的字典
        """
        return {
            "total_prompt": self.total["prompt"],
            "total_completion": self.total["completion"],
            "total_tokens": self.total["total"],
            "total_calls": sum(s["calls"] for s in self.stages.values()),
            "stages": self.stages
        }


def enable_llm_tracking(token_tracker: LLMTokenTracker):
    """启用 LLM 调用追踪（兼容所有阶段：LOAD, EXTRACT, SEARCH, QA）

    通过 monkey-patch OpenAIClient.chat，自动拦截所有 LLM 调用并记录 token 消耗。
    阶段识别通过调用栈分析实现：
      - QA    ← 调用栈中有 show_retrieval_info
      - LOAD  ← 调用栈中有 DocumentLoader / DocumentProcessor
      - EXTRACT ← 调用栈中有 EventProcessor / EventExtractor
      - SEARCH  ← 调用栈中有 RecallSearcher / SAGSearcher

    Args:
        token_tracker: LLMTokenTracker 实例，用于接收记录结果
    """
    from pipeline.core.ai import llm
    original_chat = llm.OpenAIClient.chat

    async def tracked_chat(self, messages, **kwargs):
        result = await original_chat(self, messages, **kwargs)
        import inspect
        frame = inspect.currentframe()
        stage = "UNKNOWN"

        try:
            # 向上查找调用栈，最多查找 30 层
            for _ in range(30):
                if not frame:
                    break
                frame = frame.f_back
                if not frame:
                    break

                # 检查函数名（用于 QA 阶段）
                func_name = frame.f_code.co_name if frame.f_code else ""
                if func_name == "show_retrieval_info":
                    stage = "QA"
                    break

                # 检查类名（用于 LOAD、EXTRACT、SEARCH 阶段）
                if 'self' in frame.f_locals:
                    obj = frame.f_locals['self']
                    class_name = obj.__class__.__name__

                    if 'DocumentLoader' in class_name or 'DocumentProcessor' in class_name:
                        stage = "LOAD"
                        break
                    elif 'EventProcessor' in class_name or 'EventExtractor' in class_name:
                        stage = "EXTRACT"
                        break
                    elif 'RecallSearcher' in class_name or 'SAGSearcher' in class_name:
                        stage = "SEARCH"
                        break
        except Exception:
            pass
        finally:
            del frame

        if hasattr(result, 'usage') and result.usage:
            await token_tracker.record(stage, result.usage)

        return result

    llm.OpenAIClient.chat = tracked_chat


__all__ = [
    'LLMTokenTracker',
    'enable_llm_tracking',
]
