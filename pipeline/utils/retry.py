"""
重试工具模块

提供异常分类和重试逻辑
"""

import asyncio
from typing import Callable, Optional, Type, Tuple
from sqlalchemy.exc import OperationalError, IntegrityError

from pipeline.exceptions import RetryableError, NetworkError, ResourceBusyError, ServiceUnavailableError


def is_retryable_db_error(error: Exception) -> bool:
    """
    判断数据库错误是否可重试

    Args:
        error: 异常对象

    Returns:
        True 表示可重试，False 表示不可重试
    """
    if isinstance(error, OperationalError):
        error_str = str(error)
        # 死锁和锁等待超时可重试
        if "1213" in error_str or "Deadlock" in error_str:
            return True
        if "1205" in error_str or "Lock wait timeout" in error_str:
            return True
        # 连接丢失可重试
        if "2013" in error_str or "Lost connection" in error_str:
            return True
        # 连接超时可重试
        if "2003" in error_str or "Can't connect" in error_str:
            return True
        # 其他 OperationalError 不可重试（如语法错误）
        return False

    # IntegrityError（唯一键冲突）不可重试
    if isinstance(error, IntegrityError):
        return False

    return False


def is_retryable_network_error(error: Exception) -> bool:
    """
    判断网络错误是否可重试

    Args:
        error: 异常对象

    Returns:
        True 表示可重试，False 表示不可重试
    """
    error_str = str(error).lower()

    # 连接超时、读取超时可重试
    if "timeout" in error_str or "timed out" in error_str:
        return True

    # 连接被拒绝、连接重置可重试
    if "connection refused" in error_str or "connection reset" in error_str:
        return True

    # 临时性网络错误可重试
    if "temporary failure" in error_str or "network unreachable" in error_str:
        return True

    return False


def is_retryable_error(error: Exception) -> bool:
    """
    判断异常是否可重试（统一入口）

    Args:
        error: 异常对象

    Returns:
        True 表示可重试，False 表示不可重试
    """
    # 自定义可重试异常
    if isinstance(error, RetryableError):
        return True

    # 数据库错误
    if is_retryable_db_error(error):
        return True

    # 网络错误
    if is_retryable_network_error(error):
        return True

    return False


async def retry_async(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
) -> any:
    """
    异步重试装饰器

    Args:
        func: 异步函数
        max_retries: 最大重试次数
        base_delay: 基础延迟（秒）
        max_delay: 最大延迟（秒）
        exponential_base: 指数退避基数
        retryable_exceptions: 可重试的异常类型（如果为None，使用is_retryable_error判断）

    Returns:
        函数执行结果

    Raises:
        最后一次重试的异常
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            last_exception = e

            # 判断是否可重试
            if retryable_exceptions:
                is_retryable = isinstance(e, retryable_exceptions)
            else:
                is_retryable = is_retryable_error(e)

            # 不可重试或已达最大重试次数
            if not is_retryable or attempt >= max_retries - 1:
                raise

            # 计算延迟时间（指数退避）
            delay = min(base_delay * (exponential_base ** attempt), max_delay)
            await asyncio.sleep(delay)

    # 理论上不会到这里，但为了类型检查
    if last_exception:
        raise last_exception
