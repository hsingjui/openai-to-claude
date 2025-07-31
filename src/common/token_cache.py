"""
简单的请求token缓存模块

基于KISS原则，使用全局字典实现请求ID与token数量的临时缓存。
主要用于在OpenAI响应缺失usage信息时提供fallback。
"""

from typing import Dict, Optional

# 全局缓存字典 - 遵循KISS原则
_cache: Dict[str, int] = {}


def cache_tokens(request_id: str, tokens: int) -> None:
    """
    缓存请求的token数量

    Args:
        request_id: 请求ID
        tokens: token数量
    """
    if request_id and tokens > 0:
        _cache[request_id] = tokens


def get_cached_tokens(request_id: str, delete=False) -> Optional[int]:
    """
    获取缓存的token数量并清理缓存

    Args:
        request_id: 请求ID

    Returns:
        缓存的token数量，如果不存在则返回None

    Note:
        使用pop()方法，获取后自动删除缓存，防止内存泄漏
    """
    if not request_id:
        return None
    if delete:
        return _cache.pop(request_id, None)
    else:
        return _cache.get(request_id, None)


def get_cache_size() -> int:
    """
    获取当前缓存大小，用于调试和监控

    Returns:
        缓存中的条目数量
    """
    return len(_cache)


def clear_cache() -> None:
    """
    清空所有缓存，用于测试或重置
    """
    _cache.clear()
