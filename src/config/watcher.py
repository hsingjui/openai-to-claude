"""配置文件监听和热重载模块

监听配置文件的变化，当配置文件被修改时自动重新加载配置。
使用 watchdog 库监听文件系统事件。
"""

import asyncio
import json
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from loguru import logger
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class ConfigFileHandler(FileSystemEventHandler):
    """配置文件变化事件处理器"""

    def __init__(self, config_path: Path, callback: Callable[[], None]):
        """
        初始化配置文件处理器

        Args:
            config_path: 要监听的配置文件路径
            callback: 配置文件变化时的回调函数
        """
        self.config_path = config_path.resolve()
        self.callback = callback
        self._last_modified = 0

    def on_modified(self, event) -> None:
        """处理文件修改事件"""
        if event.is_directory:
            return

        # 检查是否是我们监听的配置文件
        event_path = Path(event.src_path).resolve()
        if event_path != self.config_path:
            return

        # 防止重复触发
        try:
            current_modified = event_path.stat().st_mtime
            if current_modified == self._last_modified:
                return
            self._last_modified = current_modified
        except OSError:
            return

        logger.info(f"配置文件已修改: {self.config_path}")

        # 延迟一点执行，确保文件写入完成
        threading.Timer(0.1, self._execute_callback).start()

    def _execute_callback(self) -> None:
        """执行回调函数"""
        try:
            self.callback()
        except Exception as e:
            logger.error(f"配置重载回调执行失败: {e}")


class ConfigWatcher:
    """配置文件监听器

    监听指定的配置文件，当文件发生变化时触发重新加载。
    """

    def __init__(self, config_path: str | None = None):
        """
        初始化配置监听器

        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        if config_path is None:
            config_path = os.getenv("CONFIG_PATH", "config/settings.json")

        self.config_path = Path(config_path).resolve()
        self.observer: Observer | None = None
        self.handler: ConfigFileHandler | None = None
        self._reload_callbacks: list[Callable[[], None]] = []
        self._async_reload_callbacks: list[Callable[[], Any]] = []
        self._executor: ThreadPoolExecutor | None = None

    def add_reload_callback(self, callback: Callable[[], Any]) -> None:
        """
        添加异步配置重载回调函数

        Args:
            callback: 异步回调函数
        """
        self._async_reload_callbacks.append(callback)

    async def start_watching(self) -> None:
        """开始监听配置文件变化"""
        if self.observer is not None:
            logger.warning("配置监听器已在运行")
            return

        if not self.config_path.exists():
            logger.warning(f"配置文件不存在，跳过监listen: {self.config_path}")
            return

        # 创建线程池执行器用于异步回调
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="config-watcher"
            )

        # 创建事件处理器
        self.handler = ConfigFileHandler(self.config_path, self._on_config_changed)

        # 创建观察者并开始监听
        self.observer = Observer()
        watch_dir = self.config_path.parent
        self.observer.schedule(self.handler, str(watch_dir), recursive=False)
        self.observer.start()

        logger.info(f"开始监听配置文件: {self.config_path}")

    def stop_watching(self) -> None:
        """停止监听配置文件变化"""
        if self.observer is None:
            return

        logger.info("停止配置文件监听")
        self.observer.stop()
        self.observer.join()
        self.observer = None
        self.handler = None

        # 关闭线程池
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _on_config_changed(self) -> None:
        """配置文件变化时的处理逻辑"""
        logger.info("检测到配置文件变化，开始重新加载...")

        # 在线程池中执行异步验证和回调
        if self._executor is not None:
            self._executor.submit(self._handle_config_change)
        else:
            logger.error("线程池执行器未初始化，跳过配置重载")

    async def _process_config_change(self) -> None:
        """处理配置变化的异步逻辑"""
        # 验证配置文件格式
        if not await self._validate_config_file():
            logger.error("配置文件格式无效，跳过重载")
            return

        # 执行同步回调
        for callback in self._reload_callbacks:
            try:
                callback()
                logger.debug(f"同步配置重载回调执行成功: {callback.__name__}")
            except Exception as e:
                logger.error(f"同步配置重载回调执行失败 {callback.__name__}: {e}")

        # 执行异步回调
        await self._execute_async_callbacks()

        logger.info("配置重载完成")

    def _handle_config_change(self) -> None:
        """在线程池中处理配置变化"""
        try:
            # 在新线程中创建事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # 运行异步验证和回调
                loop.run_until_complete(self._process_config_change())
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"配置变化处理失败: {e}")

    async def _execute_async_callbacks(self) -> None:
        """执行异步回调函数"""
        for callback in self._async_reload_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"异步配置重载回调执行失败 {callback.__name__}: {e}")

    async def _validate_config_file(self) -> bool:
        """验证配置文件格式是否正确"""
        try:
            import aiofiles

            async with aiofiles.open(self.config_path, encoding="utf-8") as f:
                content = await f.read()
                json.loads(content)
            return True
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"配置文件验证失败: {e}")
            return False

    def __enter__(self):
        """上下文管理器入口"""
        self.start_watching()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        # 忽略异常信息，直接停止监听
        self.stop_watching()
