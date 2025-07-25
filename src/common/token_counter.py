import json
from typing import Any

import tiktoken


class TokenCounter:
    """Token计数器，基于Node.js实现完整功能复现"""

    def __init__(self):
        self.encoder = tiktoken.get_encoding("cl100k_base")

    async def count_tokens(
        self,
        messages: list[Any] = None,
        system: Any = None,
        tools: list[Any] = None,
    ) -> int:
        """计算完整请求的token总数

        Args:
            messages: 消息列表
            system: 系统提示
            tools: 工具定义

        Returns:
            int: 总计token数量
        """
        total_tokens = 0

        # 计算消息token
        if messages:
            for message in messages:
                content = (
                    message.content
                    if hasattr(message, "content")
                    else message.get("content", "")
                )

                if isinstance(content, str):
                    total_tokens += len(self.encoder.encode(content))
                elif isinstance(content, list):
                    for content_part in content:
                        if hasattr(content_part, "type"):
                            # 处理Pydantic模型
                            part_type = content_part.type
                            if part_type == "text":
                                text = getattr(content_part, "text", "")
                                total_tokens += len(self.encoder.encode(str(text)))
                            elif part_type == "tool_use":
                                input_data = getattr(content_part, "input", {})
                                total_tokens += len(
                                    self.encoder.encode(
                                        json.dumps(input_data, ensure_ascii=False)
                                    )
                                )
                        elif isinstance(content_part, dict):
                            # 处理字典
                            part_type = content_part.get("type")
                            if part_type == "text":
                                text = content_part.get("text", "")
                                total_tokens += len(self.encoder.encode(str(text)))
                            elif part_type == "tool_use":
                                input_data = content_part.get("input", {})
                                total_tokens += len(
                                    self.encoder.encode(
                                        json.dumps(input_data, ensure_ascii=False)
                                    )
                                )

        # 计算系统提示token
        if system:
            if isinstance(system, str):
                total_tokens += len(self.encoder.encode(system))
            elif isinstance(system, list):
                for item in system:
                    if hasattr(item, "type") and item.type == "text":
                        # 处理Pydantic模型
                        text_content = getattr(item, "text", "")
                        total_tokens += len(self.encoder.encode(str(text_content)))
                    elif isinstance(item, dict) and item.get("type") == "text":
                        # 处理字典
                        text_content = item.get("text")
                        total_tokens += len(self.encoder.encode(str(text_content)))

        # 计算工具定义token
        if tools:
            for tool in tools:
                if hasattr(tool, "name") and hasattr(tool, "description"):
                    # 处理Pydantic模型
                    name = str(getattr(tool, "name", ""))
                    description = str(getattr(tool, "description", ""))
                elif isinstance(tool, dict):
                    # 处理字典
                    name = str(tool.get("name", ""))
                    description = str(tool.get("description", ""))
                else:
                    name = str(getattr(tool, "name", ""))
                    description = str(getattr(tool, "description", ""))

                total_tokens += len(self.encoder.encode(name + description))

                schema = (
                    getattr(tool, "input_schema", None)
                    if hasattr(tool, "input_schema")
                    else tool.get("input_schema")
                )
                if schema:
                    total_tokens += len(
                        self.encoder.encode(json.dumps(schema, ensure_ascii=False))
                    )

        return total_tokens

    async def count_simple_tokens(self, text: str) -> int:
        """简单字符串token计数

        Args:
            text: 要计数的文本

        Returns:
            int: text的token数量
        """
        if not text:
            return 0
        return len(self.encoder.encode(text))


# 全局实例
token_counter = TokenCounter()
