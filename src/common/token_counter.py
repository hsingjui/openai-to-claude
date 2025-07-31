import json
from typing import Any

import tiktoken


class TokenCounter:
    """Token计数器，基于Node.js实现完整功能复现"""

    def __init__(self):
        self.encoder = tiktoken.get_encoding("o200k_base")

    def _extract_text_content(self, obj, field_name: str) -> str:
        """统一提取文本内容的方法，简化代码重复"""
        if hasattr(obj, field_name):
            return str(getattr(obj, field_name, ""))
        elif isinstance(obj, dict):
            return str(obj.get(field_name, ""))
        return ""

    def _process_content_part(self, content_part) -> list[str]:
        """处理消息内容部分，返回文本列表"""
        texts = []
        part_type = (
            content_part.type
            if hasattr(content_part, "type")
            else content_part.get("type") if isinstance(content_part, dict) else None
        )

        if part_type == "text":
            text = self._extract_text_content(content_part, "text")
            if text:
                texts.append(text)
        elif part_type == "tool_use":
            input_data = (
                getattr(content_part, "input", {})
                if hasattr(content_part, "input")
                else content_part.get("input", {})
            )
            if input_data:
                texts.append(json.dumps(input_data, ensure_ascii=False))

        return texts

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
        # 收集所有文本内容到单个列表，遵循KISS原则
        text_parts = []

        # 处理消息内容
        if messages:
            for message in messages:
                # 获取消息内容
                content = (
                    message.content
                    if hasattr(message, "content")
                    else message.get("content", "") if isinstance(message, dict) else ""
                )

                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    for content_part in content:
                        text_parts.extend(self._process_content_part(content_part))

        # 处理系统提示
        if system:
            if isinstance(system, str):
                text_parts.append(system)
            elif isinstance(system, list):
                for item in system:
                    item_type = (
                        item.type
                        if hasattr(item, "type")
                        else item.get("type") if isinstance(item, dict) else None
                    )
                    if item_type == "text":
                        text_content = self._extract_text_content(item, "text")
                        if text_content:
                            text_parts.append(text_content)

        # 处理工具定义
        if tools:
            for tool in tools:
                # 统一获取name和description
                name = self._extract_text_content(tool, "name")
                description = self._extract_text_content(tool, "description")

                if name:
                    text_parts.append(name)
                if description:
                    text_parts.append(description)

                # 处理schema
                schema = (
                    getattr(tool, "input_schema", None)
                    if hasattr(tool, "input_schema")
                    else tool.get("input_schema") if isinstance(tool, dict) else None
                )
                if schema:
                    text_parts.append(json.dumps(schema, ensure_ascii=False))

        # 一次性拼接所有文本并计算token（KISS原则）
        combined_text = "".join(text_parts)
        return len(self.encoder.encode(combined_text))

    def count_response_tokens(self, content_blocks: list) -> int:
        """计算响应内容的token数量

        Args:
            content_blocks: Anthropic格式的内容块列表

        Returns:
            int: 响应内容的token数量
        """
        # 收集所有文本内容到单个列表，遵循KISS原则
        text_parts = []

        # 处理内容块
        if content_blocks:
            for block in content_blocks:
                # 处理文本内容
                if hasattr(block, "text") and block.text:
                    text_parts.append(str(block.text))
                elif isinstance(block, dict) and block.get("text"):
                    text_parts.append(str(block["text"]))

                # 处理思考内容
                if hasattr(block, "thinking") and block.thinking:
                    text_parts.append(str(block.thinking))
                elif isinstance(block, dict) and block.get("thinking"):
                    text_parts.append(str(block["thinking"]))

                # 处理工具调用内容
                if hasattr(block, "input") and block.input:
                    text_parts.append(json.dumps(block.input, ensure_ascii=False))
                elif isinstance(block, dict) and block.get("input"):
                    text_parts.append(json.dumps(block["input"], ensure_ascii=False))

                # 处理工具名称
                if hasattr(block, "name") and block.name:
                    text_parts.append(str(block.name))
                elif isinstance(block, dict) and block.get("name"):
                    text_parts.append(str(block["name"]))

        # 一次性拼接所有文本并计算token（KISS原则）
        combined_text = "".join(text_parts)
        return len(self.encoder.encode(combined_text))


# 全局实例
token_counter = TokenCounter()
