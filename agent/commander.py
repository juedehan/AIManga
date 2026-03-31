import importlib.util
import sys
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Literal, Optional
from contextlib import AsyncExitStack

_BOOTSTRAP_PATH = Path(__file__).resolve().parents[1] / "bootstrap.py"
_BOOTSTRAP_SPEC = importlib.util.spec_from_file_location("aimanga_bootstrap", _BOOTSTRAP_PATH)
if _BOOTSTRAP_SPEC is None or _BOOTSTRAP_SPEC.loader is None:
    raise ImportError(f"无法加载 bootstrap: {_BOOTSTRAP_PATH}")
_bootstrap = importlib.util.module_from_spec(_BOOTSTRAP_SPEC)
_BOOTSTRAP_SPEC.loader.exec_module(_bootstrap)
_bootstrap.ensure_project_root()

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from model.factory import chat_model
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
from utils.prompt_loader import load_system_prompts
from utils.logger_handler import logger
from middleware.middleware import (
    log_before_agent,
    log_after_agent,
    log_before_model,
    log_after_model,
)

from tools.commander_tools import CommanderToolbox

from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession
from mcp.types import CallToolResult, TextContent


class CommanderInterruptedError(Exception):
    """当前轮次被外部中断。"""


@dataclass
class CommanderRequestRuntime:
    session_id: str
    user_query: str
    pending_events: list[dict[str, Any]] = field(default_factory=list)
    cancel_checker: Callable[[], bool] | None = None


class Commander:
    def __init__(
        self,
        mode: Literal["mcp", "langchain"] | None = None,
        mcp_server_path: str | None = None,
        mcp_server_command: str | None = None,
    ):
        commander_conf = agent_conf.get("commander", {})
        self.mode = mode or commander_conf.get("default_mode", "langchain")
        server_relpath = mcp_server_path or commander_conf.get("mcp_server_path", "mcp/mcp_server.py")
        self.mcp_server_path = get_abs_path(server_relpath)
        configured_command = commander_conf.get("mcp_server_command")
        self.mcp_server_command = mcp_server_command or configured_command or sys.executable

        self.agent = None
        self.langchain_tools = []
        self.toolbox = CommanderToolbox()
        self.exit_stack = AsyncExitStack()
        self.mcp_session: Optional[ClientSession] = None
        self.request_runtime: ContextVar[CommanderRequestRuntime | None] = ContextVar(
            "commander_request_runtime",
            default=None,
        )

    def _require_request_runtime(self) -> CommanderRequestRuntime:
        runtime = self.request_runtime.get()
        if runtime is None:
            raise RuntimeError("当前请求缺少 Commander 会话上下文")
        return runtime

    def _check_cancelled(self) -> None:
        runtime = self._require_request_runtime()
        if runtime.cancel_checker and runtime.cancel_checker():
            raise CommanderInterruptedError("当前轮次已被中断")

    def _queue_runtime_event(self, event: dict[str, Any]) -> None:
        runtime = self._require_request_runtime()
        runtime.pending_events.append(event)

    def _queue_process_event(self, stage: str, message: str) -> None:
        self._queue_runtime_event(
            {
                "event": "process",
                "stage": stage,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    def _drain_runtime_events(self) -> list[dict[str, Any]]:
        runtime = self._require_request_runtime()
        queued = list(runtime.pending_events)
        runtime.pending_events.clear()
        return queued

    def _extract_text_from_call_result(self, result: CallToolResult) -> str:
        text_parts: list[str] = []
        for item in result.content:
            if isinstance(item, TextContent):
                text_parts.append(item.text)
                continue

            item_type = getattr(item, "type", None)
            if item_type == "text":
                text = getattr(item, "text", "")
                if text:
                    text_parts.append(text)

        if text_parts:
            return "\n".join(part.strip() for part in text_parts if part and part.strip()).strip()

        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return str(structured)
        return ""

    def _is_image_path(self, value: str) -> bool:
        if not value:
            return False
        candidate = Path(value.strip())
        return candidate.is_absolute() and candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and candidate.exists()

    async def _call_mcp_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if self.mcp_session is None:
            raise RuntimeError("MCP Session 尚未初始化")

        tool_start_messages = {
            "get_features": "正在调用 get_features，提取角色特征。",
            "opt_prompts": "正在调用 opt_prompts，优化绘画提示词。",
            "get_portrait": "正在调用 get_portrait，开始出图。",
            "adjust_existing_portrait": "正在调用 adjust_existing_portrait，调整已有肖像。",
        }
        tool_done_messages = {
            "get_features": "get_features 已完成。",
            "opt_prompts": "opt_prompts 已完成。",
            "get_portrait": "get_portrait 已完成。",
            "adjust_existing_portrait": "adjust_existing_portrait 已完成。",
        }

        self._check_cancelled()
        self._queue_process_event("tool_call", tool_start_messages.get(tool_name, f"正在调用 {tool_name}。"))
        result = await self.mcp_session.call_tool(tool_name, arguments=arguments)
        self._check_cancelled()
        text = self._extract_text_from_call_result(result)

        if getattr(result, "isError", False):
            raise RuntimeError(text or f"MCP tool 调用失败: {tool_name}")

        self._queue_process_event("tool_done", tool_done_messages.get(tool_name, f"{tool_name} 已完成。"))

        if tool_name in {"get_portrait", "adjust_existing_portrait"} and self._is_image_path(text):
            self._queue_runtime_event({"event": "image", "image_path": text.strip()})
            self._queue_process_event("image_ready", "已收到生成图片。")

        return text

    def _coerce_message_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
                else:
                    text_parts.append(str(item))
            return "".join(text_parts)
        if content is None:
            return ""
        return str(content)

    def _extract_delta_text(self, previous: str, current: str) -> str:
        if not current or current == previous:
            return ""
        if previous and current.startswith(previous):
            return current[len(previous) :]
        return current

    def _build_mcp_proxy_tools(self):
        @tool("get_features", description="首次生成角色肖像时，从小说知识库提取人物在指定场景下的特征描述。")
        async def get_features(character_name: str, scene: str) -> str:
            """首次生成角色肖像时，从小说知识库提取人物在指定场景下的特征描述。"""
            runtime = self._require_request_runtime()
            return await self._call_mcp_tool(
                "get_features",
                {
                    "session_id": runtime.session_id,
                    "user_query": runtime.user_query,
                    "character_name": character_name,
                    "scene": scene,
                },
            )

        @tool("opt_prompts", description="把文学性人物描述转换为可直接用于文生图的最终绘画提示词。")
        async def opt_prompts(literary_description: str, gender: Optional[Literal["男", "女"]] = "未提及") -> str:
            """把文学性人物描述转换为可直接用于文生图的最终绘画提示词。"""
            runtime = self._require_request_runtime()
            return await self._call_mcp_tool(
                "opt_prompts",
                {
                    "session_id": runtime.session_id,
                    "user_query": runtime.user_query,
                    "literary_description": literary_description,
                    "gender": gender,
                },
            )

        @tool("get_portrait", description="根据最终提示词生成人物肖像图，并在当前会话下保存角色记忆。")
        async def get_portrait(
            optimized_prompt: str,
            character_name: str,
            character_name_pinyin: str,
            scene: str = "未提及",
            gender: Optional[Literal["男", "女"]] = "未提及",
        ) -> str:
            """根据最终提示词生成人物肖像图，并在当前会话下保存角色记忆。"""
            runtime = self._require_request_runtime()
            return await self._call_mcp_tool(
                "get_portrait",
                {
                    "session_id": runtime.session_id,
                    "user_query": runtime.user_query,
                    "optimized_prompt": optimized_prompt,
                    "character_name": character_name,
                    "character_name_pinyin": character_name_pinyin,
                    "scene": scene,
                    "gender": gender,
                },
            )

        @tool("adjust_existing_portrait", description="对当前会话已生成的角色肖像做局部调整，并复用角色记忆。")
        async def adjust_existing_portrait(
            modification_request: str,
            character_name: Optional[str] = None,
            character_name_pinyin: Optional[str] = None,
        ) -> str:
            """对当前会话已生成的角色肖像做局部调整，并复用角色记忆。"""
            runtime = self._require_request_runtime()
            payload = {
                "session_id": runtime.session_id,
                "user_query": runtime.user_query,
                "modification_request": modification_request,
            }
            if character_name is not None:
                payload["character_name"] = character_name
            if character_name_pinyin is not None:
                payload["character_name_pinyin"] = character_name_pinyin
            return await self._call_mcp_tool("adjust_existing_portrait", payload)

        return [get_features, opt_prompts, get_portrait, adjust_existing_portrait]

    async def setup(self):
        if self.mode == "langchain":
            logger.info("\033[33m正在以本地 LangChain Tool 模式启动......\033[0m")

            self.langchain_tools = [
                tool(self.toolbox.get_features),
                tool(self.toolbox.opt_prompts),
                tool(self.toolbox.get_portrait),
                tool(self.toolbox.adjust_existing_portrait),
            ]

            tool_names = [t.name for t in self.langchain_tools]
            logger.info(f"\033[33m本地工具装载成功: {tool_names}\033[0m")

        elif self.mode == "mcp":
            logger.info("\033[33mMCP Client 启动成功，准备启动 MCP Server\033[0m")

            server_params = StdioServerParameters(
                command=self.mcp_server_command,
                args=[self.mcp_server_path],
            )

            logger.info("\033[33m正在启动 MCP Server\033[0m")
            read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))

            logger.info("\033[33m正在建立 Client Session 对象\033[0m")
            self.mcp_session = await self.exit_stack.enter_async_context(ClientSession(read, write))

            logger.info("\033[33m成功建立 Client Session 对象，即将握手初始化\033[0m")
            await self.mcp_session.initialize()

            logger.info("\033[33m握手初始化成功，正在构建 MCP 代理工具\033[0m")
            self.langchain_tools = self._build_mcp_proxy_tools()

            tool_names = [t.name for t in self.langchain_tools]
            logger.info(f"\033[33m成功构建并装载 MCP 代理工具: {tool_names}\033[0m")

        else:
            raise ValueError(f"Commander init error: unknown mode '{self.mode}'")

        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts("commander"),
            tools=self.langchain_tools,
            middleware=[
                log_before_agent,
                log_after_agent,
                log_before_model,
                log_after_model,
            ],
        )

    async def execute_stream(
        self,
        user_query: str,
        session_id: str,
        should_cancel: Callable[[], bool] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        if self.agent is None:
            raise RuntimeError("Commander 尚未初始化，请先调用 await commander.setup()")

        runtime = CommanderRequestRuntime(
            session_id=session_id,
            user_query=user_query,
            cancel_checker=should_cancel,
        )
        token = self.request_runtime.set(runtime)

        if self.mode == "langchain":
            self.toolbox.set_request_context(session_id=session_id, user_query=user_query)

        input_dict = {"messages": [("human", user_query)]}
        latest_full_text = ""

        try:
            self._queue_process_event("request_received", "已收到请求，准备处理。")
            for event in self._drain_runtime_events():
                yield event

            self._check_cancelled()
            self._queue_process_event("agent_start", "主控代理开始分析请求。")
            for event in self._drain_runtime_events():
                yield event

            if hasattr(self.agent, "astream"):
                async for step in self.agent.astream(
                    input_dict,
                    stream_mode="values",
                    context={"agent_name": "commander"},
                ):
                    self._check_cancelled()
                    for event in self._drain_runtime_events():
                        yield event

                    latest_message = step["messages"][-1]
                    if isinstance(latest_message, AIMessage):
                        current_text = self._coerce_message_text(latest_message.content).strip()
                        delta_text = self._extract_delta_text(latest_full_text, current_text)
                        if delta_text:
                            latest_full_text = current_text
                            yield {"event": "delta", "text": delta_text}
            else:
                response = await self.agent.ainvoke(input_dict, context={"agent_name": "commander"})
                self._check_cancelled()
                for event in self._drain_runtime_events():
                    yield event
                latest_message = response["messages"][-1]
                if isinstance(latest_message, AIMessage):
                    current_text = self._coerce_message_text(latest_message.content).strip()
                    if current_text:
                        latest_full_text = current_text
                        yield {"event": "delta", "text": current_text}

            self._check_cancelled()
            self._queue_process_event("reply_complete", "当前回复已完成。")
            for event in self._drain_runtime_events():
                yield event

            if not latest_full_text:
                yield {"event": "delta", "text": "未能生成有效回复"}
        finally:
            self.request_runtime.reset(token)

    async def execute(self, user_query: str, session_id: str) -> str:
        chunks: list[str] = []
        async for event in self.execute_stream(user_query=user_query, session_id=session_id):
            if event.get("event") == "delta":
                chunks.append(event.get("text", ""))

        final_text = "".join(chunks).strip()
        return final_text or "未能生成有效回复"

    async def close(self):
        if self.mode == "mcp":
            await self.exit_stack.aclose()
            logger.info("\033[33mMCP 连接已关闭。\033[0m")


if __name__ == "__main__":
    import asyncio

    async def main():
        commander = Commander(mode="langchain")

        try:
            await commander.setup()
            print("========== 测试 ==========")
            res = await commander.execute("刚才那张照片变成白色裙子", session_id="demo-session")
            print(res)
        finally:
            await commander.close()

    asyncio.run(main())
