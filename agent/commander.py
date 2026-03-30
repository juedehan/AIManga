import importlib.util
from pathlib import Path
from typing import Literal, Optional
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

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession


class Commander:
    def __init__(
        self,
        mode: Literal["mcp", "langchain"] | None = None,
        mcp_server_path: str | None = None,
    ):
        commander_conf = agent_conf.get("commander", {})
        self.mode = mode or commander_conf.get("default_mode", "langchain")
        server_relpath = mcp_server_path or commander_conf.get("mcp_server_path", "mcp/mcp_server.py")
        self.mcp_server_path = get_abs_path(server_relpath)
        self.mcp_server_command = commander_conf.get("mcp_server_command", "python3")

        self.agent = None
        self.langchain_tools = []
        self.toolbox = CommanderToolbox()
        self.exit_stack = AsyncExitStack()
        self.mcp_session: Optional[ClientSession] = None

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

            logger.info("\033[33m握手初始化成功，正在获取工具\033[0m")
            self.langchain_tools = await load_mcp_tools(self.mcp_session)

            tool_names = [t.name for t in self.langchain_tools]
            logger.info(f"\033[33m成功从 MCP Server 获取并装载工具: {tool_names}\033[0m")

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

    async def execute(self, user_query: str, session_id: str) -> str:
        if self.agent is None:
            raise RuntimeError("Commander 尚未初始化，请先调用 await commander.setup()")

        if self.mode == "langchain":
            self.toolbox.set_request_context(session_id=session_id, user_query=user_query)

        input_dict = {"messages": [("human", user_query)]}
        response = await self.agent.ainvoke(input_dict, context={"agent_name": "commander"})

        latest_message = response["messages"][-1]
        if isinstance(latest_message, AIMessage) and latest_message.content:
            return latest_message.content.strip()

        return "未能生成有效回复"

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
            res = await commander.execute("画出日常生活里高欣欣的肖像", session_id="demo-session")
            print(res)
        finally:
            await commander.close()

    asyncio.run(main())
