from typing import Literal, Optional

from contextlib import AsyncExitStack

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from utils.logger_handler import logger
from middleware.middleware import (
    log_before_agent,
    log_after_agent,
    log_before_model,
    log_after_model,
)

# 本地工具
from tools.commander_tools import get_features, opt_prompts, get_portrait

# MCP 相关
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientSession


class Commander:
    def __init__(
        self,
        mode: Literal["mcp", "langchain"] = "langchain",
        mcp_server_path: str = "/home/hjz/AI漫改/mcp/mcp_server.py",
    ):
        self.mode = mode
        self.mcp_server_path = mcp_server_path

        self.agent = None
        self.langchain_tools = []

        # 仅在MCP模式下有用
        self.exit_stack = AsyncExitStack()
        self.mcp_session: Optional[ClientSession] = None

    async def setup(self):
        """
        根据mode初始化工具和 Agent
        """
        if self.mode == "langchain":
            logger.info("\033[33m正在以本地 LangChain Tool 模式启动......\033[0m")

            self.langchain_tools = [
                tool(get_features),
                tool(opt_prompts),
                tool(get_portrait),
            ]

            tool_names = [t.name for t in self.langchain_tools]
            logger.info(f"\033[33m本地工具装载成功: {tool_names}\033[0m")

        elif self.mode == "mcp":
            logger.info("\033[33mMCP Client 启动成功，准备启动 MCP Server\033[0m")

            server_params = StdioServerParameters(
                command="python",
                args=[self.mcp_server_path],
            )

            # 建立stdio通道
            logger.info("\033[33m正在启动 MCP Server\033[0m")
            read, write = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )

            # 建立MCP会话
            logger.info("\033[33m正在建立 Client Session 对象\033[0m")
            self.mcp_session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )

            # 握手初始化
            logger.info("\033[33m成功建立 Client Session 对象，即将握手初始化\033[0m")
            await self.mcp_session.initialize()

            # 将 MCP Tools 转为 LangChain Tools
            logger.info("\033[33m握手初始化成功，正在获取工具\033[0m")
            self.langchain_tools = await load_mcp_tools(self.mcp_session)

            tool_names = [t.name for t in self.langchain_tools]
            logger.info(f"\033[33m成功从 MCP Server 获取并装载工具: {tool_names}\033[0m")

        else:
            raise ValueError(f"Commander init error: unknown mode '{self.mode}'")

        # 不管是哪种模式，最后统一创建 Agent
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

    async def execute(self, user_query: str) -> str:
        """
        接收用户指令，Commander 自主决定调用哪些工具
        """
        if self.agent is None:
            raise RuntimeError("Commander 尚未初始化，请先调用 await commander.setup()")

        input_dict = {
            "messages": [
                ("human", user_query),
            ]
        }

        # 统一使用异步调用，兼容 langchain/mcp 两种模式
        response = await self.agent.ainvoke(input_dict,context={"agent_name": "commander"})

        latest_message = response["messages"][-1]
        if isinstance(latest_message, AIMessage) and latest_message.content:
            return latest_message.content.strip()

        return "未能生成有效回复"

    async def close(self):
        """
        关闭 MCP 连接
        """
        if self.mode == "mcp":
            await self.exit_stack.aclose()
            logger.info("\033[33mMCP 连接已关闭。\033[0m")


# ----------------------------
# 测试入口
# ----------------------------
if __name__ == "__main__":
    import asyncio

    async def main():
        commander = Commander(mode="langchain")

        try:
            await commander.setup()

            print("========== 测试 ==========")
            res = await commander.execute("日常生活中林月的肖像")
            print(res)

        finally:
            await commander.close()

    asyncio.run(main())