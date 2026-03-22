import os
import sys

#把项目根目录加入 Python 的模块搜索路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.logger_handler import logger
from mcp.server.fastmcp import FastMCP
from tools.commander_tools import get_portrait, get_features, opt_prompts

mcp = FastMCP("mcp_server")

mcp.add_tool(get_features)
mcp.add_tool(opt_prompts)
mcp.add_tool(get_portrait)

if __name__ == "__main__":
    logger.info(f"\033[33mMCP Server 启动成功\033[0m")
    mcp.run("stdio")
