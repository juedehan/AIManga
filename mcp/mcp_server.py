import importlib.util
from pathlib import Path

_BOOTSTRAP_PATH = Path(__file__).resolve().parents[1] / "bootstrap.py"
_BOOTSTRAP_SPEC = importlib.util.spec_from_file_location("aimanga_bootstrap", _BOOTSTRAP_PATH)
if _BOOTSTRAP_SPEC is None or _BOOTSTRAP_SPEC.loader is None:
    raise ImportError(f"无法加载 bootstrap: {_BOOTSTRAP_PATH}")
_bootstrap = importlib.util.module_from_spec(_BOOTSTRAP_SPEC)
_BOOTSTRAP_SPEC.loader.exec_module(_bootstrap)
_bootstrap.ensure_project_root()
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
