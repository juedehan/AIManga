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
from tools.agent_tools import (
    load_character_portrait_memory,
    rag_retrieve_context_impl,
    resolve_latest_character_for_session,
    save_character_portrait_memory,
    update_character_portrait_memory,
)
from tools.commander_tools import (
    mcp_adjust_existing_portrait,
    mcp_get_features,
    mcp_get_portrait,
    mcp_opt_prompts,
)

mcp = FastMCP("mcp_server")

mcp.add_tool(mcp_get_features, name="get_features")
mcp.add_tool(mcp_opt_prompts, name="opt_prompts")
mcp.add_tool(mcp_get_portrait, name="get_portrait")
mcp.add_tool(mcp_adjust_existing_portrait, name="adjust_existing_portrait")

mcp.add_tool(rag_retrieve_context_impl, name="rag_retrieve_context")
mcp.add_tool(resolve_latest_character_for_session)
mcp.add_tool(load_character_portrait_memory)
mcp.add_tool(save_character_portrait_memory)
mcp.add_tool(update_character_portrait_memory)

if __name__ == "__main__":
    logger.info(f"\033[33mMCP Server 启动成功\033[0m")
    mcp.run("stdio")
