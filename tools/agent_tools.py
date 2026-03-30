from langchain_core.tools import tool

from memory.portrait_memory_store import PortraitMemoryStore
from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.logger_handler import logger

rag = RagSummarizeService()
memory_conf = agent_conf.get("memory", {})
portrait_memory_store = PortraitMemoryStore(
    store_path=memory_conf.get("store_path", "memory/portrait_memory.sqlite3"),
    max_adjustment_history=memory_conf.get("max_adjustment_history", 10),
    enabled=memory_conf.get("enabled", True),
)


def rag_retrieve_context_impl(query: str) -> str:
    """从向量存储中检索参考资料。"""
    logger.info(f"\033[34m[rag_summarize]调用工具\033[0m")
    return rag.rag_retrieve_context(query)


@tool(description="从向量存储中检索参考资料")
def rag_retrieve_context(query: str) -> str:
    """从向量存储中检索参考资料。"""
    return rag_retrieve_context_impl(query)


def resolve_latest_character_for_session(session_id: str) -> str | None:
    """解析某个会话最近一次成功生成或调整过的角色名。"""
    return portrait_memory_store.resolve_latest_character(session_id)


def load_character_portrait_memory(session_id: str, character_name: str | None = None) -> dict | None:
    """按会话和角色读取肖像记忆，未提供角色时回退到最近角色。"""
    resolved_character = character_name or resolve_latest_character_for_session(session_id)
    if not resolved_character:
        return None
    return portrait_memory_store.get_character_memory(session_id, resolved_character)


def save_character_portrait_memory(
    session_id: str,
    character_name: str,
    character_name_pinyin: str,
    latest_final_prompt: str,
    latest_image_path: str,
    latest_scene: str | None = None,
    latest_gender: str | None = None,
    last_user_request: str | None = None,
) -> dict | None:
    """保存某角色在当前会话下最新一次成功出图的结果。"""
    return portrait_memory_store.save_character_memory(
        session_id=session_id,
        character_name=character_name,
        character_name_pinyin=character_name_pinyin,
        latest_final_prompt=latest_final_prompt,
        latest_image_path=latest_image_path,
        latest_scene=latest_scene,
        latest_gender=latest_gender,
        last_user_request=last_user_request,
    )


def update_character_portrait_memory(
    session_id: str,
    character_name: str,
    latest_final_prompt: str,
    latest_image_path: str,
    modification_request: str,
    character_name_pinyin: str | None = None,
    latest_scene: str | None = None,
    latest_gender: str | None = None,
    last_user_request: str | None = None,
) -> dict | None:
    """更新某角色在当前会话下的最新提示词、出图结果和修改历史。"""
    return portrait_memory_store.update_character_memory(
        session_id=session_id,
        character_name=character_name,
        latest_final_prompt=latest_final_prompt,
        latest_image_path=latest_image_path,
        modification_request=modification_request,
        character_name_pinyin=character_name_pinyin,
        latest_scene=latest_scene,
        latest_gender=latest_gender,
        last_user_request=last_user_request,
    )

