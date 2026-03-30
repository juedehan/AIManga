import importlib.util
from pathlib import Path

_BOOTSTRAP_PATH = Path(__file__).resolve().parents[1] / "bootstrap.py"
_BOOTSTRAP_SPEC = importlib.util.spec_from_file_location("aimanga_bootstrap", _BOOTSTRAP_PATH)
if _BOOTSTRAP_SPEC is None or _BOOTSTRAP_SPEC.loader is None:
    raise ImportError(f"无法加载 bootstrap: {_BOOTSTRAP_PATH}")
_bootstrap = importlib.util.module_from_spec(_BOOTSTRAP_SPEC)
_BOOTSTRAP_SPEC.loader.exec_module(_bootstrap)
_bootstrap.ensure_project_root()
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from middleware.middleware import log_before_agent, log_after_agent, log_before_model, log_after_model
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts


class PortraitAdjustmentAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts("portrait_adjustment_agent"),
            tools=[],
            middleware=[log_before_agent, log_after_agent, log_before_model, log_after_model],
        )

    def execute(
        self,
        base_prompt: str,
        modification_request: str,
        character_name: str = "",
    ) -> str:
        """根据历史最终提示词和用户新的修改要求输出完整的新提示词。"""
        input_dict = {
            "messages": [
                (
                    "human",
                    f"角色名：{character_name}\n历史最终提示词：{base_prompt}\n本轮修改要求：{modification_request}",
                )
            ]
        }

        response = self.agent.invoke(input_dict, context={"agent_name": "portrait_adjustment_agent"})
        latest_message = response["messages"][-1]

        if isinstance(latest_message, AIMessage) and latest_message.content:
            return latest_message.content.strip()

        return ""
