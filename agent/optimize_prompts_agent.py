from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from middleware.middleware import log_before_agent, log_after_agent, log_before_model, log_after_model
from typing import Literal, Optional



class Optimize_prompts_agent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts("optimize_prompts_agent"),
            tools=[],  # 这个 agent负责文本重构和视觉翻译，不需要RAG工具
            middleware=[log_before_agent, log_after_agent, log_before_model, log_after_model]
        )

    def execute(self, character_description: str,gender: Optional[Literal["男", "女"]] = "未提及") -> str:
        """
        接收 Features_agent 提取的文学性侧写，一次性输出结构化的 AI 绘画提示词
        """
        input_dict = {
            "messages": [
                ("human", f"人物描述：{character_description}  性别：{gender}")
            ]
        }

        # 使用 invoke 一次性执行完毕，方便下游直接拿完整的字符串进行解析
        response = self.agent.invoke(input_dict,context={"agent_name": "optimize_prompts_agent"})

        # 从最终的 state 中提取最新的一条 AIMessage
        latest_message = response["messages"][-1]

        if isinstance(latest_message, AIMessage) and latest_message.content:
            return latest_message.content.strip()

        return ""