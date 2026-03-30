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

from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from tools.agent_tools import rag_retrieve_context
from middleware.middleware import log_before_agent,log_after_agent,log_before_model,log_after_model



class Features_agent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts("features_agent"),
            tools=[rag_retrieve_context],
            middleware=[log_before_agent,log_after_agent,log_before_model,log_after_model]
        )

    def execute_stream(self, character_name: str,scene:str):
        input_dict = {
            "messages": [
                ("human", f"{character_name}  场景：{scene}"),
            ]
        }
        """
        self.agent.stream(..., stream_mode="values") 返回的是一个迭代器。
        每次迭代拿到的 step，本质上就是当前时刻的 state 快照。
    
        state 中最核心的字段是 messages，它保存了当前 Agent 执行过程中的所有消息，
        例如 HumanMessage、AIMessage、ToolMessage 等。
    
        step 的结构大致如下：
        {
            "messages": [
                HumanMessage(...),
                AIMessage(tool_calls=[...]),
                ToolMessage(...),
                AIMessage(content="青灵的外貌是...")
            ]
        }
    
        注意：
        1. step["messages"] 包含的是“到当前为止的全部消息”，
           不是只有最新生成的那一条。
        2. 每次 stream 产出一个 step，可以理解为 Agent 完成了一个执行阶段，
           例如：
           - 接收用户输入
           - 模型思考并决定是否调用工具
           - 工具执行并返回结果
           - 模型基于工具结果生成最终回答
        3. step["messages"][-1] 表示当前这个阶段最新产生的一条消息。
        4. 这里只关心 AIMessage，并且要求 content 非空，
           这样可以过滤掉 HumanMessage、ToolMessage，以及仅包含 tool_calls 的空 AIMessage。
        5. yield 的作用是把模型生成的文本分段返回给外层调用者（line68 的 chunk）
           从而实现流式输出。
        """

        for step in self.agent.stream(input_dict, stream_mode="values",context={"agent_name": "features_agent"}):
            latest_message = step["messages"][-1]
            if isinstance(latest_message, AIMessage):
                if latest_message.content:
                    yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    agent = Features_agent()

    for chunk in agent.execute_stream("给我青灵的人物特征，包括身材，五官，性格，外貌，气质等"):
        print(chunk) 
