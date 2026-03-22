from typing import Callable
from utils.prompt_loader import load_system_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import (
    after_agent,
    after_model,
    before_agent,
    before_model,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger

@before_agent
def log_before_agent(state: AgentState, runtime: Runtime) -> None:
    """
    Agent 执行之前的拦截钩子。

    应用场景：
    - 记录 Agent 启动日志
    - 验证输入参数
    - 修改初始状态
    """
    print(f"before_{runtime.context['agent_name']}", state)
    logger.info(f"[log_before_agent] Starting agent with {len(state['messages'])} messages")

@after_agent
def log_after_agent(state: AgentState, runtime: Runtime) -> None:
    """
    Agent 执行之后的拦截钩子。

    应用场景：
    - 记录 Agent 完成日志
    - 统计执行时间
    - 保存对话历史
    """
    print(f"after_{runtime.context['agent_name']}", state)
    logger.info(f"[log_after_agent] Agent completed with {len(state['messages'])} messages")

@before_model
def log_before_model(
        state: AgentState,          # 整个Agent中的状态记录
        runtime: Runtime,           # 记录了整个执行过程中的上下文信息
):         # 在模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")
    print("before_model",state)
    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content}")

    return None

@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> None:
    """
    模型执行之后的拦截钩子。

    应用场景：
    - 记录模型返回结果
    - 分析模型输出
    - 修改模型响应
    """
    if state["messages"]:
        logger.info(f"[log_after_model]模型调用完成")
    print("after_model", state)