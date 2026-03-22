from utils.logger_handler import logger
from langchain_core.tools import tool

from rag.rag_service import RagSummarizeService

rag = RagSummarizeService()

@tool(description="从向量存储中检索参考资料")
def rag_retrieve_context(query: str) -> str:
    logger.info(f"\033[34m[rag_summarize]调用工具\033[0m")
    return rag.rag_retrieve_context(query)



