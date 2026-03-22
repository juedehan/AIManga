import os
from abc import ABC, abstractmethod
from typing import Optional, Union, Any
from http import HTTPStatus

import dashscope
from dotenv import load_dotenv
from openai import OpenAI

from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi

from utils.config_handler import load_model_config

# 确保环境变量被加载
load_dotenv()
model_conf = load_model_config()
class RerankModel:
    """
    百炼 rerank 客户端封装
    用法：
        rerank_model.rerank(
            query="什么是文本排序模型",
            documents=["...", "..."],
            top_n=5
        )
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        documents: list[str],
        rerank_top_k: int = 5,
        return_documents: bool = True,
    ) -> Any:
        resp = dashscope.TextReRank.call(
            model=self.model_name,
            query=query,
            documents=documents,
            top_n=rerank_top_k,
            return_documents=return_documents,
        )

        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(f"Rerank 调用失败: {resp}")

        return resp

# 扩充类型提示：加入 OpenAI 类型，以兼容文生图客户端，后续可加入视频模型，语音模型等
ModelType = Optional[Embeddings | BaseChatModel | OpenAI | RerankModel]

class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> ModelType:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> ModelType:
        return ChatTongyi(model=model_conf["chat_model_name"])


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> ModelType:
        return DashScopeEmbeddings(model=model_conf["embedding_model_name"])


class ImageModelFactory(BaseModelFactory):
    def generator(self) -> ModelType:
        return OpenAI(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            # api_key 会自动读取 os.getenv("ARK_API_KEY") 或者 os.getenv("OPENAI_API_KEY")
        )

class RerankModelFactory(BaseModelFactory):
    def generator(self) -> ModelType:
        return RerankModel(model_name=model_conf["rerank_model_name"])

# ================= 实例化 =================

chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
image_model = ImageModelFactory().generator()
rerank_model = RerankModelFactory().generator()