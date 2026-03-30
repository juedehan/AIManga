from abc import ABC, abstractmethod
from typing import Optional, Any
from http import HTTPStatus

import dashscope
from dotenv import load_dotenv
from openai import OpenAI

from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi

from utils.config_handler import model_conf

load_dotenv()


class RerankModel:
    """
    百炼 rerank 客户端封装
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

        if not resp.status_code == HTTPStatus.OK:
            raise RuntimeError(f"Rerank 调用失败: {resp}")

        return resp


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
        return OpenAI(base_url=model_conf["image_base_url"])


class RerankModelFactory(BaseModelFactory):
    def generator(self) -> ModelType:
        return RerankModel(model_name=model_conf["rerank_model_name"])


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
image_model = ImageModelFactory().generator()
rerank_model = RerankModelFactory().generator()
