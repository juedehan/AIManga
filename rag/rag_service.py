
"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from utils.config_handler import rag_conf
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model,rerank_model
from utils.logger_handler import logger


def print_prompt(prompt):
    logger.info("\033[34m" + "\n" + "=" * 20 + "\n" + prompt.to_string() + "\n" + "=" * 20 + "\033[0m")
    return prompt


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()
        self.rerank_top_k = rag_conf["rerank_top_k"]
        self.relevance_score = rag_conf["relevance_score"]

    def _init_chain(self):
        # 此处的chain只负责rag获得传入agent的model前的prompt，不再用model进行总结 ，当生成的prompt过长时，可以在rag时使用model总结
        chain = self.prompt_template | print_prompt #| self.model | StrOutputParser()
        return chain

    def do_rerank(self, query: str, docs: list[Document]) -> list[Document]:
        if not docs:
            return []

        documents = [doc.page_content for doc in docs]

        """
               官方SDK输出示例
            {
               "status_code": 200,
               "request_id": "4b0805c0-6b36-490d-8bc1-4365f4c89905",
               "code": "",
               "message": "",
               "output": {
                   "results": [
                       {
                           "index": 0,
                           "relevance_score": 0.9334521178273196,
                           "document": {
                               "text": "文本排序模型广泛用于搜索引擎和推荐系统中，它们根据文本相关性对候选文本进行排序"
                           }
                       },
                       {
                           "index": 2,
                           "relevance_score": 0.34100082626411193,
                           "document": {
                               "text": "预训练语言模型的发展给文本排序模型带来了新的进展"
                           }
                       }
                   ]
               },
               "usage": {
                   "total_tokens": 79
               }
           }
               """
        resp = rerank_model.rerank(
            query=query,
            documents=documents,
            rerank_top_k=min(self.rerank_top_k, len(documents)),
            return_documents=False,
        )


        results = resp.output.results

        # 相关性过滤
        ranked_docs = [
            docs[item.index]
            for item in results
            if item.relevance_score > self.relevance_score
        ]
        # 如果全被过滤掉，至少保留前1条
        if not ranked_docs and results:
            ranked_docs = [docs[results[0].index]]

        return ranked_docs


    def retriever_docs(self, query: str) -> list[Document]:
        # 先召回
        docs = self.retriever.invoke(query)
        # 再rerank
        ranked_docs = self.do_rerank(query, docs)
        return ranked_docs

    def rag_retrieve_context(self, query: str) -> str:

        context_docs = self.retriever_docs(query)

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】\n{doc.page_content} \n>>>> 参考元数据：{doc.metadata}\n"
            #context += f">>>>>> 参考资料{counter} <<<<<<\n{doc.page_content} \n 【参考元数据】：{doc.metadata}\n"
        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )


if __name__ == '__main__':
    rag = RagSummarizeService()
    rag.rag_retrieve_context("没有打斗时，关于青灵的外貌描写")
    #print(rag.rag_retrieve_context("关于青灵的外貌描写"))
