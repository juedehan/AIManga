import hashlib
import os
import shutil
import importlib.util
from pathlib import Path
from typing import Iterable

_BOOTSTRAP_PATH = Path(__file__).resolve().parents[1] / "bootstrap.py"
_BOOTSTRAP_SPEC = importlib.util.spec_from_file_location("aimanga_bootstrap", _BOOTSTRAP_PATH)
if _BOOTSTRAP_SPEC is None or _BOOTSTRAP_SPEC.loader is None:
    raise ImportError(f"无法加载 bootstrap: {_BOOTSTRAP_PATH}")
_bootstrap = importlib.util.module_from_spec(_BOOTSTRAP_SPEC)
_BOOTSTRAP_SPEC.loader.exec_module(_bootstrap)
_bootstrap.ensure_project_root()
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embed_model
from utils.config_handler import chroma_conf, rag_conf, vector_store_conf
from utils.file_handler import (
    get_file_md5_hex,
    listdir_with_allowed_type,
    pdf_loader,
    txt_loader,
)
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class VectorStoreService:
    def __init__(self):
        """初始化路径、配置、持久化向量库和文本切分器。"""
        self.persist_directory = get_abs_path(chroma_conf["persist_directory"])
        self.md5_store_path = get_abs_path(chroma_conf["md5_hex_store"])
        self.known_characters = vector_store_conf.get("known_characters", [])
        self.scene_keywords = vector_store_conf.get("scene_keywords", {})
        self.content_type_keywords = vector_store_conf.get("content_type_keywords", {})

        os.makedirs(self.persist_directory, exist_ok=True)
        self.vector_store = self._create_vector_store()
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf.get("chunk_size", 500),
            chunk_overlap=chroma_conf.get("chunk_overlap", 60),
            separators=chroma_conf.get("separators", ["\n\n", "\n", ". ", f"{chr(33)} ", "?", ""]),
            length_function=len,
        )

    def _create_vector_store(self) -> Chroma:
        """创建绑定统一 embedding 模型的 Chroma 集合。"""
        return Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=self.persist_directory,
        )

    def get_retriever(self, search_kwargs: dict | None = None):
        """返回基于当前向量库的检索器。"""
        kwargs = {"k": rag_conf.get("k", 5)}
        if search_kwargs:
            kwargs.update(search_kwargs)
        return self.vector_store.as_retriever(search_kwargs=kwargs)

    def get_vectorstore(self) -> Chroma:
        """暴露底层的 Chroma 向量库实例。"""
        return self.vector_store

    def reset_store(self) -> None:
        """ 删除持久化数据并从头重建向量库。"""
        try:
            self.vector_store.delete_collection()
        except Exception as exc:
            logger.warning(f"[重建知识库]删除旧 collection 失败，继续清理磁盘目录: {exc}")

        if os.path.isdir(self.persist_directory):
            shutil.rmtree(self.persist_directory, ignore_errors=True)
        os.makedirs(self.persist_directory, exist_ok=True)

        if os.path.exists(self.md5_store_path):
            os.remove(self.md5_store_path)

        self.vector_store = self._create_vector_store()
        logger.info("[重建知识库]向量数据库和 md5 索引已重置")

    def _read_processed_md5s(self) -> set[str]:
        """读取已经入库过的源文件哈希集合。"""
        if not os.path.exists(self.md5_store_path):
            return set()

        with open(self.md5_store_path, "r", encoding="utf-8") as file_obj:
            return {line.strip() for line in file_obj if line.strip()}

    def _append_processed_md5(self, md5_hex: str) -> None:
        """追加保存一个新处理完成的源文件哈希。"""
        with open(self.md5_store_path, "a", encoding="utf-8") as file_obj:
            file_obj.write(md5_hex + "\n")

    def _load_documents_for_path(self, read_path: str) -> list[Document]:
        """根据文件类型分派到对应的文档加载器。"""
        ext = os.path.splitext(read_path)[1].lower()
        if ext == ".txt":
            return txt_loader(read_path)
        if ext == ".pdf":
            return pdf_loader(read_path)

        logger.warning(f"[加载知识库]{read_path} 文件类型 {ext} 当前未实现 loader，跳过")
        return []

    def _extract_chapter(self, source_path: str) -> str:
        """从源文件名中提取章节标识。"""
        return os.path.splitext(os.path.basename(source_path))[0]

    def _classify_scene(self, text: str) -> str:
        """通过关键词命中次数给文本打场景标签。"""
        if not self.scene_keywords:
            return "unknown"

        scene_scores = {
            name: sum(1 for keyword in keywords if keyword in text)
            for name, keywords in self.scene_keywords.items()
        }
        best_scene = max(scene_scores, key=scene_scores.get)
        if scene_scores[best_scene] == 0:
            return "unknown"
        return best_scene

    def _classify_content_type(self, text: str) -> str:
        """通过关键词命中次数给文本打内容类型标签。"""
        if not self.content_type_keywords:
            return "unknown"

        type_scores = {
            name: sum(1 for keyword in keywords if keyword in text)
            for name, keywords in self.content_type_keywords.items()
        }
        best_type = max(type_scores, key=type_scores.get)
        if type_scores[best_type] == 0:
            return "unknown"
        return best_type

    def _extract_character_mentions(self, text: str) -> str:
        """提取文本中命中的已知角色名，并拼成逗号分隔字符串。"""
        matched = [name for name in self.known_characters if name in text]
        return ",".join(matched)

    def _build_text_hash(self, text: str) -> str:
        """为分块文本计算稳定的内容哈希。"""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _build_metadata(self, source_path: str, chunk: Document, chunk_index: int, source_md5: str) -> dict:
        """为单个文本分块组装写入向量库前的 metadata。"""
        text = chunk.page_content.strip()
        # 防止修改原数据
        metadata = dict(chunk.metadata or {})
        metadata.update(
            {
                "source": source_path,
                "chapter": self._extract_chapter(source_path),
                "scene_hint": self._classify_scene(text),
                "character_mentions": self._extract_character_mentions(text),
                "content_type": self._classify_content_type(text),
                "text_hash": self._build_text_hash(text),
                "chunk_index": chunk_index,
                "source_md5": source_md5,
            }
        )
        return metadata

    def _enrich_chunks(
        self,
        split_documents: Iterable[Document],
        source_path: str,
        source_md5: str,
    ) -> list[Document]:
        """清洗切分结果，并为每个分块附加生成好的 metadata。"""
        enriched_documents: list[Document] = []
        for index, chunk in enumerate(split_documents):
            text = chunk.page_content.strip()
            if not text:
                continue
            enriched_documents.append(
                Document(
                    page_content=text,
                    metadata=self._build_metadata(source_path, chunk, index, source_md5),
                )
            )
        return enriched_documents

    def load_document(self, force_rebuild: bool = False) -> dict:
        """把支持的源文件导入向量库，并返回入库统计信息。"""
        if force_rebuild:
            self.reset_store()

        processed_md5s = set() if force_rebuild else self._read_processed_md5s()
        allowed_files_path = sorted(
            listdir_with_allowed_type(
                get_abs_path(chroma_conf["data_path"]),
                tuple(chroma_conf["allow_knowledge_file_type"]),
            )
        )

        ingest_stats = {
            "processed_files": 0,
            "skipped_files": 0,
            "ingested_chunks": 0,
        }

        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)
            if not md5_hex:
                ingest_stats["skipped_files"] += 1
                continue

            if md5_hex in processed_md5s:
                logger.info(f"[加载知识库]{path} 内容已存在知识库内，跳过")
                ingest_stats["skipped_files"] += 1
                continue

            try:
                documents = self._load_documents_for_path(path)
                if not documents:
                    logger.warning(f"[加载知识库]{path} 内没有有效文本内容，跳过")
                    ingest_stats["skipped_files"] += 1
                    continue

                split_documents = self.spliter.split_documents(documents)
                enriched_documents = self._enrich_chunks(split_documents, path, md5_hex)
                if not enriched_documents:
                    logger.warning(f"[加载知识库]{path} 分片后没有有效文本内容，跳过")
                    ingest_stats["skipped_files"] += 1
                    continue

                self.vector_store.add_documents(enriched_documents)
                self._append_processed_md5(md5_hex)
                processed_md5s.add(md5_hex)

                ingest_stats["processed_files"] += 1
                ingest_stats["ingested_chunks"] += len(enriched_documents)
                logger.info(
                    f"[加载知识库]{path} 内容加载成功，新增 {len(enriched_documents)} 个 chunk"
                )
            except Exception as exc:
                logger.error(f"[加载知识库]{path} 加载失败：{exc}", exc_info=True)
                ingest_stats["skipped_files"] += 1

        logger.info(f"[加载知识库]完成，统计信息: {ingest_stats}")
        return ingest_stats


if __name__ == "__main__":
    persist_directory = get_abs_path(chroma_conf["persist_directory"])
    md5_store_path = get_abs_path(chroma_conf["md5_hex_store"])

    service = VectorStoreService()

    # if os.path.isdir(persist_directory):
    #     shutil.rmtree(persist_directory, ignore_errors=True)
    # if os.path.exists(md5_store_path):
    #     os.remove(md5_store_path)

    service.load_document(force_rebuild=False)

    # 删除指定元数据的向量
    # vectorstore = service.get_vectorstore()
    # res = vectorstore.delete(where = {"source" : f"/home/hjz/localrepo/AIManga/data/chapter8.txt"} )
