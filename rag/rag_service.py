import re
import importlib.util
from copy import deepcopy
from dataclasses import dataclass, field
from math import log2
from pathlib import Path
from typing import Any, Iterable

import yaml
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

_BOOTSTRAP_PATH = Path(__file__).resolve().parents[1] / "bootstrap.py"
_BOOTSTRAP_SPEC = importlib.util.spec_from_file_location("aimanga_bootstrap", _BOOTSTRAP_PATH)
if _BOOTSTRAP_SPEC is None or _BOOTSTRAP_SPEC.loader is None:
    raise ImportError(f"无法加载 bootstrap: {_BOOTSTRAP_PATH}")
_bootstrap = importlib.util.module_from_spec(_BOOTSTRAP_SPEC)
_BOOTSTRAP_SPEC.loader.exec_module(_bootstrap)
_bootstrap.ensure_project_root()

from model.factory import rerank_model
from rag.vector_store import VectorStoreService
from utils.config_handler import rag_conf, rag_service_conf, vector_store_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.prompt_loader import load_rag_prompts

ATTRIBUTE_GROUPS = rag_service_conf.get("attribute_groups", {})
SCENE_QUERY_KEYWORDS = rag_service_conf.get("scene_query_keywords", {})
SCENE_QUERY_ALIASES = rag_service_conf.get("scene_query_aliases", {})
STOP_CANDIDATES = set(rag_service_conf.get("stop_candidates", []))
APPEARANCE_ATTRIBUTE_GROUPS = set(rag_service_conf.get("appearance_attribute_groups", []))
EMOTION_ATTRIBUTE_GROUPS = set(rag_service_conf.get("emotion_attribute_groups", []))
DAILY_NEGATIVE_SCENE_TOKENS = rag_service_conf.get("daily_negative_scene_tokens", [])
QUERY_VARIANT_TRIGGER_GROUPS = rag_service_conf.get("query_variant_trigger_groups", {})


@dataclass
class RetrievalIntent:
    raw_query: str
    character: str | None = None
    attribute_terms: list[str] = field(default_factory=list)
    attribute_groups: list[str] = field(default_factory=list)
    required_scene: str | None = None
    excluded_scenes: list[str] = field(default_factory=list)
    content_types: list[str] = field(default_factory=list)
    query_variants: list[str] = field(default_factory=list)
    retry_count: int = 0


@dataclass
class EvidenceItem:
    content: str
    metadata: dict[str, Any]
    rerank_score: float
    match_reasons: list[str] = field(default_factory=list)


@dataclass
class EvalSample:
    query: str
    target_character: str | None = None
    required_scene: str | None = None
    required_attributes: list[str] = field(default_factory=list)
    gold_spans: list[str] = field(default_factory=list)
    gold_sources: list[str] = field(default_factory=list)
    expect_no_answer: bool = False


@dataclass
class EvalResult:
    query: str
    mode: str
    retrieved_count: int
    top3_hit: int
    top3_precision: float
    ndcg3: float
    tp: int
    fp: int
    tn: int
    fn: int


@dataclass
class EvalReport:
    mode: str
    sample_count: int
    metrics: dict[str, float]
    results: list[EvalResult] = field(default_factory=list)


def print_prompt(prompt):
    """打印最终拼装出的 RAG prompt，便于调试检索上下文。"""
    logger.info("\033[34m" + "\n" + "=" * 20 + "\n" + prompt.to_string() + "\n" + "=" * 20 + "\033[0m")
    return prompt


class RagSummarizeService:
    def __init__(self):
        """初始化向量库、提示词模板和检索相关配置。"""
        self.vector_store_service = VectorStoreService()
        self.vector_store = self.vector_store_service.get_vectorstore()
        self.retriever = self.vector_store_service.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.chain = self._init_chain()

        self.final_k = rag_conf.get("k", 4)
        self.candidate_k = rag_conf.get("candidate_k", max(self.final_k, 8))
        self.rerank_top_k = rag_conf.get("rerank_top_k", max(self.final_k, 8))
        self.relevance_score = rag_conf.get("relevance_score", 0.45)
        self.enable_agentic_retry = rag_conf.get("enable_agentic_retry", True)
        self.low_confidence_min_evidence = rag_conf.get("low_confidence_min_evidence", 2)
        self.low_confidence_avg_score = rag_conf.get("low_confidence_avg_score", 0.55)
        self.max_query_variants = rag_conf.get("max_query_variants", 6)
        self.known_characters = vector_store_conf.get("known_characters", [])

    def _init_chain(self):
        """构建只负责渲染和打印 prompt 的轻量链路。"""
        return self.prompt_template | print_prompt

    def _dedupe_preserve_order(self, items: Iterable[str]) -> list[str]:
        """按原始顺序去重，过滤空项。"""
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            if not item or item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    def _extract_query_character(self, query: str) -> str | None:
        """从查询中提取目标角色名，优先匹配白名单角色。"""
        for name in self.known_characters:
            if name in query:
                return name

        match = re.search(r"([\u4e00-\u9fff]{2,4})(?=的)", query)
        if match:
            candidate = match.group(1)
            if candidate not in STOP_CANDIDATES:
                return candidate
        return None

    def _extract_attributes(self, query: str) -> tuple[list[str], list[str]]:
        """从查询中抽取命中的属性关键词及其所属属性组。"""
        matched_terms: list[str] = []
        matched_groups: list[str] = []
        for group_name, keywords in ATTRIBUTE_GROUPS.items():
            group_terms = [keyword for keyword in keywords if keyword in query]
            if group_terms:
                matched_terms.extend(group_terms)
                matched_groups.append(group_name)

        return self._dedupe_preserve_order(matched_terms), self._dedupe_preserve_order(matched_groups)

    def _extract_scene_constraints(self, query: str) -> tuple[str | None, list[str]]:
        """识别查询中的场景要求，并给出需要排除的场景。"""
        lowered = query.replace(" ", "")
        if any(token in lowered for token in DAILY_NEGATIVE_SCENE_TOKENS):
            return "daily", ["battle"]

        if any(token in query for token in SCENE_QUERY_KEYWORDS["daily"]):
            return "daily", ["battle"]

        if any(token in query for token in SCENE_QUERY_KEYWORDS["battle"]):
            return "battle", ["daily"]

        return None, []

    def _infer_content_types(self, attribute_groups: list[str]) -> list[str]:
        """根据属性组推断更适合检索的内容类型标签。"""
        if not attribute_groups:
            return []

        content_types: list[str] = []
        if any(group in APPEARANCE_ATTRIBUTE_GROUPS for group in attribute_groups):
            content_types.append("appearance")
        if any(group in EMOTION_ATTRIBUTE_GROUPS for group in attribute_groups):
            content_types.append("emotion")
        return self._dedupe_preserve_order(content_types)

    def _build_query_variants(
        self,
        raw_query: str,
        character: str | None,
        attribute_terms: list[str],
        required_scene: str | None,
    ) -> list[str]:
        """围绕角色、属性和场景生成多路检索查询变体。"""
        variants = [raw_query]
        joined_attributes = " ".join(attribute_terms) if attribute_terms else ""

        if character:
            variants.append(character)
            if joined_attributes:
                variants.append(f"{character} {joined_attributes}")
                variants.append(f"{character} 描写 {joined_attributes}")
            else:
                variants.append(f"{character} 描写")

        if required_scene:
            for scene_alias in SCENE_QUERY_ALIASES.get(required_scene, []):
                if character and joined_attributes:
                    variants.append(f"{scene_alias} {character} {joined_attributes}")
                elif character:
                    variants.append(f"{scene_alias} {character} 描写")
                else:
                    variants.append(f"{scene_alias} {raw_query}")

        if character and any(group in set(QUERY_VARIANT_TRIGGER_GROUPS.get("appearance", [])) for group in self._dedupe_preserve_order(self._extract_attributes(raw_query)[1])):
            variants.append(f"{character} 外貌 衣着")
        if character and any(group in set(QUERY_VARIANT_TRIGGER_GROUPS.get("expression", [])) for group in self._dedupe_preserve_order(self._extract_attributes(raw_query)[1])):
            variants.append(f"{character} 气质 神态 眼神")

        return self._dedupe_preserve_order(variants)[: self.max_query_variants]

    def analyze_query(self, query: str) -> RetrievalIntent:
        """把原始查询解析为结构化的检索意图对象。"""
        character = self._extract_query_character(query)
        attribute_terms, attribute_groups = self._extract_attributes(query)
        required_scene, excluded_scenes = self._extract_scene_constraints(query)
        content_types = self._infer_content_types(attribute_groups)
        query_variants = self._build_query_variants(query, character, attribute_terms, required_scene)
        return RetrievalIntent(
            raw_query=query,
            character=character,
            attribute_terms=attribute_terms,
            attribute_groups=attribute_groups,
            required_scene=required_scene,
            excluded_scenes=excluded_scenes,
            content_types=content_types,
            query_variants=query_variants,
        )

    def do_rerank(self, query: str, docs: list[Document]) -> list[tuple[Document, float]]:
        """使用 rerank 模型对候选文档重新排序并附上分数。"""
        if not docs:
            return []

        documents = [doc.page_content for doc in docs]
        resp = rerank_model.rerank(
            query=query,
            documents=documents,
            rerank_top_k=min(self.rerank_top_k, len(documents)),
            return_documents=False,
        )

        ranked_docs: list[tuple[Document, float]] = []
        for item in resp.output.results:
            ranked_docs.append((docs[item.index], float(item.relevance_score)))
        return ranked_docs

    def _build_search_filters(self, intent: RetrievalIntent, use_scene_filter: bool) -> list[dict | None]:
        """根据检索意图构造 metadata 过滤条件集合。"""
        filters: list[dict | None] = [None]
        if use_scene_filter and intent.required_scene:
            filters.insert(0, {"scene_hint": intent.required_scene})
        if use_scene_filter:
            for content_type in intent.content_types:
                filters.append({"content_type": content_type})
        return filters

    def _search_candidates(self, intent: RetrievalIntent, use_scene_filter: bool) -> list[Document]:
        """基于查询变体和过滤条件进行多路召回，并对候选去重。"""
        filters = self._build_search_filters(intent, use_scene_filter)
        docs_by_hash: dict[str, Document] = {}

        for query_variant in intent.query_variants:
            for filter_payload in filters:
                try:
                    search_kwargs = {"k": self.candidate_k}
                    if filter_payload:
                        search_kwargs["filter"] = filter_payload
                    docs = self.vector_store.similarity_search(query_variant, **search_kwargs)
                except Exception as exc:
                    logger.warning(
                        f"[RAG]query={query_variant} filter={filter_payload} 检索失败，回退为无过滤搜索: {exc}"
                    )
                    docs = self.vector_store.similarity_search(query_variant, k=self.candidate_k)

                for doc in docs:
                    content = doc.page_content.strip()
                    if not content:
                        continue
                    text_hash = doc.metadata.get("text_hash") if doc.metadata else None
                    dedupe_key = text_hash or f"{doc.metadata.get('source', '')}:{hash(content)}"
                    docs_by_hash[dedupe_key] = doc

        return list(docs_by_hash.values())

    def _matches_character(self, intent: RetrievalIntent, evidence: EvidenceItem) -> bool:
        """判断证据是否与目标角色匹配。"""
        if not intent.character:
            return True
        mentions = evidence.metadata.get("character_mentions", "")
        return intent.character in evidence.content or intent.character in mentions.split(",")

    def _matched_attribute_groups(self, text: str, attribute_groups: list[str]) -> list[str]:
        """统计文本命中了哪些目标属性组。"""
        matched: list[str] = []
        for group_name in attribute_groups:
            keywords = ATTRIBUTE_GROUPS.get(group_name, [])
            if any(keyword in text for keyword in keywords):
                matched.append(group_name)
        return matched

    def _matches_attributes(self, intent: RetrievalIntent, evidence: EvidenceItem) -> bool:
        """判断证据是否覆盖了查询要求的属性信息。"""
        if not intent.attribute_groups:
            return True

        matched_groups = self._matched_attribute_groups(evidence.content, intent.attribute_groups)
        if matched_groups:
            return True

        content_type = evidence.metadata.get("content_type")
        if content_type == "appearance" and any(group in APPEARANCE_ATTRIBUTE_GROUPS for group in intent.attribute_groups):
            return True
        if content_type == "emotion" and any(group in EMOTION_ATTRIBUTE_GROUPS for group in intent.attribute_groups):
            return True
        return False

    def _build_match_reasons(self, intent: RetrievalIntent, doc: Document, rerank_score: float) -> list[str]:
        """为一条证据生成命中原因，方便调试和评估。"""
        reasons: list[str] = []
        text = doc.page_content
        metadata = doc.metadata or {}

        if intent.character and (intent.character in text or intent.character in metadata.get("character_mentions", "")):
            reasons.append("character")
        if intent.required_scene and metadata.get("scene_hint") == intent.required_scene:
            reasons.append("scene")
        if metadata.get("scene_hint") == "unknown":
            reasons.append("scene_unknown")
        if intent.content_types and metadata.get("content_type") in intent.content_types:
            reasons.append("content_type")

        matched_groups = self._matched_attribute_groups(text, intent.attribute_groups)
        if matched_groups:
            reasons.append("attributes:" + ",".join(matched_groups))
        if rerank_score >= self.low_confidence_avg_score:
            reasons.append("strong_rerank")

        return reasons

    def _build_evidence_items(
        self,
        intent: RetrievalIntent,
        ranked_docs: list[tuple[Document, float]],
    ) -> list[EvidenceItem]:
        """把重排后的文档转换为统一的证据对象。"""
        evidence_items: list[EvidenceItem] = []
        for doc, rerank_score in ranked_docs:
            evidence_items.append(
                EvidenceItem(
                    content=doc.page_content,
                    metadata=dict(doc.metadata or {}),
                    rerank_score=rerank_score,
                    match_reasons=self._build_match_reasons(intent, doc, rerank_score),
                )
                )
        return evidence_items

    def _filter_evidence(self, intent: RetrievalIntent, evidence_items: list[EvidenceItem]) -> list[EvidenceItem]:
        """按分数、角色、属性和场景约束过滤最终证据。"""
        filtered_items: list[EvidenceItem] = []
        for evidence in evidence_items:
            if evidence.rerank_score < self.relevance_score:
                continue
            if intent.excluded_scenes and evidence.metadata.get("scene_hint") in intent.excluded_scenes:
                continue
            if intent.required_scene and evidence.metadata.get("scene_hint") not in {intent.required_scene, "unknown"}:
                continue
            if not self._matches_character(intent, evidence):
                continue
            if not self._matches_attributes(intent, evidence):
                continue
            filtered_items.append(evidence)

        filtered_items.sort(
            key=lambda item: (item.rerank_score, len(item.match_reasons)),
            reverse=True,
        )
        return filtered_items[: self.final_k]

    def _retrieve_once(self, intent: RetrievalIntent, use_scene_filter: bool) -> list[EvidenceItem]:
        """执行一轮完整的召回、重排和过滤流程。"""
        candidates = self._search_candidates(intent, use_scene_filter=use_scene_filter)
        ranked_docs = self.do_rerank(intent.raw_query, candidates)
        evidence_items = self._build_evidence_items(intent, ranked_docs)
        return self._filter_evidence(intent, evidence_items)

    def _should_retry(self, intent: RetrievalIntent, evidence_items: list[EvidenceItem]) -> bool:
        """判断当前证据质量是否过低，是否需要放宽条件重试。"""
        if not self.enable_agentic_retry:
            return False
        if intent.retry_count >= 1:
            return False
        if not evidence_items:
            return True
        if len(evidence_items) < self.low_confidence_min_evidence:
            return True

        avg_score = sum(item.rerank_score for item in evidence_items) / len(evidence_items)
        if avg_score < self.low_confidence_avg_score:
            return True
        if intent.required_scene and not any(item.metadata.get("scene_hint") == intent.required_scene for item in evidence_items):
            return True
        if intent.attribute_groups and not any(self._matches_attributes(intent, item) for item in evidence_items):
            return True
        return False

    def _build_retry_intent(self, intent: RetrievalIntent) -> RetrievalIntent:
        """基于当前检索意图构造一份更宽松的重试版本。"""
        retry_intent = deepcopy(intent)
        retry_intent.retry_count += 1
        retry_variants = list(retry_intent.query_variants)

        if retry_intent.character and retry_intent.attribute_terms:
            retry_variants.append(f"{retry_intent.character} 描写 {' '.join(retry_intent.attribute_terms)}")
            retry_variants.append(f"{retry_intent.character} 相关片段 {' '.join(retry_intent.attribute_terms)}")
        elif retry_intent.character:
            retry_variants.append(f"{retry_intent.character} 相关描写")

        if retry_intent.required_scene == "daily":
            retry_variants.append(f"{retry_intent.character or retry_intent.raw_query} 平时 生活")
        elif retry_intent.required_scene == "battle":
            retry_variants.append(f"{retry_intent.character or retry_intent.raw_query} 战斗 交手")

        retry_intent.required_scene = None
        retry_intent.query_variants = self._dedupe_preserve_order(retry_variants)[: self.max_query_variants]
        return retry_intent

    def _score_evidence_set(self, evidence_items: list[EvidenceItem]) -> float:
        """为一组证据计算整体质量分，用于比较首轮与重试结果。"""
        return sum(item.rerank_score for item in evidence_items) + len(evidence_items) * 0.2

    def retrieve_baseline_evidence(self, query: str) -> list[EvidenceItem]:
        """执行不带复杂约束的基础检索流程，供评测对比。"""
        docs = self.retriever.invoke(query)
        ranked_docs = self.do_rerank(query, docs)
        intent = RetrievalIntent(raw_query=query, query_variants=[query])
        evidence_items = self._build_evidence_items(intent, ranked_docs)

        filtered = [item for item in evidence_items if item.rerank_score >= self.relevance_score]
        if not filtered and evidence_items:
            filtered = [evidence_items[0]]
        return filtered[: self.final_k]

    def retrieve_evidence(self, intent: RetrievalIntent, allow_retry: bool = True) -> list[EvidenceItem]:
        """执行增强检索流程，并在低置信度时进行一次重试。"""
        first_pass = self._retrieve_once(intent, use_scene_filter=True)
        if not allow_retry or not self._should_retry(intent, first_pass):
            return first_pass

        retry_intent = self._build_retry_intent(intent)
        second_pass = self._retrieve_once(retry_intent, use_scene_filter=False)
        if self._score_evidence_set(second_pass) > self._score_evidence_set(first_pass):
            return second_pass
        return first_pass

    def retriever_docs(self, query: str) -> list[Document]:
        """对外返回检索到的文档列表，供上层直接消费。"""
        intent = self.analyze_query(query)
        evidence_items = self.retrieve_evidence(intent)
        return [Document(page_content=item.content, metadata=item.metadata) for item in evidence_items]

    def _format_context(self, evidence_items: list[EvidenceItem]) -> str:
        """把最终证据格式化为可注入提示词的上下文文本。"""
        if not evidence_items:
            return "【检索结果】未找到足够匹配的参考资料，请不要补充原文中不存在的细节。"

        context_parts: list[str] = []
        for index, evidence in enumerate(evidence_items, start=1):
            metadata = evidence.metadata
            context_parts.append(
                "\n".join(
                    [
                        f"【参考资料{index}】",
                        evidence.content,
                        f">>>> 参考元数据：{metadata}",
                        f">>>> rerank_score：{evidence.rerank_score:.4f}",
                        f">>>> 命中原因：{', '.join(evidence.match_reasons) if evidence.match_reasons else '无'}",
                    ]
                )
            )
        return "\n".join(context_parts)

    def rag_retrieve_context(self, query: str) -> str:
        """对外提供完整的 RAG 上下文构造入口。"""
        intent = self.analyze_query(query)
        evidence_items = self.retrieve_evidence(intent)
        context = self._format_context(evidence_items)
        return self.chain.invoke({"input": query, "context": context})

    def _load_eval_samples(self, dataset: str | list[dict[str, Any]]) -> list[EvalSample]:
        """读取检索评测样本，支持路径和内存对象两种输入。"""
        if isinstance(dataset, str):
            dataset_path = get_abs_path(dataset)
            with open(dataset_path, "r", encoding="utf-8") as file_obj:
                raw_samples = yaml.load(file_obj, Loader=yaml.FullLoader)
        else:
            raw_samples = dataset

        samples: list[EvalSample] = []
        for item in raw_samples or []:
            samples.append(
                EvalSample(
                    query=item["query"],
                    target_character=item.get("target_character"),
                    required_scene=item.get("required_scene"),
                    required_attributes=item.get("required_attributes", []),
                    gold_spans=item.get("gold_spans", []),
                    gold_sources=item.get("gold_sources", []),
                    expect_no_answer=item.get("expect_no_answer", False),
                )
            )
        return samples

    def _evidence_matches_sample(self, evidence: EvidenceItem, sample: EvalSample) -> bool:
        """判断一条证据是否命中评测样本的 gold 条件。"""
        if sample.expect_no_answer:
            return False

        if sample.target_character and sample.target_character not in evidence.content and sample.target_character not in evidence.metadata.get("character_mentions", ""):
            return False

        if sample.gold_spans and any(span in evidence.content for span in sample.gold_spans):
            return True

        source = str(evidence.metadata.get("source", ""))
        chapter = str(evidence.metadata.get("chapter", ""))
        for gold_source in sample.gold_sources:
            if gold_source in source or gold_source == chapter:
                return True
        return False

    def _compute_ndcg(self, relevances: list[int], k: int = 3) -> float:
        """计算前 k 条结果的 nDCG 指标。"""
        truncated = relevances[:k]
        dcg = sum(rel / log2(index + 2) for index, rel in enumerate(truncated))
        ideal = sorted(relevances, reverse=True)[:k]
        idcg = sum(rel / log2(index + 2) for index, rel in enumerate(ideal))
        if idcg == 0:
            return 0.0
        return dcg / idcg

    def _evaluate_sample(self, sample: EvalSample, evidence_items: list[EvidenceItem], mode: str) -> EvalResult:
        """计算单条评测样本在当前模式下的检索指标。"""
        top5 = evidence_items[:5]
        relevances = [1 if self._evidence_matches_sample(item, sample) else 0 for item in top5]
        top3_hit = 1 if any(relevances[:3]) else 0
        top3_precision = sum(relevances[:3]) / 3
        ndcg3 = self._compute_ndcg(relevances, k=3)
        tp = fp = tn = fn = 0

        if sample.expect_no_answer:
            if evidence_items:
                fp = 1
            else:
                tn = 1
        elif top3_hit:
            tp = 1
        else:
            fn = 1

        return EvalResult(
            query=sample.query,
            mode=mode,
            retrieved_count=len(evidence_items),
            top3_hit=top3_hit,
            top3_precision=top3_precision,
            ndcg3=ndcg3,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
        )

    def evaluate_retrieval(self, dataset: str | list[dict[str, Any]], mode: str = "enhanced_retry") -> EvalReport:
        """运行整套检索评测并汇总为报告。"""
        samples = self._load_eval_samples(dataset)
        results: list[EvalResult] = []

        for sample in samples:
            if mode == "baseline":
                evidence_items = self.retrieve_baseline_evidence(sample.query)
            elif mode == "enhanced":
                evidence_items = self.retrieve_evidence(self.analyze_query(sample.query), allow_retry=False)
            elif mode == "enhanced_retry":
                evidence_items = self.retrieve_evidence(self.analyze_query(sample.query), allow_retry=True)
            else:
                raise ValueError(f"unknown evaluation mode: {mode}")

            results.append(self._evaluate_sample(sample, evidence_items, mode))

        sample_count = len(results) or 1
        metrics = {
            "Recall@3": sum(result.top3_hit for result in results) / sample_count,
            "Precision@3": sum(result.top3_precision for result in results) / sample_count,
            "nDCG@3": sum(result.ndcg3 for result in results) / sample_count,
            "TP": sum(result.tp for result in results),
            "FP": sum(result.fp for result in results),
            "TN": sum(result.tn for result in results),
            "FN": sum(result.fn for result in results),
        }

        logger.info(f"[RAG-EVAL][{mode}] {metrics}")
        return EvalReport(mode=mode, sample_count=len(results), metrics=metrics, results=results)


if __name__ == '__main__':
    rag = RagSummarizeService()
    print(rag.rag_retrieve_context("没有打斗时，关于青灵的外貌描写"))
