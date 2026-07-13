"""
검색기 모듈 (Hybrid Search)

벡터 유사도(60%) + BM25 키워드(40%) 하이브리드 검색을 수행합니다.
EnsembleRetriever로 두 결과를 RRF 방식으로 병합합니다.
"""

from typing import List, Dict, Optional, Any
import json
import logging
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from domain.rag.vector_store import get_vector_store, VectorStore
from config.settings import get_settings

logger = logging.getLogger("RAG")

# BM25 문서 풀 캐시 (filter JSON → list[dict])
_bm25_cache: Dict[str, List[Dict]] = {}


class _VectorStoreRetriever(BaseRetriever):
    """VectorStore를 LangChain BaseRetriever로 래핑"""

    vector_store: Any
    where_filter: dict
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        results = self.vector_store.search(
            query=query, top_k=self.k, where=self.where_filter
        )
        return [
            Document(page_content=r["document"], metadata=r["metadata"])
            for r in results
        ]


def _load_bm25_docs(
    vector_store: VectorStore, where_filter: Optional[dict]
) -> List[Dict]:
    """필터에 맞는 전체 문서를 Qdrant에서 로드 (캐시 적용)."""
    cache_key = json.dumps(where_filter or {}, sort_keys=True, ensure_ascii=False)
    if cache_key in _bm25_cache:
        return _bm25_cache[cache_key]

    from qdrant_client.models import Filter
    from domain.rag.vector_store import _build_qdrant_filter, get_qdrant_client
    from config.settings import get_settings

    settings = get_settings()
    # 싱글톤 client 재사용 — 새 인스턴스 생성 금지
    client = get_qdrant_client()
    name = settings.QDRANT_COLLECTION_NAME
    qdrant_filter = _build_qdrant_filter(where_filter)

    docs: List[Dict] = []
    offset = None
    try:
        while True:
            results, next_offset = client.scroll(
                collection_name=name,
                scroll_filter=qdrant_filter,
                with_payload=True,
                with_vectors=False,
                limit=1000,
                offset=offset,
            )
            for point in results:
                payload = dict(point.payload)
                text = payload.pop("document", "")
                payload.pop("_original_id", None)
                if text:
                    docs.append({"document": text, "metadata": payload})
            if next_offset is None:
                break
            offset = next_offset
    except Exception as e:
        logger.error(f"BM25 문서 로드 실패: {e}")

    _bm25_cache[cache_key] = docs
    logger.info(f"[RAG] BM25 문서 풀 로드: {len(docs)}개 (filter={cache_key[:50]})")
    return docs


def _docs_to_hybrid_results(docs: List[Document]) -> List[Dict]:
    """LangChain Document 목록을 기존 dict 포맷으로 변환"""
    return [
        {"document": d.page_content, "metadata": d.metadata, "distance": 0.5}
        for d in docs
    ]


class Retriever:
    """Hybrid Search 기반 문서 검색 관리"""

    def __init__(self):
        self.vector_store = get_vector_store()
        settings = get_settings()
        self.top_k = settings.RETRIEVAL_TOP_K

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict] = None,
        document_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Hybrid Search (벡터 60% + BM25 40%)

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수 (None이면 설정값 사용)
            where: ChromaDB 스타일 필터
            document_type: 문서 종류 한정 필터
        """
        if not query.strip():
            return []

        k = top_k or self.top_k

        if where is None:
            where = {"is_latest": True}

        if document_type:
            if "$and" in where:
                where["$and"].append({"document_type": document_type})
            else:
                where = {"$and": [where, {"document_type": document_type}]}

        logger.info(f"[RAG] Hybrid Search 시작 | query={query[:50]} | filter={where}")

        try:
            # ── 벡터 검색 Retriever ─────────────────────────────────────
            vector_retriever = _VectorStoreRetriever(
                vector_store=self.vector_store,
                where_filter=where,
                k=k,
            )

            # ── BM25 Retriever ──────────────────────────────────────────
            bm25_docs = _load_bm25_docs(self.vector_store, where)
            if not bm25_docs:
                # 문서 풀 없으면 벡터 검색만
                results = self.vector_store.search(query, top_k=k, where=where)
                logger.info(f"[RAG] Vector-only search (BM25 풀 없음) | {len(results)}건")
                return results

            bm25_retriever = BM25Retriever.from_texts(
                texts=[d["document"] for d in bm25_docs],
                metadatas=[d["metadata"] for d in bm25_docs],
                k=k,
            )
            logger.info(f"[RAG] BM25 검색 준비 완료 | 후보 {len(bm25_docs)}건")

            # ── Ensemble (RRF 병합) ─────────────────────────────────────
            ensemble = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.6, 0.4],
            )
            lc_results = ensemble.invoke(query)[:k]
            results = _docs_to_hybrid_results(lc_results)
            logger.info(f"[RAG] Hybrid 통합 결과 | {len(results)}건")
            if results:
                top_meta = results[0].get("metadata", {})
                logger.info(
                    f"[RAG] 상위 조문: {top_meta.get('조문번호', '없음')} | 페이지: {top_meta.get('페이지', 0)}"
                )
            return results

        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            # 폴백: 순수 벡터 검색
            return self.vector_store.search(query, top_k=k, where=where)

    def search_history(
        self,
        query: str,
        regulation_num: str,
        document_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """전체 이력 검색 — 특정 조문의 모든 버전을 시간순으로 반환"""
        filters: Dict = {"조문번호": regulation_num}
        if document_type:
            filters = {
                "$and": [
                    {"조문번호": regulation_num},
                    {"document_type": document_type},
                ]
            }

        try:
            results = self.vector_store.search(
                query=query,
                top_k=top_k or 50,
                where=filters,
            )
            results.sort(
                key=lambda x: x.get("metadata", {}).get("버전", ""),
                reverse=True,
            )
            return results
        except Exception as e:
            logger.error(f"History search error: {e}")
            return []

    def search_by_regulation_number(self, regulation_num: str) -> List[Dict]:
        """조문번호로 정확히 해당 조문만 검색.

        벡터 유사도 검색이 아닌 Qdrant 메타데이터 정확 필터(조문번호 exact match)를
        사용한다. is_latest 제한 없이 모든 버전에서 조회해 개정 이력 조회에도
        사용할 수 있게 한다.
        """
        import re
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        from domain.rag.vector_store import get_qdrant_client

        normalized = re.sub(r"\s+", "", regulation_num)
        settings = get_settings()
        client = get_qdrant_client()
        name = settings.QDRANT_COLLECTION_NAME

        qdrant_filter = Filter(
            must=[FieldCondition(key="조문번호", match=MatchValue(value=normalized))]
        )

        docs: List[Dict] = []
        offset = None
        try:
            while True:
                points, next_offset = client.scroll(
                    collection_name=name,
                    scroll_filter=qdrant_filter,
                    with_payload=True,
                    with_vectors=False,
                    limit=100,
                    offset=offset,
                )
                for point in points:
                    payload = dict(point.payload)
                    text = payload.pop("document", "")
                    payload.pop("_original_id", None)
                    docs.append({"document": text, "metadata": payload})
                if next_offset is None:
                    break
                offset = next_offset
        except Exception as e:
            logger.error(f"Search by regulation number failed: {e}")
            return []

        docs.sort(key=lambda d: d["metadata"].get("버전", ""))
        return docs

    def search_with_filters(
        self,
        query: str,
        version: Optional[str] = None,
        chapter: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """필터를 포함한 고급 검색"""
        filters: Dict = {}
        if version:
            filters["버전"] = version
        if chapter:
            filters["장번호"] = chapter

        try:
            return self.vector_store.search(
                query=query,
                top_k=top_k or self.top_k,
                where=filters if filters else None,
            )
        except Exception as e:
            logger.error(f"Filtered search failed: {e}")
            return []

    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """문서 ID로 특정 문서 조회"""
        try:
            return self.vector_store.get_by_id(doc_id)
        except Exception as e:
            logger.error(f"Failed to get document by id: {e}")
            return None


def get_retriever() -> Retriever:
    """검색기 인스턴스 반환"""
    return Retriever()
