"""
벡터 데이터베이스 관리 (Qdrant)

전력시장운영규칙 조문을 벡터로 저장하고 유사도 검색을 수행합니다.
로컬(파일)/클라우드 자동 전환, 재인덱싱 방지, 배치 처리 지원.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from pathlib import Path
from functools import lru_cache
import json
import logging
import uuid
from typing import List, Dict, Optional
from config.settings import get_settings
from domain.rag.embedder import get_embedder

logger = logging.getLogger(__name__)


def _str_to_point_id(s: str) -> str:
    """문자열 ID를 Qdrant UUID로 결정론적 변환"""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))


# search_filter에 포함될 수 있는 내부 메타 키 — Qdrant 페이로드에 없으므로 필터에서 제외
_INTERNAL_FILTER_KEYS = {"mode"}


def _build_qdrant_filter(where: Optional[Dict]) -> Optional[Filter]:
    """ChromaDB 스타일 where 조건을 Qdrant Filter로 변환.
    내부 메타 키(mode 등)는 Qdrant 페이로드에 없으므로 자동으로 제외한다."""
    if not where:
        return None

    if "$and" in where:
        must = []
        for cond in where["$and"]:
            for key, value in cond.items():
                if not key.startswith("$") and key not in _INTERNAL_FILTER_KEYS:
                    must.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=must) if must else None

    must = []
    for key, value in where.items():
        if not key.startswith("$") and key not in _INTERNAL_FILTER_KEYS:
            must.append(FieldCondition(key=key, match=MatchValue(value=value)))
    return Filter(must=must) if must else None


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """로컬(path)/클라우드(url) 자동 전환 — 싱글톤 (로컬 파일 잠금 방지)"""
    settings = get_settings()
    if settings.QDRANT_URL:
        return QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
        )
    Path(settings.QDRANT_DB_PATH).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=settings.QDRANT_DB_PATH)


def ensure_collection() -> None:
    """컬렉션이 있으면 재사용, 없으면 생성 (재인덱싱 방지 핵심)"""
    settings = get_settings()
    client = get_qdrant_client()
    name = settings.QDRANT_COLLECTION_NAME

    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=settings.VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info(f"[Qdrant] 새 컬렉션 생성: {name}")
        print(f"[Qdrant] 새 컬렉션 생성: {name}")
    else:
        count = client.count(name).count
        logger.info(f"[Qdrant] 기존 컬렉션 재사용: {count}개 청크")
        print(f"[Qdrant] 기존 컬렉션 재사용: {count}개 청크")


def get_collection_stats() -> dict:
    """총 청크 수, 버전 목록, 최신 버전 반환"""
    settings = get_settings()
    client = get_qdrant_client()
    name = settings.QDRANT_COLLECTION_NAME

    try:
        existing = [c.name for c in client.get_collections().collections]
        if name not in existing:
            return {"total_chunks": 0, "versions": [], "latest_version": ""}

        total = client.count(name).count
        if total == 0:
            return {"total_chunks": 0, "versions": [], "latest_version": ""}

        versions: set = set()
        latest_version = ""
        offset = None

        while True:
            results, next_offset = client.scroll(
                collection_name=name,
                with_payload=["버전", "is_latest"],
                with_vectors=False,
                limit=1000,
                offset=offset,
            )
            for point in results:
                v = point.payload.get("버전", "")
                if v:
                    versions.add(v)
                if point.payload.get("is_latest") and v:
                    latest_version = v
            if next_offset is None:
                break
            offset = next_offset

        return {
            "total_chunks": total,
            "versions": sorted(versions, reverse=True),
            "latest_version": latest_version,
        }
    except Exception as e:
        logger.error(f"Failed to get collection stats: {e}")
        return {"total_chunks": 0, "versions": [], "latest_version": ""}


class VectorStore:
    """Qdrant 벡터 데이터베이스 관리"""

    def __init__(self):
        settings = get_settings()
        # get_qdrant_client()는 lru_cache 싱글톤 — 로컬 파일 잠금 충돌 없음
        self.client = get_qdrant_client()
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.embedder = get_embedder()
        ensure_collection()  # 이미 같은 싱글톤 client 재사용

    def add_documents_with_embeddings(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        ids: List[str],
        metadatas: List[Dict],
        batch_size: int = 100,
    ) -> int:
        """사전 계산된 임베딩으로 Qdrant에 저장. batch_size개씩 배치 처리."""
        added = 0
        for start in range(0, len(documents), batch_size):
            end = min(start + batch_size, len(documents))
            points = []
            for i in range(start, end):
                payload = {
                    **metadatas[i],
                    "document": documents[i],
                    "_original_id": ids[i],
                }
                points.append(
                    PointStruct(
                        id=_str_to_point_id(ids[i]),
                        vector=embeddings[i],
                        payload=payload,
                    )
                )
            try:
                self.client.upsert(collection_name=self.collection_name, points=points)
                added += end - start
                logger.info(f"Qdrant upsert {end}/{len(documents)}")
            except Exception as e:
                logger.error(f"Qdrant 배치 저장 실패 ({start}~{end}): {e}", exc_info=True)
        return added

    def add_documents(
        self,
        documents: List[str],
        ids: List[str],
        metadatas: List[Dict],
    ) -> None:
        """소량 문서 추가 (임베딩 내부 생성)"""
        try:
            embs = self.embedder.embed_documents(documents)
            self.add_documents_with_embeddings(documents, embs, ids, metadatas)
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
    ) -> List[Dict]:
        """벡터 유사도 검색 (Qdrant) — query_points() 사용 (>=1.12)"""
        try:
            query_vector = self.embedder.embed_text(query)
            qdrant_filter = _build_qdrant_filter(where)

            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )
            hits = response.points

            formatted = []
            for hit in hits:
                payload = dict(hit.payload)
                doc_text = payload.pop("document", "")
                payload.pop("_original_id", None)

                # JSON 문자열 역직렬화
                parsed: Dict = {}
                for key, value in payload.items():
                    if isinstance(value, str) and (
                        value.startswith("[") or value.startswith("{")
                    ):
                        try:
                            parsed[key] = json.loads(value)
                        except Exception:
                            parsed[key] = value
                    else:
                        parsed[key] = value

                formatted.append(
                    {
                        "document": doc_text,
                        "metadata": parsed,
                        "distance": 1 - hit.score,  # cosine similarity → distance
                    }
                )
            return formatted
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_by_id(self, doc_id: str) -> Optional[Dict]:
        """원본 문자열 ID로 문서 조회"""
        try:
            point_id = _str_to_point_id(doc_id)
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
            if results:
                payload = dict(results[0].payload)
                doc_text = payload.pop("document", "")
                payload.pop("_original_id", None)
                return {"id": doc_id, "document": doc_text, "metadata": payload}
            return None
        except Exception as e:
            logger.error(f"Failed to get document by id: {e}")
            return None

    def update_latest_flag(self, new_version: str, document_type: str) -> None:
        """같은 document_type 내에서만 is_latest를 False로 전환"""
        try:
            filter_obj = Filter(
                must=[
                    FieldCondition(key="is_latest", match=MatchValue(value=True)),
                    FieldCondition(key="document_type", match=MatchValue(value=document_type)),
                ]
            )
            count_before = self.client.count(
                self.collection_name, count_filter=filter_obj
            ).count
            if count_before == 0:
                return

            self.client.set_payload(
                collection_name=self.collection_name,
                payload={"is_latest": False},
                points=filter_obj,
            )
            logger.info(
                f"is_latest → False for {count_before} chunks "
                f"(document_type={document_type}, new_version={new_version})"
            )
        except Exception as e:
            logger.error(f"Failed to update latest flag: {e}")
            raise

    def rollback_latest_flag(self, document_type: str) -> None:
        """저장 실패 시 이전 버전의 is_latest를 True로 복구"""
        try:
            filter_obj = Filter(
                must=[
                    FieldCondition(key="is_latest", match=MatchValue(value=False)),
                    FieldCondition(key="document_type", match=MatchValue(value=document_type)),
                ]
            )
            version_map: Dict[str, list] = {}
            offset = None

            while True:
                results, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_obj,
                    with_payload=["버전"],
                    with_vectors=False,
                    limit=1000,
                    offset=offset,
                )
                for point in results:
                    v = point.payload.get("버전", "")
                    if v:
                        version_map.setdefault(v, []).append(point.id)
                if next_offset is None:
                    break
                offset = next_offset

            if not version_map:
                return

            latest = sorted(version_map.keys(), reverse=True)[0]
            rollback_ids = version_map[latest]
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={"is_latest": True},
                points=rollback_ids,
            )
            logger.info(f"[Rollback] is_latest 복구: {latest} ({len(rollback_ids)}개)")
        except Exception as e:
            logger.error(f"[Rollback] 실패: {e}")

    def get_all_versions(self, document_type: str) -> List[str]:
        """특정 문서 종류의 모든 버전 목록 반환 (시간 역순)"""
        try:
            filter_obj = Filter(
                must=[FieldCondition(key="document_type", match=MatchValue(value=document_type))]
            )
            versions: set = set()
            offset = None

            while True:
                results, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_obj,
                    with_payload=["버전"],
                    with_vectors=False,
                    limit=1000,
                    offset=offset,
                )
                for point in results:
                    v = point.payload.get("버전", "")
                    if v:
                        versions.add(v)
                if next_offset is None:
                    break
                offset = next_offset

            return sorted(versions, reverse=True)
        except Exception as e:
            logger.error(f"Failed to get versions: {e}")
            return []

    def get_collection_stats(self) -> Dict:
        """컬렉션 통계 반환"""
        try:
            count = self.client.count(self.collection_name).count
            return {"total_documents": count}
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"total_documents": 0}

    def delete_documents(self, ids: List[str]) -> None:
        """문서 삭제"""
        try:
            point_ids = [_str_to_point_id(doc_id) for doc_id in ids]
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids,
            )
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise

    def update_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict],
    ) -> None:
        """문서 업데이트 (삭제 후 재추가)"""
        try:
            self.delete_documents(ids)
            self.add_documents(documents, ids, metadatas)
        except Exception as e:
            logger.error(f"Failed to update documents: {e}")
            raise


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """벡터 스토어 인스턴스 반환 (싱글톤)"""
    return VectorStore()


def reset_collection() -> None:
    """Qdrant DB 디렉토리 전체 삭제 후 재생성. 싱글톤 캐시도 함께 초기화.

    Qdrant 로컬 모드에서는 delete_collection()이 디스크 데이터를 즉시 정리하지
    않으므로 qdrant_db 디렉토리를 직접 삭제하는 방식을 사용한다.
    """
    import shutil
    settings = get_settings()

    # ① 싱글톤 캐시 무효화 (클라이언트/스토어 모두)
    get_qdrant_client.cache_clear()
    get_vector_store.cache_clear()

    # ② 디렉토리 삭제 (클라우드 모드에서는 컬렉션만 삭제)
    if not settings.QDRANT_URL:
        db_path = Path(settings.QDRANT_DB_PATH)
        if db_path.exists():
            shutil.rmtree(db_path)
            logger.info(f"[Qdrant] DB 디렉토리 삭제: {db_path}")
    else:
        # 클라우드: client API로 컬렉션 삭제
        client = get_qdrant_client()
        existing = [c.name for c in client.get_collections().collections]
        if settings.QDRANT_COLLECTION_NAME in existing:
            client.delete_collection(settings.QDRANT_COLLECTION_NAME)

    # ③ 컬렉션 재생성 (캐시 클리어 후 새 클라이언트로)
    ensure_collection()
    logger.info("[Qdrant] 컬렉션 초기화 완료")


def get_versions_by_document_type(document_type: str) -> List[Dict]:
    """특정 document_type의 버전 목록을 날짜 오름차순으로 반환.

    반환 형식:
        [{"버전": "2025-02-12", "is_latest": False, "chunk_count": 850},
         {"버전": "2026-05-20", "is_latest": True,  "chunk_count": 900}]
    """
    from collections import Counter
    settings = get_settings()
    client = get_qdrant_client()
    name = settings.QDRANT_COLLECTION_NAME

    try:
        existing = [c.name for c in client.get_collections().collections]
        if name not in existing:
            return []

        all_points: List = []
        offset = None
        filt = Filter(must=[FieldCondition(key="document_type", match=MatchValue(value=document_type))])
        while True:
            batch, next_offset = client.scroll(
                collection_name=name,
                scroll_filter=filt,
                limit=1000,
                offset=offset,
                with_payload=["버전", "is_latest"],
                with_vectors=False,
            )
            all_points.extend(batch)
            if next_offset is None:
                break
            offset = next_offset

        version_counts: Counter = Counter(p.payload.get("버전", "") for p in all_points)
        latest_map: Dict[str, bool] = {}
        for p in all_points:
            v = p.payload.get("버전", "")
            if p.payload.get("is_latest"):
                latest_map[v] = True
            elif v not in latest_map:
                latest_map[v] = False

        versions = [
            {"버전": v, "is_latest": latest_map.get(v, False), "chunk_count": cnt}
            for v, cnt in version_counts.items() if v
        ]
        versions.sort(key=lambda x: x["버전"])
        return versions
    except Exception as e:
        logger.error(f"get_versions_by_document_type 실패: {e}")
        return []
