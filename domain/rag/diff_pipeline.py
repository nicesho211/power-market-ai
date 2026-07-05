"""
규정 개정 비교 파이프라인

개정 전후의 규정을 비교하여 변경 사항을 분석합니다.
"""

from typing import Dict, List, Tuple
import logging
from domain.rag.retriever import get_retriever

logger = logging.getLogger(__name__)


class DiffPipeline:
    """규정 개정 비교"""
    
    def __init__(self):
        """비교 파이프라인 초기화"""
        self.retriever = get_retriever()
    
    def search_regulation_versions(
        self,
        regulation_num: str,
        current_version: str,
        previous_version: str
    ) -> Tuple[Dict, Dict]:
        """
        규정의 현재 버전과 이전 버전 검색
        
        Args:
            regulation_num (str): 조문번호 (예: "제2.4.1조")
            current_version (str): 현재 버전 (YYYY-MM-DD)
            previous_version (str): 이전 버전 (YYYY-MM-DD)
            
        Returns:
            Tuple[Dict, Dict]: (현재 규정, 이전 규정)
        """
        try:
            # 현재 버전 검색
            current_results = self.retriever.search_with_filters(
                query=regulation_num,
                version=current_version,
                top_k=5
            )
            
            # 이전 버전 검색
            previous_results = self.retriever.search_with_filters(
                query=regulation_num,
                version=previous_version,
                top_k=5
            )
            
            current_doc = current_results[0] if current_results else None
            previous_doc = previous_results[0] if previous_results else None
            
            return current_doc, previous_doc
        except Exception as e:
            logger.error(f"Failed to search regulation versions: {e}")
            return None, None
    
    def compare_regulations(
        self,
        current: Dict,
        previous: Dict
    ) -> Dict:
        """
        두 규정을 비교
        
        Args:
            current (Dict): 현재 규정
            previous (Dict): 이전 규정
            
        Returns:
            Dict: 비교 결과
            {
                "regulation_num": str,
                "current_version": str,
                "previous_version": str,
                "current_content": str,
                "previous_content": str,
                "is_changed": bool,
                "change_type": str,  # 신설/수정/삭제/유지
                "key_changes": List[str],
                "summary": str
            }
        """
        if not current or not previous:
            return {
                "is_changed": False,
                "change_type": "유지",
                "key_changes": [],
                "summary": "비교 불가능한 구간입니다."
            }
        
        current_text = current.get("document", "")
        previous_text = previous.get("document", "")
        current_metadata = current.get("metadata", {})
        previous_metadata = previous.get("metadata", {})
        
        # 기본 정보
        regulation_num = current_metadata.get("조문번호", "")
        current_version = current_metadata.get("버전", "")
        previous_version = previous_metadata.get("버전", "")
        
        # 변경 여부 확인
        is_changed = current_text != previous_text
        
        # 변경 타입
        change_type = current_metadata.get("개정유형", "유지")
        
        # 주요 변경사항 추출 (간단한 diff 구현)
        key_changes = self._extract_key_changes(current_text, previous_text)
        
        return {
            "regulation_num": regulation_num,
            "current_version": current_version,
            "previous_version": previous_version,
            "current_content": current_text,
            "previous_content": previous_text,
            "is_changed": is_changed,
            "change_type": change_type,
            "key_changes": key_changes,
            "summary": self._generate_diff_summary(
                change_type, key_changes, regulation_num
            )
        }
    
    def _extract_key_changes(
        self,
        current: str,
        previous: str
    ) -> List[str]:
        """
        두 텍스트의 주요 변경사항 추출
        
        Args:
            current (str): 현재 텍스트
            previous (str): 이전 텍스트
            
        Returns:
            List[str]: 주요 변경사항 리스트
        """
        changes = []
        
        # 간단한 변경 감지
        current_lines = set(current.split('\n'))
        previous_lines = set(previous.split('\n'))
        
        # 추가된 라인
        added = current_lines - previous_lines
        for line in added:
            if line.strip():
                changes.append(f"추가: {line.strip()[:100]}")
        
        # 제거된 라인
        removed = previous_lines - current_lines
        for line in removed:
            if line.strip():
                changes.append(f"제거: {line.strip()[:100]}")
        
        return changes[:5]  # 최대 5개까지만
    
    def _generate_diff_summary(
        self,
        change_type: str,
        key_changes: List[str],
        regulation_num: str
    ) -> str:
        """
        개정 요약 생성
        
        Args:
            change_type (str): 개정 유형
            key_changes (List[str]): 주요 변경사항
            regulation_num (str): 조문번호
            
        Returns:
            str: 개정 요약
        """
        if change_type == "신설":
            return f"{regulation_num}이 새로 신설되었습니다."
        elif change_type == "수정":
            return f"{regulation_num}이 수정되었습니다. 주요 내용: {'; '.join(key_changes[:2])}"
        elif change_type == "삭제":
            return f"{regulation_num}이 삭제되었습니다."
        else:
            return f"{regulation_num}에 변경사항이 없습니다."


def get_diff_pipeline() -> DiffPipeline:
    """비교 파이프라인 인스턴스 반환"""
    return DiffPipeline()


def compare_versions(
    query: str,
    document_type: str = "전력시장운영규칙",
    latest_version: str = None,
    prev_version: str = None,
    top_k: int = 5,
) -> Dict:
    """저장된 버전 목록에서 최신/이전 버전을 자동 선택해 비교.

    버전이 1개뿐이면 비교 불가 안내를 담은 dict 반환.

    Returns:
        {"latest_version", "prev_version", "latest_docs", "prev_docs",
         "available_versions", "error"}
    """
    from domain.rag.vector_store import get_versions_by_document_type, get_vector_store

    version_list = get_versions_by_document_type(document_type)
    available_sorted = sorted(v["버전"] for v in version_list)

    if not available_sorted:
        return {
            "latest_version": None, "prev_version": None,
            "latest_docs": [], "prev_docs": [],
            "available_versions": [],
            "error": "인덱싱된 문서가 없습니다. PDF를 먼저 업로드해주세요.",
        }

    if latest_version is None:
        latest_version = available_sorted[-1]
    if prev_version is None:
        candidates = [v for v in available_sorted if v < latest_version]
        prev_version = candidates[-1] if candidates else None

    if prev_version is None:
        return {
            "latest_version": latest_version, "prev_version": None,
            "latest_docs": [], "prev_docs": [],
            "available_versions": available_sorted,
            "error": (
                f"비교할 이전 버전이 없습니다. "
                f"현재 저장된 버전: {', '.join(available_sorted)}. "
                "이전 버전 PDF를 업로드한 뒤 비교해주세요."
            ),
        }

    vs = get_vector_store()
    latest_docs = vs.search(query, top_k=top_k, where={"버전": latest_version})
    prev_docs   = vs.search(query, top_k=top_k, where={"버전": prev_version})

    return {
        "latest_version": latest_version,
        "prev_version": prev_version,
        "latest_docs": latest_docs,
        "prev_docs": prev_docs,
        "available_versions": available_sorted,
        "error": None,
    }
