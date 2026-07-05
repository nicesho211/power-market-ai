"""
규정 개정 이력 관리자

조문별 개정 이력을 추적하고 조회합니다.
"""

from typing import Dict, List, Optional
import logging
from domain.rag.retriever import get_retriever
import json

logger = logging.getLogger(__name__)


class HistoryManager:
    """규정 개정 이력 관리"""
    
    def __init__(self):
        """이력 관리자 초기화"""
        self.retriever = get_retriever()
    
    def get_history(self, regulation_num: str) -> Dict:
        """
        조문의 개정 이력 조회
        
        Args:
            regulation_num (str): 조문번호 (예: "제2.4.1조")
            
        Returns:
            Dict: 개정 이력
            {
                "regulation_num": str,
                "history": [
                    {
                        "버전": str,
                        "시행일": str,
                        "개정유형": str,
                        "변경요지": str
                    },
                    ...
                ]
            }
        """
        try:
            # 해당 조문의 모든 버전 검색 (필터 없음 - 모든 버전)
            results = self.retriever.search_by_regulation_number(regulation_num)
            
            if not results:
                return {
                    "regulation_num": regulation_num,
                    "found": False,
                    "history": []
                }
            
            # 메타데이터에서 개정이력 추출
            history_data = self._extract_histories(results)
            
            return {
                "regulation_num": regulation_num,
                "found": True,
                "history": history_data
            }
        except Exception as e:
            logger.error(f"Failed to get history for {regulation_num}: {e}")
            return {
                "regulation_num": regulation_num,
                "found": False,
                "history": [],
                "error": str(e)
            }
    
    def _extract_histories(self, results: List[Dict]) -> List[Dict]:
        """
        검색 결과에서 이력 정보 추출
        
        Args:
            results (List[Dict]): 검색 결과
            
        Returns:
            List[Dict]: 이력 리스트 (시간 순서로 정렬)
        """
        histories = []
        
        for result in results:
            metadata = result.get("metadata", {})
            
            # 개정이력 파싱
            revision_history = metadata.get("개정이력", "[]")
            if isinstance(revision_history, str):
                try:
                    revision_history = json.loads(revision_history)
                except:
                    revision_history = []
            
            # 각 이력 항목 추가
            if isinstance(revision_history, list):
                for item in revision_history:
                    if isinstance(item, dict):
                        histories.append({
                            "버전": item.get("버전", metadata.get("버전", "")),
                            "시행일": item.get("시행일", ""),
                            "개정유형": item.get("유형", metadata.get("개정유형", "유지")),
                            "변경요지": item.get("요지", "")
                        })
        
        # 버전순으로 정렬 (역순)
        histories.sort(key=lambda x: x.get("버전", ""), reverse=True)
        
        return histories
    
    def get_revision_count(self, regulation_num: str) -> int:
        """
        조문의 개정 횟수 조회
        
        Args:
            regulation_num (str): 조문번호
            
        Returns:
            int: 개정 횟수
        """
        history = self.get_history(regulation_num)
        return len(history.get("history", []))
    
    def get_latest_revision_date(self, regulation_num: str) -> Optional[str]:
        """
        조문의 최신 개정일 조회
        
        Args:
            regulation_num (str): 조문번호
            
        Returns:
            Optional[str]: 최신 개정일 (YYYY-MM-DD) 또는 None
        """
        history = self.get_history(regulation_num)
        histories = history.get("history", [])
        
        if histories:
            return histories[0].get("시행일", "")
        return None
    
    def compare_amendments(
        self,
        regulation_num: str,
        version1: str,
        version2: str
    ) -> Dict:
        """
        두 버전의 개정 내용 비교
        
        Args:
            regulation_num (str): 조문번호
            version1 (str): 첫 번째 버전 (YYYY-MM-DD)
            version2 (str): 두 번째 버전 (YYYY-MM-DD)
            
        Returns:
            Dict: 비교 결과
        """
        # 추후 확장 - 현재는 구조만 정의
        return {
            "regulation_num": regulation_num,
            "version1": version1,
            "version2": version2,
            "differences": []
        }


def get_history_manager() -> HistoryManager:
    """
    이력 관리자 인스턴스 반환
    
    Returns:
        HistoryManager: 이력 관리자 인스턴스
    """
    return HistoryManager()
