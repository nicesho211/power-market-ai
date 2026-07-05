"""
텍스트 청킹 모듈

문서를 크기 제한이 있는 청크로 나누고 메타데이터를 추가합니다.
조문 기반으로 지능적으로 청킹합니다.
"""

import re
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class Chunker:
    """문서 청킹 관리"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 100, version: str = "2025-04-10"):
        """
        청커 초기화
        
        Args:
            chunk_size (int): 청크 최대 크기
            overlap (int): 청크 간 중복 크기
            version (str): 규정 버전 (YYYY-MM-DD 형식)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.version = version
    
    def chunk_document(self, content: str, filename: str) -> List[Dict]:
        """
        문서를 청크로 분할하고 메타데이터 추가
        
        Args:
            content (str): 문서 내용
            filename (str): 파일명
            
        Returns:
            List[Dict]: 청크 리스트 [{"text": str, "metadata": dict, "chunk_id": str}]
        """
        chunks = []
        
        # 조문 단위로 분할 시도 (제X.X.X조 패턴)
        article_pattern = r'(제\d+\.\d+\.\d+조)'
        articles = re.split(article_pattern, content)
        
        chunk_id = 0
        current_text = ""
        
        for i, segment in enumerate(articles):
            if re.match(r'제\d+\.\d+\.\d+조', segment):
                # 조문번호
                if current_text.strip():
                    # 이전 청크 저장
                    chunks.extend(self._create_chunks(
                        current_text, chunk_id, filename, ""
                    ))
                    chunk_id += len(chunks)
                
                # 새로운 조문 시작
                article_num = segment
                current_text = segment + "\n"
                
                # 다음 내용 추가
                if i + 1 < len(articles):
                    current_text += articles[i + 1]
                    i += 1
            else:
                current_text += segment
        
        # 마지막 청크 처리
        if current_text.strip():
            chunks.extend(self._create_chunks(
                current_text, chunk_id, filename, ""
            ))
        
        logger.info(f"Created {len(chunks)} chunks from {filename}")
        return chunks
    
    def _create_chunks(
        self,
        text: str,
        start_id: int,
        filename: str,
        article_num: str
    ) -> List[Dict]:
        """
        텍스트를 크기 제한으로 분할
        
        Args:
            text (str): 텍스트
            start_id (int): 시작 청크 ID
            filename (str): 파일명
            article_num (str): 조문번호
            
        Returns:
            List[Dict]: 청크 리스트
        """
        chunks = []
        
        # 단순 크기 기반 분할
        lines = text.split('\n')
        current_chunk = ""
        chunk_id = start_id
        
        for line in lines:
            if len(current_chunk) + len(line) > self.chunk_size and current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": self._extract_metadata(
                        current_chunk, filename, article_num
                    ),
                    "chunk_id": f"{filename}_{chunk_id:04d}"
                })
                current_chunk = line
                chunk_id += 1
            else:
                current_chunk += line + "\n"
        
        # 마지막 청크
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": self._extract_metadata(
                    current_chunk, filename, article_num
                ),
                "chunk_id": f"{filename}_{chunk_id:04d}"
            })
        
        return chunks
    
    def _extract_metadata(
        self,
        chunk_text: str,
        filename: str,
        article_num: str
    ) -> Dict:
        """
        청크에서 메타데이터 추출
        
        Args:
            chunk_text (str): 청크 텍스트
            filename (str): 파일명
            article_num (str): 조문번호
            
        Returns:
            Dict: 메타데이터
        """
        # 조문번호 추출 (제X.X.X조 패턴)
        article_match = re.search(r'(제\d+\.\d+\.\d+조)', chunk_text)
        조문번호 = article_match.group(1) if article_match else article_num
        
        # 장, 절 추출
        if 조문번호:
            parts = 조문번호.replace('제', '').replace('조', '').split('.')
            if len(parts) >= 2:
                장번호 = f"제{parts[0]}장"
                절번호 = f"제{parts[1]}절" if len(parts) > 1 else ""
            else:
                장번호 = ""
                절번호 = ""
        else:
            장번호 = ""
            절번호 = ""
        
        return {
            "조문번호": 조문번호,
            "장번호": 장번호,
            "절번호": 절번호,
            "장제목": "",  # 별도 추출 필요
            "조제목": "",
            "페이지": 0,
            "버전": self.version,
            "is_latest": True,
            "이전버전": None,
            "개정유형": "유지",  # 신설/수정/삭제/유지
            "개정이력": "[]",  # JSON 문자열
            "최초제정일": "2021-01-01",
            "개정횟수": 0,
            "키워드": ""
        }


def get_chunker(chunk_size: int = 500, overlap: int = 100) -> Chunker:
    """
    청커 인스턴스 반환
    
    Args:
        chunk_size (int): 청크 최대 크기
        overlap (int): 청크 간 중복 크기
        
    Returns:
        Chunker: 청커 인스턴스
    """
    return Chunker(chunk_size, overlap)
