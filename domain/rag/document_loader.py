"""
문서 로더 모듈

PDF 파일로부터 전력시장운영규칙을 로드합니다.
pymupdf4llm을 사용하여 PDF를 마크다운으로 변환합니다.
"""

import pymupdf4llm
from pathlib import Path
from typing import List, Dict
import logging
from config.settings import get_settings

logger = logging.getLogger(__name__)


class DocumentLoader:
    """PDF 문서 로더"""
    
    def __init__(self):
        """문서 로더 초기화"""
        settings = get_settings()
        self.pdf_path = Path(settings.pdf_path)
    
    def load_pdfs(self) -> List[Dict]:
        """
        PDF 디렉토리의 모든 파일을 로드
        
        Returns:
            List[Dict]: 문서 리스트 [{"filename": str, "content": str}]
        """
        if not self.pdf_path.exists():
            logger.warning(f"PDF path not found: {self.pdf_path}")
            return []
        
        documents = []
        pdf_files = list(self.pdf_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_path}")
            return []
        
        for pdf_file in pdf_files:
            try:
                doc = self._load_pdf(pdf_file)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to load PDF {pdf_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(documents)} PDF files")
        return documents
    
    def _load_pdf(self, pdf_path: Path) -> Dict:
        """
        단일 PDF 파일 로드
        
        Args:
            pdf_path (Path): PDF 파일 경로
            
        Returns:
            Dict: {"filename": str, "content": str}
        """
        try:
            # pymupdf4llm을 사용하여 PDF를 마크다운으로 변환
            md_text = pymupdf4llm.to_markdown(str(pdf_path))
            
            return {
                "filename": pdf_path.name,
                "content": md_text
            }
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
            return None
    
    def load_pdf_by_name(self, filename: str) -> Dict:
        """
        파일명으로 특정 PDF 로드
        
        Args:
            filename (str): PDF 파일명
            
        Returns:
            Dict: 문서 정보 또는 None
        """
        pdf_file = self.pdf_path / filename
        
        if not pdf_file.exists():
            logger.error(f"PDF file not found: {pdf_file}")
            return None
        
        return self._load_pdf(pdf_file)


def detect_document_type(filename: str, extracted_text: str) -> str:
    """
    파일명과 추출 텍스트 첫 500자로 문서 종류를 자동 분류한다.

    Args:
        filename (str): PDF 파일명
        extracted_text (str): PDF에서 추출한 텍스트

    Returns:
        str: "전력시장운영규칙" | "정산세부규정" | "기타"
    """
    keywords = {
        "전력시장운영규칙": ["전력시장운영규칙", "전력시장 운영규칙"],
        "정산세부규정": ["정산세부규정", "정산 세부규정"],
    }
    target = filename + extracted_text[:500]
    for doc_type, kws in keywords.items():
        if any(kw in target for kw in kws):
            return doc_type
    return "기타"


def get_document_loader() -> DocumentLoader:
    """
    문서 로더 인스턴스 반환

    Returns:
        DocumentLoader: 문서 로더 인스턴스
    """
    return DocumentLoader()
