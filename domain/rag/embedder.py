"""
텍스트 임베딩 모듈

OpenAI의 text-embedding-3-large를 사용합니다.
대형 문서 인덱싱을 위해 배치 처리 함수를 제공합니다.
"""

from infrastructure.llm_client import get_embeddings
from typing import List, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class Embedder:
    """텍스트 임베딩 관리 클래스"""

    def __init__(self):
        self.embeddings = get_embeddings()

    def embed_text(self, text: str) -> List[float]:
        """단일 텍스트 임베딩"""
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트를 한 번의 API 호출로 임베딩"""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"Failed to embed documents: {e}")
            raise

    def get_dimension(self) -> int:
        return 3072  # text-embedding-3-large


def get_embedder() -> Embedder:
    return Embedder()


def embed_texts_batch(
    texts: List[str],
    batch_size: int = 100,
    progress_callback: Optional[Callable] = None,
) -> List[List[float]]:
    """
    대형 텍스트 목록을 batch_size 단위로 나눠 임베딩.
    429/토큰 초과를 방지하고 진행률을 콜백으로 보고한다.

    Args:
        texts: 임베딩할 텍스트 목록
        batch_size: 한 번에 전송할 최대 텍스트 수 (기본 100)
        progress_callback: (step, current, total, message) → None

    Returns:
        List[List[float]]: 임베딩 벡터 목록 (texts와 동일한 순서)
    """
    embeddings_model = get_embeddings()
    all_vectors: List[List[float]] = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        try:
            vectors = embeddings_model.embed_documents(batch)
            all_vectors.extend(vectors)
        except Exception as e:
            logger.error(f"배치 임베딩 실패 ({i}~{i+len(batch)}): {e}")
            # 빈 벡터로 채워서 인덱스 정합성 유지
            all_vectors.extend([[0.0] * 3072] * len(batch))

        done = min(i + batch_size, total)
        if progress_callback:
            progress_callback(
                "embed",
                done,
                total,
                f"임베딩 중... {done}/{total}개 ({done/total:.0%})",
            )

    return all_vectors
