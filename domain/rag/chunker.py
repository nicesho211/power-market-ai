"""
텍스트 청킹 모듈

문서를 크기 제한이 있는 청크로 나누고 메타데이터를 추가합니다.
조문 기반으로 지능적으로 청킹합니다.
"""

import re
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# PDF → 마크다운 변환 시 조문번호에 띄어쓰기가 삽입되는 경우가 있어 (예: "제 12.7.3.7 조")
# 숫자/점/글자 사이 공백을 모두 허용하는 패턴을 사용한다.
ARTICLE_PATTERN = r"제\s*\d+\s*\.\s*\d+(?:\s*\.\s*\d+)*\s*조"
CHAPTER_PATTERN = r"제\s*\d+\s*장"
SECTION_PATTERN = r"제\s*\d+\s*절"

# 표/산식이 많은 조문 구간에서 다음 조문 경계가 정규식에 잡히지 않아 하나의 조문에
# 청크가 과도하게 쌓이는 경우(과병합)를 막기 위한 상한. 초과분은 마지막 청크로 병합한다.
MAX_CHUNKS_PER_ARTICLE = 10

# 전력시장 전문용어 — 청크 텍스트에 등장하면 키워드로 자동 추출
POWER_TERMS = [
    "SMP", "계통한계가격", "급전순위", "발전량", "전력수요",
    "예비력", "용량요금", "정산", "입찰", "변동비", "고정비",
    "신재생", "원자력", "LNG", "유연탄", "태양광", "수력",
    "한전", "전력거래소", "KPX", "계통운영", "부하차단",
    "피크", "기저", "첨두", "최대수요", "최소수요",
    "전력시장", "직접구매", "소규모", "분산형",
    "가중평균", "한계발전기", "변동비용", "연료비",
    "공급능력", "공급예비력", "수요예측", "수요반응"
]


def _normalize_article_num(raw: str) -> str:
    """
    띄어쓰기가 섞인 조문번호를 표준 형식으로 정규화
    예: "제 12.7.3.7 조" → "제12.7.3.7조"
        "제 2 . 4 . 1 조" → "제2.4.1조"
    """
    normalized = re.sub(r'\s+', '', raw)
    return normalized


def _split_table_blocks(text: str) -> List[Tuple[str, str]]:
    """표/산식 블록("|"가 3개 이상인 줄이 연속되는 구간)을 일반 텍스트와 분리한다.
    표 블록은 크기 제한과 무관하게 하나의 청크로 유지해 중간에 잘리지 않게 한다."""
    lines = text.split("\n")
    blocks: List[Tuple[str, str]] = []
    current_block: List[str] = []
    in_table = False

    for line in lines:
        is_table_line = line.count("|") >= 3
        if is_table_line and not in_table:
            if current_block:
                blocks.append(("\n".join(current_block), "text"))
                current_block = []
            in_table = True
        elif not is_table_line and in_table:
            blocks.append(("\n".join(current_block), "table"))
            current_block = []
            in_table = False
        current_block.append(line)

    if current_block:
        block_type = "table" if in_table else "text"
        blocks.append(("\n".join(current_block), block_type))

    return blocks


def _extract_keywords(text: str) -> List[str]:
    """청크 텍스트에서 전력시장 전문용어 추출"""
    found = []
    for term in POWER_TERMS:
        if term in text:
            found.append(term)
    return found


def _extract_article_title(text: str, article_num: str) -> str:
    """
    조문번호 다음 줄의 제목 추출
    예: "제2.4.1조(계통한계가격의 산정)" → "계통한계가격의 산정"
        "제2.4.1조 계통한계가격의 산정" → "계통한계가격의 산정"
    """
    # 조문번호 표기의 띄어쓰기를 정규화한 뒤 탐색해야 아래 정규식이 안정적으로 매칭된다
    normalized_text = re.sub(
        ARTICLE_PATTERN,
        lambda m: _normalize_article_num(m.group()),
        text
    )

    # 괄호 안 제목 패턴: 제X.X.X조(제목)
    paren_match = re.search(
        r'제\s*[\d.]+\s*조\s*[(\[（]([^)\]）]+)[)\]）]',
        normalized_text
    )
    if paren_match:
        title = paren_match.group(1).strip()
        if len(title) >= 2:
            return title

    # 괄호 없이 바로 오는 제목 패턴: 제X.X.X조 제목
    # 주의: "제X.X.X조 제5항의 규정..." 처럼 같은 조문 내 다른 항을 인용하는 경우,
    # "제" 뒤에 숫자가 오면 문자 클래스가 거기서 멈춰 공백 포함 "제 " → strip 후 "제"만
    # 남는 오탐이 발생한다. 2자 미만이면 제목이 아닌 것으로 판단해 버린다.
    space_match = re.search(
        r'제\s*[\d.]+\s*조\s+([가-힣a-zA-Z ]{2,20})',
        normalized_text
    )
    if space_match:
        title = space_match.group(1).strip()
        if len(title) >= 2:
            return title

    return ""


class Chunker:
    """문서 청킹 관리"""

    def __init__(self, chunk_size: int = 800, overlap: int = 100, version: str = "2025-04-10"):
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
        matches = list(re.finditer(ARTICLE_PATTERN, content))
        chunk_id = 0

        if not matches:
            # 조문번호가 전혀 없는 문서(용어집/서식 등) — 통째로 크기 기반 분할
            if content.strip():
                chunks.extend(self._create_chunks(content, chunk_id, filename, ""))
            logger.info(f"Created {len(chunks)} chunks from {filename}")
            return chunks

        # 첫 조문 이전 서문(목차/전문 등)은 조문번호 없이 별도 청크로 저장
        preamble = content[:matches[0].start()]
        if preamble.strip():
            new_chunks = self._create_chunks(preamble, chunk_id, filename, "")
            chunks.extend(new_chunks)
            chunk_id += len(new_chunks)

        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
            article_text = content[start:end]
            article_num = _normalize_article_num(match.group())

            new_chunks = self._create_chunks(article_text, chunk_id, filename, article_num)
            chunks.extend(new_chunks)
            chunk_id += len(new_chunks)

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
        chunk_id = start_id
        current_chunk = ""

        def _flush(text_to_flush: str) -> None:
            nonlocal chunk_id
            if text_to_flush.strip():
                chunks.append({
                    "text": text_to_flush.strip(),
                    "metadata": self._extract_metadata(
                        text_to_flush, filename, article_num
                    ),
                    "chunk_id": f"{filename}_{chunk_id:04d}"
                })
                chunk_id += 1

        # 표/산식 블록은 별도로 분리해 크기 제한과 무관하게 하나의 청크로 유지한다
        # (표 중간이 잘려 산식/컬럼 의미가 끊기는 것을 방지)
        for block_text, block_type in _split_table_blocks(text):
            if block_type == "table":
                _flush(current_chunk)
                current_chunk = ""
                _flush(block_text)
                continue

            # 일반 텍스트 블록: 기존과 동일하게 줄 단위 크기 기반 분할
            for line in block_text.split('\n'):
                if len(current_chunk) + len(line) > self.chunk_size and current_chunk.strip():
                    _flush(current_chunk)
                    current_chunk = line
                else:
                    current_chunk += line + "\n"

        # 마지막 청크
        _flush(current_chunk)

        # 조문당 청크 수 상한 — 표/산식 구간에서 다음 조문 경계 인식이 깨져 하나의
        # 조문에 청크가 과도하게 쌓이는 경우, 초과분을 하나로 뭉치지 않고 chunk_size
        # 단위로 다시 분할한다. (하나로 뭉치면 수만 자짜리 거대 청크가 생겨 임베딩
        # 품질이 오히려 나빠지므로, 개수 상한보다 "청크당 크기 제한"을 우선한다)
        if len(chunks) > MAX_CHUNKS_PER_ARTICLE:
            overflow_text = "\n\n".join(
                c["text"] for c in chunks[MAX_CHUNKS_PER_ARTICLE - 1:]
            )
            chunks = chunks[:MAX_CHUNKS_PER_ARTICLE - 1]

            resplit_chunk = ""
            for line in overflow_text.split('\n'):
                if len(resplit_chunk) + len(line) > self.chunk_size and resplit_chunk.strip():
                    _flush(resplit_chunk)
                    resplit_chunk = line
                else:
                    resplit_chunk += line + "\n"
            _flush(resplit_chunk)

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
            article_num (str): 조문번호 (chunk_document에서 이미 정규화되어 전달됨)

        Returns:
            Dict: 메타데이터
        """
        # 조문번호 — 상위에서 전달되지 않은 경우 청크 텍스트에서 재탐색 (띄어쓰기 허용 + 정규화)
        조문번호 = article_num
        if not 조문번호:
            article_match = re.search(ARTICLE_PATTERN, chunk_text)
            조문번호 = _normalize_article_num(article_match.group()) if article_match else ""

        # 장, 절 추출 — 기본은 조문번호에서 파생
        if 조문번호:
            parts = 조문번호.replace('제', '').replace('조', '').split('.')
            if len(parts) >= 2:
                장번호 = f"제{parts[0]}장"
                절번호 = f"제{parts[1]}절"
            else:
                장번호 = ""
                절번호 = ""
        else:
            장번호 = ""
            절번호 = ""

        # 텍스트에 "제X장"/"제X절" 헤더가 직접 있으면 그 값을 우선 사용 (띄어쓰기 허용)
        chapter_match = re.search(CHAPTER_PATTERN, chunk_text)
        if chapter_match:
            장번호 = re.sub(r'\s+', '', chapter_match.group())

        section_match = re.search(SECTION_PATTERN, chunk_text)
        if section_match:
            절번호 = re.sub(r'\s+', '', section_match.group())

        return {
            "조문번호": 조문번호,
            "장번호": 장번호,
            "절번호": 절번호,
            "장제목": "",  # 별도 추출 필요
            "조제목": _extract_article_title(chunk_text, 조문번호),
            "페이지": 0,
            "버전": self.version,
            "is_latest": True,
            "이전버전": None,
            "개정유형": "유지",  # 신설/수정/삭제/유지
            "개정이력": "[]",  # JSON 문자열
            "최초제정일": "2021-01-01",
            "개정횟수": 0,
            "키워드": _extract_keywords(chunk_text)
        }


def get_chunker(chunk_size: int = 800, overlap: int = 100) -> Chunker:
    """
    청커 인스턴스 반환

    Args:
        chunk_size (int): 청크 최대 크기
        overlap (int): 청크 간 중복 크기

    Returns:
        Chunker: 청커 인스턴스
    """
    return Chunker(chunk_size, overlap)
