"""
의도 분류기

사용자 질문을 의도별로 분류합니다.
키워드 필터 + LLM 분류 조합 방식을 사용합니다.
"""

import json
import logging
import re
from datetime import datetime
from typing import Dict, Optional
from infrastructure.llm_client import get_llm
from infrastructure.prompt_manager import get_intent_classifier_prompt

# 조문번호 패턴: "제2.4.1조", "제 2.4.1 조", "제3장" 등
_ARTICLE_PATTERN = re.compile(r"제\s*\d+(?:\.\d+)*\s*조")

logger = logging.getLogger(__name__)


class IntentClassifier:
    """질문 의도 분류기"""
    
    # 키워드 필터
    RAG_KEYWORDS = ["조항", "규정", "조문", "개정", "전력시장운영규칙", "정산규정",
                     "규칙", "조", "절", "내용", "뭔가"]
    HISTORY_KEYWORDS = ["이력", "히스토리", "언제부터", "몇 번", "변천", "개정이력",
                        "변경 이력", "언제", "몰"]
    ANALYSIS_KEYWORDS = ["SMP", "방향성", "발전량", "수요", "분석", "올랐", "내렸",
                         "추이", "패턴", "추이", "어떻게", "추세", "예상"]
    DIFF_KEYWORDS = ["개정 전", "개정 후", "비교", "차이", "뭐가 바뀌었", "달라",
                     "수정", "변경"]
    # rag_diff는 규정 개정 비교 전용 — "비교/차이/변경" 같은 범용 단어가
    # SMP 데이터 비교 질문까지 오탐하지 않도록 규정 문맥 키워드 동반 여부로 한정한다.
    REGULATION_CONTEXT_KEYWORDS = ["조항", "규정", "조문", "개정", "전력시장운영규칙",
                                   "정산규정", "규칙"]
    
    def __init__(self):
        """분류기 초기화"""
        self.llm = get_llm()
    
    def classify(self, query: str) -> Dict:
        """
        질문을 의도별로 분류
        
        Args:
            query (str): 사용자 질문
            
        Returns:
            Dict: 분류 결과
            {
                "intent": str,  # rag|rag_diff|rag_history|analysis_fixed|analysis_plan|complex|clarify
                "confidence": float,  # 0.0~1.0
                "reason": str,
                "search_filter": dict  # 필터 정보
            }
        """
        # Step 1: 키워드 기반 필터
        keyword_result = self._keyword_filter(query)
        
        if keyword_result["confidence"] >= 0.8:
            logger.info(f"Intent: {keyword_result['intent']} (keyword filter)")
            return keyword_result
        
        # Step 2: LLM 기반 분류 (애매한 경우)
        llm_result = self._llm_classify(query)
        
        # confidence 합산 (가중 평균)
        final_confidence = (
            keyword_result["confidence"] * 0.4 +
            llm_result["confidence"] * 0.6
        )
        
        # 더 높은 confidence를 가진 결과 선택
        if llm_result["confidence"] > keyword_result["confidence"]:
            llm_result["confidence"] = final_confidence
            logger.info(f"Intent: {llm_result['intent']} (LLM classifier)")
            return llm_result
        else:
            keyword_result["confidence"] = final_confidence
            logger.info(f"Intent: {keyword_result['intent']} (keyword filter + LLM)")
            return keyword_result
    
    def _keyword_filter(self, query: str) -> Dict:
        """키워드 기반 의도 필터링 (date_filter 포함)"""
        query_lower = query.lower()

        NOT_APPLICABLE = {
            "period_type": "not_applicable",
            "n_days": None,
            "start_date": None,
            "end_date": None
        }
        TODAY = {
            "period_type": "today",
            "n_days": None,
            "start_date": None,
            "end_date": None
        }

        # 개정 비교 (우선 처리) — 단, "비교/차이/변경" 같은 범용 단어만으로는
        # 오탐하지 않도록 규정 문맥 키워드가 함께 있고 분석 키워드가 없을 때만 매칭
        has_diff_kw = any(kw in query_lower for kw in self.DIFF_KEYWORDS)
        has_regulation_ctx = any(kw in query_lower for kw in self.REGULATION_CONTEXT_KEYWORDS) \
            or bool(_ARTICLE_PATTERN.search(query))
        has_analysis_kw = any(kw in query_lower for kw in self.ANALYSIS_KEYWORDS)

        if has_diff_kw and has_regulation_ctx and not has_analysis_kw:
            return {
                "intent": "rag_diff",
                "confidence": 0.85,
                "reason": "개정 비교 키워드 감지",
                "search_filter": {"is_latest": True, "mode": "diff"},
                "date_filter": NOT_APPLICABLE
            }

        # 개정 이력
        if any(kw in query_lower for kw in self.HISTORY_KEYWORDS):
            # 조문번호가 있으면 해당 조문의 이력, 없으면 규칙 전체 개정 버전 목록
            article_match = _ARTICLE_PATTERN.search(query)
            sf = {"mode": "history"}
            if article_match:
                sf["조문번호"] = article_match.group(0).replace(" ", "")
            return {
                "intent": "rag_history",
                "confidence": 0.8,
                "reason": "개정 이력 키워드 감지" + (f" (조문: {sf.get('조문번호', '')})" if "조문번호" in sf else " (전체 버전 목록)"),
                "search_filter": sf,
                "date_filter": NOT_APPLICABLE
            }

        # 데이터 분석 (SMP 방향성)
        if any(kw in query_lower for kw in self.ANALYSIS_KEYWORDS):
            intent = "analysis_fixed" if ("SMP" in query or "방향성" in query) else "analysis_plan"

            # 질문에 "오늘" 외의 기간 표현(작년/올해/지난/최근 등)이 있으면
            # date_filter를 TODAY로 단정하지 말고, 정확한 기간 파싱을 위해
            # confidence를 낮춰 LLM 분류(_llm_classify)로 위임한다.
            if self._has_date_hint(query_lower):
                return {
                    "intent": intent,
                    "confidence": 0.5,
                    "reason": "분석 키워드 + 기간 표현 감지 → LLM 기간 파싱 위임",
                    "search_filter": {},
                    "date_filter": TODAY
                }

            return {
                "intent": intent,
                "confidence": 0.88 if intent == "analysis_fixed" else 0.75,
                "reason": ("SMP 정형 분석 키워드 감지" if intent == "analysis_fixed"
                           else "비정형 데이터 분석 키워드 감지"),
                "search_filter": {},
                "date_filter": TODAY
            }

        # 규정 조회
        if any(kw in query_lower for kw in self.RAG_KEYWORDS):
            return {
                "intent": "rag",
                "confidence": 0.8,
                "reason": "규정 조회 키워드 감지",
                "search_filter": {"is_latest": True, "mode": "single"},
                "date_filter": NOT_APPLICABLE
            }

        # 기본값 (불명확)
        return {
            "intent": "clarify",
            "confidence": 0.3,
            "reason": "키워드 매칭 실패",
            "search_filter": {},
            "date_filter": NOT_APPLICABLE
        }
    
    # "오늘"이 아닌 기간을 가리키는 표현 — 있으면 date_filter를 TODAY로 단정하면 안 됨
    _DATE_HINT_KEYWORDS = ["작년", "올해", "재작년", "어제", "그제", "지난", "최근",
                           "이번주", "지난주", "이번달", "지난달", "여름", "겨울",
                           "봄", "가을", "부터", "까지"]
    _DATE_HINT_PATTERN = re.compile(r"\d+\s*(년|월|일)")

    def _has_date_hint(self, query_lower: str) -> bool:
        """질문에 '오늘' 외의 명시적 기간 표현이 있는지 감지"""
        if any(kw in query_lower for kw in self._DATE_HINT_KEYWORDS):
            return True
        return bool(self._DATE_HINT_PATTERN.search(query_lower))

    def _llm_classify(self, query: str) -> Dict:
        """LLM 기반 의도 분류 (date_filter 포함 확장 스키마)"""
        try:
            formatted_prompt = self._build_llm_prompt(query)

            response = self.llm.invoke(formatted_prompt)
            response_text = response.content

            result = self._parse_llm_response(response_text)
            return result
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return {
                "intent": "clarify",
                "confidence": 0.2,
                "reason": f"LLM 오류: {str(e)}",
                "search_filter": {},
                "date_filter": {
                    "period_type": "not_applicable",
                    "n_days": None,
                    "start_date": None,
                    "end_date": None
                }
            }

    def _build_llm_prompt(self, user_query: str) -> str:
        """date_filter 포함 LLM 분류 프롬프트 (Few-Shot 12개 예시)"""
        today = datetime.now().date()
        today_str = today.strftime("%Y-%m-%d")
        this_year = today.year
        last_year = this_year - 1
        year_before_last = this_year - 2

        return f"""당신은 전력시장 Q&A를 분류하는 전문가입니다.
사용자 질문을 아래 형식에 맞게 분류하세요.

오늘 날짜: {today_str} (지난달/이번달/지난주/상반기/하반기 등은 이 날짜 기준으로 계산)

[출력 형식]
intent: [rag|rag_diff|rag_history|analysis_fixed|analysis_plan|complex|clarify]
confidence: [0.0~1.0]
search_filter_is_latest: [True|False|None]
search_filter_mode: [single|diff|history|None]
date_filter_period_type: [today|yesterday|last_n_days|this_week|last_week|custom_range|multi_range|ambiguous|not_applicable]
date_filter_n_days: [숫자 또는 None]
date_filter_start_date: [YYYY-MM-DD 또는 None]
date_filter_end_date: [YYYY-MM-DD 또는 None]
date_filter_ranges: [multi_range일 때만 JSON 배열 한 줄로, 그 외는 None]
reason: [분류 근거 한 줄]

[핵심 원칙]
상대적 시간 표현이라도 네가 날짜로 변환 가능하면 절대 ambiguous로 분류하지 마라.
ambiguous는 "예전에", "한참 전에", "언젠가", "옛날에"처럼 정말 날짜를 특정할 수 없는 경우에만 사용한다.

[period_type 정의]
- 단일 기간 질문 → custom_range (start_date/end_date를 직접 채움)
- 오늘/어제/최근 N일/이번주/지난주처럼 "오늘"을 기준으로 상대 계산되는 단순 기간은
  today/yesterday/last_n_days/this_week/last_week 그대로 사용 (날짜 자체는 Python이 계산)
- 두 기간 이상을 비교하는 질문 → multi_range (date_filter_ranges에 JSON 배열로 채움)
- 정말 날짜 특정이 불가능한 경우에만 → ambiguous

[분류 지침]
- 규정 관련(rag/rag_diff/rag_history)이면 date_filter_period_type은 반드시 not_applicable
- 분석 관련(analysis_fixed/analysis_plan/complex)이면 기간 표현에 따라 period_type을 채움
- complex일 때는 search_filter(규정 검색용)와 date_filter(데이터 수집용) 모두 채움

[analysis_fixed vs analysis_plan 구분 기준]
- 단일 기간의 SMP 방향성/추이 질문 → analysis_fixed
- 두 기간을 비교하거나 패턴 분석처럼 복잡한 분석 질문 → analysis_plan
예:
- "오늘 SMP 방향성?" → analysis_fixed
- "지난 3일 SMP 동향?" → analysis_fixed
- "작년 가을 SMP 어땠어?" → analysis_fixed (단일 기간)
- "작년 여름이랑 올해 여름 SMP 패턴 비교해줘" → analysis_plan (두 기간 비교)
- "{last_year}년과 {this_year}년 여름 SMP 어떻게 달랐어?" → analysis_plan

[날짜 변환 규칙 — 현재 연도 기준: {this_year}년]

계절 변환:
- 봄 = 3월 1일 ~ 5월 31일
- 여름 = 6월 1일 ~ 8월 31일
- 가을 = 9월 1일 ~ 11월 30일
- 겨울 = 12월 1일 ~ 익년 2월 28일 (연도를 걸치는 것에 주의)

분기 변환:
- 1분기 = 1월 1일 ~ 3월 31일
- 2분기 = 4월 1일 ~ 6월 30일
- 3분기 = 7월 1일 ~ 9월 30일
- 4분기 = 10월 1일 ~ 12월 31일

기타 표현 변환:
- 작년 → {last_year}년
- 올해 → {this_year}년
- 재작년 → {year_before_last}년
- 지난달 → 전월 1일 ~ 전월 말일 ({today_str} 기준으로 계산)
- 이번달 → 당월 1일 ~ 오늘 ({today_str})
- 상반기 → 1월 1일 ~ 6월 30일
- 하반기 → 7월 1일 ~ 12월 31일
- 지난주 → 저번 주 월요일 ~ 저번 주 일요일 ({today_str} 기준으로 계산)
- N주 전 → 해당 주 월요일 ~ 일요일

이 규칙들로 변환 가능한 표현은 절대 ambiguous로 분류하지 마라.
단일 기간이면 custom_range(start_date/end_date), 두 기간 비교면 multi_range(date_filter_ranges)로 채운다.

[두 기간 비교 시 date_filter 처리 방식]
두 기간을 비교하는 질문(analysis_plan)은 period_type을 multi_range로 하고,
date_filter_ranges에 아래처럼 JSON 배열을 한 줄로 채운다:
date_filter_ranges: [{{"label": "작년 여름", "start_date": "{last_year}-06-01", "end_date": "{last_year}-08-31"}}, {{"label": "올해 여름", "start_date": "{this_year}-06-01", "end_date": "{this_year}-08-31"}}]
이 경우 date_filter_start_date, date_filter_end_date, date_filter_n_days는 모두 None으로 둔다.
반대로 custom_range(단일 기간)일 때는 date_filter_ranges를 None으로 둔다.

[예시 12개]
Q: "SMP 산정 방식이 어떻게 돼?"
intent: rag
confidence: 0.92
search_filter_is_latest: True
search_filter_mode: single
date_filter_period_type: not_applicable
date_filter_n_days: None
date_filter_start_date: None
date_filter_end_date: None
date_filter_ranges: None
reason: 규정 조회 키워드 감지

Q: "이번에 직접구매 조항 뭐가 바뀌었어?"
intent: rag_diff
confidence: 0.90
search_filter_is_latest: True
search_filter_mode: diff
date_filter_period_type: not_applicable
date_filter_n_days: None
date_filter_start_date: None
date_filter_end_date: None
date_filter_ranges: None
reason: 개정 비교 키워드 감지

Q: "제2.4.1조 지금까지 몇 번 바뀌었어?"
intent: rag_history
confidence: 0.88
search_filter_is_latest: None
search_filter_mode: history
date_filter_period_type: not_applicable
date_filter_n_days: None
date_filter_start_date: None
date_filter_end_date: None
date_filter_ranges: None
reason: 개정 이력 키워드 감지

Q: "오늘 SMP 방향성 어때?"
intent: analysis_fixed
confidence: 0.92
search_filter_is_latest: None
search_filter_mode: None
date_filter_period_type: today
date_filter_n_days: None
date_filter_start_date: None
date_filter_end_date: None
date_filter_ranges: None
reason: SMP 방향성 + 오늘 기간 감지

Q: "지난 3일치 SMP 동향 알려줘"
intent: analysis_fixed
confidence: 0.90
search_filter_is_latest: None
search_filter_mode: None
date_filter_period_type: last_n_days
date_filter_n_days: 3
date_filter_start_date: None
date_filter_end_date: None
date_filter_ranges: None
reason: SMP 분석 + last_n_days(3) 기간 감지

Q: "최근에 SMP 어땠어?"
intent: analysis_fixed
confidence: 0.60
search_filter_is_latest: None
search_filter_mode: None
date_filter_period_type: ambiguous
date_filter_n_days: None
date_filter_start_date: None
date_filter_end_date: None
date_filter_ranges: None
reason: SMP 분석이나 기간 불명확 → ambiguous

Q: "작년 가을 SMP 어땠어?"
intent: analysis_fixed
confidence: 0.90
search_filter_is_latest: None
search_filter_mode: None
date_filter_period_type: custom_range
date_filter_n_days: None
date_filter_start_date: {last_year}-09-01
date_filter_end_date: {last_year}-11-30
date_filter_ranges: None
reason: 단일 기간(작년 가을)을 날짜로 변환 → analysis_fixed

Q: "작년 여름이랑 올해 여름 SMP 패턴 비교해줘"
intent: analysis_plan
confidence: 0.90
search_filter_is_latest: None
search_filter_mode: None
date_filter_period_type: multi_range
date_filter_n_days: None
date_filter_start_date: None
date_filter_end_date: None
date_filter_ranges: [{{"label": "작년 여름", "start_date": "{last_year}-06-01", "end_date": "{last_year}-08-31"}}, {{"label": "올해 여름", "start_date": "{this_year}-06-01", "end_date": "{this_year}-08-31"}}]
reason: 두 기간 비교 분석 → analysis_plan, 계절 표현을 날짜로 변환

Q: "올해 1분기랑 2분기 SMP 비교해줘"
intent: analysis_plan
confidence: 0.90
search_filter_is_latest: None
search_filter_mode: None
date_filter_period_type: multi_range
date_filter_n_days: None
date_filter_start_date: None
date_filter_end_date: None
date_filter_ranges: [{{"label": "1분기", "start_date": "{this_year}-01-01", "end_date": "{this_year}-03-31"}}, {{"label": "2분기", "start_date": "{this_year}-04-01", "end_date": "{this_year}-06-30"}}]
reason: 두 분기 비교 분석 → analysis_plan, 분기 표현을 날짜로 변환

Q: "작년 겨울 SMP는?"
intent: analysis_fixed
confidence: 0.88
search_filter_is_latest: None
search_filter_mode: None
date_filter_period_type: custom_range
date_filter_n_days: None
date_filter_start_date: {last_year}-12-01
date_filter_end_date: {this_year}-02-28
date_filter_ranges: None
reason: 겨울은 연도를 걸치므로 작년 12월~올해 2월로 변환 → analysis_fixed

Q: "2024년이랑 2025년 여름 SMP 어떻게 달랐어?"
intent: analysis_plan
confidence: 0.90
search_filter_is_latest: None
search_filter_mode: None
date_filter_period_type: multi_range
date_filter_n_days: None
date_filter_start_date: None
date_filter_end_date: None
date_filter_ranges: [{{"label": "2024년 여름", "start_date": "2024-06-01", "end_date": "2024-08-31"}}, {{"label": "2025년 여름", "start_date": "2025-06-01", "end_date": "2025-08-31"}}]
reason: 연도가 명시된 두 기간 비교 → analysis_plan

Q: "예전에 SMP 어땠더라?"
intent: analysis_fixed
confidence: 0.55
search_filter_is_latest: None
search_filter_mode: None
date_filter_period_type: ambiguous
date_filter_n_days: None
date_filter_start_date: None
date_filter_end_date: None
date_filter_ranges: None
reason: "예전에"는 날짜 특정이 정말 불가능함 → 진짜 ambiguous

[사용자 질문]
{user_query}"""

    def _parse_llm_response(self, response_text: str) -> Dict:
        """LLM 응답 파싱 (date_filter 포함)"""
        try:
            lines = response_text.strip().split('\n')

            intent = "clarify"
            confidence = 0.5
            reason = ""
            search_filter_is_latest = None
            search_filter_mode = None
            date_period_type = "not_applicable"
            date_n_days = None
            date_start = None
            date_end = None
            date_ranges = None

            for line in lines:
                line = line.strip()
                if line.startswith("intent:"):
                    val = line.split(":", 1)[1].strip()
                    valid_intents = ["rag", "rag_diff", "rag_history",
                                    "analysis_fixed", "analysis_plan", "complex", "clarify"]
                    intent = val if val in valid_intents else "clarify"

                elif line.startswith("confidence:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except Exception:
                        confidence = 0.5

                elif line.startswith("reason:"):
                    reason = line.split(":", 1)[1].strip()

                elif line.startswith("search_filter_is_latest:"):
                    val = line.split(":", 1)[1].strip()
                    if val == "True":
                        search_filter_is_latest = True
                    elif val == "False":
                        search_filter_is_latest = False

                elif line.startswith("search_filter_mode:"):
                    val = line.split(":", 1)[1].strip()
                    search_filter_mode = None if val == "None" else val

                elif line.startswith("date_filter_period_type:"):
                    date_period_type = line.split(":", 1)[1].strip()

                elif line.startswith("date_filter_n_days:"):
                    val = line.split(":", 1)[1].strip()
                    try:
                        date_n_days = int(val) if val != "None" else None
                    except Exception:
                        date_n_days = None

                elif line.startswith("date_filter_start_date:"):
                    val = line.split(":", 1)[1].strip()
                    date_start = None if val == "None" else val

                elif line.startswith("date_filter_end_date:"):
                    val = line.split(":", 1)[1].strip()
                    date_end = None if val == "None" else val

                elif line.startswith("date_filter_ranges:"):
                    val = line.split(":", 1)[1].strip()
                    if val and val != "None":
                        try:
                            parsed = json.loads(val)
                            if isinstance(parsed, list):
                                date_ranges = parsed
                        except (json.JSONDecodeError, TypeError):
                            logger.warning(f"date_filter_ranges JSON 파싱 실패: {val}")
                            date_ranges = None

            # search_filter 구성
            search_filter = {}
            if search_filter_is_latest is not None:
                search_filter["is_latest"] = search_filter_is_latest
            if search_filter_mode and search_filter_mode != "None":
                search_filter["mode"] = search_filter_mode

            return {
                "intent": intent,
                "confidence": min(max(confidence, 0.0), 1.0),
                "reason": reason,
                "search_filter": search_filter,
                "date_filter": {
                    "period_type": date_period_type,
                    "n_days": date_n_days,
                    "start_date": date_start,
                    "end_date": date_end,
                    "ranges": date_ranges
                }
            }

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {
                "intent": "clarify",
                "confidence": 0.2,
                "reason": "응답 파싱 실패",
                "search_filter": {},
                "date_filter": {
                    "period_type": "not_applicable",
                    "n_days": None,
                    "start_date": None,
                    "end_date": None
                }
            }


def get_intent_classifier() -> IntentClassifier:
    """
    의도 분류기 인스턴스 반환
    
    Returns:
        IntentClassifier: 의도 분류기 인스턴스
    """
    return IntentClassifier()
