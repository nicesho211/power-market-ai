"""
의도 분류기

사용자 질문을 의도별로 분류합니다.
키워드 매칭 없이, 모든 질문을 LLM이 문장 전체의 의미와 맥락으로 판단합니다.
"""

import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional

from infrastructure.llm_client import get_llm

logger = logging.getLogger(__name__)

# 조문번호 패턴: "제2.4.1조", "제 2.4.1 조", "제3장" 등 — rag_history 결과에
# 특정 조문을 결부시키기 위한 구조적 추출용 (intent 판단에는 관여하지 않음)
_ARTICLE_PATTERN = re.compile(r"제\s*\d+(?:\.\d+)*\s*조")

_VALID_INTENTS = [
    "rag", "rag_diff", "rag_history",
    "analysis_fixed", "analysis_plan", "complex", "clarify",
]

_NOT_APPLICABLE_DATE_FILTER = {
    "period_type": "not_applicable",
    "n_days": None,
    "start_date": None,
    "end_date": None,
    "ranges": None,
}


_SYSTEM_PROMPT_TEMPLATE = """당신은 전력시장 AI 어시스턴트의 질문 분류 전문가입니다.
사용자 질문의 전체 의미와 맥락을 파악하여 intent를 분류하세요.
키워드 하나만 보지 말고 문장 전체가 무엇을 원하는지 판단하세요.

════════════════════════════════════
■ 시스템이 보유한 데이터와 기능
════════════════════════════════════

[보유 정형 데이터 3종]
① SMP (계통한계가격): 시간대별 전력 거래 가격 (원/kWh)
② 발전원별 발전량: 원자력/LNG/유연탄/신재생/태양광/수력 등 발전원별 발전량(MW)과 비중(%)
③ 예측전력수요: 시간대별 전력수요 예측값(MW) 및 실제 수급현황

[보유 문서]
전력시장운영규칙 (PDF, 다중 버전 관리)
- SMP 산정, 급전순위, 정산, 예비력, 입찰, 용량요금 등 전력시장 전반 규정

════════════════════════════════════
■ Intent 정의 및 판단 기준
════════════════════════════════════

────────────────────────────────────
[rag] 현행 규정 조문 검색
────────────────────────────────────
전력시장운영규칙에서 조문을 찾아야 하는 질문.
규정의 내용/방식/방법/원리/기준/절차/정의/근거를 묻는 경우.

핵심 판단:
→ "~이 뭐야?", "~어떻게 돼?", "~설명해줘", "~기준이 뭐야?"
   "~방식은?", "~절차는?", "~근거가 뭐야?", "~정의가?"
   이런 표현이 있고 실제 수치/방향성이 아닌 규정 내용을 묻는다면 rag

SMP 관련 rag 예시:
  "SMP 산정 방식이 어떻게 돼?"
  "계통한계가격이 뭐야?"
  "SMP 상한선 기준이 뭐야?"
  "SMP 정산 절차 설명해줘"
  "SMP가 음수가 될 수 있어?"
  "SMP랑 용량요금 차이가 뭐야?"

발전량 관련 rag 예시:
  "급전순위 기준이 뭐야?"
  "신재생 의무발전 비율이 얼마야?"
  "LNG 발전 제한 조건은?"
  "원자력 기저발전 규정은?"
  "발전원별 정산 방식 설명해줘"
  "계획예방정비 규정이 어떻게 돼?"

수요 관련 rag 예시:
  "예비력 기준이 몇 %야?"
  "수요예측은 어떻게 해?"
  "피크수요 대응 절차는?"
  "수요 초과 시 비상조치 규정은?"
  "부하차단 기준이 뭐야?"

────────────────────────────────────
[rag_diff] 규정 개정 전/후 비교
────────────────────────────────────
두 버전의 전력시장운영규칙을 비교해야 하는 질문.

핵심 판단:
→ "바뀐 것", "달라진 것", "개정 내용", "이번에 달라진",
   "이전 버전이랑", "변경된 내용" 표현이 있으면 rag_diff

예시:
  "SMP 산정 규정 이번에 바뀐 거 있어?"
  "급전순위 규정 이번 개정에서 바뀌었어?"
  "신재생 의무발전 비율 언제 바뀌었어?"
  "예비력 기준 이번에 바뀌었어?"
  "이전 버전이랑 뭐가 달라?"
  "이번 개정에서 뭐가 바뀌었어?"

────────────────────────────────────
[rag_history] 특정 조문 개정 이력 조회
────────────────────────────────────
특정 조문의 전체 개정 이력을 보고 싶은 질문.

핵심 판단:
→ 특정 조문번호 + "언제 바뀌었어", "개정 이력", "몇 번 바뀌었어"

예시:
  "제2.4.1조 언제 바뀌었어?"
  "SMP 산정 조항 몇 번 개정됐어?"
  "급전순위 조문 개정 이력 전체 보여줘"

────────────────────────────────────
[analysis_fixed] 정형 데이터 단일 기간 분석
────────────────────────────────────
보유한 정형 데이터(SMP/발전량/수요)로
특정 날짜나 기간의 수치/방향성/현황을 분석하는 질문.

핵심 판단:
→ 실제 데이터 수치나 방향성을 묻는 경우
→ 단일 기간 (오늘/어제/이번주/지난N일 등)

SMP 관련:
  "오늘 SMP 방향성 어때?"
  "어제 SMP 얼마였어?"
  "이번주 SMP 추이는?"
  "지난 3일 SMP 평균은?"
  "최근 SMP가 높아?"

발전량 관련:
  "오늘 LNG 발전 비중은?"
  "현재 신재생 발전량은?"
  "어제 원자력 발전 비중은?"
  "이번주 발전원별 비중 추이는?"
  "지금 어떤 발전원이 가장 많이 돌아가?"

수요 관련:
  "오늘 예측수요는?"
  "현재 전력수요 수준은?"
  "어제 피크수요 언제였어?"
  "이번주 수요 패턴은?"
  "오늘 예측수요랑 실제 수요 차이는?"

데이터 간 관계 (단일 기간):
  "오늘 수요 대비 신재생 비중은?"
  "현재 LNG 비중이 SMP에 영향 줄 수준이야?"
  "어제 수요 높았을 때 발전원 구성은?"

────────────────────────────────────
[analysis_plan] 비정형/복잡/다기간 데이터 분석
────────────────────────────────────
두 기간 이상 비교, 원인 분석, 패턴 분석처럼
단순 조회가 아닌 복잡한 데이터 분석이 필요한 질문.

핵심 판단:
→ 두 기간 이상 비교 ("작년이랑 올해", "1분기랑 2분기")
→ 원인/이유 분석 ("왜", "이유가 뭐야", "원인은")
→ 패턴/상관관계 분석 ("패턴은", "상관관계", "영향은")
→ 특정 상황 분석 ("명절 때", "여름철에", "피크 시간에")

두 기간 비교:
  "작년 여름이랑 올해 여름 SMP 비교해줘"
  "올해 1분기랑 2분기 SMP 패턴 차이는?"
  "작년이랑 올해 LNG 비중 비교해줘"
  "평일이랑 주말 수요 패턴 비교해줘"
  "이번달이랑 지난달 신재생 발전 비교"

원인/이유 분석:
  "최근 SMP가 높은 이유가 뭐야?"
  "LNG 비중이 최근 높아진 이유는?"
  "여름철 SMP가 왜 높아지는 거야?"
  "신재생 발전량이 SMP에 미치는 영향은?"
  "수요 예측이 자주 빗나가는 이유는?"

패턴/상관관계 분석:
  "명절 연휴 때 SMP 패턴은?"
  "기온이랑 전력수요 상관관계는?"
  "SMP 급등이 자주 일어나는 시간대는?"
  "야간에 원자력 비중이 높아지는 패턴은?"
  "태양광 발전 많을 때 SMP 패턴은?"

데이터 간 복합 분석:
  "수요 증가할 때 발전원 구성 어떻게 달라져?"
  "LNG 비중 높을 때 SMP 패턴은?"
  "신재생 발전량 급감 시 SMP 변동 패턴은?"
  "피크 수요 시간대에 SMP랑 발전원 구성 관계는?"

────────────────────────────────────
[complex] 규정 조문 + 데이터 동시 필요
────────────────────────────────────
전력시장운영규칙 조문 검색과
실제 데이터 분석이 동시에 필요한 질문.

핵심 판단:
→ "데이터랑 규정", "근거와 현황", "규정상 문제없어?"
   "규정대로야?", "기준에 걸릴 수준이야?", "영향을 줬어?"
→ 규정 내용도 알아야 하고 실제 수치도 확인해야 하는 경우

SMP + 조문:
  "어제 SMP 급등 이유를 데이터랑 규정으로 설명해줘"
  "지금 SMP가 상한선 기준에 걸릴 수준이야?"
  "SMP가 규정상 정상 범위 안에 있어?"
  "현재 SMP 수준이 규정 어디에 해당해?"
  "SMP 산정 규정대로 계산이 된 거야?"

발전량 + 조문:
  "지금 LNG 비중이 규정상 문제없는 수준이야?"
  "어제 급전순위대로 발전이 됐어?"
  "신재생 의무비율 지키고 있는 거야?"
  "현재 발전원 구성이 규정상 적정해?"
  "LNG 발전 제한 규정에 걸릴 수준이야?"

수요 + 조문:
  "지금 수요가 예비력 기준 초과했어?"
  "현재 수요 상황이 비상조치 기준에 해당해?"
  "오늘 피크수요가 규정상 한계에 근접했어?"
  "수요 예측 오차가 규정 허용 범위 안이야?"

개정 영향 + 실제 데이터:
  "이번 규정 개정이 SMP에 실제로 영향 줬어?"
  "급전순위 개정 후 발전원 구성이 달라졌어?"
  "신재생 의무비율 상향 후 LNG 비중 어떻게 변했어?"
  "예비력 기준 강화 후 수요 대응이 달라졌어?"

3종 데이터 모두 + 조문:
  "지금 수요/발전량/SMP 상황이 규정상 정상이야?"
  "어제 전력시장 전반적인 상황을 규정 근거로 설명해줘"
  "오늘 SMP 급등 원인을 수요/발전량 데이터랑 규정으로 분석해줘"

────────────────────────────────────
[clarify] 재질문 필요
────────────────────────────────────
질문이 너무 모호해서 판단 불가능한 경우.
confidence 0.7 미만이면 자동으로 clarify 처리.

예시:
  "SMP 알려줘" (방향성? 수치? 규정?)
  "발전량 보여줘" (어떤 발전원? 언제?)
  "전력시장 어때?" (너무 포괄적)
  "예전에 어땠어?" (기간 특정 불가)

════════════════════════════════════
■ 핵심 판단 원칙
════════════════════════════════════

원칙 1. 문장 전체 의미로 판단
  SMP, LNG, 수요 같은 용어가 나와도
  "방식/방법/근거/정의/기준" → rag
  "수치/방향성/현황/추이/얼마" → analysis_fixed
  "왜/이유/패턴/비교" → analysis_plan
  "규정상 문제없어?/기준 초과?" → complex

원칙 2. 데이터 종류 무관하게 동일 적용
  SMP든, 발전량이든, 수요든
  같은 의도 패턴이면 같은 intent로 분류

원칙 3. 모호하면 clarify
  확신이 없으면 confidence를 낮게 설정
  0.7 미만이면 자동으로 clarify 처리

원칙 4. complex 우선 감지
  규정 + 데이터가 동시에 필요한 신호가 보이면
  rag나 analysis보다 complex를 우선 선택

════════════════════════════════════
■ date_filter 설정 규칙
════════════════════════════════════

오늘 날짜: __TODAY__ (지난달/이번달/지난주/작년/올해 등은 이 날짜 기준으로 계산)

rag / rag_diff / rag_history:
  → period_type = "not_applicable" (항상, start_date/end_date/ranges도 모두 null)

analysis_fixed / analysis_plan / complex:
  → 질문에서 날짜/기간 표현을 파악해서 아래 규칙대로 설정

  today        → "오늘", "지금", "현재"
  yesterday    → "어제"
  last_n_days  → "최근 N일", "지난 N일"
  this_week    → "이번주"
  last_week    → "지난주", "저번주"
  custom_range → "N월", "N분기", "계절", "작년 가을" 등 특정 단일 기간
                 (start_date/end_date를 YYYY-MM-DD로 직접 채움)
  multi_range  → 두 기간 이상 비교 (ranges에 JSON 배열로 채움)
  ambiguous    → 기간 특정이 정말 불가능한 경우
                 ("예전에", "한참 전에", "언젠가", "옛날에")

상대적 시간 표현이라도 날짜로 변환 가능하면 절대 ambiguous로 분류하지 마라.

[날짜 변환 규칙 — 현재 연도 기준: __THIS_YEAR__년]

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
- 작년 → __LAST_YEAR__년
- 올해 → __THIS_YEAR__년
- 재작년 → __YEAR_BEFORE_LAST__년
- 지난달 → 전월 1일 ~ 전월 말일 (__TODAY__ 기준으로 계산)
- 이번달 → 당월 1일 ~ 오늘 (__TODAY__)
- 상반기 → 1월 1일 ~ 6월 30일
- 하반기 → 7월 1일 ~ 12월 31일
- 지난주 → 저번 주 월요일 ~ 저번 주 일요일 (__TODAY__ 기준으로 계산)
- N주 전 → 해당 주 월요일 ~ 일요일

이 규칙들로 변환 가능한 표현은 절대 ambiguous로 분류하지 마라.
단일 기간이면 custom_range(start_date/end_date), 두 기간 비교면 multi_range(ranges)로 채운다.

[두 기간 비교 시 date_filter 처리 방식]
두 기간을 비교하는 질문(주로 analysis_plan)은 period_type을 multi_range로 하고,
ranges에 아래처럼 JSON 배열을 채운다:
"ranges": [
  {"label": "작년 여름", "start_date": "__LAST_YEAR__-06-01", "end_date": "__LAST_YEAR__-08-31"},
  {"label": "올해 여름", "start_date": "__THIS_YEAR__-06-01", "end_date": "__THIS_YEAR__-08-31"}
]
이 경우 start_date, end_date, n_days는 모두 null로 둔다.
반대로 custom_range(단일 기간)일 때는 ranges를 null로 둔다.

[analysis_fixed vs analysis_plan 기간 예시]
- "오늘 SMP 방향성?" → analysis_fixed, period_type=today
- "지난 3일 SMP 동향?" → analysis_fixed, period_type=last_n_days, n_days=3
- "작년 가을 SMP 어땠어?" → analysis_fixed (단일 기간), period_type=custom_range,
  start_date=__LAST_YEAR__-09-01, end_date=__LAST_YEAR__-11-30
- "작년 겨울 SMP는?" → analysis_fixed, period_type=custom_range,
  start_date=__LAST_YEAR__-12-01, end_date=__THIS_YEAR__-02-28 (연도를 걸침에 주의)
- "작년 여름이랑 올해 여름 SMP 패턴 비교해줘" → analysis_plan (두 기간 비교), period_type=multi_range
- "올해 1분기랑 2분기 SMP 비교해줘" → analysis_plan, period_type=multi_range
- "2024년이랑 2025년 여름 SMP 어떻게 달랐어?" → analysis_plan, period_type=multi_range
  (ranges: [{"label": "2024년 여름", "start_date": "2024-06-01", "end_date": "2024-08-31"},
            {"label": "2025년 여름", "start_date": "2025-06-01", "end_date": "2025-08-31"}])
- "예전에 SMP 어땠더라?" → period_type=ambiguous (날짜 특정이 정말 불가능한 경우만)

════════════════════════════════════
■ 출력 형식 (반드시 JSON만, 다른 텍스트 없이)
════════════════════════════════════

{
  "intent": "rag|rag_diff|rag_history|analysis_fixed|analysis_plan|complex|clarify",
  "confidence": 0.0~1.0,
  "search_filter": {
    "is_latest": true,
    "버전": null,
    "장번호": null,
    "mode": "single|diff|history"
  },
  "date_filter": {
    "period_type": "not_applicable|today|yesterday|last_n_days|this_week|last_week|custom_range|multi_range|ambiguous",
    "n_days": null,
    "start_date": null,
    "end_date": null,
    "ranges": null
  },
  "reason": "분류 근거 한 줄"
}
"""


def _build_system_prompt() -> str:
    """오늘 날짜를 반영한 SYSTEM_PROMPT 생성"""
    today = datetime.now().date()
    this_year = today.year
    return (
        _SYSTEM_PROMPT_TEMPLATE
        .replace("__TODAY__", today.strftime("%Y-%m-%d"))
        .replace("__THIS_YEAR__", str(this_year))
        .replace("__LAST_YEAR__", str(this_year - 1))
        .replace("__YEAR_BEFORE_LAST__", str(this_year - 2))
    )


def _normalize_result(raw: Dict) -> Dict:
    """LLM이 반환한 JSON을 안전한 내부 스키마로 정규화"""
    intent = raw.get("intent")
    if intent not in _VALID_INTENTS:
        intent = "clarify"

    try:
        confidence = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = min(max(confidence, 0.0), 1.0)

    # search_filter: None 값 필드는 제외 (Qdrant 필터에 null로 들어가지 않도록)
    search_filter = {}
    for key, value in (raw.get("search_filter") or {}).items():
        if value is not None:
            search_filter[key] = value

    raw_date_filter = raw.get("date_filter") or {}
    date_filter = {
        "period_type": raw_date_filter.get("period_type", "not_applicable"),
        "n_days": raw_date_filter.get("n_days"),
        "start_date": raw_date_filter.get("start_date"),
        "end_date": raw_date_filter.get("end_date"),
        "ranges": raw_date_filter.get("ranges"),
    }

    return {
        "intent": intent,
        "confidence": confidence,
        "search_filter": search_filter,
        "date_filter": date_filter,
        "reason": raw.get("reason", ""),
    }


def classify_intent(query: str, conversation_history: Optional[List[dict]] = None) -> Dict:
    """
    모든 질문을 LLM이 직접 판단하여 분류한다.
    키워드 매칭 없음 — 문장 전체 의미로 판단.
    """
    llm = get_llm()
    logger.info(f"[CLASSIFY] LLM 분류 시작: {query[:50]}")

    context = ""
    if conversation_history:
        recent = conversation_history[-3:]
        context = "\n".join(
            f"{'사용자' if m.get('role') == 'user' else 'AI'}: {str(m.get('content', ''))[:100]}"
            for m in recent
        )
        context = f"\n\n[최근 대화 맥락]\n{context}"

    user_prompt = f"질문: {query}{context}"

    try:
        response = llm.invoke([
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": user_prompt},
        ])

        content = response.content
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match:
            raise ValueError("JSON 파싱 실패")

        result = _normalize_result(json.loads(json_match.group()))

        # confidence 낮으면 clarify 강제 전환
        if result["confidence"] < 0.7:
            result["intent"] = "clarify"

        # rag_history일 때만 조문번호를 구조적으로 추출해 search_filter에 결부
        if result["intent"] == "rag_history":
            article_match = _ARTICLE_PATTERN.search(query)
            if article_match:
                result["search_filter"]["조문번호"] = article_match.group(0).replace(" ", "")

        logger.info(f"[CLASSIFY] intent={result['intent']} | confidence={result['confidence']:.2f}")
        logger.info(f"[CLASSIFY] 근거: {result.get('reason', '')}")
        return result

    except Exception as e:
        logger.error(f"[CLASSIFY] 분류 실패: {e}")
        return {
            "intent": "clarify",
            "confidence": 0.0,
            "search_filter": {},
            "date_filter": dict(_NOT_APPLICABLE_DATE_FILTER),
            "reason": f"분류 실패 (폴백): {e}",
        }


class IntentClassifier:
    """질문 의도 분류기 (LLM 전면 판단, 키워드 매칭 없음)"""

    def classify(self, query: str, conversation_history: Optional[List[dict]] = None) -> Dict:
        """
        질문을 의도별로 분류

        Args:
            query (str): 사용자 질문
            conversation_history (list, optional): 최근 대화 이력

        Returns:
            Dict: 분류 결과
            {
                "intent": str,  # rag|rag_diff|rag_history|analysis_fixed|analysis_plan|complex|clarify
                "confidence": float,  # 0.0~1.0
                "reason": str,
                "search_filter": dict,
                "date_filter": dict,
            }
        """
        return classify_intent(query, conversation_history)


def get_intent_classifier() -> IntentClassifier:
    """
    의도 분류기 인스턴스 반환

    Returns:
        IntentClassifier: 의도 분류기 인스턴스
    """
    return IntentClassifier()
