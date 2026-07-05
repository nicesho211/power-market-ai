"""
프롬프트 관리

LangGraph의 각 노드에서 사용할 프롬프트 템플릿을 중앙에서 관리합니다.
"""

from typing import Dict


class PromptManager:
    """프롬프트 템플릿 관리"""
    
    @staticmethod
    def get_intent_classifier_prompt() -> str:
        """의도 분류기 프롬프트"""
        return """당신은 전력시장 Q&A를 분류하는 전문가입니다.

사용자의 질문을 다음 중 하나의 의도로 분류하세요:
- rag: 단순 규정 조항 조회 (예: "제2.4.1조가 뭔가요?")
- rag_diff: 개정 전후 비교 (예: "이번 개정에서 뭐가 바뀌었어?")
- rag_history: 개정 이력 조회 (예: "이 조항이 몇 번 개정됐어?")
- analysis_fixed: SMP 방향성 추정 (정형 파이프라인, 예: "SMP가 올라갈까?")
- analysis_plan: 비정형 데이터 분석 (LLM Planning, 예: "최근 LNG 발전 비중이 어떻게 됐나?")
- complex: 규정 + 데이터 복합 분석 (예: "이 규정이 SMP에 어떤 영향을 미쳤나?")

사용자 질문:
{user_query}

출력 형식:
intent: [위 중 하나]
confidence: [0.0~1.0]
reason: [분류 근거]"""
    
    @staticmethod
    def get_rag_retrieval_prompt() -> str:
        """RAG 검색 프롬프트"""
        return """당신은 전력시장운영규칙 전문가입니다.

주어진 규정 내용을 바탕으로 사용자의 질문에 명확하게 답하세요.

규정 내용:
{context}

사용자 질문:
{user_query}

답변 작성 기준:
1. 규정에 있는 내용만 답변하기
2. 조문번호, 페이지 명시하기
3. 확실하지 않으면 "추정/추가 확인 필요" 표시하기"""
    
    @staticmethod
    def get_diff_analysis_prompt() -> str:
        """개정 비교 분석 프롬프트"""
        return """당신은 전력시장운영규칙 개정 전문가입니다.

개정 전후의 규정 내용을 비교하여 변경 사항을 설명하세요.

[개정 전]
{before_content}

[개정 후]
{after_content}

변경 요지 및 업무 영향을 분석해주세요."""
    
    @staticmethod
    def get_smp_analysis_prompt() -> str:
        """SMP 방향성 분석 프롬프트"""
        return """당신은 전력시장 분석 전문가입니다.

다음 스코어링 결과를 바탕으로 SMP 방향성을 자연어로 분석해주세요.
(LLM이 스코어링을 하는 것이 아니라, 이미 계산된 스코어링 결과를 설명하는 것입니다.)

스코어링 결과:
{scoring_result}

분석 포인트:
1. 각 지표별 현황 설명
2. 방향성의 신뢰도 평가
3. 주목할 만한 변화 포인트"""
    
    @staticmethod
    def get_period_comparison_prompt() -> str:
        """기간별 SMP 비교 분석 프롬프트 (multi_range)"""
        return """당신은 전력시장 분석 전문가입니다.

다음은 서로 다른 기간의 SMP 통계입니다. (LLM이 통계를 계산하는 것이 아니라,
이미 계산된 기간별 통계를 비교 설명하는 것입니다.)

기간별 통계:
{comparison_result}

분석 포인트:
1. 기간 간 평균/최고/최저 SMP 차이
2. 어느 기간이 더 높거나 낮은 경향을 보이는지
3. 데이터를 가져오지 못한 기간이 있다면 그 점을 명시"""

    @staticmethod
    def get_planning_analyzer_prompt() -> str:
        """비정형 데이터 분석 계획 프롬프트"""
        return """당신은 전력시장 데이터 분석가입니다.

사용자의 분석 요청에 대해 필요한 데이터 수집 및 분석 계획을 세우세요.

사용자 요청:
{user_query}

다음 데이터 소스를 활용할 수 있습니다:
- SMP (계통한계가격): 일별 시간대별 데이터
- 발전원별 발전량: 수력, 유류, 유연탄, 원자력, 양수, LNG, 국내탄, 신재생, 태양광
- 현재 수급현황: 실시간 수요, 공급능력, 공급예비력, 예비율

분석 계획을 단계별로 제시하세요:
1단계: 필요 데이터 정의
2단계: 데이터 수집 방법
3단계: 분석 방법
4단계: 예상 결과 및 시각화"""
    
    @staticmethod
    def get_output_formatter_prompt() -> str:
        """출력 포맷팅 프롬프트"""
        return """당신은 전력시장 AI 어시스턴트입니다.

다음 분석 결과를 고정 템플릿에 맞게 포맷팅하세요:

분석 결과 타입: {result_type}
내용: {content}
소스/근거: {source}

포맷팅 규칙:
1. 📌 요약: 1~3줄 핵심 답변
2. 📖 근거: 제시한 데이터 또는 조문 명시
3. 🔄 변경/분석 포인트: 개정 사항 또는 주요 분석 내용
4. ⚠ 불확실성: 근거 미검출 시 "추정/추가 확인 필요" 표시
5. 📎 면책: 항상 포함

최종 포맷된 답변을 제시하세요."""
    
    @staticmethod
    def get_clarification_prompt() -> str:
        """명확화 요청 프롬프트"""
        return """당신은 전력시장 Q&A 어시스턴트입니다.

사용자의 질문이 모호한 경우 추가 정보를 요청하세요.

사용자 질문:
{user_query}

분류 이유: {reason}

질문이 모호한 이유를 설명하고, 명확하게 하기 위해 필요한 정보를 요청하세요.

예시 추가 질문들을 제시해주세요."""
    
    @staticmethod
    def get_all_prompts() -> Dict[str, str]:
        """모든 프롬프트 반환"""
        return {
            "intent_classifier": PromptManager.get_intent_classifier_prompt(),
            "rag_retrieval": PromptManager.get_rag_retrieval_prompt(),
            "diff_analysis": PromptManager.get_diff_analysis_prompt(),
            "smp_analysis": PromptManager.get_smp_analysis_prompt(),
            "planning_analyzer": PromptManager.get_planning_analyzer_prompt(),
            "output_formatter": PromptManager.get_output_formatter_prompt(),
            "clarification": PromptManager.get_clarification_prompt(),
        }


def get_rag_prompt(query: str = "", context: str = "", history: list = None) -> str:
    """RAG 검색 프롬프트 반환 — query/context/history를 받아 포맷팅된 프롬프트 반환"""
    template = PromptManager.get_rag_retrieval_prompt()
    history_text = ""
    if history:
        history_text = "\n".join(
            f"[{h.get('role', 'user')}] {h.get('content', '')}" for h in history[-5:]
        )
    try:
        return template.format(user_query=query, context=context)
    except KeyError:
        return template


def get_intent_classifier_prompt() -> str:
    """의도 분류기 프롬프트 반환"""
    return PromptManager.get_intent_classifier_prompt()


def get_analysis_prompt(query: str, analysis_data: dict) -> str:
    """분석 결과 요약을 위한 프롬프트 반환"""
    prompt = PromptManager.get_smp_analysis_prompt()
    return prompt.format(scoring_result=analysis_data)


def get_planning_prompt(user_query: str) -> str:
    """비정형 데이터 분석 계획 프롬프트 반환"""
    prompt = PromptManager.get_planning_analyzer_prompt()
    return prompt.format(user_query=user_query)


def get_output_formatter_prompt(result_type: str, content: str, source: str) -> str:
    """출력 포맷터용 프롬프트 반환"""
    prompt = PromptManager.get_output_formatter_prompt()
    return prompt.format(result_type=result_type, content=content, source=source)


def get_clarification_prompt(user_query: str, reason: str) -> str:
    """명확화 요청 프롬프트 반환"""
    prompt = PromptManager.get_clarification_prompt()
    return prompt.format(user_query=user_query, reason=reason)
