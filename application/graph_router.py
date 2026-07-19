"""
LangGraph 워크플로우 라우터

의도 분류 결과에 따른 다양한 노드 및 엣지를 관리합니다.
AgentState 기반으로 상태를 추적합니다.
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph import StateGraph
from langgraph.types import Send
import logging
import json
import re
import time
import pandas as pd

from application.intent_classifier import get_intent_classifier
from domain.rag.retriever import get_retriever
from domain.rag.diff_pipeline import get_diff_pipeline
from domain.rag.history_manager import get_history_manager
from domain.analysis.direction_estimator import get_direction_estimator
from domain.analysis.smp_analyzer import get_smp_analyzer
from domain.analysis.mcp_client import fetch_smp_range
from infrastructure.llm_client import get_llm, get_llm_for_planning
from infrastructure.prompt_manager import PromptManager
from datetime import datetime
from domain.analysis.date_resolver import resolve_date_range

logger = logging.getLogger("GRAPH")


def _format_diff_unavailable(result: Dict) -> str:
    """compare_versions()가 비교 불가를 반환했을 때 사용자에게 보여줄 메시지."""
    versions = result.get("available_versions", [])
    ver_str = ", ".join(versions) if versions else "없음"
    return (
        f"⚠️ 개정 비교를 수행할 수 없습니다.\n\n"
        f"**사유:** {result.get('error', '')}\n\n"
        f"**현재 저장된 버전:** {ver_str}\n\n"
        "두 개 이상의 버전이 인덱싱되어야 비교가 가능합니다. "
        "사이드바에서 이전 버전 PDF를 업로드해주세요."
    )


# 사용 가능한 스킬 정의
SKILLS: Dict[str, Dict] = {
    "rag_skill": {
        "description": "전력시장운영규칙 현행 조문 검색 및 Q&A. 규정 내용/방식/기준/정의를 찾을 때 사용",
        "node": "rag_node",
        "output_field": "rag_result",
    },
    "analysis_skill": {
        "description": "SMP/발전원별 발전량/예측전력수요 실제 데이터 수집 및 분석. 현재 수치/방향성/원인/패턴 분석 시 사용",
        "node": "analysis_fixed_node",
        "output_field": "analysis_result",
    },
    "diff_skill": {
        "description": "전력시장운영규칙 두 버전 간 개정 전/후 조문 비교. 개정 내용/변경사항 확인 시 사용",
        "node": "rag_diff_node",
        "output_field": "rag_result",
    },
    "history_skill": {
        "description": "특정 조문의 전체 개정 이력 조회. 특정 조문이 언제 몇 번 바뀌었는지 확인 시 사용",
        "node": "rag_history_node",
        "output_field": "rag_result",
    },
}

SKILL_TO_NODE: Dict[str, str] = {name: info["node"] for name, info in SKILLS.items()}


def _prefer_new(current: Any, incoming: Any) -> Any:
    """병렬(Send) 노드가 동시에 같은 채널에 값을 반환할 때의 충돌 리듀서.

    executor_router가 parallel=True일 때 Send로 rag_node/analysis_fixed_node 등을
    동시 실행하면, 각 노드가 (자신이 건드리지 않은 필드까지 포함해) state 전체를
    반환하기 때문에 LangGraph 기본 채널(단일 writer만 허용)은
    `InvalidUpdateError: Can receive only one value per step`로 죽는다.
    두 브랜치가 반환하는 미변경 필드 값은 항상 동일하므로, 값이 있는 쪽(진짜로
    갱신된 값)을 우선 채택하면 병렬 실행에서도 안전하게 병합된다.
    """
    return incoming if incoming else current


class AgentState(TypedDict):
    """LangGraph 에이전트 상태"""
    query: Annotated[str, _prefer_new]
    intent: Annotated[str, _prefer_new]
    confidence: Annotated[float, _prefer_new]
    search_filter: Annotated[dict, _prefer_new]
    rag_result: Annotated[str, _prefer_new]
    analysis_result: Annotated[dict, _prefer_new]
    direction_result: Annotated[dict, _prefer_new]
    final_answer: Annotated[str, _prefer_new]
    needs_clarification: Annotated[bool, _prefer_new]
    conversation_history: Annotated[List[Dict], _prefer_new]
    progress_steps: Annotated[List[Dict], _prefer_new]
    # 에러 핸들링 필드 (A섹션)
    error: Annotated[Optional[Dict], _prefer_new]
    # 날짜 필터 필드 (B섹션)
    date_filter: Annotated[Optional[Dict], _prefer_new]
    clarify_message: Annotated[str, _prefer_new]
    # Agent Planning 관련 신규 필드
    plan: Annotated[Optional[Dict], _prefer_new]
    # {"skills_needed": [...], "parallel": bool, "execution_order": [...], "reason": str}
    completed_skills: Annotated[List[str], _prefer_new]
    # 로그 소요시간 계산용 (run() 시작 시각)
    _start_time: Annotated[Optional[float], _prefer_new]


class GraphRouter:
    """LangGraph 기반 워크플로우 라우터"""
    
    def __init__(self):
        """라우터 초기화"""
        self.classifier = get_intent_classifier()
        self.llm = get_llm()
        self.llm_planning = get_llm_for_planning()
        self.retriever = get_retriever()
        self.diff_pipeline = get_diff_pipeline()
        self.history_manager = get_history_manager()
        self.direction_estimator = get_direction_estimator()
        self.smp_analyzer = get_smp_analyzer()
        
        # LangGraph 구성
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """LangGraph 구성"""
        workflow = StateGraph(AgentState)

        # 노드 추가
        workflow.add_node("classify_node", self._classify_node)
        workflow.add_node("rag_node", self._rag_node)
        workflow.add_node("rag_diff_node", self._rag_diff_node)
        workflow.add_node("rag_history_node", self._rag_history_node)
        workflow.add_node("analysis_fixed_node", self._analysis_fixed_node)
        workflow.add_node("analysis_plan_node", self._analysis_plan_node)
        workflow.add_node("clarify_node", self._clarify_node)
        workflow.add_node("format_output_node", self._format_output_node)
        # Agent Planning 노드
        workflow.add_node("planning_node", self._planning_node)
        workflow.add_node("executor_node", self._executor_node)
        workflow.add_node("merge_node", self._merge_node)

        # 엣지 추가
        workflow.set_entry_point("classify_node")

        # classify_node → 라우팅 (complex는 planning_node로)
        workflow.add_conditional_edges(
            "classify_node",
            self._route_after_classify,
            {
                "rag": "rag_node",
                "rag_diff": "rag_diff_node",
                "rag_history": "rag_history_node",
                "analysis_fixed": "analysis_fixed_node",
                "analysis_plan": "analysis_plan_node",
                "complex": "planning_node",
                "clarify": "clarify_node",
                "format_output_node": "format_output_node",
            },
        )

        # 단순 처리 노드 → format_output_node
        workflow.add_edge("rag_node", "format_output_node")
        workflow.add_edge("rag_diff_node", "format_output_node")
        workflow.add_edge("rag_history_node", "format_output_node")
        workflow.add_edge("analysis_fixed_node", "format_output_node")
        workflow.add_edge("analysis_plan_node", "format_output_node")

        # Agent Planning 흐름
        workflow.add_edge("planning_node", "executor_node")
        workflow.add_conditional_edges(
            "executor_node",
            self._executor_router,
            {
                "rag_node": "rag_node",
                "analysis_fixed_node": "analysis_fixed_node",
                "rag_diff_node": "rag_diff_node",
                "rag_history_node": "rag_history_node",
                "merge_node": "merge_node",
            },
        )
        # 병렬 실행 후 merge_node로 수렴
        workflow.add_edge("merge_node", "format_output_node")

        workflow.set_finish_point("format_output_node")
        workflow.set_finish_point("clarify_node")

        return workflow.compile()
    
    # ===== 노드 구현 =====
    
    def _classify_node(self, state: AgentState) -> AgentState:
        """Step 1: 질문 분류"""
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info(f"[CHAT] 사용자 질문: {state['query']}")

        try:
            classification = self.classifier.classify(
                state["query"], state.get("conversation_history")
            )

            state["intent"] = classification["intent"]
            state["confidence"] = classification["confidence"]
            state["search_filter"] = classification.get("search_filter", {})
            state["date_filter"] = classification.get("date_filter", {
                "period_type": "not_applicable",
                "n_days": None,
                "start_date": None,
                "end_date": None
            })

            logger.info(f"[CLASSIFY] intent={state['intent']} | confidence={state['confidence']:.2f}")
            logger.info(f"[CLASSIFY] 분류 근거: {classification.get('reason', '')}")
            logger.info(f"[CLASSIFY] date_filter: {state.get('date_filter', {})}")

            # 모호한 기간 표현 → clarify로 전환 (B.2.6)
            if state["date_filter"].get("period_type") == "ambiguous":
                state["needs_clarification"] = True
                state["intent"] = "clarify"
                state["clarify_message"] = (
                    "어느 기간을 분석해드릴까요? "
                    "예: 오늘 / 어제 / 지난 3일 / 이번주 / 2025년 4월"
                )

            state["progress_steps"].append({
                "step": 1,
                "name": "질문 분류",
                "status": "완료",
                "detail": (
                    f"Intent: {classification['intent']} (confidence: {classification['confidence']:.2f})\n"
                    f"이유: {classification.get('reason', '')}\n"
                    f"기간: {state['date_filter'].get('period_type', 'N/A')}"
                )
            })

        except Exception as e:
            logger.error(f"Classify node error: {e}")
            state["error"] = {
                "node": "classify_node",
                "stage": "질문 분류",
                "reason": f"LLM JSON 파싱 실패: {str(e)}",
                "recoverable": True,
                "message": "질문을 정확히 이해하지 못했어요. 조금 더 구체적으로 말씀해주시겠어요?"
            }
            state["intent"] = "clarify"
            state["progress_steps"].append({
                "step": 1,
                "name": "질문 분류",
                "status": "실패",
                "detail": str(e)
            })

        return state
    
    def _route_after_classify(self, state: AgentState) -> str:
        """classify 이후 라우팅 — error 존재 시 format_output_node로 직행"""
        if state.get("error"):
            return "format_output_node"
        return state.get("intent", "clarify")
    
    def _rag_node(self, state: AgentState) -> AgentState:
        """단순 규정 Q&A"""
        logger.info(f"[RAG] Hybrid Search 시작 | query={state['query'][:50]} | filter={state.get('search_filter', {})}")

        try:
            # Step 2: 문서 검색
            results = self.retriever.search(
                query=state["query"],
                where=state["search_filter"] or {"is_latest": True}
            )

            logger.info(f"[RAG] 검색 완료 | {len(results) if results else 0}건")
            if results:
                top_meta = results[0].get("metadata", {})
                logger.info(
                    f"[RAG] 상위 결과: {top_meta.get('조문번호', '')} "
                    f"(distance={results[0].get('distance', 0):.2f})"
                )

            if not results:
                state["error"] = {
                    "node": "rag_node",
                    "stage": "문서 검색",
                    "reason": "관련 규정 검색 결과 0건",
                    "recoverable": True,
                    "message": "관련 규정을 찾지 못했습니다. 질문을 다르게 표현해보시거나, 조문번호를 알고 계시면 알려주세요."
                }
                state["rag_result"] = "검색결과없음"
                state["progress_steps"].append({
                    "step": 2,
                    "name": "문서 검색",
                    "status": "실패",
                    "detail": "검색 결과 0건"
                })
                return state

            state["progress_steps"].append({
                "step": 2,
                "name": "문서 검색",
                "status": "완료",
                "detail": f"검색 결과: {len(results)}건"
            })

            # 최상위 결과 선택
            top_result = results[0]
            doc_text = top_result.get("document", "")
            metadata = top_result.get("metadata", {})

            # Step 3: LLM을 통한 답변 생성
            prompt = PromptManager.get_rag_retrieval_prompt()
            formatted_prompt = prompt.format(
                context=doc_text,
                user_query=state["query"]
            )

            response = self.llm.invoke(formatted_prompt)

            state["progress_steps"].append({
                "step": 3,
                "name": "LLM 답변 생성",
                "status": "완료",
                "detail": f"조문: {metadata.get('조문번호', 'N/A')}"
            })

            state["rag_result"] = response.content

        except Exception as e:
            logger.error(f"RAG node error: {e}")
            state["error"] = {
                "node": "rag_node",
                "stage": "문서 검색/답변 생성",
                "reason": str(e),
                "recoverable": False,
                "message": "시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
            }
            state["rag_result"] = ""

        return state
    
    def _rag_diff_node(self, state: AgentState) -> AgentState:
        """개정 전후 비교 — compare_versions()로 자동 버전 탐지."""
        logger.info("=== RAG Diff Node ===")

        try:
            from domain.rag.diff_pipeline import compare_versions
            result = compare_versions(
                query=state["query"],
                document_type="전력시장운영규칙",
                top_k=5,
            )

            # 비교 불가 (버전 없음 / 이전 버전 없음)
            if result["error"]:
                state["error"] = {
                    "node": "rag_diff_node",
                    "stage": "버전 탐지",
                    "reason": result["error"],
                    "recoverable": True,
                    "message": result["error"],
                }
                state["rag_result"] = _format_diff_unavailable(result)
                state["progress_steps"].append({
                    "step": 2, "name": "개정 비교", "status": "실패",
                    "detail": result["error"],
                })
                return state

            latest_docs = result["latest_docs"]
            prev_docs   = result["prev_docs"]
            latest_v    = result["latest_version"]
            prev_v      = result["prev_version"]

            logger.info(
                f"[RAG] 버전 자동 탐지 | latest={latest_v} prev={prev_v} | "
                f"latest_docs={len(latest_docs)}건 prev_docs={len(prev_docs)}건"
            )

            latest_text = "\n\n".join(d["document"] for d in latest_docs) if latest_docs else "(검색 결과 없음)"
            prev_text   = "\n\n".join(d["document"] for d in prev_docs)   if prev_docs   else "(검색 결과 없음)"

            prompt = PromptManager.get_diff_analysis_prompt()
            formatted_prompt = prompt.format(
                before_content=prev_text,
                after_content=latest_text,
            )
            response = self.llm.invoke(formatted_prompt)

            state["progress_steps"].append({
                "step": 2, "name": "개정 비교", "status": "완료",
                "detail": f"{prev_v} → {latest_v} 비교 완료",
            })
            state["rag_result"] = (
                f"**버전 비교: {prev_v} → {latest_v}**\n\n"
                + response.content
            )

        except Exception as e:
            logger.error(f"RAG Diff node error: {e}")
            state["error"] = {
                "node": "rag_diff_node",
                "stage": "개정 비교",
                "reason": str(e),
                "recoverable": False,
                "message": "개정 비교 중 시스템 오류가 발생했습니다.",
            }
            state["rag_result"] = ""

        return state
    
    def _rag_history_node(self, state: AgentState) -> AgentState:
        """개정 이력 조회.

        search_filter["조문번호"] 존재 → 특정 조문 이력 조회
        search_filter["조문번호"] 없음 → 전체 인덱싱 버전 목록 반환
        """
        logger.info("=== RAG History Node ===")

        try:
            sf = state.get("search_filter") or {}
            조문번호 = sf.get("조문번호")

            # ── 케이스 A: 조문번호 없음 → 전체 규칙 개정 버전 목록 ────────────
            if not 조문번호:
                from domain.rag.vector_store import get_versions_by_document_type
                versions = get_versions_by_document_type("전력시장운영규칙")
                if not versions:
                    state["rag_result"] = (
                        "⚠️ 인덱싱된 문서가 없습니다. 사이드바에서 PDF를 먼저 업로드해주세요."
                    )
                    state["progress_steps"].append({
                        "step": 2, "name": "개정 버전 목록 조회", "status": "실패",
                        "detail": "인덱싱된 문서 없음",
                    })
                else:
                    lines = ["**📋 저장된 전력시장운영규칙 버전 목록**\n"]
                    for v in sorted(versions, key=lambda x: x["버전"]):
                        latest_mark = " ✅ (최신)" if v["is_latest"] else ""
                        lines.append(f"- **{v['버전']}**{latest_mark} — {v['chunk_count']:,}개 조문")
                    lines.append(
                        "\n특정 조문의 이력을 조회하려면 조문번호를 명시해 주세요.\n"
                        "예: \"제2.4.1조 개정 이력 알려줘\""
                    )
                    state["rag_result"] = "\n".join(lines)
                    state["progress_steps"].append({
                        "step": 2, "name": "개정 버전 목록 조회", "status": "완료",
                        "detail": f"{len(versions)}개 버전 반환",
                    })
                return state

            # ── 케이스 B: 조문번호 있음 → 해당 조문 개정 이력 ─────────────────
            history = self.history_manager.get_history(조문번호)

            if not history or not history.get("history"):
                state["error"] = {
                    "node": "rag_history_node",
                    "stage": "개정 이력 조회",
                    "reason": "해당 조문의 개정 이력을 찾을 수 없음",
                    "recoverable": True,
                    "message": (
                        f"'{조문번호}'의 개정 이력을 찾지 못했습니다. "
                        "조문번호를 정확히 입력해주세요 (예: 제2.4.1조)."
                    ),
                }
                state["rag_result"] = "검색결과없음"
                state["progress_steps"].append({
                    "step": 2, "name": "개정 이력 조회", "status": "실패",
                    "detail": f"{조문번호} 이력 없음",
                })
                return state

            history_text = f"조문: {history['regulation_num']}\n"
            history_text += f"개정 이력 ({len(history['history'])}건):\n"
            for item in history.get("history", []):
                history_text += f"- {item.get('시행일', 'N/A')}: {item.get('변경요지', 'N/A')}\n"

            state["progress_steps"].append({
                "step": 2, "name": "개정 이력 조회", "status": "완료",
                "detail": f"{조문번호} 이력 {len(history['history'])}건 조회 완료",
            })
            state["rag_result"] = history_text

        except Exception as e:
            logger.error(f"RAG History node error: {e}")
            state["error"] = {
                "node": "rag_history_node",
                "stage": "개정 이력 조회",
                "reason": str(e),
                "recoverable": False,
                "message": "이력 조회 중 시스템 오류가 발생했습니다.",
            }
            state["rag_result"] = ""

        return state
    
    # SMP 방향성 스코어링을 트리거하는 키워드. 이 키워드가 없으면 일반 데이터 조회(Mode B)로 처리한다
    # (예: "예비력 기준 초과했어?", "LNG 비중은?" 같은 질문이 항상 SMP 방향성 스코어만
    #  받아 질문과 무관한 답변이 나오던 경직성 문제를 해결하기 위해 추가)
    _SMP_DIRECTION_KEYWORDS = [
        "방향성", "방향", "올랐어", "내렸어", "상승", "하락",
        "보합", "전망", "추정", "브리핑"
    ]

    def _analysis_fixed_node(self, state: AgentState) -> AgentState:
        """정형 데이터 분석 진입점.

        Mode A: SMP 방향성 스코어링 (query에 방향성 관련 키워드가 있을 때, 기존 로직 그대로)
        Mode B: 그 외 일반 데이터 조회 (LNG 비중, 예비력 초과 여부 등 — 실제 데이터를 가져와
                LLM이 질문에 맞춰 답변, 방향성 스코어에 억지로 끼워맞추지 않음)
        """
        query = state.get("query", "")
        is_direction_query = any(kw in query for kw in self._SMP_DIRECTION_KEYWORDS)

        if is_direction_query:
            return self._run_smp_direction_scoring(state)
        return self._run_general_data_query(state)

    def _run_smp_direction_scoring(self, state: AgentState) -> AgentState:
        """SMP 방향성 추정 (정형 파이프라인) — 다중 날짜 지원. (기존 _analysis_fixed_node 로직 그대로)"""
        logger.info("[ANALYSIS] 날짜 계산 시작")

        try:
            # date_filter에서 날짜 리스트 계산 (없으면 오늘)
            date_filter = state.get("date_filter") or {"period_type": "today"}
            logger.info(f"[ANALYSIS] period_type={date_filter.get('period_type', '')}")
            date_list = resolve_date_range(date_filter)
            if not date_list:
                date_list = [datetime.now().strftime("%Y%m%d")]

            logger.info(
                f"[ANALYSIS] 분석 날짜: {date_list[:3]}{'...' if len(date_list) > 3 else ''}"
            )

            state["progress_steps"].append({
                "step": len(state["progress_steps"]) + 1,
                "name": "데이터 수집",
                "status": "진행중",
                "detail": f"기간: {date_list[0]} ~ {date_list[-1]} ({len(date_list)}일)"
            })

            # 단일 날짜: 기존 방식 (direction_estimator 사용)
            if len(date_list) == 1:
                target_date = date_list[0]
                direction_result = self.direction_estimator.estimate_direction(target_date)

                if direction_result.get("direction") == "분석불가":
                    state["error"] = {
                        "node": "analysis_fixed_node",
                        "stage": "데이터 수집",
                        "reason": f"{target_date} SMP/발전량 데이터 응답 없음 (공공데이터 API)",
                        "recoverable": True,
                        "message": f"{target_date} SMP 데이터를 가져오지 못했습니다. 다른 날짜로 다시 시도해주시겠어요?"
                    }
                    state["progress_steps"].append({
                        "step": len(state["progress_steps"]) + 1,
                        "name": "데이터 수집",
                        "status": "실패",
                        "detail": direction_result.get("error", "데이터 없음")
                    })
                    return state

                state["direction_result"] = direction_result

            else:
                # 다중 날짜: estimate_direction_batch로 병렬 처리
                batch_results = self.direction_estimator.estimate_direction_batch(date_list)

                success_dates = [d for d, r in batch_results.items() if r.get("direction") != "분석불가"]
                failed_dates  = [d for d, r in batch_results.items() if r.get("direction") == "분석불가"]

                if not success_dates:
                    state["error"] = {
                        "node": "analysis_fixed_node",
                        "stage": "데이터 수집",
                        "reason": f"조회 기간({date_list[0]}~{date_list[-1]}) 데이터 없음",
                        "recoverable": True,
                        "message": "해당 기간의 SMP 데이터를 가져오지 못했습니다. 다른 기간으로 다시 시도해주세요."
                    }
                    state["progress_steps"].append({
                        "step": len(state["progress_steps"]) + 1,
                        "name": "데이터 수집",
                        "status": "실패",
                        "detail": "전 기간 데이터 없음"
                    })
                    return state

                # 가장 최근 성공 날짜 기준으로 direction_result 사용 (date_list 정렬 순서에 의존하지 않음)
                state["direction_result"] = batch_results[max(success_dates)]

                detail_msg = f"{len(success_dates)}일 성공"
                if failed_dates:
                    detail_msg += f", {len(failed_dates)}일 실패 ({', '.join(failed_dates)})"
                state["progress_steps"].append({
                    "step": len(state["progress_steps"]) + 1,
                    "name": "데이터 수집",
                    "status": "완료",
                    "detail": detail_msg
                })

            state["progress_steps"].append({
                "step": len(state["progress_steps"]) + 1,
                "name": "데이터 수집",
                "status": "완료",
                "detail": "SMP/발전량 데이터 수집 완료"
            })

            # LLM 요약 (스코어링 결과 기반 — LLM이 방향성을 직접 판단하지 않음)
            prompt = PromptManager.get_smp_analysis_prompt()
            formatted_prompt = prompt.format(
                scoring_result=str(state["direction_result"])
            )
            response = self.llm.invoke(formatted_prompt)
            state["analysis_result"]["summary"] = response.content

            dr = state["direction_result"]
            state["progress_steps"].append({
                "step": len(state["progress_steps"]) + 1,
                "name": "방향성 분석",
                "status": "완료",
                "detail": f"방향성: {dr.get('direction', 'N/A')} {dr.get('direction_emoji', '')} (점수: {dr.get('score', 0)}/{dr.get('max_score', 3)})"
            })

        except Exception as e:
            logger.error(f"Analysis Fixed node error: {e}")
            state["error"] = {
                "node": "analysis_fixed_node",
                "stage": "데이터 수집/분석",
                "reason": str(e),
                "recoverable": False,
                "message": "SMP 분석 중 시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
            }
            state["progress_steps"].append({
                "step": len(state["progress_steps"]) + 1,
                "name": "데이터 수집/분석",
                "status": "실패",
                "detail": str(e)
            })

        return state

    def _run_general_data_query(self, state: AgentState) -> AgentState:
        """SMP 방향성 스코어링 대상이 아닌 일반 데이터 질의 처리.
        (예: "현재 LNG 발전 비중은?", "현재 전력수요가 예비력 기준 초과했어?")
        실제 API로 데이터를 수집하고 LLM이 질문에 맞춰 답변한다."""
        from domain.analysis.mcp_client import fetch_smp, fetch_generation, fetch_current_demand

        query = state.get("query", "")

        try:
            date_filter = state.get("date_filter") or {"period_type": "today"}
            date_list = resolve_date_range(date_filter)
            if not date_list:
                date_list = [datetime.now().strftime("%Y%m%d")]
            dates = date_list[-7:]  # 최대 최근 7일

            state["progress_steps"].append({
                "step": len(state["progress_steps"]) + 1,
                "name": "데이터 수집",
                "status": "진행중",
                "detail": f"기간: {dates[0]} ~ {dates[-1]} ({len(dates)}일)"
            })

            smp_dfs, gen_dfs = [], []
            for date in dates:
                sdf = fetch_smp(date)
                gdf = fetch_generation(date)
                if not sdf.empty:
                    smp_dfs.append(sdf)
                if not gdf.empty:
                    gen_dfs.append(gdf)
            demand_df = fetch_current_demand()

            if not smp_dfs and not gen_dfs and demand_df.empty:
                state["error"] = {
                    "node": "analysis_fixed_node",
                    "stage": "데이터 수집",
                    "reason": f"조회 기간({dates[0]}~{dates[-1]}) 데이터 없음 (공공데이터 API)",
                    "recoverable": True,
                    "message": "관련 데이터를 가져오지 못했습니다. 다른 기간으로 다시 시도해주시겠어요?"
                }
                state["progress_steps"].append({
                    "step": len(state["progress_steps"]) + 1,
                    "name": "데이터 수집",
                    "status": "실패",
                    "detail": "SMP/발전량/수급현황 데이터 전부 없음"
                })
                return state

            data_summary_lines = []
            if smp_dfs:
                smp_df = pd.concat(smp_dfs, ignore_index=True)
                data_summary_lines.append(f"SMP 평균: {smp_df['smp'].mean():.1f} 원/kWh")

            if gen_dfs:
                gen_df = pd.concat(gen_dfs, ignore_index=True)
                gen_by_source = gen_df.groupby("source")["gen_mw"].mean()
                total = gen_by_source.sum()
                data_summary_lines.append("발전원별 평균 발전량 및 비중:")
                for source, mw in gen_by_source.sort_values(ascending=False).items():
                    pct = (mw / total * 100) if total > 0 else 0
                    data_summary_lines.append(f"  {source}: {mw:,.0f}MW ({pct:.1f}%)")

            if not demand_df.empty:
                row = demand_df.iloc[0]
                data_summary_lines.append(
                    f"현재 전력수요: {row.get('demand_mw', 0):,.0f}MW / "
                    f"공급능력: {row.get('supply_mw', 0):,.0f}MW / "
                    f"예비력: {row.get('reserve_mw', 0):,.0f}MW "
                    f"(예비율 {row.get('reserve_rate', 0):.1f}%)"
                )

            data_summary = "\n".join(data_summary_lines)

            state["progress_steps"].append({
                "step": len(state["progress_steps"]) + 1,
                "name": "데이터 수집",
                "status": "완료",
                "detail": f"{dates[0]}~{dates[-1]} 데이터 수집 완료"
            })

            prompt = f"""질문: {query}

[수집된 실제 데이터]
{data_summary}

위 실제 데이터를 바탕으로 질문에 정확하게 답해줘.
수치를 직접 인용하면서 설명해줘.
데이터에 없는 내용(예: 규정상 기준값)은 추정/확인 필요라고 명시해줘.
"""
            response = self.llm.invoke(prompt)
            state["analysis_result"]["summary"] = response.content

            state["progress_steps"].append({
                "step": len(state["progress_steps"]) + 1,
                "name": "데이터 조회 완료",
                "status": "완료",
                "detail": query[:30]
            })

        except Exception as e:
            logger.error(f"General data query error: {e}")
            state["error"] = {
                "node": "analysis_fixed_node",
                "stage": "데이터 조회",
                "reason": str(e),
                "recoverable": False,
                "message": "데이터 조회 중 시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
            }

        return state

    def _analysis_plan_node(self, state: AgentState) -> AgentState:
        """비정형 데이터 분석 — multi_range(두 기간 이상 비교)는 실제 데이터 수집·비교,
        그 외는 LLM Planning (실패 시 Fixed Pipeline으로 폴백)"""
        logger.info("[ANALYSIS] 날짜 계산 시작")

        date_filter = state.get("date_filter") or {}
        logger.info(f"[ANALYSIS] period_type={date_filter.get('period_type', '')}")
        date_result = resolve_date_range(date_filter)

        if isinstance(date_result, dict) and date_result:
            labels = list(date_result.keys())
            logger.info(f"[ANALYSIS] 비교 기간: {labels}")
            return self._analysis_plan_multi_range(state, date_result)

        try:
            date_list = date_result if isinstance(date_result, list) else []
            if not date_list:
                from datetime import timedelta as _timedelta
                today = datetime.now()
                date_list = [
                    (today - _timedelta(days=i)).strftime("%Y%m%d")
                    for i in range(7, 0, -1)
                ]
                logger.info("[ANALYSIS_PLAN] 기간 특정 불가 → 최근 7일 기본값으로 원인 분석 진행")

            # 계획 텍스트만 생성하고 끝내지 않고, 실제 데이터를 수집해 원인을 분석한다
            state = self._run_cause_analysis(state, date_list)

            if state.get("error"):
                raise RuntimeError(state["error"].get("reason", "데이터 수집 실패"))

        except Exception as e:
            logger.warning(f"Analysis Plan node failed, falling back to fixed pipeline: {e}")
            state["progress_steps"].append({
                "step": len(state["progress_steps"]) + 1,
                "name": "원인 분석",
                "status": "실패",
                "detail": f"폴백 시도: {str(e)}"
            })

            # Fixed Pipeline으로 자동 폴백
            state["error"] = None
            state = self._analysis_fixed_node(state)

            # 폴백도 실패한 경우
            if state.get("error"):
                state["error"] = {
                    "node": "analysis_plan_node",
                    "stage": "비정형 데이터 분석",
                    "reason": f"원인 분석 실패 및 Fixed 파이프라인 폴백도 실패: {str(e)}",
                    "recoverable": True,
                    "message": "복잡한 분석에 실패해 기본 방식으로 재시도했지만 데이터를 가져오지 못했습니다."
                }
            else:
                state["progress_steps"].append({
                    "step": len(state["progress_steps"]) + 1,
                    "name": "분석 방식 전환",
                    "status": "완료",
                    "detail": "Fixed Pipeline으로 전환 성공"
                })

        return state

    def _run_cause_analysis(self, state: AgentState, date_list: List[str]) -> AgentState:
        """analysis_plan_node의 단일 기간 경로 — 계획 텍스트만 생성하던 것을
        실제 SMP/발전량 데이터 수집 + LLM 원인 분석 실행으로 대체한다."""
        from domain.analysis.mcp_client import fetch_smp, fetch_generation

        dates = date_list[-30:]  # 최대 30일

        smp_dfs, gen_dfs = [], []
        for date in dates:
            sdf = fetch_smp(date)
            gdf = fetch_generation(date)
            if not sdf.empty:
                smp_dfs.append(sdf)
            if not gdf.empty:
                gen_dfs.append(gdf)

        if not smp_dfs:
            state["error"] = {
                "node": "analysis_plan_node",
                "stage": "데이터 수집",
                "reason": f"조회 기간({dates[0]}~{dates[-1]}) SMP 데이터 없음",
                "recoverable": True,
                "message": "해당 기간의 SMP 데이터를 가져오지 못했습니다. 다른 기간으로 다시 시도해주세요."
            }
            return state

        smp_df = pd.concat(smp_dfs, ignore_index=True)
        gen_df = pd.concat(gen_dfs, ignore_index=True) if gen_dfs else pd.DataFrame()

        avg_smp = float(smp_df["smp"].mean())
        max_smp = float(smp_df["smp"].max())
        min_smp = float(smp_df["smp"].min())

        gen_summary = ""
        if not gen_df.empty and "source" in gen_df.columns:
            gen_by_source = gen_df.groupby("source")["gen_mw"].mean()
            total = gen_by_source.sum()
            lines = []
            for source, mw in gen_by_source.sort_values(ascending=False).items():
                pct = (mw / total * 100) if total > 0 else 0
                lines.append(f"{source}: {mw:,.0f}MW ({pct:.1f}%)")
            gen_summary = "\n".join(lines)

        state["progress_steps"].append({
            "step": len(state["progress_steps"]) + 1,
            "name": "데이터 수집",
            "status": "완료",
            "detail": f"기간: {dates[0]} ~ {dates[-1]} ({len(dates)}일) | SMP {len(smp_df)}건 수집"
        })

        prompt = f"""질문: {state.get('query', '')}

[수집된 실제 데이터 — {dates[0]}~{dates[-1]}]
SMP 평균: {avg_smp:.1f} 원/kWh
SMP 최고: {max_smp:.1f} 원/kWh
SMP 최저: {min_smp:.1f} 원/kWh
발전원별 평균 발전량 및 비중:
{gen_summary if gen_summary else "(발전량 데이터 없음)"}

위 실제 데이터를 바탕으로 질문에 답해줘.
추측이나 일반적인 설명이 아니라 위에 제시된 실제 수치를 근거로 인용해서 설명해줘.
데이터에 없는 내용은 추정이라고 명시해줘.
"""
        try:
            response = self.llm.invoke(prompt)
            analysis_text = response.content
        except Exception as e:
            logger.error(f"원인 분석 LLM 호출 실패: {e}")
            analysis_text = (
                f"SMP 평균 {avg_smp:.1f}원/kWh, 최고 {max_smp:.1f}원/kWh, "
                f"최저 {min_smp:.1f}원/kWh (자연어 요약 생성 실패)"
            )

        state["analysis_result"]["summary"] = analysis_text
        state["analysis_result"]["avg_smp"] = round(avg_smp, 1)
        state["analysis_result"]["max_smp"] = round(max_smp, 1)
        state["analysis_result"]["min_smp"] = round(min_smp, 1)

        state["progress_steps"].append({
            "step": len(state["progress_steps"]) + 1,
            "name": "원인 분석",
            "status": "완료",
            "detail": f"SMP 평균 {avg_smp:.1f}원/kWh 기준 분석 완료"
        })

        return state

    def _analysis_plan_multi_range(self, state: AgentState, date_result: Dict[str, List[str]]) -> AgentState:
        """두 기간 이상 비교 — 기간별로 실제 SMP 데이터를 수집해 통계를 비교한다"""
        period_dfs: Dict[str, pd.DataFrame] = {}
        failed_periods: List[str] = []

        for label, dates in date_result.items():
            df = fetch_smp_range(dates[0], dates[-1]) if dates else pd.DataFrame()
            if not df.empty:
                period_dfs[label] = df
            else:
                failed_periods.append(label)

        detail = f"성공: {', '.join(period_dfs.keys()) if period_dfs else '없음'}"
        if failed_periods:
            detail += f" / 실패: {', '.join(failed_periods)}"
        state["progress_steps"].append({
            "step": len(state["progress_steps"]) + 1,
            "name": "기간별 데이터 수집",
            "status": "완료" if period_dfs else "실패",
            "detail": detail
        })

        if not period_dfs:
            state["error"] = {
                "node": "analysis_plan_node",
                "stage": "기간별 데이터 수집",
                "reason": f"비교 대상 전체 기간({', '.join(date_result.keys())}) 데이터 없음",
                "recoverable": True,
                "message": "비교하려는 기간의 SMP 데이터를 가져오지 못했습니다. 다른 기간으로 다시 시도해주세요."
            }
            return state

        comparison = {
            label: {
                "평균SMP": round(float(df["smp"].mean()), 2),
                "최고SMP": round(float(df["smp"].max()), 2),
                "최저SMP": round(float(df["smp"].min()), 2),
            }
            for label, df in period_dfs.items()
        }
        state["analysis_result"]["comparison"] = comparison

        state["progress_steps"].append({
            "step": len(state["progress_steps"]) + 1,
            "name": "기간 비교 분석",
            "status": "완료",
            "detail": "; ".join(f"{label} 평균 {stats['평균SMP']}원" for label, stats in comparison.items())
        })

        try:
            prompt = PromptManager.get_period_comparison_prompt()
            formatted_prompt = prompt.format(comparison_result=str(comparison))
            response = self.llm.invoke(formatted_prompt)
            state["analysis_result"]["summary"] = response.content
        except Exception as e:
            logger.warning(f"기간 비교 요약 생성 실패: {e}")

        return state
    
    def _complex_node(self, state: AgentState) -> AgentState:
        """규정 + 데이터 분석 복합 (부분 성공 허용)"""
        logger.info("=== Complex Node ===")

        # RAG 검색 시도
        state["error"] = None
        state = self._rag_node(state)
        rag_error = state.get("error")

        # 분석 시도 (이전 에러 클리어 후)
        state["error"] = None
        state = self._analysis_fixed_node(state)
        analysis_error = state.get("error")

        # 부분 성공 처리
        if rag_error and analysis_error:
            state["error"] = {
                "node": "complex_node",
                "stage": "복합 분석",
                "reason": f"규정 검색 실패: {rag_error['reason']} / 데이터 분석 실패: {analysis_error['reason']}",
                "recoverable": True,
                "message": "규정 정보와 데이터 분석을 모두 가져오지 못했습니다."
            }
        elif rag_error:
            # RAG만 실패 → 분석 결과만 사용 (부분 성공)
            state["error"] = None
            state["rag_result"] = "⚠️ 규정 정보 검색에 실패했습니다."
            state["progress_steps"].append({
                "step": len(state["progress_steps"]) + 1,
                "name": "복합 분석",
                "status": "부분 성공",
                "detail": f"데이터 분석 성공 / 규정 검색 실패: {rag_error['reason']}"
            })
        elif analysis_error:
            # 분석만 실패 → RAG 결과만 사용 (부분 성공)
            state["error"] = None
            state["progress_steps"].append({
                "step": len(state["progress_steps"]) + 1,
                "name": "복합 분석",
                "status": "부분 성공",
                "detail": f"규정 검색 성공 / 데이터 분석 실패: {analysis_error['reason']}"
            })
        else:
            state["error"] = None

        return state
    
    # ===== Agent Planning 노드 =====

    def _planning_node(self, state: AgentState) -> AgentState:
        """LLM이 질문을 분석해서 실행 계획을 수립하는 Node."""
        logger.info("[PLANNING] LLM Plan 수립 시작")

        system_prompt = f"""당신은 전력시장 AI 어시스턴트의 실행 계획 수립 담당자입니다.
사용자 질문을 분석해서 필요한 스킬과 실행 순서를 결정하세요.

════════════════════════════════════
■ 사용 가능한 스킬 4가지
════════════════════════════════════

rag_skill:
  현행 전력시장운영규칙 조문 검색
  → "방식", "기준", "규정", "절차", "정의", "뭐야?" 표현 시 사용

analysis_skill:
  SMP/발전량/수요 실제 데이터 수집 및 분석
  → "지금", "현재", "오늘", "어제", "수치", "얼마", "이유", "원인" 표현 시 사용

diff_skill:
  규정 개정 전/후 비교
  → "개정", "바뀐 것", "달라진 것", "이번에 변경" 표현 시 사용
  → rag_skill 대신 사용 (개정 비교는 diff_skill이 담당)

history_skill:
  특정 조문 개정 이력 조회
  → 특정 조문번호 + "언제 바뀌었어", "개정 이력" 표현 시 사용
  → rag_skill 대신 사용 (이력 조회는 history_skill이 담당)

════════════════════════════════════
■ 스킬 선택 필수 규칙
════════════════════════════════════

규칙 1. 실시간 데이터가 필요하면 → analysis_skill 반드시 포함
  "지금", "현재", "오늘", "어제", "최근", "실제로" 표현

규칙 2. 현행 규정 내용이 필요하면 → rag_skill 반드시 포함
  "규정은", "기준이", "방식은", "절차는", "정의는" 표현
  단, 개정 비교라면 diff_skill로 대체

규칙 3. 데이터 + 현행 규정 동시 필요 → rag_skill + analysis_skill, parallel=true
  "규정상 문제없어?", "기준 초과했어?", "맞는 수준이야?"
  "데이터랑 규정으로", "근거로 설명해줘"

규칙 4. 개정 비교 + 데이터 영향 → diff_skill + analysis_skill, parallel=true
  "개정 내용이 실제로 영향 줬어?"
  "바뀐 규정 후에 SMP가 어떻게 됐어?"

규칙 5. 개정 이력 + 현행 내용 → history_skill + rag_skill, parallel=true
  "이 조문 몇 번 바뀌었고 현재는 어떻게 돼?"

규칙 6. parallel 기준
  True: 두 스킬이 서로의 결과에 의존하지 않을 때 (대부분의 복합 질의)
  False: 앞 스킬 결과가 뒤 스킬에 필요할 때 (거의 없음)

════════════════════════════════════
■ Few-Shot 예시
════════════════════════════════════

Q: "어제 SMP 급등 이유를 데이터랑 규정 근거로 설명해줘"
→ skills=["rag_skill", "analysis_skill"], parallel=true
   이유: 규정 근거(rag) + 실제 데이터 분석(analysis) 동시 필요

Q: "지금 LNG 비중이 규정상 문제없는 수준이야?"
→ skills=["rag_skill", "analysis_skill"], parallel=true
   이유: 현행 LNG 규정(rag) + 현재 LNG 실측치(analysis) 둘 다 필요

Q: "현재 전력수요가 예비력 기준 초과했어?"
→ skills=["rag_skill", "analysis_skill"], parallel=true
   이유: 예비력 기준 규정(rag) + 현재 수요 실측치(analysis) 둘 다 필요

Q: "이번 개정 내용이 SMP에 실제로 영향 줬어?"
→ skills=["diff_skill", "analysis_skill"], parallel=true
   이유: 개정 전후 비교(diff) + 개정 전후 SMP 데이터 비교(analysis) 필요

Q: "급전순위 규정 이번에 바뀐 거 있어?"
→ skills=["diff_skill"], parallel=false
   이유: 규정 개정 비교만 필요, 데이터 불필요

Q: "제2.4.1조 언제 바뀌었고 현재 내용은?"
→ skills=["history_skill", "rag_skill"], parallel=true
   이유: 개정 이력(history) + 현행 조문 내용(rag) 동시 필요

════════════════════════════════════
■ 출력 형식 (JSON만, 다른 텍스트 없이)
════════════════════════════════════

{{
    "skills_needed": ["스킬명1", "스킬명2"],
    "parallel": true,
    "execution_order": [["스킬명1", "스킬명2"], ["merge"]],
    "reason": "판단 근거 한 줄"
}}"""

        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"질문: {state['query']}"},
            ])
            json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
            plan = json.loads(json_match.group())
        except Exception as e:
            logger.warning(f"Planning LLM 파싱 실패, 기본값 사용: {e}")
            plan = {
                "skills_needed": ["rag_skill", "analysis_skill"],
                "parallel": True,
                "execution_order": [["rag_skill", "analysis_skill"], ["merge"]],
                "reason": f"Plan 파싱 실패 → 기본값(RAG+분석 병렬) 사용",
            }

        logger.info(f"[PLANNING] skills={plan.get('skills_needed', [])}")
        logger.info(f"[PLANNING] parallel={plan.get('parallel', False)}")
        logger.info(f"[PLANNING] 판단근거: {plan.get('reason', '')}")

        state["plan"] = plan
        state["completed_skills"] = []
        state["progress_steps"].append({
            "step": len(state["progress_steps"]) + 1,
            "name": "실행 계획 수립 (Agent Planning)",
            "status": "완료",
            "detail": (
                f"필요 스킬: {plan['skills_needed']}\n"
                f"실행 방식: {'병렬' if plan['parallel'] else '순차'}\n"
                f"판단 근거: {plan['reason']}"
            ),
        })
        return state

    def _executor_node(self, state: AgentState) -> AgentState:
        """Plan을 읽고 실행 방식을 결정하는 Node."""
        plan = state.get("plan") or {}
        skills = plan.get("skills_needed", ["rag_skill", "analysis_skill"])
        parallel = plan.get("parallel", True)

        logger.info(f"[EXECUTOR] {'병렬' if parallel else '순차'} 실행 시작 → {skills}")

        state["progress_steps"].append({
            "step": len(state["progress_steps"]) + 1,
            "name": "스킬 실행",
            "status": "진행중",
            "detail": f"{'병렬' if parallel else '순차'} 실행 시작: {skills}",
        })
        return state

    def _executor_router(self, state: AgentState):
        """executor_node 이후 라우팅 — parallel이면 Send API로 동시 실행"""
        plan = state.get("plan") or {}
        skills = plan.get("skills_needed", ["rag_skill", "analysis_skill"])
        parallel = plan.get("parallel", True)

        valid_skills = [s for s in skills if s in SKILL_TO_NODE]
        if not valid_skills:
            return "merge_node"

        if parallel and len(valid_skills) > 1:
            # 병렬 실행: Send API로 동시 실행
            return [
                Send(SKILL_TO_NODE[skill], {**state, "current_skill": skill})
                for skill in valid_skills
            ]
        else:
            # 순차 실행: 첫 번째 스킬 Node로 라우팅 (나머지는 merge_node에서 처리)
            first_skill = valid_skills[0]
            return SKILL_TO_NODE[first_skill]

    def _merge_node(self, state: AgentState) -> AgentState:
        """병렬/순차 실행된 스킬 결과를 통합하는 Node."""
        has_rag = bool(state.get("rag_result"))
        has_analysis = bool(state.get("analysis_result") or state.get("direction_result"))
        logger.info(
            f"[MERGE] rag_result={'✅' if has_rag else '❌'} | "
            f"analysis_result={'✅' if has_analysis else '❌'}"
        )
        logger.info("[MERGE] 결과 통합 완료")

        detail_parts = []
        if state.get("rag_result"):
            detail_parts.append("규정 검색 결과 수신")
        if state.get("analysis_result") or state.get("direction_result"):
            detail_parts.append("데이터 분석 결과 수신")

        state["progress_steps"].append({
            "step": len(state["progress_steps"]) + 1,
            "name": "결과 통합",
            "status": "완료",
            "detail": "\n".join(detail_parts) if detail_parts else "통합 완료",
        })
        return state

    def _clarify_node(self, state: AgentState) -> AgentState:
        """추가 질문 요청 — classify_node가 설정한 clarify_message 우선 사용"""
        logger.info("=== Clarify Node ===")

        # classify_node가 준비한 구체적 메시지가 있으면 LLM 호출 없이 바로 사용
        if state.get("clarify_message"):
            state["final_answer"] = state["clarify_message"]
            state["needs_clarification"] = True
            return state

        try:
            prompt = PromptManager.get_clarification_prompt()
            formatted_prompt = prompt.format(
                user_query=state["query"],
                reason="질문이 모호하여 명확한 의도 파악이 어렵습니다."
            )

            response = self.llm.invoke(formatted_prompt)
            state["final_answer"] = response.content
            state["needs_clarification"] = True

        except Exception as e:
            logger.error(f"Clarify node error: {e}")
            state["final_answer"] = "질문 내용을 좀 더 구체적으로 말씀해주시겠어요?"

        return state
    
    def _format_output_node(self, state: AgentState) -> AgentState:
        """출력 포맷팅 — error 존재 시 오류 전용 템플릿 출력"""
        logger.info("[FORMAT] 최종 답변 생성 시작")

        try:
            # 오류 전용 템플릿 (error 필드 존재 시)
            if state.get("error"):
                error = state["error"]
                logger.error(f"[ERROR] node={error.get('node')} | stage={error.get('stage')}")
                logger.error(f"[ERROR] reason={error.get('reason')}")
                logger.error(f"[ERROR] recoverable={error.get('recoverable')}")
                user_msg = error.get("message", "")
                if not user_msg:
                    user_msg = ("잠시 후 다시 시도하거나 관리자에게 문의해주세요."
                                if not error.get("recoverable")
                                else "다시 한번 시도해주세요.")

                output = (
                    f"⚠️ 처리 중 문제가 발생했습니다\n\n"
                    f"📍 어느 단계: {error.get('stage', '알 수 없음')}\n"
                    f"📍 무엇이 문제: {error.get('reason', '알 수 없는 오류')}\n\n"
                    f"💡 이렇게 해보세요\n{user_msg}\n\n"
                    f"📎 면책\n본 시스템은 참고용이며, 오류 발생 시 직접 데이터를 확인하시기 바랍니다."
                )
                state["final_answer"] = output
                state["progress_steps"].append({
                    "step": len(state["progress_steps"]) + 1,
                    "name": "오류 출력",
                    "status": "완료",
                    "detail": f"오류 단계: {error.get('stage', '')}"
                })
                elapsed = round(time.time() - (state.get("_start_time") or time.time()), 1)
                logger.info(f"[FORMAT] 답변 생성 완료 | 소요시간: {elapsed}초")
                logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                return state

            # 정상 출력
            rag_result = state.get("rag_result", "")
            direction_result = state.get("direction_result") or {}
            analysis_result = state.get("analysis_result") or {}
            has_rag = bool(rag_result) and rag_result not in ("검색결과없음", "")
            has_analysis = bool(direction_result.get("direction")) or bool(analysis_result)

            # complex 등에서 rag_result와 analysis_result/direction_result가 모두 채워진 경우
            # 규정 근거와 데이터 분석을 하나의 답변으로 통합 (한쪽만 보여주고 버리지 않도록)
            if has_rag and has_analysis:
                output = self._format_complex_answer(
                    rag_result=rag_result,
                    analysis_result=analysis_result,
                    direction_result=direction_result,
                    query=state.get("query", ""),
                )
            elif direction_result.get("direction"):
                output = self._format_direction_result(direction_result)
                summary = analysis_result.get("summary", "")
                if summary:
                    output += f"\n\n🔄 분석 포인트\n{summary}"
            elif has_rag:
                output = rag_result
            elif analysis_result.get("comparison"):
                output = self._format_period_comparison(analysis_result["comparison"])
                summary = analysis_result.get("summary", "")
                if summary:
                    output += f"\n\n🔄 분석 포인트\n{summary}"
            elif analysis_result.get("summary"):
                output = analysis_result["summary"]
            elif analysis_result:
                output = str(analysis_result)
            else:
                output = "처리 결과를 생성하지 못했습니다."

            output += "\n\n📎 면책\n본 답변은 참고용이며, 최종 판단은 담당자 확인이 필요합니다."
            state["final_answer"] = output

            state["progress_steps"].append({
                "step": len(state["progress_steps"]) + 1,
                "name": "출력 포맷팅",
                "status": "완료",
                "detail": "최종 답변 생성 완료"
            })

        except Exception as e:
            logger.error(f"Format output node error: {e}")
            state["final_answer"] = (
                f"⚠️ 처리 중 문제가 발생했습니다\n\n"
                f"📍 어느 단계: 출력 포맷팅\n"
                f"📍 무엇이 문제: {str(e)}\n\n"
                f"💡 이렇게 해보세요\n잠시 후 다시 시도해주세요.\n\n"
                f"📎 면책\n본 시스템은 참고용이며, 오류 발생 시 직접 데이터를 확인하시기 바랍니다."
            )

        elapsed = round(time.time() - (state.get("_start_time") or time.time()), 1)
        logger.info(f"[FORMAT] 답변 생성 완료 | 소요시간: {elapsed}초")
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return state

    def _format_complex_answer(
        self,
        rag_result: str,
        analysis_result: Dict,
        direction_result: Dict,
        query: str,
    ) -> str:
        """규정 근거(rag_result) + 데이터 분석(analysis_result/direction_result)을
        하나의 답변으로 통합. LLM을 한 번 더 호출해 두 결과를 자연스럽게 연결한다.

        complex 질의에서 규정 검색과 데이터 분석이 모두 성공했는데도
        format_output_node가 direction_result만 보여주고 rag_result를 버리던
        병합 버그를 해결하기 위해 추가된 경로."""
        data_summary = ""
        if direction_result.get("direction"):
            data_summary = (
                f"방향성: {direction_result.get('direction')} "
                f"{direction_result.get('direction_emoji', '')} "
                f"({direction_result.get('score', 0)}/{direction_result.get('max_score', 3)}점)\n"
                f"지표: {direction_result.get('indicators', {})}"
            )
            if analysis_result.get("summary"):
                data_summary += f"\n분석 요약: {analysis_result['summary']}"
        elif analysis_result:
            data_summary = str(analysis_result)

        prompt = f"""사용자 질문: {query}

[데이터 분석 결과]
{data_summary}

[관련 규정 조문]
{rag_result}

위 두 가지 정보를 바탕으로 질문에 대한 통합 답변을 작성해줘.
- 먼저 데이터 분석 결과를 설명하고
- 그 다음 관련 규정 근거를 인용해서 설명해줘
- 두 내용이 자연스럽게 연결되도록 작성해줘
- 근거 없는 내용을 추가로 지어내지 말고, 위에 제공된 데이터/조문 범위 안에서만 답변해줘
"""
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Complex answer merge failed: {e}")
            # LLM 병합 실패 시에도 두 결과를 모두 보여주기 (최소한 데이터 유실은 없게)
            direction_text = self._format_direction_result(direction_result) if direction_result.get("direction") else data_summary
            return f"{direction_text}\n\n📖 관련 규정\n{rag_result}"

    def _format_direction_result(self, result: Dict) -> str:
        """방향성 결과 포맷팅"""
        output = f"📊 SMP 방향성 분석\n"
        output += f"방향성: {result['direction']} {result['direction_emoji']}\n"
        output += f"스코어: {result['score']}/{result['max_score']}\n"
        return output

    def _format_period_comparison(self, comparison: Dict[str, Dict[str, float]]) -> str:
        """기간별 SMP 비교 결과를 표 형태로 포맷팅"""
        lines = ["📊 기간별 SMP 비교", "", "| 기간 | 평균SMP | 최고SMP | 최저SMP |", "|------|---------|---------|---------|"]
        for label, stats in comparison.items():
            lines.append(
                f"| {label} | {stats.get('평균SMP', 'N/A')} | {stats.get('최고SMP', 'N/A')} | {stats.get('최저SMP', 'N/A')} |"
            )
        return "\n".join(lines)
    
    def _resolve_followup_query(self, query: str, conversation_history: List[Dict]) -> str:
        """clarify(재질문) 직후의 짧은 후속 답변("지난 3일" 등)을 원래 보류 중이던
        질문과 합쳐 하나의 완결된 질문으로 재구성한다.

        문제: 직전 답변이 "어느 기간을 분석해드릴까요?" 같은 clarify 재질문이었을 때,
        사용자가 "지난 3일"처럼 기간만 답하면 그 문장 단독으로는 주제(예: LNG 비중,
        SMP 등)가 전혀 없어 intent 분류도, 이후 데이터 분석 프롬프트도 원래 질문을
        전혀 알지 못한 채 처리되어 버린다. 직전 turn이 clarify였을 때만 원래 질문과
        이번 답변을 합쳐 state["query"] 자체를 재구성함으로써, 분류/분석 양쪽 모두
        하나의 완결된 질문을 보게 만든다.
        """
        if not conversation_history:
            return query

        # conversation_history 마지막이 이번 턴 사용자 메시지 자신이면 그 이전부터 탐색
        history = conversation_history
        if history and history[-1].get("role") == "user" and history[-1].get("content") == query:
            history = history[:-1]

        if not history or history[-1].get("role") != "assistant":
            return query

        last_assistant = history[-1]
        if last_assistant.get("intent") != "clarify":
            return query

        # clarify를 유발한 직전 사용자 질문을 원래 질문으로 간주
        for msg in reversed(history[:-1]):
            if msg.get("role") == "user":
                original_question = msg.get("content", "")
                if original_question and original_question != query:
                    merged = f"{original_question} ({query})"
                    logger.info(
                        f"[FOLLOWUP] clarify 후속 답변 병합: '{query}' + "
                        f"'{original_question}' -> '{merged}'"
                    )
                    return merged
                break

        return query

    def run(self, query: str, conversation_history: List = None) -> Dict:
        """
        워크플로우 실행

        Args:
            query (str): 사용자 질문
            conversation_history (List): 대화 이력

        Returns:
            Dict: 최종 답변
        """
        conversation_history = conversation_history or []
        effective_query = self._resolve_followup_query(query, conversation_history)

        initial_state = AgentState(
            query=effective_query,
            intent="",
            confidence=0.0,
            search_filter={},
            rag_result="",
            analysis_result={},
            direction_result={},
            final_answer="",
            needs_clarification=False,
            conversation_history=conversation_history or [],
            progress_steps=[],
            error=None,
            date_filter={"period_type": "today", "n_days": None, "start_date": None, "end_date": None},
            clarify_message="",
            plan=None,
            completed_skills=[],
            _start_time=time.time(),
        )
        
        # 그래프 실행
        result = self.graph.invoke(initial_state)
        
        return {
            "answer": result.get("final_answer", ""),
            "intent": result.get("intent", ""),
            "progress": result.get("progress_steps", []),
            "needs_clarification": result.get("needs_clarification", False),
            "plan": result.get("plan"),
        }


def get_graph_router() -> GraphRouter:
    """
    그래프 라우터 인스턴스 반환
    
    Returns:
        GraphRouter: 그래프 라우터 인스턴스
    """
    return GraphRouter()
