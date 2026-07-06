"""
LangGraph 워크플로우 라우터

의도 분류 결과에 따른 다양한 노드 및 엣지를 관리합니다.
AgentState 기반으로 상태를 추적합니다.
"""

from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph
from langgraph.types import Send
import logging
import json
import re
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

logger = logging.getLogger(__name__)


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
        "description": "전력시장운영규칙 조문 검색 및 Q&A",
        "node": "rag_node",
        "output_field": "rag_result",
    },
    "analysis_skill": {
        "description": "SMP/발전량/수요 데이터 수집 및 방향성 분석",
        "node": "analysis_fixed_node",
        "output_field": "analysis_result",
    },
    "diff_skill": {
        "description": "규정 개정 전/후 조문 비교",
        "node": "rag_diff_node",
        "output_field": "rag_result",
    },
    "history_skill": {
        "description": "특정 조문의 전체 개정 이력 조회",
        "node": "rag_history_node",
        "output_field": "rag_result",
    },
}

SKILL_TO_NODE: Dict[str, str] = {name: info["node"] for name, info in SKILLS.items()}


class AgentState(TypedDict):
    """LangGraph 에이전트 상태"""
    query: str
    intent: str
    confidence: float
    search_filter: dict
    rag_result: str
    analysis_result: dict
    direction_result: dict
    final_answer: str
    needs_clarification: bool
    conversation_history: List[Dict]
    progress_steps: List[Dict]
    # 에러 핸들링 필드 (A섹션)
    error: Optional[Dict]
    # 날짜 필터 필드 (B섹션)
    date_filter: Optional[Dict]
    clarify_message: str
    # Agent Planning 관련 신규 필드
    plan: Optional[Dict]
    # {"skills_needed": [...], "parallel": bool, "execution_order": [...], "reason": str}
    completed_skills: List[str]


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
        logger.info("=== Classify Node ===")

        try:
            classification = self.classifier.classify(state["query"])

            state["intent"] = classification["intent"]
            state["confidence"] = classification["confidence"]
            state["search_filter"] = classification.get("search_filter", {})
            state["date_filter"] = classification.get("date_filter", {
                "period_type": "not_applicable",
                "n_days": None,
                "start_date": None,
                "end_date": None
            })

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
        logger.info("=== RAG Node ===")

        try:
            # Step 2: 문서 검색
            results = self.retriever.search(
                query=state["query"],
                where=state["search_filter"] or {"is_latest": True}
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
    
    def _analysis_fixed_node(self, state: AgentState) -> AgentState:
        """SMP 방향성 추정 (정형 파이프라인) — 다중 날짜 지원"""
        logger.info("=== Analysis Fixed Node ===")

        try:
            # date_filter에서 날짜 리스트 계산 (없으면 오늘)
            date_filter = state.get("date_filter") or {"period_type": "today"}
            date_list = resolve_date_range(date_filter)
            if not date_list:
                date_list = [datetime.now().strftime("%Y%m%d")]

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
    
    def _analysis_plan_node(self, state: AgentState) -> AgentState:
        """비정형 데이터 분석 — multi_range(두 기간 이상 비교)는 실제 데이터 수집·비교,
        그 외는 LLM Planning (실패 시 Fixed Pipeline으로 폴백)"""
        logger.info("=== Analysis Plan Node ===")

        date_filter = state.get("date_filter") or {}
        date_result = resolve_date_range(date_filter)

        if isinstance(date_result, dict) and date_result:
            return self._analysis_plan_multi_range(state, date_result)

        try:
            prompt = PromptManager.get_planning_analyzer_prompt()
            formatted_prompt = prompt.format(user_query=state["query"])
            response = self.llm_planning.invoke(formatted_prompt)

            if not response.content or len(response.content.strip()) < 10:
                raise ValueError("LLM Planning 결과가 비정상입니다 (응답 너무 짧음)")

            state["analysis_result"]["plan"] = response.content
            state["progress_steps"].append({
                "step": len(state["progress_steps"]) + 1,
                "name": "분석 계획 수립",
                "status": "완료",
                "detail": "LLM Planning 완료"
            })

        except Exception as e:
            logger.warning(f"Analysis Plan node failed, falling back to fixed pipeline: {e}")
            state["progress_steps"].append({
                "step": len(state["progress_steps"]) + 1,
                "name": "분석 계획 수립",
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
                    "reason": f"LLM Planning 실패 및 Fixed 파이프라인 폴백도 실패: {str(e)}",
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
        logger.info("=== Planning Node ===")

        skills_desc = "\n".join(
            f"- {name}: {info['description']}" for name, info in SKILLS.items()
        )
        system_prompt = f"""당신은 전력시장 AI 어시스턴트의 실행 계획 수립 담당자입니다.
사용자 질문을 분석해서 필요한 스킬과 실행 순서를 결정하세요.

사용 가능한 스킬:
{skills_desc}

병렬 실행 조건: 두 스킬이 서로의 결과에 의존하지 않을 때
순차 실행 조건: 앞 스킬의 결과가 뒤 스킬의 입력으로 필요할 때

반드시 아래 JSON 형식으로만 답하세요. 다른 텍스트 없이 JSON만:
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
        logger.info("=== Executor Node ===")
        plan = state.get("plan") or {}
        skills = plan.get("skills_needed", ["rag_skill", "analysis_skill"])
        parallel = plan.get("parallel", True)

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
        logger.info("=== Merge Node ===")
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
        logger.info("=== Format Output Node ===")

        try:
            # 오류 전용 템플릿 (error 필드 존재 시)
            if state.get("error"):
                error = state["error"]
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
                return state

            # 정상 출력 (기존 로직 유지)
            if state.get("direction_result") and state["direction_result"].get("direction"):
                output = self._format_direction_result(state["direction_result"])
                summary = state.get("analysis_result", {}).get("summary", "")
                if summary:
                    output += f"\n\n🔄 분석 포인트\n{summary}"
            elif state.get("rag_result") and state["rag_result"] not in ("검색결과없음", ""):
                output = state["rag_result"]
            elif state.get("analysis_result", {}).get("comparison"):
                output = self._format_period_comparison(state["analysis_result"]["comparison"])
                summary = state.get("analysis_result", {}).get("summary", "")
                if summary:
                    output += f"\n\n🔄 분석 포인트\n{summary}"
            elif state.get("analysis_result"):
                output = str(state["analysis_result"])
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

        return state
    
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
    
    def run(self, query: str, conversation_history: List = None) -> Dict:
        """
        워크플로우 실행
        
        Args:
            query (str): 사용자 질문
            conversation_history (List): 대화 이력
            
        Returns:
            Dict: 최종 답변
        """
        initial_state = AgentState(
            query=query,
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
        )
        
        # 그래프 실행
        result = self.graph.invoke(initial_state)
        
        return {
            "answer": result.get("final_answer", ""),
            "intent": result.get("intent", ""),
            "progress": result.get("progress_steps", []),
            "needs_clarification": result.get("needs_clarification", False)
        }


def get_graph_router() -> GraphRouter:
    """
    그래프 라우터 인스턴스 반환
    
    Returns:
        GraphRouter: 그래프 라우터 인스턴스
    """
    return GraphRouter()
