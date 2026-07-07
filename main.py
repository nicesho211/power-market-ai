"""
메인 앱 (Streamlit)

전력시장 AI 어시스턴트의 메인 엔트리 포인트입니다.
"""

from config.logging_config import setup_logging
setup_logging()

import streamlit as st
from presentation.sidebar import render_sidebar, show_instructions
from presentation.chat_ui import render_chat_ui, render_conversation_sidebar
from presentation.chart_view import render_chart_view
from application.graph_router import get_graph_router
from config.settings import validate_settings
from domain.rag.vector_store import ensure_collection
from infrastructure.llm_client import get_llm, get_embeddings
import logging

logger = logging.getLogger("APP")

# Streamlit 페이지 설정
st.set_page_config(
    page_title="⚡ Power Market AI Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

LIGHT_CSS = """
<style>
/* ── 전역 ──────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    background-color: #F8FAFC;
    color: #0F172A;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── 사이드바 ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
    box-shadow: 2px 0 8px rgba(0,0,0,0.04);
}
[data-testid="stSidebar"] * { color: #334155; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #0F172A !important; }

/* ── 지표 카드 ───────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 4px 16px rgba(37,99,235,0.06);
    transition: box-shadow 0.2s;
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.08), 0 8px 24px rgba(37,99,235,0.10);
}
[data-testid="stMetricValue"] {
    color: #2563EB !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}
[data-testid="stMetricLabel"] { color: #64748B !important; font-size: 0.8rem !important; }
[data-testid="stMetricDelta"] svg { display: none; }

/* ── 채팅 메시지 ─────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: #FFFFFF;
    border: 1px solid #F1F5F9;
    border-radius: 14px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
[data-testid="stChatMessage"][data-testid*="user"] {
    background: #EFF6FF;
    border-color: #DBEAFE;
}

/* ── 채팅 입력창 ─────────────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    background-color: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 12px !important;
    color: #0F172A !important;
    font-size: 0.95rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}

/* ── 버튼 ────────────────────────────────────────────────── */
.stButton > button {
    background: #2563EB;
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.875rem;
    padding: 8px 18px;
    transition: background 0.2s, transform 0.15s, box-shadow 0.2s;
    box-shadow: 0 1px 3px rgba(37,99,235,0.3);
}
.stButton > button:hover {
    background: #1D4ED8;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(37,99,235,0.35);
}
.stButton > button:active { transform: translateY(0); }

/* ── 탭 ──────────────────────────────────────────────────── */
[data-testid="stTabs"] button {
    color: #64748B;
    font-weight: 500;
    border-bottom: 2px solid transparent;
    transition: color 0.2s;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #2563EB !important;
    border-bottom-color: #2563EB !important;
    font-weight: 600;
}

/* ── 익스팬더 ─────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* ── 상태 컨테이너 ───────────────────────────────────────── */
[data-testid="stStatus"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
}

/* ── h1 그라디언트 ───────────────────────────────────────── */
h1 {
    background: linear-gradient(135deg, #1E40AF 0%, #2563EB 50%, #0EA5E9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800;
    letter-spacing: -0.5px;
}

/* ── 구분선 ──────────────────────────────────────────────── */
hr { border-color: #F1F5F9 !important; }

/* ── 스크롤바 ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #F8FAFC; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }

/* ── 알림/info/warning ──────────────────────────────────── */
[data-testid="stInfo"] {
    background: #EFF6FF;
    border-left-color: #2563EB;
    color: #1E3A8A;
}
[data-testid="stWarning"] {
    background: #FFFBEB;
    border-left-color: #F59E0B;
    color: #92400E;
}
[data-testid="stSuccess"] {
    background: #F0FDF4;
    border-left-color: #10B981;
    color: #065F46;
}

/* ── 파일 업로더 ─────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1.5px dashed #CBD5E1;
    border-radius: 10px;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #2563EB;
}

/* ── selectbox / dropdown ───────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: #FFFFFF;
    border: 1.5px solid #CBD5E1;
    border-radius: 8px;
    color: #0F172A;
}

/* ── 프로그레스 바 ───────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #2563EB, #0EA5E9);
    border-radius: 4px;
}
[data-testid="stProgress"] > div {
    background: #E2E8F0;
    border-radius: 4px;
}
</style>
"""


@st.cache_resource(show_spinner="⚡ Power Market AI 초기화 중...")
def init_resources():
    """앱 시작 시 Qdrant 컬렉션 확인 + LLM/임베딩 클라이언트 + SMP 임계값 웜업"""
    logger.info("⚡ Power Market AI Assistant 시작")
    ensure_collection()
    get_llm()
    get_embeddings()
    _warm_up_smp_thresholds()
    logger.info("✅ 리소스 초기화 완료")
    return True


def _warm_up_smp_thresholds():
    """SMP 방향성 임계값(과거 7일 baseline)을 앱 시작 시 미리 계산해둔다.
    이걸 안 하면 첫 SMP 분석 질의가 baseline 계산 때문에 느려진다."""
    from domain.analysis.direction_estimator import get_direction_estimator
    try:
        get_direction_estimator().warm_up()
    except Exception:
        logger.exception("SMP 임계값 warm-up 실패 (첫 질의 시 재계산됩니다)")


def render_top_metrics():
    """상단 실시간 지표 카드 3개 렌더링"""
    from domain.analysis.data_fetcher import fetch_smp, fetch_current_demand
    from datetime import datetime

    today = datetime.now().strftime("%Y%m%d")
    col1, col2, col3 = st.columns(3)

    try:
        smp_df = fetch_smp(today)
        latest_smp = float(smp_df["smp"].iloc[-1]) if not smp_df.empty else 0.0
        prev_smp = float(smp_df["smp"].iloc[-2]) if len(smp_df) > 1 else latest_smp
    except Exception:
        latest_smp, prev_smp = 0.0, 0.0

    try:
        demand_df = fetch_current_demand()
        demand_mw = float(demand_df["demand_mw"].iloc[0]) if not demand_df.empty else 0.0
    except Exception:
        demand_mw = 0.0

    with col1:
        st.metric(
            "⚡ 현재 SMP",
            f"{latest_smp:.1f} 원/kWh" if latest_smp else "데이터 없음",
            f"{latest_smp - prev_smp:+.1f}" if latest_smp else None,
        )
    with col2:
        st.metric(
            "🔋 현재 전력수요",
            f"{demand_mw:,.0f} MW" if demand_mw else "데이터 없음",
        )
    with col3:
        direction = st.session_state.get("last_direction", "분석 필요")
        emoji = st.session_state.get("last_direction_emoji", "❓")
        st.metric("📊 SMP 방향성", f"{emoji} {direction}")


def main():
    """메인 앱"""

    # 리소스 초기화 (Qdrant 컬렉션 확인 + LLM 웜업)
    init_resources()

    # 라이트 테마 CSS 주입
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)

    # 상단 실시간 지표 카드
    render_top_metrics()
    st.divider()

    # 사이드바 렌더링
    render_sidebar()
    
    # 설정 검증
    validation = validate_settings()
    if not validation["valid"]:
        st.error("🚨 설정 오류로 인해 앱을 실행할 수 없습니다.")
        st.info("**.env** 파일을 확인하고 필수 API 키를 설정해주세요.")
        return
    
    # 메인 탭
    tab1, tab2, tab3 = st.tabs(["💬 채팅", "📈 시각화", "ℹ️ 정보"])
    
    with tab1:
        st.header("⚡ 전력시장 AI 어시스턴트")
        st.markdown("""
        전력시장운영규칙 기반 규정 Q&A와 SMP 방향성 추정을 제공합니다.
        모든 답변에는 근거와 출처가 포함됩니다.
        """)
        
        # 라우터 초기화
        try:
            router = get_graph_router()
            render_chat_ui(router)
        except Exception as e:
            st.error(f"❌ 앱 초기화 실패: {str(e)}")
            logger.exception("App initialization failed")
        
        # 대화 이력 사이드바
        render_conversation_sidebar()
    
    with tab2:
        try:
            render_chart_view()
        except Exception as e:
            st.error(f"❌ 차트 로딩 실패: {str(e)}")
            logger.exception("Chart rendering failed")
    
    with tab3:
        st.header("서비스 정보")
        
        st.subheader("📌 개요")
        st.markdown("""
        **전력시장 AI 어시스턴트**는 전력거래 실무 담당자를 위한 챗봇으로,
        다음 기능을 제공합니다:
        
        - 🏛️ **규정 조회**: 전력시장운영규칙 기반 Q&A
        - 📜 **개정 비교**: 규정 개정 전후 비교
        - 📋 **이력 조회**: 조문별 개정 이력
        - 📊 **SMP 분석**: 공개 데이터 기반 방향성 추정
        """)
        
        st.subheader("🔧 기술 스택")
        st.markdown("""
        - **LLM**: GPT-5.4 (Azure OpenAI)
        - **Agent Framework**: LangGraph + Agent Planning
        - **Vector DB**: Qdrant (로컬/클라우드 자동 전환)
        - **Search**: Hybrid (벡터 60% + BM25 40%)
        - **Embeddings**: text-embedding-3-large (3072차원)
        - **UI**: Streamlit (다크 테마)
        - **Visualization**: Plotly
        """)
        
        st.subheader("📊 데이터 소스")
        st.markdown("""
        - **공공데이터포털**: SMP, 발전원별 발전량, 전력수급현황
        - **규정 데이터**: 전력시장운영규칙 PDF
        """)
        
        st.subheader("⚠️ 면책 사항")
        st.warning("""
        - 본 서비스는 참고용입니다.
        - 최종 판단은 담당자 확인이 필수입니다.
        - 근거 없는 환각 답변을 방지하기 위해 모든 답변에 출처를 표시합니다.
        """)
        
        st.subheader("💡 사용 팁")
        st.markdown("""
        **더 나은 결과를 위해:**
        1. 구체적인 조문번호나 용어를 사용하세요.
        2. 규정 질의는 "제X.X.X조", "정산규정" 등 명확히 표현하세요.
        3. 분석 요청은 "SMP 분석", "발전 추이" 등으로 요청하세요.
        """)


if __name__ == "__main__":
    main()
