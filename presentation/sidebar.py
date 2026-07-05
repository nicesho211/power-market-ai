"""
사이드바 모듈

Streamlit 사이드바를 구성합니다.
설정, 도움말, 데이터 소스 정보 등을 제공합니다.
PDF 업로드 및 다중 버전 인덱싱 UI를 포함합니다.
"""

import streamlit as st
import threading
import time
from config.settings import validate_settings
from domain.rag.vector_store import ensure_collection, get_collection_stats
import logging

logger = logging.getLogger(__name__)

# ── 배경 스레드 ↔ Streamlit 메인 스레드 공유 상태 ─────────────────────────
# st.session_state는 배경 스레드에서 쓰면 StopException 발생.
# 순수 Python dict는 GIL로 보호되므로 단일 사용자 환경에서 안전하게 사용 가능.
_IDX: dict = {
    "running":  False,
    "done":     False,
    "filename": "",
    "progress": {},
    "result":   None,
}


def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        # ── 로고 ──────────────────────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center; padding: 20px 0 16px;">
            <div style="font-size: 2.2rem; line-height:1;">⚡</div>
            <div style="font-size: 1.05rem; font-weight: 700; margin-top: 6px;
                background: linear-gradient(135deg, #1E40AF, #2563EB, #0EA5E9);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                background-clip: text;">
                Power Market AI
            </div>
            <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 2px;
                letter-spacing: 0.5px;">
                전력시장 AI 어시스턴트
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── 인덱싱 상태 카드 ──────────────────────────────────────────────
        try:
            ensure_collection()
            stats = get_collection_stats()
            if stats["total_chunks"] > 0:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #F0FDF4, #ECFDF5);
                    border: 1px solid #BBF7D0; border-radius: 10px;
                    padding: 14px 16px; margin-bottom: 8px;
                    box-shadow: 0 1px 4px rgba(16,185,129,0.08);">
                    <div style="color: #059669; font-weight: 600; font-size: 0.82rem;
                        text-transform: uppercase; letter-spacing: 0.5px;">
                        ✅ 인덱싱 완료
                    </div>
                    <div style="color: #065F46; font-size: 1.5rem;
                        font-weight: 700; margin: 4px 0 2px; letter-spacing: -0.5px;">
                        {stats['total_chunks']:,}개 조문
                    </div>
                    <div style="color: #6B7280; font-size: 0.73rem;">
                        최신 버전: {stats['latest_version'] or '알 수 없음'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #FFFBEB; border: 1px solid #FDE68A;
                    border-radius: 10px; padding: 14px 16px; margin-bottom: 8px;">
                    <div style="color: #D97706; font-weight: 600; font-size: 0.82rem;">
                        ⚠️ 인덱싱된 문서 없음
                    </div>
                    <div style="color: #78716C; font-size: 0.8rem; margin-top: 4px;">
                        아래에서 PDF를 업로드해주세요
                    </div>
                </div>
                """, unsafe_allow_html=True)
        except Exception:
            pass

        st.markdown("---")
        st.header("📚 전력시장 AI 어시스턴트")

        st.markdown("""
        **기능:**
        - 규정 조회: 전력시장운영규칙 Q&A
        - 개정 비교: 개정 전후 규정 비교
        - 이력 조회: 규정 개정 이력
        - 분석: SMP 방향성 추정
        """)

        st.markdown("---")

        # 설정 검증
        st.subheader("💾 설정 상태")
        validation = validate_settings()

        if validation["valid"]:
            st.success("✅ 모든 설정이 정상입니다.")
        else:
            st.error("⚠️ 설정 오류 발생")
            for error in validation["errors"]:
                st.error(f"  • {error}")

        st.markdown("---")

        # PDF 업로드 섹션 (다중 버전 지원)
        _render_pdf_upload_section()

        st.markdown("---")

        # 데이터 소스
        st.subheader("📊 데이터 소스")
        st.info("""
        **공공데이터포털 연동:**
        - SMP (계통한계가격)
        - 발전원별 발전량
        - 전력수급현황

        **규정 데이터:**
        - 전력시장운영규칙 PDF
        """)

        st.markdown("---")

        # 도움말
        st.subheader("❓ 사용 방법")

        with st.expander("예제 질문"):
            st.markdown("""
            **규정 조회:**
            - "제2.4.1조가 뭔가요?"
            - "계통한계가격이란?"

            **개정 비교:**
            - "이번 개정에서 뭐가 바뀌었어?"

            **이력 조회:**
            - "이 조항이 몇 번 개정됐어?"

            **SMP 분석:**
            - "오늘 SMP 방향성 어때?"
            - "지난 3일치 SMP 동향 알려줘"
            """)

        st.markdown("---")

        # 면책 문구
        st.caption("""
        ⚠️ **면책**
        본 서비스는 참고용이며,
        최종 판단은 담당자 확인이 필수입니다.
        """)


def _render_pdf_upload_section():
    """PDF 업로드 및 인덱싱 UI.
    인덱싱 중에는 실시간 진행률 화면으로 전환되고,
    완료되면 결과를 표시한 뒤 업로드 UI로 돌아온다.
    """
    st.subheader("📄 규정 PDF 업로드")

    # ── 인덱싱 진행 중 → 진행률 화면 표시 ─────────────────────────────────
    if _IDX["running"]:
        _show_live_progress()
        return

    # 현재 저장된 버전 목록 참고 표시
    try:
        stats = get_collection_stats()
        if stats["versions"]:
            st.caption(f"저장된 버전: {', '.join(stats['versions'])}")
    except Exception:
        pass

    uploaded_file = st.file_uploader(
        "전력시장운영규칙 또는 정산세부규정 PDF를 업로드하세요",
        type=["pdf"],
        key="pdf_uploader"
    )

    if uploaded_file is not None:
        # 버전 날짜 입력 (사용자 직접 입력)
        version_input = st.text_input(
            "📅 버전 날짜 (시행일 기준, YYYY-MM-DD)",
            placeholder="예: 2026-05-20",
            help="PDF 파일의 시행일을 입력하세요. 비워두면 파일명에서 자동 추출합니다.",
            key="version_date_input",
        )

        # 문서 종류 자동 감지 (첫 페이지 텍스트로 미리보기)
        try:
            file_bytes = uploaded_file.read()
            import tempfile, os, pymupdf4llm
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                preview_text = pymupdf4llm.to_markdown(tmp_path, pages=[0])
            except Exception:
                preview_text = ""
            finally:
                os.unlink(tmp_path)

            from domain.rag.document_loader import detect_document_type
            auto_type = detect_document_type(uploaded_file.name, preview_text)

        except Exception:
            file_bytes = uploaded_file.getvalue()
            auto_type = "기타"

        st.info(f"📋 감지된 문서 종류: **{auto_type}**")

        doc_type_options = ["전력시장운영규칙", "정산세부규정", "기타"]
        default_idx = doc_type_options.index(auto_type) if auto_type in doc_type_options else 2

        doc_type_override = st.selectbox(
            "문서 종류 확인/수정",
            options=doc_type_options,
            index=default_idx,
            key="doc_type_select"
        )

        # 날짜 형식 검증
        import re as _re
        version_ok = not version_input or bool(_re.match(r"^\d{4}-\d{2}-\d{2}$", version_input.strip()))
        if version_input and not version_ok:
            st.error("날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력하세요.")

        if st.button("🚀 업로드 및 인덱싱 시작", key="start_indexing", disabled=not version_ok):
            _bytes    = file_bytes
            _filename = uploaded_file.name
            _doctype  = doc_type_override
            _version  = version_input.strip() if version_input and version_ok else None

            # 버전 날짜 기준 중복 체크
            _should_index = True
            if _version:
                _stats = get_collection_stats()
                if _version in _stats["versions"]:
                    st.warning(f"⚠️ {_version} 버전이 이미 인덱싱되어 있습니다.")
                    st.info("다른 날짜 버전을 업로드하거나, 관리자 설정에서 초기화 후 다시 시도하세요.")
                    _should_index = False

            if _should_index:
                _IDX["running"]  = True
                _IDX["done"]     = False
                _IDX["filename"] = _filename
                _IDX["result"]   = None
                _IDX["progress"] = {
                    "step": "extract", "current": 0, "total": 1,
                    "message": "⏳ 인덱싱 준비 중..."
                }

                def _bg():
                    from application.indexing_service import run_indexing_stream
                    try:
                        for update in run_indexing_stream(_bytes, _filename, _doctype, _version):
                            if update["type"] == "progress":
                                _IDX["progress"] = update
                            elif update["type"] == "result":
                                _IDX["result"] = update
                    except Exception as e:
                        _IDX["result"] = {
                            "success": False, "message": f"오류: {e}",
                            "chunk_count": 0, "failed_chunks": 0,
                            "detection_rate": 0.0, "version": "", "elapsed_seconds": 0
                        }
                    finally:
                        _IDX["done"] = True

                threading.Thread(target=_bg, daemon=True).start()
                st.rerun()

    # ── 관리자: 컬렉션 초기화 ──────────────────────────────────────────────
    st.markdown("---")
    with st.expander("⚠️ 관리자 설정"):
        st.warning("아래 버튼을 누르면 모든 인덱싱 데이터가 삭제됩니다.")
        confirm_reset = st.checkbox("모든 인덱싱 데이터를 삭제하겠습니다", key="confirm_reset")
        if st.button("🗑️ 전체 데이터 초기화", type="secondary",
                     key="reset_collection_btn", disabled=not confirm_reset):
            try:
                from domain.rag.vector_store import reset_collection
                reset_collection()
                st.success("✅ 초기화 완료. PDF를 다시 업로드해주세요.")
                st.rerun()
            except Exception as e:
                st.error(f"초기화 실패: {e}")


def _show_live_progress() -> None:
    """
    배경 스레드 인덱싱 진행 중 실시간 표시.
    _IDX dict를 읽고 0.5초마다 st.rerun()으로 UI를 갱신.
    """
    st.caption(f"📄 {_IDX['filename']}")

    # ── 완료 ─────────────────────────────────────────────────────────────
    if _IDX["done"]:
        _IDX["running"] = False
        result = _IDX.get("result") or {}

        if result.get("success"):
            st.success(
                f"✅ 인덱싱 완료!  \n"
                f"청크: **{result['chunk_count']:,}개** | "
                f"버전: {result.get('version', '')} | "
                f"소요: {result.get('elapsed_seconds', '?')}초"
            )
            if result.get("failed_chunks", 0) > 0:
                st.warning(f"⚠️ 저장 실패(건너뜀): {result['failed_chunks']:,}개")
            if result.get("detection_rate", 1.0) < 0.5:
                st.warning(
                    f"⚠️ 조문 감지율 낮음 ({result['detection_rate']:.1%}) — "
                    "검색 품질이 낮을 수 있습니다."
                )
        else:
            st.error(f"❌ {result.get('message', '알 수 없는 오류')}")
        return

    # ── 진행 중 ───────────────────────────────────────────────────────────
    prog    = _IDX["progress"]
    step    = prog.get("step", "")
    current = prog.get("current", 0)
    total   = prog.get("total", 1)
    message = prog.get("message", "⏳ 처리 중...")

    pct = min(current / total, 1.0) if total > 0 else 0.0
    st.progress(pct)

    if step == "embed" and total > 0:
        st.markdown(f"🔢 임베딩: **{current:,} / {total:,}개** ({pct:.0%})")
    else:
        st.markdown(message)

    # 0.5초 후 재실행 → _IDX 최신값 표시
    time.sleep(0.5)
    st.rerun()


def show_instructions():
    """지침 표시"""
    st.info("""
    📋 **사용 지침**
    
    이 어시스턴트는 다음을 제공합니다:
    
    1. **규정 Q&A** - 전력시장운영규칙 기반 질의응답
    2. **개정 비교** - 규정 개정 전후 비교
    3. **이력 조회** - 조문별 개정 이력
    4. **SMP 분석** - 공개 데이터 기반 SMP 방향성 추정
    
    모든 답변에는 근거 출처가 포함됩니다.
    """)
