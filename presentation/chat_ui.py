"""
채팅 UI 모듈

Streamlit을 사용하여 채팅 UI를 구성합니다.
사용자 입력, 메시지 표시, 대화 이력 관리를 담당합니다.
"""

import streamlit as st
from typing import List, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def render_chat_ui(router):
    """
    채팅 UI 렌더링
    
    Args:
        router: GraphRouter 인스턴스
    """
    st.header("💬 채팅")
    
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # 진행 단계 표시
            if "progress" in message and message["progress"]:
                with st.expander("📊 처리 과정 보기"):
                    for step in message["progress"]:
                        icon = "✅" if step["status"] == "완료" else "❌"
                        st.caption(f'{icon} Step {step["step"]}. {step["name"]}')
                        if step.get("detail"):
                            for line in step["detail"].split("\n"):
                                if line.strip():
                                    st.caption(f"   → {line.strip()}")
    
    # 사용자 입력
    user_input = st.chat_input("질문을 입력하세요...")
    
    if user_input:
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # 대화 이력에 추가
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("분석 중..."):
                try:
                    # 라우터 실행
                    result = router.run(
                        user_input,
                        conversation_history=st.session_state.messages
                    )
                    
                    # 응답 표시
                    st.markdown(result["answer"])
                    
                    # 진행 단계 표시
                    if result.get("progress"):
                        with st.expander("📊 처리 과정 보기"):
                            for step in result["progress"]:
                                icon = "✅" if step["status"] == "완료" else "❌"
                                st.caption(f'{icon} Step {step["step"]}. {step["name"]}')
                                if step.get("detail"):
                                    for line in step["detail"].split("\n"):
                                        if line.strip():
                                            st.caption(f"   → {line.strip()}")
                    
                    # 대화 이력에 추가
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "progress": result.get("progress", []),
                        "intent": result.get("intent", ""),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    error_msg = f"❌ 오류 발생: {str(e)}\n\n다시 시도해주세요."
                    st.error(error_msg)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": datetime.now().isoformat()
                    })


def render_conversation_sidebar():
    """대화 이력 표시"""
    with st.sidebar:
        st.markdown("---")
        st.subheader("💬 대화 이력")
        
        if "messages" in st.session_state and st.session_state.messages:
            # 최근 대화 5개 표시
            for message in st.session_state.messages[-10:]:
                if message["role"] == "user":
                    st.caption(f"👤 {message['content'][:50]}...")
        else:
            st.caption("대화 이력이 없습니다.")
        
        # 초기화 버튼
        if st.button("🗑️ 대화 이력 초기화"):
            st.session_state.messages = []
            st.rerun()
