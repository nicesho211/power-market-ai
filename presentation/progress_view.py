"""
진행 상황 표시 모듈

Streamlit의 st.status를 활용하여 각 처리 단계를 시각적으로 표시합니다.
"""

import streamlit as st
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def render_progress_view(progress_steps: List[Dict]):
    """
    진행 상황을 시각적으로 표시
    
    Args:
        progress_steps (List[Dict]): 진행 단계 리스트
    """
    if not progress_steps:
        return
    
    with st.status("분석 중...", expanded=True) as status:
        for step in progress_steps:
            step_num = step.get("step", 0)
            step_name = step.get("name", "")
            step_status = step.get("status", "완료")
            step_detail = step.get("detail", "")
            
            # 상태 아이콘
            if step_status == "완료":
                icon = "✅"
                status_text = "완료"
            elif step_status == "진행중":
                icon = "⏳"
                status_text = "진행중"
            else:
                icon = "❌"
                status_text = "실패"
            
            # 단계 표시
            st.write(f'{icon} Step {step_num}. {step_name}')
            
            # 상세 정보 표시
            if step_detail:
                for line in step_detail.split("\n"):
                    if line.strip():
                        st.caption(f"   → {line.strip()}")
        
        # 마지막 단계가 완료면 상태 업데이트
        if progress_steps and progress_steps[-1].get("status") == "완료":
            status.update(label="분석 완료!", state="complete")


def render_simple_progress(message: str, progress_value: float = None):
    """
    간단한 진행률 표시
    
    Args:
        message (str): 메시지
        progress_value (float): 진행률 (0.0~1.0)
    """
    if progress_value is not None:
        st.progress(progress_value, text=message)
    else:
        with st.spinner(message):
            pass
