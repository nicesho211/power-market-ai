"""
차트 표시 모듈

Plotly 차트를 Streamlit에 표시합니다.
"""

import streamlit as st
from domain.analysis.chart_builder import get_chart_builder
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def render_chart_view():
    """차트 뷰 렌더링"""
    st.header("📈 데이터 시각화")
    
    chart_builder = get_chart_builder()
    
    # 탭 구성
    tab1, tab2, tab3 = st.tabs(["SMP 차트", "발전량 차트", "비교 분석"])
    
    with tab1:
        st.subheader("SMP 및 수요 예측")
        
        # 날짜 선택
        selected_date = st.date_input(
            "날짜 선택",
            value=datetime.now().date(),
            max_value=datetime.now().date()
        )
        
        date_str = selected_date.strftime("%Y%m%d")
        
        try:
            fig = chart_builder.build_smp_chart(date_str)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"차트 생성 실패: {str(e)}")
    
    with tab2:
        st.subheader("발전원별 발전량")
        
        # 날짜 선택
        selected_date = st.date_input(
            "날짜 선택",
            value=datetime.now().date(),
            max_value=datetime.now().date(),
            key="gen_date"
        )
        
        date_str = selected_date.strftime("%Y%m%d")
        
        try:
            fig = chart_builder.build_generation_chart(date_str)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"차트 생성 실패: {str(e)}")
    
    with tab3:
        st.subheader("SMP 일별 비교")
        
        # 비교 날짜 선택
        num_days = st.slider("비교 일수", min_value=2, max_value=7, value=3)
        
        today = datetime.now().date()
        date_list = [
            (today - timedelta(days=i)).strftime("%Y%m%d")
            for i in range(num_days)
        ]
        
        try:
            fig = chart_builder.build_comparative_chart(date_list)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"차트 생성 실패: {str(e)}")


def render_direction_gauge(score: int, max_score: int = 3):
    """방향성 게이지 표시"""
    chart_builder = get_chart_builder()
    
    try:
        fig = chart_builder.build_indicator_gauge(score, max_score)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"게이지 생성 실패: {str(e)}")
