"""
차트 빌더

Plotly를 사용하여 SMP 및 발전량 데이터를 시각화합니다.
Streamlit과 통합하여 대시보드용 차트를 생성합니다.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
import logging
from domain.analysis.mcp_client import fetch_smp, fetch_generation

logger = logging.getLogger(__name__)


DARK_LAYOUT = {
    "paper_bgcolor": "#FFFFFF",
    "plot_bgcolor": "#F8FAFC",
    "font": {"color": "#334155", "family": "Inter, -apple-system, sans-serif"},
    "xaxis": {
        "gridcolor": "#F1F5F9",
        "zerolinecolor": "#E2E8F0",
        "linecolor": "#E2E8F0",
        "tickfont": {"color": "#64748B", "size": 11},
    },
    "yaxis": {
        "gridcolor": "#F1F5F9",
        "zerolinecolor": "#E2E8F0",
        "linecolor": "#E2E8F0",
        "tickfont": {"color": "#64748B", "size": 11},
    },
    "legend": {
        "bgcolor": "rgba(255,255,255,0.9)",
        "bordercolor": "#E2E8F0",
        "borderwidth": 1,
        "font": {"color": "#334155"},
    },
}

SOURCE_COLORS = {
    "원자력": "#2563EB",
    "LNG":   "#F59E0B",
    "유연탄": "#94A3B8",
    "신재생": "#10B981",
    "태양광": "#EAB308",
    "수력":  "#0EA5E9",
    "유류":  "#F97316",
    "양수":  "#8B5CF6",
    "국내탄": "#6B7280",
}


def _hex_to_rgba(hex_color: str, alpha: float = 0.7) -> str:
    """#RRGGBB → rgba(r,g,b,a) 변환"""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _base_layout() -> dict:
    """xaxis/yaxis 제외한 DARK_LAYOUT 복사본 반환 (update_layout 중복 키 방지)"""
    return {k: v for k, v in DARK_LAYOUT.items() if k not in ("xaxis", "yaxis")}


class ChartBuilder:
    """차트 빌더"""

    def __init__(self):
        pass
    
    def build_smp_chart(self, date: str) -> go.Figure:
        """
        일일 SMP 차트 구성
        
        Args:
            date (str): 날짜 (YYYYMMDD)
            
        Returns:
            go.Figure: Plotly Figure 객체
        """
        try:
            df = fetch_smp(date)
            
            if df.empty:
                return self._get_empty_figure("No data available")
            
            fig = go.Figure()

            # SMP 라인 (블루)
            fig.add_trace(go.Scatter(
                x=df["hour"],
                y=df["smp"],
                mode="lines+markers",
                name="SMP (원/kWh)",
                line=dict(color="#2563EB", width=2.5),
                marker=dict(size=5, color="#2563EB"),
                fill="tozeroy",
                fillcolor="rgba(37,99,235,0.07)",
                yaxis="y",
            ))

            # 평균선 (앰버 점선)
            if not df["smp"].empty:
                avg_smp = df["smp"].mean()
                fig.add_hline(
                    y=avg_smp,
                    line=dict(color="#F59E0B", dash="dash", width=1.5),
                    annotation_text=f"평균 {avg_smp:.1f}",
                    annotation_font_color="#92400E",
                )

            # 수요 예측 (보조 축, 에메랄드)
            fig.add_trace(go.Scatter(
                x=df["hour"],
                y=df["forecast_demand"],
                mode="lines+markers",
                name="수요 예측 (MW)",
                line=dict(color="#10B981", width=2, dash="dot"),
                marker=dict(size=4, color="#10B981"),
                yaxis="y2",
            ))

            layout = {
                **DARK_LAYOUT,
                "title": dict(text=f"SMP 및 전력수요 추이 ({date})",
                              font=dict(color="#0F172A", size=15, family="Inter, sans-serif")),
                "xaxis": {**DARK_LAYOUT["xaxis"], "title": dict(text="시간대", font=dict(color="#64748B"))},
                "yaxis": {
                    **DARK_LAYOUT["yaxis"],
                    "title": dict(text="SMP (원/kWh)", font=dict(color="#2563EB")),
                    "tickfont": dict(color="#2563EB", size=11),
                },
                "yaxis2": dict(
                    title=dict(text="전력수요 (MW)", font=dict(color="#10B981")),
                    tickfont=dict(color="#10B981", size=11),
                    overlaying="y",
                    side="right",
                    gridcolor="#F1F5F9",
                    linecolor="#E2E8F0",
                ),
                "hovermode": "x unified",
                "height": 480,
                "margin": dict(l=60, r=60, t=50, b=50),
            }
            fig.update_layout(**layout)
            
            return fig
        except Exception as e:
            logger.error(f"Failed to build SMP chart: {e}")
            return self._get_empty_figure(f"Error: {str(e)}")
    
    def build_generation_chart(self, date: str) -> go.Figure:
        """
        발전원별 발전량 차트 구성
        
        Args:
            date (str): 날짜 (YYYYMMDD)
            
        Returns:
            go.Figure: Plotly Figure 객체
        """
        try:
            df = fetch_generation(date)
            
            if df.empty:
                return self._get_empty_figure("No generation data available")
            
            # 발전원별로 피벗
            pivot_df = df.pivot_table(
                index="hour",
                columns="source",
                values="gen_mw",
                fill_value=0
            )
            
            fig = go.Figure()

            for source in pivot_df.columns:
                color = SOURCE_COLORS.get(source, "#94A3B8")
                fig.add_trace(go.Scatter(
                    x=pivot_df.index,
                    y=pivot_df[source],
                    mode="lines",
                    name=source,
                    stackgroup="one",
                    line=dict(color=color, width=1),
                    fillcolor=_hex_to_rgba(color) if color.startswith("#") else color,
                ))

            fig.update_layout(
                **_base_layout(),
                title=dict(text=f"발전원별 발전량 추이 ({date})",
                           font=dict(color="#0F172A", size=15, family="Inter, sans-serif")),
                xaxis={**DARK_LAYOUT["xaxis"], "title": dict(text="시간대", font=dict(color="#64748B"))},
                yaxis={**DARK_LAYOUT["yaxis"], "title": dict(text="발전량 (MW)", font=dict(color="#64748B"))},
                hovermode="x unified",
                height=480,
                margin=dict(l=60, r=20, t=50, b=50),
            )
            
            return fig
        except Exception as e:
            logger.error(f"Failed to build generation chart: {e}")
            return self._get_empty_figure(f"Error: {str(e)}")
    
    def build_comparative_chart(
        self,
        date_list: List[str]
    ) -> go.Figure:
        """
        여러 날짜의 SMP를 비교하는 차트
        
        Args:
            date_list (List[str]): 날짜 리스트
            
        Returns:
            go.Figure: 비교 차트
        """
        try:
            neon_palette = ["#00D4FF", "#00FF94", "#FF6B35", "#FFD700", "#4A9EFF", "#9B59B6", "#FF8C00"]
            fig = go.Figure()

            for i, date in enumerate(date_list):
                df = fetch_smp(date)
                if not df.empty:
                    color = neon_palette[i % len(neon_palette)]
                    fig.add_trace(go.Scatter(
                        x=df["hour"],
                        y=df["smp"],
                        mode="lines+markers",
                        name=date,
                        line=dict(color=color, width=2),
                    ))

            fig.update_layout(
                **_base_layout(),
                title=dict(text="SMP 일별 비교",
                           font=dict(color="#0F172A", size=15, family="Inter, sans-serif")),
                xaxis={**DARK_LAYOUT["xaxis"], "title": dict(text="시간대", font=dict(color="#64748B"))},
                yaxis={**DARK_LAYOUT["yaxis"], "title": dict(text="SMP (원/kWh)", font=dict(color="#64748B"))},
                hovermode="x unified",
                height=480,
                margin=dict(l=60, r=20, t=50, b=50),
            )
            
            return fig
        except Exception as e:
            logger.error(f"Failed to build comparative chart: {e}")
            return self._get_empty_figure(f"Error: {str(e)}")
    
    def build_indicator_gauge(self, score: int, max_score: int = 3) -> go.Figure:
        """
        SMP 방향성 스코어 게이지 차트
        
        Args:
            score (int): 현재 스코어
            max_score (int): 최대 스코어
            
        Returns:
            go.Figure: 게이지 차트
        """
        percentage = (score / max_score) * 100 if max_score > 0 else 0
        
        bar_color = "#10B981" if score >= 2 else "#EF4444" if score == 0 else "#2563EB"
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            title={"text": "SMP 방향성 스코어", "font": {"color": "#0F172A", "size": 15}},
            number={"font": {"color": "#2563EB", "size": 40}},
            delta={
                "reference": max_score / 2,
                "increasing": {"color": "#10B981"},
                "decreasing": {"color": "#EF4444"},
            },
            gauge={
                "axis": {"range": [0, max_score], "tickcolor": "#94A3B8",
                         "tickfont": {"color": "#64748B"}},
                "bar": {"color": bar_color},
                "bgcolor": "#F1F5F9",
                "bordercolor": "#E2E8F0",
                "steps": [
                    {"range": [0, max_score / 3], "color": "#FEE2E2"},
                    {"range": [max_score / 3, 2 * max_score / 3], "color": "#DBEAFE"},
                ],
                "threshold": {
                    "line": {"color": "#F59E0B", "width": 3},
                    "thickness": 0.75,
                    "value": max_score,
                },
            },
        ))

        fig.update_layout(
            **_base_layout(),
            height=380,
            margin=dict(l=30, r=30, t=40, b=20),
        )
        return fig
    
    def _get_source_color(self, source: str) -> str:
        """발전원별 다크 테마 색상 코드"""
        return SOURCE_COLORS.get(source, "#94A3B8")
    
    def _get_empty_figure(self, message: str) -> go.Figure:
        """빈 차트 반환"""
        fig = go.Figure()
        fig.add_annotation(text=message, showarrow=False, font=dict(color="#94A3B8", size=14))
        fig.update_layout(**_base_layout(), height=300)
        return fig


def get_chart_builder() -> ChartBuilder:
    """
    차트 빌더 인스턴스 반환
    
    Returns:
        ChartBuilder: 차트 빌더 인스턴스
    """
    return ChartBuilder()
