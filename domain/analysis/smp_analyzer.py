"""
SMP 분석기

SMP, 발전량, 수요 데이터를 종합적으로 분석합니다.
"""

import pandas as pd
from typing import Dict, List
import logging
from domain.analysis.mcp_client import fetch_smp, fetch_generation, fetch_current_demand
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SMPAnalyzer:
    """SMP 분석"""
    
    def __init__(self):
        """분석기 초기화"""
        pass
    
    def get_daily_stats(self, date: str) -> Dict:
        """
        특정 날짜의 SMP 일일 통계
        
        Args:
            date (str): 날짜 (YYYYMMDD)
            
        Returns:
            Dict: 일일 통계
            {
                "date": str,
                "smp_avg": float,
                "smp_min": float,
                "smp_max": float,
                "smp_std": float,
                "demand_avg": float,
                "demand_max": float
            }
        """
        try:
            df = fetch_smp(date)
            
            if df.empty:
                logger.warning(f"No data for {date}")
                return {
                    "date": date,
                    "smp_avg": 0,
                    "smp_min": 0,
                    "smp_max": 0,
                    "smp_std": 0,
                    "demand_avg": 0,
                    "demand_max": 0
                }
            
            return {
                "date": date,
                "smp_avg": float(df["smp"].mean()),
                "smp_min": float(df["smp"].min()),
                "smp_max": float(df["smp"].max()),
                "smp_std": float(df["smp"].std()),
                "demand_avg": float(df["forecast_demand"].mean()),
                "demand_max": float(df["forecast_demand"].max())
            }
        except Exception as e:
            logger.error(f"Failed to get daily stats for {date}: {e}")
            return {}
    
    def get_period_stats(self, start_date: str, end_date: str) -> Dict:
        """
        특정 기간의 SMP 통계
        
        Args:
            start_date (str): 시작 날짜 (YYYYMMDD)
            end_date (str): 종료 날짜 (YYYYMMDD)
            
        Returns:
            Dict: 기간 통계
        """
        try:
            # 날짜 범위의 평균값 계산
            start = datetime.strptime(start_date, "%Y%m%d")
            end = datetime.strptime(end_date, "%Y%m%d")
            
            daily_stats = []
            current = start
            
            while current <= end:
                date_str = current.strftime("%Y%m%d")
                stats = self.get_daily_stats(date_str)
                if stats.get("smp_avg"):
                    daily_stats.append(stats)
                current += timedelta(days=1)
            
            if not daily_stats:
                return {"message": "No data available"}
            
            smp_values = [s["smp_avg"] for s in daily_stats]
            demand_values = [s["demand_avg"] for s in daily_stats]
            
            return {
                "start_date": start_date,
                "end_date": end_date,
                "days": len(daily_stats),
                "smp_avg": sum(smp_values) / len(smp_values),
                "smp_min": min(smp_values),
                "smp_max": max(smp_values),
                "demand_avg": sum(demand_values) / len(demand_values)
            }
        except Exception as e:
            logger.error(f"Failed to get period stats: {e}")
            return {"message": f"Error: {e}"}
    
    def analyze_demand_pattern(self, date: str) -> Dict:
        """
        특정 날짜의 수요 패턴 분석
        
        Args:
            date (str): 날짜 (YYYYMMDD)
            
        Returns:
            Dict: 수요 패턴 분석 결과
        """
        try:
            df = fetch_smp(date)
            
            if df.empty:
                return {"message": "No data"}
            
            demand = df["forecast_demand"].values
            
            return {
                "date": date,
                "peak_hour": int(df.loc[df["forecast_demand"].idxmax(), "hour"]),
                "peak_demand": float(df["forecast_demand"].max()),
                "low_hour": int(df.loc[df["forecast_demand"].idxmin(), "hour"]),
                "low_demand": float(df["forecast_demand"].min()),
                "pattern": self._classify_demand_pattern(demand)
            }
        except Exception as e:
            logger.error(f"Failed to analyze demand pattern: {e}")
            return {}
    
    def _classify_demand_pattern(self, demand_values: list) -> str:
        """수요 패턴 분류"""
        if len(demand_values) < 2:
            return "unknown"
        
        # 첫 절반과 두 번째 절반 비교
        mid = len(demand_values) // 2
        first_half_avg = sum(demand_values[:mid]) / mid
        second_half_avg = sum(demand_values[mid:]) / (len(demand_values) - mid)
        
        ratio = second_half_avg / first_half_avg if first_half_avg > 0 else 1
        
        if ratio > 1.1:
            return "상승추세"
        elif ratio < 0.9:
            return "하강추세"
        else:
            return "안정적"


def get_smp_analyzer() -> SMPAnalyzer:
    """
    SMP 분석기 인스턴스 반환
    
    Returns:
        SMPAnalyzer: SMP 분석기 인스턴스
    """
    return SMPAnalyzer()
