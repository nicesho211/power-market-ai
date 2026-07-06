"""
SMP 방향성 추정기

Python 고정 로직으로 SMP 방향성을 추정합니다.
LLM은 이 결과를 받아 자연어 요약만 담당합니다.
"""

import pandas as pd
from typing import Dict, List
import logging
from concurrent.futures import ThreadPoolExecutor
from domain.analysis.mcp_client import fetch_smp, fetch_generation
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DirectionEstimator:
    """SMP 방향성 추정 (Python 고정 로직)"""

    # 클래스 레벨 임계값 캐시 (24시간 유효)
    _threshold_cache: Dict = {}
    _threshold_cached_at: datetime = None
    CACHE_TTL_HOURS: int = 24

    def __init__(self, lookback_days: int = 7):
        """
        추정기 초기화

        Args:
            lookback_days (int): 임계값 계산을 위한 과거 일수
        """
        self.lookback_days = lookback_days
    
    def estimate_direction(self, target_date: str) -> Dict:
        """
        특정 날짜의 SMP 방향성 추정
        
        Args:
            target_date (str): 목표 날짜 (YYYYMMDD)
            
        Returns:
            Dict: 방향성 추정 결과
            {
                "direction": str,  # 상승/보합/하락
                "direction_emoji": str,  # ⬆/➡/⬇
                "score": int,  # 0~3
                "max_score": int,  # 3
                "indicators": dict,  # 각 지표별 상세 정보
                "thresholds": dict,  # 사용된 임계값
                "timestamp": str
            }
        """
        try:
            # 임계값: 캐시 유효하면 재사용, 만료/없으면 병렬 계산
            thresholds = self._get_cached_thresholds(target_date)

            # 현재 데이터 조회 (target_date)
            current_smp = fetch_smp(target_date)
            current_gen = fetch_generation(target_date)

            if current_smp.empty or current_gen.empty:
                logger.warning(f"Incomplete data for {target_date}")
                return self._get_empty_result(target_date)

            # 지표값 계산
            indicators = self._calculate_indicators(current_smp, current_gen, thresholds)

            # 스코어 계산 (Python 고정 로직)
            score = self._calculate_score(indicators)

            # 방향성 결정
            direction, emoji = self._determine_direction(score)

            return {
                "direction": direction,
                "direction_emoji": emoji,
                "score": score,
                "max_score": 3,
                "indicators": indicators,
                "thresholds": thresholds,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to estimate direction: {e}")
            return self._get_empty_result(target_date)
    
    def _get_cached_thresholds(self, reference_date: str) -> Dict:
        """
        임계값 캐시 조회 — 유효하면 재사용, 만료 시 과거 데이터 병렬 fetch 후 계산.
        캐시 유효 기간: 24시간 (CACHE_TTL_HOURS)
        """
        now = datetime.now()
        if (
            DirectionEstimator._threshold_cache
            and DirectionEstimator._threshold_cached_at is not None
            and (now - DirectionEstimator._threshold_cached_at).total_seconds() < self.CACHE_TTL_HOURS * 3600
        ):
            logger.info("임계값 캐시 히트 — API 호출 생략")
            return DirectionEstimator._threshold_cache

        logger.info(f"임계값 캐시 만료/없음 — 과거 {self.lookback_days}일 병렬 fetch 시작")
        thresholds = self._fetch_and_calc_thresholds(reference_date)
        DirectionEstimator._threshold_cache = thresholds
        DirectionEstimator._threshold_cached_at = now
        return thresholds

    def _fetch_and_calc_thresholds(self, reference_date: str) -> Dict:
        """과거 lookback_days 동안의 데이터를 병렬로 가져와 임계값 계산"""
        start_dt = datetime.strptime(reference_date, "%Y%m%d") - timedelta(days=self.lookback_days)
        end_dt = datetime.strptime(reference_date, "%Y%m%d") - timedelta(days=1)

        date_range = []
        cur = start_dt
        while cur <= end_dt:
            date_range.append(cur.strftime("%Y%m%d"))
            cur += timedelta(days=1)

        with ThreadPoolExecutor(max_workers=2) as executor:
            smp_results = list(executor.map(fetch_smp, date_range))
            gen_results = list(executor.map(fetch_generation, date_range))

        past_smp_list = [df for df in smp_results if not df.empty]
        past_gen_list = [df for df in gen_results if not df.empty]

        if past_smp_list and past_gen_list:
            past_smp = pd.concat(past_smp_list, ignore_index=True)
            past_gen = pd.concat(past_gen_list, ignore_index=True)
            return self._calculate_thresholds(past_smp, past_gen)
        return self._get_default_thresholds()

    def estimate_direction_batch(self, target_dates: List[str]) -> Dict[str, Dict]:
        """
        여러 날짜를 한 번에 분석.
        임계값은 캐시를 공유하고, target_dates의 현재 데이터만 병렬 fetch.

        Args:
            target_dates (List[str]): 분석할 날짜 목록 (YYYYMMDD)

        Returns:
            Dict[str, Dict]: {날짜: direction_result}
        """
        if not target_dates:
            return {}

        # 임계값 1회 확보 (캐시 or 신규 계산)
        thresholds = self._get_cached_thresholds(min(target_dates))

        # target_dates의 SMP + 발전량 병렬 fetch
        with ThreadPoolExecutor(max_workers=2) as executor:
            smp_futures = {d: executor.submit(fetch_smp, d) for d in target_dates}
            gen_futures = {d: executor.submit(fetch_generation, d) for d in target_dates}

        results = {}
        for d in target_dates:
            try:
                smp_df = smp_futures[d].result()
                gen_df = gen_futures[d].result()

                if smp_df.empty or gen_df.empty:
                    results[d] = self._get_empty_result(d)
                    continue

                indicators = self._calculate_indicators(smp_df, gen_df, thresholds)
                score = self._calculate_score(indicators)
                direction, emoji = self._determine_direction(score)

                results[d] = {
                    "direction": direction,
                    "direction_emoji": emoji,
                    "score": score,
                    "max_score": 3,
                    "indicators": indicators,
                    "thresholds": thresholds,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Batch estimate failed for {d}: {e}")
                results[d] = self._get_empty_result(d)

        return results

    def warm_up(self, reference_date: str | None = None) -> None:
        """앱 시작 시 임계값 캐시를 미리 채운다 (첫 사용자 질의의 지연 방지)"""
        ref = reference_date or datetime.now().strftime("%Y%m%d")
        self._get_cached_thresholds(ref)

    def invalidate_threshold_cache(self) -> None:
        """임계값 캐시 강제 초기화 (수동 갱신 시 호출)"""
        DirectionEstimator._threshold_cache = {}
        DirectionEstimator._threshold_cached_at = None
        logger.info("임계값 캐시 초기화 완료")

    def _calculate_thresholds(self, past_smp: pd.DataFrame, past_gen: pd.DataFrame) -> Dict:
        """과거 데이터로부터 임계값 계산"""
        # 수요 임계값: 과거 평균 * 1.10 (0이면 기본값 사용)
        demand_avg = past_smp["forecast_demand"].mean()
        if pd.isna(demand_avg) or demand_avg <= 0:
            demand_avg = 74200  # 기본 평균 (육지 기준 74,200MW)
        demand_threshold = demand_avg * 1.10
        
        # LNG 발전 비중 평균
        past_gen_by_source = past_gen.groupby("source")["gen_mw"].sum()
        total_gen = past_gen_by_source.sum()
        
        lng_avg = (
            past_gen_by_source.get("LNG", 0) / total_gen
            if total_gen > 0 else 0.31
        )
        
        # 신재생 발전 비중 평균
        renewable_sources = ["신재생", "태양광"]
        renewable_gen = sum(
            past_gen_by_source.get(source, 0) for source in renewable_sources
        )
        renewable_avg = renewable_gen / total_gen if total_gen > 0 else 0.18
        
        return {
            "수요_임계값": demand_threshold,
            "LNG_비중_평균": lng_avg,
            "신재생_비중_평균": renewable_avg,
            "SMP_평균": past_smp["smp"].mean(),
            "기준기간": f"최근 {self.lookback_days}일"
        }
    
    def _get_default_thresholds(self) -> Dict:
        """기본 임계값"""
        return {
            "수요_임계값": 81620,  # MW
            "LNG_비중_평균": 0.31,
            "신재생_비중_평균": 0.18,
            "SMP_평균": 132.4,
            "기준기간": f"최근 {self.lookback_days}일"
        }
    
    def _calculate_indicators(
        self,
        current_smp: pd.DataFrame,
        current_gen: pd.DataFrame,
        thresholds: Dict
    ) -> Dict:
        """현재 지표값 계산"""
        # 현재 수요
        current_demand = current_smp["forecast_demand"].mean()
        if pd.isna(current_demand):
            current_demand = 0.0
        threshold_demand = thresholds["수요_임계값"]
        demand_vs_threshold = (current_demand / threshold_demand) if threshold_demand > 0 else 0.0
        
        # 현재 LNG 비중
        current_gen_by_source = current_gen.groupby("source")["gen_mw"].sum()
        total_gen = current_gen_by_source.sum()
        
        if total_gen > 0:
            lng_ratio = current_gen_by_source.get("LNG", 0) / total_gen
            renewable_sources = ["신재생", "태양광"]
            renewable_gen = sum(
                current_gen_by_source.get(source, 0) for source in renewable_sources
            )
            renewable_ratio = renewable_gen / total_gen
        else:
            lng_ratio = 0
            renewable_ratio = 0
        
        return {
            "전력수요": {
                "현재": f"{current_demand:,.0f}MW",
                "임계값": f"{thresholds['수요_임계값']:,.0f}MW",
                "평균대비": f"{(demand_vs_threshold - 1) * 100:+.1f}%",
                "기여": "+1점" if current_demand > thresholds["수요_임계값"] else "0점"
            },
            "LNG발전비중": {
                "현재": f"{lng_ratio * 100:.1f}%",
                "임계값": f"{thresholds['LNG_비중_평균'] * 100:.1f}%",
                "평균대비": f"{(lng_ratio - thresholds['LNG_비중_평균']) * 100:+.1f}%p",
                "기여": "+1점" if lng_ratio > thresholds["LNG_비중_평균"] else "0점"
            },
            "신재생발전량": {
                "현재": f"{renewable_ratio * 100:.1f}%",
                "임계값": f"{thresholds['신재생_비중_평균'] * 100:.1f}%",
                "평균대비": f"{(renewable_ratio - thresholds['신재생_비중_평균']) * 100:+.1f}%p",
                "기여": "+1점" if renewable_ratio < thresholds["신재생_비중_평균"] else "0점"
            }
        }
    
    def _calculate_score(self, indicators: Dict) -> int:
        """
        Python 고정 로직으로 스코어 계산
        LLM이 직접 방향성을 판단하면 안 됨!
        """
        score = 0
        
        # 전력수요: 임계값 초과 시 +1점
        demand_str = indicators["전력수요"]["현재"]
        demand_val = float(demand_str.replace(",", "").replace("MW", "").strip())
        threshold_str = indicators["전력수요"]["임계값"]
        threshold_val = float(threshold_str.replace(",", "").replace("MW", "").strip())
        
        if demand_val > threshold_val:
            score += 1
        
        # LNG 비중: 임계값 초과 시 +1점
        lng_str = indicators["LNG발전비중"]["현재"].replace("%", "").strip()
        lng_val = float(lng_str)
        lng_threshold_str = indicators["LNG발전비중"]["임계값"].replace("%", "").strip()
        lng_threshold_val = float(lng_threshold_str)
        
        if lng_val > lng_threshold_val:
            score += 1
        
        # 신재생 비중: 임계값 미만 시 +1점
        renewable_str = indicators["신재생발전량"]["현재"].replace("%", "").strip()
        renewable_val = float(renewable_str)
        renewable_threshold_str = indicators["신재생발전량"]["임계값"].replace("%", "").strip()
        renewable_threshold_val = float(renewable_threshold_str)
        
        if renewable_val < renewable_threshold_val:
            score += 1
        
        return score
    
    def _determine_direction(self, score: int) -> tuple:
        """스코어에 따른 방향성 결정"""
        if score >= 2:
            return "상승", "⬆"
        elif score == 1:
            return "보합", "➡"
        else:
            return "하락", "⬇"
    
    def _get_empty_result(self, target_date: str) -> Dict:
        """빈 결과 반환"""
        return {
            "direction": "분석불가",
            "direction_emoji": "❓",
            "score": 0,
            "max_score": 3,
            "indicators": {},
            "thresholds": {},
            "timestamp": datetime.now().isoformat(),
            "error": "데이터 부족으로 분석 불가"
        }


def get_direction_estimator(lookback_days: int = 7) -> DirectionEstimator:
    """
    추정기 인스턴스 반환
    
    Args:
        lookback_days (int): 임계값 계산을 위한 과거 일수
        
    Returns:
        DirectionEstimator: 추정기 인스턴스
    """
    return DirectionEstimator(lookback_days)
