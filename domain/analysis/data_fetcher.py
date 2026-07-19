"""
공공데이터포털 API 호출 모듈
SMP, 발전량, 전력수급현황 데이터를 On-the-fly로 수집합니다.
DB 저장 없이 즉시 처리 후 반환합니다.
"""

import logging
import time
import requests
import pandas as pd
from functools import lru_cache
from datetime import datetime, timedelta
from config.settings import get_settings

logger = logging.getLogger("API")

_DEMAND_CACHE_TTL = 30  # 초 — KPX 수급현황 자체가 5분 주기 갱신이라 짧은 TTL로 충분
_demand_cache: tuple[float, pd.DataFrame] | None = None


def _call_api(base_url: str, params: dict) -> dict:
    """공통 API 호출 함수"""
    try:
        from urllib.parse import unquote_plus
        settings = get_settings()
        service_key = settings.PUBLIC_DATA_API_KEY
        if "%" in service_key:
            service_key = unquote_plus(service_key)
        params["serviceKey"] = service_key
        params["dataType"] = "JSON"
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"[API 오류] {base_url}: {e}")
        return {}


def _validate_date(date: str) -> str | None:
    """날짜 유효성 사전 체크 — 문제 있으면 오류 메시지 반환, 정상이면 None"""
    try:
        target = datetime.strptime(date, "%Y%m%d").date()
    except ValueError:
        return f"날짜 형식 오류: {date} (YYYYMMDD 형식으로 입력하세요)"
    today = datetime.now().date()
    if target > today:
        return f"아직 데이터가 없는 날짜입니다 ({date}는 미래 날짜)"
    if target < today - timedelta(days=365):
        return f"조회 가능 범위를 초과했습니다 ({date}는 1년 이상 과거)"
    return None


@lru_cache(maxsize=32)
def fetch_smp(date: str, area: str = "01") -> pd.DataFrame:
    """SMP 데이터 조회 (date 파라미터명 사용)"""
    logger.info(f"[API] fetch_smp 호출 | date={date}")

    # 사전 날짜 유효성 체크 (미래/과거 1년 초과 시 API 호출 안 함)
    err_msg = _validate_date(date)
    if err_msg:
        logger.warning(f"[API] SMP 날짜 체크 실패 | {err_msg}")
        return pd.DataFrame(columns=["date", "hour", "smp", "region", "forecast_demand"])

    params = {
        "pageNo": 1,
        "numOfRows": 24,
        "date": date
    }
    data = _call_api(
        "https://apis.data.go.kr/B552115/SmpWithForecastDemand/getSmpWithForecastDemand",
        params
    )
    items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    if not items:
        logger.warning(f"[API] SMP 응답 없음 | date={date}")
        return pd.DataFrame(columns=["date", "hour", "smp", "region", "forecast_demand"])
    rows = []
    for item in items:
        rows.append({
            "date": item.get("date", date),
            "hour": int(item.get("hour", 0)),
            "smp": float(item.get("smp", 0)),
            "region": item.get("areaName", "육지" if area == "01" else "제주"),
            "forecast_demand": float(item.get("slfd", item.get("fcstDemand", 0)))
        })
    df = pd.DataFrame(rows)
    logger.info(f"[API] SMP 응답 수신 | {len(df)}개 시간대 | 평균={df['smp'].mean():.1f} 원/kWh")
    return df


@lru_cache(maxsize=32)
def fetch_generation(date: str) -> pd.DataFrame:
    """발전원별 발전량 조회 (baseDate 파라미터명 사용 — fetch_smp의 date와 다름)"""
    logger.info(f"[API] fetch_generation 호출 | baseDate={date}")

    # 사전 날짜 유효성 체크 (미래/과거 1년 초과 시 API 호출 안 함)
    err_msg = _validate_date(date)
    if err_msg:
        logger.warning(f"[API] 발전량 날짜 체크 실패 | {err_msg}")
        return pd.DataFrame(columns=["date", "hour", "source", "gen_mw"])

    params = {
        "pageNo": 1,
        "numOfRows": 24,
        "baseDate": date
    }
    data = _call_api(
        "https://apis.data.go.kr/B552115/PwrAmountByGen/getPwrAmountByGen",
        params
    )
    SOURCE_MAP = {
        "fuelPwr1": "수력",
        "fuelPwr2": "유류",
        "fuelPwr3": "유연탄",
        "fuelPwr4": "원자력",
        "fuelPwr5": "양수",
        "fuelPwr6": "LNG",
        "fuelPwr7": "국내탄",
        "fuelPwr8": "신재생",
        "fuelPwr9": "태양광"
    }
    items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    if not items:
        logger.warning(f"[API] 발전량 응답 없음 | baseDate={date}")
        return pd.DataFrame(columns=["date", "hour", "source", "gen_mw"])
    rows = []
    for item in items:
        base_dt = item.get("baseDatetime", "")
        hour = int(base_dt[8:10]) if len(base_dt) >= 10 else int(item.get("tradeHour", 0))
        for field, source_name in SOURCE_MAP.items():
            rows.append({
                "date": date,
                "hour": hour,
                "source": source_name,
                "gen_mw": float(item.get(field, 0))
            })
    df = pd.DataFrame(rows)
    logger.info(f"[API] 발전량 응답 수신 | {len(SOURCE_MAP)}개 발전원 | {df['hour'].nunique()}개 시간대")
    for source_name in SOURCE_MAP.values():
        avg_mw = df.loc[df["source"] == source_name, "gen_mw"].mean()
        logger.info(f"[API]   {source_name}: {avg_mw:,.0f} MW (일평균)")
    return df


def fetch_current_demand() -> pd.DataFrame:
    """현재 전력수급현황 조회 (짧은 TTL 캐시 적용).

    사이드바/상단 지표 카드가 매 Streamlit rerun(=사용자 클릭 한 번)마다
    이 함수를 부르므로, 캐시 없이는 클릭할 때마다 외부 API 왕복이 발생해
    화면이 느려진다. KPX 수급현황 자체가 5분 주기 갱신이라 30초 TTL로도
    체감 실시간성은 유지된다."""
    global _demand_cache
    now = time.monotonic()
    if _demand_cache is not None:
        cached_at, cached_df = _demand_cache
        if now - cached_at < _DEMAND_CACHE_TTL:
            return cached_df

    logger.info("[API] fetch_current_demand 호출")
    data = _call_api(
        "https://openapi.kpx.or.kr/openapi/sukub5mToday/getSukub5mToday",
        {}
    )
    items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    if not items:
        logger.warning("[API] 수급현황 응답 없음")
        result = pd.DataFrame(columns=["datetime", "demand_mw", "supply_mw", "reserve_mw", "reserve_rate"])
        _demand_cache = (now, result)
        return result
    item = items[0] if isinstance(items, list) else items
    demand_mw = float(item.get("curr", 0))
    logger.info(f"[API] 수급현황 수신 | 수요={demand_mw:,.0f} MW")
    result = pd.DataFrame([{
        "datetime": item.get("baseDatetime", ""),
        "demand_mw": demand_mw,
        "supply_mw": float(item.get("suppAbility", 0)),
        "reserve_mw": float(item.get("suppReserve", 0)),
        "reserve_rate": float(item.get("suppReserveRate", 0))
    }])
    _demand_cache = (now, result)
    return result
