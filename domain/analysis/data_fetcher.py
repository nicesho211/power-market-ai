"""
공공데이터포털 API 호출 모듈
SMP, 발전량, 전력수급현황 데이터를 On-the-fly로 수집합니다.
DB 저장 없이 즉시 처리 후 반환합니다.
"""

import requests
import pandas as pd
from functools import lru_cache
from datetime import datetime, timedelta
from config.settings import get_settings


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
        print(f"[API 오류] {base_url}: {e}")
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
    # 사전 날짜 유효성 체크 (미래/과거 1년 초과 시 API 호출 안 함)
    err_msg = _validate_date(date)
    if err_msg:
        print(f"[SMP 날짜 체크] {err_msg}")
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
        print(f"[SMP 파싱] {date} 데이터 없음")
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
    return pd.DataFrame(rows)


@lru_cache(maxsize=32)
def fetch_generation(date: str) -> pd.DataFrame:
    """발전원별 발전량 조회 (baseDate 파라미터명 사용 — fetch_smp의 date와 다름)"""
    # 사전 날짜 유효성 체크 (미래/과거 1년 초과 시 API 호출 안 함)
    err_msg = _validate_date(date)
    if err_msg:
        print(f"[발전량 날짜 체크] {err_msg}")
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
        print(f"[발전량 파싱] {date} 데이터 없음")
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
    return pd.DataFrame(rows)


def fetch_current_demand() -> pd.DataFrame:
    """현재 전력수급현황 조회 (실시간, 캐시 미적용)"""
    data = _call_api(
        "https://openapi.kpx.or.kr/openapi/sukub5mToday/getSukub5mToday",
        {}
    )
    items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    if not items:
        print("[수급현황 파싱] 데이터 없음")
        return pd.DataFrame(columns=["datetime", "demand_mw", "supply_mw", "reserve_mw", "reserve_rate"])
    item = items[0] if isinstance(items, list) else items
    return pd.DataFrame([{
        "datetime": item.get("baseDatetime", ""),
        "demand_mw": float(item.get("curr", 0)),
        "supply_mw": float(item.get("suppAbility", 0)),
        "reserve_mw": float(item.get("suppReserve", 0)),
        "reserve_rate": float(item.get("suppReserveRate", 0))
    }])
