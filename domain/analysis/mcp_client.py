"""
공공데이터 API 클라이언트

한국전력거래소 API를 통해 SMP, 발전원별 발전량, 전력수급현황 데이터를 수집합니다.
CLAUDE.md 섹션 3의 API 명세를 기준으로 구현됩니다.
"""

import requests
import pandas as pd
import time
from config.settings import get_settings
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# lru_cache 대신 커스텀 캐시 (빈 DataFrame은 저장하지 않아 429 오염 방지)
_SMP_CACHE: dict = {}
_GEN_CACHE: dict = {}

# 실패(429/데이터없음) 응답용 단기 캐시 — 짧은 TTL 동안은 재호출 없이 바로 빈 DF 반환
# (실패는 캐시되지 않아 rate limit에 걸릴 때마다 매번 재시도하며 점점 느려지는 것을 방지)
_SMP_FAIL_CACHE: dict = {}
_GEN_FAIL_CACHE: dict = {}
_FAIL_CACHE_TTL_SECONDS = 60

# 실시간 수급현황 캐시 (5분 간격 데이터이므로 60초 캐시해도 안전)
_DEMAND_CACHE: dict = {}
_DEMAND_CACHE_TTL_SECONDS = 60


def _is_fail_cached(cache: dict, key) -> bool:
    failed_at = cache.get(key)
    return failed_at is not None and (datetime.now() - failed_at).total_seconds() < _FAIL_CACHE_TTL_SECONDS


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


def _call_api(base_url: str, params: dict, max_retries: int = 2) -> dict:
    """공통 API 호출 함수 — 429 발생 시 최대 2회 재시도"""
    from urllib.parse import unquote_plus
    settings = get_settings()
    service_key = settings.public_data_api_key
    if "%" in service_key:
        service_key = unquote_plus(service_key)
    params["serviceKey"] = service_key
    params["dataType"] = "JSON"

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(base_url, params=params, timeout=15)
            if response.status_code == 429:
                wait = 1.5 * (attempt + 1)
                logger.warning(f"429 rate limit — {wait:.1f}초 대기 후 재시도 ({attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries:
                logger.error(f"[API 오류] {base_url}: {e}")
                return {}
            time.sleep(1.0)
    return {}


def fetch_smp(date: str, area: str = "01") -> pd.DataFrame:
    """
    SMP (계통한계가격) 및 수요예측 데이터 조회

    Args:
        date (str): 조회 날짜 (YYYYMMDD 형식)
        area (str): 지역코드 ("01": 육지, "02": 제주)

    Returns:
        pd.DataFrame: SMP 데이터
        컬럼: [date, hour, smp, region, forecast_demand]
    """
    cache_key = (date, area)
    if cache_key in _SMP_CACHE:
        logger.info(f"[API] SMP 캐시 재사용 | date={date}")
        return _SMP_CACHE[cache_key]

    if _is_fail_cached(_SMP_FAIL_CACHE, cache_key):
        return pd.DataFrame(columns=["date", "hour", "smp", "region", "forecast_demand"])

    err_msg = _validate_date(date)
    if err_msg:
        logger.warning(f"[SMP 날짜 체크] {err_msg}")
        return pd.DataFrame(columns=["date", "hour", "smp", "region", "forecast_demand"])

    logger.info(f"[API] fetch_smp 호출 | date={date}")
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
        logger.warning(f"No SMP data for {date} with date param")
        _SMP_FAIL_CACHE[cache_key] = datetime.now()
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
    logger.info(f"[API] SMP 응답 수신 | date={date} | {len(df)}개 시간대")
    if not df.empty:
        _SMP_CACHE[cache_key] = df  # 성공 결과만 캐시
        _SMP_FAIL_CACHE.pop(cache_key, None)
    return df


def fetch_generation(date: str) -> pd.DataFrame:
    """
    발전원별 발전량 (계통기준) 조회

    Args:
        date (str): 조회 날짜 (YYYYMMDD 형식)

    Returns:
        pd.DataFrame: 발전원별 발전량
        컬럼: [date, hour, source, gen_mw]
        source: 수력, 유류, 유연탄, 원자력, 양수, LNG, 국내탄, 신재생, 태양광
    """
    if date in _GEN_CACHE:
        logger.info(f"[API] 발전량 캐시 재사용 | date={date}")
        return _GEN_CACHE[date]

    if _is_fail_cached(_GEN_FAIL_CACHE, date):
        return pd.DataFrame(columns=["date", "hour", "source", "gen_mw"])

    err_msg = _validate_date(date)
    if err_msg:
        logger.warning(f"[발전량 날짜 체크] {err_msg}")
        return pd.DataFrame(columns=["date", "hour", "source", "gen_mw"])

    logger.info(f"[API] fetch_generation 호출 | date={date}")
    params = {
        "pageNo": 1,
        "numOfRows": 24,
        "baseDate": date
    }

    data = _call_api(
        "https://apis.data.go.kr/B552115/PwrAmountByGen/getPwrAmountByGen",
        params
    )

    items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])

    if not items:
        logger.warning(f"No generation data for {date} with baseDate param")
        _GEN_FAIL_CACHE[date] = datetime.now()
        return pd.DataFrame(columns=["date", "hour", "source", "gen_mw"])
    
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
    
    rows = []
    for item in items:
        # baseDatetime 형식: "YYYYMMDDHHMMSS"
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
    logger.info(f"[API] 발전량 응답 수신 | date={date} | {len(df)}행")
    if not df.empty:
        _GEN_CACHE[date] = df  # 성공 결과만 캐시
        _GEN_FAIL_CACHE.pop(date, None)
    return df


def fetch_current_demand() -> pd.DataFrame:
    """
    현재 전력수급현황 조회 (실시간, 60초 캐시 — 원본 데이터가 5분 간격이라 안전)

    Returns:
        pd.DataFrame: 현재 수급현황
        컬럼: [datetime, demand_mw, supply_mw, reserve_mw, reserve_rate]
    """
    cached_at = _DEMAND_CACHE.get("at")
    if cached_at and (datetime.now() - cached_at).total_seconds() < _DEMAND_CACHE_TTL_SECONDS:
        logger.info("[API] 수급현황 캐시 재사용")
        return _DEMAND_CACHE["data"]

    logger.info("[API] fetch_current_demand 호출")
    data = _call_api(
        "https://openapi.kpx.or.kr/openapi/sukub5mToday/getSukub5mToday",
        {}
    )

    items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])

    if not items:
        logger.warning("No current demand data")
        return pd.DataFrame(
            columns=["datetime", "demand_mw", "supply_mw", "reserve_mw", "reserve_rate"]
        )

    item = items[0] if isinstance(items, list) else items

    result = pd.DataFrame([{
        "datetime": item.get("baseDatetime", ""),
        "demand_mw": float(item.get("curr", 0)),
        "supply_mw": float(item.get("suppAbility", 0)),
        "reserve_mw": float(item.get("suppReserve", 0)),
        "reserve_rate": float(item.get("suppReserveRate", 0))
    }])
    _DEMAND_CACHE["data"] = result
    _DEMAND_CACHE["at"] = datetime.now()
    return result


def fetch_smp_range(start_date: str, end_date: str, area: str = "01") -> pd.DataFrame:
    """
    특정 기간의 SMP 데이터 조회
    
    Args:
        start_date (str): 시작 날짜 (YYYYMMDD)
        end_date (str): 종료 날짜 (YYYYMMDD)
        area (str): 지역코드
        
    Returns:
        pd.DataFrame: SMP 데이터 (full range)
    """
    all_data = []
    
    # 날짜 범위 생성
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    
    current = start
    while current <= end:
        date_str = current.strftime("%Y%m%d")
        df = fetch_smp(date_str, area)
        if not df.empty:
            all_data.append(df)
        current += timedelta(days=1)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame(columns=["date", "hour", "smp", "region", "forecast_demand"])


def fetch_generation_range(start_date: str, end_date: str) -> pd.DataFrame:
    """
    특정 기간의 발전원별 발전량 조회
    
    Args:
        start_date (str): 시작 날짜 (YYYYMMDD)
        end_date (str): 종료 날짜 (YYYYMMDD)
        
    Returns:
        pd.DataFrame: 발전원별 발전량 (full range)
    """
    all_data = []
    
    # 날짜 범위 생성
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    
    current = start
    while current <= end:
        date_str = current.strftime("%Y%m%d")
        df = fetch_generation(date_str)
        if not df.empty:
            all_data.append(df)
        current += timedelta(days=1)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame(columns=["date", "hour", "source", "gen_mw"])
