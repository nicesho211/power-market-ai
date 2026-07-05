"""
날짜 범위 계산 모듈

LLM이 분류한 period_type을 받아 Python 고정 로직으로 실제 날짜 리스트를 계산한다.
LLM이 직접 날짜를 계산하지 않고 이 함수가 담당한다.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Union


def resolve_date_range(date_filter: dict) -> Union[List[str], Dict[str, List[str]]]:
    """
    date_filter의 period_type에 따라 날짜 리스트(YYYYMMDD) 계산

    Args:
        date_filter (dict): {
            "period_type": "today"|"yesterday"|"last_n_days"|
                           "this_week"|"last_week"|"custom_range"|"multi_range"|
                           "ambiguous"|"not_applicable",
            "n_days": int | None,
            "start_date": "YYYY-MM-DD" | None,
            "end_date": "YYYY-MM-DD" | None,
            "ranges": [{"label": str, "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}, ...] | None
        }

    Returns:
        List[str]: YYYYMMDD 형식 날짜 리스트 (빈 리스트 = 계산 불가)
        단, period_type이 "multi_range"이면 {label: [YYYYMMDD, ...]} 형태의 dict를 반환한다.
    """
    today = datetime.now().date()
    period_type = date_filter.get("period_type", "not_applicable")

    if period_type == "multi_range":
        result: Dict[str, List[str]] = {}
        for r in date_filter.get("ranges", []) or []:
            start_str = r.get("start_date")
            end_str = r.get("end_date")
            label = r.get("label") or f"{start_str}~{end_str}"
            if not start_str or not end_str:
                continue
            try:
                start = datetime.strptime(start_str, "%Y-%m-%d").date()
                end = datetime.strptime(end_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            total_days = (end - start).days
            if total_days < 0 or total_days > 365:
                continue
            result[label] = [
                (start + timedelta(days=i)).strftime("%Y%m%d")
                for i in range(total_days + 1)
            ]
        return result

    if period_type == "today":
        return [today.strftime("%Y%m%d")]

    elif period_type == "yesterday":
        return [(today - timedelta(days=1)).strftime("%Y%m%d")]

    elif period_type == "last_n_days":
        n = date_filter.get("n_days") or 3
        try:
            n = int(n)
        except (TypeError, ValueError):
            n = 3
        return [
            (today - timedelta(days=i)).strftime("%Y%m%d")
            for i in range(1, n + 1)
        ]

    elif period_type == "this_week":
        monday = today - timedelta(days=today.weekday())
        days_from_monday = (today - monday).days
        return [
            (monday + timedelta(days=i)).strftime("%Y%m%d")
            for i in range(days_from_monday + 1)
        ]

    elif period_type == "last_week":
        this_monday = today - timedelta(days=today.weekday())
        last_monday = this_monday - timedelta(days=7)
        last_sunday = this_monday - timedelta(days=1)
        return [
            (last_monday + timedelta(days=i)).strftime("%Y%m%d")
            for i in range((last_sunday - last_monday).days + 1)
        ]

    elif period_type == "custom_range":
        start_str = date_filter.get("start_date")
        end_str = date_filter.get("end_date")
        if not start_str or not end_str:
            return []
        try:
            start = datetime.strptime(start_str, "%Y-%m-%d").date()
            end = datetime.strptime(end_str, "%Y-%m-%d").date()
        except ValueError:
            return []
        total_days = (end - start).days
        if total_days < 0 or total_days > 365:
            return []
        return [
            (start + timedelta(days=i)).strftime("%Y%m%d")
            for i in range(total_days + 1)
        ]

    # ambiguous, not_applicable, unknown
    return []
