"""
데이터 분석 파이프라인 테스트

API 연결, 데이터 수집, 분석, 시각화를 검증합니다.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from domain.analysis.mcp_client import (
    fetch_smp, fetch_generation, fetch_current_demand
)
from domain.analysis.smp_analyzer import get_smp_analyzer
from domain.analysis.direction_estimator import get_direction_estimator
from domain.analysis.chart_builder import get_chart_builder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_fetch_smp():
    """SMP 데이터 조회 테스트"""
    print("\n=== Test 1: Fetch SMP ===")
    
    try:
        # 과거 날짜 사용 (데이터 존재 확률 높음)
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        
        df = fetch_smp(target_date)
        
        if df.empty:
            print(f"⚠️  No SMP data for {target_date}")
        else:
            print(f"✅ SMP data fetched successfully")
            print(f"   - Date: {target_date}")
            print(f"   - Records: {len(df)}")
            print(f"   - Columns: {list(df.columns)}")
            print(f"   - SMP range: {df['smp'].min():.2f} ~ {df['smp'].max():.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Fetch SMP test failed: {e}")
        return False


def test_fetch_generation():
    """발전원별 발전량 조회 테스트"""
    print("\n=== Test 2: Fetch Generation ===")
    
    try:
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        
        df = fetch_generation(target_date)
        
        if df.empty:
            print(f"⚠️  No generation data for {target_date}")
        else:
            print(f"✅ Generation data fetched successfully")
            print(f"   - Date: {target_date}")
            print(f"   - Records: {len(df)}")
            print(f"   - Sources: {df['source'].unique().tolist()}")
        
        return True
    except Exception as e:
        print(f"❌ Fetch generation test failed: {e}")
        return False


def test_fetch_current_demand():
    """현재 수급현황 조회 테스트"""
    print("\n=== Test 3: Fetch Current Demand ===")
    
    try:
        df = fetch_current_demand()
        
        if df.empty:
            print(f"⚠️  No current demand data")
        else:
            print(f"✅ Current demand fetched successfully")
            print(f"   - Columns: {list(df.columns)}")
            if not df.empty:
                row = df.iloc[0]
                print(f"   - Timestamp: {row.get('datetime', 'N/A')}")
                print(f"   - Demand: {row.get('demand_mw', 0):,.0f} MW")
        
        return True
    except Exception as e:
        print(f"❌ Fetch current demand test failed: {e}")
        return False


def test_smp_analyzer():
    """SMP 분석기 테스트"""
    print("\n=== Test 4: SMP Analyzer ===")
    
    try:
        analyzer = get_smp_analyzer()
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        
        # 일일 통계
        daily_stats = analyzer.get_daily_stats(target_date)
        
        print(f"✅ SMP Analyzer test successful")
        if daily_stats.get("smp_avg"):
            print(f"   - Date: {daily_stats.get('date')}")
            print(f"   - SMP Average: {daily_stats.get('smp_avg', 0):.2f} ₩/MWh")
            print(f"   - SMP Range: {daily_stats.get('smp_min', 0):.2f} ~ {daily_stats.get('smp_max', 0):.2f}")
        else:
            print(f"   - No data available")
        
        return True
    except Exception as e:
        print(f"❌ SMP Analyzer test failed: {e}")
        return False


def test_direction_estimator():
    """방향성 추정기 테스트"""
    print("\n=== Test 5: Direction Estimator ===")
    
    try:
        estimator = get_direction_estimator()
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        
        # 방향성 추정
        result = estimator.estimate_direction(target_date)
        
        print(f"✅ Direction Estimator test successful")
        print(f"   - Date: {target_date}")
        print(f"   - Direction: {result.get('direction', 'N/A')} {result.get('direction_emoji', '')}")
        print(f"   - Score: {result.get('score', 0)}/{result.get('max_score', 3)}")
        
        if result.get("indicators"):
            print(f"   - Indicators:")
            for key, value in result["indicators"].items():
                print(f"     • {key}: {value.get('현재')} (기여: {value.get('기여')})")
        
        return True
    except Exception as e:
        print(f"❌ Direction Estimator test failed: {e}")
        return False


def test_chart_builder():
    """차트 빌더 테스트"""
    print("\n=== Test 6: Chart Builder ===")
    
    try:
        builder = get_chart_builder()
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        
        # SMP 차트 생성
        smp_fig = builder.build_smp_chart(target_date)
        print(f"✅ SMP chart created successfully")
        
        # 발전량 차트 생성
        gen_fig = builder.build_generation_chart(target_date)
        print(f"✅ Generation chart created successfully")
        
        # 게이지 차트 생성
        gauge_fig = builder.build_indicator_gauge(2, 3)
        print(f"✅ Gauge chart created successfully")
        
        return True
    except Exception as e:
        print(f"❌ Chart Builder test failed: {e}")
        return False


def run_all_tests():
    """모든 테스트 실행"""
    print("\\n" + "="*50)
    print("  DATA ANALYSIS PIPELINE TEST")
    print("="*50)
    
    tests = [
        ("Fetch SMP", test_fetch_smp),
        ("Fetch Generation", test_fetch_generation),
        ("Fetch Current Demand", test_fetch_current_demand),
        ("SMP Analyzer", test_smp_analyzer),
        ("Direction Estimator", test_direction_estimator),
        ("Chart Builder", test_chart_builder),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append(False)
    
    # 요약
    print("\n" + "="*50)
    print("  TEST SUMMARY")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All analysis tests passed!")
        return True
    else:
        print("❌ Some analysis tests failed.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
