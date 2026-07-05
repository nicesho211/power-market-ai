"""
환경 설정 테스트

requirements, .env, API 연결 등을 검증합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import validate_settings, get_settings
from infrastructure.llm_client import get_llm, get_embeddings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_settings_validation():
    """설정 검증 테스트"""
    print("\n=== Test 1: Settings Validation ===")
    validation = validate_settings()
    
    if validation["valid"]:
        print("✅ All settings are valid")
        return True
    else:
        print("❌ Settings validation failed:")
        for error in validation["errors"]:
            print(f"  - {error}")
        return False


def test_settings_loading():
    """설정 로딩 테스트"""
    print("\n=== Test 2: Settings Loading ===")
    
    try:
        settings = get_settings()
        
        print(f"✅ Settings loaded successfully")
        print(f"   - OpenAI Model: {settings.OPENAI_MODEL}")
        print(f"   - Embedding Model: {settings.EMBEDDING_MODEL}")
        print(f"   - Collection Name: {settings.QDRANT_COLLECTION_NAME}")
        return True
    except Exception as e:
        print(f"❌ Failed to load settings: {e}")
        return False


def test_llm_client():
    """LLM 클라이언트 테스트"""
    print("\n=== Test 3: LLM Client ===")
    
    try:
        llm = get_llm()
        print(f"✅ LLM client initialized: {type(llm).__name__}")
        
        # 간단한 호출 테스트
        response = llm.invoke("안녕하세요")
        print(f"✅ LLM call successful (response length: {len(response.content)})")
        return True
    except Exception as e:
        print(f"❌ LLM client test failed: {e}")
        return False


def test_embeddings():
    """임베딩 테스트"""
    print("\n=== Test 4: Embeddings ===")
    
    try:
        embeddings = get_embeddings()
        print(f"✅ Embeddings initialized: {type(embeddings).__name__}")
        
        # 임베딩 차원 확인
        test_embedding = embeddings.embed_query("test")
        print(f"✅ Embedding dimension: {len(test_embedding)}")
        
        return True
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        return False


def test_paths():
    """경로 존재 여부 테스트"""
    print("\n=== Test 5: Paths ===")
    
    settings = get_settings()

    paths = {"PDF_PATH": settings.PDF_PATH}
    if not settings.QDRANT_URL:
        paths["QDRANT_DB_PATH"] = settings.QDRANT_DB_PATH
    else:
        print(f"ℹ️  QDRANT_URL 설정됨 (클라우드 모드) — QDRANT_DB_PATH 로컬 경로 체크는 건너뜀")

    all_exist = True
    for name, path in paths.items():
        path_obj = Path(path)
        if path_obj.exists():
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name} does not exist: {path}")
            all_exist = False

    return all_exist


def run_all_tests():
    """모든 테스트 실행"""
    print("\\n" + "="*50)
    print("  ENVIRONMENT VALIDATION TEST")
    print("="*50)
    
    tests = [
        ("Settings Validation", test_settings_validation),
        ("Settings Loading", test_settings_loading),
        ("LLM Client", test_llm_client),
        ("Embeddings", test_embeddings),
        ("Paths", test_paths),
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
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
