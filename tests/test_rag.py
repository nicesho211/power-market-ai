"""
RAG 파이프라인 테스트

벡터 스토어, 검색, 개정 비교 등을 검증합니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from domain.rag.embedder import get_embedder
from domain.rag.vector_store import get_vector_store
from domain.rag.chunker import get_chunker
from domain.rag.retriever import get_retriever
from domain.rag.diff_pipeline import get_diff_pipeline
from domain.rag.history_manager import get_history_manager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_embedder():
    """임베딩 테스트"""
    print("\n=== Test 1: Embedder ===")
    
    try:
        embedder = get_embedder()
        
        # 단일 텍스트 임베딩
        text = "계통한계가격이란 무엇인가?"
        embedding = embedder.embed_text(text)
        
        print(f"✅ Single text embedding successful")
        print(f"   - Text: {text}")
        print(f"   - Dimension: {len(embedding)}")
        
        # 여러 텍스트 임베딩
        texts = ["SMP", "발전량", "수요"]
        embeddings = embedder.embed_documents(texts)
        
        print(f"✅ Multiple texts embedding successful")
        print(f"   - Count: {len(embeddings)}")
        
        return True
    except Exception as e:
        print(f"❌ Embedder test failed: {e}")
        return False


def test_vector_store():
    """벡터 스토어 테스트"""
    print("\n=== Test 2: Vector Store ===")
    
    try:
        vs = get_vector_store()
        
        # 컬렉션 통계
        stats = vs.get_collection_stats()
        print(f"✅ Vector store initialized")
        print(f"   - Total documents: {stats.get('total_documents', 0)}")
        
        # 테스트 문서 추가
        test_docs = ["계통한계가격(SMP) 결정 규칙", "발전기 입찰 규칙"]
        test_ids = ["test_001", "test_002"]
        test_metadata = [
            {
                "조문번호": "제2.4.1조",
                "페이지": 42,
                "버전": "2025-04-10",
                "is_latest": True
            },
            {
                "조문번호": "제3.1.2조",
                "페이지": 85,
                "버전": "2025-04-10",
                "is_latest": True
            }
        ]
        
        vs.add_documents(test_docs, test_ids, test_metadata)
        print(f"✅ Documents added successfully")
        
        return True
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False


def test_chunker():
    """청커 테스트"""
    print("\n=== Test 3: Chunker ===")
    
    try:
        chunker = get_chunker()
        
        # 테스트 문서
        test_content = """
        제2.4.1조 계통한계가격의 정의
        계통한계가격(SMP)은 모든 발전기가 동일한 가격으로 입찰하는 경우의 
        한계 발전기의 가격을 의미한다.
        
        제2.4.2조 계통한계가격의 결정
        계통한계가격은 시간대별로 결정되며, 공급곡선과 수요를 고려한다.
        """
        
        chunks = chunker.chunk_document(test_content, "test.pdf")
        
        print(f"✅ Chunking successful")
        print(f"   - Number of chunks: {len(chunks)}")
        
        if chunks:
            print(f"   - First chunk size: {len(chunks[0]['text'])} chars")
            print(f"   - Article: {chunks[0]['metadata']['조문번호']}")
        
        return True
    except Exception as e:
        print(f"❌ Chunker test failed: {e}")
        return False


def test_retriever():
    """검색기 테스트"""
    print("\n=== Test 4: Retriever ===")
    
    try:
        retriever = get_retriever()
        
        # 검색 수행
        query = "SMP"
        results = retriever.search(query, top_k=3)
        
        print(f"✅ Retrieval successful")
        print(f"   - Query: {query}")
        print(f"   - Results: {len(results)}")
        
        if results:
            for i, result in enumerate(results[:1]):
                print(f"   - Result {i+1}: {result['metadata'].get('조문번호', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Retriever test failed: {e}")
        return False


def test_diff_pipeline():
    """개정 비교 파이프라인 테스트"""
    print("\n=== Test 5: Diff Pipeline ===")
    
    try:
        diff_pipeline = get_diff_pipeline()
        
        # 테스트 문서 생성
        current = {
            "document": "수정된 규정 내용",
            "metadata": {
                "조문번호": "제2.4.1조",
                "버전": "2025-04-10",
                "개정유형": "수정"
            }
        }
        
        previous = {
            "document": "기존 규정 내용",
            "metadata": {
                "조문번호": "제2.4.1조",
                "버전": "2025-02-01",
                "개정유형": "유지"
            }
        }
        
        # 비교 수행
        result = diff_pipeline.compare_regulations(current, previous)
        
        print(f"✅ Diff pipeline test successful")
        print(f"   - Is changed: {result['is_changed']}")
        print(f"   - Change type: {result['change_type']}")
        
        return True
    except Exception as e:
        print(f"❌ Diff pipeline test failed: {e}")
        return False


def test_history_manager():
    """이력 관리자 테스트"""
    print("\n=== Test 6: History Manager ===")
    
    try:
        history_manager = get_history_manager()
        
        # 테스트 조문번호
        regulation = "제2.4.1조"
        
        # 이력 조회
        history = history_manager.get_history(regulation)
        
        print(f"✅ History manager test successful")
        print(f"   - Regulation: {regulation}")
        print(f"   - Found: {history.get('found')}")
        print(f"   - History count: {len(history.get('history', []))}")
        
        # 개정 횟수
        count = history_manager.get_revision_count(regulation)
        print(f"   - Revision count: {count}")
        
        return True
    except Exception as e:
        print(f"❌ History manager test failed: {e}")
        return False


def run_all_tests():
    """모든 테스트 실행"""
    print("\\n" + "="*50)
    print("  RAG PIPELINE TEST")
    print("="*50)
    
    tests = [
        ("Embedder", test_embedder),
        ("Vector Store", test_vector_store),
        ("Chunker", test_chunker),
        ("Retriever", test_retriever),
        ("Diff Pipeline", test_diff_pipeline),
        ("History Manager", test_history_manager),
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
        print("✅ All RAG tests passed!")
        return True
    else:
        print("❌ Some RAG tests failed.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
