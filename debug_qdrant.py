import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from domain.rag.vector_store import get_collection_stats, get_versions_by_document_type

stats = get_collection_stats()
print(f"총 청크 수: {stats['total_chunks']}")
print(f"버전 목록: {stats['versions']}")
print(f"최신 버전: {stats['latest_version']}")

versions = get_versions_by_document_type("전력시장운영규칙")
print("\n=== 버전별 상세 ===")
for v in versions:
    print(f"  버전={v['버전']} | is_latest={v['is_latest']} | chunk_count={v['chunk_count']}")

# 검색 테스트
print("\n=== 벡터 검색 테스트 ===")
from domain.rag.vector_store import get_vector_store
vs = get_vector_store()
results = vs.search("계통한계가격 산정 방법", top_k=3, where={"is_latest": True})
print(f"is_latest=True 검색 결과: {len(results)}건")
for r in results:
    print(f"  조문번호={r['metadata'].get('조문번호','?')} | 버전={r['metadata'].get('버전','?')} | score={1-r['distance']:.3f}")
    print(f"  내용: {r['document'][:80]}...")
