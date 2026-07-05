#!/usr/bin/env python
"""RAG 파이프라인 검증 (Step 6)"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("Step 6: RAG Pipeline Validation")
print("=" * 60)

# 1. 임베더 테스트
try:
    from domain.rag.embedder import get_embedder
    embedder = get_embedder()
    test_vec = embedder.embed_text("테스트")
    print(f"✅ Embedder: {len(test_vec)}차원 작동")
except Exception as e:
    print(f"❌ Embedder: {e}")

# 2. 벡터 스토어 테스트
try:
    from domain.rag.vector_store import get_vector_store
    vs = get_vector_store()
    stats = vs.get_collection_stats()
    print(f"✅ Vector Store: {stats}")
except Exception as e:
    print(f"❌ Vector Store: {e}")

# 3. 청커 테스트
try:
    from domain.rag.chunker import get_chunker
    chunker = get_chunker()
    chunks = chunker.chunk_document("제2.4.1조 테스트\n문서 내용입니다.", "test.pdf")
    print(f"✅ Chunker: {len(chunks)}개 청크 생성")
except Exception as e:
    print(f"❌ Chunker: {e}")

# 4. 문서 로더 테스트
try:
    from domain.rag.document_loader import get_document_loader
    loader = get_document_loader()
    docs = loader.load_pdfs()
    print(f"✅ Document Loader: {len(docs)}개 PDF")
except Exception as e:
    print(f"❌ Document Loader: {e}")

# 5. 검색기 테스트
try:
    from domain.rag.retriever import get_retriever
    retriever = get_retriever()
    print(f"✅ Retriever: 초기화 성공")
except Exception as e:
    print(f"❌ Retriever: {e}")

# 6. 개정 비교 테스트
try:
    from domain.rag.diff_pipeline import get_diff_pipeline
    diff = get_diff_pipeline()
    print(f"✅ Diff Pipeline: 초기화 성공")
except Exception as e:
    print(f"❌ Diff Pipeline: {e}")

# 7. 이력 관리자 테스트
try:
    from domain.rag.history_manager import get_history_manager
    history = get_history_manager()
    print(f"✅ History Manager: 초기화 성공")
except Exception as e:
    print(f"❌ History Manager: {e}")

print("=" * 60)
print("Step 6 검증 완료")
print("=" * 60)
