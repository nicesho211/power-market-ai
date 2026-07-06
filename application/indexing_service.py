"""
PDF 인덱싱 서비스

PDF 업로드 → 텍스트 추출 → 청킹 → 임베딩 → ChromaDB 저장까지
전체 흐름을 관리한다.

run_indexing_stream() : 제너레이터 — 진행 상황을 yield로 실시간 보고.
                        배경 스레드에서 실행하고 st.session_state로 UI에 전달한다.
run_indexing()        : 제너레이터를 소비해 최종 결과 dict 반환 (하위 호환).
"""

import re
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Generator
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import get_settings
from domain.rag.document_loader import get_document_loader, detect_document_type
from domain.rag.chunker import get_chunker
from domain.rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def run_indexing_stream(
    file_bytes: bytes,
    filename: str,
    document_type: Optional[str] = None,
    version: Optional[str] = None,
) -> Generator[dict, None, None]:
    """
    인덱싱 파이프라인 제너레이터.

    yield 형식:
        {"type":"progress", "step":str, "current":int, "total":int, "message":str}
        {"type":"result",   "success":bool, ...결과 필드}
    """
    t_start = time.time()
    settings = get_settings()
    pdf_dir = Path(settings.pdf_path)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    save_path = pdf_dir / filename
    # 파일이 이미 존재하면 클린업에서 삭제하지 않음 (기존 원본 보호)
    _pre_existed = save_path.exists()

    def _prog(step, current, total, msg):
        return {"type": "progress", "step": step,
                "current": current, "total": total, "message": msg}

    def _fail(message, **kw):
        return {
            "type": "result", "success": False, "message": message,
            "document_type": document_type or "",
            "version": kw.get("version", ""),
            "chunk_count": 0, "failed_chunks": 0,
            "detection_rate": 0.0,
            "elapsed_seconds": round(time.time() - t_start, 1),
        }

    # ── Step 1: 파일 저장 ───────────────────────────────────────────────────
    yield _prog("extract", 0, 4, "💾 파일 저장 중...")
    try:
        save_path.write_bytes(file_bytes)
    except Exception as e:
        yield _fail(f"파일 저장 실패: {e}")
        return

    try:
        # ── Step 2: PDF → 텍스트 추출 ───────────────────────────────────────
        yield _prog("extract", 1, 4, "📄 PDF 텍스트 추출 중... (1,300페이지는 약 30~60초 소요)")
        t2 = time.time()

        # 페이지 수 파악 후 배치 추출 (100페이지씩 yield)
        import pymupdf4llm
        try:
            import pymupdf
            fitz_doc = pymupdf.open(str(save_path))
            total_pages = len(fitz_doc)
            fitz_doc.close()
        except Exception:
            total_pages = 0

        if total_pages > 100:
            # 100페이지 배치로 분할 추출 → 진행률 표시
            text_parts = []
            page_batch = 100
            for start_p in range(0, total_pages, page_batch):
                end_p = min(start_p + page_batch, total_pages)
                part = pymupdf4llm.to_markdown(
                    str(save_path), pages=list(range(start_p, end_p))
                )
                text_parts.append(part)
                yield _prog(
                    "extract", end_p, total_pages,
                    f"📄 페이지 추출: {end_p}/{total_pages}페이지 ({end_p/total_pages:.0%})"
                )
            extracted_text = "\n".join(text_parts)
        else:
            loader = get_document_loader()
            doc = loader._load_pdf(save_path)
            extracted_text = doc["content"] if doc else ""

        if not extracted_text:
            save_path.unlink(missing_ok=True)
            yield _fail("텍스트 추출 실패 (0자). PDF에 텍스트가 포함되어 있는지 확인해주세요.")
            return

        t2_done = round(time.time() - t2, 1)
        logger.info(f"[Step 2] 추출 완료: {len(extracted_text):,}자 ({t2_done}초)")

        # 조문 감지율
        matches = re.findall(r"제\s*\d+\.\d+(?:\.\d+)*\s*조", extracted_text)
        total_paragraphs = max(extracted_text.count("\n\n"), 1)
        detection_rate = len(matches) / total_paragraphs

        if not document_type:
            document_type = detect_document_type(filename, extracted_text)

        yield _prog("extract", 4, 4,
                    f"✅ 텍스트 추출 완료: {len(extracted_text):,}자 ({t2_done}초) | {document_type}")

        # ── Step 3: 청킹 ────────────────────────────────────────────────────
        yield _prog("chunk", 1, 4, "✂️ 조문 단위 청킹 중...")
        t3 = time.time()
        chunker = get_chunker()
        chunks = chunker.chunk_document(extracted_text, filename)
        t3_done = round(time.time() - t3, 1)

        if not chunks:
            save_path.unlink(missing_ok=True)
            yield _fail("청크 생성 실패 (0개).")
            return

        total_chunks = len(chunks)
        logger.info(f"[Step 3] 청킹 완료: {total_chunks}개 ({t3_done}초)")
        yield _prog("chunk", 2, 4, f"✅ 청킹 완료: {total_chunks:,}개 ({t3_done}초)")

        # 사용자 입력 버전 우선, 없으면 파일명에서 추출
        if not version:
            version = _extract_version(filename)

        # ── Step 4: is_latest 플래그 전환 ───────────────────────────────────
        vector_store = get_vector_store()
        vector_store.update_latest_flag(new_version=version, document_type=document_type)

        # ── Step 5a: 데이터 준비 ─────────────────────────────────────────────
        documents_list: list[str] = []
        ids: list[str] = []
        metadatas: list[dict] = []

        for i, chunk in enumerate(chunks):
            meta = {
                **(chunk.get("metadata", {})),
                "document_type": document_type,
                "버전": version,
                "is_latest": True,
                "파일명": filename,
            }
            documents_list.append(chunk.get("text", chunk.get("content", "")))
            ids.append(f"{filename}_{version}_{i}")
            metadatas.append(meta)

        # ── Step 5b: 임베딩 — ThreadPoolExecutor(max_workers=2) 병렬 처리 ──
        from infrastructure.llm_client import get_embeddings
        embeddings_model = get_embeddings()
        batch_size = 100   # 100개씩 → 더 자주 진행률 갱신
        batches = [documents_list[i:i + batch_size]
                   for i in range(0, total_chunks, batch_size)]
        n_batches = len(batches)

        yield _prog("embed", 0, total_chunks,
                    f"🔢 임베딩 시작: {total_chunks:,}개 청크 / {n_batches}배치 (병렬 4)")

        all_embeddings: list = [None] * n_batches
        completed_chunks = 0
        t5b = time.time()

        with ThreadPoolExecutor(max_workers=4) as pool:
            future_to_idx = {
                pool.submit(embeddings_model.embed_documents, batch): i
                for i, batch in enumerate(batches)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    all_embeddings[idx] = future.result()
                except Exception as e:
                    logger.error(f"배치 {idx} 임베딩 실패: {e}")
                    all_embeddings[idx] = [[0.0] * 3072] * len(batches[idx])
                completed_chunks += len(batches[idx])
                yield _prog(
                    "embed", min(completed_chunks, total_chunks), total_chunks,
                    f"🔢 임베딩: {completed_chunks:,}/{total_chunks:,}개 ({completed_chunks/total_chunks:.0%})"
                )

        flat_embeddings = [vec for batch_vecs in all_embeddings for vec in batch_vecs]
        t5b_done = round(time.time() - t5b, 1)
        logger.info(f"[Step 5b] 임베딩 완료: {total_chunks}개 ({t5b_done}초)")

        # ── Step 5c: Qdrant 저장 ────────────────────────────────────────────
        yield _prog("store", 3, 4, f"💾 Qdrant 저장 중... ({total_chunks:,}개)")
        t5c = time.time()

        success_chunks = vector_store.add_documents_with_embeddings(
            documents=documents_list,
            embeddings=flat_embeddings,
            ids=ids,
            metadatas=metadatas,
            batch_size=300,
        )
        failed_chunks = total_chunks - success_chunks
        t5c_done = round(time.time() - t5c, 1)

        if success_chunks == 0:
            _rollback_latest_flag(vector_store, document_type)
            save_path.unlink(missing_ok=True)
            yield _fail("저장 전체 실패.", version=version)
            return

        elapsed = round(time.time() - t_start, 1)
        logger.info(f"[Step 5c] 저장 완료: {success_chunks}개 ({t5c_done}초)")

        msg = (
            f"인덱싱 완료 — 버전: {version}, 종류: {document_type}, "
            f"저장: {success_chunks:,}개, 총 소요: {elapsed}초"
        )
        if failed_chunks:
            msg += f" (실패: {failed_chunks}개)"
        if detection_rate < 0.5:
            msg += f"\n⚠️ 조문 감지율 낮음 ({detection_rate:.1%})"

        yield {
            "type": "result", "success": True,
            "message": msg,
            "document_type": document_type,
            "version": version,
            "chunk_count": success_chunks,
            "failed_chunks": failed_chunks,
            "detection_rate": detection_rate,
            "elapsed_seconds": elapsed,
        }

    except Exception as e:
        logger.error(f"[indexing] 오류: {e}", exc_info=True)
        if not _pre_existed:
            save_path.unlink(missing_ok=True)
        yield _fail(f"인덱싱 중 오류 발생: {e}")


def run_indexing(
    file_bytes: bytes,
    filename: str,
    document_type: Optional[str] = None,
    version: Optional[str] = None,
    progress_callback=None,
) -> Dict:
    """하위 호환 래퍼 — 제너레이터를 소비해 최종 결과 dict 반환."""
    result = {}
    for update in run_indexing_stream(file_bytes, filename, document_type, version):
        if update["type"] == "result":
            result = {k: v for k, v in update.items() if k != "type"}
    return result


def _extract_version(filename: str) -> str:
    m = re.search(r"(\d{4})[-_](\d{1,2})[-_](\d{1,2})", filename)
    if m:
        y, mo, d = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"
    return datetime.now().strftime("%Y-%m-%d")


def _rollback_latest_flag(vector_store, document_type: str) -> None:
    try:
        vector_store.rollback_latest_flag(document_type)
    except Exception as e:
        logger.error(f"[Rollback] 실패: {e}")
