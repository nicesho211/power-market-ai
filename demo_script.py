"""
Power Market AI Assistant — 데모 영상 자동 제작 스크립트 (v5, 로컬 실행 기준)

Playwright로 로컬 Streamlit 앱(localhost:8501)을 자동 조작하며 브라우저 화면을 녹화하고,
동시에 pyautogui로 터미널 창(콘솔 로그) 영역을 캡처한 뒤 ffmpeg로 두 화면을 좌(65%)/우(35%)로
합성한다. LangGraph 노드가 남기는 [PLANNING]/[EXECUTOR]/[MERGE] 로그를 demo_console.log
파일에서 직접 감지해 자막 타이밍과 터미널 강조 구간을 정한다. 고정 sleep은 사용하지 않는다.

── 실행 전 준비 ──────────────────────────────────────────────
1. 터미널 창을 화면 오른쪽에 배치해두고 (정확한 픽셀 위치는 무관 — 실제 창 위치를
   실행 시점에 조회하므로 대략 브라우저와 겹치지 않는 곳이면 됨), 아래 PowerShell 명령으로
   Streamlit을 실행. Windows 콘솔 기본 인코딩(cp949)에서 로그의 한글/이모지가 깨지거나
   출력 자체가 크래시나므로 콘솔/파이프 인코딩을 UTF-8로 맞춰야 한다. Windows PowerShell 5.1의
   Tee-Object는 -Encoding 파라미터가 없으므로 ForEach-Object + Add-Content로 대체:
     chcp 65001 > $null
     [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
     $OutputEncoding = [System.Text.Encoding]::UTF8
     $Host.UI.RawUI.WindowTitle = "DEMO_TERMINAL"
     cd "C:\\Users\\user\\Desktop\\Claude AI\\power-market-ai"
     python -m streamlit run main.py 2>&1 | ForEach-Object { $_; Add-Content -Path demo_console.log -Value $_ -Encoding utf8 }

2. 브라우저에서 http://localhost:8501 이 정상적으로 열리는지 확인 (Playwright는 별도
   headless 브라우저로 접속하므로, 이 확인은 서버가 떠 있는지 점검하는 용도)

3. pyautogui가 없으면 설치:
     pip install pyautogui

4. 이 스크립트 실행 (5개 시나리오: A=규정 단순질의, B=SMP 방향성, C=개정 비교, complex=복합 질의,
   multiturn=멀티턴):
     python demo_script.py --test-scenario A   # 시나리오 하나만 먼저 검증 (권장 순서: A→B→C→complex→multiturn)
     python demo_script.py                     # 전체 파이프라인 (인덱싱 + 5개 시나리오 + 요약)
── ──────────────────────────────────────────────────────────

결과물 (demo_output/):
    전체 실행 : demo_final.mp4, demo_subtitles.srt, timestamp_log.txt
    시나리오 테스트: demo_test_<이름>_final.mp4, demo_test_<이름>_subtitles.srt
"""
from __future__ import annotations

import argparse
import ctypes
import asyncio
import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from playwright.async_api import Page, async_playwright

if sys.platform == "win32":
    # Windows 콘솔 기본 cp949 인코딩에서 한글/이모지 출력 시 크래시 방지
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ──────────────────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────────────────
URL = "http://localhost:8501"
ROOT = Path(__file__).parent
OUT_DIR = ROOT / "demo_output"
PDF_DIR = ROOT / "data" / "pdf"
LOG_FILE = ROOT / "demo_console.log"

# 최종 화면 구성: 왼쪽 65% 브라우저 + 오른쪽 35% 터미널
TOTAL_W, TOTAL_H = 1920, 1080
BROWSER_W = round(TOTAL_W * 0.65)
TERMINAL_W = TOTAL_W - BROWSER_W
VIEWPORT = {"width": BROWSER_W, "height": TOTAL_H}

RAW_VIDEO_DIR = OUT_DIR / "raw_video"
TERMINAL_FRAMES_DIR = OUT_DIR / "terminal_frames"
TERMINAL_RAW_PATH = OUT_DIR / "terminal_raw.mp4"
COMPOSITE_RAW_PATH = OUT_DIR / "composite_raw.mp4"
TTS_DIR = OUT_DIR / "tts"
TIMESTAMP_LOG_PATH = OUT_DIR / "timestamp_log.txt"
SUBTITLES_SRT_PATH = OUT_DIR / "demo_subtitles.srt"
FINAL_MP4_PATH = OUT_DIR / "demo_final.mp4"

TERMINAL_CAPTURE_FPS = 2.0

# 로컬은 Cloud보다 빠르므로 timeout을 절반으로 줄임
TIMEOUT_GENERAL = 30_000    # 일반 질의
TIMEOUT_COMPLEX = 60_000    # 복합 질의
TIMEOUT_INDEXING = 1_200_000  # 인덱싱 — 실측 시 PDF 텍스트 추출 413초 + Qdrant 저장 278초 등
                               # 합계 721초가 나온 적이 있어 10분(600초)으로는 부족했다. 20분으로 여유.

QUERY_A_RAG = "SMP 산정 방식이 어떻게 돼?"
QUERY_B_ANALYSIS = "오늘 SMP 방향성 어때?"
QUERY_C_DIFF = "이번 개정에서 뭐가 바뀌었어?"
SCENARIO_1_QUERY = "어제 SMP 급등 이유를 데이터랑 규정 근거로 설명해줘"
SCENARIO_2_TURN_1 = "LNG 비중이 높을 때 SMP 패턴은?"
SCENARIO_2_TURN_2 = "지난 3일"

# STEP 번호 ↔ SUBTITLES 인덱스는 1:1 (SUBTITLES[i-1]이 STEP i의 나레이션).
# 1~7  : 인덱싱 파트 (run_indexing_part, 고정)
# 8~11 : 시나리오 A - 규정 단순 질의 (rag)
# 12~15: 시나리오 B - SMP 방향성 분석 (analysis_fixed)
# 16~19: 시나리오 C - 규정 개정 비교 (rag_diff)
# 20~24: 시나리오 D - 복합 질의 (complex, Agent Planning)
# 25~27: 시나리오 E - 멀티턴 대화
# 28   : 성과 요약
STEPS_A = (8, 9, 10, 11)
STEPS_B = (12, 13, 14, 15)
STEPS_C = (16, 17, 18, 19)
STEPS_COMPLEX = (20, 21, 22, 23, 24)
STEPS_MULTITURN = (25, 26, 27)
STEP_SUMMARY = 28

SUBTITLES = [
    "전력거래 담당자는 1,327페이지 전력시장운영규칙을 매번 수동으로 검색합니다. 조문 하나를 찾는 데 평균 30분이 소요됩니다.",
    "이 문제를 해결하기 위해 LangGraph 기반 Agent Planning과 Hybrid Search RAG로 규정 조회와 SMP 데이터 분석을 자동화했습니다.",
    "전력시장운영규칙 PDF를 업로드합니다. 버전 날짜를 입력하면 여러 버전을 구분해서 관리할 수 있습니다.",
    "pymupdf4llm으로 텍스트를 추출하고 조문 단위로 청킹합니다. 100페이지씩 병렬로 처리합니다.",
    "text-embedding-3-large로 임베딩을 생성합니다. 100개씩 배치 처리로 속도를 높였습니다.",
    "인덱싱 완료. 9,333개 조문이 Qdrant Cloud에 영구 저장됩니다. 서버를 재시작해도 다시 인덱싱할 필요가 없습니다.",
    "두 버전의 전력시장운영규칙이 저장되어 있습니다. 개정 전후 비교도 자동으로 가능합니다.",
    # 시나리오 A — 규정 단순 질의 (rag)
    "현행 규정 단순 질의 시나리오입니다. SMP 산정 방식을 물어보겠습니다.",
    "LLM이 방식이라는 표현을 보고 데이터 분석이 아닌 조문 검색으로 정확히 분류합니다. Intent: rag",
    "Hybrid Search로 벡터 60%와 BM25 40%를 결합해 관련 조문을 검색합니다.",
    "전력시장운영규칙 관련 조문을 정확히 인용한 답변이 생성됩니다. 출처 조문번호가 함께 표시됩니다.",
    # 시나리오 B — SMP 방향성 분석 (analysis_fixed)
    "SMP 방향성 분석 시나리오입니다. 오늘 방향성을 물어보겠습니다.",
    "공공데이터포털 API에서 SMP, 발전량, 수요 데이터를 실시간으로 수집합니다.",
    "수요, LNG 비중, 신재생 비중 3가지 지표를 기준값과 비교합니다. LLM이 아닌 Python 고정 스코어링입니다.",
    "방향성 판단 완료. 실제 데이터 기반 수치와 함께 상승, 보합, 하락 중 하나로 브리핑됩니다.",
    # 시나리오 C — 규정 개정 비교 (rag_diff)
    "규정 개정 비교 시나리오입니다. 이번 개정 내용을 물어보겠습니다.",
    "개정 관련 표현을 감지하여 두 버전 비교 모드로 분류합니다.",
    "최신 버전과 직전 버전을 자동으로 탐지합니다. 별도 버전 지정 없이 자동으로 비교합니다.",
    "개정 전후 변경된 조문이 자동으로 비교되어 출력됩니다.",
    # 시나리오 D — 복합 질의 (complex, Agent Planning)
    "네 번째 시나리오입니다. 규정 근거와 실제 데이터 분석이 동시에 필요한 복합 질의입니다.",
    "LLM이 먼저 실행 계획을 수립합니다. 필요한 스킬과 병렬 실행 여부를 스스로 판단합니다. 이것이 진짜 Agent Planning입니다.",
    "LangGraph Send API로 규정 검색과 데이터 분석이 동시에 실행됩니다. 순차 실행 대비 응답 시간이 30% 단축됩니다.",
    "두 스킬의 결과가 모두 도착하면 merge_node가 통합합니다. rag_result와 analysis_result 둘 다 존재합니다.",
    "규정 근거와 실제 수치가 통합된 답변이 생성됩니다. 이것이 이 시스템의 핵심 가치입니다.",
    # 시나리오 E — 멀티턴 대화
    "다섯 번째 시나리오입니다. 기간을 지정하지 않은 패턴 분석 질의입니다.",
    "시스템이 기간을 되묻습니다. 지난 3일이라고 답하면 원래 질문과 자동으로 병합되어 재처리됩니다.",
    "지난 3일치 실제 데이터를 수집해서 LNG 비중과 SMP의 상관 패턴을 분석합니다.",
    # 성과 요약
    "Intent 분류 100%, 조문번호 메타데이터 98.2%, 복합 시나리오 7건 전부 처리. 1,327페이지 규정을 자연어 하나로 즉시 조회합니다.",
]


# ──────────────────────────────────────────────────────────────────────────
# 타임스탬프 기록
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Recorder:
    start_time: float
    entries: list[dict] = field(default_factory=list)

    def mark(self, step: int, label: str, extra: str = "") -> float:
        elapsed = round(time.time() - self.start_time, 2)
        self.entries.append({"step": step, "t": elapsed, "label": label, "extra": extra})
        print(f"  [t={elapsed:7.2f}s] STEP {step}. {label} {extra}".rstrip())
        return elapsed

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for e in self.entries:
                f.write(f"{e['step']}\t{e['step']}\t{e['t']}\t{e['label']}\t{e['extra']}\n")


# ──────────────────────────────────────────────────────────────────────────
# 로컬 Streamlit 앱 접속
# 로컬 실행은 Streamlit Community Cloud와 달리 앱 콘텐츠가 최상위 문서에
# 바로 렌더링된다 ("/~/+/" 중첩 iframe 없음) → page 셀렉터를 그대로 사용한다.
# ──────────────────────────────────────────────────────────────────────────
async def goto_app(page: Page, timeout: int = 90_000) -> None:
    await page.goto(URL, wait_until="domcontentloaded", timeout=timeout)
    await page.wait_for_selector('[data-testid="stChatInput"]', timeout=timeout)


# ──────────────────────────────────────────────────────────────────────────
# 이벤트 감지 헬퍼 (고정 sleep 대신 사용)
# ──────────────────────────────────────────────────────────────────────────
async def wait_for_stable_text(page: Page, selector: str, timeout: int = 60000) -> str:
    """텍스트가 1.5초 동안 변하지 않으면 완료로 판단.

    selector가 여러 요소에 매치될 경우 DOM/렌더링 순서상 마지막 요소(.last)를 쓴다.
    ':last-child' CSS 의사클래스는 "자신의 부모 안에서 마지막 자식"이라는 뜻이라
    Streamlit이 각 stChatMessage를 별도 래퍼 div로 감싸는 구조에서는 (사용자 말풍선도
    자기 부모 안의 마지막 자식이므로) 최신 assistant 답변이 아니라 그보다 먼저 렌더링된
    엉뚱한 말풍선을 집어올 수 있다 — 실제로 겪은 문제."""
    loc = page.locator(selector).last
    prev_text, stable_count = "", 0
    start = time.time()
    while (time.time() - start) < timeout / 1000:
        try:
            current = await loc.inner_text()
        except Exception:
            current = ""
        if current and current == prev_text:
            stable_count += 1
            if stable_count >= 3:
                return current
        else:
            stable_count = 0
        prev_text = current
        await asyncio.sleep(0.5)
    return prev_text


async def wait_for_spinner_gone(page: Page, timeout: int = 60000) -> None:
    """Streamlit 로딩 스피너 사라질 때까지 대기"""
    try:
        await page.wait_for_selector('[data-testid="stSpinner"]', state="hidden", timeout=timeout)
    except Exception:
        pass


def _log_size() -> int:
    try:
        return LOG_FILE.stat().st_size
    except FileNotFoundError:
        return 0


async def wait_for_log(keyword: str, since_pos: int, timeout: float = 60.0) -> bool:
    """demo_console.log에서 since_pos 바이트 이후 새로 추가된 내용에 keyword가
    등장할 때까지 대기한다. since_pos를 호출 시점의 파일 크기로 주면, 이전 실행에서
    이미 쌓여있던 같은 키워드 로그를 새 이벤트로 오인하지 않는다."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
                f.seek(since_pos)
                if keyword in f.read():
                    return True
        except (FileNotFoundError, PermissionError):
            # PermissionError: PowerShell(Add-Content)이 매 줄마다 파일을 열고/닫으며
            # 쓰는 순간과 겹치면 Windows 파일 잠금으로 순간적으로 못 읽을 수 있다.
            # 폴링 루프가 0.3초 후 다시 시도하므로 그냥 건너뛴다.
            pass
        await asyncio.sleep(0.3)
    return False


def extract_log_context(keyword: str, context_lines: int = 3) -> list[str]:
    """로그 파일에서 keyword가 포함된 가장 최근 줄 주변 컨텍스트를 추출"""
    try:
        lines = [l for l in LOG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
    except (FileNotFoundError, PermissionError):
        return []
    idxs = [i for i, l in enumerate(lines) if keyword in l]
    if not idxs:
        return []
    last = idxs[-1]
    return lines[max(0, last - 1): last + context_lines]


# ──────────────────────────────────────────────────────────────────────────
# 터미널 화면 캡처 (pyautogui) — 브라우저 옆 오른쪽 35% 영역
# ──────────────────────────────────────────────────────────────────────────
TERMINAL_WINDOW_TITLE = "DEMO_TERMINAL"  # 터미널 창 실행 시 $Host.UI.RawUI.WindowTitle로 설정해둘 것


def _find_terminal_window_region() -> tuple[int, int, int, int]:
    """TERMINAL_WINDOW_TITLE 창을 찾아 실제 클라이언트 영역(화면 좌표)을 반환한다.
    창을 못 찾으면 화면 오른쪽 35% 고정 영역으로 폴백한다.
    (창 테두리/타이틀바 두께와 문자 셀 단위 리사이즈 스냅으로 인해 창 크기를
    화면 좌표와 픽셀 단위로 정확히 맞추기 어려우므로, 고정 좌표를 가정하는 대신
    실제 창 위치를 매번 조회해 캡처 영역 불일치를 근본적으로 막는다.)"""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)
    fb_w = int(screen_w * 0.35)
    fb_w -= fb_w % 2
    fb_h = screen_h - (screen_h % 2)
    fallback = (int(screen_w * 0.65), 0, fb_w, fb_h)

    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    found: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def _enum(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        if buf.value == TERMINAL_WINDOW_TITLE:
            found.append(hwnd)
        return True

    user32.EnumWindows(_enum, 0)
    if not found:
        print(f"  경고: '{TERMINAL_WINDOW_TITLE}' 창을 찾지 못해 화면 오른쪽 35% 고정 영역으로 대체합니다.")
        return fallback

    if len(found) > 1:
        # 이전 실행에서 닫지 않고 남겨둔 '{TERMINAL_WINDOW_TITLE}' 창이 있으면
        # 열거 순서상 어느 쪽이 잡힐지 보장할 수 없어 엉뚱한(죽은) 창이 녹화될 수 있다.
        # 실제로 겪은 문제 — 추측으로 하나를 고르는 대신 즉시 중단시켜 사용자가
        # 중복 창을 정리하게 한다.
        raise RuntimeError(
            f"'{TERMINAL_WINDOW_TITLE}' 제목의 창이 {len(found)}개 발견되었습니다. "
            "이전 녹화/테스트에서 닫지 않은 터미널 창이 남아있으면 엉뚱한 창이 캡처될 수 있습니다. "
            "지금 실행 중인 터미널 창 하나만 남기고 나머지를 모두 닫은 뒤 다시 시도하세요."
        )

    hwnd = found[0]
    rect = RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    if w <= 0 or h <= 0:
        print(f"  경고: '{TERMINAL_WINDOW_TITLE}' 창 크기가 비정상({w}x{h})이라 고정 영역으로 대체합니다.")
        return fallback
    # libx264 yuv420p는 짝수 가로/세로만 허용 → 홀수면 인코딩이 즉시 실패한다.
    w -= w % 2
    h -= h % 2
    return (pt.x, pt.y, w, h)


class TerminalRecorder:
    """별도 스레드에서 터미널 창 영역을 일정 간격으로 스크린샷 캡처.
    터미널 창의 실제 위치/크기를 매번 조회하므로 화면 배치가 픽셀 단위로
    정확하지 않아도 캡처 내용은 항상 창과 일치한다."""

    def __init__(self, out_dir: Path, fps: float = 2.0):
        try:
            import pyautogui
        except ImportError as e:
            raise RuntimeError(
                "pyautogui가 설치되어 있지 않습니다. 'pip install pyautogui' 실행 후 다시 시도하세요."
            ) from e
        self._pyautogui = pyautogui
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.interval = 1.0 / fps
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.frame_count = 0
        self.region = _find_terminal_window_region()
        print(f"  터미널 캡처 영역: {self.region}")

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            try:
                img = self._pyautogui.screenshot(region=self.region)
                img.save(str(self.out_dir / f"frame_{self.frame_count:06d}.png"))
                self.frame_count += 1
            except Exception as e:
                print(f"  경고: 터미널 캡처 실패: {e}")
            elapsed = time.time() - t0
            time.sleep(max(0.0, self.interval - elapsed))

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


def build_terminal_video(frames_dir: Path, fps: float, out_path: Path) -> Path | None:
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        print("  ⚠️ 터미널 캡처 프레임이 없습니다. 터미널 화면 없이 진행합니다.")
        return None
    pattern = str(frames_dir / "frame_%06d.png")
    _run([
        "ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "20",
        str(out_path),
    ])
    return out_path


def composite_browser_and_terminal(
    browser_video: Path,
    terminal_video: Path | None,
    out_path: Path,
    browser_w: int,
    total_w: int,
    height: int,
    highlight_windows: list[tuple[float, float]],
) -> Path:
    """브라우저 녹화(좌) + 터미널 캡처(우)를 hstack으로 합성.
    highlight_windows 구간에는 터미널 패널에 노란 테두리를 그려 강조한다."""
    term_w = total_w - browser_w
    box_filters = [
        f"drawbox=x={browser_w}:y=0:w={term_w}:h={height}:color=yellow@0.9:t=6:"
        f"enable='between(t,{s},{e})'"
        for s, e in highlight_windows
    ]

    if terminal_video and terminal_video.exists():
        filter_complex = (
            f"[0:v]scale={browser_w}:{height}[br];"
            f"[1:v]scale={term_w}:{height}[term];"
            f"[br][term]hstack=inputs=2[v0]"
        )
        if box_filters:
            filter_complex += ";[v0]" + ",".join(box_filters) + "[v]"
        else:
            filter_complex += ";[v0]null[v]"
        cmd = [
            "ffmpeg", "-y", "-i", str(browser_video), "-i", str(terminal_video),
            "-filter_complex", filter_complex, "-map", "[v]", "-shortest",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20", str(out_path),
        ]
    else:
        # 터미널 캡처 실패 시 브라우저 화면만 왼쪽에 두고 나머지는 검정으로 채움
        vf = f"scale={browser_w}:{height},pad={total_w}:{height}:0:0:color=black"
        cmd = [
            "ffmpeg", "-y", "-i", str(browser_video),
            "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "20", str(out_path),
        ]
    _run(cmd)
    return out_path


# ──────────────────────────────────────────────────────────────────────────
# 로그 강조 배너 (브라우저 쪽, Playwright 녹화 영상에 그대로 캡처됨)
# ──────────────────────────────────────────────────────────────────────────
_OVERLAY_JS = """
(({ title, lines, caption }) => {
    const old = document.getElementById('__demo_overlay__');
    if (old) old.remove();

    const box = document.createElement('div');
    box.id = '__demo_overlay__';
    box.style.cssText = `
        position: fixed; top: 12%; right: 3%; width: 46%; z-index: 999999;
        background: rgba(15, 23, 42, 0.96); border: 3px solid #FBBF24;
        border-radius: 14px; padding: 20px 24px; box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        font-family: 'Consolas', 'D2Coding', monospace; color: #F8FAFC;
        animation: __demo_pulse__ 1.2s ease-in-out infinite;
    `;

    const styleTag = document.createElement('style');
    styleTag.textContent = `
        @keyframes __demo_pulse__ {
            0%, 100% { box-shadow: 0 8px 32px rgba(251,191,36,0.25); }
            50% { box-shadow: 0 8px 40px rgba(251,191,36,0.65); }
        }
    `;
    document.head.appendChild(styleTag);

    const titleEl = document.createElement('div');
    titleEl.textContent = title;
    titleEl.style.cssText = 'color:#FBBF24; font-weight:700; font-size:1.05rem; margin-bottom:10px;';
    box.appendChild(titleEl);

    lines.forEach((line) => {
        const p = document.createElement('div');
        p.textContent = line;
        p.style.cssText = 'font-size:0.92rem; line-height:1.5; white-space:pre-wrap; word-break:break-all;';
        box.appendChild(p);
    });

    if (caption) {
        const arrow = document.createElement('div');
        arrow.textContent = '← ' + caption;
        arrow.style.cssText = 'margin-top:14px; color:#38BDF8; font-weight:700; font-size:1rem;';
        box.appendChild(arrow);
    }

    document.body.appendChild(box);
})(__ARGS__)
"""


async def show_highlight(page: Page, title: str, lines: list[str], caption: str) -> None:
    script = _OVERLAY_JS.replace(
        "__ARGS__", json.dumps({"title": title, "lines": lines, "caption": caption}, ensure_ascii=False)
    )
    try:
        await page.evaluate(script)
    except Exception as e:
        print(f"  ⚠️ 오버레이 표시 실패: {e}")


async def clear_highlight(page: Page) -> None:
    try:
        await page.evaluate("document.getElementById('__demo_overlay__')?.remove()")
    except Exception:
        pass


def _pdf_version(pdf_path: Path) -> str:
    """파일명에서 YYMMDD 패턴을 찾아 YYYY-MM-DD 버전 문자열로 변환"""
    m = re.search(r"(\d{2})(\d{2})(\d{2})", pdf_path.stem)
    return f"20{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else "2026-05-20"


# ──────────────────────────────────────────────────────────────────────────
# STEP 1~7. 인덱싱 파트
# ──────────────────────────────────────────────────────────────────────────
async def run_indexing_part(page: Page, rec: Recorder) -> None:
    await goto_app(page)
    await page.wait_for_selector('[data-testid="stMetric"]', timeout=30000)
    rec.mark(1, "앱_초기화면_SMP카드_등장")

    # 이미 인덱싱된 버전 확인 (사이드바 "저장된 버전: ..." 캡션 또는 "인덱싱 완료" 카드)
    sidebar_text = await page.inner_text('[data-testid="stSidebar"]')
    indexed_versions = set(re.findall(r"20\d{2}-\d{2}-\d{2}", sidebar_text))

    # STEP 2. 사이드바 PDF 업로드 위젯 클릭
    uploader = page.locator('[data-testid="stFileUploader"]').first
    try:
        await uploader.scroll_into_view_if_needed(timeout=10000)
    except Exception:
        pass
    rec.mark(2, "사이드바_PDF_업로드_위젯_클릭")

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    pending = [p for p in pdf_files if _pdf_version(p) not in indexed_versions]

    if not pending:
        # 모든 버전이 이미 인덱싱되어 있음 → STEP 3~5는 짧게 생략 표기
        rec.mark(3, "PDF_업로드_및_버전날짜입력_생략", "이미완료")
        rec.mark(4, "인덱싱_시작_생략", "이미완료")
        rec.mark(5, "임베딩_로그_확대_생략", "이미완료")
        rec.mark(6, "인덱싱_완료_감지_생략_이미완료")
    else:
        pdf_path = pending[0]
        version = _pdf_version(pdf_path)
        since_pos = _log_size()

        await uploader.locator("input[type=file]").set_input_files(str(pdf_path))
        await page.wait_for_timeout(1500)  # 업로드 후 UI 갱신 대기 (파일 파싱은 짧음)

        version_input = page.get_by_placeholder(re.compile("예:"))
        try:
            await version_input.fill(version)
        except Exception:
            pass
        rec.mark(3, "PDF_업로드_및_버전날짜입력", version)

        start_btn = page.get_by_role("button", name=re.compile("업로드 및 인덱싱 시작"))
        try:
            await start_btn.click(timeout=10000)
        except Exception as e:
            print(f"  경고: 인덱싱 시작 버튼 클릭 실패: {e}")

        # STEP 4. 인덱싱 시작 — indexing_service의 [INDEX] 인덱싱 시작 로그로 실제 시작을 확인
        # (버튼 클릭만으로는 백엔드가 실제로 처리를 시작했는지 알 수 없다 — 다른 시나리오와
        # 동일하게 터미널 로그를 근거로 삼는다.)
        await wait_for_log("[INDEX] 인덱싱 시작", since_pos, timeout=30)
        rec.mark(4, "인덱싱_시작")

        # STEP 5. 텍스트 추출 + 조문 단위 청킹 완료 감지 (1,300여 페이지 처리 — 수 분 소요)
        await wait_for_log("[INDEX] 청킹 완료", since_pos, timeout=TIMEOUT_INDEXING / 1000)
        rec.mark(5, "청킹_완료_감지")

        # STEP 6. 임베딩 생성 완료 감지
        await wait_for_log("[INDEX] 임베딩 완료", since_pos, timeout=TIMEOUT_INDEXING / 1000)
        rec.mark(6, "임베딩_완료_감지")

        # Qdrant 저장 완료(=인덱싱 전체 완료) 확인 후 UI에도 반영됐는지 확인
        await wait_for_log("[INDEX] Qdrant 저장 완료", since_pos, timeout=TIMEOUT_INDEXING / 1000)
        try:
            await page.wait_for_function(
                "document.body.innerText.includes('인덱싱 완료!') && "
                "document.body.innerText.includes('소요:')",
                timeout=30000,
            )
        except Exception as e:
            print(f"  경고: 인덱싱 완료 UI 반영 감지 실패: {e}")

    # STEP 7. 버전 목록 확인
    await page.wait_for_timeout(1000)
    rec.mark(7, "버전_목록_확인")


def _chat_input(page: Page):
    # .last: 절전/네트워크 끊김으로 세션이 재연결되면 이전 페이지가 완전히
    # 언마운트되기 전에 새 인스턴스가 겹쳐 렌더링되어 동일 selector가 2개
    # 잡힐 수 있다(strict mode violation). 나중에 마운트된(활성) 쪽을 선택.
    return page.locator('[data-testid="stChatInput"] textarea').last


# ──────────────────────────────────────────────────────────────────────────
# 시나리오 A. 규정 단순 질의 (rag)
# ──────────────────────────────────────────────────────────────────────────
async def run_scenario_rag(page: Page, rec: Recorder, steps: tuple[int, int, int, int]) -> None:
    s_input, s_classify, s_search, s_done = steps

    if not LOG_FILE.exists():
        print(f"  ⚠️ {LOG_FILE} 이 없습니다. 로그 강조 없이 진행합니다.")
    since_pos = _log_size()

    chat_input = _chat_input(page)
    await chat_input.click()
    await chat_input.fill(QUERY_A_RAG)
    await chat_input.press("Enter")
    rec.mark(s_input, "규정_단순질의_입력", QUERY_A_RAG)

    # intent=rag_diff 등 다른 intent와 겹치지 않도록 이번 요청 이후(since_pos)
    # 새로 쌓인 로그에서만 "intent=rag"를 찾는다.
    await wait_for_log("intent=rag", since_pos, timeout=30)
    rec.mark(s_classify, "Intent_rag_분류_감지")

    await wait_for_log("[RAG]", since_pos, timeout=30)
    rec.mark(s_search, "Hybrid_Search_실행_감지")

    # format_output_node가 실제로 최종 답변을 완성한 뒤에만 "완료"로 본다.
    # (스피너가 진행 단계 사이에 순간적으로 사라졌다 나타나거나, progress_steps
    # 패널이 갱신 없이 잠시 멈춰있는 동안 wait_for_stable_text가 아직 완성되지
    # 않은 중간 상태를 "안정됨"으로 오판할 수 있어, 로그로 완료를 먼저 확정한다.)
    await wait_for_log("답변 생성 완료", since_pos, timeout=TIMEOUT_GENERAL / 1000)
    await wait_for_spinner_gone(page, timeout=TIMEOUT_GENERAL)
    await wait_for_stable_text(page, '[data-testid="stChatMessage"]', timeout=TIMEOUT_GENERAL)
    rec.mark(s_done, "규정_답변_완료")
    await page.wait_for_timeout(1000)


# ──────────────────────────────────────────────────────────────────────────
# 시나리오 B. SMP 방향성 분석 (analysis_fixed)
# ──────────────────────────────────────────────────────────────────────────
async def run_scenario_analysis(page: Page, rec: Recorder, steps: tuple[int, int, int, int]) -> None:
    s_input, s_api, s_score, s_done = steps
    since_pos = _log_size()

    chat_input = _chat_input(page)
    await chat_input.click()
    await chat_input.fill(QUERY_B_ANALYSIS)
    await chat_input.press("Enter")
    rec.mark(s_input, "SMP_방향성_질의_입력", QUERY_B_ANALYSIS)

    await wait_for_log("[API]", since_pos, timeout=30)
    rec.mark(s_api, "공공데이터_API_호출_감지")

    await wait_for_log("[SCORE]", since_pos, timeout=30)
    rec.mark(s_score, "스코어링_로직_실행_감지")

    # SCORE 계산 후에도 LLM 요약 호출(수 초~10여 초)이 더 남아있으므로, 그 사이 progress
    # 패널이 잠깐 멈춰있는 상태를 wait_for_stable_text가 "완료"로 오판하지 않도록
    # format_output_node의 실제 완료 로그를 먼저 기다린다.
    await wait_for_log("답변 생성 완료", since_pos, timeout=60)
    await wait_for_spinner_gone(page, timeout=TIMEOUT_GENERAL)
    answer = await wait_for_stable_text(page, '[data-testid="stChatMessage"]', timeout=TIMEOUT_GENERAL)
    if not any(k in answer for k in ("상승", "보합", "하락")):
        print("  경고: 방향성 답변에서 상승/보합/하락 키워드를 찾지 못했습니다.")
    rec.mark(s_done, "방향성_답변_완료")
    await page.wait_for_timeout(1000)


# ──────────────────────────────────────────────────────────────────────────
# 시나리오 C. 규정 개정 비교 (rag_diff)
# ──────────────────────────────────────────────────────────────────────────
async def run_scenario_diff(page: Page, rec: Recorder, steps: tuple[int, int, int, int]) -> None:
    s_input, s_classify, s_version, s_done = steps
    since_pos = _log_size()

    chat_input = _chat_input(page)
    await chat_input.click()
    await chat_input.fill(QUERY_C_DIFF)
    await chat_input.press("Enter")
    rec.mark(s_input, "개정_비교_질의_입력", QUERY_C_DIFF)

    await wait_for_log("intent=rag_diff", since_pos, timeout=30)
    rec.mark(s_classify, "Intent_rag_diff_분류_감지")

    await wait_for_log("[RAG]", since_pos, timeout=60)
    rec.mark(s_version, "버전_자동탐지_감지")

    await wait_for_log("답변 생성 완료", since_pos, timeout=60)
    await wait_for_spinner_gone(page, timeout=TIMEOUT_GENERAL)
    await wait_for_stable_text(page, '[data-testid="stChatMessage"]', timeout=TIMEOUT_GENERAL)
    rec.mark(s_done, "개정비교_답변_완료")
    await page.wait_for_timeout(1000)


# ──────────────────────────────────────────────────────────────────────────
# 시나리오 D. 복합 질의 (complex, Agent Planning)
# ──────────────────────────────────────────────────────────────────────────
async def run_scenario_complex(page: Page, rec: Recorder, steps: tuple[int, int, int, int, int]) -> list[tuple[float, float]]:
    """복합 질의 시나리오. 반환값은 터미널 패널에 노란 테두리를 그릴 (start, end) 구간 목록."""
    s_input, s_planning, s_executor, s_merge, s_done = steps
    highlight_windows: list[tuple[float, float]] = []

    if not LOG_FILE.exists():
        print(f"  ⚠️ {LOG_FILE} 이 없습니다. 'python -m streamlit run main.py 2>&1 | tee demo_console.log'로 "
              "앱을 먼저 실행했는지 확인하세요. 로그 강조 없이 진행합니다.")

    since_pos = _log_size()

    chat_input = _chat_input(page)
    await chat_input.click()
    await chat_input.fill(SCENARIO_1_QUERY)
    await chat_input.press("Enter")
    rec.mark(s_input, "복합_질의_입력", SCENARIO_1_QUERY)

    # PLANNING 로그 감지 → 브라우저 배너 + 터미널 강조 구간 기록
    found = await wait_for_log("[PLANNING]", since_pos, timeout=60)
    lines = extract_log_context("[PLANNING]", context_lines=3) if found else []
    await show_highlight(
        page, "🧠 Agent Planning",
        lines or ["[PLANNING] LLM이 실행 계획을 수립 중..."],
        "LLM이 직접 실행 계획 수립!",
    )
    t_start = rec.mark(s_planning, "PLANNING_로그_강조")
    await page.wait_for_timeout(5000)
    highlight_windows.append((t_start, t_start + 5.0))

    # EXECUTOR 병렬 실행 로그 강조
    found = await wait_for_log("[EXECUTOR]", since_pos, timeout=60)
    lines = extract_log_context("[EXECUTOR]", context_lines=2) if found else []
    await show_highlight(page, "⚙️ Parallel Execution", lines or ["[EXECUTOR] 병렬 실행 시작"], "병렬 실행 시작!")
    t_start = rec.mark(s_executor, "EXECUTOR_병렬실행_로그_강조")
    await page.wait_for_timeout(4000)
    highlight_windows.append((t_start, t_start + 4.0))

    # MERGE 로그 강조
    found = await wait_for_log("[MERGE]", since_pos, timeout=90)
    lines = extract_log_context("[MERGE]", context_lines=2) if found else []
    await show_highlight(page, "🔗 Merge", lines or ["[MERGE] 결과 통합 완료"], "결과 통합 완료!")
    t_start = rec.mark(s_merge, "결과_통합_로그_강조")
    await page.wait_for_timeout(3000)
    highlight_windows.append((t_start, t_start + 3.0))
    await clear_highlight(page)

    # 복합 답변 완료 감지 — merge_node 이후에도 통합 답변용 LLM 호출이 남아있으므로
    # format_output_node의 실제 완료 로그를 먼저 기다린 뒤에만 "완료"로 본다.
    await wait_for_log("답변 생성 완료", since_pos, timeout=TIMEOUT_COMPLEX / 1000)
    await wait_for_spinner_gone(page, timeout=TIMEOUT_COMPLEX)
    await wait_for_stable_text(page, '[data-testid="stChatMessage"]', timeout=TIMEOUT_COMPLEX)
    rec.mark(s_done, "복합_답변_완료")
    await page.wait_for_timeout(1500)

    return highlight_windows


# ──────────────────────────────────────────────────────────────────────────
# 시나리오 E. 멀티턴 대화
# ──────────────────────────────────────────────────────────────────────────
async def run_scenario_multiturn(page: Page, rec: Recorder, steps: tuple[int, int, int]) -> None:
    s_turn1, s_clarify, s_done = steps
    chat_input = _chat_input(page)

    await chat_input.click()
    await chat_input.fill(SCENARIO_2_TURN_1)
    await chat_input.press("Enter")
    rec.mark(s_turn1, "멀티턴_질의_입력", SCENARIO_2_TURN_1)

    await wait_for_spinner_gone(page, timeout=TIMEOUT_GENERAL)
    await wait_for_stable_text(page, '[data-testid="stChatMessage"]', timeout=TIMEOUT_GENERAL)
    rec.mark(s_clarify, "clarify_재질문_감지")
    await page.wait_for_timeout(1000)

    since_pos = _log_size()
    await chat_input.click()
    await chat_input.fill(SCENARIO_2_TURN_2)
    await chat_input.press("Enter")

    await wait_for_log("답변 생성 완료", since_pos, timeout=TIMEOUT_COMPLEX / 1000)
    await wait_for_spinner_gone(page, timeout=TIMEOUT_COMPLEX)
    await wait_for_stable_text(page, '[data-testid="stChatMessage"]', timeout=TIMEOUT_COMPLEX)
    rec.mark(s_done, "멀티턴_분석결과_표시")
    await page.wait_for_timeout(1500)


# ──────────────────────────────────────────────────────────────────────────
# 성과 요약
# ──────────────────────────────────────────────────────────────────────────
async def run_summary(page: Page, rec: Recorder, step: int) -> None:
    await page.evaluate("window.scrollTo(0, 0)")
    await page.wait_for_timeout(800)

    summary_js = """
    (() => {
        document.getElementById('__demo_summary__')?.remove();
        const box = document.createElement('div');
        box.id = '__demo_summary__';
        box.style.cssText = `
            position: fixed; top: 30%; left: 50%; transform: translateX(-50%);
            z-index: 999999; background: rgba(15,23,42,0.96); border: 3px solid #34D399;
            border-radius: 16px; padding: 26px 34px; box-shadow: 0 8px 40px rgba(0,0,0,0.5);
            font-family: 'Inter', sans-serif; color: #F8FAFC; text-align: left; min-width: 420px;
        `;
        box.innerHTML = `
            <div style="font-size:1.1rem;font-weight:700;color:#34D399;margin-bottom:14px;">✅ 성과 요약</div>
            <div style="font-size:1rem;line-height:2;">
              Intent 분류 정확도&nbsp;&nbsp;&nbsp;<b style="float:right;">100%</b><br/>
              조문번호 메타데이터&nbsp;&nbsp;<b style="float:right;">98.2%</b><br/>
              복합 시나리오 성공&nbsp;&nbsp;&nbsp;&nbsp;<b style="float:right;">7 / 7</b>
            </div>
        `;
        document.body.appendChild(box);
    })()
    """
    try:
        await page.evaluate(summary_js)
    except Exception as e:
        print(f"  ⚠️ 요약 오버레이 실패: {e}")

    rec.mark(step, "성과_요약_화면")
    await page.wait_for_timeout(5000)


# ──────────────────────────────────────────────────────────────────────────
# 시나리오 테스트 모드 설정 — 이름 → (SUBTITLES 슬라이스 범위, 로컬 STEP 개수)
# 전체 실행(mode="full")은 STEPS_A/B/C/COMPLEX/MULTITURN/STEP_SUMMARY(전역 번호)를 쓰고,
# 개별 시나리오 테스트는 항상 로컬 1..k 번호를 써서 그 시나리오의 SUBTITLES 슬라이스와
# 1:1 대응시킨다 (build_final_video가 STEP 번호=슬라이스 인덱스로 가정하기 때문).
# ──────────────────────────────────────────────────────────────────────────
SCENARIO_SLICES: dict[str, tuple[int, int]] = {
    "A": (7, 11),
    "B": (11, 15),
    "C": (15, 19),
    "complex": (19, 24),
    "multiturn": (24, 27),
}


# ──────────────────────────────────────────────────────────────────────────
# 메인 실행 (Playwright + 터미널 캡처 + 합성)
# ──────────────────────────────────────────────────────────────────────────
async def record_run(mode: str) -> tuple[Path, list[tuple[float, float]], list[str]]:
    """mode="full": 인덱싱+5개 시나리오 전체. 그 외("A","B","C","complex","multiturn"):
    goto_app 후 해당 시나리오 하나만 로컬 STEP 1..k 번호로 실행."""
    OUT_DIR.mkdir(exist_ok=True)
    RAW_VIDEO_DIR.mkdir(exist_ok=True)

    if TERMINAL_FRAMES_DIR.exists():
        for f in TERMINAL_FRAMES_DIR.glob("*.png"):
            f.unlink()
    TERMINAL_FRAMES_DIR.mkdir(exist_ok=True)

    term_rec = TerminalRecorder(TERMINAL_FRAMES_DIR, fps=TERMINAL_CAPTURE_FPS)

    highlight_windows: list[tuple[float, float]] = []
    subtitles_used: list[str] = SUBTITLES

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=str(RAW_VIDEO_DIR),
            record_video_size=VIEWPORT,
        )
        page = await context.new_page()

        # 브라우저 녹화 시작(context 생성) 직후 Recorder/터미널 캡처를 시작해
        # 세 시계(Playwright 영상, Recorder, 터미널 캡처)의 t=0을 최대한 맞춘다.
        rec = Recorder(start_time=time.time())
        term_rec.start()

        try:
            if mode == "full":
                await run_indexing_part(page, rec)
                await run_scenario_rag(page, rec, STEPS_A)
                await run_scenario_analysis(page, rec, STEPS_B)
                await run_scenario_diff(page, rec, STEPS_C)
                highlight_windows = await run_scenario_complex(page, rec, STEPS_COMPLEX)
                await run_scenario_multiturn(page, rec, STEPS_MULTITURN)
                await run_summary(page, rec, STEP_SUMMARY)
            else:
                start, end = SCENARIO_SLICES[mode]
                subtitles_used = SUBTITLES[start:end]
                await goto_app(page)
                if mode == "A":
                    await run_scenario_rag(page, rec, (1, 2, 3, 4))
                elif mode == "B":
                    await run_scenario_analysis(page, rec, (1, 2, 3, 4))
                elif mode == "C":
                    await run_scenario_diff(page, rec, (1, 2, 3, 4))
                elif mode == "complex":
                    highlight_windows = await run_scenario_complex(page, rec, (1, 2, 3, 4, 5))
                elif mode == "multiturn":
                    await run_scenario_multiturn(page, rec, (1, 2, 3))
        finally:
            term_rec.stop()
            video_path_obj = page.video
            await context.close()
            await browser.close()

        raw_video_path = Path(await video_path_obj.path()) if video_path_obj else None

    rec.save(TIMESTAMP_LOG_PATH)
    print(f"\n✅ 브라우저 녹화 완료: {raw_video_path}")
    print(f"✅ 타임스탬프 로그 저장: {TIMESTAMP_LOG_PATH}")
    print(f"✅ 터미널 캡처 프레임: {term_rec.frame_count}장")

    terminal_video = build_terminal_video(TERMINAL_FRAMES_DIR, TERMINAL_CAPTURE_FPS, TERMINAL_RAW_PATH)

    composite_path = COMPOSITE_RAW_PATH if mode == "full" else (OUT_DIR / f"demo_test_{mode}.raw.mp4")
    composite_browser_and_terminal(
        browser_video=raw_video_path,
        terminal_video=terminal_video,
        out_path=composite_path,
        browser_w=BROWSER_W, total_w=TOTAL_W, height=TOTAL_H,
        highlight_windows=highlight_windows,
    )
    print(f"✅ 브라우저+터미널 화면 합성 완료: {composite_path}")

    return composite_path, highlight_windows, subtitles_used


# ──────────────────────────────────────────────────────────────────────────
# 후처리: 자막(.srt) 생성, TTS 합성, 배속 조정, 최종 mp4 렌더링
# ──────────────────────────────────────────────────────────────────────────
def _srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_subtitles_srt(timestamps: list[float], durations: list[float], subtitles: list[str],
                         srt_path: Path = SUBTITLES_SRT_PATH) -> None:
    lines = []
    for i, (text, start, dur) in enumerate(zip(subtitles, timestamps, durations), start=1):
        end = start + dur
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(start)} --> {_srt_timestamp(end)}")
        lines.append(text)
        lines.append("")
    srt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ 자막 파일 저장: {srt_path}")


def synthesize_tts(subtitles: list[str]) -> list[Path | None]:
    """subtitles와 동일한 순서/길이로 반환 (실패한 항목은 None — 인덱스 정렬 유지가 핵심).
    파일명은 텍스트 내용의 해시로 캐싱한다 — 전체 실행과 개별 시나리오 테스트가 서로 다른
    STEP 번호 체계(전역 vs 로컬 1..k)를 쓰므로, 위치 기반 파일명(01.mp3 등)을 쓰면
    시나리오 테스트에서 엉뚱한 텍스트의 캐시된 오디오를 재생하는 사고가 날 수 있다."""
    import hashlib
    from gtts import gTTS

    TTS_DIR.mkdir(exist_ok=True)
    paths: list[Path | None] = []
    for i, text in enumerate(subtitles, start=1):
        h = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
        out = TTS_DIR / f"{h}.mp3"
        if not out.exists():
            try:
                gTTS(text=text, lang="ko").save(str(out))
            except Exception as e:
                print(f"  경고: TTS 생성 실패 (자막 {i}): {e}")
                paths.append(None)
                continue
        paths.append(out)
    ok = sum(1 for p in paths if p is not None)
    print(f"✅ TTS {ok}/{len(subtitles)}개 생성 완료")
    return paths


def _run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


# 나레이션 재생이 끝난 뒤 자막/화면이 살짝 더 머무는 여유 시간
NARRATION_PAD_SEC = 0.6
MIN_SEGMENT_SEC = 2.5
MAX_SEGMENT_SEC = 14.0

# STEP별 구간 길이를 나레이션 길이만으로 정하면, 실제 화면 소스가 나레이션보다
# 훨씬 긴 구간(예: EXECUTOR 병렬실행 로그가 실제로는 90초 가까이 쌓이는데
# 그 부분 나레이션은 12초 안팎)은 배속이 과해져 로그가 읽을 수 없을 만큼
# 빨리 지나간다. 그런 STEP은 나레이션 길이와 무관하게 최소 재생 길이를 강제한다.
# 전체 실행과 개별 시나리오 테스트는 STEP 번호 체계가 다르므로(전역 vs 로컬 1..k),
# 번호가 아니라 자막 텍스트로 대상 구간을 식별한다.
SEGMENT_TEXT_MIN_OVERRIDE: dict[str, float] = {
    "LangGraph Send API로 규정 검색과 데이터 분석이 동시에 실행됩니다. 순차 실행 대비 응답 시간이 30% 단축됩니다.": 24.0,
    # 인덱싱 청킹/임베딩 구간도 실제로는 수 분 걸리므로(1,300여 페이지), 나레이션 길이
    # (~8초)로만 압축하면 터미널 로그가 스크롤하는 모습을 보여줄 새도 없이 지나간다.
    "pymupdf4llm으로 텍스트를 추출하고 조문 단위로 청킹합니다. 100페이지씩 병렬로 처리합니다.": 20.0,
    "text-embedding-3-large로 임베딩을 생성합니다. 100개씩 배치 처리로 속도를 높였습니다.": 20.0,
}
# 원본 영상에서 한 구간을 위해 최소한 이만큼의 실제 화면 소스를 확보
# (STEP 이벤트가 밀리초 단위로 붙어있는 경우, 얼어붙은 한 프레임을 늘리는 대신
#  다음 구간의 초반부를 조금 빌려와 자연스러운 정지 화면처럼 보이게 한다)
MIN_SOURCE_SEC = 1.0


def build_final_video(raw_video: Path, entries: list[dict], subtitles: list[str] = SUBTITLES,
                       out_path: Path = FINAL_MP4_PATH, srt_path: Path = SUBTITLES_SRT_PATH,
                       tmp_dir: Path | None = None) -> Path:
    """자막 나레이션(TTS) 길이를 기준으로 각 구간의 영상 재생 속도를 맞춰
    화면·자막·음성이 항상 정확히 같은 구간 경계에서 시작/종료하도록 만든다.
    (이벤트 발생 간격을 그대로 자막 타이밍에 쓰면, 나레이션이 몇 초 걸리는 사이
     다음 자막의 오디오가 먼저 시작되어 여러 음성이 겹치는 문제가 생긴다.)

    entries의 STEP 번호는 subtitles 슬라이스와 1:1 대응하는 1..len(subtitles) 로컬 번호여야
    한다 (전체 실행은 STEP 1..28 = SUBTITLES 전체, 개별 시나리오 테스트는 그 시나리오만의
    로컬 1..k 번호)."""
    if not entries:
        raise RuntimeError("타임스탬프 기록이 없습니다.")

    n = len(subtitles)
    raw_duration = _ffprobe_duration(raw_video)

    # STEP 1..n에 대응하는 원본 구간 (raw_start, raw_end)
    step_to_t = {e["step"]: e["t"] for e in entries}
    raw_segments: list[tuple[float, float]] = []
    for idx in range(1, n + 1):
        s = step_to_t.get(idx, step_to_t.get(idx - 1, 0.0))
        nxt_idx = idx + 1
        e = step_to_t.get(nxt_idx, min(s + 6.0, raw_duration))
        e = max(e, s + 0.04)  # 최소 1프레임 폭 보장
        raw_segments.append((s, e))

    tmp_dir = tmp_dir or (OUT_DIR / "_segments")
    tmp_dir.mkdir(exist_ok=True)

    # 1) TTS 먼저 생성 → 실제 나레이션 길이를 구간 길이의 기준으로 삼는다
    tts_paths = synthesize_tts(subtitles)
    targets: list[float] = []
    for i, tp in enumerate(tts_paths):
        min_sec = SEGMENT_TEXT_MIN_OVERRIDE.get(subtitles[i], MIN_SEGMENT_SEC)
        max_sec = max(MAX_SEGMENT_SEC, min_sec)  # 오버라이드가 기본 상한보다 크면 상한도 함께 올림
        dur = _ffprobe_duration(tp) if tp else 0.0
        target = dur + NARRATION_PAD_SEC if dur > 0 else min_sec
        targets.append(max(min_sec, min(max_sec, target)))

    # 2) 각 구간을 target 길이에 맞춰 setpts로 시간 재조정 (필요하면 다음 구간 시작부를 살짝 빌려온다)
    concat_list_path = tmp_dir / "concat.txt"
    seg_paths = []
    for i, ((s, e), target) in enumerate(zip(raw_segments, targets)):
        src_end = e
        if (src_end - s) < MIN_SOURCE_SEC:
            src_end = min(s + MIN_SOURCE_SEC, raw_duration)
        src_dur = max(src_end - s, 0.04)

        seg_out = tmp_dir / f"seg_{i:03d}.mp4"
        pts_factor = target / src_dur
        vf = f"setpts={pts_factor}*PTS"
        cmd = [
            "ffmpeg", "-y", "-ss", f"{s}", "-to", f"{src_end}", "-i", str(raw_video),
            "-vf", vf, "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            str(seg_out),
        ]
        try:
            _run(cmd)
            seg_paths.append(seg_out)
        except subprocess.CalledProcessError as ex:
            print(f"  경고: 구간 {i} 인코딩 실패, 건너뜀: {ex}")

    with open(concat_list_path, "w", encoding="utf-8") as f:
        for sp in seg_paths:
            f.write(f"file '{sp.as_posix()}'\n")

    speed_adjusted = tmp_dir / "speed_adjusted.mp4"
    _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list_path),
          "-c", "copy", str(speed_adjusted)])

    # 3) 새 타임라인: 각 구간의 시작/길이 = 영상 구간과 완전히 동일 (동기화 보장)
    sub_starts: list[float] = []
    acc = 0.0
    for target in targets:
        sub_starts.append(acc)
        acc += target

    build_subtitles_srt(sub_starts, targets, subtitles, srt_path=srt_path)

    # 4) TTS 오디오를 정확히 같은 위치에 배치 (target >= 나레이션 길이이므로 겹치지 않음)
    audio_mixed = tmp_dir / "tts_mixed.mp3"
    valid = [(i, tp) for i, tp in enumerate(tts_paths) if tp is not None]
    if valid:
        inputs = []
        filter_parts = []
        for j, (i, tp) in enumerate(valid):
            inputs += ["-i", str(tp)]
            delay_ms = int(sub_starts[i] * 1000)
            filter_parts.append(f"[{j}:a]adelay={delay_ms}|{delay_ms}[a{j}]")
        amix_inputs = "".join(f"[a{j}]" for j in range(len(valid)))
        filter_complex = ";".join(filter_parts) + f";{amix_inputs}amix=inputs={len(valid)}:normalize=0[aout]"
        try:
            _run(["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex,
                  "-map", "[aout]", str(audio_mixed)])
        except subprocess.CalledProcessError as ex:
            print(f"  경고: TTS 합성 실패, 자막만 사용: {ex}")
            audio_mixed = None
    else:
        audio_mixed = None

    # 5) 자막 번인 + 오디오 합성
    srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")
    if audio_mixed and audio_mixed.exists():
        _run([
            "ffmpeg", "-y", "-i", str(speed_adjusted), "-i", str(audio_mixed),
            "-vf", f"subtitles='{srt_escaped}'",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-shortest",
            str(out_path),
        ])
    else:
        _run([
            "ffmpeg", "-y", "-i", str(speed_adjusted),
            "-vf", f"subtitles='{srt_escaped}'",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            str(out_path),
        ])

    print(f"\n🎉 최종 영상 완료: {out_path} (총 길이 약 {acc:.1f}초)")
    return out_path


# ──────────────────────────────────────────────────────────────────────────
# 엔트리 포인트
# ──────────────────────────────────────────────────────────────────────────
def _read_entries() -> list[dict]:
    entries = []
    with open(TIMESTAMP_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                entries.append({"step": int(parts[0]), "t": float(parts[2])})
    return entries


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-scenario", choices=["A", "B", "C", "complex", "multiturn"], default=None,
        help="지정한 시나리오 1개만 단독 실행/렌더링 (인덱싱 생략, 자막-화면-터미널 동기화 검증용)",
    )
    parser.add_argument("--test", action="store_true",
                         help="--test-scenario complex 의 별칭 (하위 호환)")
    parser.add_argument("--skip-postprocess", action="store_true", help="녹화만 하고 ffmpeg 후처리는 건너뜀")
    parser.add_argument("--postprocess-only", action="store_true",
                         help="재녹화 없이 기존 합성 영상 + timestamp_log.txt로 후처리만 재실행 (전체 실행 전용)")
    args = parser.parse_args()

    mode = args.test_scenario or ("complex" if args.test else "full")

    if args.postprocess_only:
        if mode != "full":
            print("❌ --postprocess-only는 전체 실행 결과에만 사용할 수 있습니다 (--test-scenario와 함께 쓸 수 없음).")
            return
        if not COMPOSITE_RAW_PATH.exists():
            print(f"❌ {COMPOSITE_RAW_PATH} 가 없습니다. 먼저 전체 녹화를 한 번 실행하세요.")
            return
        print(f"기존 합성 영상 재사용: {COMPOSITE_RAW_PATH}")
        build_final_video(COMPOSITE_RAW_PATH, _read_entries(), SUBTITLES)
        return

    composite_path, _, subtitles_used = await record_run(mode)

    if mode != "full":
        test_out = OUT_DIR / f"demo_test_{mode}.mp4"
        if composite_path and composite_path.exists():
            if composite_path != test_out:
                _run(["ffmpeg", "-y", "-i", str(composite_path), "-c", "copy", str(test_out)])
            print(f"\n✅ 테스트 영상(합성) 저장: {test_out}")
        if args.skip_postprocess:
            print("후처리 건너뜀 (--skip-postprocess)")
            return
        test_final = OUT_DIR / f"demo_test_{mode}_final.mp4"
        test_srt = OUT_DIR / f"demo_test_{mode}_subtitles.srt"
        build_final_video(composite_path, _read_entries(), subtitles_used,
                           out_path=test_final, srt_path=test_srt)
        print(f"   → {test_final} 에서 자막-화면-터미널 로그 동기화를 확인한 뒤 "
              "'python demo_script.py' (전체 실행)로 진행하세요.")
        return

    if args.skip_postprocess:
        print("후처리 건너뜀 (--skip-postprocess)")
        return

    build_final_video(composite_path, _read_entries(), SUBTITLES)


def _prevent_sleep(enable: bool) -> None:
    """Windows 절전/화면 잠금으로 인한 pyautogui 캡처 실패 및 브라우저 세션 재연결
    (긴 인덱싱 대기 중 실제로 발생: 캡처 실패 다수 + 페이지가 중복 마운트되어
    'strict mode violation: resolved to 2 elements' 크래시) 방지."""
    if sys.platform != "win32":
        return
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED if enable else 0)
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception as e:
        print(f"  경고: 절전 방지 설정 실패: {e}")


if __name__ == "__main__":
    _prevent_sleep(True)
    try:
        asyncio.run(main())
    finally:
        _prevent_sleep(False)
