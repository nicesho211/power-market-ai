"""
전역 로깅 설정 모듈
앱 전체에서 사용할 콘솔 로그 포맷과 레벨을 초기화한다.
"""
import logging
import sys
from pathlib import Path

# Streamlit을 백그라운드 프로세스로 띄운 경우 콘솔 출력이 사용자에게 보이지 않으므로,
# 같은 로그를 파일로도 남겨 텍스트 에디터/tail로 직접 확인할 수 있게 한다.
LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "app.log"


def setup_logging():
    """
    전체 앱에서 사용할 로깅 설정.
    콘솔 + logs/app.log 파일에 [시간] [모듈명] 메시지 형식으로 출력.
    """
    log_format = "[%(asctime)s] [%(name)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    if sys.platform == "win32":
        # Windows 콘솔 기본 cp949 인코딩에서 로그 메시지의 이모지(⚡✅⚠️ 등) 출력 시
        # UnicodeEncodeError로 로깅 자체가 깨지는 것을 방지
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
            sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 루트 로거 설정 (콘솔 핸들러)
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        stream=sys.stdout,   # 콘솔 출력
        force=True           # 기존 핸들러 초기화
    )

    # 파일 핸들러 추가 (콘솔 접근이 불가능한 환경에서도 로그 확인 가능)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logging.getLogger().addHandler(file_handler)

    # 외부 라이브러리 로그 레벨 조정 (너무 많이 찍히면 노이즈)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.getLogger("APP").info("⚡ Power Market AI Assistant 로깅 시작")
