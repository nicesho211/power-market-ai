"""
Streamlit Community Cloud 로그인 세션 저장 스크립트

demo_script.py가 "Manage app" 로그 패널(인증 필요)을 열람할 수 있도록,
사용자가 직접 로그인한 브라우저 세션(쿠키)을 파일로 저장한다.

사용법:
    python demo_login.py

브라우저 창이 뜨면 GitHub 계정으로 Streamlit Community Cloud에 로그인한 뒤,
데모 대상 앱 페이지가 정상적으로 보이면 터미널로 돌아와 Enter를 누르면 된다.
(대시보드 화면 텍스트를 자동 감지하는 대신, 확실하게 사용자가 직접 신호를 준다.)
"""
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

TARGET_APP_URL = "https://power-market-ai-mjzl4hfvu2y4ijbbenhyq7.streamlit.app"
STATE_PATH = Path(__file__).parent / "demo_output" / "auth_state.json"


async def main() -> None:
    STATE_PATH.parent.mkdir(exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1280, "height": 900})
        page = await context.new_page()
        await page.goto(TARGET_APP_URL, wait_until="domcontentloaded")

        print("=" * 60)
        print("브라우저 창에서 GitHub 계정으로 Streamlit Community Cloud에 로그인해주세요.")
        print("로그인 후 이 앱 페이지 우측 하단(또는 하단)에 'Manage app' 버튼이")
        print("보이는지 확인하세요. (로그인 안 해도 앱 자체는 보일 수 있습니다.)")
        print()
        print(">>> 로그인을 마쳤으면 이 터미널로 돌아와서 Enter 키를 눌러주세요. <<<")
        print("=" * 60)

        await asyncio.to_thread(input, "")

        await context.storage_state(path=str(STATE_PATH))
        print(f"✅ 로그인 세션 저장 완료 → {STATE_PATH}")
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
