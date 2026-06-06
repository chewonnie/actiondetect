"""Streamlit 대시보드를 헤드리스 브라우저로 렌더링해 스크린샷 저장.
streamlit 서버는 외부에서 띄워 두고 URL만 넘긴다.
실행: python scripts/shot_dashboard.py <url> <out_png>
"""
import sys
import time

from playwright.sync_api import sync_playwright

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8599/"
out = sys.argv[2] if len(sys.argv) > 2 else "runs/baseline12/dashboard_screenshot.png"

with sync_playwright() as p:
    b = p.chromium.launch(args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1600, "height": 1200})
    pg.goto(url, wait_until="networkidle", timeout=90000)
    # Streamlit이 스크립트(YoloDetector 로드 포함)를 다 돌릴 때까지 대기:
    # 제목 + 마지막 패널 헤더가 보일 때까지 + 차트 렌더 여유.
    try:
        pg.wait_for_selector("text=고령자 일상행동 모니터링 대시보드", timeout=60000)
        pg.wait_for_selector("text=알림", timeout=60000)
    except Exception as e:
        print("WARN selector wait:", e)
    time.sleep(8)  # plotly 차트 그려질 여유
    pg.screenshot(path=out, full_page=True)
    b.close()
print("SAVED", out)
