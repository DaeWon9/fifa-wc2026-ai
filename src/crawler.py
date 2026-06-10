import os
import streamlit as st
from typing import List, Dict
from playwright.sync_api import sync_playwright

CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
MOBILE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/16.0 Mobile/15E148 Safari/604.1"
)
NAVER_BASE = "https://m.sports.naver.com"

NEWS_URL     = "https://m.sports.naver.com/fifaworldcup2026/news"
SCHEDULE_URL = "https://m.sports.naver.com/fifaworldcup2026/schedule"
RECORD_URL   = "https://m.sports.naver.com/fifaworldcup2026/record?seasonCode=3F9X"
VIDEO_URL    = "https://m.sports.naver.com/fifaworldcup2026/video"
PREDICT_URL  = "https://m.sports.naver.com/fifaworldcup2026/predict"


def _open_page(p, url: str):
    """공용 브라우저/페이지 생성 (sync_playwright 블록 안에서 호출)"""
    launch_args = ["--no-sandbox", "--disable-dev-shm-usage", "--ignore-certificate-errors"]
    # 로컬(Mac): 시스템 Chrome 우선 사용 / 클라우드: playwright 번들 Chromium 사용
    if os.path.exists(CHROME_PATH):
        browser = p.chromium.launch(
            headless=True,
            executable_path=CHROME_PATH,
            args=launch_args,
        )
    else:
        browser = p.chromium.launch(headless=True, args=launch_args)
    ctx = browser.new_context(
        user_agent=MOBILE_UA,
        locale="ko-KR",
        timezone_id="Asia/Seoul",
        viewport={"width": 390, "height": 844},
        ignore_https_errors=True,
    )
    page = ctx.new_page()
    page.goto(url, wait_until="domcontentloaded", timeout=20000)
    page.wait_for_timeout(3000)
    return browser, page


def _abs(href: str) -> str:
    if not href:
        return ""
    return href if href.startswith("http") else NAVER_BASE + href


# ─── 뉴스 ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=180, show_spinner=False)
def fetch_news(max_items: int = 10) -> List[Dict]:
    results: List[Dict] = []
    try:
        with sync_playwright() as p:
            browser, page = _open_page(p, NEWS_URL)
            items = page.query_selector_all('[class*="NewsItem_news_item"]')
            for el in items[:max_items]:
                title_el = el.query_selector('[class*="NewsItem_title"]')
                link_el  = el.query_selector('[class*="NewsItem_link_news"], a')
                info_el  = el.query_selector('[class*="NewsItem_sub_info"]')
                press_el = el.query_selector('[class*="NewsItem_press"]')
                img_el   = el.query_selector('img')
                title = title_el.inner_text().strip() if title_el else ""
                href  = _abs(link_el.get_attribute("href") or "") if link_el else ""
                info  = info_el.inner_text().strip() if info_el else ""
                press = press_el.inner_text().strip() if press_el else ""
                img   = (img_el.get_attribute("src") or "") if img_el else ""
                if title:
                    results.append({"title": title, "url": href, "info": info,
                                    "press": press, "img": img})
            browser.close()
    except Exception as e:
        results.append({"title": f"[오류] {e}", "url": "", "info": "", "press": "", "img": ""})
    return results


# ─── 경기 일정 ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=120, show_spinner=False)
def fetch_schedule() -> List[Dict]:
    results: List[Dict] = []
    try:
        with sync_playwright() as p:
            browser, page = _open_page(p, SCHEDULE_URL)
            items = page.query_selector_all('[class*="MatchBox_match_item"]')
            for el in items:
                teams    = el.query_selector_all('[class*="MatchBoxHeadToHeadArea_team_name"]')
                time_el  = el.query_selector('[class*="MatchBox_time"]')
                status   = el.query_selector('[class*="MatchBox_status"]')
                add_info = el.query_selector('[class*="MatchBox_add_info"]')
                link_el  = el.query_selector('a[href]')
                home = teams[0].inner_text().strip() if len(teams) > 0 else ""
                away = teams[1].inner_text().strip() if len(teams) > 1 else ""
                raw_time   = time_el.inner_text().strip() if time_el else ""
                # "경기 시간\n04:00" → "04:00" 로 정리
                time_txt   = raw_time.split("\n")[-1].strip() if "\n" in raw_time else raw_time
                status_txt = status.inner_text().strip() if status else ""
                group_txt  = add_info.inner_text().strip() if add_info else ""
                match_url  = _abs(link_el.get_attribute("href") or "") if link_el else ""
                if home and away:
                    results.append({
                        "home": home, "away": away,
                        "time": time_txt, "status": status_txt,
                        "group": group_txt, "url": match_url,
                    })
            browser.close()
    except Exception as e:
        results.append({"home": f"[오류] {e}", "away": "", "time": "", "status": "", "group": "", "url": ""})
    return results


# ─── 순위 ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=180, show_spinner=False)
def fetch_rankings() -> List[Dict]:
    results: List[Dict] = []
    try:
        with sync_playwright() as p:
            browser, page = _open_page(p, RECORD_URL)
            groups = page.query_selector_all('[class*="TeamRanking_group"]')
            for g in groups:
                raw = g.inner_text().strip()
                if raw and len(raw) > 5 and not any(skip in raw for skip in ["안내", "확인하세요"]):
                    results.append({"raw": raw})
            browser.close()
    except Exception as e:
        results.append({"raw": f"[오류] {e}"})
    return results


# ─── 영상 ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_videos(max_items: int = 9) -> List[Dict]:
    results: List[Dict] = []
    try:
        with sync_playwright() as p:
            browser, page = _open_page(p, VIDEO_URL)
            items = page.query_selector_all(
                '[class*="CountryCollection_list"] li, '
                '[class*="Ranking_list"] li, '
                '[class*="video"] li'
            )
            for el in items[:max_items]:
                title_el = el.query_selector('[class*="title"], strong, em, p, span')
                thumb_el = el.query_selector('img')
                title = title_el.inner_text().strip() if title_el else el.inner_text().strip()
                thumb = (thumb_el.get_attribute("src") or "") if thumb_el else ""
                # 영상 링크는 onClick 기반이므로 naver sports 영상 페이지로 안내
                if title and len(title) > 2:
                    results.append({"title": title, "thumb": thumb, "url": VIDEO_URL})
            browser.close()
    except Exception as e:
        results.append({"title": f"[오류] {e}", "thumb": "", "url": ""})
    return results


# ─── 승부예측 ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=120, show_spinner=False)
def fetch_predictions() -> List[Dict]:
    results: List[Dict] = []
    seen = set()
    try:
        with sync_playwright() as p:
            browser, page = _open_page(p, PREDICT_URL)
            # Match_match_item 안에 MatchBox_match_box가 있는 구조
            # outermost match_item 만 선택하기 위해 date 정보 포함 여부로 필터
            items = page.query_selector_all('[class*="Match_match_item"]')
            for el in items:
                full_text = el.inner_text().strip()
                # 날짜/상태 정보 포함한 카드만 (상위 컨테이너)
                if "예정" not in full_text and "종료" not in full_text and "진행" not in full_text:
                    continue
                home_el = el.query_selector('[class*="MatchBox_home"] [class*="MatchBox_name"]')
                away_el = el.query_selector('[class*="MatchBox_away"] [class*="MatchBox_name"]')
                draw_el = el.query_selector('[class*="MatchBox_draw"]')
                home = home_el.inner_text().strip() if home_el else ""
                away = away_el.inner_text().strip() if away_el else ""
                if not home or not away:
                    continue
                key = f"{home}_{away}"
                if key in seen:
                    continue
                seen.add(key)

                # 참여자 수 파싱
                participant = ""
                if "명 참여" in full_text:
                    for part in full_text.split("\n"):
                        if "명 참여" in part:
                            participant = part.strip()
                            break

                # 상태 파싱
                status = "예정"
                for s in ["종료", "진행중", "예정"]:
                    if s in full_text:
                        status = s
                        break

                # 마감 시간
                deadline = ""
                for part in full_text.split("\n"):
                    if "후 마감" in part:
                        deadline = part.strip()
                        break

                results.append({
                    "home": home,
                    "away": away,
                    "draw_text": draw_el.inner_text().strip() if draw_el else "무승부",
                    "status": status,
                    "deadline": deadline,
                    "participant": participant,
                })
    except Exception as e:
        results.append({"home": f"[오류] {e}", "away": "", "draw_text": "", "status": "", "deadline": "", "participant": ""})
    return results


# ─── 챗봇용 컨텍스트 생성 ──────────────────────────────────────────────────────

def fetch_live_context(question: str) -> str:
    """질문 내용에 따라 적합한 실시간 데이터를 텍스트로 반환"""
    q = question.lower()
    lines = []

    want_news     = any(k in q for k in ["뉴스", "소식", "최신", "기사"])
    want_schedule = any(k in q for k in ["일정", "경기", "스케줄", "시간", "언제", "몇 시"])
    want_rank     = any(k in q for k in ["순위", "랭킹", "성적", "조별", "승점"])
    want_predict  = any(k in q for k in ["예측", "승부", "예상", "이길", "질"])
    want_video    = any(k in q for k in ["영상", "하이라이트", "동영상", "vod"])

    # 키워드 미매칭이면 뉴스+일정 기본 제공
    if not any([want_news, want_schedule, want_rank, want_predict, want_video]):
        want_news = want_schedule = True

    if want_news:
        news = fetch_news(6)
        lines.append("=== 📰 최신 월드컵 뉴스 ===")
        for n in news:
            if not n["title"].startswith("[오류]"):
                lines.append(f"- [{n['press']}] {n['title']} ({n['info']})")

    if want_schedule:
        sched = fetch_schedule()
        lines.append("\n=== 📅 경기 일정 ===")
        for s in sched:
            if not s["home"].startswith("[오류]"):
                lines.append(f"- {s['group']} | {s['home']} vs {s['away']} | {s['time']} | {s['status']}")

    if want_rank:
        ranks = fetch_rankings()
        lines.append("\n=== 🏆 순위 ===")
        for r in ranks[:3]:
            lines.append(r["raw"][:300])

    if want_predict:
        preds = fetch_predictions()
        lines.append("\n=== 🔮 승부예측 현황 ===")
        for pred in preds[:5]:
            if not pred["home"].startswith("[오류]"):
                lines.append(
                    f"- {pred['home']} vs {pred['away']} "
                    f"({pred['status']}) | {pred['participant']} | {pred['deadline']}"
                )

    if want_video:
        vids = fetch_videos(5)
        lines.append("\n=== 🎬 영상 목록 ===")
        for v in vids:
            if not v["title"].startswith("[오류]"):
                lines.append(f"- {v['title']}")

    return "\n".join(lines)
