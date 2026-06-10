import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import subprocess
import sys
import warnings
import ssl

import httpx
import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
ssl._create_default_https_context = ssl._create_unverified_context

load_dotenv()

# Streamlit Cloud 환경에서 playwright Chromium 자동 설치
@st.cache_resource(show_spinner=False)
def _install_playwright_browsers():
    chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if not os.path.exists(chrome_path):
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True,
        )

_install_playwright_browsers()

st.set_page_config(
    page_title="FIFA World Cup 2026 AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #4da6ff; }
    .section-header { font-size: 1.3rem; font-weight: 700; color: #ff6b6b; margin-top: 1rem; }

    /* 뉴스 카드 */
    .news-card {
        border: 1px solid #3a3a3a;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
        background: #1e2130;
        color: #e8e8e8;
    }
    .news-title { font-weight: 600; font-size: 0.97rem; color: #ffffff; line-height: 1.45; }
    .news-meta  { font-size: 0.78rem; color: #aaaaaa; margin-top: 6px; }
    .news-link  { font-size: 0.78rem; color: #4da6ff; text-decoration: none; }

    /* 경기 카드 */
    .match-card {
        border-left: 4px solid #ff6b6b;
        padding: 10px 14px;
        margin-bottom: 8px;
        background: #1e2130;
        border-radius: 4px;
    }
    .match-teams { font-weight: 700; font-size: 1.05rem; color: #ffffff; }
    .match-info  { font-size: 0.85rem; color: #aaaaaa; margin-top: 4px; }

    /* 순위 블록 */
    .rank-block {
        background: #1e2130;
        border: 1px solid #3a3a3a;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
        white-space: pre-wrap;
        font-size: 0.85rem;
        color: #e8e8e8;
        line-height: 1.6;
    }

    /* 승부예측 카드 */
    .predict-card {
        background: #1e2130;
        border: 1px solid #3a3a3a;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .predict-teams { font-size: 1.1rem; font-weight: 700; color: #ffffff; text-align: center; }
    .predict-meta  { font-size: 0.82rem; color: #aaaaaa; text-align: center; margin-top: 6px; }

    /* 영상 카드 */
    .video-card {
        border: 1px solid #3a3a3a;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 8px;
        background: #1e2130;
        color: #e8e8e8;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── 모듈 임포트 ────────────────────────────────────────────────────────────────
from src.rag import build_rag_pipeline, is_regulation_query
from src.crawler import (
    fetch_news, fetch_schedule, fetch_rankings,
    fetch_videos, fetch_predictions, fetch_live_context,
)
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ FIFA WC 2026 AI")
    st.divider()
    st.markdown("**챗봇 답변 방식**")
    st.markdown("- 🔵 FIFA 규정 질문 → RAG 벡터 검색")
    st.markdown("- 🔴 뉴스·일정·순위 → 네이버 스포츠 실시간")
    st.divider()
    st.markdown("**모델:** gpt-4o-mini")
    st.markdown("**임베딩:** paraphrase-multilingual-MiniLM")
    st.markdown("**데이터 출처:** 네이버 스포츠 WC2026")
    st.divider()
    if st.button("🔄 실시간 데이터 새로고침", use_container_width=True):
        st.cache_data.clear()
        st.toast("데이터를 새로고침했습니다!", icon="✅")

# ── RAG 파이프라인 초기화 ───────────────────────────────────────────────────
rag_chain, llm = build_rag_pipeline()

# ── 탭 레이아웃 ────────────────────────────────────────────────────────────────
tab_chat, tab_news, tab_schedule, tab_rank, tab_video, tab_predict = st.tabs([
    "🤖 AI 챗봇",
    "📰 뉴스",
    "📅 경기 일정",
    "🏆 순위",
    "🎬 영상",
    "🔮 승부 예측",
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1: AI 챗봇
# ═══════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown('<p class="main-title">FIFA World Cup 2026 AI 챗봇</p>', unsafe_allow_html=True)
    st.caption("FIFA 규정 질문은 📄 RAG로, 뉴스·일정·순위·예측 질문은 🌐 네이버 스포츠 실시간 데이터로 답변합니다.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── 1. 스크롤 가능한 메시지 영역 ──────────────────────────────────
    chat_area = st.container(height=560, border=False)
    with chat_area:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("en_query"):
                    with chat_area:
                        st.caption(f"🔍 검색어(EN): {msg['en_query']}")
                if msg.get("sources"):
                    with st.expander("📄 참고한 FIFA 규정 원문 보기"):
                        for i, doc in enumerate(msg["sources"], 1):
                            st.markdown(f"**[참고 {i}] 페이지 {doc.metadata.get('page', '?') + 1}**")
                            st.text(doc.page_content)
                            st.divider()
                if msg.get("live_context"):
                    with st.expander("🌐 참고한 실시간 데이터 보기"):
                        st.text(msg["live_context"])

    # ── 2. 입력창 — 컨테이너 바깥에 위치해 하단 고정처럼 표시 ──────────
    if user_input := st.chat_input("무엇이든 물어보세요! (규정·뉴스·일정·순위·예측)"):

        st.session_state.messages.append({"role": "user", "content": user_input})

        msg_data: dict = {"role": "assistant"}

        if is_regulation_query(user_input):
            with st.spinner("📄 FIFA 규정을 검색 중입니다..."):
                result = rag_chain.invoke(user_input)
            msg_data["content"]  = result["answer"]
            msg_data["en_query"] = result["en_query"]
            msg_data["sources"]  = result["context"]
        else:
            with st.spinner("🌐 네이버 스포츠 실시간 데이터 수집 중..."):
                live_ctx = fetch_live_context(user_input)
            user_msg = f"[실시간 데이터]\n{live_ctx}\n\n[질문]\n{user_input}"
            with st.spinner("💬 답변 생성 중..."):
                response = llm.invoke([
                    SystemMessage(content=(
                        "당신은 FIFA 2026 월드컵 전문 AI 어시스턴트입니다. "
                        "아래 [실시간 데이터]를 최대한 활용하여 사용자 질문에 한국어로 친절하게 답변하세요. "
                        "데이터가 부족하면 FIFA 2026 월드컵 일반 지식으로 보완하세요."
                    )),
                    HumanMessage(content=user_msg),
                ])
            msg_data["content"]      = response.content
            msg_data["live_context"] = live_ctx

        st.session_state.messages.append(msg_data)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════
# TAB 2: 뉴스
# ═══════════════════════════════════════════════════════════════════
with tab_news:
    st.markdown('<p class="section-header">📰 월드컵 최신 뉴스</p>', unsafe_allow_html=True)
    st.caption("출처: 네이버 스포츠 FIFA 월드컵 2026")

    with st.spinner("뉴스를 불러오는 중..."):
        news_items = fetch_news(12)

    if news_items and not news_items[0]["title"].startswith("[오류]"):
        cols = st.columns(2)
        for idx, item in enumerate(news_items):
            col = cols[idx % 2]
            with col:
                link_html = f'<a class="news-link" href="{item["url"]}" target="_blank">기사 보기 →</a>' if item.get("url") else ""
                st.markdown(
                    f'<div class="news-card">'
                    f'<p class="news-title">{item["title"]}</p>'
                    f'<p class="news-meta">{item.get("press","")} · {item.get("info","")}</p>'
                    f'{link_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.warning(f"뉴스를 불러오지 못했습니다: {news_items[0]['title'] if news_items else ''}")
        if st.button("🔄 뉴스 다시 불러오기"):
            st.cache_data.clear()
            st.rerun()


# ═══════════════════════════════════════════════════════════════════
# TAB 3: 경기 일정
# ═══════════════════════════════════════════════════════════════════
with tab_schedule:
    st.markdown('<p class="section-header">📅 경기 일정</p>', unsafe_allow_html=True)
    st.caption("출처: 네이버 스포츠 FIFA 월드컵 2026 일정")

    with st.spinner("경기 일정을 불러오는 중..."):
        schedule_items = fetch_schedule()

    if schedule_items and not schedule_items[0]["home"].startswith("[오류]"):
        for item in schedule_items:
            status_color = {"종료": "#888", "예정": "#1a3c6e", "진행중": "#c8102e"}.get(item["status"], "#444")
            link_html = f'<a href="{item["url"]}" target="_blank" style="text-decoration:none;color:inherit;">' if item.get("url") else ""
            link_close = "</a>" if item.get("url") else ""
            st.markdown(
                f'{link_html}'
                f'<div class="match-card">'
                f'<p class="match-teams">⚽ {item["home"]} vs {item["away"]}</p>'
                f'<p class="match-info">'
                f'  {item["group"]} &nbsp;|&nbsp; {item["time"]} &nbsp;|&nbsp; '
                f'  <span style="color:{status_color};font-weight:600;">{item["status"]}</span>'
                f'</p>'
                f'</div>'
                f'{link_close}',
                unsafe_allow_html=True,
            )
    else:
        st.warning("경기 일정을 불러오지 못했습니다.")

    st.divider()
    st.markdown("#### 📌 2026 FIFA 북중미 월드컵 주요 일정")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
| 단계 | 기간 |
|------|------|
| 조별 예선 | 2026.06.11 ~ 07.03 |
| 16강 | 2026.07.04 ~ 07.07 |
| 8강 | 2026.07.09 ~ 07.11 |
| 4강 | 2026.07.14 ~ 07.15 |
| 3·4위전 | 2026.07.18 |
| **결승** | **2026.07.19** |
""")
    with col2:
        st.markdown("""
| 개최국 | 주요 도시 |
|--------|----------|
| 미국 🇺🇸 | 뉴욕, LA, 달라스, 마이애미 등 11개 |
| 캐나다 🇨🇦 | 토론토, 밴쿠버 |
| 멕시코 🇲🇽 | 멕시코시티, 과달라하라 |
""")


# ═══════════════════════════════════════════════════════════════════
# TAB 4: 순위
# ═══════════════════════════════════════════════════════════════════
with tab_rank:
    st.markdown('<p class="section-header">🏆 조별 순위</p>', unsafe_allow_html=True)
    st.caption("출처: 네이버 스포츠 FIFA 월드컵 2026 기록")

    with st.spinner("순위를 불러오는 중..."):
        rank_items = fetch_rankings()

    if rank_items and not rank_items[0]["raw"].startswith("[오류]"):
        for item in rank_items:
            raw = item["raw"]
            if raw:
                st.markdown(
                    f'<div class="rank-block">{raw}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("순위 데이터를 불러오지 못했습니다. (조별 경기 시작 전이거나 일시적 오류)")

    st.divider()
    st.markdown("#### 🌍 2026 FIFA 월드컵 참가 그룹")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**유럽 (UEFA) · 16팀**\n- 🇫🇷 프랑스 · 🇪🇸 스페인\n- 🏴󠁧󠁢󠁥󠁮󠁧󠁿 잉글랜드 · 🇩🇪 독일\n- 🇵🇹 포르투갈 · 🇮🇹 이탈리아")
    with c2:
        st.markdown("**아시아 (AFC) · 8팀**\n- 🇯🇵 일본 · 🇰🇷 한국\n- 🇮🇷 이란 · 🇦🇺 호주\n- 🇸🇦 사우디 · 🇶🇦 카타르")
    with c3:
        st.markdown("**남미 (CONMEBOL) · 6팀**\n- 🇧🇷 브라질 · 🇦🇷 아르헨티나\n- 🇺🇾 우루과이 · 🇨🇴 콜롬비아")


# ═══════════════════════════════════════════════════════════════════
# TAB 5: 영상
# ═══════════════════════════════════════════════════════════════════
with tab_video:
    st.markdown('<p class="section-header">🎬 월드컵 영상 / 하이라이트</p>', unsafe_allow_html=True)
    st.caption("출처: 네이버 스포츠 FIFA 월드컵 2026 영상")

    with st.spinner("영상을 불러오는 중..."):
        video_items = fetch_videos(9)

    if video_items and not video_items[0]["title"].startswith("[오류]"):
        cols = st.columns(3)
        for idx, v in enumerate(video_items):
            col = cols[idx % 3]
            with col:
                link_html = f'<a style="color:#4da6ff;font-size:0.82rem;" href="{v["url"]}" target="_blank">▶ 영상 보러가기</a>' if v.get("url") else ""
                st.markdown(
                    f'<div class="video-card">'
                    f'<strong style="color:#ffffff;">{v["title"]}</strong><br/>{link_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("영상을 불러오지 못했습니다.")

    st.divider()
    st.markdown("#### 🔗 관련 영상 채널")
    st.markdown("- [네이버 스포츠 WC2026 영상](https://m.sports.naver.com/fifaworldcup2026/video)")
    st.markdown("- [네이버 치지직 월드컵 채널](https://chzzk.naver.com/8af5e1ebd972659e03dde5521047e231/videos?videoType=&sortType=LATEST&page=1)")


# ═══════════════════════════════════════════════════════════════════
# TAB 6: 승부 예측
# ═══════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown('<p class="section-header">🔮 승부 예측</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("#### 네이버 스포츠 실시간 예측 현황")
        st.caption("출처: 네이버 스포츠 FIFA 월드컵 2026 승부예측")
        with st.spinner("예측 데이터를 불러오는 중..."):
            pred_items = fetch_predictions()

        if pred_items and not pred_items[0]["home"].startswith("[오류]"):
            for pred in pred_items:
                status_color = {"종료": "#888", "예정": "#1a3c6e", "진행중": "#c8102e"}.get(pred["status"], "#444")
                st.markdown(
                    f'<div class="predict-card">'
                    f'<p class="predict-teams">🏠 {pred["home"]} &nbsp; vs &nbsp; {pred["away"]} ✈️</p>'
                    f'<p class="predict-meta">'
                    f'  <span style="color:{status_color};font-weight:600;">{pred["status"]}</span>'
                    f'  &nbsp;|&nbsp; {pred.get("deadline","")}'
                    f'  &nbsp;|&nbsp; {pred.get("participant","")}'
                    f'</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("예측 데이터를 불러오지 못했습니다.")

    with col_b:
        st.markdown("#### AI 승부 예측")
        st.caption("GPT-4o-mini가 최신 데이터를 바탕으로 경기 결과를 예측합니다.")
        home_team = st.text_input("🏠 홈 팀", placeholder="예: 대한민국")
        away_team = st.text_input("✈️ 원정 팀", placeholder="예: 체코")
        extra     = st.text_area("추가 정보 (선택)", height=70,
                                 placeholder="예: 조별 예선 1차전, 비 예보...")
        predict_btn = st.button("⚡ AI 예측 시작", type="primary", use_container_width=True)

        if predict_btn:
            if not home_team or not away_team:
                st.warning("두 팀 이름을 모두 입력해주세요.")
            else:
                with st.spinner(f"🔮 {home_team} vs {away_team} 분석 중..."):
                    # 실시간 예측 현황 컨텍스트
                    live_pred_ctx = ""
                    if pred_items:
                        lines = []
                        for pr in pred_items[:5]:
                            if not pr["home"].startswith("[오류]"):
                                lines.append(f"- {pr['home']} vs {pr['away']}: {pr['participant']} 참여")
                        live_pred_ctx = "\n".join(lines)

                    extra_line = f"추가 정보: {extra}" if extra else ""
                    pred_line  = f"네이버 스포츠 예측 현황:\n{live_pred_ctx}" if live_pred_ctx else ""
                    prompt = (
                        "당신은 FIFA 2026 월드컵 전문 분석가입니다.\n\n"
                        f"경기: {home_team} vs {away_team} (FIFA 2026 북중미 월드컵)\n"
                        f"{extra_line}\n"
                        f"{pred_line}\n\n"
                        "다음 형식으로 한국어 분석 리포트를 작성해 주세요:\n\n"
                        f"**1. 팀 분석**\n"
                        f"- {home_team}: 강점·약점·최근 폼\n"
                        f"- {away_team}: 강점·약점·최근 폼\n\n"
                        "**2. 핵심 변수 3가지**\n\n"
                        "**3. 예측 결과**\n"
                        "- 예상 스코어\n"
                        "- 승/무/패 확률 (합계 100%)\n\n"
                        "**4. 주목 선수** (각 팀 1명씩)"
                    )

                    resp = llm.invoke([
                        SystemMessage(content="당신은 FIFA 2026 월드컵 전문 분석가입니다."),
                        HumanMessage(content=prompt),
                    ])
                st.markdown(f"**⚽ {home_team} vs {away_team} 분석 결과**")
                st.markdown(resp.content)

    st.divider()
    st.markdown("#### 💡 빠른 AI 예측")
    quick_matches = [("대한민국", "체코"), ("브라질", "아르헨티나"), ("프랑스", "잉글랜드"), ("일본", "독일")]
    q_cols = st.columns(len(quick_matches))
    for i, (home, away) in enumerate(quick_matches):
        with q_cols[i]:
            if st.button(f"{home} vs {away}", key=f"q{i}", use_container_width=True):
                with st.spinner(f"🔮 {home} vs {away} 예측 중..."):
                    resp = llm.invoke([
                        SystemMessage(content="당신은 FIFA 2026 월드컵 전문 분석가입니다."),
                        HumanMessage(content=(
                            f"FIFA 2026 월드컵 {home} vs {away} 경기를 간략 분석하고 "
                            f"예상 스코어와 승/무/패 확률(합계 100%)을 한국어로 예측해 주세요."
                        )),
                    ])
                st.markdown(f"**{home} vs {away}**")
                st.markdown(resp.content)
