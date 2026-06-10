import os
import warnings
import ssl

import httpx
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda

warnings.filterwarnings("ignore")
ssl._create_default_https_context = ssl._create_unverified_context

PDF_PATH   = "./FWC26_regulations_EN.pdf"
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE    = 1000   # 800 → 1000 (조항 전체 포함 확률 향상)
CHUNK_OVERLAP = 150    # 100 → 150 (경계 손실 감소)
RETRIEVER_K   = 7      # 4 → 7 (더 많은 후보)
FETCH_K       = 25     # MMR 후보풀

# ── 규정 질의 키워드 ──────────────────────────────────────────────────────────
REGULATION_KEYWORDS = [
    # 일반 규정 용어
    "규정", "룰", "rule", "article", "조항", "제도", "조문", "법규",
    # 심판/경기진행
    "심판", "주심", "부심", "var", "비디오판독", "오프사이드", "반칙",
    "파울", "핸드볼", "페널티", "프리킥", "코너킥", "스로인", "골킥",
    "킥오프", "드롭볼",
    # 카드/징계
    "레드카드", "옐로우카드", "경고", "퇴장", "징계", "제재", "정지",
    # 선수/팀 구성
    "선수등록", "출전자격", "교체", "선수교체", "로스터", "스쿼드",
    "몇 명", "인원", "골키퍼", "유니폼", "킷",
    # 경기 방식
    "경기시간", "경기 시간", "추가시간", "연장전", "승부차기", "페널티킥",
    "득점", "동점", "승자", "승리조건", "조별", "토너먼트", "결선",
    "16강", "8강", "4강", "결승", "3위",
    # 대회 구조
    "조편성", "시드", "추첨", "참가국", "참가팀", "개최", "개최국",
    "상금", "포상", "트로피",
    # 도핑/의료
    "도핑", "약물", "의료", "부상",
    # FIFA 관련 용어
    "피파규정", "fifa regulation", "피파", "fifa",
    # 등록/행정
    "등록", "자격", "국적", "귀화", "이적",
]


def is_regulation_query(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in REGULATION_KEYWORDS)


@st.cache_resource(show_spinner=False)
def build_rag_pipeline():
    loader = PyPDFLoader(PDF_PATH)
    pages  = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # DB가 없거나 청크 설정이 바뀐 경우 재구축
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embedding_model,
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=CHROMA_DIR,
        )

    # 로컬: .env / Streamlit Cloud: secrets 둘 다 지원
    api_key = (
        st.secrets.get("OPENAI_API_KEY")
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets
        else os.getenv("OPENAI_API_KEY")
    )
    http_client = httpx.Client(verify=False)
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=api_key,
        http_client=http_client,
        temperature=0,
    )

    # ── 답변 프롬프트 ──────────────────────────────────────────────────────────
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "당신은 FIFA World Cup 26 규정(Regulations) 전문 분석가입니다.\n"
            "제공된 [Context]를 최대한 활용해 [Question]에 한국어로 상세히 답변하세요.\n\n"
            "답변 원칙:\n"
            "1. Context에서 관련 내용을 찾아 가능한 한 구체적으로 설명하세요.\n"
            "2. 근거가 되는 Article 번호(예: Article 12.3)를 반드시 함께 인용하세요.\n"
            "3. Context에 직접적인 내용이 없더라도 관련 조항을 참고해 최선의 답변을 제공하세요.\n"
            "4. 정말 관련 내용이 전혀 없을 때만 '제공된 FIFA 규정에서 확인할 수 없습니다'라고 답하세요.\n"
            "5. 답변은 명확하고 읽기 쉽게 구조화하세요(필요시 번호 목록 활용).\n\n"
            "[Context]\n{context}\n\n"
            "[Question]\n{question}"
        ),
    )

    # ── 다중 쿼리 번역 프롬프트 ───────────────────────────────────────────────
    multi_translate_prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "You are a FIFA regulations search expert. "
            "Given the user question below, generate exactly 2 distinct English search queries "
            "to retrieve relevant passages from the FIFA World Cup 26 Regulations PDF.\n\n"
            "Rules:\n"
            "- Query 1: concise and direct (key terms only)\n"
            "- Query 2: expanded with synonyms and related FIFA terminology\n"
            "- Use official FIFA terms: Article, Match Officials, VAR, Extra Time, "
            "Penalty Kick, Squad, Substitution, Disciplinary, etc.\n"
            "- Output ONLY two lines, no numbering, no explanation.\n\n"
            "User question: {question}\n"
            "Query 1:\n"
            "Query 2:"
        ),
    )
    multi_translate_chain = multi_translate_prompt | llm | StrOutputParser()

    def translate_to_queries(question: str) -> list[str]:
        """한국어 질문 → 영어 검색 쿼리 2개 반환"""
        has_korean = any("가" <= ch <= "힣" for ch in question)
        if not has_korean:
            return [question, question]
        raw = multi_translate_chain.invoke({"question": question}).strip()
        lines = [ln.strip().strip('"').strip("'") for ln in raw.splitlines() if ln.strip()]
        if len(lines) >= 2:
            return lines[:2]
        return [lines[0], lines[0]] if lines else [question, question]

    # ── MMR 리트리버 (다양성 + 관련성 동시 확보) ──────────────────────────────
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_K, "fetch_k": FETCH_K, "lambda_mult": 0.6},
    )

    def multi_retrieve(queries: list[str]):
        """두 쿼리로 검색 후 중복 제거, 최대 RETRIEVER_K+2개 반환"""
        seen, docs = set(), []
        for q in queries:
            for doc in retriever.invoke(q):
                key = doc.page_content[:80]
                if key not in seen:
                    seen.add(key)
                    docs.append(doc)
        return docs[: RETRIEVER_K + 2]

    def format_docs(docs):
        parts = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "?")
            parts.append(f"[청크 {i} | p.{page + 1}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    # ── 체인 구성 ─────────────────────────────────────────────────────────────
    rag_chain = (
        RunnableParallel(
            question=RunnablePassthrough(),
            queries=RunnableLambda(translate_to_queries),
        )
        .assign(
            en_query=lambda x: x["queries"][0],          # UI 표시용 대표 쿼리
            context=lambda x: multi_retrieve(x["queries"]),
        )
        .assign(
            answer=(
                lambda x: prompt_template.invoke(
                    {"context": format_docs(x["context"]), "question": x["question"]}
                )
            )
            | llm
            | StrOutputParser()
        )
    )

    return rag_chain, llm
