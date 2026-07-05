# ⚡ Power Market AI Assistant

전력거래 실무 담당자를 위한 AI 어시스턴트.
전력시장운영규칙 기반 규정 Q&A와 공개 데이터 기반 SMP 방향성 추정을 하나의 챗봇으로 제공합니다.

---

## 🎯 주요 기능

| 기능 | 설명 | 예시 질문 |
|------|------|---------|
| 규정 Q&A | 전력시장운영규칙 기반 질의응답 | "제2.4.1조가 뭔가요?" |
| 개정 비교 | 규정 개정 전후 조문 비교 | "이번 개정에서 뭐가 바뀌었어?" |
| 이력 조회 | 조문별 전체 개정 이력 | "이 조항이 몇 번 개정됐어?" |
| SMP 분석 | 데이터 기반 방향성 추정 | "오늘 SMP 올라갈까?" |
| Agent Planning | LLM이 실행 계획 수립 후 병렬/순차 실행 | "어제 SMP 급등 이유를 데이터랑 규정으로 설명해줘" |

---

## 🔧 기술 스택

| 구분 | 기술 |
|------|------|
| LLM | Azure OpenAI GPT-5.4 |
| Agent Framework | LangGraph + Agent Planning |
| Vector DB | **Qdrant** (로컬/클라우드 자동 전환) |
| Search | **Hybrid Search** (벡터 60% + BM25 40%) |
| 임베딩 | text-embedding-3-large (3072차원) |
| PDF 파싱 | pymupdf4llm |
| 데이터 수집 | 공공데이터포털 API |
| UI | Streamlit (다크 테마) |
| 시각화 | Plotly |

---

## 🚀 로컬 설치 방법

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 환경변수 설정
cp .env.example .env
# .env 파일 편집 후 API 키 입력

# 3. 앱 실행
streamlit run main.py
```

---

## ⚙️ 환경변수 설정 (.env.example 참고)

```env
# Azure OpenAI (필수)
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://skax.ai-talentlab.com

# 공공데이터포털 (필수)
PUBLIC_DATA_API_KEY=your_key

# Qdrant: URL 없으면 로컬, 있으면 클라우드 자동 전환
QDRANT_URL=
QDRANT_API_KEY=
```

---

## ☁️ Streamlit Cloud 배포 방법

1. GitHub에 소스코드 push (`.env`, `qdrant_db/` 는 .gitignore로 제외됨)
2. [share.streamlit.io](https://share.streamlit.io) 접속 → GitHub 연동
3. `main.py` 선택 후 Secrets에 환경변수 입력:
   ```
   AZURE_OPENAI_API_KEY = "..."
   QDRANT_URL = "https://xxxx.qdrant.io"
   QDRANT_API_KEY = "..."
   ```
4. Deploy!

> **Qdrant Cloud 사용 시**: [cloud.qdrant.io](https://cloud.qdrant.io) 에서 Free tier 클러스터 생성 후 URL/API Key를 Secrets에 입력하면, 로컬에서 인덱싱한 데이터를 클라우드에서도 영구 사용 가능합니다.

---

## 🏗️ 프로젝트 구조

```
power-market-ai/
├── main.py                   # Streamlit 진입점 (다크 테마 + 실시간 지표)
├── config/settings.py        # 환경변수 관리 (Qdrant 포함)
├── infrastructure/           # LLM 클라이언트, 프롬프트
├── domain/
│   ├── rag/
│   │   ├── vector_store.py   # Qdrant (로컬/클라우드 자동 전환)
│   │   └── retriever.py      # Hybrid Search (벡터 + BM25)
│   └── analysis/             # SMP 분석, 차트 (다크 테마)
├── application/
│   └── graph_router.py       # LangGraph + Agent Planning
└── presentation/             # Streamlit UI
```

---

## 🛡️ 핵심 원칙

- 근거 없는 확정 답변 절대 금지
- 모든 답변에 출처(조문번호+페이지+인용) 또는 데이터 근거 필수
- SMP 방향성은 Python 고정 로직으로만 계산 (LLM 판단 없음)
- 모든 답변 하단에 면책 문구 자동 출력

---

**Version**: 2.0.0 (Qdrant + Hybrid Search + Agent Planning + 모던 UI)
