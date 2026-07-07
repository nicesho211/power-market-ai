"""
Azure OpenAI LLM 및 임베딩 클라이언트 모듈
AzureChatOpenAI와 AzureOpenAIEmbeddings를 싱글톤으로 관리합니다.
"""

import logging
from functools import lru_cache
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from config.settings import get_settings

logger = logging.getLogger("LLM")


@lru_cache()
def get_llm() -> AzureChatOpenAI:
    settings = get_settings()
    llm = AzureChatOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_deployment=settings.OPENAI_MODEL,
        temperature=0.1
    )
    logger.info(f"[LLM] AzureChatOpenAI 초기화 완료 | model={settings.OPENAI_MODEL}")
    return llm


@lru_cache()
def get_llm_for_planning() -> AzureChatOpenAI:
    """Planning 전용 LLM: 보다 창의적 응답을 위해 온도를 높여 인스턴스화합니다."""
    settings = get_settings()
    return AzureChatOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_deployment=settings.OPENAI_MODEL,
        temperature=0.7
    )


@lru_cache()
def get_embeddings() -> AzureOpenAIEmbeddings:
    settings = get_settings()
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_deployment=settings.EMBEDDING_MODEL
    )
    logger.info(f"[EMBED] AzureOpenAIEmbeddings 초기화 완료 | model={settings.EMBEDDING_MODEL}")
    return embeddings


def generate_rag_answer(query: str, context: str, history: list) -> dict:
    """RAG 답변 생성 — Structured Output으로 JSON 반환"""
    from infrastructure.prompt_manager import get_rag_prompt
    llm = get_llm()
    prompt = get_rag_prompt(query, context, history)
    response = llm.invoke(prompt)
    return {"answer": response.content}


def generate_analysis_answer(query: str, analysis_data: dict) -> dict:
    """분석 답변 생성 — 스코어링 결과를 받아 자연어 요약"""
    from infrastructure.prompt_manager import get_analysis_prompt
    llm = get_llm()
    prompt = get_analysis_prompt(query, analysis_data)
    response = llm.invoke(prompt)
    return {"answer": response.content}


def generate_diff_summary(diff_result: dict) -> str:
    """개정 전/후 변경 요지 요약"""
    llm = get_llm()
    prompt = f"""
아래 전력시장운영규칙 조문 변경 내용을 분석해서 변경 요지와 업무 영향을 요약해줘.

변경 전: {diff_result.get('변경전', '')}
변경 후: {diff_result.get('변경후', '')}

형식:
- 변경 요지: (1~2줄)
- 업무 영향: (1~2줄)
"""
    response = llm.invoke(prompt)
    return response.content


def generate_history_summary(history_data: list) -> str:
    """전체 개정 이력 요약"""
    llm = get_llm()
    prompt = f"아래 개정 이력을 바탕으로 주요 변경 흐름을 2~3줄로 요약해줘.\n\n{history_data}"
    response = llm.invoke(prompt)
    return response.content
