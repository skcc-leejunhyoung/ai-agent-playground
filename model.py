# model.py

import os
import openai
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# Azure OpenAI 환경 변수
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_API_KEY = os.getenv("AOAI_API_KEY")
AOAI_DEPLOY_GPT4O = os.getenv("AOAI_DEPLOY_GPT4O")
AOAI_DEPLOY_GPT4O_MINI = os.getenv("AOAI_DEPLOY_GPT4O_MINI")

# OpenAI 클라이언트 초기화
openai.api_type = "azure"
openai.api_base = AOAI_ENDPOINT
openai.api_key = AOAI_API_KEY
openai.api_version = "2024-02-15-preview"


def stream_chat(
    user_prompt: str, system_prompt: str = "", use_gpt4o_mini: bool = False
):
    """
    유저 프롬프트와 시스템 프롬프트를 받아서 OpenAI GPT-4o 또는 GPT-4o-mini로 스트리밍 응답을 반환합니다.
    """
    deployment_name = AOAI_DEPLOY_GPT4O_MINI if use_gpt4o_mini else AOAI_DEPLOY_GPT4O

    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_prompt})

    try:
        # Chat Completion 요청
        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            stream=True,  # 스트리밍 활성화
        )

        # 스트리밍 응답 생성기 반환
        for chunk in response:
            if "choices" in chunk:
                delta = chunk["choices"][0]["delta"]
                content = delta.get("content")
                if content:
                    yield content  # 스트림 응답 보내기

    except Exception as e:
        yield f"[ERROR] {str(e)}"
