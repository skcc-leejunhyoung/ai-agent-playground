# model.py

import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

##########

AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_API_KEY = os.getenv("AOAI_API_KEY")
AOAI_DEPLOY_GPT4O = os.getenv("AOAI_DEPLOY_GPT4O")  # ex) gpt-4o
AOAI_DEPLOY_GPT4O_MINI = os.getenv("AOAI_DEPLOY_GPT4O_MINI")  # ex) gpt-4o-mini
AOAI_API_VERSION = "2024-02-15-preview"  # Azure OpenAI 최신 API 버전

client = AzureOpenAI(
    azure_endpoint=AOAI_ENDPOINT,
    api_key=AOAI_API_KEY,
    api_version=AOAI_API_VERSION,
)

##########


def stream_chat(
    user_prompt: str, system_prompt: str = "", use_gpt4o_mini: bool = False
):
    """
    유저 프롬프트와 시스템 프롬프트를 받아서 Azure OpenAI GPT-4o로 스트리밍 응답을 반환합니다.
    """
    deployment_name = AOAI_DEPLOY_GPT4O_MINI if use_gpt4o_mini else AOAI_DEPLOY_GPT4O

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            stream=True,
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

    except Exception as e:
        yield f"[ERROR] {str(e)}"
