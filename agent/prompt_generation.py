from langgraph.graph import StateGraph
import os
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


##########


load_dotenv()

AOAI_API_KEY = os.getenv("AOAI_API_KEY")
AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_DEPLOY_GPT4O = os.getenv("AOAI_DEPLOY_GPT4O")
AOAI_API_VERSION = "2024-02-15-preview"

llm_gpt4o = AzureChatOpenAI(
    openai_api_key=AOAI_API_KEY,
    azure_endpoint=AOAI_ENDPOINT,
    api_version=AOAI_API_VERSION,
    deployment_name=AOAI_DEPLOY_GPT4O,
    temperature=0.7,
    max_tokens=1000,
    model_kwargs={"response_format": {"type": "json_object"}},
)


##########


class PromptOutput(BaseModel):
    system_prompt: str = Field(
        ..., description="AI 응답을 위한 최적화된 시스템 프롬프트입니다."
    )
    user_prompt_template: str = Field(
        ..., description="사용자 입력을 포함할 수 있는 사용자 프롬프트 템플릿입니다."
    )
    reasoning: str = Field(
        ...,
        description="이 프롬프트들이 사용자 의도를 어떻게 효과적으로 반영하는지에 대한 설명입니다. markdown 형식으로 작성해주세요.",
    )


prompt_parser = PydanticOutputParser(pydantic_object=PromptOutput)


##########


def run_prompt_generation_agent(user_intention, user_sys_params=None):
    """
    사용자 의도(user_intention)와 선택적 시스템 파라미터를 받아
    최적화된 프롬프트를 생성하는 에이전트를 실행합니다.

    Args:
        user_intention (str): 사용자의 목적과 원하는 AI 응답 유형에 대한 설명
        user_sys_params (str, optional): 생성할 프롬프트 유형 ("user" 또는 "system")
                                        기본값은 "system"

    Returns:
        dict: 최종 생성된 프롬프트와 분석 결과를 포함하는 상태 딕셔너리
    """

    if user_sys_params is None or user_sys_params not in ["user", "system"]:
        user_sys_params = "system"

    state = {
        "user_intention": user_intention,
        "prompt_type": user_sys_params,
        "prompt_draft": None,
        "prompt_refined": None,
        "final_prompt": None,
    }

    def analyze_intention(state):
        """사용자 의도를 분석하고 초기 프롬프트 초안을 작성합니다."""
        user_intention = state["user_intention"]
        prompt_type = state["prompt_type"]

        type_guidance = ""
        if prompt_type == "user":
            type_guidance = "사용자 프롬프트를 생성해주세요. 사용자 프롬프트는 AI에게 직접 전달되는 질문이나 지시사항입니다."
        else:  # system
            type_guidance = "시스템 프롬프트를 생성해주세요. 시스템 프롬프트는 AI의 역할과 동작을 정의하는 지침입니다."

        system_message = f"""당신은 전문 프롬프트 엔지니어입니다.
사용자의 의도와 요구사항을 분석하여 효과적인 AI 프롬프트를 설계해주세요.
{type_guidance}
사용자 의도를 이해하고 초기 프롬프트 초안을 작성하세요.
다음 JSON 형식으로 응답해주세요:
{{
  "system_prompt": "초기 시스템 프롬프트 초안",
  "user_prompt_template": "초기 사용자 프롬프트 템플릿",
  "reasoning": "이 프롬프트가 사용자 의도에 어떻게 부합하는지에 대한 초기 분석"
}}"""

        prompt_input = (
            f"다음 사용자 의도를 분석하고 초기 프롬프트 초안을 작성해주세요:\n\n"
            f"사용자 의도: {user_intention}\n\n"
            f"프롬프트 타입: {prompt_type} 프롬프트\n\n"
            "JSON 형식으로 초기 시스템 프롬프트와 사용자 프롬프트 템플릿을 제안해주세요."
        )

        try:
            response = llm_gpt4o.invoke(
                [
                    SystemMessage(content=system_message),
                    HumanMessage(content=prompt_input),
                ]
            )

            parsed_output = PromptOutput.model_validate_json(response.content)

            state["prompt_draft"] = {
                "system_prompt": parsed_output.system_prompt.strip(),
                "user_prompt_template": parsed_output.user_prompt_template.strip(),
                "reasoning": parsed_output.reasoning.strip(),
            }

        except Exception as e:
            print(f"[ERROR] 의도 분석 중 오류 발생: {e}")
            state["prompt_draft"] = {
                "system_prompt": "오류로 인해 초기 프롬프트를 생성하지 못했습니다.",
                "user_prompt_template": "오류로 인해 사용자 프롬프트 템플릿을 생성하지 못했습니다.",
                "reasoning": f"오류 발생: {str(e)}",
            }

        return state

    def refine_prompt(state):
        """초기 프롬프트 초안을 개선합니다."""
        draft = state["prompt_draft"]
        user_intention = state["user_intention"]
        prompt_type = state["prompt_type"]

        type_specific_guidance = ""
        if prompt_type == "user":
            type_specific_guidance = "사용자 프롬프트 템플릿에 더 집중하여 개선하세요."
        else:
            type_specific_guidance = "시스템 프롬프트에 더 집중하여 개선하세요."

        system_message = f"""당신은 최고 수준의 프롬프트 엔지니어입니다.
초기 프롬프트 초안을 검토하고 더 효과적이고 정교한 프롬프트로 개선해주세요.
{type_specific_guidance}
명확성, 효과성, 사용자 의도 반영 측면에서 개선점을 찾아 적용하세요.
다음 JSON 형식으로 응답해주세요:
{{
  "system_prompt": "개선된 시스템 프롬프트",
  "user_prompt_template": "개선된 사용자 프롬프트 템플릿",
  "reasoning": "개선 사항에 대한 상세 설명과 효과적인 이유"
}}"""

        prompt_input = (
            f"다음 초기 프롬프트 초안을 검토하고 개선해주세요:\n\n"
            f"사용자 의도: {user_intention}\n\n"
            f"프롬프트 타입: {prompt_type} 프롬프트\n\n"
            f"초기 시스템 프롬프트:\n\"{draft['system_prompt']}\"\n\n"
            f"초기 사용자 프롬프트 템플릿:\n\"{draft['user_prompt_template']}\"\n\n"
            "이 프롬프트를 더 효과적으로 개선하고 JSON 형식으로 응답해주세요."
        )

        try:
            response = llm_gpt4o.invoke(
                [
                    SystemMessage(content=system_message),
                    HumanMessage(content=prompt_input),
                ]
            )

            parsed_output = PromptOutput.model_validate_json(response.content)

            state["prompt_refined"] = {
                "system_prompt": parsed_output.system_prompt.strip(),
                "user_prompt_template": parsed_output.user_prompt_template.strip(),
                "reasoning": parsed_output.reasoning.strip(),
            }

        except Exception as e:
            print(f"[ERROR] 프롬프트 개선 중 오류 발생: {e}")
            state["prompt_refined"] = state["prompt_draft"]

        return state

    def finalize_prompt(state):
        """최종 프롬프트를 결정하고 평가합니다."""
        refined = state["prompt_refined"]
        user_intention = state["user_intention"]
        prompt_type = state["prompt_type"]

        type_specific_guidance = ""
        if prompt_type == "user":
            type_specific_guidance = "사용자 프롬프트 템플릿이 사용자 의도를 직접적으로 표현하는지 확인하세요."
        else:
            type_specific_guidance = (
                "시스템 프롬프트가 AI의 역할과 행동을 명확하게 정의하는지 확인하세요."
            )

        system_message = f"""당신은 프롬프트 엔지니어링 전문가입니다.
개선된 프롬프트를 최종 검토하고, 사용자 의도에 가장 적합한 최종 버전을 확정해주세요.
{type_specific_guidance}
최종 프롬프트는 구체적이고, 명확하며, 사용자 의도를 정확히 충족해야 합니다.
다음 JSON 형식으로 응답해주세요:
{{
  "system_prompt": "최종 확정된 시스템 프롬프트",
  "user_prompt_template": "최종 확정된 사용자 프롬프트 템플릿",
  "reasoning": "최종 프롬프트가 사용자 의도를 어떻게 충족하는지와 예상되는 효과에 대한 최종 평가"
}}"""

        prompt_input = (
            f"다음 개선된 프롬프트를 최종 검토하고 확정해주세요:\n\n"
            f"사용자 의도: {user_intention}\n\n"
            f"프롬프트 타입: {prompt_type} 프롬프트\n\n"
            f"개선된 시스템 프롬프트:\n\"{refined['system_prompt']}\"\n\n"
            f"개선된 사용자 프롬프트 템플릿:\n\"{refined['user_prompt_template']}\"\n\n"
            "최종 확정된 프롬프트를 JSON 형식으로 제공하고 그 효과에 대한 최종 평가를 해주세요."
        )

        try:
            response = llm_gpt4o.invoke(
                [
                    SystemMessage(content=system_message),
                    HumanMessage(content=prompt_input),
                ]
            )

            parsed_output = PromptOutput.model_validate_json(response.content)

            state["final_prompt"] = {
                "system_prompt": parsed_output.system_prompt.strip(),
                "user_prompt_template": parsed_output.user_prompt_template.strip(),
                "reasoning": parsed_output.reasoning.strip(),
            }

        except Exception as e:
            print(f"[ERROR] 프롬프트 확정 중 오류 발생: {e}")
            state["final_prompt"] = state["prompt_refined"]

        return state

    def generate_report(state):
        """프롬프트 생성 과정과 최종 결과에 대한 보고서를 생성합니다."""
        final = state["final_prompt"]
        prompt_type = state["prompt_type"]

        print("--프롬프트 생성 최종 결과--\n")

        if prompt_type == "user":
            print(
                f"🔹 사용자 프롬프트 템플릿 (주요 결과물):\n{final['user_prompt_template']}\n"
            )
            print(f"🔹 추천 시스템 프롬프트:\n{final['system_prompt']}\n")
        else:
            print(f"🔹 시스템 프롬프트 (주요 결과물):\n{final['system_prompt']}\n")
            print(f"🔹 추천 사용자 프롬프트 템플릿:\n{final['user_prompt_template']}\n")

        print(f"🔹 프롬프트 설계 근거:\n{final['reasoning']}\n")

        return state

    ##########

    graph = StateGraph(dict)

    graph.add_node("Analyze Intention", analyze_intention)
    graph.add_node("Refine Prompt", refine_prompt)
    graph.add_node("Finalize Prompt", finalize_prompt)
    graph.add_node("Generate Report", generate_report)

    graph.set_entry_point("Analyze Intention")
    graph.add_edge("Analyze Intention", "Refine Prompt")
    graph.add_edge("Refine Prompt", "Finalize Prompt")
    graph.add_edge("Finalize Prompt", "Generate Report")

    graph_executor = graph.compile()

    result = graph_executor.invoke(state)
    return result


def generate_prompt_by_intention(user_intention, prompt_type):
    """
    사용자 의도를 기반으로 프롬프트를 생성합니다.

    Args:
        user_intention (str): 사용자가 원하는 프롬프트의 의도/목적
        prompt_type (str): 'user' 또는 'system' 프롬프트 유형

    Returns:
        dict: 생성된 프롬프트 정보를 담은 딕셔너리
    """
    result = run_prompt_generation_agent(user_intention, prompt_type)

    if result and result.get("final_prompt"):
        final_prompt = result["final_prompt"]

        if prompt_type == "user":
            return {
                "prompt": final_prompt["user_prompt_template"],
                "reasoning": final_prompt["reasoning"],
            }
        else:  # system
            return {
                "prompt": final_prompt["system_prompt"],
                "reasoning": final_prompt["reasoning"],
            }

    return {
        "prompt": f"생성 실패: {prompt_type} 프롬프트 ({user_intention})",
        "reasoning": "프롬프트 생성 중 오류가 발생했습니다.",
    }
