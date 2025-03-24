# agent/eval_agent.py

from langgraph.graph import StateGraph
import pandas as pd
import os
from dotenv import load_dotenv
import json

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
    max_tokens=2000,
    model_kwargs={"response_format": {"type": "json_object"}},
)


##########


class SystemPromptOutput(BaseModel):
    system_prompt: str = Field(..., description="최종 개선된 시스템 프롬프트입니다.")
    reason: str = Field(
        ...,
        description="프롬프트를 이렇게 개선한 이유와 설명입니다. 최대한 구체적으로 작성하되 markdown 포맷으로 보기좋게 한국어로 작성하세요.",
    )


system_prompt_parser = PydanticOutputParser(pydantic_object=SystemPromptOutput)


##########


def run_eval_agent(results, project_id):
    if not results:
        print("⚠️ 평가할 결과가 없습니다.")
        return None

    df = pd.DataFrame(results)

    state = {
        "df": df,
        "analysis": None,
        "best_prompts": None,
        "improved_prompt": None,
        "project_id": project_id,
    }

    def load_results(state):
        return state

    def analyze_results(state):
        df = state["df"]
        grouped = (
            df[df["eval_pass"] == "O"]
            .groupby(["user_prompt", "model", "system_prompt"])
            .size()
            .reset_index(name="eval_pass_O_count")
        )
        state["analysis"] = grouped
        return state

    def find_best_prompts(state):
        df = state["df"]
        # 각 프롬프트별 전체 시도 횟수와 성공 횟수를 계산
        total_counts = (
            df.groupby(["model", "system_prompt"])
            .size()
            .reset_index(name="total_count")
        )
        pass_counts = (
            df[df["eval_pass"] == "O"]
            .groupby(["model", "system_prompt"])
            .size()
            .reset_index(name="eval_pass_O_count")
        )

        # 두 데이터프레임 병합
        merged_df = pd.merge(
            total_counts, pass_counts, on=["model", "system_prompt"], how="left"
        )
        merged_df["success_rate"] = (
            merged_df["eval_pass_O_count"] / merged_df["total_count"]
        ) * 100

        # 성공 횟수로 정렬하고 각 모델별 최고 성과 프롬프트 선택
        best_prompts = (
            merged_df.sort_values("eval_pass_O_count", ascending=False)
            .groupby("model")
            .first()
            .reset_index()
        )

        # needs_improvement 필드 추가
        best_prompts["needs_improvement"] = best_prompts["success_rate"] < 90

        state["best_prompts"] = best_prompts
        return state

    def suggest_improved_prompt(state):
        best_prompts = state["best_prompts"]
        df = state["df"]

        if best_prompts.empty:
            print("[ERROR] 개선할 프롬프트가 없습니다.")
            return state

        top_row = best_prompts.sort_values("eval_pass_O_count", ascending=False).iloc[0]

        if not top_row["needs_improvement"]:
            print(
                f"[INFO] 현재 프롬프트의 성공률이 {top_row['success_rate']:.1f}%로 충분히 높아 개선이 필요하지 않습니다."
            )
            return state

        model = top_row["model"]
        system_prompt = top_row["system_prompt"]

        # 1단계: 실패 사례 분석
        failed_cases = df[
            (df["model"] == model)
            & (df["system_prompt"] == system_prompt)
            & (df["eval_pass"] == "X")
        ][["user_prompt", "result", "eval_keyword"]].to_dict("records")

        analysis_system_message = """당신은 뛰어난 AI 프롬프트 엔지니어입니다.
실패 사례들을 분석하여 문제점을 파악하고, 적용 가능한 프롬프팅 기법을 제안해주세요.
다음 JSON 형식으로 응답해주세요:
{
    "failure_analysis": "실패 사례들의 공통적인 문제점과 패턴 분석",
    "improvement_techniques": "적용할 수 있는 프롬프팅 기법들과 그 이유 (예: Chain of Thought, Few-shot 등)"
}"""

        analysis_input = (
            f"다음은 현재 시스템 프롬프트와 실패한 케이스들입니다:\n\n"
            f"현재 시스템 프롬프트:\n{system_prompt}\n\n"
            f"실패 케이스들:\n"
        )

        for i, case in enumerate(failed_cases, 1):
            analysis_input += (
                f"\n케이스 {i}:\n"
                f"사용자 입력: {case['user_prompt']}\n"
                f"AI 응답: {case['result']}\n"
                f"실패 이유: {case['eval_keyword']}\n"
            )

        try:
            # 1단계: 실패 분석 및 프롬프팅 기법 추천
            analysis_response = llm_gpt4o.invoke(
                [
                    SystemMessage(content=analysis_system_message),
                    HumanMessage(content=analysis_input),
                ]
            )
            analysis_result = json.loads(analysis_response.content)

            # 2단계: 개선된 프롬프트 생성
            improvement_system_message = """당신은 뛰어난 AI 프롬프트 엔지니어입니다.
실패 분석 결과와 추천된 프롬프팅 기법을 바탕으로, 개선된 시스템 프롬프트를 생성해주세요.
다음 JSON 형식으로 응답해주세요:
{
    "system_prompt": "개선된 시스템 프롬프트",
    "reason": "프롬프트를 이렇게 개선한 이유에 대한 상세 설명(markdown 포맷)"
}"""

            improvement_input = (
                f"현재 시스템 프롬프트:\n{system_prompt}\n\n"
                f"실패 분석 결과:\n{analysis_result['failure_analysis']}\n\n"
                f"추천된 프롬프팅 기법:\n{analysis_result['improvement_techniques']}\n\n"
                f"위 분석 결과와 추천된 프롬프팅 기법을 적용하여 개선된 시스템 프롬프트를 생성해주세요."
            )

            improvement_response = llm_gpt4o.invoke(
                [
                    SystemMessage(content=improvement_system_message),
                    HumanMessage(content=improvement_input),
                ]
            )
            improvement_result = json.loads(improvement_response.content)

            state["improved_prompt"] = {
                "model": model,
                "original_prompt": system_prompt,
                "improved_prompt": improvement_result["system_prompt"],
                "failure_analysis": analysis_result["failure_analysis"],
                "improvement_techniques": analysis_result["improvement_techniques"],
                "reason": improvement_result["reason"],
            }

        except Exception as e:
            print(f"[ERROR] 프롬프트 개선 중 오류 발생: {e}")

        return state

    def report_results(state):
        analysis = state["best_prompts"]
        print("--최종 평가 및 개선 결과 리포트--\n")

        for _, row in analysis.iterrows():
            print(
                f"Model: {row['model']}\n"
                f"Best System Prompt: {row['system_prompt']}\n"
                f"Pass Count: {row['eval_pass_O_count']}\n"
                f"Success Rate: {row['success_rate']:.1f}%\n"
            )

        if state["improved_prompt"]:
            improved = state["improved_prompt"]
            print("\n--프롬프트 개선 분석--")
            print(f"\n실패 사례 분석:\n{improved['failure_analysis']}")
            print(f"\n제안된 프롬프팅 기법:\n{improved['improvement_techniques']}")
            print(
                f"\n개선된 시스템 프롬프트 ({improved['model']}):\n➡️ {improved['improved_prompt']}"
            )
            print(f"\n개선 이유:\n{improved['reason']}\n")

        return state

    ##########

    graph = StateGraph(dict)

    graph.add_node("Load Results", load_results)
    graph.add_node("Analyze Results", analyze_results)
    graph.add_node("Find Best Prompts", find_best_prompts)
    graph.add_node("Suggest Improved Prompt", suggest_improved_prompt)
    graph.add_node("Report Results", report_results)

    graph.set_entry_point("Load Results")
    graph.add_edge("Load Results", "Analyze Results")
    graph.add_edge("Analyze Results", "Find Best Prompts")
    graph.add_edge("Find Best Prompts", "Suggest Improved Prompt")
    graph.add_edge("Suggest Improved Prompt", "Report Results")

    graph_executor = graph.compile()

    result = graph_executor.invoke(state)
    return result
