# agent/eval_agent.py

from langgraph.graph import StateGraph
import pandas as pd
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
    max_tokens=500,
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
        passed_df = df[df["eval_pass"] == "O"]
        grouped = (
            passed_df.groupby(["model", "system_prompt"])
            .size()
            .reset_index(name="eval_pass_O_count")
        )
        best_prompts = (
            grouped.sort_values("eval_pass_O_count", ascending=False)
            .groupby("model")
            .first()
            .reset_index()
        )
        state["best_prompts"] = best_prompts
        return state

    def suggest_improved_prompt(state):
        best_prompts = state["best_prompts"]

        top_row = best_prompts.sort_values("eval_pass_O_count", ascending=False).iloc[0]
        model = top_row["model"]
        system_prompt = top_row["system_prompt"]
        pass_count = top_row["eval_pass_O_count"]

        format_instructions = system_prompt_parser.get_format_instructions()

        prompt_input = (
            f"다음 시스템 프롬프트를 사용하여 {model} 모델이 {pass_count}개의 성공을 거두었습니다.\n\n"
            f'시스템 프롬프트:\n"{system_prompt}"\n\n'
            "이 시스템 프롬프트를 개선하여 더 나은 결과를 얻기 위해 수정하거나 보완해 주세요.\n\n"
            "개선한 이유를 상세하게 설명하고, 아래 포맷에 맞춰 작성해 주세요.\n\n"
            f"{format_instructions}"
        )

        try:
            response = llm_gpt4o.invoke(
                [
                    SystemMessage(
                        content="당신은 뛰어난 AI 프롬프트 엔지니어입니다. 아래 포맷을 따라 개선된 시스템 프롬프트와 개선 이유를 제안하세요."
                    ),
                    HumanMessage(content=prompt_input),
                ]
            )

            parsed_output = system_prompt_parser.parse(response.content)
            improved_prompt = parsed_output.system_prompt.strip()
            reason = parsed_output.reason.strip()

            state["improved_prompt"] = {
                "model": model,
                "improved_prompt": improved_prompt,
                "reason": reason,
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
            )

        if state["improved_prompt"]:
            improved = state["improved_prompt"]
            print(
                f"\n개선된 시스템 프롬프트 ({improved['model']})\n➡️ {improved['improved_prompt']}\n"
                f"개선 이유: {improved['reason']}\n"
            )

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
