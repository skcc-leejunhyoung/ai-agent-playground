# agent/prompt_generation.py

import os
import json
import asyncio
import queue as queue_module
from typing import List, Dict, Any, Optional

from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from openai import AzureOpenAI
from pydantic import BaseModel, Field

#############
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AOAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AOAI_DEPLOY_GPT4O")
AZURE_OPENAI_API_VERSION = "2024-10-21"

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)

################################
# 2) Pydantic Models
################################


class RoleGuidanceOutput(BaseModel):
    role: str
    instructions: str
    information: str


class OutputExampleOutput(BaseModel):
    output_example: str


class RoleSummaryOutput(BaseModel):
    summary: List[str]


class ConflictEvaluationOutput(BaseModel):
    has_conflicts: bool
    conflicts: Optional[List[str]] = None
    resolution: Optional[str] = None


class ConflictResolutionOutput(BaseModel):
    role: str
    instructions: str
    information: str
    output_example: str


class CoverageEvaluationOutput(BaseModel):
    is_complete: bool
    missing_items: Optional[List[str]] = None


class CoverageFixOutput(BaseModel):
    role: str
    instructions: str
    information: str
    output_example: str


class FinalSystemPromptOutput(BaseModel):
    system_prompt: str
    reasoning: str


class InputExampleEvaluation(BaseModel):
    has_example: bool
    input_example: Optional[str] = None


class IntentionSummaryOutput(BaseModel):
    summary: List[str]


class UserPromptGenerationOutput(BaseModel):
    instructions: str
    information: str
    output_example: str


class UserPromptCoverageEvaluation(BaseModel):
    is_complete: bool
    missing_items: Optional[List[str]] = None


class UserPromptFixOutput(BaseModel):
    user_prompt: str


class FinalUserPromptOutput(BaseModel):
    user_prompt_template: str
    reasoning: str


################################
# 3) 비동기 함수
################################


async def run_prompt_generation_agent_async(
    user_intention: str,
    user_sys_params: Optional[str] = None,
    status_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    구조화된 Outputs를 각 단계에서 parse(...)로 받아서 state를 채운다.
    status_callback: 상태 업데이트를 위한 콜백 함수 (선택적)
    """

    print("DEBUG: Enter run_prompt_generation_agent_async()")
    if user_sys_params not in ["user", "system"]:
        user_sys_params = "system"

    # 상태 업데이트 함수
    def update_status(message, state="running"):
        if status_callback:
            status_callback(message, state)
        print(f"STATUS: {message} ({state})")

    state = {
        "user_intention": user_intention,
        "prompt_type": user_sys_params,
        # ...
        "role_guidance": None,
        "output_example": None,
        "role_summary": None,
        "conflict_evaluation": None,
        "coverage_evaluation": None,
        "final_prompt": None,
        "input_example_evaluation": None,
        "intention_summary": None,
        "user_prompt_generation": None,
        "user_prompt_coverage": None,
    }

    MAX_RETRY = 20

    async def parse_chat(system_content: str, user_content: str, pydantic_model):
        """
        구조화된 응답을 pydantic_model로 파싱
        """
        print(f"DEBUG: 파싱 시작 - {pydantic_model.__name__}")
        response = client.beta.chat.completions.parse(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,  # 실제 배포 이름
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            response_format=pydantic_model,
        )
        print(f"DEBUG: 파싱 완료 - {pydantic_model.__name__}")
        return response.choices[0].message.parsed

    ####### 시스템 프롬프트 노드들 #######
    async def node_system_1_role_guidance(state):
        print("DEBUG: 시작 - node_system_1_role_guidance")
        system_msg = "당신은 AI 프롬프트 전문가입니다. 역할/작동방식을 ##역할, ##지시 사항, ##정보로 나누어 JSON으로 출력하세요."
        user_msg = state["user_intention"]
        try:
            parsed = await parse_chat(system_msg, user_msg, RoleGuidanceOutput)
            state["role_guidance"] = {
                "role": parsed.role,
                "instructions": parsed.instructions,
                "information": parsed.information,
            }
        except Exception as e:
            print("[ERROR] node_system_1_role_guidance:", e)
            state["role_guidance"] = {
                "role": f"(오류){e}",
                "instructions": f"(오류){e}",
                "information": f"(오류){e}",
            }
        print("DEBUG: 종료 - node_system_1_role_guidance")
        return state

    async def node_system_2_output_example(state):
        print("DEBUG: 시작 - node_system_2_output_example")
        system_msg = "당신은 AI 프롬프트 전문가입니다. 역할/작동방식으로부터 출력 예시를 JSON으로 생성하세요."
        user_msg = state["user_intention"]
        try:
            parsed = await parse_chat(system_msg, user_msg, OutputExampleOutput)
            state["output_example"] = {"output_example": parsed.output_example}
        except Exception as e:
            print("[ERROR] node_system_2_output_example:", e)
            state["output_example"] = {"output_example": f"(오류){e}"}
        print("DEBUG: 종료 - node_system_2_output_example")
        return state

    async def node_system_3_role_summary(state):
        print("DEBUG: 시작 - node_system_3_role_summary")
        system_msg = "당신은 AI 프롬프트 전문가입니다. 역할/작동방식을 개괄식으로 빠짐없이 나열해 JSON 배열로 정제하세요."
        user_msg = state["user_intention"]
        try:
            parsed = await parse_chat(system_msg, user_msg, RoleSummaryOutput)
            state["role_summary"] = {"summary": parsed.summary}
        except Exception as e:
            print("[ERROR] node_system_3_role_summary:", e)
            state["role_summary"] = {"summary": [f"(오류){e}"]}
        print("DEBUG: 종료 - node_system_3_role_summary")
        return state

    async def node_system_4_conflict_evaluation(state):
        print("DEBUG: 시작 - node_system_4_conflict_evaluation")
        system_msg = "당신은 AI 프롬프트 전문가입니다. 각 항목이 서로 모순되거나 충돌하는 지시, 정보가 없는지 평가하세요."
        rg = state["role_guidance"]
        oe = state["output_example"]
        user_msg = f"""##역할: {rg['role']}
##지시 사항: {rg['instructions']}
##정보: {rg['information']}
##출력 예시: {oe['output_example']}"""
        try:
            parsed = await parse_chat(system_msg, user_msg, ConflictEvaluationOutput)
            state["conflict_evaluation"] = {
                "has_conflicts": parsed.has_conflicts,
                "conflicts": parsed.conflicts,
                "resolution": parsed.resolution,
            }
            print(f"DEBUG: 충돌 평가 결과 - has_conflicts: {parsed.has_conflicts}")
        except Exception as e:
            print("[ERROR] node_system_4_conflict_evaluation:", e)
            state["conflict_evaluation"] = {
                "has_conflicts": False,
                "conflicts": None,
                "resolution": None,
            }
        print("DEBUG: 종료 - node_system_4_conflict_evaluation")
        return state

    async def node_system_4_1_conflict_resolution(state):
        print("DEBUG: 시작 - node_system_4_1_conflict_resolution")
        system_msg = "당신은 AI 프롬프트 전문가입니다. 충돌을 해결하는 새로운 항목들을 생성하세요."
        rg = state["role_guidance"]
        oe = state["output_example"]
        ce = state["conflict_evaluation"]
        user_msg = f"""##역할: {rg['role']}
##지시 사항: {rg['instructions']}
##정보: {rg['information']}
##출력 예시: {oe['output_example']}

발견된 충돌: {ce['conflicts']}
해결안: {ce['resolution']}"""
        try:
            parsed = await parse_chat(system_msg, user_msg, ConflictResolutionOutput)
            state["role_guidance"] = {
                "role": parsed.role,
                "instructions": parsed.instructions,
                "information": parsed.information,
            }
            state["output_example"] = {"output_example": parsed.output_example}
        except Exception as e:
            print("[ERROR] node_system_4_1_conflict_resolution:", e)
        print("DEBUG: 종료 - node_system_4_1_conflict_resolution")
        return state

    async def node_system_5_coverage_evaluation(state):
        print("DEBUG: 시작 - node_system_5_coverage_evaluation")
        system_msg = "당신은 AI 프롬프트 전문가입니다. 개괄식으로 정제한 역할/작동방식의 모든 항목을 만족하는지 평가하세요."
        rg = state["role_guidance"]
        oe = state["output_example"]
        rs = state["role_summary"]
        user_msg = f"""##역할: {rg['role']}
##지시 사항: {rg['instructions']}
##정보: {rg['information']}
##출력 예시: {oe['output_example']}

요구사항 목록: {rs['summary']}"""
        try:
            parsed = await parse_chat(system_msg, user_msg, CoverageEvaluationOutput)
            state["coverage_evaluation"] = {
                "is_complete": parsed.is_complete,
                "missing_items": parsed.missing_items,
            }
            print(f"DEBUG: 커버리지 평가 결과 - is_complete: {parsed.is_complete}")
        except Exception as e:
            print("[ERROR] node_system_5_coverage_evaluation:", e)
            state["coverage_evaluation"] = {"is_complete": True, "missing_items": None}
        print("DEBUG: 종료 - node_system_5_coverage_evaluation")
        return state

    async def node_system_5_1_coverage_fix(state):
        print("DEBUG: 시작 - node_system_5_1_coverage_fix")
        system_msg = "당신은 AI 프롬프트 전문가입니다. 만족하지 못한 항목을 수정하세요."
        rg = state["role_guidance"]
        oe = state["output_example"]
        ce = state["coverage_evaluation"]
        rs = state["role_summary"]
        user_msg = f"""##역할: {rg['role']}
##지시 사항: {rg['instructions']}
##정보: {rg['information']}
##출력 예시: {oe['output_example']}

누락된 항목: {ce['missing_items']}
요구사항 목록: {rs['summary']}"""
        try:
            parsed = await parse_chat(system_msg, user_msg, CoverageFixOutput)
            state["role_guidance"] = {
                "role": parsed.role,
                "instructions": parsed.instructions,
                "information": parsed.information,
            }
            state["output_example"] = {"output_example": parsed.output_example}
        except Exception as e:
            print("[ERROR] node_system_5_1_coverage_fix:", e)
        print("DEBUG: 종료 - node_system_5_1_coverage_fix")
        return state

    async def node_system_final(state):
        print("DEBUG: 시작 - node_system_final")
        system_msg = (
            "당신은 AI 프롬프트 전문가입니다. 최종 시스템 프롬프트를 생성하세요."
        )
        rg = state["role_guidance"]
        oe = state["output_example"]
        user_msg = f"""##역할: {rg['role']}
##지시 사항: {rg['instructions']}
##정보: {rg['information']}
##출력 예시: {oe['output_example']}"""
        try:
            parsed = await parse_chat(system_msg, user_msg, FinalSystemPromptOutput)
            state["final_prompt"] = {
                "system_prompt": parsed.system_prompt,
                "user_prompt_template": "{{사용자 입력}}",
                "reasoning": parsed.reasoning,
            }
        except Exception as e:
            print("[ERROR] node_system_final:", e)
            state["final_prompt"] = {
                "system_prompt": "(오류)시스템프롬프트",
                "user_prompt_template": "{{사용자 입력}}",
                "reasoning": str(e),
            }
        print("DEBUG: 종료 - node_system_final")
        return state

    ####### 유저 프롬프트 노드들 #######
    async def node_user_1_input_example_check(state):
        print("DEBUG: 시작 - node_user_1_input_example_check")
        system_msg = "당신은 AI 프롬프트 전문가입니다. 의도/목적에 입력 예시가 있는지 확인하세요."
        user_msg = state["user_intention"]
        try:
            parsed = await parse_chat(system_msg, user_msg, InputExampleEvaluation)
            state["input_example_evaluation"] = {
                "has_example": parsed.has_example,
                "input_example": parsed.input_example,
            }
            print(f"DEBUG: 입력 예시 확인 결과 - has_example: {parsed.has_example}")
        except Exception as e:
            print("[ERROR] node_user_1_input_example_check:", e)
            state["input_example_evaluation"] = {
                "has_example": False,
                "input_example": None,
            }
        print("DEBUG: 종료 - node_user_1_input_example_check")
        return state

    async def node_user_2_intention_summary(state):
        print("DEBUG: 시작 - node_user_2_intention_summary")
        system_msg = "당신은 AI 프롬프트 전문가입니다. 의도/목적을 개괄식으로 빠짐없이 나열해 정제하세요."
        user_msg = state["user_intention"]
        try:
            parsed = await parse_chat(system_msg, user_msg, IntentionSummaryOutput)
            state["intention_summary"] = {"summary": parsed.summary}
        except Exception as e:
            print("[ERROR] node_user_2_intention_summary:", e)
            state["intention_summary"] = {"summary": [f"(오류){e}"]}
        print("DEBUG: 종료 - node_user_2_intention_summary")
        return state

    async def node_user_3_prompt_generation(state):
        print("DEBUG: 시작 - node_user_3_prompt_generation")
        system_msg = "당신은 AI 프롬프트 전문가입니다. 의도/목적으로부터 지시사항, 정보, 출력 예시를 생성하세요."
        ie = state["input_example_evaluation"]
        user_msg = f"""의도/목적: {state["user_intention"]}
입력 예시 포함 여부: {ie["has_example"]}
입력 예시: {ie["input_example"]}"""
        try:
            parsed = await parse_chat(system_msg, user_msg, UserPromptGenerationOutput)
            # 프롬프트 조합
            if ie["has_example"] and ie["input_example"]:
                user_prompt = (
                    f"{parsed.instructions}\n\n"
                    f"{parsed.information}\n\n"
                    f"입력 예시: {ie['input_example']}\n\n"
                    f"출력 예시: {parsed.output_example}"
                )
            else:
                user_prompt = (
                    f"{parsed.instructions}\n\n"
                    f"{parsed.information}\n\n"
                    f"출력 예시: {parsed.output_example}"
                )
            state["user_prompt_generation"] = {
                "user_prompt": user_prompt,
                "instructions": parsed.instructions,
                "information": parsed.information,
                "output_example": parsed.output_example,
            }
        except Exception as e:
            print("[ERROR] node_user_3_prompt_generation:", e)
            state["user_prompt_generation"] = {
                "user_prompt": f"(오류){e}",
                "instructions": f"(오류){e}",
                "information": f"(오류){e}",
                "output_example": f"(오류){e}",
            }
        print("DEBUG: 종료 - node_user_3_prompt_generation")
        return state

    async def node_user_4_coverage_evaluation(state):
        print("DEBUG: 시작 - node_user_4_coverage_evaluation")
        system_msg = "당신은 AI 프롬프트 전문가입니다. 프롬프트가 의도/목적의 모든 항목을 만족하는지 평가하세요."
        up = state["user_prompt_generation"]["user_prompt"]
        it = state["intention_summary"]
        ie = state["input_example_evaluation"]
        user_msg = f"""생성된 프롬프트:
{up}

입력 예시 포함 여부: {ie["has_example"]}
입력 예시: {ie["input_example"]}

의도/목적 요약: {it["summary"]}"""
        try:
            parsed = await parse_chat(
                system_msg, user_msg, UserPromptCoverageEvaluation
            )
            state["user_prompt_coverage"] = {
                "is_complete": parsed.is_complete,
                "missing_items": parsed.missing_items,
            }
            print(
                f"DEBUG: 유저 프롬프트 커버리지 평가 결과 - is_complete: {parsed.is_complete}"
            )
        except Exception as e:
            print("[ERROR] node_user_4_coverage_evaluation:", e)
            state["user_prompt_coverage"] = {"is_complete": True, "missing_items": None}
        print("DEBUG: 종료 - node_user_4_coverage_evaluation")
        return state

    async def node_user_4_1_coverage_fix(state):
        print("DEBUG: 시작 - node_user_4_1_coverage_fix")
        system_msg = "당신은 AI 프롬프트 전문가입니다. 만족하지 못한 항목을 수정하세요."
        up = state["user_prompt_generation"]["user_prompt"]
        uc = state["user_prompt_coverage"]
        it = state["intention_summary"]
        user_msg = f"""현재 프롬프트:
{up}

누락된 항목: {uc["missing_items"]}
의도/목적 요약: {it["summary"]}"""
        try:
            parsed = await parse_chat(system_msg, user_msg, UserPromptFixOutput)
            state["user_prompt_generation"]["user_prompt"] = parsed.user_prompt
        except Exception as e:
            print("[ERROR] node_user_4_1_coverage_fix:", e)
        print("DEBUG: 종료 - node_user_4_1_coverage_fix")
        return state

    async def node_user_final(state):
        print("DEBUG: 시작 - node_user_final")
        system_msg = "당신은 AI 프롬프트 전문가입니다. 최종 유저 프롬프트를 생성하세요."
        up = state["user_prompt_generation"]["user_prompt"]
        try:
            parsed = await parse_chat(system_msg, up, FinalUserPromptOutput)
            state["final_prompt"] = {
                "system_prompt": "당신은 유용한 AI 어시스턴트입니다.",
                "user_prompt_template": parsed.user_prompt_template,
                "reasoning": parsed.reasoning,
            }
        except Exception as e:
            print("[ERROR] node_user_final:", e)
            state["final_prompt"] = {
                "system_prompt": "당신은 유용한 AI 어시스턴트입니다.",
                "user_prompt_template": f"(오류){e}",
                "reasoning": f"(오류){e}",
            }
        print("DEBUG: 종료 - node_user_final")
        return state

    ################################
    # 시스템 프롬프트 생성 흐름
    ################################
    async def system_prompt_workflow(state):
        print("DEBUG: 시스템 프롬프트 생성 워크플로우 시작")
        update_status("시스템 프롬프트 생성 워크플로우 시작")

        # 1, 2, 3번 노드 병렬 처리
        print("DEBUG: 병렬 실행 - 노드 1, 2, 3")
        update_status("역할/작동방식 분석 중 (1/5)")
        node1_task = asyncio.create_task(node_system_1_role_guidance(state.copy()))
        node2_task = asyncio.create_task(node_system_2_output_example(state.copy()))
        node3_task = asyncio.create_task(node_system_3_role_summary(state.copy()))

        # 병렬 작업 완료 대기
        node1_result, node2_result, node3_result = await asyncio.gather(
            node1_task, node2_task, node3_task
        )
        print("DEBUG: 병렬 실행 완료 - 노드 1, 2, 3")

        # 결과 병합
        state["role_guidance"] = node1_result["role_guidance"]
        state["output_example"] = node2_result["output_example"]
        state["role_summary"] = node3_result["role_summary"]

        # 4번 노드 - 충돌 평가 및 해결
        update_status("항목 간 충돌 검토 중 (2/5)")
        state = await node_system_4_conflict_evaluation(state)
        retry_count = 0
        while state["conflict_evaluation"].get("has_conflicts", False):
            print(f"DEBUG: 충돌 발견, 재시도 {retry_count+1}/{MAX_RETRY}")
            update_status(f"충돌 해결 중 (재시도 {retry_count+1}/{MAX_RETRY})")
            if retry_count >= MAX_RETRY:
                print("DEBUG: 최대 재시도 횟수 초과, 충돌 해결 중단")
                break
            state = await node_system_4_1_conflict_resolution(state)
            state = await node_system_4_conflict_evaluation(state)
            retry_count += 1

        # 5번 노드 - 커버리지 평가 및 수정
        update_status("요구사항 충족도 검토 중 (3/5)")
        state = await node_system_5_coverage_evaluation(state)
        retry_count = 0
        while not state["coverage_evaluation"].get("is_complete", True):
            print(f"DEBUG: 커버리지 부족, 재시도 {retry_count+1}/{MAX_RETRY}")
            update_status(f"누락된 항목 보완 중 (재시도 {retry_count+1}/{MAX_RETRY})")
            if retry_count >= MAX_RETRY:
                print("DEBUG: 최대 재시도 횟수 초과, 커버리지 수정 중단")
                break
            state = await node_system_5_1_coverage_fix(state)
            state = await node_system_5_coverage_evaluation(state)
            retry_count += 1

        # 최종 시스템 프롬프트 생성
        update_status("최종 시스템 프롬프트 생성 중 (4/5)")
        state = await node_system_final(state)
        update_status("시스템 프롬프트 생성 완료 (5/5)", "complete")
        print("DEBUG: 시스템 프롬프트 생성 워크플로우 완료")
        return state

    ################################
    # 유저 프롬프트 생성 흐름
    ################################
    async def user_prompt_workflow(state):
        print("DEBUG: 유저 프롬프트 생성 워크플로우 시작")
        update_status("유저 프롬프트 생성 워크플로우 시작")

        # 1, 2번 노드 병렬 처리
        print("DEBUG: 병렬 실행 - 노드 1, 2")
        update_status("의도/목적 분석 중 (1/4)")
        node1_task = asyncio.create_task(node_user_1_input_example_check(state.copy()))
        node2_task = asyncio.create_task(node_user_2_intention_summary(state.copy()))

        # 병렬 작업 완료 대기
        node1_result, node2_result = await asyncio.gather(node1_task, node2_task)
        print("DEBUG: 병렬 실행 완료 - 노드 1, 2")

        # 결과 병합
        state["input_example_evaluation"] = node1_result["input_example_evaluation"]
        state["intention_summary"] = node2_result["intention_summary"]

        # 프롬프트 생성
        update_status("프롬프트 초안 생성 중 (2/4)")
        state = await node_user_3_prompt_generation(state)

        # 4번 노드 - 커버리지 평가 및 수정
        update_status("요구사항 충족도 검토 중 (3/4)")
        state = await node_user_4_coverage_evaluation(state)
        retry_count = 0
        while not state["user_prompt_coverage"].get("is_complete", True):
            print(
                f"DEBUG: 유저 프롬프트 커버리지 부족, 재시도 {retry_count+1}/{MAX_RETRY}"
            )
            update_status(f"누락된 항목 보완 중 (재시도 {retry_count+1}/{MAX_RETRY})")
            if retry_count >= MAX_RETRY:
                print("DEBUG: 최대 재시도 횟수 초과, 유저 프롬프트 커버리지 수정 중단")
                break
            state = await node_user_4_1_coverage_fix(state)
            state = await node_user_4_coverage_evaluation(state)
            retry_count += 1

        # 최종 유저 프롬프트 생성
        update_status("최종 유저 프롬프트 생성 중 (4/4)")
        state = await node_user_final(state)
        update_status("유저 프롬프트 생성 완료", "complete")
        print("DEBUG: 유저 프롬프트 생성 워크플로우 완료")
        return state

    ################################
    # 메인 워크플로우 실행
    ################################
    if user_sys_params == "system":
        state = await system_prompt_workflow(state)
    else:
        state = await user_prompt_workflow(state)

    print("DEBUG: Exit run_prompt_generation_agent_async()")
    return state


################################
# 4) 동기 래퍼
################################
def run_prompt_generation_agent(
    user_intention: str,
    user_sys_params: Optional[str] = None,
    status_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    print("DEBUG: Enter run_prompt_generation_agent (SYNC WRAPPER)")
    result = asyncio.run(
        run_prompt_generation_agent_async(
            user_intention, user_sys_params, status_callback
        )
    )
    print("DEBUG: Exit run_prompt_generation_agent (SYNC WRAPPER)")
    return result


################################
# 5) 스트리밍 함수
################################
def _run_agent_and_yield(
    user_intention: str, prompt_type: str, status_callback: Optional[callable] = None
):
    """단계별 결과를 yield"""
    yield f"'{prompt_type}' 프롬프트 생성을 위한 분석을 시작합니다...\n"
    try:
        result = run_prompt_generation_agent(
            user_intention, prompt_type, status_callback
        )
        final_prompt = result.get("final_prompt", {})
        if prompt_type == "system":
            sp = final_prompt.get("system_prompt", "")
            yield f"## 최종 시스템 프롬프트\n\n```\n{sp}\n```\n\n"
            yield f"### 추천 사용자 프롬프트 템플릿\n```\n{final_prompt.get('user_prompt_template','')}\n```\n\n"
        else:
            up = final_prompt.get("user_prompt_template", "")
            yield f"## 최종 사용자 프롬프트\n\n```\n{up}\n```\n\n"
            yield f"### 추천 시스템 프롬프트\n```\n{final_prompt.get('system_prompt','')}\n```\n\n"

        yield f"### 설계 근거\n\n{final_prompt.get('reasoning','')}\n\n"
        data = {
            "prompt_type": prompt_type,
            "prompt": sp if prompt_type == "system" else up,
            "reasoning": final_prompt.get("reasoning", ""),
        }
        yield f"__RESULT_DATA__:{json.dumps(data)}"

    except Exception as e:
        yield f"오류 발생: {str(e)}\n"
        data = {
            "prompt_type": prompt_type,
            "prompt": f"(오류) {prompt_type}",
            "reasoning": f"(오류) {str(e)}",
        }
        yield f"__RESULT_DATA__:{json.dumps(data)}"


def run_prompt_generation_agent_streaming(
    user_intention: str,
    prompt_type: Optional[str] = None,
    status_callback: Optional[callable] = None,
):
    """별도 스레드 + Queue로 스트리밍"""
    ctx = get_script_run_ctx()
    if prompt_type not in ["user", "system"]:
        prompt_type = "system"

    q = queue_module.Queue()

    def worker():
        add_script_run_ctx(thread=None, ctx=ctx)
        for chunk in _run_agent_and_yield(user_intention, prompt_type, status_callback):
            q.put(chunk)
        q.put(None)

    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(worker)

    while True:
        item = q.get()
        if item is None:
            break
        yield item
    executor.shutdown(wait=True)


################################
# 6) 최종 외부 공개 함수
################################
def generate_prompt_by_intention_streaming(
    user_intention: str, prompt_type: str, status_callback: Optional[callable] = None
):
    yield from run_prompt_generation_agent_streaming(
        user_intention, prompt_type, status_callback
    )


def generate_prompt_by_intention(
    user_intention: str, prompt_type: str, status_callback: Optional[callable] = None
):
    result = run_prompt_generation_agent(user_intention, prompt_type, status_callback)
    fp = result.get("final_prompt", {})
    if "system_prompt" in fp or "user_prompt_template" in fp:
        if prompt_type == "system":
            return {
                "prompt": fp.get("system_prompt", ""),
                "reasoning": fp.get("reasoning", ""),
            }
        else:
            return {
                "prompt": fp.get("user_prompt_template", ""),
                "reasoning": fp.get("reasoning", ""),
            }
    else:
        return {"prompt": f"생성 실패: {prompt_type}", "reasoning": "오류 발생"}
