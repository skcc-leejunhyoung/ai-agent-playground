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
    user_intention: str, user_sys_params: Optional[str] = None
) -> Dict[str, Any]:
    """
    구조화된 Outputs를 각 단계에서 parse(...)로 받아서 state를 채운다.
    """

    print("DEBUG: Enter run_prompt_generation_agent_async()")
    if user_sys_params not in ["user", "system"]:
        user_sys_params = "system"

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
        response = client.beta.chat.completions.parse(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,  # 실제 배포 이름
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            response_format=pydantic_model,
        )
        return response.choices[0].message.parsed

    ####### 1) role guidance #######
    async def generate_role_guidance(state):
        system_msg = "당신은 AI 프롬프트 전문가입니다. AI 역할/동작을 구조화된 JSON으로 분리해 주세요."
        user_msg = state["user_intention"]
        try:
            parsed = await parse_chat(system_msg, user_msg, RoleGuidanceOutput)
            state["role_guidance"] = {
                "role": parsed.role,
                "instructions": parsed.instructions,
                "information": parsed.information,
            }
        except Exception as e:
            print("[ERROR] generate_role_guidance:", e)
            state["role_guidance"] = {
                "role": f"(오류){e}",
                "instructions": f"(오류){e}",
                "information": f"(오류){e}",
            }
        return state

    ####### 2) output example #######
    async def generate_output_example(state):
        system_msg = "JSON으로 출력 예시를 주세요. 필드는 output_example만."
        user_msg = state["user_intention"]
        try:
            parsed = await parse_chat(system_msg, user_msg, OutputExampleOutput)
            state["output_example"] = {"output_example": parsed.output_example}
        except Exception as e:
            print("[ERROR] generate_output_example:", e)
            state["output_example"] = {"output_example": f"(오류){e}"}
        return state

    async def summarize_role(state):
        system_msg = "JSON으로 role summary를 주세요. 필드는 summary: string[]"
        user_msg = state["user_intention"]
        try:
            parsed = await parse_chat(system_msg, user_msg, RoleSummaryOutput)
            state["role_summary"] = {"summary": parsed.summary}
        except Exception as e:
            print("[ERROR] summarize_role:", e)
            state["role_summary"] = {"summary": [f"(오류){e}"]}
        return state

    async def evaluate_conflicts(state):
        system_msg = "Check conflicts in JSON. has_conflicts, conflicts, resolution"
        rg = state["role_guidance"]
        oe = state["output_example"]
        user_msg = f"""역할: {rg['role']}
지시: {rg['instructions']}
정보: {rg['information']}
출력예시: {oe['output_example']}"""
        try:
            parsed = await parse_chat(system_msg, user_msg, ConflictEvaluationOutput)
            state["conflict_evaluation"] = {
                "has_conflicts": parsed.has_conflicts,
                "conflicts": parsed.conflicts,
                "resolution": parsed.resolution,
            }
        except Exception as e:
            print("[ERROR] evaluate_conflicts:", e)
            state["conflict_evaluation"] = {
                "has_conflicts": False,
                "conflicts": None,
                "resolution": None,
            }
        return state

    async def resolve_conflicts(state):
        system_msg = "JSON for conflict resolution. Fields: role, instructions, information, output_example"
        rg = state["role_guidance"]
        oe = state["output_example"]
        ce = state["conflict_evaluation"]
        user_msg = f"""역할: {rg['role']}
지시사항: {rg['instructions']}
info: {rg['information']}
output_ex: {oe['output_example']}

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
            state["conflict_evaluation"] = {
                "has_conflicts": False,
                "conflicts": None,
                "resolution": None,
            }
        except Exception as e:
            print("[ERROR] resolve_conflicts:", e)
        return state

    async def evaluate_coverage(state):
        system_msg = "JSON coverage check -> is_complete, missing_items"
        rg = state["role_guidance"]
        oe = state["output_example"]
        rs = state["role_summary"]
        user_msg = f"""역할: {rg['role']}
지시: {rg['instructions']}
info: {rg['information']}
출력예시: {oe['output_example']}

summary: {rs['summary']}"""
        try:
            parsed = await parse_chat(system_msg, user_msg, CoverageEvaluationOutput)
            state["coverage_evaluation"] = {
                "is_complete": parsed.is_complete,
                "missing_items": parsed.missing_items,
            }
        except Exception as e:
            print("[ERROR] evaluate_coverage:", e)
            state["coverage_evaluation"] = {"is_complete": True, "missing_items": None}
        return state

    async def fix_coverage(state):
        system_msg = (
            "JSON coverage fix -> role, instructions, information, output_example"
        )
        rg = state["role_guidance"]
        oe = state["output_example"]
        ce = state["coverage_evaluation"]
        rs = state["role_summary"]
        user_msg = f"""role: {rg['role']}
instructions: {rg['instructions']}
info: {rg['information']}
output_ex: {oe['output_example']}

missing: {ce['missing_items']}
all req: {rs['summary']}"""
        try:
            parsed = await parse_chat(system_msg, user_msg, CoverageFixOutput)
            state["role_guidance"] = {
                "role": parsed.role,
                "instructions": parsed.instructions,
                "information": parsed.information,
            }
            state["output_example"] = {"output_example": parsed.output_example}
            state["coverage_evaluation"] = {"is_complete": True, "missing_items": None}
        except Exception as e:
            print("[ERROR] fix_coverage:", e)
        return state

    async def finalize_system_prompt(state):
        system_msg = "JSON final system prompt -> system_prompt, reasoning"
        rg = state["role_guidance"]
        oe = state["output_example"]
        user_msg = f"""role: {rg['role']}
instructions: {rg['instructions']}
info: {rg['information']}
out_ex: {oe['output_example']}"""
        try:
            parsed = await parse_chat(system_msg, user_msg, FinalSystemPromptOutput)
            state["final_prompt"] = {
                "system_prompt": parsed.system_prompt,
                "user_prompt_template": "{{사용자 입력}}",
                "reasoning": parsed.reasoning,
            }
        except Exception as e:
            print("[ERROR] finalize_system_prompt:", e)
            state["final_prompt"] = {
                "system_prompt": "(오류)시스템프롬프트",
                "user_prompt_template": "{{사용자 입력}}",
                "reasoning": str(e),
            }
        return state

    ###### 유저 프롬프트 ######
    async def check_input_example(state):
        system_msg = "JSON -> has_example(bool), input_example(str?)"
        user_msg = state["user_intention"]
        try:
            parsed = await parse_chat(system_msg, user_msg, InputExampleEvaluation)
            state["input_example_evaluation"] = {
                "has_example": parsed.has_example,
                "input_example": parsed.input_example,
            }
        except Exception as e:
            print("[ERROR] check_input_example:", e)
            state["input_example_evaluation"] = {
                "has_example": False,
                "input_example": None,
            }
        return state

    async def summarize_intention(state):
        system_msg = "JSON -> summary: string[]"
        user_msg = state["user_intention"]
        try:
            parsed = await parse_chat(system_msg, user_msg, IntentionSummaryOutput)
            state["intention_summary"] = {"summary": parsed.summary}
        except Exception as e:
            print("[ERROR] summarize_intention:", e)
            state["intention_summary"] = {"summary": [f"(오류){e}"]}
        return state

    async def generate_user_prompt(state):
        system_msg = "JSON -> instructions, information, output_example"
        ie = state["input_example_evaluation"]
        user_msg = f"""의도: {state["user_intention"]}
has_example: {ie["has_example"]}
example: {ie["input_example"]}"""
        try:
            parsed = await parse_chat(system_msg, user_msg, UserPromptGenerationOutput)
            # 조합
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
            print("[ERROR] generate_user_prompt:", e)
            state["user_prompt_generation"] = {
                "user_prompt": f"(오류){e}",
                "instructions": f"(오류){e}",
                "information": f"(오류){e}",
                "output_example": f"(오류){e}",
            }
        return state

    async def evaluate_user_prompt_coverage(state):
        system_msg = "JSON -> is_complete(bool), missing_items?"
        up = state["user_prompt_generation"]["user_prompt"]
        it = state["intention_summary"]
        ie = state["input_example_evaluation"]
        user_msg = f"""user_prompt:
{up}

has_example: {ie["has_example"]}
input_example: {ie["input_example"]}

intention_summary: {it["summary"]}"""
        try:
            parsed = await parse_chat(
                system_msg, user_msg, UserPromptCoverageEvaluation
            )
            state["user_prompt_coverage"] = {
                "is_complete": parsed.is_complete,
                "missing_items": parsed.missing_items,
            }
        except Exception as e:
            print("[ERROR] evaluate_user_prompt_coverage:", e)
            state["user_prompt_coverage"] = {"is_complete": True, "missing_items": None}
        return state

    async def fix_user_prompt(state):
        system_msg = "JSON -> user_prompt"
        up = state["user_prompt_generation"]["user_prompt"]
        uc = state["user_prompt_coverage"]
        it = state["intention_summary"]
        user_msg = f"""현재 user_prompt: {up}
missing: {uc["missing_items"]}
all: {it["summary"]}"""
        try:
            parsed = await parse_chat(system_msg, user_msg, UserPromptFixOutput)
            state["user_prompt_generation"]["user_prompt"] = parsed.user_prompt
            state["user_prompt_coverage"] = {"is_complete": True, "missing_items": None}
        except Exception as e:
            print("[ERROR] fix_user_prompt:", e)
        return state

    async def finalize_user_prompt(state):
        system_msg = "JSON -> user_prompt_template, reasoning"
        up = state["user_prompt_generation"]["user_prompt"]
        try:
            parsed = await parse_chat(system_msg, up, FinalUserPromptOutput)
            state["final_prompt"] = {
                "system_prompt": "당신은 유용한 AI 어시스턴트입니다...",
                "user_prompt_template": parsed.user_prompt_template,
                "reasoning": parsed.reasoning,
            }
        except Exception as e:
            print("[ERROR] finalize_user_prompt:", e)
            state["final_prompt"] = {
                "system_prompt": "당신은 유용한 AI 어시스턴트입니다.",
                "user_prompt_template": f"(오류){e}",
                "reasoning": f"(오류){e}",
            }
        return state

    ################################
    # 실제 단계 수행
    ################################
    if user_sys_params == "system":
        # 1) 역할/예시/요약 병렬
        rg_task = asyncio.create_task(generate_role_guidance(state.copy()))
        oe_task = asyncio.create_task(generate_output_example(state.copy()))
        rs_task = asyncio.create_task(summarize_role(state.copy()))
        rg_result, oe_result, rs_result = await asyncio.gather(
            rg_task, oe_task, rs_task
        )

        state["role_guidance"] = rg_result["role_guidance"]
        state["output_example"] = oe_result["output_example"]
        state["role_summary"] = rs_result["role_summary"]

        # 충돌 루프
        state = await evaluate_conflicts(state)
        ccount = 0
        while state["conflict_evaluation"]["has_conflicts"]:
            ccount += 1
            if ccount > MAX_RETRY:
                break
            state = await resolve_conflicts(state)
            state = await evaluate_conflicts(state)

        # 커버리지 루프
        state = await evaluate_coverage(state)
        ccount = 0
        while not state["coverage_evaluation"]["is_complete"]:
            ccount += 1
            if ccount > MAX_RETRY:
                break
            state = await fix_coverage(state)
            state = await evaluate_coverage(state)

        # 최종
        state = await finalize_system_prompt(state)

    else:
        # user prompt path
        ie_task = asyncio.create_task(check_input_example(state.copy()))
        is_task = asyncio.create_task(summarize_intention(state.copy()))
        ie_res, is_res = await asyncio.gather(ie_task, is_task)
        state["input_example_evaluation"] = ie_res["input_example_evaluation"]
        state["intention_summary"] = is_res["intention_summary"]

        state = await generate_user_prompt(state)
        state = await evaluate_user_prompt_coverage(state)

        ccount = 0
        while not state["user_prompt_coverage"]["is_complete"]:
            ccount += 1
            if ccount > MAX_RETRY:
                break
            state = await fix_user_prompt(state)
            state = await evaluate_user_prompt_coverage(state)

        state = await finalize_user_prompt(state)

    return state


################################
# 4) 동기 래퍼
################################
def run_prompt_generation_agent(
    user_intention: str, user_sys_params: Optional[str] = None
) -> Dict[str, Any]:
    print("DEBUG: Enter run_prompt_generation_agent (SYNC WRAPPER)")
    result = asyncio.run(
        run_prompt_generation_agent_async(user_intention, user_sys_params)
    )
    print("DEBUG: Exit run_prompt_generation_agent (SYNC WRAPPER)")
    return result


################################
# 5) 스트리밍 함수
################################
def _run_agent_and_yield(user_intention: str, prompt_type: str):
    """단계별 결과를 yield"""
    yield f"'{prompt_type}' 프롬프트 생성을 위한 분석을 시작합니다...\n"
    try:
        result = run_prompt_generation_agent(user_intention, prompt_type)
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
    user_intention: str, prompt_type: Optional[str] = None
):
    """별도 스레드 + Queue로 스트리밍"""
    ctx = get_script_run_ctx()
    if prompt_type not in ["user", "system"]:
        prompt_type = "system"

    q = queue_module.Queue()

    def worker():
        add_script_run_ctx(thread=None, ctx=ctx)
        for chunk in _run_agent_and_yield(user_intention, prompt_type):
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
def generate_prompt_by_intention_streaming(user_intention: str, prompt_type: str):
    yield from run_prompt_generation_agent_streaming(user_intention, prompt_type)


def generate_prompt_by_intention(user_intention: str, prompt_type: str):
    result = run_prompt_generation_agent(user_intention, prompt_type)
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
