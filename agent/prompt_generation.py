# agent/prompt_generation.py

import os
import json
import asyncio
import time
from typing import List, Optional, AsyncGenerator, Generator

from dotenv import load_dotenv

from openai import AzureOpenAI
import pydantic
from pydantic import BaseModel

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
# 1) Pydantic Models
################################


class OpenAiResponseFormatGenerator(pydantic.json_schema.GenerateJsonSchema):
    def generate(self, schema, mode="validation"):
        json_schema = super().generate(schema, mode=mode)
        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": json_schema.pop("title"),
                "schema": json_schema,
            },
        }
        return json_schema


class StrictBaseModel(BaseModel):
    class Config:
        extra = "forbid"

    @classmethod
    def model_json_schema(cls, **kwargs):
        return super().model_json_schema(
            schema_generator=OpenAiResponseFormatGenerator, **kwargs
        )


class RoleGuidanceOutput(StrictBaseModel):
    role: str
    instructions: str
    information: str


class OutputExampleOutput(StrictBaseModel):
    output_example: str


class RoleSummaryOutput(StrictBaseModel):
    summary: List[str]


class ConflictEvaluationOutput(StrictBaseModel):
    has_conflicts: bool
    conflicts: Optional[List[str]] = None
    resolution: Optional[str] = None


class ConflictResolutionOutput(StrictBaseModel):
    role: str
    instructions: str
    information: str
    output_example: str


class CoverageEvaluationOutput(StrictBaseModel):
    is_complete: bool
    missing_items: Optional[List[str]] = None


class CoverageFixOutput(StrictBaseModel):
    role: str
    instructions: str
    information: str
    output_example: str


class FinalSystemPromptOutput(StrictBaseModel):
    system_prompt: str
    reasoning: str


################################
# 2) 노드 처리용 비동기 + 스트리밍 "Generator" 함수
################################


async def parse_chat_streaming_gen(
    system_content: str,
    user_content: str,
    pydantic_model,
    max_retries: int = 3,
) -> AsyncGenerator[str, None]:
    """
    실제 LLM 스트리밍을 async generator로 만들어,
      1) 토큰을 partial로 yield
      2) 전체가 끝나면 pydantic parse 시도 → 성공하면 최종 JSON을 __STEP_FINAL__ 형식으로 yield
         (파싱 실패 시 raw 출력 or 에러 메시지)

    호출 측에서는:
        async for output in parse_chat_streaming_gen(...):
            yield output
    """
    for attempt in range(max_retries):
        try:
            # Pydantic 모델로부터 JSON Schema 생성
            schema = pydantic_model.model_json_schema()

            # AzureOpenAI Chat Completions API 호출
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                response_format=schema,
                stream=True,
            )

            full_text = ""
            # 1) partial 스트리밍
            for chunk in response:
                if not hasattr(chunk, "choices") or len(chunk.choices) == 0:
                    continue
                delta_obj = chunk.choices[0].delta
                if hasattr(delta_obj, "content") and delta_obj.content is not None:
                    token = delta_obj.content
                    full_text += token
                    yield token  # partial 토큰

            # 2) 최종 parse 시도
            try:
                parsed_json = json.loads(full_text)
                # pydantic parse
                parsed_result = pydantic_model.parse_obj(parsed_json)
                # 성공하면 "__FINAL_PARSE__:{...}" 형식으로 내보냄
                yield f"__FINAL_PARSE__:{json.dumps(parsed_json)}"
                yield f"__PARSED_OBJECT__:{parsed_result}"
            except Exception as e:
                # parse 실패
                yield f"__FINAL_PARSE_ERROR__:{str(e)}"
            return

        except Exception as e:
            if attempt < max_retries - 1:
                # 재시도 전에 잠시 쉼
                await asyncio.sleep(1)
                continue
            else:
                yield f"__FINAL_PARSE_ERROR__:{str(e)}"
                return


################################
# 3) 시스템 프롬프트 생성 메인 로직
################################


async def run_prompt_generation_agent_async(
    user_intention: str,
    status_callback: Optional[callable] = None,
) -> AsyncGenerator[str, None]:
    """
    시스템 프롬프트 생성 프로세스:
      1. (병렬) Node1(역할/지시사항/정보 구분), Node2(출력 예시), Node3(역할 요약) → 결과가 모두 끝나면
         *1,2,3 결과 종합 정제*
      2. (순차) Node4(충돌 평가) → (필요 시 Node4.1 충돌 해결) → Pass될 때까지
      3. (순차) Node5(커버리지 평가) → (필요 시 Node5.1 보완) → Pass될 때까지
      4. (순차) Node6(최종 시스템 프롬프트 출력)

    각 노드 함수는 parse_chat_streaming_gen(...)을 호출해 partial 토큰을 yield 하고,
    parse 완료 시점에 "__FINAL_PARSE__:{...}"가 나옴. 이를 상위에서 잡아 state에 반영.
    """

    def update_status(msg: str, state="running"):
        if status_callback:
            status_callback(msg, state)
        print(f"[STATUS] {msg} ({state})")

    ################################
    # 노드 함수들 (각각 parse_chat_streaming_gen 호출)
    ################################

    async def node_system_1_role_guidance(state: dict):
        """
        1번 노드: 역할 / 지시사항 / 정보 구분
        """
        system_msg = '##Role: "I am a versatile system designed to assist in providing structured and accurate responses."\n##Instructions: "Interpret the user\'s inputs, maintain clarity, and adhere to professional communication standards."\n##Information: "Focus on understanding and addressing the given instructions effectively, providing informative and precise outcomes."\n\nPlease provide input in the format provided above to ensure the intended task is clear.'
        user_msg = state["user_intention"]
        async for part in parse_chat_streaming_gen(
            system_msg, user_msg, RoleGuidanceOutput
        ):
            yield f"__NODE1_PARTIAL__:{part}"

    async def node_system_2_output_example(state: dict):
        """
        2번 노드: 출력 예시 생성
        """
        system_msg = "You are an assistant specialized in generating 'example outputs' for specified roles or functionalities. Analyze the user's provided role or functionality description, understand the specified intent and scope, and craft a well-structured and relevant example output adhering to the defined role or task."
        user_msg = state["user_intention"]
        async for part in parse_chat_streaming_gen(
            system_msg, user_msg, OutputExampleOutput
        ):
            yield f"__NODE2_PARTIAL__:{part}"

    async def node_system_3_role_summary(state: dict):
        """
        3번 노드: 역할 요약
        """
        system_msg = "You refine, condense, and clarify user-provided definitions or instructions into concise and precise descriptions."
        user_msg = state["user_intention"]
        async for part in parse_chat_streaming_gen(
            system_msg, user_msg, RoleSummaryOutput
        ):
            yield f"__NODE3_PARTIAL__:{part}"

    async def node_system_4_conflict_evaluation(state: dict):
        """
        4번 노드: 충돌 평가
        """
        rg = state["role_guidance"]
        oe = state["output_example"]
        system_msg = "##role : Evaluator of system prompt coherence ##instructions : 1. Evaluate whether the specified roles, instructions, information, input examples, and output examples in the created system prompt have any contradictions, informational conflicts, or unclear directives. 2. Upon analyzing user-provided content categorized into roles, instructions, data, inputs, and outputs, assess for coherency, identifying any contradictions or inconsistencies among them. 3. Ensure a thorough evaluation process and resolve identified mismatches, ensuring overall harmony and accuracy in the provided content. ##information :  Focus on systematic evaluation of potential conflicts or unclear instructions between the components of a given system prompt, ensuring a coherent and seamless design for effective use. ##output_example :  Checking Alignment of Attributes\n\nTo ensure the entailed segments: ##role, ##instructions, ##information, ##input examples, and ##output examples are consistent and do not contradict one another, follow the procedure below:\n\n1. Analyze each segment to identify core requirements and intentions.\n2. Compare instructions against the provided role definition; ensure all tasks are feasible within the specified role.\n3. Validate information relevancy and harmony with the role's goals.\n4. Cross-reference examples (both input and output) with structured information and instructions for alignment.\n\nExample analysis:\n- Role: Data summarizer.\n- Instructions: Create concise summaries of provided text.\n- Information: Provides a maximum of 200-word summary.\n- Input Example: A 500-word news article.\n- Output Example: a 150-word summary.\n\nAfter evaluation, the segments all align, showing no contradiction. "
        user_msg = (
            f"(역할:{rg['role']}, 지시:{rg['instructions']}, "
            f"정보:{rg['information']}, 예시:{oe['output_example']})"
        )
        async for part in parse_chat_streaming_gen(
            system_msg, user_msg, ConflictEvaluationOutput
        ):
            yield f"__NODE4_PARTIAL__:{part}"

    async def node_system_4_1_conflict_resolution(state: dict):
        """
        4.1번 노드: 충돌 해결
        """
        ce = state["conflict_evaluation"]
        rg = state["role_guidance"]
        oe = state["output_example"]
        system_msg = "[시스템] Node4.1: 충돌 해결"
        user_msg = (
            f"충돌:{ce['conflicts']}, 해결안:{ce['resolution']}, "
            f"현재 역할:{rg['role']}, 지시:{rg['instructions']}, 정보:{rg['information']}, "
            f"출력 예시:{oe['output_example']}"
        )
        async for part in parse_chat_streaming_gen(
            system_msg, user_msg, ConflictResolutionOutput
        ):
            yield f"__NODE4_1_PARTIAL__:{part}"

    async def node_system_5_coverage_evaluation(state: dict):
        """
        5번 노드: 커버리지 평가
        """
        rg = state["role_guidance"]
        oe = state["output_example"]
        rs = state["role_summary"]
        system_msg = "##Role : You are an advanced data and functionality compliance review assistant. ##instructions : Given ##role, ##instructions, ##information, and the output example from processing data inputs, carefully review the criteria match of all parameters and ensure the generated result adheres entirely to the initial structure and specifications. Identify mismatches and correct them. ##information : The system should clarify the consistency between input-derived parameters and the generated outputs. Any inconsistency should be iteratively fixed until the output conforms satisfactorily to predefined standards. ##output_example : When validating against the specified roles and functionality, ensure all input data aligns with the outlined requirements and produce a compliance report summarizing the validation process."
        user_msg = (
            f"역할:{rg['role']}, 지시:{rg['instructions']}, 정보:{rg['information']}, "
            f"요약:{rs['summary']}, 예시:{oe['output_example']}"
        )
        async for part in parse_chat_streaming_gen(
            system_msg, user_msg, CoverageEvaluationOutput
        ):
            yield f"__NODE5_PARTIAL__:{part}"

    async def node_system_5_1_coverage_fix(state: dict):
        """
        5.1번 노드: 커버리지 보완
        """
        rg = state["role_guidance"]
        oe = state["output_example"]
        ce = state["coverage_evaluation"]
        rs = state["role_summary"]
        system_msg = "[시스템] Node5.1: 보완"
        user_msg = (
            f"누락:{ce['missing_items']}, 요약:{rs['summary']}, "
            f"현재 역할:{rg['role']}, 지시:{rg['instructions']}, 정보:{rg['information']}, "
            f"출력 예시:{oe['output_example']}"
        )
        async for part in parse_chat_streaming_gen(
            system_msg, user_msg, CoverageFixOutput
        ):
            yield f"__NODE5_1_PARTIAL__:{part}"

    async def node_system_6_final(state: dict):
        """
        6번 노드: 최종 시스템 프롬프트 생성
        """
        rg = state["role_guidance"]
        oe = state["output_example"]
        system_msg = "당신은 시스템 프롬프트 작성 전문가입니다. 유저 프롬프트에 전달된 정보를 빠짐없이 system_prompt 에 마크다운 형태로 작성하세요."
        user_msg = (
            f"역할:{rg['role']}, 지시:{rg['instructions']}, "
            f"정보:{rg['information']}, 예시:{oe['output_example']}"
        )
        async for part in parse_chat_streaming_gen(
            system_msg, user_msg, FinalSystemPromptOutput
        ):
            yield f"__NODE6_PARTIAL__:{part}"

    ################################
    # 상태 저장용 dict
    ################################
    state = {
        "user_intention": user_intention,  # 유저 입력
        "role_guidance": {
            "role": "",
            "instructions": "",
            "information": "",
        },  # 1번 노드 결과
        "output_example": {"output_example": ""},  # 2번 노드 결과
        "role_summary": {"summary": []},  # 3번 노드 결과
        "conflict_evaluation": {
            "has_conflicts": False,
            "conflicts": [],
            "resolution": "",
        },  # 4번 노드 결과
        "coverage_evaluation": {
            "is_complete": False,
            "missing_items": [],
        },  # 5번 노드 결과
        "final_prompt": {},  # 6번 노드 결과
    }

    update_status("시스템 프롬프트 생성을 시작합니다.", "info")

    ################################
    # 1) 1,2,3 번 노드를 병렬 수행 (round-robin 방식)
    ################################
    update_status("1,2,3번 노드 (병렬) 시작", "info")

    # 생성기 생성
    node1_gen = node_system_1_role_guidance(state.copy())
    node2_gen = node_system_2_output_example(state.copy())
    node3_gen = node_system_3_role_summary(state.copy())

    # 생성기와 관련 상태
    generators = [
        (node1_gen, "node1", False, ""),
        (node2_gen, "node2", False, ""),
        (node3_gen, "node3", False, ""),
    ]

    parsed_results = {"node1": None, "node2": None, "node3": None}

    try:
        # 모든 생성기가 완료될 때까지 라운드 로빈 방식으로 실행
        while not all(done for _, _, done, _ in generators):
            for i, (gen, name, done, buffer) in enumerate(generators):
                if done:
                    continue

                try:
                    # 현재 생성기에서 다음 부분 가져오기
                    part = await asyncio.wait_for(gen.asend(None), timeout=0.1)
                    yield part

                    # 버퍼 업데이트
                    new_buffer = buffer + part
                    generators[i] = (gen, name, done, new_buffer)

                    # FINAL_PARSE 확인
                    if "__FINAL_PARSE__" in new_buffer:
                        final_parse_index = new_buffer.find("__FINAL_PARSE__")
                        json_str = new_buffer[final_parse_index:].replace(
                            "__FINAL_PARSE__:", ""
                        )
                        parsed_results[name] = json.loads(json_str)
                        generators[i] = (gen, name, True, new_buffer)

                    elif "__FINAL_PARSE_ERROR__" in new_buffer:
                        generators[i] = (gen, name, True, new_buffer)

                except asyncio.TimeoutError:
                    # 타임아웃 발생 시 다음 생성기로 넘어감
                    continue
                except StopAsyncIteration:
                    # 생성기 완료
                    generators[i] = (gen, name, True, buffer)
                except Exception as e:
                    # 오류 처리
                    yield f"__{name.upper()}_ERROR__: {str(e)}"
                    generators[i] = (gen, name, True, buffer)

    finally:
        # 모든 생성기를 안전하게 닫음
        for gen, name, _, _ in generators:
            try:
                await gen.aclose()
            except Exception:
                pass

    # 결과 state 반영
    if parsed_results["node1"]:
        state["role_guidance"]["role"] = parsed_results["node1"].get("role", "")
        state["role_guidance"]["instructions"] = parsed_results["node1"].get(
            "instructions", ""
        )
        state["role_guidance"]["information"] = parsed_results["node1"].get(
            "information", ""
        )

    if parsed_results["node2"]:
        state["output_example"]["output_example"] = parsed_results["node2"].get(
            "output_example", ""
        )

    if parsed_results["node3"]:
        state["role_summary"]["summary"] = parsed_results["node3"].get("summary", [])

    update_status("1,2,3번 노드 (병렬) 완료", "info")
    time.sleep(2)

    ################################
    # 2) 노드4 (충돌 평가)
    ################################
    update_status("4번 노드 (충돌 평가) 시작", "info")
    node4_gen = node_system_4_conflict_evaluation(state)
    node4_parsed_json = None

    async for part in node4_gen:
        yield part
        if part.startswith("__FINAL_PARSE__"):
            raw_json = part.replace("__FINAL_PARSE__:", "")
            node4_parsed_json = json.loads(raw_json)
        elif part.startswith("__PARSED_OBJECT__"):
            pass

    update_status("4번 노드 완료", "info")
    time.sleep(2)

    if node4_parsed_json:
        state["conflict_evaluation"]["has_conflicts"] = node4_parsed_json.get(
            "has_conflicts", False
        )
        state["conflict_evaluation"]["conflicts"] = node4_parsed_json.get(
            "conflicts", []
        )
        state["conflict_evaluation"]["resolution"] = node4_parsed_json.get(
            "resolution", ""
        )

    # 만약 충돌이 있다면 해결 노드 4.1 -> 다시 4번 노드를 반복 등 로직이 가능하나,
    # 여기서는 단순 예시로 1회만 시연하고 pass 처리 생략

    ################################
    # 3) 노드5 (커버리지 평가)
    ################################
    update_status("5번 노드 (커버리지 평가) 시작", "info")
    node5_gen = node_system_5_coverage_evaluation(state)
    node5_parsed_json = None

    async for part in node5_gen:
        yield part
        if part.startswith("__FINAL_PARSE__"):
            raw_json = part.replace("__FINAL_PARSE__:", "")
            node5_parsed_json = json.loads(raw_json)
        elif part.startswith("__PARSED_OBJECT__"):
            pass

    update_status("5번 노드 완료", "info")
    time.sleep(2)
    if node5_parsed_json:
        state["coverage_evaluation"]["is_complete"] = node5_parsed_json.get(
            "is_complete", False
        )
        state["coverage_evaluation"]["missing_items"] = node5_parsed_json.get(
            "missing_items", []
        )

    # 만약 누락된 사항 있다면 5.1 → 재평가 등 로직

    update_status("5.1 (커버리지 보완)", "info")
    node5_1_gen = node_system_5_1_coverage_fix(state)
    node5_1_parsed_json = None

    async for part in node5_1_gen:
        yield part
        if part.startswith("__FINAL_PARSE__"):
            raw_json = part.replace("__FINAL_PARSE__:", "")
            node5_1_parsed_json = json.loads(raw_json)
        elif part.startswith("__PARSED_OBJECT__"):
            pass

    update_status("5.1번 노드 완료", "info")
    time.sleep(2)

    if node5_1_parsed_json:
        # 보완된 결과를 state에 반영(필요시)
        state["role_guidance"]["role"] = node5_1_parsed_json.get("role", "")
        state["role_guidance"]["instructions"] = node5_1_parsed_json.get(
            "instructions", ""
        )
        state["role_guidance"]["information"] = node5_1_parsed_json.get(
            "information", ""
        )
        state["output_example"]["output_example"] = node5_1_parsed_json.get(
            "output_example", ""
        )

    ################################
    # 4) 노드6 (최종)
    ################################
    update_status("6번 노드 (최종) 시작", "info")
    node6_gen = node_system_6_final(state)
    node6_parsed_json = None

    async for part in node6_gen:
        yield part
        if part.startswith("__FINAL_PARSE__"):
            raw_json = part.replace("__FINAL_PARSE__:", "")
            node6_parsed_json = json.loads(raw_json)
        elif part.startswith("__PARSED_OBJECT__"):
            pass

    update_status("6번 노드 완료", "info")
    time.sleep(2)

    if node6_parsed_json:
        state["final_prompt"] = node6_parsed_json

    # 마지막 결과 (예: "__RESULT_DATA__:{...}")
    final_data = {
        "final_prompt": {
            "system_prompt": state["final_prompt"].get("system_prompt", ""),
            "reasoning": state["final_prompt"].get("reasoning", ""),
        }
    }
    yield f"__RESULT_DATA__:{json.dumps(final_data)}"


def run_prompt_generation_agent_streaming(
    user_intention: str, status_callback: Optional[callable] = None
) -> Generator[str, None, None]:
    """
    동기 -> async generator consume
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async_gen = run_prompt_generation_agent_async(user_intention, status_callback)

    async def consume():
        async for token in async_gen:
            yield token

    gen = consume()

    try:
        while True:
            token = loop.run_until_complete(anext(gen))
            yield token
    except StopAsyncIteration:
        pass
    finally:
        loop.close()


def generate_system_prompt_by_intention_streaming(
    user_intention: str, status_callback: Optional[callable] = None
) -> Generator[str, None, None]:
    """
    외부에서 직접 호출할 수 있는 진입점.
    예시:
        for token in generate_system_prompt_by_intention_streaming("..."):
            # token 별로 스트리밍
            ...
    """
    for token in run_prompt_generation_agent_streaming(user_intention, status_callback):
        yield token
