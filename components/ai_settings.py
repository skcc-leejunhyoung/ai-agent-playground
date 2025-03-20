# components/ai_settings.py

import streamlit as st
from st_diff_viewer import diff_viewer
from streamlit_monaco import st_monaco
import uuid
import time
import json

from model import stream_chat
from agent.eval_tools import run_evaluation
from database import (
    add_system_prompt,
    get_system_prompts,
    add_user_prompt,
    get_user_prompts,
    add_model,
    get_models,
    add_result,
    get_results_by_project,
    update_system_prompt,
    update_user_prompt,
    update_project_excluded_prompts,
    get_project_excluded_prompts,
)
from agent.prompt_generation import generate_system_prompt_by_intention_streaming

##########


one_dark_pro_styles = {
    "variables": {
        "dark": {
            "diffViewerBackground": "#282C34",
            "diffViewerColor": "#ABB2BF",
            "addedBackground": "#31392b",
            "addedColor": "#98C379",
            "removedBackground": "#3b282c",
            "removedColor": "#E06C75",
            "wordAddedBackground": "#264F78",
            "wordRemovedBackground": "#8B0000",
            "addedGutterBackground": "#31392b",
            "removedGutterBackground": "#3b282c",
        }
    }
}


##########


def generate_uuid():
    return str(uuid.uuid4().int)[:18]


def ai_settings_ui(project_id):
    """
    Playground의 왼쪽 UI 컴포넌트를 담당하는 함수입니다.
    project_id를 입력하면 시스템, 유저 프롬프트, 모델 선택 등 모든 설정 UI를 렌더링하고,
    실행 버튼을 클릭하면 결과 데이터를 session_state["results"]에 저장합니다.
    """

    try:
        excluded_system_prompts, excluded_user_prompts = get_project_excluded_prompts(
            project_id
        )
    except Exception as e:
        print(f"[ERROR] 제외된 프롬프트 목록 조회 실패: {e}")
        excluded_system_prompts, excluded_user_prompts = [], []

    if "excluded_system_prompt_ids" not in st.session_state:
        st.session_state["excluded_system_prompt_ids"] = excluded_system_prompts
    if "excluded_user_prompt_ids" not in st.session_state:
        st.session_state["excluded_user_prompt_ids"] = excluded_user_prompts

    def update_excluded_prompts():
        try:
            update_project_excluded_prompts(
                project_id,
                st.session_state.get("excluded_system_prompt_ids", []),
                st.session_state.get("excluded_user_prompt_ids", []),
            )
        except Exception as e:
            print(f"[ERROR] 제외된 프롬프트 목록 업데이트 실패: {e}")
            st.toast("제외된 프롬프트 목록 저장 중 오류가 발생했습니다.", icon="⚠️")

    with st.container():
        header_col1, header_col2 = st.columns([2, 1], vertical_alignment="center")
        with header_col1:
            st.write("_프롬프팅 Agent를 결합한 playground_")

        with header_col2:
            model_count = 2 if st.session_state.get("model_toggle") else 1
            system_count = (
                len(st.session_state.get("system_prompts", ["dummy"]))
                if st.session_state.get("system_toggle")
                else 1
            )
            user_count = (
                len(st.session_state.get("user_prompts", ["dummy"]))
                if st.session_state.get("user_toggle")
                else 1
            )
            total_runs = model_count * system_count * user_count
            execute_label = f"**:red[{total_runs} | Execute]**"
            execute_button = st.button(execute_label, use_container_width=True)

        ##########

        with st.expander(
            "System Prompt", expanded=st.session_state.get("system_expanded", False)
        ):
            system_compare_toggle = st.toggle(
                "다중 System Prompt 활성화", key="system_toggle"
            )
            if system_compare_toggle:
                btn_generate_sys = False
                db_system_prompts = get_system_prompts(project_id)
                if "excluded_system_prompt_ids" not in st.session_state:
                    st.session_state["excluded_system_prompt_ids"] = []
                st.session_state["system_prompts"] = [
                    {"id": row["id"], "prompt": row["prompt"]}
                    for row in db_system_prompts
                    if row["id"] not in st.session_state["excluded_system_prompt_ids"]
                ]
                if "system_select_1_idx" not in st.session_state:
                    st.session_state["system_select_1_idx"] = 0
                if "system_select_2_idx" not in st.session_state:
                    st.session_state["system_select_2_idx"] = 1
                prompt_count = len(st.session_state["system_prompts"])
                available_idx_for_1 = list(range(prompt_count))
                available_idx_for_2 = list(range(prompt_count))
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        with st.container():
                            c1, c2 = st.columns([10, 1])
                            with c1:
                                idx_1 = st.selectbox(
                                    "sys 1 선택",
                                    options=available_idx_for_1,
                                    index=st.session_state["system_select_1_idx"],
                                    format_func=lambda x: f"{x+1}. {st.session_state['system_prompts'][x]['prompt'][:30]}...",
                                    key="system_select_1",
                                    label_visibility="collapsed",
                                )
                                if idx_1 == st.session_state["system_select_2_idx"]:
                                    temp = st.session_state["system_select_1_idx"]
                                    st.session_state["system_select_1_idx"] = (
                                        st.session_state["system_select_2_idx"]
                                    )
                                    st.session_state["system_select_2_idx"] = temp
                                    st.rerun()
                                else:
                                    st.session_state["system_select_1_idx"] = idx_1
                            with c2:
                                btn_update_sys_1 = st.button(
                                    ":material/save:",
                                    type="tertiary",
                                    use_container_width=True,
                                    key=f"update_sys_{idx_1}_top",
                                )
                            editor_1 = st_monaco(
                                value=st.session_state["system_prompts"][idx_1][
                                    "prompt"
                                ],
                                language="markdown",
                                height="300px",
                                theme="vs-dark",
                            )
                            if editor_1 is not None:
                                st.session_state["system_prompts"][idx_1][
                                    "prompt"
                                ] = editor_1
                            if btn_update_sys_1:
                                update_system_prompt(
                                    st.session_state["system_prompts"][idx_1]["id"],
                                    st.session_state["system_prompts"][idx_1]["prompt"],
                                )
                                st.toast("Left System Prompt updated")
                                time.sleep(0.7)
                                st.rerun()
                    with col2:
                        with st.container():
                            c1, c2, c3, c4 = st.columns([9, 1, 1, 1])
                            with c1:
                                idx_2 = st.selectbox(
                                    "sys 2 선택",
                                    options=available_idx_for_2,
                                    index=st.session_state["system_select_2_idx"],
                                    format_func=lambda x: f"{x+1}. {st.session_state['system_prompts'][x]['prompt'][:30]}...",
                                    key="system_select_2",
                                    label_visibility="collapsed",
                                )
                                if idx_2 == st.session_state["system_select_1_idx"]:
                                    temp = st.session_state["system_select_2_idx"]
                                    st.session_state["system_select_2_idx"] = (
                                        st.session_state["system_select_1_idx"]
                                    )
                                    st.session_state["system_select_1_idx"] = temp
                                    st.rerun()
                                else:
                                    st.session_state["system_select_2_idx"] = idx_2
                            with c2:
                                editor_2_btn = st.button(
                                    ":material/save:",
                                    type="tertiary",
                                    use_container_width=True,
                                    key=f"update_sys_{idx_2}_top",
                                )
                            with c3:
                                if st.button(
                                    ":material/add:",
                                    type="tertiary",
                                    use_container_width=True,
                                    key="add_system_prompt",
                                ):
                                    new_prompt = (
                                        f"You are helpful assistant! #{generate_uuid()}"
                                    )
                                    new_id = add_system_prompt(new_prompt, project_id)
                                    st.session_state["system_prompts"].append(
                                        {"id": new_id, "prompt": new_prompt}
                                    )
                                    st.session_state["system_select_2_idx"] = (
                                        len(st.session_state["system_prompts"]) - 1
                                    )
                                    st.rerun()
                            with c4:
                                if st.button(
                                    ":material/remove:",
                                    type="tertiary",
                                    use_container_width=True,
                                    key="remove_system_prompt",
                                ):
                                    remove_idx = st.session_state["system_select_2_idx"]
                                    if len(st.session_state["system_prompts"]) <= 2:
                                        st.toast(
                                            "시스템 프롬프트는 최소 2개 필요합니다.",
                                            icon="⚠️",
                                        )
                                    else:
                                        removed = st.session_state[
                                            "system_prompts"
                                        ].pop(remove_idx)
                                        st.session_state.setdefault(
                                            "excluded_system_prompt_ids", []
                                        ).append(removed["id"])
                                        new_len = len(
                                            st.session_state["system_prompts"]
                                        )
                                        st.session_state["system_select_2_idx"] = max(
                                            0, min(remove_idx, new_len - 1)
                                        )
                                        st.session_state["system_select_1_idx"] = max(
                                            0,
                                            min(
                                                st.session_state["system_select_1_idx"],
                                                new_len - 1,
                                            ),
                                        )
                                        if (
                                            new_len > 1
                                            and st.session_state["system_select_1_idx"]
                                            == st.session_state["system_select_2_idx"]
                                        ):
                                            st.session_state["system_select_1_idx"] = (
                                                st.session_state["system_select_2_idx"]
                                                + 1
                                            ) % new_len
                                        st.rerun()
                            editor_2 = st_monaco(
                                value=st.session_state["system_prompts"][idx_2][
                                    "prompt"
                                ],
                                language="markdown",
                                height="300px",
                                theme="vs-dark",
                            )
                            if editor_2 is not None:
                                st.session_state["system_prompts"][idx_2][
                                    "prompt"
                                ] = editor_2
                            if editor_2_btn:
                                update_system_prompt(
                                    st.session_state["system_prompts"][idx_2]["id"],
                                    st.session_state["system_prompts"][idx_2]["prompt"],
                                )
                                st.toast("Right System Prompt updated")
                                time.sleep(0.7)
                                st.rerun()
                    split_view_sys = st.toggle(
                        "Split View", value=True, key="system_split_view"
                    )
                    diff_viewer(
                        st.session_state["system_prompts"][idx_1]["prompt"],
                        st.session_state["system_prompts"][idx_2]["prompt"],
                        split_view=split_view_sys,
                        use_dark_theme=True,
                        styles=one_dark_pro_styles,
                    )
            else:
                db_system_prompts = get_system_prompts(project_id)
                if "excluded_system_prompt_ids" not in st.session_state:
                    st.session_state["excluded_system_prompt_ids"] = []
                st.session_state["system_prompts"] = [
                    {"id": row["id"], "prompt": row["prompt"]}
                    for row in db_system_prompts
                    if row["id"] not in st.session_state["excluded_system_prompt_ids"]
                ]
                if "system_single_idx" not in st.session_state:
                    st.session_state["system_single_idx"] = 0
                prompt_count = len(st.session_state["system_prompts"])

                if (
                    prompt_count > 0
                    and st.session_state["system_single_idx"] >= prompt_count
                ):
                    st.session_state["system_single_idx"] = 0

                c1, c2, c3, c4, c5 = st.columns([9, 1, 1, 1, 1])
                with c1:
                    if prompt_count > 0:
                        idx = st.selectbox(
                            "System Prompt 선택",
                            options=list(range(prompt_count)),
                            index=st.session_state["system_single_idx"],
                            format_func=lambda x: f"{x+1}. {st.session_state['system_prompts'][x]['prompt'][:30]}...",
                            key="system_single_select",
                            label_visibility="collapsed",
                        )
                        st.session_state["system_single_idx"] = idx
                    else:
                        st.info("사용 가능한 System Prompt가 없습니다.")
                        idx = 0

                with c2:
                    if prompt_count > 0:
                        btn_update_sys = st.button(
                            ":material/save:",
                            key=f"update_system_single_top_{idx}",
                            type="tertiary",
                            use_container_width=True,
                        )

                with c3:
                    btn_generate_sys = st.button(
                        ":material/Cognition:",
                        key=f"generate_system_single_top_{idx}",
                        type="tertiary",
                        use_container_width=True,
                    )

                with c4:
                    if st.button(
                        ":material/add:",
                        type="tertiary",
                        use_container_width=True,
                        key="add_system_prompt_single",
                    ):
                        new_prompt = f"You are helpful assistant! #{generate_uuid()}"
                        new_id = add_system_prompt(new_prompt, project_id)
                        st.session_state["system_prompts"].append(
                            {"id": new_id, "prompt": new_prompt}
                        )
                        st.session_state["system_single_idx"] = (
                            len(st.session_state["system_prompts"]) - 1
                        )
                        st.rerun()

                with c5:
                    if prompt_count > 0 and st.button(
                        ":material/remove:",
                        type="tertiary",
                        use_container_width=True,
                        key="remove_system_prompt_single",
                    ):
                        if len(st.session_state["system_prompts"]) <= 2:
                            st.toast(
                                "최소 2개의 시스템 프롬프트가 필요합니다.", icon="⚠️"
                            )
                        else:
                            removed = st.session_state["system_prompts"].pop(idx)
                            st.session_state.setdefault(
                                "excluded_system_prompt_ids", []
                            ).append(removed["id"])
                            try:
                                update_excluded_prompts()
                            except Exception as e:
                                print(f"[ERROR] 제외된 프롬프트 저장 실패: {e}")

                            if st.session_state["system_single_idx"] >= len(
                                st.session_state["system_prompts"]
                            ):
                                st.session_state["system_single_idx"] = max(
                                    0, len(st.session_state["system_prompts"]) - 1
                                )
                            st.rerun()

                if prompt_count > 0:
                    editor_single = st_monaco(
                        value=st.session_state["system_prompts"][idx]["prompt"],
                        language="markdown",
                        height="300px",
                        theme="vs-dark",
                    )
                    if editor_single is not None:
                        st.session_state["system_prompts"][idx][
                            "prompt"
                        ] = editor_single
                    if btn_update_sys:
                        prompt_id = st.session_state["system_prompts"][idx]["id"]
                        prompt_text = st.session_state["system_prompts"][idx]["prompt"]
                        if prompt_id:
                            update_system_prompt(prompt_id, prompt_text)
                            st.toast("System Prompt updated")
                            time.sleep(0.7)
                            st.rerun()
                        else:
                            st.warning("저장할 수 없는 프롬프트입니다.", icon="⚠️")
                    system_single_text = (
                        st.session_state["system_prompts"][idx]["prompt"]
                        if prompt_count > 0
                        else ""
                    )
                else:
                    system_single_text = ""

        ##########

        with st.expander(
            "User Prompt", expanded=st.session_state.get("user_expanded", True)
        ):
            user_compare_toggle = st.toggle(
                "다중 User Prompt 활성화", key="user_toggle"
            )
            if user_compare_toggle:
                db_user_prompts = get_user_prompts(project_id)
                if "excluded_user_prompt_ids" not in st.session_state:
                    st.session_state["excluded_user_prompt_ids"] = []
                st.session_state["user_prompts"] = [
                    {
                        "id": row["id"],
                        "prompt": row["prompt"],
                        "eval_method": row["eval_method"],
                        "eval_keyword": row["eval_keyword"],
                    }
                    for row in db_user_prompts
                    if row["id"] not in st.session_state["excluded_user_prompt_ids"]
                ]
                if "user_select_1_idx" not in st.session_state:
                    st.session_state["user_select_1_idx"] = 0
                if "user_select_2_idx" not in st.session_state:
                    st.session_state["user_select_2_idx"] = 1
                prompt_count = len(st.session_state["user_prompts"])
                available_idx_for_1 = list(range(prompt_count))
                available_idx_for_2 = list(range(prompt_count))
                col1, col2 = st.columns(2)
                with col1:
                    c1, c2 = st.columns([10, 1])
                    with c1:
                        idx_1 = st.selectbox(
                            "usr 1 선택",
                            options=available_idx_for_1,
                            index=st.session_state["user_select_1_idx"],
                            format_func=lambda x: f"{x+1}. {st.session_state['user_prompts'][x]['prompt'][:30]}...",
                            key="user_select_1",
                            label_visibility="collapsed",
                        )
                        if idx_1 == st.session_state["user_select_2_idx"]:
                            temp = st.session_state["user_select_1_idx"]
                            st.session_state["user_select_1_idx"] = st.session_state[
                                "user_select_2_idx"
                            ]
                            st.session_state["user_select_2_idx"] = temp
                            st.rerun()
                        else:
                            st.session_state["user_select_1_idx"] = idx_1
                    with c2:
                        btn_update_usr_1 = st.button(
                            ":material/save:",
                            type="tertiary",
                            use_container_width=True,
                            key=f"update_usr_{idx_1}_top",
                        )

                    editor_1 = st_monaco(
                        value=st.session_state["user_prompts"][idx_1]["prompt"],
                        language="markdown",
                        height="300px",
                        theme="vs-dark",
                    )
                    if editor_1 is not None:
                        st.session_state["user_prompts"][idx_1]["prompt"] = editor_1

                with col2:
                    c1, c2, c3, c4 = st.columns([9, 1, 1, 1])
                    with c1:
                        idx_2 = st.selectbox(
                            "usr 2 선택",
                            options=available_idx_for_2,
                            index=st.session_state["user_select_2_idx"],
                            format_func=lambda x: f"{x+1}. {st.session_state['user_prompts'][x]['prompt'][:30]}...",
                            key="user_select_2",
                            label_visibility="collapsed",
                        )
                        if idx_2 == st.session_state["user_select_1_idx"]:
                            temp = st.session_state["user_select_2_idx"]
                            st.session_state["user_select_2_idx"] = st.session_state[
                                "user_select_1_idx"
                            ]
                            st.session_state["user_select_1_idx"] = temp
                            st.rerun()
                        else:
                            st.session_state["user_select_2_idx"] = idx_2
                    with c2:
                        btn_update_usr_2 = st.button(
                            ":material/save:",
                            type="tertiary",
                            use_container_width=True,
                            key=f"update_usr_{idx_2}_top",
                        )
                    with c3:
                        if st.button(
                            ":material/add:",
                            type="tertiary",
                            use_container_width=True,
                            key="add_user_prompt",
                        ):
                            new_prompt = f"User Prompt #{generate_uuid()}"
                            add_user_prompt(new_prompt, project_id)
                            st.session_state["user_prompts"].append(
                                {"id": None, "prompt": new_prompt}
                            )
                            st.session_state["user_select_2_idx"] = (
                                len(st.session_state["user_prompts"]) - 1
                            )
                            st.rerun()
                    with c4:
                        if st.button(
                            ":material/remove:",
                            type="tertiary",
                            use_container_width=True,
                            key="remove_user_prompt",
                        ):
                            remove_idx = st.session_state["user_select_2_idx"]
                            if len(st.session_state["user_prompts"]) <= 2:
                                st.toast(
                                    "유저 프롬프트는 최소 2개 필요합니다.", icon="⚠️"
                                )
                            else:
                                removed = st.session_state["user_prompts"].pop(
                                    remove_idx
                                )
                                st.session_state.setdefault(
                                    "excluded_user_prompt_ids", []
                                ).append(removed["id"])
                                new_len = len(st.session_state["user_prompts"])
                                st.session_state["user_select_2_idx"] = max(
                                    0, min(remove_idx, new_len - 1)
                                )
                                st.session_state["user_select_1_idx"] = max(
                                    0,
                                    min(
                                        st.session_state["user_select_1_idx"],
                                        new_len - 1,
                                    ),
                                )
                                if (
                                    new_len > 1
                                    and st.session_state["user_select_1_idx"]
                                    == st.session_state["user_select_2_idx"]
                                ):
                                    st.session_state["user_select_1_idx"] = (
                                        st.session_state["user_select_2_idx"] + 1
                                    ) % new_len
                                st.rerun()
                    editor_2 = st_monaco(
                        value=st.session_state["user_prompts"][idx_2]["prompt"],
                        language="markdown",
                        height="300px",
                        theme="vs-dark",
                    )
                    if editor_2 is not None:
                        st.session_state["user_prompts"][idx_2]["prompt"] = editor_2

                split_view_usr = st.toggle(
                    "Split View", value=True, key="user_split_view"
                )
                diff_viewer(
                    st.session_state["user_prompts"][idx_1]["prompt"],
                    st.session_state["user_prompts"][idx_2]["prompt"],
                    split_view=split_view_usr,
                    use_dark_theme=True,
                    styles=one_dark_pro_styles,
                )
                col1, col2 = st.columns(2)
                with col1:
                    method_options = ["pass", "rule"]
                    current_method = (
                        st.session_state["user_prompts"][idx_1].get("eval_method")
                        or "pass"
                    )
                    current_keyword = (
                        st.session_state["user_prompts"][idx_1].get("eval_keyword")
                        or ""
                    )
                    eval_method_1 = st.segmented_control(
                        "평가 방법 선택 (eval_method)",
                        options=method_options,
                        default=(
                            current_method
                            if current_method in method_options
                            else method_options[0]
                        ),
                        key=f"user_eval_method_select_1_{idx_1}",
                        label_visibility="collapsed",
                    )
                    st.session_state["user_prompts"][idx_1][
                        "eval_method"
                    ] = eval_method_1
                    if eval_method_1 == "rule":
                        eval_keyword_1 = st.text_area(
                            "평가 keyword 입력",
                            height=68,
                            value=current_keyword,
                            key=f"user_eval_keyword_1_{idx_1}",
                            label_visibility="collapsed",
                        )
                        st.session_state["user_prompts"][idx_1][
                            "eval_keyword"
                        ] = eval_keyword_1

                    if btn_update_usr_1:
                        update_user_prompt(
                            st.session_state["user_prompts"][idx_1]["id"],
                            new_prompt=st.session_state["user_prompts"][idx_1][
                                "prompt"
                            ],
                            eval_method=st.session_state["user_prompts"][idx_1][
                                "eval_method"
                            ],
                            eval_keyword=st.session_state["user_prompts"][idx_1][
                                "eval_keyword"
                            ],
                        )
                        st.toast("Left User Prompt updated")
                        time.sleep(0.7)
                        st.rerun()
                with col2:
                    method_options = ["pass", "rule"]
                    current_method = (
                        st.session_state["user_prompts"][idx_2].get("eval_method")
                        or "pass"
                    )
                    current_keyword = (
                        st.session_state["user_prompts"][idx_2].get("eval_keyword")
                        or ""
                    )
                    eval_method_2 = st.segmented_control(
                        "평가 방법 선택 (eval_method)",
                        options=method_options,
                        default=(
                            current_method
                            if current_method in method_options
                            else method_options[0]
                        ),
                        key=f"user_eval_method_select_2_{idx_2}",
                        label_visibility="collapsed",
                    )
                    st.session_state["user_prompts"][idx_2][
                        "eval_method"
                    ] = eval_method_2
                    if eval_method_2 == "rule":
                        eval_keyword_2 = st.text_area(
                            "평가 keyword 입력",
                            value=current_keyword,
                            height=68,
                            key=f"user_eval_keyword_2_{idx_2}",
                            label_visibility="collapsed",
                        )
                        st.session_state["user_prompts"][idx_2][
                            "eval_keyword"
                        ] = eval_keyword_2

                    if btn_update_usr_2:
                        update_user_prompt(
                            st.session_state["user_prompts"][idx_2]["id"],
                            new_prompt=st.session_state["user_prompts"][idx_2][
                                "prompt"
                            ],
                            eval_method=st.session_state["user_prompts"][idx_2][
                                "eval_method"
                            ],
                            eval_keyword=st.session_state["user_prompts"][idx_2][
                                "eval_keyword"
                            ],
                        )
                        st.toast("Right User Prompt updated")
                        time.sleep(0.7)
                        st.rerun()
            else:
                db_user_prompts = get_user_prompts(project_id)
                if "excluded_user_prompt_ids" not in st.session_state:
                    st.session_state["excluded_user_prompt_ids"] = []
                st.session_state["user_prompts"] = [
                    {
                        "id": row["id"],
                        "prompt": row["prompt"],
                        "eval_method": row["eval_method"],
                        "eval_keyword": row["eval_keyword"],
                    }
                    for row in db_user_prompts
                    if row["id"] not in st.session_state["excluded_user_prompt_ids"]
                ]
                if "user_single_idx" not in st.session_state:
                    st.session_state["user_single_idx"] = 0
                prompt_count = len(st.session_state["user_prompts"])

                if (
                    prompt_count > 0
                    and st.session_state["user_single_idx"] >= prompt_count
                ):
                    st.session_state["user_single_idx"] = 0

                c1, c2, c3, c4 = st.columns([9, 1, 1, 1])
                with c1:
                    if prompt_count > 0:
                        idx = st.selectbox(
                            "User Prompt 선택",
                            options=list(range(prompt_count)),
                            index=st.session_state["user_single_idx"],
                            format_func=lambda x: f"{x+1}. {st.session_state['user_prompts'][x]['prompt'][:30]}...",
                            key="user_single_select",
                            label_visibility="collapsed",
                        )
                        st.session_state["user_single_idx"] = idx
                    else:
                        st.info("사용 가능한 User Prompt가 없습니다.")
                        idx = 0

                with c2:
                    if prompt_count > 0:
                        btn_update_usr = st.button(
                            ":material/save:",
                            key=f"update_user_single_top_{idx}",
                            type="tertiary",
                            use_container_width=True,
                        )

                with c3:
                    if st.button(
                        ":material/add:",
                        type="tertiary",
                        use_container_width=True,
                        key="add_user_prompt_single",
                    ):
                        new_prompt = f"User Prompt #{generate_uuid()}"
                        new_id = add_user_prompt(new_prompt, project_id)
                        st.session_state["user_prompts"].append(
                            {
                                "id": new_id,
                                "prompt": new_prompt,
                                "eval_method": "pass",
                                "eval_keyword": "",
                            }
                        )
                        st.session_state["user_single_idx"] = (
                            len(st.session_state["user_prompts"]) - 1
                        )
                        st.rerun()

                with c4:
                    if prompt_count > 0 and st.button(
                        ":material/remove:",
                        type="tertiary",
                        use_container_width=True,
                        key="remove_user_prompt_single",
                    ):
                        if len(st.session_state["user_prompts"]) <= 2:
                            st.toast("최소 2개의 유저 프롬프트가 필요합니다.", icon="⚠️")
                        else:
                            removed = st.session_state["user_prompts"].pop(idx)
                            st.session_state.setdefault(
                                "excluded_user_prompt_ids", []
                            ).append(removed["id"])
                            try:
                                update_excluded_prompts()
                            except Exception as e:
                                print(f"[ERROR] 제외된 프롬프트 저장 실패: {e}")

                            if st.session_state["user_single_idx"] >= len(
                                st.session_state["user_prompts"]
                            ):
                                st.session_state["user_single_idx"] = max(
                                    0, len(st.session_state["user_prompts"]) - 1
                                )
                            st.rerun()

                if prompt_count > 0:
                    editor_single = st_monaco(
                        value=st.session_state["user_prompts"][idx]["prompt"],
                        language="markdown",
                        height="300px",
                        theme="vs-dark",
                    )
                    if editor_single is not None:
                        st.session_state["user_prompts"][idx]["prompt"] = editor_single

                    method_options = ["pass", "rule"]
                    current_method = (
                        st.session_state["user_prompts"][idx].get("eval_method")
                        or "pass"
                    )
                    current_keyword = (
                        st.session_state["user_prompts"][idx].get("eval_keyword") or ""
                    )
                    eval_method_single = st.segmented_control(
                        "평가 방법 선택 (eval_method)",
                        options=method_options,
                        default=(
                            current_method
                            if current_method in method_options
                            else method_options[0]
                        ),
                        key=f"user_eval_method_select_single_{idx}",
                        label_visibility="collapsed",
                    )
                    st.session_state["user_prompts"][idx][
                        "eval_method"
                    ] = eval_method_single
                    if eval_method_single == "rule":
                        eval_keyword_single = st.text_area(
                            "평가 keyword 입력",
                            height=68,
                            value=current_keyword,
                            key=f"user_eval_keyword_single_{idx}",
                            label_visibility="collapsed",
                        )
                        st.session_state["user_prompts"][idx][
                            "eval_keyword"
                        ] = eval_keyword_single

                    if btn_update_usr:
                        update_user_prompt(
                            st.session_state["user_prompts"][idx]["id"],
                            new_prompt=st.session_state["user_prompts"][idx]["prompt"],
                            eval_method=st.session_state["user_prompts"][idx][
                                "eval_method"
                            ],
                            eval_keyword=st.session_state["user_prompts"][idx].get(
                                "eval_keyword", ""
                            ),
                        )
                        st.toast("User Prompt updated")
                        time.sleep(0.7)
                        st.rerun()
                    user_single_text = (
                        st.session_state["user_prompts"][idx]["prompt"]
                        if prompt_count > 0
                        else ""
                    )
                else:
                    user_single_text = ""

        ##########

        with st.expander("Model", expanded=False):
            model_compare_toggle = st.toggle("다중 Model 활성화", key="model_toggle")
            if model_compare_toggle:
                db_models = get_models(project_id)
                st.session_state["model_names"] = [
                    row["model_name"] for row in db_models
                ]
                if not st.session_state["model_names"]:
                    add_model("GPT-4o", project_id)
                    add_model("GPT-4o mini", project_id)
                    st.session_state["model_names"] = ["GPT-4o", "GPT-4o mini"]

                if "model_1_name" not in st.session_state:
                    st.session_state["model_1_name"] = st.session_state["model_names"][
                        0
                    ]
                if "model_2_name" not in st.session_state:
                    if len(st.session_state["model_names"]) > 1:
                        st.session_state["model_2_name"] = st.session_state[
                            "model_names"
                        ][1]
                    else:
                        st.session_state["model_2_name"] = st.session_state[
                            "model_names"
                        ][0]

                col1, col2 = st.columns(2)
                with col1:
                    model_1 = st.selectbox(
                        "모델 선택 1",
                        st.session_state["model_names"],
                        index=st.session_state["model_names"].index(
                            st.session_state["model_1_name"]
                        ),
                        key="model_selector_1",
                        label_visibility="collapsed",
                    )
                    st.session_state["model_1_name"] = model_1

                    if (
                        model_1 == st.session_state["model_2_name"]
                        and len(st.session_state["model_names"]) > 1
                    ):
                        available_models = [
                            m for m in st.session_state["model_names"] if m != model_1
                        ]
                        if available_models:
                            st.session_state["model_2_name"] = available_models[0]
                            st.rerun()

                with col2:
                    available_models = st.session_state["model_names"]
                    if len(available_models) > 1:
                        available_models = [
                            m for m in st.session_state["model_names"] if m != model_1
                        ]

                    if st.session_state["model_2_name"] in available_models:
                        default_idx = available_models.index(
                            st.session_state["model_2_name"]
                        )
                    else:
                        default_idx = 0

                    model_2 = st.selectbox(
                        "모델 선택 2",
                        available_models,
                        index=min(default_idx, len(available_models) - 1),
                        key="model_selector_2",
                        label_visibility="collapsed",
                    )
                    st.session_state["model_2_name"] = model_2

                model_list = [model_1, model_2]

                if model_1 == model_2:
                    st.warning(
                        "두 모델이 동일합니다. 다른 결과를 비교하려면 서로 다른 모델을 선택하세요."
                    )
            else:
                single_model = st.selectbox(
                    "모델 선택",
                    ("GPT-4o", "GPT-4o mini"),
                    index=0,
                    key="model_selector_single",
                    label_visibility="collapsed",
                )
                model_list = [single_model]

        ##########

        if execute_button:
            if system_compare_toggle:
                system_prompts = st.session_state["system_prompts"]
            else:
                system_prompts = [{"prompt": system_single_text or ""}]

            if user_compare_toggle:
                user_prompts = st.session_state["user_prompts"]
            else:
                user_prompts = [
                    {
                        "prompt": user_single_text or "",
                        "eval_method": st.session_state["user_prompts"][idx].get(
                            "eval_method", "pass"
                        ),
                        "eval_keyword": st.session_state["user_prompts"][idx].get(
                            "eval_keyword", ""
                        ),
                    }
                ]

            st.toast(
                f"|| Model: {len(model_list)}개 || system_prompt: {len(system_prompts)}개 || user_prompt: {len(user_prompts)}개 ||"
            )

            project_dict = dict(st.session_state["project"])
            current_session_id = project_dict.get("current_session_id", 0)
            new_session_id = current_session_id + 1

            from database import update_project_session_id

            update_project_session_id(project_id, new_session_id)

            project_dict["current_session_id"] = new_session_id
            st.session_state["project"] = project_dict

            st.session_state["results"] = []
            st.session_state["system_count"] = len(system_prompts)
            st.session_state["user_count"] = len(user_prompts)
            st.session_state["model_names"] = model_list

            total_combinations = (
                len(model_list) * len(system_prompts) * len(user_prompts)
            )

            current_run = 1
            toast_msg = st.toast(
                f"[{current_run}/{total_combinations}] 실행을 시작합니다..."
            )

            for model_name in model_list:
                use_gpt4o_mini = model_name == "GPT-4o mini"

                for sys_idx, sys_prompt in enumerate(system_prompts, 1):
                    for user_idx, user_prompt in enumerate(user_prompts, 1):

                        toast_msg.toast(
                            f"🚀 [{current_run}/{total_combinations}] {model_name} | Sys{sys_idx} + User{user_idx} 실행 중..."
                        )

                        full_response = ""
                        for token in stream_chat(
                            system_prompt=sys_prompt["prompt"],
                            user_prompt=user_prompt["prompt"],
                            use_gpt4o_mini=use_gpt4o_mini,
                        ):
                            full_response += token

                        st.session_state["results"].append(full_response)

                        add_result(
                            sys_prompt["prompt"],
                            user_prompt["prompt"],
                            model_name,
                            full_response,
                            new_session_id,
                            project_id,
                            "X",
                            eval_method=user_prompt["eval_method"],
                            eval_keyword=user_prompt["eval_keyword"],
                        )

                        toast_msg.toast(
                            f"[{current_run}/{total_combinations}] {model_name} | Sys{sys_idx} + User{user_idx} 실행 완료!"
                        )
                        current_run += 1

            run_evaluation(project_id, new_session_id)

            latest_results = get_results_by_project(project_id)
            st.session_state["results"] = [
                {
                    "result": r["result"],
                    "eval_pass": r["eval_pass"],
                    "eval_method": r["eval_method"],
                    "eval_keyword": r["eval_keyword"],
                }
                for r in latest_results
                if r["session_id"] == new_session_id
            ]

            toast_msg.toast(f":green[모든 실행이 완료되었습니다!]", icon="🎉")
            time.sleep(1)
            st.rerun()

        if btn_generate_sys:
            generate_system_prompt_dialog()

        if (
            "select_new_system_prompt" in st.session_state
            and st.session_state["select_new_system_prompt"]
        ):
            st.session_state["system_expanded"] = True
            st.session_state["select_new_system_prompt"] = False

        if (
            "select_new_user_prompt" in st.session_state
            and st.session_state["select_new_user_prompt"]
        ):
            st.session_state["user_expanded"] = True
            st.session_state["select_new_user_prompt"] = False


@st.dialog("System Prompt 자동 생성")
def generate_system_prompt_dialog():
    user_intention = st.text_area(
        "AI 시스템의 역할/작동 방식을 자세히 설명해주세요.",
        # value 선입력
        value="""
[시스템 프롬프트 생성 프로세스]
1. 유저가 입력한 역할/작동방식으로부터 ##역할, ##지시 사항, ##정보 로 나누어 생성하기
2. 유저가 입력한 역할/작동방식으로부터 ##출력 예시 생성하기.
3. 유저가 입력한 역할/작동방식을 개괄식으로 빠짐없이 나열해 정제하기.
4. ##역할, ##지시 사항, ##정보, ##입력 예시, ##출력 예시 를 검토해 각 항목이 서로 모순되거나 충돌하는 지시, 정보가 없는지 평가하기. pass시 다음노드로, fail시 충돌을 해결하는 새로운 노드를 거친뒤 재평가. pass까지 반복
5. ##역할, ##지시 사항, ##정보, ##출력 예시 를 검토해 개괄식으로 정제한 역할/작동방식의 모든 항목을 만족하는지 평가하기. pass시 다음노드로, fail시 만족하지 못한 항목을 수정하는 새로운 노드를 거친뒤 재평가. pass까지 반복

이때 1,2,3번은 병렬처리(비동기)로 해주고 1,2,3 작업이 모두끝나면 4로, 그다음 5로 순차적으로 넘겨주고 최종 output을 배출해.

위 시스템 프롬프트 생성 프로세스의 1. 유저가 입력한 역할/작동방식으로부터 ##역할, ##지시 사항, ##정보 로 나누어 생성하기 부분을 위한 시스템 프롬프트 작성해줘.
""",
        height=100,
        placeholder="예: 고객 문의에 친절하게 응답하는 전문 상담원 역할 + 화자의 감정을 분석하여 공감있는 응대...",
    )

    generate_btn = st.button(
        "✨ 프롬프트 생성", use_container_width=True, type="primary"
    )

    # 노드별 컨테이너
    partial_containers = {
        "__NODE1_PARTIAL__": st.empty(),
        "__NODE2_PARTIAL__": st.empty(),
        "__NODE3_PARTIAL__": st.empty(),
        "__NODE4_PARTIAL__": st.empty(),
        "__NODE4_1_PARTIAL__": st.empty(),
        "__NODE5_PARTIAL__": st.empty(),
        "__NODE5_1_PARTIAL__": st.empty(),
        "__NODE6_PARTIAL__": st.empty(),
    }
    partial_texts = {k: "" for k in partial_containers}

    # 결과 파싱용 컨테이너 추가
    result_containers = {k: st.empty() for k in partial_containers}
    final_container = st.empty()

    def parse_node_content(node_key, content):
        """노드 내용에서 JSON 파싱 시도"""
        if "__FINAL_PARSE__:" in content:
            try:
                # 다음 노드 마커나 텍스트가 있을 경우 잘라내기
                if "__PARSED_OBJECT__:" in content:
                    json_part = content.split("__PARSED_OBJECT__:", 1)[0].strip()
                    json_part = json_part.split("__FINAL_PARSE__:", 1)[1].strip()
                else:
                    json_part = content.split("__FINAL_PARSE__:", 1)[1].strip()

                data = json.loads(json_part)
                result_containers[node_key].markdown(
                    f"**{node_key.replace('__', '')} 결과**:\n\n```json\n{json.dumps(data, ensure_ascii=False, indent=2)}\n```"
                )
                return True
            except Exception as e:
                result_containers[node_key].error(f"파싱 에러 ({node_key}): {e}")
        return False

    if generate_btn and user_intention:
        for token in generate_system_prompt_by_intention_streaming(user_intention):
            # 최종 결과 처리
            if token.startswith("__RESULT_DATA__"):
                json_str = token.replace("__RESULT_DATA__:", "")
                try:
                    data = json.loads(json_str)
                    fp = data.get("final_prompt", {})
                    sp = fp.get("system_prompt", "")
                    rs = fp.get("reasoning", "")
                    if sp:
                        st.success("System Prompt created!")
                        final_container.markdown(
                            f"**System Prompt**\n```\n{sp}\n```\n\n**Reasoning**:\n```\n{rs}\n```"
                        )
                except Exception as e:
                    st.error(f"파싱 에러: {e}")
                continue

            # 노드별 내용 처리
            if ":" in token:
                node_prefix, content = token.split(":", 1)
                if node_prefix in partial_containers:
                    partial_texts[node_prefix] += content

                    # __FINAL_PARSE__가 포함되어 있으면 해당 노드의 컨테이너를 비우고 결과만 표시
                    if "__FINAL_PARSE__:" in partial_texts[node_prefix]:
                        partial_containers[
                            node_prefix
                        ].empty()  # 원본 텍스트 컨테이너를 비움
                        parse_node_content(node_prefix, partial_texts[node_prefix])
                    else:
                        partial_containers[node_prefix].markdown(
                            f"{node_prefix}:\n\n{partial_texts[node_prefix]}"
                        )
                else:
                    # 알 수 없는 노드
                    st.info(token)
            elif token.startswith("__FINAL_PARSE_ERROR__"):
                err = token.replace("__FINAL_PARSE_ERROR__:", "")
                st.error(f"Parse Error: {err}")
            else:
                # 그 외 토큰은 일반 정보로 표시
                st.info(token)

    elif generate_btn:
        st.warning("역할/작동 방식을 입력해주세요.")

    if st.button("close", use_container_width=True):
        st.rerun()
