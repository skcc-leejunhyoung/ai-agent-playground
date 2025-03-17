# components/ai_settings.py

import streamlit as st
from st_diff_viewer import diff_viewer
from streamlit_monaco import st_monaco
import uuid
import time

from model import stream_chat
from agent.eval_agent import run_evaluation
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
)


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

        with st.expander("System Prompt", expanded=False):
            system_compare_toggle = st.toggle(
                "다중 System Prompt 활성화", key="system_toggle"
            )
            if system_compare_toggle:
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
                available_idx_for_1 = [
                    i
                    for i in range(prompt_count)
                    if i != st.session_state["system_select_2_idx"]
                ]
                available_idx_for_2 = [
                    i
                    for i in range(prompt_count)
                    if i != st.session_state["system_select_1_idx"]
                ]
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        with st.container():
                            c1, c2 = st.columns([10, 1])
                            with c1:
                                idx_1 = st.selectbox(
                                    "sys 1 선택",
                                    options=available_idx_for_1,
                                    index=(
                                        available_idx_for_1.index(
                                            st.session_state["system_select_1_idx"]
                                        )
                                        if st.session_state["system_select_1_idx"]
                                        in available_idx_for_1
                                        else 0
                                    ),
                                    format_func=lambda x: f"{x+1}. {st.session_state['system_prompts'][x]['prompt'][:30]}...",
                                    key="system_select_1",
                                    label_visibility="collapsed",
                                )
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
                                    index=(
                                        available_idx_for_2.index(
                                            st.session_state["system_select_2_idx"]
                                        )
                                        if st.session_state["system_select_2_idx"]
                                        in available_idx_for_2
                                        else 0
                                    ),
                                    format_func=lambda x: f"{x+1}. {st.session_state['system_prompts'][x]['prompt'][:30]}...",
                                    key="system_select_2",
                                    label_visibility="collapsed",
                                )
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
                                        if (
                                            st.session_state["system_select_1_idx"]
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
                st.session_state["system_prompts"] = [
                    {"id": row["id"], "prompt": row["prompt"]}
                    for row in db_system_prompts
                ]
                if "system_single_idx" not in st.session_state:
                    st.session_state["system_single_idx"] = 0
                prompt_count = len(st.session_state["system_prompts"])
                c1, c2 = st.columns([10, 1])
                with c1:
                    idx = st.selectbox(
                        "System Prompt 선택",
                        options=list(range(prompt_count)),
                        index=st.session_state["system_single_idx"],
                        format_func=lambda x: f"{x+1}. {st.session_state['system_prompts'][x]['prompt'][:30]}...",
                        key="system_single_select",
                        label_visibility="collapsed",
                    )
                    st.session_state["system_single_idx"] = idx
                with c2:
                    btn_update_sys = st.button(
                        ":material/save:",
                        key=f"update_system_single_top_{idx}",
                        type="tertiary",
                        use_container_width=True,
                    )
                editor_single = st_monaco(
                    value=st.session_state["system_prompts"][idx]["prompt"],
                    language="markdown",
                    height="300px",
                    theme="vs-dark",
                )
                if editor_single is not None:
                    st.session_state["system_prompts"][idx]["prompt"] = editor_single
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
                system_single_text = st.session_state["system_prompts"][idx]["prompt"]

        ##########

        with st.expander("User Prompt", expanded=True):
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
                available_idx_for_1 = [
                    i
                    for i in range(prompt_count)
                    if i != st.session_state["user_select_2_idx"]
                ]
                available_idx_for_2 = [
                    i
                    for i in range(prompt_count)
                    if i != st.session_state["user_select_1_idx"]
                ]
                col1, col2 = st.columns(2)
                with col1:
                    c1, c2 = st.columns([10, 1])
                    with c1:
                        idx_1 = st.selectbox(
                            "usr 1 선택",
                            options=available_idx_for_1,
                            index=(
                                available_idx_for_1.index(
                                    st.session_state["user_select_1_idx"]
                                )
                                if st.session_state["user_select_1_idx"]
                                in available_idx_for_1
                                else 0
                            ),
                            format_func=lambda x: f"{x+1}. {st.session_state['user_prompts'][x]['prompt'][:30]}...",
                            key="user_select_1",
                            label_visibility="collapsed",
                        )
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
                            index=(
                                available_idx_for_2.index(
                                    st.session_state["user_select_2_idx"]
                                )
                                if st.session_state["user_select_2_idx"]
                                in available_idx_for_2
                                else 0
                            ),
                            format_func=lambda x: f"{x+1}. {st.session_state['user_prompts'][x]['prompt'][:30]}...",
                            key="user_select_2",
                            label_visibility="collapsed",
                        )
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
                                if (
                                    st.session_state["user_select_1_idx"]
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
                st.session_state["user_prompts"] = [
                    {
                        "id": row["id"],
                        "prompt": row["prompt"],
                        "eval_method": row["eval_method"],
                        "eval_keyword": row["eval_keyword"],
                    }
                    for row in db_user_prompts
                ]
                if "user_single_idx" not in st.session_state:
                    st.session_state["user_single_idx"] = 0
                prompt_count = len(st.session_state["user_prompts"])
                c1, c2 = st.columns([10, 1])
                with c1:
                    idx = st.selectbox(
                        "User Prompt 선택",
                        options=list(range(prompt_count)),
                        index=st.session_state["user_single_idx"],
                        format_func=lambda x: f"{x+1}. {st.session_state['user_prompts'][x]['prompt'][:30]}...",
                        key="user_single_select",
                        label_visibility="collapsed",
                    )
                    st.session_state["user_single_idx"] = idx
                with c2:
                    btn_update_usr = st.button(
                        ":material/save:",
                        key=f"update_user_single_top_{idx}",
                        type="tertiary",
                        use_container_width=True,
                    )
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
                    st.session_state["user_prompts"][idx].get("eval_method") or "pass"
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
                user_single_text = st.session_state["user_prompts"][idx]["prompt"]

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
                col1, col2 = st.columns(2)
                with col1:
                    model_1 = st.selectbox(
                        "모델 선택 1",
                        st.session_state["model_names"],
                        index=0,
                        key="model_selector_1",
                        label_visibility="collapsed",
                    )
                with col2:
                    model_2 = st.selectbox(
                        "모델 선택 2",
                        st.session_state["model_names"],
                        index=1 if len(st.session_state["model_names"]) > 1 else 0,
                        key="model_selector_2",
                        label_visibility="collapsed",
                    )
                model_list = [model_1, model_2]
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
                            f"🚀 [{current_run}/{total_combinations}] {model_name} | Sys{sys_idx} + User{user_idx} 실행 완료!"
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
