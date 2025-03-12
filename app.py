# app.py

import streamlit as st
from st_diff_viewer import diff_viewer
from streamlit_monaco import st_monaco

from model import stream_chat
from components.result_card import result_cards

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

st.set_page_config(page_title="Prompt Playground", layout="wide")

if "results" not in st.session_state:
    st.session_state["results"] = []
    st.session_state["system_count"] = 1
    st.session_state["user_count"] = 1
    st.session_state["model_names"] = []

##########

st.header("Prompt Playground", divider="blue")

header_col1, header_col2 = st.columns([8, 1.3], vertical_alignment="center")

with header_col1:
    st.write("_프롬프팅 에이전트를 결합한 playground_")

with header_col2:
    model_count = 2 if st.session_state.get("model_toggle") else 1
    system_count = 2 if st.session_state.get("system_toggle") else 1
    user_count = 2 if st.session_state.get("user_toggle") else 1

    total_runs = model_count * system_count * user_count
    execute_label = f"**:red[{total_runs} | Execute]**"

    execute_button = st.button(execute_label, use_container_width=True)

##########
with st.expander("System Prompt", expanded=False):
    system_compare_toggle = st.toggle("다중 System Prompt 활성화", key="system_toggle")

    if system_compare_toggle:
        prompt_count = 10
        if "system_prompts" not in st.session_state:
            st.session_state["system_prompts"] = [
                f"You are helpful assistant! #{i+1}" for i in range(prompt_count)
            ]

        if "system_select_1_idx" not in st.session_state:
            st.session_state["system_select_1_idx"] = 0
        if "system_select_2_idx" not in st.session_state:
            st.session_state["system_select_2_idx"] = 1

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

        col1, col2 = st.columns(2)

        with col1:
            idx_1 = st.selectbox(
                "sys 1 선택",
                options=available_idx_for_1,
                index=(
                    available_idx_for_1.index(st.session_state["system_select_1_idx"])
                    if st.session_state["system_select_1_idx"] in available_idx_for_1
                    else 0
                ),
                format_func=lambda x: f"{x+1}",
                key="system_select_1",
                label_visibility="collapsed",
            )
            st.session_state["system_select_1_idx"] = idx_1

            editor_1 = st_monaco(
                value=st.session_state["system_prompts"][idx_1],
                language="markdown",
                height="300px",
                theme="vs-dark",
            )
            if editor_1 is not None:
                st.session_state["system_prompts"][idx_1] = editor_1

        with col2:
            idx_2 = st.selectbox(
                "sys 2 선택",
                options=available_idx_for_2,
                index=(
                    available_idx_for_2.index(st.session_state["system_select_2_idx"])
                    if st.session_state["system_select_2_idx"] in available_idx_for_2
                    else 0
                ),
                format_func=lambda x: f"{x+1}",
                key="system_select_2",
                label_visibility="collapsed",
            )
            st.session_state["system_select_2_idx"] = idx_2

            editor_2 = st_monaco(
                value=st.session_state["system_prompts"][idx_2],
                language="markdown",
                height="300px",
                theme="vs-dark",
            )
            if editor_2 is not None:
                st.session_state["system_prompts"][idx_2] = editor_2

        split_view_sys = st.toggle("Split View", value=True, key="system_split_view")

        diff_viewer(
            st.session_state["system_prompts"][idx_1],
            st.session_state["system_prompts"][idx_2],
            split_view=split_view_sys,
            use_dark_theme=True,
            styles=one_dark_pro_styles,
        )

    else:
        system_single_text = st_monaco(
            value="You are helpful assistant!",
            language="markdown",
            height="300px",
            theme="vs-dark",
        )


##########
with st.expander("User Prompt", expanded=False):
    user_compare_toggle = st.toggle("다중 User Prompt 활성화", key="user_toggle")

    if user_compare_toggle:
        prompt_count = 10
        if "user_prompts" not in st.session_state:
            st.session_state["user_prompts"] = [
                f"User Prompt {i+1}" for i in range(prompt_count)
            ]

        if "user_select_1_idx" not in st.session_state:
            st.session_state["user_select_1_idx"] = 0
        if "user_select_2_idx" not in st.session_state:
            st.session_state["user_select_2_idx"] = 1

        available_idx_for_1 = [
            i for i in range(prompt_count) if i != st.session_state["user_select_2_idx"]
        ]
        available_idx_for_2 = [
            i for i in range(prompt_count) if i != st.session_state["user_select_1_idx"]
        ]

        col1, col2 = st.columns(2)

        with col1:
            idx_1 = st.selectbox(
                "usr 1 선택",
                options=available_idx_for_1,
                index=(
                    available_idx_for_1.index(st.session_state["user_select_1_idx"])
                    if st.session_state["user_select_1_idx"] in available_idx_for_1
                    else 0
                ),
                format_func=lambda x: f"{x+1}",
                key="user_select_1",
                label_visibility="collapsed",
            )
            st.session_state["user_select_1_idx"] = idx_1

            editor_1 = st_monaco(
                value=st.session_state["user_prompts"][idx_1],
                language="markdown",
                height="300px",
                theme="vs-dark",
            )
            if editor_1 is not None:
                st.session_state["user_prompts"][idx_1] = editor_1

        with col2:
            idx_2 = st.selectbox(
                "usr 2 선택",
                options=available_idx_for_2,
                index=(
                    available_idx_for_2.index(st.session_state["user_select_2_idx"])
                    if st.session_state["user_select_2_idx"] in available_idx_for_2
                    else 0
                ),
                format_func=lambda x: f"{x+1}",
                key="user_select_2",
                label_visibility="collapsed",
            )
            st.session_state["user_select_2_idx"] = idx_2

            editor_2 = st_monaco(
                value=st.session_state["user_prompts"][idx_2],
                language="markdown",
                height="300px",
                theme="vs-dark",
            )
            if editor_2 is not None:
                st.session_state["user_prompts"][idx_2] = editor_2

        split_view_usr = st.toggle("Split View", value=True, key="user_split_view")

        diff_viewer(
            st.session_state["user_prompts"][idx_1],
            st.session_state["user_prompts"][idx_2],
            split_view=split_view_usr,
            use_dark_theme=True,
            styles=one_dark_pro_styles,
        )

    else:
        user_single_text = st_monaco(
            value="Harry James Potter, Hermione Jean Granger, Ronald Bilius Weasley 중에서 'r'이 가장 많이 들어간 단어는 뭐야?",
            language="markdown",
            height="300px",
            theme="vs-dark",
        )


##########
with st.expander("Model", expanded=False):
    model_compare_toggle = st.toggle("다중 Model 활성화", key="model_toggle")

    if model_compare_toggle:
        col1, col2 = st.columns(2)

        with col1:
            model_1 = st.selectbox(
                "모델 선택 1",
                ("GPT-4o", "GPT-4o mini"),
                index=0,
                key="model_selector_1",
                label_visibility="collapsed",
            )

        with col2:
            model_2 = st.selectbox(
                "모델 선택 2",
                ("GPT-4o", "GPT-4o mini"),
                index=1,
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
        system_prompts = [system_single_text or ""]

    if user_compare_toggle:
        user_prompts = st.session_state["user_prompts"]
    else:
        user_prompts = [user_single_text or ""]

    st.info(
        f"||  모델:  {len(model_list)}개  || 시스템 프롬프트:  {len(system_prompts)}개  || 유저 프롬프트:  {len(user_prompts)}개  || 조합으로 실행됩니다."
    )

    st.session_state["results"] = []
    st.session_state["system_count"] = len(system_prompts)
    st.session_state["user_count"] = len(user_prompts)
    st.session_state["model_names"] = model_list

    with st.spinner("Processing.."):
        for model_name in model_list:
            use_gpt4o_mini = True if model_name == "GPT-4o mini" else False

            for sys_idx, sys_prompt in enumerate(system_prompts, 1):
                for user_idx, user_prompt in enumerate(user_prompts, 1):
                    full_response = ""

                    for token in stream_chat(
                        system_prompt=sys_prompt,
                        user_prompt=user_prompt,
                        use_gpt4o_mini=use_gpt4o_mini,
                    ):
                        full_response += token

                    result_markdown = full_response
                    st.session_state["results"].append(result_markdown)

##########
if st.session_state["results"]:
    result_cards(
        st.session_state["results"],
        system_count=st.session_state.get("system_count", 1),
        user_count=st.session_state.get("user_count", 1),
        model_names=st.session_state.get("model_names", []),
        height=500,
    )
