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

st.set_page_config(page_title="텍스트 비교기", layout="wide")

if "results" not in st.session_state:
    st.session_state["results"] = []
    st.session_state["system_count"] = 1
    st.session_state["user_count"] = 1
    st.session_state["model_names"] = []

##########

st.title("Prompt Playground")

header_col1, header_col2 = st.columns([8, 1.3])

with header_col1:
    st.markdown("프롬프팅 에이전트를 결합한 playground")

with header_col2:
    model_count = (
        2
        if st.session_state.get("model_names")
        and len(st.session_state["model_names"]) > 1
        else 1
    )
    system_count = 2 if st.session_state.get("system_toggle") else 1
    user_count = 2 if st.session_state.get("user_toggle") else 1

    total_runs = model_count * system_count * user_count
    execute_label = f"{total_runs} | Execute"

    execute_button = st.button(execute_label, use_container_width=True)

##########
with st.expander("System Prompt", expanded=False):
    system_compare_toggle = st.toggle("다중 System Prompt 활성화", key="system_toggle")

    if system_compare_toggle:
        col1, col2 = st.columns(2)

        with col1:
            system_old_text = st_monaco(
                value="You are helpful assistant!",
                language="markdown",
                height="300px",
                theme="vs-dark",
            )

        with col2:
            system_new_text = st_monaco(
                value="너는 고양이야. 앞으로 어떤 질문이 들어와도 야옹이라고 대답해",
                language="markdown",
                height="300px",
                theme="vs-dark",
            )

        split_view_sys = st.toggle("Split View", value=True, key="system_split_view")

        if (system_old_text or "").strip() == "" and (
            system_new_text or ""
        ).strip() == "":
            st.warning("System Prompt를 입력해주세요!")
        else:
            diff_viewer(
                system_old_text or "",
                system_new_text or "",
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
        col1, col2 = st.columns(2)

        with col1:
            user_old_text = st_monaco(
                value="미국의 화폐단위가 뭐야?",
                language="markdown",
                height="300px",
                theme="vs-dark",
            )

        with col2:
            user_new_text = st_monaco(
                value="미국의 수도가 어디야?",
                language="markdown",
                height="300px",
                theme="vs-dark",
            )
        split_view_usr = st.toggle("Split View", value=True, key="user_split_view")

        if (user_old_text or "").strip() == "" and (user_new_text or "").strip() == "":
            st.warning("User Prompt를 입력해주세요!")
        else:
            diff_viewer(
                user_old_text or "",
                user_new_text or "",
                split_view=split_view_usr,
                use_dark_theme=True,
                styles=one_dark_pro_styles,
            )
    else:
        user_single_text = st_monaco(
            value="대한민국의 수도가 어디야?",
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
        system_prompts = [system_old_text or "", system_new_text or ""]
    else:
        system_prompts = [system_single_text or ""]

    if user_compare_toggle:
        user_prompts = [user_old_text or "", user_new_text or ""]
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

########## Results ##########
if st.session_state["results"]:
    result_cards(
        st.session_state["results"],
        system_count=st.session_state.get("system_count", 1),
        user_count=st.session_state.get("user_count", 1),
        model_names=st.session_state.get("model_names", []),
        height=500,
    )
