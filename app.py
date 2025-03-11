# app.py

import streamlit as st
from st_diff_viewer import diff_viewer
from streamlit_monaco import st_monaco
import streamlit.components.v1 as components

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

##########

if "results" not in st.session_state:
    st.session_state["results"] = []

##########

st.title("Prompt Playground")

header_col1, header_col2 = st.columns([8, 1])

with header_col1:
    st.markdown("프롬프팅 에이전트를 결합한 playground")

with header_col2:
    execute_button = st.button("Execute")

##########

with st.expander("System Prompt", expanded=False):
    sub_col1, sub_col2 = st.columns([8, 2])

    with sub_col1:
        st.subheader("System Prompt 비교")

    with sub_col2:
        system_compare_toggle = st.toggle("비교 활성화", key="system_toggle")

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
    sub_col1, sub_col2 = st.columns([8, 2])

    with sub_col1:
        st.subheader("User Prompt 비교")

    with sub_col2:
        user_compare_toggle = st.toggle("비교 활성화", key="user_toggle")

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
        f"총 {len(system_prompts)}개의 시스템 프롬프트와 {len(user_prompts)}개의 유저 프롬프트 조합으로 실행됩니다."
    )

    st.session_state["results"] = []

    with st.spinner("Processing.."):
        for sys_idx, sys_prompt in enumerate(system_prompts, 1):
            for user_idx, user_prompt in enumerate(user_prompts, 1):
                full_response = ""

                for token in stream_chat(
                    system_prompt=sys_prompt, user_prompt=user_prompt
                ):
                    full_response += token

                result_markdown = full_response
                st.session_state["results"].append(result_markdown)


if st.session_state["results"]:
    result_cards(st.session_state["results"], height=500)
