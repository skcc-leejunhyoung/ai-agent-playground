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

st.title("Prompt Playground")

header_col1, header_col2 = st.columns([8, 1])

with header_col1:
    st.markdown("프롬프팅 에이전트를 결합한 playground")

with header_col2:
    execute_button = st.button("Execute")

with st.expander("System Prompt", expanded=False):
    st.subheader("System Prompt 비교")

    col1, col2 = st.columns(2)

    with col1:
        system_old_text = st_monaco(
            value="시스템 프롬프트 OLD",
            language="markdown",
            height="300px",
            theme="vs-dark",
        )

    with col2:
        system_new_text = st_monaco(
            value="시스템 프롬프트 NEW",
            language="markdown",
            height="300px",
            theme="vs-dark",
        )

    split_view_sys = st.toggle("System Split View", value=True)

    if (system_old_text or "").strip() == "" and (system_new_text or "").strip() == "":
        st.warning("System Prompt의 텍스트를 입력해주세요!")
    else:
        diff_viewer(
            system_old_text or "",
            system_new_text or "",
            split_view=split_view_sys,
            use_dark_theme=True,
            styles=one_dark_pro_styles,
        )


with st.expander("User Prompt", expanded=False):
    st.subheader("User Prompt 비교")

    col1, col2 = st.columns(2)

    with col1:
        user_old_text = st_monaco(
            value="유저 프롬프트 OLD",
            language="markdown",
            height="300px",
            theme="vs-dark",
        )

    with col2:
        user_new_text = st_monaco(
            value="유저 프롬프트 NEW",
            language="markdown",
            height="300px",
            theme="vs-dark",
        )

    split_view_user = st.toggle("User Split View", value=True)

    if (user_old_text or "").strip() == "" and (user_new_text or "").strip() == "":
        st.warning("User Prompt의 텍스트를 입력해주세요!")
    else:
        diff_viewer(
            user_old_text or "",
            user_new_text or "",
            split_view=split_view_user,
            use_dark_theme=True,
            styles=one_dark_pro_styles,
        )

# 예시: 카드에 들어갈 내용을 리스트로 전달
result = [
    "##첫 번째 결과입니다.",
    "#두 번째 결과입니다.#두 번째 결과입니다.#두 번째 결과입니다.#두 번째 결과입니다.#두 번째 결과입니다.#두 번째 결과입니다.#두 번째 결과입니다.#두 번째 결과입니다.#두 번째 결과입니다.#두 번째 결과입니다.#두 번째 결과입니다.#두 번째 결과입니다.#두 번째 결과입니다.#두 번째 결과입니다.#두 번째 결과입니다.",
    "세 번째 결과입니다.",
    "네 번째 결과입니다.",
    "다섯 번째 결과입니다.",
    "여섯 번째 결과입니다.",
    "일곱 번째 결과입니다.",
    "여덟 번째 결과입니다.",
]

# 렌더링 호출
result_cards(result, height=500)
