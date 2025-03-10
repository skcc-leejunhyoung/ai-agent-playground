import streamlit as st
from st_diff_viewer import diff_viewer

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

st.markdown("프롬프팅 에이전트를 결합한 playground")

col1, col2 = st.columns(2)

with col1:
    old_text = st.text_area("[Old]", height=300, value="a = 0")

with col2:
    new_text = st.text_area("[New]", height=300, value="a = 1")

split_view = st.toggle("Split View", value=True)

if old_text.strip() == "" and new_text.strip() == "":
    st.warning("비교할 텍스트를 입력해주세요!")
else:
    diff_viewer(
        old_text,
        new_text,
        split_view=split_view,
        use_dark_theme=True,
        styles=one_dark_pro_styles,
    )
