import streamlit as st
import difflib

st.title("텍스트 Diff 뷰어")

# 사용자 입력 텍스트
text1 = st.text_area("원본 텍스트", height=200)
text2 = st.text_area("비교할 텍스트", height=200)

if st.button("비교하기"):
    # 라인 단위로 나누기
    text1_lines = text1.splitlines()
    text2_lines = text2.splitlines()

    # 차이 계산
    diff = difflib.ndiff(text1_lines, text2_lines)

    # HTML 형식으로 변환
    diff_html = difflib.HtmlDiff().make_table(
        text1_lines, text2_lines, "원본", "비교 대상"
    )

    st.markdown(diff_html, unsafe_allow_html=True)
