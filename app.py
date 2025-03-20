# app.py

import streamlit as st
import time
import io
import os

from agent.rag_module import process_for_rag
from database import (
    init_db,
    get_projects,
    delete_project,
    create_project,
    add_system_prompt,
    add_user_prompt,
    add_model,
    import_project_csv,
)


##########

init_db()

st.set_page_config(page_title="Prompt Playground", layout="wide")

st.markdown(
    """
    <style>
        .centered-title {
            text-align: center;
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .card-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 2rem;
            margin-top: 2rem;
        }
        .project-card {
            background-color: #ffffff10;
            border-radius: 1rem;
            padding: 1.5rem;
            width: 300px;
            height: 200px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            cursor: pointer;
            text-decoration: none;
            color: inherit;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .project-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
        }
        .project-card.add {
            justify-content: center;
            align-items: center;
            font-size: 4rem;
        }
        .outer-card-wrapper:hover .close-button{
            transform: translateY(-5px);
            transition: transform 0.2s ease;
        }
        .project-card h2 {
            overflow-x: auto;
            white-space: nowrap;
        }
        .project-card i {
            display: block;
            width: 250px;
            height: 80px;
            overflow-y: auto;
            overflow-x: hidden;
            white-space: pre-wrap;
            overflow-wrap: break-word;  
        }
    </style>
    """,
    unsafe_allow_html=True,
)


##########


projects = get_projects()

default_project = None
for proj in projects:
    if proj["project_name"] == "default":
        default_project = proj
        break

if not default_project:
    project_id = create_project("default", "This is auto-generated project.")
    add_system_prompt("You are helpful assistant!", project_id)
    add_system_prompt(
        "You always answer 'meow' regardless of the question.", project_id
    )
    add_user_prompt("What is the capital of France?", project_id)
    add_user_prompt("What is the currency of USA?", project_id)
    add_model("GPT-4o", project_id)
    add_model("GPT-4o mini", project_id)
    st.toast("default 프로젝트가 생성되었습니다")
    projects = get_projects()


##########


if "add_card_state" not in st.session_state:
    st.session_state.add_card_state = False
if "task_state" not in st.session_state:
    st.session_state.task_state = None
if "new_project_name" not in st.session_state:
    st.session_state.new_project_name = ""
    st.session_state.new_project_description = ""
    st.session_state.uploaded_file = None
if "delete_project_id" not in st.session_state:
    st.session_state.delete_project_id = None

##########


st.markdown(
    '<h1 class="centered-title">프로젝트를 <span style="color:#1f77b4;">선택</span>하세요</h1>',
    unsafe_allow_html=True,
)
st.divider()

query_params = st.query_params

html_cards = """<div class="card-container">"""

for proj in projects:
    proj_dict = dict(proj)
    project_name = proj_dict["project_name"]
    project_id = proj_dict["project_id"]
    session_id = proj_dict.get("current_session_id", 0)
    description = proj_dict.get("description", "")

    html_cards += f"""
    <div class="project-card" style="position: relative;">
        <a href="/main_playground?selected_project={project_id}" style="flex-grow:1; text-decoration: none; color: inherit;">
            <h2>{project_name}</h2>
            <i>{description}</i>
        </a>
        <a href="?delete_project_id={project_id}" 
            style="position: absolute; top: 10px; right: 15px; color: #bbb; text-decoration: none; font-size: 1.5rem;">
            ✕
        </a>
    </div>
"""


if st.session_state.get("add_card_state") == False:
    html_cards += """
        <a href="?add_project=true" class="project-card add">
            ➕
        </a>
    """
else:
    html_cards += """
    <div class="outer-card-wrapper" style="position: relative; display: inline-block;">
        <div class="project-card add" style="font-size: 1rem; padding: 1rem; padding-top: 2.1rem;">
            <form method="get" style="width: 100%; display: flex; flex-direction: column; align-items: stretch;">
                <input type="text" 
                    name="new_project_name" 
                    placeholder="프로젝트 이름 입력" 
                    style="width: 100%; 
                            padding: 6px; 
                            margin-bottom: 0.5rem; 
                            border-radius: 5px; 
                            border: none; 
                            font-size: 0.9rem;" 
                    required />
                <textarea name="new_project_description"
                        placeholder="프로젝트 설명 입력"
                        rows="3"
                        style="width: 100%;
                                height: 35px;
                                padding: 6px;
                                margin-bottom: 0.5rem;
                                border-radius: 5px;
                                border: none;
                                font-size: 0.9rem;
                                resize: none;"></textarea>
                <div style="display: flex; gap: 10px;">
                    <a href="?upload_csv=true" 
                        style="width: 48%; 
                                padding: 8px; 
                                border-radius: 5px; 
                                background-color: #28b5d3; 
                                color: white; 
                                text-align: center;
                                font-size: 0.9rem; 
                                text-decoration: none;">
                        CSV 업로드
                    </a>
                    <div style="width: 4%"></div>
                    <button type="submit" 
                            style="width: 48%; 
                                padding: 8px; 
                                border-radius: 5px; 
                                background-color: #1f77b4; 
                                color: white; 
                                border: none; 
                                cursor: pointer; 
                                font-size: 0.9rem;">
                        ﹢ 생성
                    </button>
                </div>
            </form>
        </div>
        <a class="close-button" href="/" 
            style="
                position: absolute; 
                top: 10px; 
                right: 15px; 
                color: #bbb; 
                text-decoration: none; 
                font-size: 1.5rem;
                cursor: pointer;
                transition: transform 0.2s ease;
            ">
            ✕
        </a>
    </div>
    """


st.html(html_cards)


##########


query_params = st.query_params

if query_params.get("add_project") == "true":
    st.session_state.add_card_state = True
    st.query_params.clear()
    st.rerun()

if query_params.get("new_project_name"):
    st.session_state.task_state = "new_project"
    st.session_state.new_project_name = query_params.get("new_project_name")
    st.session_state.new_project_description = query_params.get(
        "new_project_description", ""
    )
    st.query_params.clear()
    st.rerun()

if query_params.get("upload_csv") == "true":
    st.session_state.task_state = "upload_csv"
    st.session_state.add_card_state = False
    st.query_params.clear()
    st.rerun()

if query_params.get("delete_project_id"):
    st.session_state.task_state = "delete_project"
    st.session_state.delete_project_id = query_params.get("delete_project_id")
    st.query_params.clear()
    st.rerun()


##########


if st.session_state.get("task_state") == "new_project":
    new_project_name = st.session_state.get("new_project_name")
    new_project_description = st.session_state.get("new_project_description")

    if new_project_name:
        existing_project = next(
            (dict(p) for p in projects if p["project_name"] == new_project_name), None
        )

        if not existing_project:
            project_id = create_project(new_project_name, new_project_description)
            st.toast(f"{new_project_name} 프로젝트가 생성되었습니다")
            st.session_state.add_card_state = False
            st.session_state.new_project_name = ""
            st.session_state.new_project_description = ""
            st.session_state.uploaded_file = None
            st.session_state.task_state = None
            time.sleep(0.5)
            st.rerun()
        else:
            st.toast(f"'{new_project_name}' 프로젝트가 이미 존재합니다.", icon="⚠️")


if st.session_state.get("task_state") == "upload_csv":
    uploaded_file = st.file_uploader("CSV 파일 업로드", type="csv")
    if uploaded_file is not None:
        file_content = uploaded_file.read()

        with st.form("csv_import_form"):
            new_project_name = st.text_input("새 프로젝트 이름")
            new_project_description = st.text_area("프로젝트 설명", height=68)
            submit_button = st.form_submit_button("Create Project from CSV")

            if submit_button:
                existing_project = next(
                    (
                        dict(p)
                        for p in projects
                        if p["project_name"] == new_project_name
                    ),
                    None,
                )

                if not existing_project:
                    csv_io = io.StringIO(file_content.decode("utf-8"))
                    project_id = import_project_csv(
                        csv_io,
                        new_project_name,
                        new_project_description,
                    )

                    st.toast(f"'{new_project_name}' 프로젝트가 생성되었습니다!")
                    st.session_state.add_card_state = False
                    st.session_state.task_state = None
                    st.session_state.uploaded_file_content = None
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.toast(
                        f"'{new_project_name}' 프로젝트가 이미 존재합니다.", icon="⚠️"
                    )


if st.session_state.get("task_state") == "delete_project":
    delete_project_id = st.session_state.get("delete_project_id")
    delete_project(int(delete_project_id))
    st.toast(f"프로젝트 ID {delete_project_id} 삭제 완료")
    st.session_state.delete_project_id = None
    st.session_state.task_state = None
    time.sleep(0.5)
    st.rerun()


# RAG 섹션
with st.container():
    st.markdown(
        '<div style="text-align: center; margin-bottom: 1rem;"><h2>RAG 학습</h2></div>',
        unsafe_allow_html=True,
    )

    # 파일 업로드 섹션
    st.markdown(
        """
        <div style="border: 1px solid #ddd; padding: 1rem; border-radius: 10px;">
            <h3 style="text-align: center;">파일 업로드로 학습</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "텍스트 또는 HTML 파일 업로드", type=["txt", "html", "htm"]
    )

    if st.button("RAG 학습 시작", use_container_width=True):
        if uploaded_file:
            try:
                from stqdm import stqdm

                with st.spinner("RAG 학습 중..."):
                    # 진행 상황 표시
                    progress_text = st.empty()
                    progress_text.text("파일 처리 중...")

                    # 파일 확장자 확인
                    file_extension = uploaded_file.name.split(".")[-1].lower()

                    # 임시 파일로 저장
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    # 문서 처리
                    documents = process_for_rag(temp_path, content_type=file_extension)

                    # 임시 파일 삭제
                    os.remove(temp_path)

                    if documents:
                        st.success(
                            f"RAG 학습 완료! {len(documents)}개의 청크가 처리되었습니다."
                        )
                    else:
                        st.error("문서 처리 중 오류가 발생했습니다.")
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")
        else:
            st.warning("파일을 업로드해주세요.")
