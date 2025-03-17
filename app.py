# app.py

import streamlit as st
from database import (
    init_db,
    get_projects,
    create_project,
    add_system_prompt,
    add_user_prompt,
    add_model,
)

##########

init_db()

st.set_page_config(page_title="프로젝트 선택", layout="wide")

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
    st.toast("기본 프로젝트가 생성되었습니다!")
    projects = get_projects()


##########


st.markdown(
    '<h1 class="centered-title">📁 프로젝트 선택하기</h1>', unsafe_allow_html=True
)
st.divider()

html_cards = """<div class="card-container">"""

for proj in projects:
    proj_dict = dict(proj)
    project_name = proj_dict["project_name"]
    project_id = proj_dict["project_id"]
    session_id = proj_dict.get("current_session_id", 0)
    description = proj_dict.get("description", "")

    html_cards += f"""
    <a href="/main_playground?selected_project={project_id}" class="project-card">
        <h2>{project_name}</h2>
        <i>{description}</i>
    </a>
    """

html_cards += """
    <a href="?add_project=true" class="project-card add">
        ➕
    </a>
</div>
"""

st.html(html_cards)


##########


query_params = st.query_params
if query_params.get("selected_project"):
    selected_project_id = query_params.get("selected_project")
    selected_project = next(
        (dict(p) for p in projects if str(p["project_id"]) == selected_project_id),
        None,
    )
    if selected_project:
        st.session_state["project"] = selected_project
        st.success(f"'{selected_project['project_name']}' 프로젝트가 선택되었습니다!")
        st.page_link("main_playground", label="Go to Playground")

if query_params.get("add_project") or st.session_state.get("show_modal"):
    st.session_state["show_modal"] = True
    with st.container():
        st.markdown(
            """
            <div class="modal-overlay">
                <div class="modal-content">
                    <h2>새 프로젝트 생성</h2>
            """,
            unsafe_allow_html=True,
        )
        new_project_name = st.text_input(
            "프로젝트 이름",
            key="modal_project_name",
            placeholder="예: GPT-테스트용 프로젝트",
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 생성", key="modal_create", use_container_width=True):
                if new_project_name:
                    project_id = create_project(new_project_name)
                    st.success(
                        f"✅ 프로젝트 '{new_project_name}' 생성 완료! (ID: {project_id})"
                    )
                    st.session_state["show_modal"] = False
                    st.experimental_rerun()
                else:
                    st.warning("⚠️ 프로젝트 이름을 입력하세요!")
        with col2:
            if st.button("❌ 닫기", key="modal_close", use_container_width=True):
                st.session_state["show_modal"] = False
                st.experimental_rerun()
        st.markdown("</div></div>", unsafe_allow_html=True)
