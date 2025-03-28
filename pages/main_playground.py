# pages/main_playground.py

import streamlit as st

from components.result_card import result_cards
from components.ai_settings import ai_settings_ui
from components.result_manager import result_manager

from database import (
    get_projects,
    get_results_by_project,
    export_project_csv,
)


##########

st.set_page_config(page_title="Prompt Playground", layout="wide")

##########

if "results" not in st.session_state:
    st.session_state["results"] = []
    st.session_state["system_count"] = 1
    st.session_state["user_count"] = 1
    st.session_state["model_names"] = []

##########

query_params = st.query_params
if "project" not in st.session_state and query_params.get("selected_project"):
    projects = get_projects()
    selected_project_id = query_params.get("selected_project")
    selected_project = next(
        (dict(p) for p in projects if str(p["project_id"]) == selected_project_id),
        None,
    )
    if selected_project:
        st.session_state["project"] = selected_project

if "project" not in st.session_state:
    st.error("프로젝트가 선택되지 않았습니다. 프로젝트 선택 페이지로 돌아가세요.")
    st.switch_page("app.py")

project = st.session_state["project"]
project_id = project["project_id"]

##########

css = """
<style>
    [data-testid="stSidebar"][aria-expanded="true"] {
        max-width: calc(100vw - 600px) !important;
        min-width: 380px;
        width: calc(100vw - 600px);
        background-color: #202833;
    }

    [data-testid="stSidebar"][aria-expanded="false"] {
        display: none;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

##########

st.header("Prompt Playground", divider="blue")

with st.container():
    ai_settings_ui(project_id)

results_data = get_results_by_project(project_id)
if results_data:
    csv_data = export_project_csv(project_id)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(
            "Back to Project Select", type="tertiary", use_container_width=True
        ):
            st.switch_page("app.py")
    with col2:
        st.download_button(
            label="Export CSV",
            data=csv_data,
            file_name=f"{st.session_state['project']['project_name']}_export.csv",
            mime="text/csv",
            type="tertiary",
            use_container_width=True,
        )
    with col3:
        if st.button("View History", type="tertiary", use_container_width=True):
            latest_session_id = max(result["session_id"] for result in results_data)

            latest_results = [
                r for r in results_data if r["session_id"] == latest_session_id
            ]

            st.session_state["results"] = [
                {
                    "result": r["result"],
                    "eval_pass": r["eval_pass"],
                    "eval_method": r["eval_method"],
                    "eval_keyword": r["eval_keyword"],
                }
                for r in latest_results
            ]

            st.session_state["system_count"] = len(
                set(r["system_prompt"] for r in latest_results)
            )
            st.session_state["user_count"] = len(
                set(r["user_prompt"] for r in latest_results)
            )
            st.session_state["model_names"] = list(
                set(r["model"] for r in latest_results)
            )

            st.toast(f"Run #{latest_session_id} Results Loaded")
else:
    csv_data = export_project_csv(project_id)
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "Back to Project Select", type="tertiary", use_container_width=True
        ):
            st.switch_page("app.py")
    with col2:
        st.download_button(
            label="Export CSV",
            data=csv_data,
            file_name=f"{st.session_state['project']['project_name']}_export.csv",
            mime="text/csv",
            type="tertiary",
            use_container_width=True,
        )


if st.session_state["results"]:
    with st.sidebar:
        container = st.container(height=9, border=False)
        with st.sidebar:
            session_id = project.get("current_session_id", 0)
            st.markdown(
                f"""
                <h2 style='font-size: 36px; color: white; margin-bottom: 0;'>
                    Run #{session_id}
                </h2>
                <hr style='margin-top: 0; border: 1px solid #ccc;' />
                """,
                unsafe_allow_html=True,
            )
        result_cards(
            st.session_state["results"],
            system_count=st.session_state["system_count"],
            user_count=st.session_state["user_count"],
            model_names=st.session_state["model_names"],
            height=500,
        )
        result_manager(project_id)
