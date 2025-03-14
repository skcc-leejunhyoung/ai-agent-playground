# components/result_manager.py

import streamlit as st

from database import get_results_by_project


def result_manager(project_id):
    option_map = {
        0: ":material/docs:",
        1: ":material/zoom_in:",
        2: ":material/zoom_out:",
        3: ":material/zoom_out_map:",
    }

    selection = st.segmented_control(
        "Tool",
        options=option_map.keys(),
        format_func=lambda option: option_map[option],
        selection_mode="single",
        label_visibility="collapsed",
    )

    if selection == 0:
        project_dict = dict(st.session_state["project"])
        current_session_id = project_dict.get("current_session_id", 0)

        all_results = get_results_by_project(project_id)

        current_session_results = [
            r for r in all_results if r["session_id"] == current_session_id
        ]

        if not current_session_results:
            st.warning(f"Session #{current_session_id}의 결과가 없습니다.")
        else:
            st.dataframe(reversed(current_session_results), use_container_width=True)
