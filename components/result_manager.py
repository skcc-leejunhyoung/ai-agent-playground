# components/result_manager.py

import streamlit as st
import time

from database import get_results_by_project
from components.evaluation_and_improvement import evaluation_and_improvement


##########


def result_manager(project_id):

    all_results = get_results_by_project(project_id)

    if "selection_change_counter" not in st.session_state:
        st.session_state["selection_change_counter"] = 0

    dynamic_key = (
        f"selection_segmented_control_{st.session_state['selection_change_counter']}"
    )

    project_dict = dict(st.session_state["project"])
    current_session_id = project_dict.get("current_session_id", 0)

    session_ids = sorted(set(r["session_id"] for r in all_results))
    if not session_ids:
        st.warning("해당 프로젝트에 실행된 세션이 없습니다.")
        return
    min_session_id = min(session_ids)
    max_session_id = max(session_ids)

    option_map = {
        0: ":material/docs:",
        1: ":material/Book_4_Spark:",
        2: ":material/arrow_back_iOS:",
        3: ":material/arrow_forward_iOS:",
    }

    selection = st.segmented_control(
        "Tool",
        options=option_map.keys(),
        format_func=lambda option: option_map[option],
        selection_mode="single",
        default=None,
        key=f"selection_segmented_control_{dynamic_key}",
        label_visibility="collapsed",
    )

    if selection == 0:
        current_session_results = [
            r for r in all_results if r["session_id"] == current_session_id
        ]

        if not current_session_results:
            st.warning(f"Session #{current_session_id}의 결과가 없습니다.")
        else:
            st.dataframe(reversed(current_session_results), use_container_width=True)

    if selection == 1:
        current_session_results = [
            r for r in all_results if r["session_id"] == current_session_id
        ]
        evaluation_and_improvement(
            current_session_id, current_session_results, project_id
        )
        # st.session_state["selection_change_counter"] += 1
        # st.session_state["selection"] = None

    if selection == 2:
        if current_session_id <= min_session_id:
            st.toast("First Session", icon="⚠️")
            time.sleep(0.3)
        else:
            new_session_id = current_session_id - 1
            project_dict["current_session_id"] = new_session_id
            st.session_state["project"] = project_dict

            latest_results = [
                r for r in all_results if r["session_id"] == new_session_id
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

        st.session_state["selection_change_counter"] += 1
        st.session_state["selection"] = None
        st.rerun()

    if selection == 3:
        if current_session_id >= max_session_id:
            st.toast("Last Session", icon="⚠️")
            time.sleep(0.3)
        else:
            new_session_id = current_session_id + 1
            project_dict["current_session_id"] = new_session_id
            st.session_state["project"] = project_dict

            latest_results = [
                r for r in all_results if r["session_id"] == new_session_id
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

        st.session_state["selection_change_counter"] += 1
        st.session_state["selection"] = None
        st.rerun()
