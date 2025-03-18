# components/result_manager.py

import streamlit as st
import time

from agent.eval_agent import run_eval_agent
from database import get_results_by_project


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
        1: ":material/wand_shine:",
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
        st.toast("세션 평가 및 개선을 시작합니다...")

        current_session_results = [
            r for r in all_results if r["session_id"] == current_session_id
        ]

        if not current_session_results:
            st.warning(f"Session #{current_session_id}의 결과가 없습니다.")
        else:
            eval_state = run_eval_agent(current_session_results, project_id)

            if eval_state and eval_state["best_prompts"] is not None:
                st.success(f"Session #{current_session_id} 평가가 완료되었습니다!")

                st.subheader("모델별 최적 시스템 프롬프트 결과")
                st.dataframe(eval_state["best_prompts"], use_container_width=True)

                improved_prompt = eval_state.get("improved_prompt")

                if improved_prompt:
                    st.subheader("개선된 시스템 프롬프트 및 모델 제안")
                    st.markdown(
                        f"""
                        **모델명**: `{improved_prompt['model']}`  
                        **개선된 시스템 프롬프트**:  
                        ```
                        {improved_prompt['improved_prompt']}
                        ```
                        """
                    )
                    st.markdown("**개선 이유 및 방향성**:  ")
                    st.markdown(improved_prompt["reason"])
                    st.session_state["selection_change_counter"] += 1
                    st.session_state["selection"] = None
                else:
                    st.info("아직 개선된 프롬프트가 없습니다.")
                    st.session_state["selection_change_counter"] += 1
                    st.session_state["selection"] = None
            else:
                st.warning(f"Session #{current_session_id} 평가 결과가 없습니다.")
                st.session_state["selection_change_counter"] += 1
                st.session_state["selection"] = None

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
