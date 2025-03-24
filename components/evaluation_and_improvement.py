# components/evaluation_and_improvement.py

import streamlit as st
import time
from streamlit_monaco import st_monaco

from agent.eval_agent import run_eval_agent
from database import add_system_prompt, update_system_prompt


def get_eval_state_key(current_session_id: int, project_id: str) -> str:
    return f"eval_state_{project_id}_{current_session_id}"


def evaluation_and_improvement(
    current_session_id: int, current_session_results: list, project_id: str
):
    eval_state_key = get_eval_state_key(current_session_id, project_id)

    if eval_state_key in st.session_state:
        eval_state = st.session_state[eval_state_key]
        st.toast(f"Session #{current_session_id}의 저장된 평가 결과를 불러왔습니다.")
    else:
        st.toast("세션 평가 및 개선을 시작합니다...")

        if not current_session_results:
            st.warning(f"Session #{current_session_id}의 결과가 없습니다.")
            return

        eval_state = run_eval_agent(current_session_results, project_id)

        if eval_state and eval_state["best_prompts"] is not None:
            st.session_state[eval_state_key] = eval_state

    if eval_state and eval_state["best_prompts"] is not None:
        st.success(f"Session #{current_session_id} 평가가 완료되었습니다!")

        col1, col2 = st.columns([0.2, 0.8])
        with col1:
            if st.button("재평가", key=f"reevaluate_{current_session_id}"):
                if eval_state_key in st.session_state:
                    del st.session_state[eval_state_key]
                st.rerun()

        st.subheader(":blue[모델별] 최적 시스템 프롬프트 결과")
        st.dataframe(eval_state["best_prompts"], use_container_width=True)

        improved_prompt = eval_state.get("improved_prompt")

        if improved_prompt:
            st.subheader(":blue[개선된] 시스템 프롬프트 및 모델 :blue[제안]")
            st.markdown(f"최적 모델: `{improved_prompt['model']}`")
            c1, c2 = st.columns([8, 2])
            with c1:
                st.markdown(":orange[**개선된 시스템 프롬프트**]")
            with c2:
                system_prompt_editor_btn = st.button(
                    ":material/add:",
                    type="tertiary",
                    use_container_width=True,
                    key=f"update_sys_prompt",
                )

            if "improved_prompt_id" not in eval_state:
                eval_state["improved_prompt_id"] = None

            system_prompt_editor = st_monaco(
                value=improved_prompt["improved_prompt"],
                language="markdown",
                height="150px",
                theme="vs-dark",
            )

            if system_prompt_editor_btn and system_prompt_editor is not None:
                if eval_state["improved_prompt_id"] is None:
                    prompt_id = add_system_prompt(system_prompt_editor, project_id)
                    eval_state["improved_prompt_id"] = prompt_id
                    st.toast("새로운 시스템 프롬프트가 저장되었습니다.")
                    time.sleep(0.4)
                else:
                    update_system_prompt(
                        eval_state["improved_prompt_id"], system_prompt_editor
                    )
                    st.toast("시스템 프롬프트가 업데이트되었습니다.")
                    time.sleep(0.4)
                st.rerun()

            st.markdown(":orange[**개선 이유 및 방향성**]  ")
            st.markdown(improved_prompt["reason"])
        else:
            st.info("개선할 프롬프트가 없습니다.")
    else:
        st.warning(f"Session #{current_session_id} 평가 결과가 없습니다.")
