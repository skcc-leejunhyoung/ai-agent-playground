import streamlit as st
from agent.eval_agent import run_eval_agent


def get_eval_state_key(current_session_id: int, project_id: str) -> str:
    return f"eval_state_{project_id}_{current_session_id}"


def evaluation_and_improvement(
    current_session_id: int, current_session_results: list, project_id: str
):
    eval_state_key = get_eval_state_key(current_session_id, project_id)

    # 이미 저장된 평가 결과가 있는지 확인
    if eval_state_key in st.session_state:
        eval_state = st.session_state[eval_state_key]
        st.toast(f"Session #{current_session_id}의 저장된 평가 결과를 불러왔습니다.")
    else:
        st.toast("세션 평가 및 개선을 시작합니다...")

        if not current_session_results:
            st.warning(f"Session #{current_session_id}의 결과가 없습니다.")
            return

        eval_state = run_eval_agent(current_session_results, project_id)

        # 평가 결과 저장
        if eval_state and eval_state["best_prompts"] is not None:
            st.session_state[eval_state_key] = eval_state

    # 평가 결과 표시
    if eval_state and eval_state["best_prompts"] is not None:
        st.success(f"Session #{current_session_id} 평가가 완료되었습니다!")

        col1, col2 = st.columns([0.2, 0.8])
        with col1:
            if st.button("재평가", key=f"reevaluate_{current_session_id}"):
                # 세션 상태에서 기존 평가 결과 삭제
                if eval_state_key in st.session_state:
                    del st.session_state[eval_state_key]
                st.rerun()

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
        else:
            st.info("아직 개선된 프롬프트가 없습니다.")
    else:
        st.warning(f"Session #{current_session_id} 평가 결과가 없습니다.")
