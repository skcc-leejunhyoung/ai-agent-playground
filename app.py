import streamlit as st
from st_diff_viewer import diff_viewer
from streamlit_monaco import st_monaco
import uuid

from model import stream_chat
from components.result_card import result_cards
from database import (
    init_db,
    create_project,
    get_projects,
    add_system_prompt,
    get_system_prompts,
    add_user_prompt,
    get_user_prompts,
    add_model,
    get_models,
    add_result,
    update_system_prompt,
    update_user_prompt,
)

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

st.set_page_config(page_title="Prompt Playground", layout="wide")

init_db()

if "results" not in st.session_state:
    st.session_state["results"] = []
    st.session_state["system_count"] = 1
    st.session_state["user_count"] = 1
    st.session_state["model_names"] = []


def generate_uuid():
    return str(uuid.uuid4().int)[:18]


##########

st.header("Prompt Playground", divider="blue")

header_col1, header_col2 = st.columns([8, 1.5], vertical_alignment="center")

with header_col1:
    st.write("_프롬프팅 에이전트를 결합한 playground_")

with header_col2:
    model_count = 2 if st.session_state.get("model_toggle") else 1
    system_count = (
        len(st.session_state.get("system_prompts", ["dummy"]))
        if st.session_state.get("system_toggle")
        else 1
    )
    user_count = (
        len(st.session_state.get("user_prompts", ["dummy"]))
        if st.session_state.get("user_toggle")
        else 1
    )

    total_runs = model_count * system_count * user_count
    execute_label = f"**:red[{total_runs} | Execute]**"
    execute_button = st.button(execute_label, use_container_width=True)


##########

with st.sidebar:
    st.header("Project Management")
    projects = get_projects()
    default_project = None
    for proj in projects:
        if proj["project_name"] == "default":
            default_project = proj
            break
    if not default_project:
        project_id = create_project("default")
        add_system_prompt("You are helpful assistant!", project_id)
        add_system_prompt(
            "You always answer 'meow' regardless of the question.", project_id
        )
        add_user_prompt("What is the capital of France?", project_id)
        add_user_prompt("What is the currency of USA?", project_id)
        add_model("GPT-4o", project_id)
        add_model("GPT-4o mini", project_id)
        st.toast("Default 프로젝트가 생성되었습니다.", icon="✨")
        default_project = {
            "project_id": project_id,
            "project_name": "default",
            "current_session_id": 0,
        }
    else:
        project_id = default_project["project_id"]
    st.write(
        f"Selected project: **{default_project['project_name']}** (ID: {project_id})"
    )
    st.session_state["project"] = default_project


##########

with st.expander("System Prompt", expanded=False):
    system_compare_toggle = st.toggle("다중 System Prompt 활성화", key="system_toggle")
    if system_compare_toggle:
        db_system_prompts = get_system_prompts(project_id)
        if "excluded_system_prompt_ids" not in st.session_state:
            st.session_state["excluded_system_prompt_ids"] = []
        st.session_state["system_prompts"] = [
            {"id": row["id"], "prompt": row["prompt"]}
            for row in db_system_prompts
            if row["id"] not in st.session_state["excluded_system_prompt_ids"]
        ]

        if "system_select_1_idx" not in st.session_state:
            st.session_state["system_select_1_idx"] = 0
        if "system_select_2_idx" not in st.session_state:
            st.session_state["system_select_2_idx"] = 1

        prompt_count = len(st.session_state["system_prompts"])

        available_idx_for_1 = [
            i
            for i in range(prompt_count)
            if i != st.session_state["system_select_2_idx"]
        ]
        available_idx_for_2 = [
            i
            for i in range(prompt_count)
            if i != st.session_state["system_select_1_idx"]
        ]

        col1, col2 = st.columns(2)
        with col1:
            c1, c2 = st.columns([10, 1])
            with c1:
                idx_1 = st.selectbox(
                    "sys 1 선택",
                    options=available_idx_for_1,
                    index=(
                        available_idx_for_1.index(
                            st.session_state["system_select_1_idx"]
                        )
                        if st.session_state["system_select_1_idx"]
                        in available_idx_for_1
                        else 0
                    ),
                    format_func=lambda x: f"{x+1}",
                    key="system_select_1",
                    label_visibility="collapsed",
                )
                st.session_state["system_select_1_idx"] = idx_1
            with c2:
                btn_update_sys_1 = st.button(
                    ":material/save:",
                    type="tertiary",
                    use_container_width=True,
                    key=f"update_sys_{idx_1}_top",
                )

            editor_1 = st_monaco(
                value=st.session_state["system_prompts"][idx_1]["prompt"],
                language="markdown",
                height="300px",
                theme="vs-dark",
            )

            if editor_1 is not None:
                st.session_state["system_prompts"][idx_1]["prompt"] = editor_1

            if btn_update_sys_1:
                update_system_prompt(
                    st.session_state["system_prompts"][idx_1]["id"],
                    st.session_state["system_prompts"][idx_1]["prompt"],
                )
                st.toast("Left System Prompt updated")

        with col2:
            c1, c2, c3, c4 = st.columns([15, 1, 1, 1])
            with c1:
                idx_2 = st.selectbox(
                    "sys 2 선택",
                    options=available_idx_for_2,
                    index=(
                        available_idx_for_2.index(
                            st.session_state["system_select_2_idx"]
                        )
                        if st.session_state["system_select_2_idx"]
                        in available_idx_for_2
                        else 0
                    ),
                    format_func=lambda x: f"{x+1}",
                    key="system_select_2",
                    label_visibility="collapsed",
                )
                st.session_state["system_select_2_idx"] = idx_2
            with c2:
                editor_2_btn = st.button(
                    ":material/save:",
                    type="tertiary",
                    use_container_width=True,
                    key=f"update_sys_{idx_2}_top",
                )
            with c3:
                if st.button(
                    ":material/add:",
                    type="tertiary",
                    use_container_width=True,
                    key="add_system_prompt",
                ):
                    new_prompt = f"You are helpful assistant! #{generate_uuid()}"
                    new_id = add_system_prompt(new_prompt, project_id)
                    st.session_state["system_prompts"].append(
                        {"id": new_id, "prompt": new_prompt}
                    )
                    st.session_state["system_select_2_idx"] = (
                        len(st.session_state["system_prompts"]) - 1
                    )
                    st.rerun()
            with c4:
                if st.button(
                    ":material/remove:",
                    type="tertiary",
                    use_container_width=True,
                    key="remove_system_prompt",
                ):
                    remove_idx = st.session_state["system_select_2_idx"]
                    if len(st.session_state["system_prompts"]) <= 2:
                        st.toast("시스템 프롬프트는 최소 2개 필요합니다.", icon="⚠️")
                    else:
                        removed = st.session_state["system_prompts"].pop(remove_idx)
                        st.session_state.setdefault(
                            "excluded_system_prompt_ids", []
                        ).append(removed["id"])
                        new_len = len(st.session_state["system_prompts"])
                        st.session_state["system_select_2_idx"] = max(
                            0, min(remove_idx, new_len - 1)
                        )
                        if (
                            st.session_state["system_select_1_idx"]
                            == st.session_state["system_select_2_idx"]
                        ):
                            st.session_state["system_select_1_idx"] = (
                                st.session_state["system_select_2_idx"] + 1
                            ) % new_len
                        st.rerun()

            editor_2 = st_monaco(
                value=st.session_state["system_prompts"][idx_2]["prompt"],
                language="markdown",
                height="300px",
                theme="vs-dark",
            )
            if editor_2 is not None:
                st.session_state["system_prompts"][idx_2]["prompt"] = editor_2

            if editor_2_btn:
                update_system_prompt(
                    st.session_state["system_prompts"][idx_2]["id"],
                    st.session_state["system_prompts"][idx_2]["prompt"],
                )
                st.toast("Right System Prompt updated")
        split_view_sys = st.toggle("Split View", value=True, key="system_split_view")
        diff_viewer(
            st.session_state["system_prompts"][idx_1]["prompt"],
            st.session_state["system_prompts"][idx_2]["prompt"],
            split_view=split_view_sys,
            use_dark_theme=True,
            styles=one_dark_pro_styles,
        )
    else:
        system_single_text = st_monaco(
            value="You are helpful assistant!",
            language="markdown",
            height="300px",
            theme="vs-dark",
        )


##########


with st.expander("User Prompt", expanded=False):
    user_compare_toggle = st.toggle("다중 User Prompt 활성화", key="user_toggle")
    if user_compare_toggle:
        db_user_prompts = get_user_prompts(project_id)
        if "excluded_user_prompt_ids" not in st.session_state:
            st.session_state["excluded_user_prompt_ids"] = []
        st.session_state["user_prompts"] = [
            {"id": row["id"], "prompt": row["prompt"]}
            for row in db_user_prompts
            if row["id"] not in st.session_state["excluded_user_prompt_ids"]
        ]
        if "user_select_1_idx" not in st.session_state:
            st.session_state["user_select_1_idx"] = 0
        if "user_select_2_idx" not in st.session_state:
            st.session_state["user_select_2_idx"] = 1

        prompt_count = len(st.session_state["user_prompts"])
        available_idx_for_1 = [
            i for i in range(prompt_count) if i != st.session_state["user_select_2_idx"]
        ]
        available_idx_for_2 = [
            i for i in range(prompt_count) if i != st.session_state["user_select_1_idx"]
        ]

        col1, col2 = st.columns(2)
        with col1:
            c1, c2 = st.columns([10, 1])
            with c1:
                idx_1 = st.selectbox(
                    "usr 1 선택",
                    options=available_idx_for_1,
                    index=(
                        available_idx_for_1.index(st.session_state["user_select_1_idx"])
                        if st.session_state["user_select_1_idx"] in available_idx_for_1
                        else 0
                    ),
                    format_func=lambda x: f"{x+1}",
                    key="user_select_1",
                    label_visibility="collapsed",
                )
                st.session_state["user_select_1_idx"] = idx_1
            with c2:
                btn_update_usr_1 = st.button(
                    ":material/save:",
                    type="tertiary",
                    use_container_width=True,
                    key=f"update_usr_{idx_1}_top",
                )
            editor_1 = st_monaco(
                value=st.session_state["user_prompts"][idx_1]["prompt"],
                language="markdown",
                height="300px",
                theme="vs-dark",
            )
            if editor_1 is not None:
                st.session_state["user_prompts"][idx_1]["prompt"] = editor_1
            if btn_update_usr_1:
                update_user_prompt(
                    st.session_state["user_prompts"][idx_1]["id"],
                    st.session_state["user_prompts"][idx_1]["prompt"],
                )
                st.toast("Left User Prompt updated")
        with col2:
            c1, c2, c3, c4 = st.columns([15, 1, 1, 1])
            with c1:
                idx_2 = st.selectbox(
                    "usr 2 선택",
                    options=available_idx_for_2,
                    index=(
                        available_idx_for_2.index(st.session_state["user_select_2_idx"])
                        if st.session_state["user_select_2_idx"] in available_idx_for_2
                        else 0
                    ),
                    format_func=lambda x: f"{x+1}",
                    key="user_select_2",
                    label_visibility="collapsed",
                )
                st.session_state["user_select_2_idx"] = idx_2
            with c2:
                editor_2_btn = st.button(
                    ":material/save:",
                    type="tertiary",
                    use_container_width=True,
                    key=f"update_usr_{idx_2}_top",
                )
            with c3:
                if st.button(
                    ":material/add:",
                    type="tertiary",
                    use_container_width=True,
                    key="add_user_prompt",
                ):
                    new_prompt = f"User Prompt #{generate_uuid()}"
                    add_user_prompt(new_prompt, project_id)
                    st.session_state["user_prompts"].append(
                        {"id": None, "prompt": new_prompt}
                    )
                    st.session_state["user_select_2_idx"] = (
                        len(st.session_state["user_prompts"]) - 1
                    )
                    st.rerun()
            with c3:
                if st.button(
                    ":material/remove:",
                    type="tertiary",
                    use_container_width=True,
                    key="remove_user_prompt",
                ):
                    remove_idx = st.session_state["user_select_2_idx"]
                    if len(st.session_state["user_prompts"]) <= 2:
                        st.toast("유저 프롬프트는 최소 2개 필요합니다.", icon="⚠️")
                    else:
                        removed = st.session_state["user_prompts"].pop(remove_idx)
                        st.session_state.setdefault(
                            "excluded_user_prompt_ids", []
                        ).append(removed["id"])
                        new_len = len(st.session_state["user_prompts"])
                        st.session_state["user_select_2_idx"] = max(
                            0, min(remove_idx, new_len - 1)
                        )
                        if (
                            st.session_state["user_select_1_idx"]
                            == st.session_state["user_select_2_idx"]
                        ):
                            st.session_state["user_select_1_idx"] = (
                                st.session_state["user_select_2_idx"] + 1
                            ) % new_len
                        st.rerun()

            editor_2 = st_monaco(
                value=st.session_state["user_prompts"][idx_2]["prompt"],
                language="markdown",
                height="300px",
                theme="vs-dark",
            )
            if editor_2 is not None:
                st.session_state["user_prompts"][idx_2]["prompt"] = editor_2
            if editor_2_btn:
                update_user_prompt(
                    st.session_state["user_prompts"][idx_2]["id"],
                    st.session_state["user_prompts"][idx_2]["prompt"],
                )
                st.toast("Right User Prompt updated")
        split_view_usr = st.toggle("Split View", value=True, key="user_split_view")
        diff_viewer(
            st.session_state["user_prompts"][idx_1]["prompt"],
            st.session_state["user_prompts"][idx_2]["prompt"],
            split_view=split_view_usr,
            use_dark_theme=True,
            styles=one_dark_pro_styles,
        )
    else:
        user_single_text = st_monaco(
            value="Harry James Potter, Hermione Jean Granger, Ronald Bilius Weasley 중에서 'r'이 가장 많이 들어간 단어는 뭐야?",
            language="markdown",
            height="300px",
            theme="vs-dark",
        )


##########


with st.expander("Model", expanded=False):
    model_compare_toggle = st.toggle("다중 Model 활성화", key="model_toggle")
    if model_compare_toggle:
        db_models = get_models(project_id)
        st.session_state["model_names"] = [row["model_name"] for row in db_models]
        if not st.session_state["model_names"]:
            add_model("GPT-4o", project_id)
            add_model("GPT-4o mini", project_id)
            st.session_state["model_names"] = ["GPT-4o", "GPT-4o mini"]

        col1, col2 = st.columns(2)
        with col1:
            model_1 = st.selectbox(
                "모델 선택 1",
                st.session_state["model_names"],
                index=0,
                key="model_selector_1",
                label_visibility="collapsed",
            )
        with col2:
            model_2 = st.selectbox(
                "모델 선택 2",
                st.session_state["model_names"],
                index=1 if len(st.session_state["model_names"]) > 1 else 0,
                key="model_selector_2",
                label_visibility="collapsed",
            )
        model_list = [model_1, model_2]
    else:
        single_model = st.selectbox(
            "모델 선택",
            ("GPT-4o", "GPT-4o mini"),
            index=0,
            key="model_selector_single",
            label_visibility="collapsed",
        )
        model_list = [single_model]


##########


if execute_button:
    if system_compare_toggle:
        system_prompts = st.session_state["system_prompts"]
    else:
        system_prompts = [{"prompt": system_single_text or ""}]

    if user_compare_toggle:
        user_prompts = st.session_state["user_prompts"]
    else:
        user_prompts = [{"prompt": user_single_text or ""}]

    st.info(
        f"|| 모델: {len(model_list)}개 || 시스템 프롬프트: {len(system_prompts)}개 || 유저 프롬프트: {len(user_prompts)}개 || 조합으로 실행됩니다."
    )

    project_dict = dict(st.session_state["project"])
    current_session_id = project_dict.get("current_session_id", 0)
    new_session_id = current_session_id + 1
    from database import update_project_session_id

    update_project_session_id(project_id, new_session_id)
    project_dict["current_session_id"] = new_session_id
    st.session_state["project"] = project_dict

    st.session_state["results"] = []
    st.session_state["system_count"] = len(system_prompts)
    st.session_state["user_count"] = len(user_prompts)
    st.session_state["model_names"] = model_list

    total_combinations = len(model_list) * len(system_prompts) * len(user_prompts)
    current_run = 1

    toast_msg = st.toast(f"[{current_run}/{total_combinations}] 실행을 시작합니다...")

    for model_name in model_list:
        use_gpt4o_mini = model_name == "GPT-4o mini"
        for sys_idx, sys_prompt in enumerate(system_prompts, 1):
            for user_idx, user_prompt in enumerate(user_prompts, 1):
                toast_msg.toast(
                    f"🚀 [{current_run}/{total_combinations}] {model_name} | Sys{sys_idx} + User{user_idx} 실행 중..."
                )
                full_response = ""
                for token in stream_chat(
                    system_prompt=sys_prompt["prompt"],
                    user_prompt=user_prompt["prompt"],
                    use_gpt4o_mini=use_gpt4o_mini,
                ):
                    full_response += token
                st.session_state["results"].append(full_response)
                add_result(
                    sys_prompt["prompt"],
                    user_prompt["prompt"],
                    model_name,
                    full_response,
                    new_session_id,
                    project_id,
                )
                toast_msg.toast(
                    f"[{current_run}/{total_combinations}] {model_name} | Sys{sys_idx} + User{user_idx} 실행 완료!"
                )
                current_run += 1

    toast_msg.toast(f":green[모든 실행이 완료되었습니다!]", icon="🎉")


##########


if st.session_state["results"]:
    result_cards(
        st.session_state["results"],
        system_count=st.session_state["system_count"],
        user_count=st.session_state["user_count"],
        model_names=st.session_state["model_names"],
        height=500,
    )
