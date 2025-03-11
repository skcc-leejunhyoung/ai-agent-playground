# components/result_card.py

import streamlit.components.v1 as components
import markdown


def result_cards(
    card_contents: list[str],
    system_count: int,
    user_count: int,
    model_names: list[str],
    height: int = 300,
):
    """
    결과 카드 리스트를 렌더링하는 함수 (카드 클릭 시 모달 팝업)

    :param card_contents: 카드에 들어갈 결과 텍스트 리스트
    :param system_count: 시스템 프롬프트 개수
    :param user_count: 유저 프롬프트 개수
    :param model_names: 모델 이름 리스트
    :param height: 카드 전체 영역 높이
    """

    cards_html = """
        <style>
        body, html {
            font-family: "Inter", system-ui, "Roboto", sans-serif;
        }

        .scrolling-wrapper {
            display: flex;
            flex-wrap: nowrap;
            overflow-x: auto;
            padding: 1rem;
            -ms-overflow-style: none;
            scrollbar-width: none;
        }

        .scrolling-wrapper::-webkit-scrollbar {
            display: none;
        }

        .card {
            flex: 0 0 auto;
            background-color: #21252B;
            color: #ABB2BF;
            border-radius: 10px;
            border: 1px solid #333;
            width: 300px;
            height: 450px;
            margin-right: 1rem;
            padding: 1rem;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            max-height: 450px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
            cursor: pointer;
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        }

        .modal {
            display: none;
            position: fixed;
            z-index: 9999;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.7);
        }

        .modal-content {
            background-color: #2B2B2B;
            color: #ABB2BF;
            margin: 10% auto;
            padding: 15px;
            border: 1px solid #888;
            width: 80%;
            border-radius: 10px;
            position: relative;
        }

        .close {
            color: #aaa;
            position: absolute;
            top: 10px;
            right: 25px;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }

        .close:hover,
        .close:focus {
            color: #fff;
            text-decoration: none;
            cursor: pointer;
        }

        </style>

        <div class="scrolling-wrapper">
    """

    modals = ""

    model_count = len(model_names)
    total_per_model = system_count * user_count

    for idx, content in enumerate(card_contents, 1):
        html_content = markdown.markdown(content)
        modal_id = f"modal_{idx}"

        # 모델, 시스템, 유저 인덱스 계산
        model_idx = (idx - 1) // total_per_model
        model_name = (
            model_names[model_idx]
            if model_idx < model_count
            else f"Model{model_idx + 1}"
        )

        sys_num = ((idx - 1) % total_per_model) // user_count + 1
        user_num = ((idx - 1) % user_count) + 1

        title = f"{model_name} + Sys{sys_num} + User{user_num}"

        cards_html += f"""
            <div class="card" onclick="openModal('{modal_id}')">
                <h4 style="color: #61AFEF;">{title}</h4>
                {html_content}
            </div>
        """

        modals += f"""
            <div id="{modal_id}" class="modal">
                <div class="modal-content">
                    <span class="close" onclick="closeModal('{modal_id}')">&times;</span>
                    <h2 style="color: #61AFEF;">{title} 상세 보기</h2>
                    {html_content}
                </div>
            </div>
        """

    cards_html += "</div>"
    cards_html += modals

    cards_html += """
        <script>
        function openModal(id) {
            var modal = document.getElementById(id);
            if (modal) {
                modal.style.display = "block";
            }
        }

        function closeModal(id) {
            var modal = document.getElementById(id);
            if (modal) {
                modal.style.display = "none";
            }
        }

        window.addEventListener("keydown", function(event) {
            if (event.key === "Escape") {
                var modals = document.getElementsByClassName("modal");
                for (var i = 0; i < modals.length; i++) {
                    modals[i].style.display = "none";
                }
            }
        });
        </script>
    """

    components.html(cards_html, height=height, scrolling=False)
