# components/result_card.py

import streamlit.components.v1 as components
import markdown
import re


def result_cards(
    card_contents: list[dict],
    system_count: int,
    user_count: int,
    model_names: list[str],
    height: int = 300,
):
    card_contents = list(reversed(card_contents))

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
            margin: 5% auto;
            padding: 20px;
            border: 1px solid #888;
            width: 80%;
            border-radius: 10px;
            position: relative;
            max-height: 80%;
            overflow-y: auto;
        }

        .highlight {
            font-weight: bold;
            padding: 2px 4px;
            border-radius: 3px;
        }

        .highlight-0 { background-color: #FFD700; color: #000; }
        .highlight-1 { background-color: #FF8C00; color: #000; }
        .highlight-2 { background-color: #FF69B4; color: #000; }
        .highlight-3 { background-color: #1E90FF; color: #fff; }
        .highlight-4 { background-color: #32CD32; color: #000; }

        .keyword-section {
            margin-top: 20px;
            padding: 15px;
            background-color: #333;
            border: 1px solid #555;
            border-radius: 8px;
        }

        .keyword-section h4 {
            margin-top: 0;
            color: #61AFEF;
        }

        .keyword-section p {
            margin: 0;
            color: #ABB2BF;
            font-size: 14px;
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

    for idx, data in enumerate(card_contents, 1):
        content = data.get("result", "")
        eval_pass = data.get("eval_pass", "")
        eval_method = data.get("eval_method", "")
        eval_keyword = data.get("eval_keyword", "")

        if eval_pass == "O":
            card_bg_color = "#152217"
        elif eval_pass == "X":
            card_bg_color = "#321516"
        else:
            card_bg_color = "#21252B"

        highlighted_content = content

        keyword_box_html = ""
        if eval_method == "rule" and eval_keyword:
            keywords = [kw.strip() for kw in eval_keyword.split(",") if kw.strip()]

            for i, kw in enumerate(keywords):
                css_class = f"highlight-{i % 5}"
                highlighted_content = re.sub(
                    f"({re.escape(kw)})",
                    rf"<span class='{css_class}'><strong>\1</strong></span>",
                    highlighted_content,
                    flags=re.IGNORECASE,
                )

            keyword_list_html = ", ".join(keywords)
            keyword_box_html = f"""
                <div class="keyword-section">
                    <h4>Eval Keywords</h4>
                    <p>{keyword_list_html}</p>
                </div>
            """

        html_content = markdown.markdown(highlighted_content)
        modal_id = f"modal_{idx}"

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
            <div class="card" onclick="openModal('{modal_id}')"
                 style="background-color: {card_bg_color};">
                <h4 style="color: #61AFEF;">{title}</h4>
                {markdown.markdown(content)}
            </div>
        """

        modals += f"""
            <div id="{modal_id}" class="modal">
                <div class="modal-content">
                    <span class="close" onclick="closeModal('{modal_id}')">&times;</span>
                    <h3 style="color: #61AFEF;">{title}</h3>
                    <div>{html_content}</div>
                    {keyword_box_html}
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
