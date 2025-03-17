# agent/eval_agent.py

import json
import re
from database import get_eval_data, update_eval_data, get_connection


def clean_text(text: str) -> str:
    """
    마크다운 및 특수문자 제거 함수
    - **Bold** -> Bold
    - 기타 특수문자 제거
    """
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def evaluate_response(response: str, keywords: list[str]) -> str:
    """
    주어진 응답(response)에 키워드가 모두 포함되면 'O', 하나라도 빠지면 'X' 반환
    - 대소문자 구분 없음
    - 특수문자 제거
    """
    normalized_response = clean_text(response).lower()

    for keyword in keywords:
        normalized_keyword = keyword.lower().strip()
        if normalized_keyword not in normalized_response:
            return "X"

    return "O"


def run_evaluation(project_id: int, session_id: int):
    """
    특정 프로젝트와 세션의 모든 result에 대해 평가 후 eval_pass 필드 업데이트

    규칙:
    - eval_method가 'pass'면 무조건 'P'
    - eval_method가 'rule'이면 eval_keyword 기준으로 result 평가하여 'O' 또는 'X'
    """
    eval_data = get_eval_data(project_id, session_id)
    print(
        f"[EVAL] 프로젝트 {project_id}, 세션 {session_id}: {len(eval_data)}개의 결과 평가 중..."
    )

    for data in eval_data:
        result_id = data["id"]

        with get_connection() as conn:
            cur = conn.cursor()

            cur.execute(
                """
                SELECT result, eval_method, eval_keyword
                FROM result
                WHERE id = ?
                """,
                (result_id,),
            )
            row = cur.fetchone()

            if row is None:
                continue

            result_content = row["result"]
            eval_method = row["eval_method"]
            eval_keyword = row["eval_keyword"]

        if eval_method == "pass":
            eval_pass = "P"

        elif eval_method == "rule":
            if "<|eot_id|>" in eval_keyword:
                keywords = [
                    kw.strip() for kw in eval_keyword.split("<|eot_id|>") if kw.strip()
                ]
            else:
                keywords = [kw.strip() for kw in eval_keyword.split(",") if kw.strip()]

            combined_text = result_content
            eval_pass = evaluate_response(combined_text, keywords)

        else:
            print(f"[EVAL] 알 수 없는 eval_method '{eval_method}' → 기본 'P' 처리")
            eval_pass = "X"

        update_eval_data(
            result_id,
            eval_pass=eval_pass,
            eval_method=eval_method,
            eval_keyword=eval_keyword,
        )

    print(f"[EVAL] 프로젝트 {project_id}, 세션 {session_id} 평가 완료")
