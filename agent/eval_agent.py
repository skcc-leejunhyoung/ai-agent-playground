# agent/eval_agent.py

import json
from database import get_eval_data, update_eval_data, get_connection


def evaluate_response(response: str, keywords: list[str]) -> str:
    """
    주어진 응답(response)에 키워드가 하나라도 포함되면 'O', 없으면 'X' 반환
    """
    for keyword in keywords:
        if keyword in response:
            return "O"
    return "X"


def run_evaluation(
    project_id: int, session_id: int, keywords: list[str], method: str = "keyword_match"
):
    """
    특정 프로젝트와 세션의 모든 result에 대해 평가 후 eval_pass, eval_method, eval_keyword 필드 업데이트

    매개변수:
      - project_id: 프로젝트 ID
      - session_id: 세션 ID
      - keywords: 평가에 사용할 키워드 리스트
      - method: 평가 방법 (기본: "keyword_match")
    """
    # get_eval_data는 eval 관련 기본 데이터만 반환하므로, 각 result의 원본 응답은 별도로 조회
    eval_data = get_eval_data(project_id, session_id)
    print(
        f"[EVAL] 프로젝트 {project_id}, 세션 {session_id}: {len(eval_data)}개의 결과 평가 중..."
    )

    for data in eval_data:
        result_id = data["id"]
        # 개별 result의 원본 응답(result 필드)을 조회
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT result FROM result WHERE id = ?", (result_id,))
            row = cur.fetchone()
            if row is None:
                continue
            result_content = row["result"]

        try:
            parsed_result = json.loads(result_content)
        except json.JSONDecodeError:
            eval_pass = "X"
        else:
            if isinstance(parsed_result, dict):
                combined_text = " ".join(map(str, parsed_result.values()))
            elif isinstance(parsed_result, str):
                combined_text = parsed_result
            else:
                combined_text = str(parsed_result)
            eval_pass = evaluate_response(combined_text, keywords)

        # update_eval_data를 사용하여 개별 result의 평가 데이터 업데이트
        update_eval_data(
            result_id,
            eval_pass=eval_pass,
            eval_method=method,
            eval_keyword=", ".join(keywords),
        )

    print(f"[EVAL] 프로젝트 {project_id}, 세션 {session_id} 평가 완료!")
