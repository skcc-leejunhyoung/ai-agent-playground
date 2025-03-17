# database.py

import sqlite3
import csv
import io
from pathlib import Path

DB_PATH = Path(__file__).parent / "playground.db"


def get_connection():
    """SQLite DB 연결"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """테이블 초기화"""
    with get_connection() as conn:
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS project (
                project_id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name TEXT NOT NULL UNIQUE,
                current_session_id INTEGER DEFAULT 0,
                description TEXT DEFAULT ''
            );
        """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS system_prompt (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                project_id INTEGER,
                FOREIGN KEY (project_id) REFERENCES project (project_id)
            );
        """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_prompt (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                project_id INTEGER,
                eval_method TEXT DEFAULT 'pass',
                eval_keyword TEXT DEFAULT '',
                FOREIGN KEY (project_id) REFERENCES project (project_id)
            );
        """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS model (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                project_id INTEGER,
                FOREIGN KEY (project_id) REFERENCES project (project_id)
            );
        """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS result (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                system_prompt TEXT NOT NULL,
                user_prompt TEXT NOT NULL,
                model TEXT NOT NULL,
                result TEXT NOT NULL,
                session_id INTEGER,
                project_id INTEGER,
                eval_pass TEXT DEFAULT 'X',
                eval_method TEXT DEFAULT 'pass',
                eval_keyword TEXT DEFAULT '',
                FOREIGN KEY (project_id) REFERENCES project (project_id)
            );
        """
        )

        conn.commit()
        print("[DB] initialized")


##########


def create_project(project_name, description=""):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO project (project_name, description) VALUES (?, ?);
        """,
            (project_name, description),
        )
        conn.commit()
        return cur.lastrowid


def get_projects():
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM project;")
        return cur.fetchall()


def update_project_session_id(project_id, new_session_id):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE project SET current_session_id = ? WHERE project_id = ?;",
            (new_session_id, project_id),
        )
        conn.commit()


def export_project_csv(project_id):
    """
    Export project data as CSV. Excludes all id columns and project_id.
    Column names are prefixed with table names for clarity.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header 정의
    writer.writerow(
        [
            "table",  # 테이블 이름
            "project_project_name",  # project
            "project_current_session_id",  # project
            "project_description",  # project
            "system_prompt_prompt",  # system_prompt
            "user_prompt_prompt",  # user_prompt
            "user_prompt_eval_method",  # user_prompt
            "user_prompt_eval_keyword",  # user_prompt
            "model_model_name",  # model
            "result_system_prompt",  # result
            "result_user_prompt",  # result
            "result_model",  # result
            "result_result",  # result
            "result_session_id",  # result
            "result_eval_pass",  # result
            "result_eval_method",  # result
            "result_eval_keyword",  # result
        ]
    )

    with get_connection() as conn:
        cur = conn.cursor()

        # PROJECT
        cur.execute(
            """
            SELECT project_name, current_session_id, description
            FROM project
            WHERE project_id = ?
        """,
            (project_id,),
        )
        proj_row = cur.fetchone()
        if proj_row:
            writer.writerow(
                [
                    "project",
                    proj_row["project_name"],
                    proj_row["current_session_id"],
                    proj_row["description"],
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

        # SYSTEM_PROMPT
        cur.execute(
            """
            SELECT prompt
            FROM system_prompt
            WHERE project_id = ?
        """,
            (project_id,),
        )
        sys_prompts = cur.fetchall()
        for sp in sys_prompts:
            writer.writerow(
                [
                    "system_prompt",
                    "",
                    "",
                    "",
                    sp["prompt"],
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

        # USER_PROMPT
        cur.execute(
            """
            SELECT prompt, eval_method, eval_keyword
            FROM user_prompt
            WHERE project_id = ?
        """,
            (project_id,),
        )
        user_prompts = cur.fetchall()
        for up in user_prompts:
            writer.writerow(
                [
                    "user_prompt",
                    "",
                    "",
                    "",
                    "",  # system_prompt_prompt 없음
                    up["prompt"],  # user_prompt_prompt
                    up["eval_method"],  # user_prompt_eval_method
                    up["eval_keyword"],  # user_prompt_eval_keyword
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

        # MODEL
        cur.execute(
            """
            SELECT model_name
            FROM model
            WHERE project_id = ?
        """,
            (project_id,),
        )
        models = cur.fetchall()
        for m in models:
            writer.writerow(
                [
                    "model",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    m["model_name"],  # model_model_name
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

        # RESULT
        cur.execute(
            """
            SELECT system_prompt, user_prompt, model, result,
                session_id, eval_pass, eval_method, eval_keyword
            FROM result
            WHERE project_id = ?
        """,
            (project_id,),
        )
        results = cur.fetchall()
        for res in results:
            writer.writerow(
                [
                    "result",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",  # 앞부분 빈칸
                    res["system_prompt"],  # result_system_prompt
                    res["user_prompt"],  # result_user_prompt
                    res["model"],  # result_model
                    res["result"],  # result_result
                    res["session_id"],  # result_session_id
                    res["eval_pass"],  # result_eval_pass
                    res["eval_method"],  # result_eval_method
                    res["eval_keyword"],  # result_eval_keyword
                ]
            )

    return output.getvalue()


def import_project_csv(csv_file, project_name, project_description):
    """
    Imports project data from CSV.
    Args:
        csv_file: StringIO
        project_name: Name of the new project
        project_description: Description for the new project
    Returns:
        New project_id
    """
    reader = csv.DictReader(csv_file)

    project_id = create_project(project_name, project_description)
    print(f"[DB] 프로젝트 '{project_name}' 생성 완료 (ID: {project_id})")

    current_session_id = 0

    for row in reader:
        table = row["table"]

        if table == "project":
            try:
                current_session_id = int(row["project_current_session_id"])
                print(f"[IMPORT] project_current_session_id 읽음: {current_session_id}")
            except (ValueError, TypeError):
                current_session_id = 0
                print(
                    f"[IMPORT] project_current_session_id 기본값 사용: {current_session_id}"
                )

        elif table == "system_prompt":
            prompt = row["system_prompt_prompt"]
            add_system_prompt(prompt, project_id)

        elif table == "user_prompt":
            prompt = row["user_prompt_prompt"]
            eval_method = row["user_prompt_eval_method"]
            eval_keyword = row["user_prompt_eval_keyword"]
            add_user_prompt(prompt, project_id, eval_method, eval_keyword)

        elif table == "model":
            model_name = row["model_model_name"]
            add_model(model_name, project_id)

        elif table == "result":
            system_prompt = row["result_system_prompt"]
            user_prompt = row["result_user_prompt"]
            model = row["result_model"]
            result_text = row["result_result"]

            try:
                session_id = int(row["result_session_id"])
            except (ValueError, TypeError):
                session_id = 0
                print(f"[IMPORT] result_session_id 기본값 사용: {session_id}")

            eval_pass = row["result_eval_pass"]
            eval_method = row["result_eval_method"]
            eval_keyword = row["result_eval_keyword"]

            add_result(
                system_prompt,
                user_prompt,
                model,
                result_text,
                session_id,
                project_id,
                eval_pass,
                eval_method,
                eval_keyword,
            )

    update_project_session_id(project_id, current_session_id)
    print(
        f"[DB] 프로젝트 {project_id}의 current_session_id = {current_session_id} 업데이트 완료"
    )

    return project_id


def delete_project(project_id):
    with get_connection() as conn:
        cur = conn.cursor()

        cur.execute("DELETE FROM system_prompt WHERE project_id = ?", (project_id,))
        cur.execute("DELETE FROM user_prompt WHERE project_id = ?", (project_id,))
        cur.execute("DELETE FROM model WHERE project_id = ?", (project_id,))
        cur.execute("DELETE FROM result WHERE project_id = ?", (project_id,))

        cur.execute("DELETE FROM project WHERE project_id = ?", (project_id,))

        conn.commit()

    print(f"[DB] 프로젝트 {project_id} 및 관련 데이터를 삭제했습니다.")


##########


def add_system_prompt(prompt, project_id):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO system_prompt (prompt, project_id)
            VALUES (?, ?);
        """,
            (prompt, project_id),
        )
        conn.commit()
        return cur.lastrowid


def get_system_prompts(project_id):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM system_prompt
            WHERE project_id = ?;
        """,
            (project_id,),
        )
        return cur.fetchall()


def update_system_prompt(prompt_id, new_prompt):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE system_prompt SET prompt = ? WHERE id = ?;",
            (new_prompt, prompt_id),
        )
        conn.commit()


##########


def add_user_prompt(prompt, project_id, eval_method="pass", eval_keyword=""):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO user_prompt (prompt, project_id, eval_method, eval_keyword)
            VALUES (?, ?, ?, ?);
        """,
            (prompt, project_id, eval_method, eval_keyword),
        )
        conn.commit()
        return cur.lastrowid


def get_user_prompts(project_id):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, prompt, eval_method, eval_keyword
            FROM user_prompt
            WHERE project_id = ?;
        """,
            (project_id,),
        )
        return cur.fetchall()


def update_user_prompt(prompt_id, new_prompt=None, eval_method=None, eval_keyword=None):
    with get_connection() as conn:
        cur = conn.cursor()

        fields = []
        values = []

        if new_prompt is not None:
            fields.append("prompt = ?")
            values.append(new_prompt)

        if eval_method is not None:
            fields.append("eval_method = ?")
            values.append(eval_method)

        if eval_keyword is not None:
            fields.append("eval_keyword = ?")
            values.append(eval_keyword)

        if not fields:
            print(f"[DB] 업데이트할 필드가 없습니다. user_prompt_id={prompt_id}")
            return

        values.append(prompt_id)

        sql = f"""
            UPDATE user_prompt
            SET {', '.join(fields)}
            WHERE id = ?
        """
        cur.execute(sql, values)
        conn.commit()

    print(f"[DB] user_prompt_id={prompt_id} 업데이트 완료")


##########


def add_model(model_name, project_id):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO model (model_name, project_id)
            VALUES (?, ?);
        """,
            (model_name, project_id),
        )
        conn.commit()
        return cur.lastrowid


def get_models(project_id):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM model
            WHERE project_id = ?;
        """,
            (project_id,),
        )
        return cur.fetchall()


##########


def add_result(
    system_prompt: str,
    user_prompt: str,
    model: str,
    result: str,
    session_id: int,
    project_id: int,
    eval_pass: str,
    eval_method: str,
    eval_keyword: str,
):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO result
            (system_prompt, user_prompt, model, result, session_id, project_id, eval_pass, eval_method, eval_keyword)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                system_prompt,
                user_prompt,
                model,
                result,
                session_id,
                project_id,
                eval_pass,
                eval_method,
                eval_keyword,
            ),
        )
        conn.commit()
        return cur.lastrowid


def get_results_by_project(project_id: int):
    """
    프로젝트 ID에 해당하는 모든 result 가져오기 (최신 순 정렬)
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT *
            FROM result
            WHERE project_id = ?
            ORDER BY session_id DESC, id DESC
            """,
            (project_id,),
        )
        rows = cur.fetchall()

    return [dict(row) for row in rows]


##########


def get_eval_data(project_id, session_id):
    """
    프로젝트와 세션에 해당하는 result의 평가 관련 데이터 가져오기.

    반환값:
        - 리스트 형태로 반환
        - 각 원소는 딕셔너리: {
            "id": result_id,
            "eval_pass": "O" 또는 "X",
            "eval_method": 평가 방법,
            "eval_keyword": 평가 키워드 리스트 (쉼표 구분 문자열)
        }
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, eval_pass, eval_method, eval_keyword
            FROM result
            WHERE project_id = ? AND session_id = ?
        """,
            (project_id, session_id),
        )

        rows = cur.fetchall()

        results = []
        for row in rows:
            results.append(
                {
                    "id": row["id"],
                    "eval_pass": row["eval_pass"],
                    "eval_method": row["eval_method"],
                    "eval_keyword": row["eval_keyword"],
                }
            )

    print(
        f"[DB] 프로젝트 {project_id}, 세션 {session_id} 평가 데이터 {len(results)}건 조회 완료"
    )
    return results


def update_eval_data(result_id, eval_pass=None, eval_method=None, eval_keyword=None):
    """
    개별 result_id에 대한 평가 데이터 업데이트.

    매개변수:
        - result_id (int): 업데이트할 대상 result ID
        - eval_pass (str | None): "O" 또는 "X" (None이면 변경 안함)
        - eval_method (str | None): 평가 방식 (None이면 변경 안함)
        - eval_keyword (str | None): 평가에 사용된 키워드 (None이면 변경 안함)
    """
    with get_connection() as conn:
        cur = conn.cursor()

        fields = []
        values = []

        if eval_pass is not None:
            fields.append("eval_pass = ?")
            values.append(eval_pass)

        if eval_method is not None:
            fields.append("eval_method = ?")
            values.append(eval_method)

        if eval_keyword is not None:
            fields.append("eval_keyword = ?")
            values.append(eval_keyword)

        if not fields:
            print(f"[DB] 업데이트할 필드가 없습니다. result_id={result_id}")
            return

        values.append(result_id)

        sql = f"""
            UPDATE result
            SET {', '.join(fields)}
            WHERE id = ?
        """
        cur.execute(sql, values)
        conn.commit()

    print(f"[DB] result_id={result_id} 평가 데이터 업데이트 완료")
