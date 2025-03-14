# database.py

import sqlite3
import json
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
                eval_method TEXT DEFAULT '',
                eval_keyword TEXT DEFAULT '',
                FOREIGN KEY (project_id) REFERENCES project (project_id)
            );
        """
        )

        conn.commit()
        print("[DB initialized]")


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


def add_result(system_prompt, user_prompt, model, result_data, session_id, project_id):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO result (system_prompt, user_prompt, model, result, session_id, project_id)
            VALUES (?, ?, ?, ?, ?, ?);
        """,
            (
                system_prompt,
                user_prompt,
                model,
                json.dumps(result_data, ensure_ascii=False),
                session_id,
                project_id,
            ),
        )
        conn.commit()
        return cur.lastrowid


def get_results(project_id):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM result
            WHERE project_id = ?;
        """,
            (project_id,),
        )
        rows = cur.fetchall()

        results = []
        for row in rows:
            item = dict(row)
            item["result"] = json.loads(item["result"])
            results.append(item)

        return results


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
