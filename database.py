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
                result TEXT NOT NULL, -- JSON 형태 문자열
                session_id INTEGER,
                project_id INTEGER,
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


def add_user_prompt(prompt, project_id):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO user_prompt (prompt, project_id)
            VALUES (?, ?);
        """,
            (prompt, project_id),
        )
        conn.commit()
        return cur.lastrowid


def get_user_prompts(project_id):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM user_prompt
            WHERE project_id = ?;
        """,
            (project_id,),
        )
        return cur.fetchall()


def update_user_prompt(prompt_id, new_prompt):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE user_prompt SET prompt = ? WHERE id = ?;",
            (new_prompt, prompt_id),
        )
        conn.commit()


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
