import sqlite3
import json
from pathlib import Path

DB_PATH = Path(__file__).parent / "playground.db"


def get_connection():
    """SQLite DB 연결"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Dict처럼 결과 반환
    return conn


def init_db():
    """테이블 초기화"""
    with get_connection() as conn:
        cur = conn.cursor()

        # 프로젝트 테이블
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS project (
                project_id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name TEXT NOT NULL UNIQUE
            );
        """
        )

        # 시스템 프롬프트 테이블
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

        # 유저 프롬프트 테이블
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

        # 모델 테이블
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

        # 결과 테이블
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS result (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                system_prompt TEXT NOT NULL,
                user_prompt TEXT NOT NULL,
                model TEXT NOT NULL,
                result TEXT NOT NULL, -- JSON 형태 문자열
                project_id INTEGER,
                FOREIGN KEY (project_id) REFERENCES project (project_id)
            );
        """
        )

        conn.commit()
        print("✅ DB 초기화 완료!")


##############################
# PROJECT
##############################


def create_project(project_name):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO project (project_name) VALUES (?);
        """,
            (project_name,),
        )
        conn.commit()
        return cur.lastrowid


def get_projects():
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM project;")
        return cur.fetchall()


##############################
# SYSTEM PROMPT
##############################


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


##############################
# USER PROMPT
##############################


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


##############################
# MODEL
##############################


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


##############################
# RESULT
##############################


def add_result(system_prompt, user_prompt, model, result_data, project_id):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO result (system_prompt, user_prompt, model, result, project_id)
            VALUES (?, ?, ?, ?, ?);
        """,
            (
                system_prompt,
                user_prompt,
                model,
                json.dumps(result_data, ensure_ascii=False),
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

        # JSON 필드 파싱해서 반환
        results = []
        for row in rows:
            item = dict(row)
            item["result"] = json.loads(item["result"])
            results.append(item)

        return results
