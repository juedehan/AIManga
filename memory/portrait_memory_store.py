import json
import sqlite3
from contextlib import closing
from datetime import datetime

from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class PortraitMemoryStore:
    def __init__(self, store_path: str, max_adjustment_history: int = 10, enabled: bool = True):
        self.enabled = enabled
        self.store_path = get_abs_path(store_path)
        self.max_adjustment_history = max_adjustment_history

        if not self.enabled:
            return

        parent_dir = self.store_path.rsplit("/", 1)[0]
        if parent_dir:
            import os

            os.makedirs(parent_dir, exist_ok=True)

        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.store_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with closing(self._connect()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS portrait_memory (
                    session_id TEXT NOT NULL,
                    character_name TEXT NOT NULL,
                    character_name_pinyin TEXT,
                    latest_final_prompt TEXT NOT NULL,
                    latest_image_path TEXT NOT NULL,
                    latest_scene TEXT,
                    latest_gender TEXT,
                    last_user_request TEXT,
                    adjustment_history TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, character_name)
                );

                CREATE TABLE IF NOT EXISTS session_latest_character (
                    session_id TEXT PRIMARY KEY,
                    character_name TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.commit()

    def _normalize_history(self, adjustment_history: list[str] | None) -> list[str]:
        history = [item.strip() for item in adjustment_history or [] if item and item.strip()]
        if self.max_adjustment_history > 0:
            history = history[-self.max_adjustment_history :]
        return history

    def _touch_latest_character(self, conn: sqlite3.Connection, session_id: str, character_name: str, updated_at: str) -> None:
        conn.execute(
            """
            INSERT INTO session_latest_character(session_id, character_name, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                character_name = excluded.character_name,
                updated_at = excluded.updated_at
            """,
            (session_id, character_name, updated_at),
        )

    def get_character_memory(self, session_id: str, character_name: str) -> dict | None:
        if not self.enabled:
            return None

        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT *
                FROM portrait_memory
                WHERE session_id = ? AND character_name = ?
                """,
                (session_id, character_name),
            ).fetchone()

        if row is None:
            return None

        record = dict(row)
        record["adjustment_history"] = json.loads(record.get("adjustment_history") or "[]")
        return record

    def resolve_latest_character(self, session_id: str) -> str | None:
        if not self.enabled:
            return None

        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT character_name
                FROM session_latest_character
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

        if row is None:
            return None
        return row["character_name"]

    def save_character_memory(
        self,
        session_id: str,
        character_name: str,
        character_name_pinyin: str,
        latest_final_prompt: str,
        latest_image_path: str,
        latest_scene: str | None = None,
        latest_gender: str | None = None,
        last_user_request: str | None = None,
    ) -> dict | None:
        if not self.enabled:
            return None

        existing = self.get_character_memory(session_id, character_name)
        adjustment_history = self._normalize_history(existing.get("adjustment_history", []) if existing else [])
        updated_at = datetime.utcnow().isoformat()

        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO portrait_memory(
                    session_id,
                    character_name,
                    character_name_pinyin,
                    latest_final_prompt,
                    latest_image_path,
                    latest_scene,
                    latest_gender,
                    last_user_request,
                    adjustment_history,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, character_name) DO UPDATE SET
                    character_name_pinyin = excluded.character_name_pinyin,
                    latest_final_prompt = excluded.latest_final_prompt,
                    latest_image_path = excluded.latest_image_path,
                    latest_scene = excluded.latest_scene,
                    latest_gender = excluded.latest_gender,
                    last_user_request = excluded.last_user_request,
                    adjustment_history = excluded.adjustment_history,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    character_name,
                    character_name_pinyin,
                    latest_final_prompt,
                    latest_image_path,
                    latest_scene,
                    latest_gender,
                    last_user_request,
                    json.dumps(adjustment_history, ensure_ascii=False),
                    updated_at,
                ),
            )
            self._touch_latest_character(conn, session_id, character_name, updated_at)
            conn.commit()

        logger.info(f"[portrait_memory] 保存角色记忆: session_id={session_id}, character={character_name}")
        return self.get_character_memory(session_id, character_name)

    def update_character_memory(
        self,
        session_id: str,
        character_name: str,
        latest_final_prompt: str,
        latest_image_path: str,
        modification_request: str,
        character_name_pinyin: str | None = None,
        latest_scene: str | None = None,
        latest_gender: str | None = None,
        last_user_request: str | None = None,
    ) -> dict | None:
        if not self.enabled:
            return None

        existing = self.get_character_memory(session_id, character_name)
        if existing is None:
            return None

        adjustment_history = self._normalize_history(
            [*existing.get("adjustment_history", []), modification_request]
        )
        updated_at = datetime.utcnow().isoformat()

        with closing(self._connect()) as conn:
            conn.execute(
                """
                UPDATE portrait_memory
                SET character_name_pinyin = ?,
                    latest_final_prompt = ?,
                    latest_image_path = ?,
                    latest_scene = ?,
                    latest_gender = ?,
                    last_user_request = ?,
                    adjustment_history = ?,
                    updated_at = ?
                WHERE session_id = ? AND character_name = ?
                """,
                (
                    character_name_pinyin or existing.get("character_name_pinyin"),
                    latest_final_prompt,
                    latest_image_path,
                    latest_scene if latest_scene is not None else existing.get("latest_scene"),
                    latest_gender if latest_gender is not None else existing.get("latest_gender"),
                    last_user_request if last_user_request is not None else existing.get("last_user_request"),
                    json.dumps(adjustment_history, ensure_ascii=False),
                    updated_at,
                    session_id,
                    character_name,
                ),
            )
            self._touch_latest_character(conn, session_id, character_name, updated_at)
            conn.commit()

        logger.info(f"[portrait_memory] 更新角色记忆: session_id={session_id}, character={character_name}")
        return self.get_character_memory(session_id, character_name)
