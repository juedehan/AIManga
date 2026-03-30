import sqlite3
from contextlib import closing
from datetime import datetime

from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class ChatHistoryStore:
    def __init__(self, store_path: str, enabled: bool = True):
        self.enabled = enabled
        self.store_path = get_abs_path(store_path)

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
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL DEFAULT '',
                    image_url TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.commit()

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str = "",
        image_url: str | None = None,
    ) -> dict | None:
        if not self.enabled:
            return None

        created_at = datetime.utcnow().isoformat()
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                """
                INSERT INTO chat_history(session_id, role, content, image_url, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, role, content, image_url, created_at),
            )
            conn.commit()
            row_id = cursor.lastrowid
            row = conn.execute(
                """
                SELECT id, session_id, role, content, image_url, created_at
                FROM chat_history
                WHERE id = ?
                """,
                (row_id,),
            ).fetchone()

        record = dict(row) if row is not None else None
        logger.info(f"[chat_history] 保存消息: session_id={session_id}, role={role}")
        return record

    def list_messages(self, session_id: str) -> list[dict]:
        if not self.enabled:
            return []

        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, role, content, image_url, created_at
                FROM chat_history
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()

        return [dict(row) for row in rows]
