"""Migration utility from SQLite checkpointer to PostgreSQL checkpointer.

Reads checkpoints, writes, and hidden threads from SQLite and populates PostgreSQL.
"""

import logging
import sqlite3
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import DATABASE_URL, DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sqlite_to_postgres_migration")


def migrate(sqlite_path: Path = DB_PATH, postgres_url: str = DATABASE_URL) -> None:
    if not sqlite_path.exists():
        logger.warning("SQLite database not found at %s. Nothing to migrate.", sqlite_path)
        return

    logger.info("Starting migration from SQLite (%s) to PostgreSQL (%s)", sqlite_path, postgres_url)

    sqlite_conn = sqlite3.connect(str(sqlite_path))
    sqlite_saver = SqliteSaver(sqlite_conn)

    with psycopg.connect(postgres_url, autocommit=True) as pg_conn:
        pg_saver = PostgresSaver(pg_conn)

        # 1. Run migrations and setup tables
        logger.info("Setting up PostgresSaver schema and tables...")
        pg_saver.setup()

        with pg_conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS hidden_threads (
                    thread_id TEXT PRIMARY KEY,
                    hidden_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

        # 2. Migrate checkpoints
        logger.info("Migrating checkpoints...")
        checkpoints = list(sqlite_saver.list(None))
        migrated_cp = 0
        for cp in reversed(checkpoints):
            try:
                pg_saver.put(
                    cp.config,
                    cp.checkpoint,
                    cp.metadata,
                    cp.checkpoint.get("channel_versions", {}),
                )
                migrated_cp += 1
            except Exception as e:
                logger.error("Failed to migrate checkpoint %s: %s", cp.config, e)
        logger.info("Checkpoints migrated: %d / %d", migrated_cp, len(checkpoints))

        # 3. Migrate writes
        logger.info("Migrating intermediate writes...")
        s_cur = sqlite_conn.cursor()
        s_cur.execute(
            """
            SELECT thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, value
            FROM writes
            """
        )
        writes = s_cur.fetchall()
        migrated_writes = 0
        with pg_conn.cursor() as cur:
            for row in writes:
                cur.execute(
                    """
                    INSERT INTO checkpoint_writes (
                        thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, idx, channel, type, blob
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO NOTHING
                    """,
                    row,
                )
                migrated_writes += 1
        logger.info("Writes migrated: %d", migrated_writes)

        # 4. Migrate hidden threads
        logger.info("Migrating hidden threads...")
        s_cur.execute("SELECT thread_id, hidden_at FROM hidden_threads")
        hidden = s_cur.fetchall()
        with pg_conn.cursor() as cur:
            for row in hidden:
                cur.execute(
                    """
                    INSERT INTO hidden_threads (thread_id, hidden_at)
                    VALUES (%s, %s)
                    ON CONFLICT (thread_id) DO NOTHING
                    """,
                    row,
                )
        logger.info("Hidden threads migrated: %d", len(hidden))

    sqlite_conn.close()
    logger.info("Migration completed successfully!")


if __name__ == "__main__":
    db_file = Path(sys.argv[1]) if len(sys.argv) > 1 else DB_PATH
    pg_url = sys.argv[2] if len(sys.argv) > 2 else DATABASE_URL
    migrate(db_file, pg_url)
