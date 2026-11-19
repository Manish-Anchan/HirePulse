import sqlite3
from typing import Dict, List, Optional
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages

from app.config import DB_PATH


# 1. State Schema for Short-Term / Working Memory
class convo(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    summarized_index: int


# 2. Checkpointer setup (SQLite persistence scoped to thread_id)
conn = sqlite3.connect(database=str(DB_PATH), check_same_thread=False)
with conn:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS hidden_threads (
            thread_id TEXT PRIMARY KEY,
            hidden_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
checkpointer = SqliteSaver(conn=conn)


# 3. Token estimation helper
def count_tokens(messages: list[BaseMessage]) -> int:
    """Rough token estimator: ~4 characters per token + 4 per message."""
    total = 0
    for m in messages:
        c = m.content if isinstance(m.content, str) else str(m.content)
        total += len(c) // 4 + 4
    return total


# 4. Progressive thread summarizer
def update_summary(existing_summary: str, new_msgs: list[BaseMessage], model) -> str:
    """Incrementally update the rolling summary of the current thread."""
    convo_lines = []
    for m in new_msgs:
        if isinstance(m, HumanMessage):
            convo_lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage) and m.content:
            convo_lines.append(f"Assistant: {m.content}")

    convo_text = "\n".join(convo_lines).strip()
    if not convo_text:
        return existing_summary

    prompt = (
        "Summarize the key facts, user preferences, and context from this conversation concisely (under 150 words):\n"
    )
    if existing_summary:
        prompt = (
            f"Current conversation summary:\n{existing_summary}\n\n"
            f"Update the summary with these new messages, keeping it concise (under 150 words):\n{convo_text}"
        )
    else:
        prompt += convo_text

    try:
        res = model.invoke([SystemMessage(content=prompt)])
        return res.content.strip() if isinstance(res.content, str) else str(res.content)
    except Exception:
        return existing_summary


# 5. Sliding window token trimmer
def safe_trim(msgs: list[BaseMessage], max_tokens: int = 2500) -> list[BaseMessage]:
    """Trim message list to stay strictly within token budget while preserving recent context."""
    try:
        trimmed = trim_messages(
            msgs,
            max_tokens=max_tokens,
            strategy="last",
            token_counter=count_tokens,
            start_on="human",
            include_system=False,
            allow_partial=False,
        )
        if trimmed:
            return trimmed
    except Exception:
        pass
    # Fallback: keep last 8 messages if trimming fails
    return msgs[-8:] if len(msgs) >= 8 else msgs


# 6. Thread retrieval & soft-delete helpers
def hide_thread(thread_id: str) -> None:
    """Soft-delete: hide a thread from the frontend while preserving history in the database."""
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO hidden_threads (thread_id) VALUES (?)",
            (str(thread_id),),
        )


def get_hidden_threads() -> set[str]:
    """Retrieve all thread IDs that have been hidden/removed from the frontend."""
    with conn:
        cur = conn.cursor()
        cur.execute("SELECT thread_id FROM hidden_threads")
        return {row[0] for row in cur.fetchall()}


def get_threads(include_hidden: bool = False) -> list[str]:
    """List unique thread IDs recorded in the SQLite checkpointer, excluding hidden threads by default."""
    hidden = set() if include_hidden else get_hidden_threads()
    all_threads = []
    seen = set()
    for checkpoint in checkpointer.list(config=None):
        tid = checkpoint.config["configurable"]["thread_id"]
        if tid not in seen and tid not in hidden:
            seen.add(tid)
            all_threads.append(tid)
    return all_threads


def get_thread_messages(chatbot, thread_id: str) -> list[dict]:
    """Retrieve and format all messages for a given thread."""
    state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
    raw_messages = state.values.get("messages", []) if state and state.values else []
    formatted = []
    for msg in raw_messages:
        if isinstance(msg, HumanMessage):
            formatted.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            formatted.append({"role": "assistant", "content": msg.content})
    return formatted
