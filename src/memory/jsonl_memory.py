"""
JSONL Short-Term Memory Manager — Single User, No Session
===========================================================
Satu file JSONL per user, append terus.
Window pruning otomatis saat get_recent_messages().
Tidak ada session ID, tidak ada index file.

File: storage/sessions/{user_id}.jsonl
Profile: storage/profiles/{user_id}.json

Format tiap baris:
{
    "turn": 42,
    "timestamp": "2026-04-06T14:30:22",
    "role": "user" | "assistant",
    "content": "..."
}
"""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConversationContext:
    """Kompatibel dengan kode lama yang pakai ctx_obj.current_topic."""
    current_topic: Optional[str] = None
    message_count: int = 0


class JSONLMemoryManager:

    def __init__(
        self,
        sessions_dir: str = "./../storage/memory/sessions",
        window_size: int = 20,
    ):
        self.sessions_dir = Path(sessions_dir)
        self.window_size  = window_size

        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    # ── Paths ──────────────────────────────────────────────────────

    def _chat_path(self, user_id: str) -> Path:
        return self.sessions_dir / f"{user_id}.jsonl"

    # ── Tulis pesan ────────────────────────────────────────────────

    def add_message(self, user_id: str, role: str, content: str):
        """Append satu turn ke file JSONL user."""
        p    = self._chat_path(user_id)
        turn = self._count_lines(p) + 1

        record = {
            "turn"     : turn,
            "timestamp": _now(),
            "role"     : role,
            "content"  : content,
        }

        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── Baca pesan ─────────────────────────────────────────────────

    def get_recent_messages(self, user_id: str, n: Optional[int] = None) -> list[dict]:
        """Return N turn terakhir (window). Default = self.window_size."""
        n = n or self.window_size
        p = self._chat_path(user_id)
        if not p.exists():
            return []

        records = []
        for line in p.read_text(encoding="utf-8").split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        return records[-n:] if len(records) > n else records

    def get_all_messages(self, user_id: str) -> list[dict]:
        return self.get_recent_messages(user_id, n=999_999)

    # ── Context object ─────────────────────────────────────────────

    def get_context(self, user_id: str) -> ConversationContext:
        messages  = self.get_recent_messages(user_id, n=3)
        last_user = next((m for m in reversed(messages) if m["role"] == "user"), None)
        topic     = last_user["content"][:80] if last_user else None
        total     = self._count_lines(self._chat_path(user_id))
        return ConversationContext(current_topic=topic, message_count=total)

    # ── Hapus riwayat ──────────────────────────────────────────────

    def clear(self, user_id: str):
        """Hapus seluruh riwayat chat (command 'clear')."""
        p = self._chat_path(user_id)
        if p.exists():
            p.unlink()
        p.touch()


    # ── Stats ──────────────────────────────────────────────────────

    def stats(self, user_id: str) -> dict:
        all_msgs = self.get_all_messages(user_id)
        recent   = self.get_recent_messages(user_id)
        return {
            "total_turns" : len(all_msgs),
            "window_turns": len(recent),
            "window_size" : self.window_size,
        }

    # ── Util ───────────────────────────────────────────────────────

    def _count_lines(self, path: Path) -> int:
        if not path.exists():
            return 0
        return sum(1 for l in path.read_text(encoding="utf-8").split("\n") if l.strip())


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")