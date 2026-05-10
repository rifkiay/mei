"""
tools/timer_tool.py — Timer Tool untuk MEI
==========================================
Mendukung start, pause, reset, dan list timer aktif.
Terintegrasi dengan NotifProactive via callback on_timer_done.

Fix v2:
  - Tambah _parse_params() di semua call() agar params bisa
    berupa str (JSON) maupun dict — konsisten dengan calendar_tool.py.

Fix v3:
  - Persistensi timer ke file JSON di storage/event/timers.json
    agar list timer yang dibuat bisa dicek validitasnya.
  - Logika threading/callback TIDAK berubah.
"""
import json
import os
import time
import threading
from typing import Callable, Optional, Union
from qwen_agent.tools.base import BaseTool, register_tool


# ── Storage path ──────────────────────────────────────────────────
_STORAGE_DIR  = r"E:\skripsi pendekatan openclaw\agent_qwen\storage\event"
_STORAGE_FILE = os.path.join(_STORAGE_DIR, "timers.json")


# ── Active timers store ───────────────────────────────────────────
_timers: dict[str, dict] = {}
_lock = threading.Lock()


# ── Shared parser (sama persis dengan calendar_tool._parse_params) ─
def _parse_params(params: Union[str, dict]) -> dict:
    if isinstance(params, str):
        try:
            return json.loads(params)
        except json.JSONDecodeError:
            return {}
    return params or {}


# ── Persistence helpers ───────────────────────────────────────────

def _ensure_storage_dir() -> None:
    """Buat folder storage/event jika belum ada."""
    os.makedirs(_STORAGE_DIR, exist_ok=True)


def _load_persisted_timers() -> dict:
    """
    Baca timers.json dari disk.
    Return dict kosong jika file belum ada atau rusak.
    """
    _ensure_storage_dir()
    if not os.path.exists(_STORAGE_FILE):
        return {}
    try:
        with open(_STORAGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_persisted_timers(data: dict) -> None:
    """
    Tulis dict timer ke timers.json (pretty-print agar mudah dibaca).
    Dipanggil setiap ada perubahan (tambah / hapus timer).
    """
    _ensure_storage_dir()
    try:
        with open(_STORAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        pass  # jangan crash hanya karena gagal tulis file


def _persist_add_timer(timer_id: str, meta: dict) -> None:
    """
    Simpan satu timer baru ke file.
    Field yang disimpan: id, label, total_seconds, started_at, status.
    """
    data = _load_persisted_timers()
    data[timer_id] = {
        "id"          : meta["id"],
        "label"       : meta["label"],
        "total_seconds": meta["total"],
        "started_at"  : meta["started_at"],
        "status"      : "running",
    }
    _save_persisted_timers(data)


def _persist_remove_timer(timer_id: str, reason: str = "done") -> None:
    """
    Tandai timer sebagai selesai/dibatalkan di file,
    lalu hapus entry-nya agar file tetap bersih.
    """
    data = _load_persisted_timers()
    if timer_id in data:
        data[timer_id]["status"] = reason   # "done" | "cancelled"
        # Hapus dari storage — hanya simpan yang masih aktif
        del data[timer_id]
    _save_persisted_timers(data)


def _persist_clear_all() -> None:
    """Kosongkan seluruh isi timers.json."""
    _save_persisted_timers({})


# ── Timer thread ──────────────────────────────────────────────────

def _run_timer(
    timer_id: str,
    duration_sec: int,
    label: str,
    on_done: Callable[[str], None],
):
    """Background thread: hitung mundur lalu panggil on_done callback."""
    end_time = time.time() + duration_sec
    while True:
        remaining = end_time - time.time()
        with _lock:
            if timer_id not in _timers:
                return  # timer di-cancel
            _timers[timer_id]["remaining"] = max(0, remaining)
            if _timers[timer_id].get("paused"):
                time.sleep(0.5)
                end_time = time.time() + _timers[timer_id]["remaining"]
                continue
        if remaining <= 0:
            with _lock:
                _timers.pop(timer_id, None)

            # ← Hapus dari persistent storage
            _persist_remove_timer(timer_id, reason="done")

            h = int(duration_sec // 3600)
            m = int((duration_sec % 3600) // 60)
            s = int(duration_sec % 60)
            dur_str = ""
            if h:             dur_str += f"{h} jam "
            if m:             dur_str += f"{m} menit "
            if s or not dur_str: dur_str += f"{s} detik"
            msg = f"Timer '{label}' ({dur_str.strip()}) sudah selesai!"
            on_done(msg)
            return
        time.sleep(0.25)


# ── Tools ─────────────────────────────────────────────────────────

@register_tool("set_timer")
class SetTimerTool(BaseTool):
    """Mulai timer countdown. Notifikasi otomatis saat selesai via proactive callback."""

    description = (
        "Buat countdown timer dengan durasi tertentu. "
        "Notifikasi otomatis dikirim saat timer habis. "
        "Gunakan saat user minta 'ingatkan saya dalam X menit/jam/detik'."
    )
    parameters = [
        {
            "name": "duration_minutes",
            "type": "number",
            "description": "Durasi timer dalam menit (boleh desimal, contoh: 1.5 = 90 detik).",
            "required": False,
        },
        {
            "name": "duration_seconds",
            "type": "integer",
            "description": "Durasi timer dalam detik. Diabaikan jika duration_minutes diset.",
            "required": False,
        },
        {
            "name": "label",
            "type": "string",
            "description": "Nama/label timer supaya mudah dikenali.",
            "required": False,
        },
    ]

    def __init__(self, on_timer_done: Callable[[str], None], **kwargs):
        super().__init__(**kwargs)
        self._on_done = on_timer_done

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        import uuid

        params = _parse_params(params)

        label = params.get("label") or "Timer"

        if params.get("duration_minutes") is not None:
            total_sec = int(float(params["duration_minutes"]) * 60)
        elif params.get("duration_seconds") is not None:
            total_sec = int(params["duration_seconds"])
        else:
            return {"status": "error", "message": "Durasi tidak diberikan."}

        if total_sec <= 0:
            return {"status": "error", "message": "Durasi harus lebih dari 0."}

        timer_id = str(uuid.uuid4())[:8]
        meta = {
            "id"        : timer_id,
            "label"     : label,
            "total"     : total_sec,
            "remaining" : total_sec,
            "paused"    : False,
            "started_at": time.time(),
        }

        with _lock:
            _timers[timer_id] = meta

        # ← Simpan ke persistent storage segera setelah dibuat
        _persist_add_timer(timer_id, meta)

        t = threading.Thread(
            target=_run_timer,
            args=(timer_id, total_sec, label, self._on_done),
            daemon=True,
        )
        t.start()

        h = total_sec // 3600
        m = (total_sec % 3600) // 60
        s = total_sec % 60
        parts = []
        if h: parts.append(f"{h} jam")
        if m: parts.append(f"{m} menit")
        if s or not parts: parts.append(f"{s} detik")
        dur_str = " ".join(parts)

        return {
            "status"  : "success",
            "timer_id": timer_id,
            "message" : f"Timer '{label}' dimulai selama {dur_str}.",
        }


@register_tool("list_timers")
class ListTimersTool(BaseTool):
    """Tampilkan semua timer yang sedang aktif."""

    description = "Tampilkan daftar timer yang sedang berjalan atau dijeda."
    parameters  = []

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        with _lock:
            active = list(_timers.values())

        if not active:
            return {"status": "success", "message": "Tidak ada timer aktif.", "timers": []}

        result = []
        for t in active:
            rem  = int(t["remaining"])
            m, s = divmod(rem, 60)
            h, m = divmod(m, 60)
            parts = []
            if h: parts.append(f"{h}j")
            parts.append(f"{m:02d}m {s:02d}d")
            status = "dijeda" if t["paused"] else "berjalan"
            result.append({
                "id"       : t["id"],
                "label"    : t["label"],
                "remaining": " ".join(parts),
                "status"   : status,
            })

        lines = "\n".join(
            f"- [{r['id']}] {r['label']} — sisa {r['remaining']} ({r['status']})"
            for r in result
        )
        return {"status": "success", "message": lines, "timers": result}


@register_tool("cancel_timer")
class CancelTimerTool(BaseTool):
    """Batalkan timer yang sedang berjalan."""

    description = "Batalkan timer berdasarkan ID atau label."
    parameters  = [
        {
            "name": "timer_id",
            "type": "string",
            "description": "ID timer (dari list_timers). Atau gunakan 'all' untuk batalkan semua.",
            "required": True,
        }
    ]

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        params = _parse_params(params)

        tid = params.get("timer_id", "").strip()
        if not tid:
            return {"status": "error", "message": "timer_id diperlukan."}

        with _lock:
            if tid == "all":
                n = len(_timers)
                ids_to_remove = list(_timers.keys())
                _timers.clear()

                # ← Hapus semua dari persistent storage
                _persist_clear_all()

                return {"status": "success", "message": f"{n} timer dibatalkan."}

            if tid in _timers:
                label = _timers.pop(tid)["label"]

                # ← Hapus dari persistent storage
                _persist_remove_timer(tid, reason="cancelled")

                return {"status": "success", "message": f"Timer '{label}' ({tid}) dibatalkan."}

            # coba match by label (case-insensitive)
            found = [k for k, v in _timers.items() if v["label"].lower() == tid.lower()]
            if found:
                matched_id = found[0]
                label = _timers.pop(matched_id)["label"]

                # ← Hapus dari persistent storage
                _persist_remove_timer(matched_id, reason="cancelled")

                return {"status": "success", "message": f"Timer '{label}' dibatalkan."}

        return {"status": "error", "message": f"Timer '{tid}' tidak ditemukan."}