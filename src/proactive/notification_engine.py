"""
proactive/notification_engine.py — Proactive Notification Engine untuk MEI
===========================================================================
Scheduler notifikasi berbasis waktu (sekali atau berulang).
Terintegrasi dengan on_proactive_trigger callback yang sudah ada di main.py.

Contoh penggunaan di main.py:
    from proactive.notification_engine import NotificationEngine

    notif_engine = NotificationEngine(on_trigger=on_proactive_trigger)
    notif_engine.start()

Contoh dari tool / agent:
    notif_engine.schedule(
        message  = "Waktunya istirahat, Rifki!",
        at_time  = "15:30",          # HH:MM format
        repeat   = "daily",          # None | "daily" | "weekdays"
        notif_id = "istirahat-sore",
    )
"""
import threading
import time
from datetime import datetime, timedelta
from typing import Callable, Optional
import pytz

TZ = pytz.timezone("Asia/Jakarta")

_REPEAT_LABELS = (None, "daily", "weekdays", "weekly")


class ScheduledNotif:
    def __init__(
        self,
        notif_id  : str,
        message   : str,
        fire_at   : datetime,           # timezone-aware
        repeat    : Optional[str],      # None | "daily" | "weekdays" | "weekly"
        tags      : list[str] | None = None,
    ):
        self.notif_id = notif_id
        self.message  = message
        self.fire_at  = fire_at
        self.repeat   = repeat
        self.tags     = tags or []
        self.fired    = False

    def next_fire(self) -> Optional[datetime]:
        """Hitung waktu tembak berikutnya untuk notif berulang."""
        if self.repeat is None:
            return None
        now  = datetime.now(TZ)
        base = self.fire_at
        if self.repeat == "daily":
            while base <= now:
                base += timedelta(days=1)
            return base
        if self.repeat == "weekdays":
            base += timedelta(days=1)
            while base.weekday() >= 5 or base <= now:
                base += timedelta(days=1)
            return base
        if self.repeat == "weekly":
            while base <= now:
                base += timedelta(weeks=1)
            return base
        return None

    def __repr__(self):
        return (
            f"<ScheduledNotif id={self.notif_id!r} "
            f"fire_at={self.fire_at.strftime('%H:%M %d/%m')} "
            f"repeat={self.repeat}>"
        )


class NotificationEngine:
    """
    Thread tunggal yang memeriksa jadwal notifikasi setiap 30 detik.
    Thread-safe: semua operasi schedule/cancel menggunakan lock.
    """

    CHECK_INTERVAL = 30  # detik

    def __init__(
        self,
        on_trigger: Callable[[str], None],
        debug     : bool = False,
    ):
        self._on_trigger  = on_trigger
        self._debug       = debug
        self._notifs      : dict[str, ScheduledNotif] = {}
        self._lock        = threading.Lock()
        self._stop_event  = threading.Event()
        self._thread      : Optional[threading.Thread] = None

    # ── Public API ────────────────────────────────────────────────

    def schedule(
        self,
        message  : str,
        at_time  : str | None          = None,   # "HH:MM" hari ini / besok
        at_dt    : datetime | None     = None,   # datetime tz-aware eksplisit
        delay_sec: int | None          = None,   # relatif dari sekarang
        repeat   : Optional[str]       = None,
        notif_id : str | None          = None,
        tags     : list[str] | None    = None,
    ) -> str:
        """
        Jadwalkan notifikasi. Kembalikan notif_id.

        Prioritas waktu: at_dt > at_time > delay_sec > 60 detik dari sekarang.
        """
        import uuid
        nid = notif_id or str(uuid.uuid4())[:8]

        now = datetime.now(TZ)

        if at_dt is not None:
            fire_at = at_dt if at_dt.tzinfo else TZ.localize(at_dt)
        elif at_time is not None:
            h, m  = (int(x) for x in at_time.split(":")[:2])
            fire_at = now.replace(hour=h, minute=m, second=0, microsecond=0)
            if fire_at <= now:
                fire_at += timedelta(days=1)
        elif delay_sec is not None:
            fire_at = now + timedelta(seconds=delay_sec)
        else:
            fire_at = now + timedelta(seconds=60)

        notif = ScheduledNotif(nid, message, fire_at, repeat, tags)
        with self._lock:
            self._notifs[nid] = notif

        if self._debug:
            print(f"  [NotifEngine] Scheduled {notif}", flush=True)

        return nid

    def cancel(self, notif_id: str) -> bool:
        with self._lock:
            removed = self._notifs.pop(notif_id, None)
        return removed is not None

    def cancel_by_tag(self, tag: str) -> int:
        with self._lock:
            to_remove = [nid for nid, n in self._notifs.items() if tag in n.tags]
            for nid in to_remove:
                del self._notifs[nid]
        return len(to_remove)

    def list_scheduled(self) -> list[dict]:
        with self._lock:
            result = []
            for n in self._notifs.values():
                result.append({
                    "id":      n.notif_id,
                    "message": n.message[:80],
                    "fire_at": n.fire_at.strftime("%Y-%m-%d %H:%M"),
                    "repeat":  n.repeat or "sekali",
                    "tags":    n.tags,
                })
        return result

    def check_now(self):
        """Paksa cek segera (bisa dipanggil dari command 'notif' di main.py)."""
        self._check()

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="NotifEngine", daemon=True
        )
        self._thread.start()
        if self._debug:
            print("  [NotifEngine] Started.", flush=True)

    def stop(self, timeout: float = 5.0):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    # ── Internal ──────────────────────────────────────────────────

    def _loop(self):
        while not self._stop_event.is_set():
            self._check()
            self._stop_event.wait(self.CHECK_INTERVAL)

    def _check(self):
        now = datetime.now(TZ)
        to_reschedule = []
        to_remove     = []

        with self._lock:
            for nid, notif in list(self._notifs.items()):
                if notif.fire_at <= now and not notif.fired:
                    to_fire = notif
                    next_t  = notif.next_fire()
                    if next_t:
                        to_reschedule.append((nid, next_t))
                    else:
                        to_remove.append(nid)

                    try:
                        self._on_trigger(notif.message)
                    except Exception as e:
                        if self._debug:
                            print(f"  [NotifEngine] callback error: {e}", flush=True)

            for nid, next_t in to_reschedule:
                if nid in self._notifs:
                    self._notifs[nid].fire_at = next_t

            for nid in to_remove:
                self._notifs.pop(nid, None)


# ── Convenience tool wrapper ──────────────────────────────────────

try:
    from qwen_agent.tools.base import BaseTool, register_tool

    @register_tool("schedule_notification")
    class ScheduleNotificationTool(BaseTool):
        """Jadwalkan notifikasi proaktif pada waktu tertentu atau setelah delay."""

        description = (
            "Jadwalkan notifikasi / pengingat untuk Rifki pada waktu tertentu. "
            "Gunakan saat user minta diingatkan pada jam tertentu, "
            "atau setelah X menit/jam."
        )
        parameters = [
            {
                "name": "message",
                "type": "string",
                "description": "Pesan notifikasi yang akan ditampilkan.",
                "required": True,
            },
            {
                "name": "at_time",
                "type": "string",
                "description": "Waktu tembak dalam format HH:MM (24h), contoh '14:30'. "
                               "Jika sudah lewat hari ini, dijadwalkan besok.",
                "required": False,
            },
            {
                "name": "delay_minutes",
                "type": "number",
                "description": "Delay dalam menit dari sekarang. Diabaikan jika at_time diset.",
                "required": False,
            },
            {
                "name": "repeat",
                "type": "string",
                "description": "Pengulangan: 'daily', 'weekdays', 'weekly', atau kosong untuk sekali.",
                "required": False,
            },
        ]

        def __init__(self, engine: "NotificationEngine", **kwargs):
            super().__init__(**kwargs)
            self._engine = engine

        def call(self, params: dict, **kwargs) -> dict:
            message = params.get("message", "").strip()
            if not message:
                return {"status": "error", "message": "Pesan notifikasi tidak boleh kosong."}

            at_time    = params.get("at_time")
            delay_min  = params.get("delay_minutes")
            repeat     = params.get("repeat") or None
            delay_sec  = int(float(delay_min) * 60) if delay_min else None

            nid = self._engine.schedule(
                message   = message,
                at_time   = at_time,
                delay_sec = delay_sec,
                repeat    = repeat,
            )

            if at_time:
                when = f"pukul {at_time}"
            elif delay_sec:
                m, s = divmod(delay_sec, 60)
                when = f"dalam {m} menit" + (f" {s} detik" if s else "")
            else:
                when = "1 menit lagi"

            rep_str = f" (berulang: {repeat})" if repeat else ""
            return {
                "status":   "success",
                "notif_id": nid,
                "message":  f"Notifikasi dijadwalkan {when}{rep_str}.",
            }

    @register_tool("list_notifications")
    class ListNotificationsTool(BaseTool):
        """Tampilkan semua notifikasi yang terjadwal."""

        description = "Tampilkan daftar notifikasi/pengingat yang masih aktif."
        parameters  = []

        def __init__(self, engine: "NotificationEngine", **kwargs):
            super().__init__(**kwargs)
            self._engine = engine

        def call(self, params: dict, **kwargs) -> dict:
            items = self._engine.list_scheduled()
            if not items:
                return {"status": "success", "message": "Tidak ada notifikasi terjadwal."}
            lines = "\n".join(
                f"- [{i['id']}] {i['fire_at']} | {i['repeat']} | {i['message']}"
                for i in items
            )
            return {"status": "success", "message": lines, "items": items}

    @register_tool("cancel_notification")
    class CancelNotificationTool(BaseTool):
        """Batalkan notifikasi berdasarkan ID."""

        description = "Batalkan notifikasi/pengingat berdasarkan ID."
        parameters  = [
            {
                "name": "notif_id",
                "type": "string",
                "description": "ID notifikasi (dari list_notifications). Gunakan 'all' untuk semua.",
                "required": True,
            }
        ]

        def __init__(self, engine: "NotificationEngine", **kwargs):
            super().__init__(**kwargs)
            self._engine = engine

        def call(self, params: dict, **kwargs) -> dict:
            nid = params.get("notif_id", "").strip()
            if nid == "all":
                items = self._engine.list_scheduled()
                for i in items:
                    self._engine.cancel(i["id"])
                return {"status": "success", "message": f"{len(items)} notifikasi dibatalkan."}
            ok = self._engine.cancel(nid)
            if ok:
                return {"status": "success", "message": f"Notifikasi '{nid}' dibatalkan."}
            return {"status": "error", "message": f"ID '{nid}' tidak ditemukan."}

except ImportError:
    pass  # qwen_agent tidak tersedia, tool wrapper dilewati