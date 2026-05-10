"""
proactive/notif_proactive.py — Unified Proactive Notification Engine
=====================================================================
Menggabungkan CalendarProactive + timer-done watcher dalam satu class,
satu background scanner thread, satu callback (on_trigger).

Arsitektur notifikasi:
  - _calendar_loop      : scan kalender tiap check_interval_sec (default 1 jam)
                          untuk debug log + spawn/refresh watcher
  - _event_watcher      : thread per event, sleep presisi sampai waktu notif
                          (30 menit, 10 menit, tepat mulai) lalu fire
  - _maybe_spawn_watcher: deteksi datetime berubah → cancel watcher lama,
                          spawn baru; guard duplikat via _watched_events

Perubahan v5:
  - FIX: prompt LLM dipisah antara 'start' dan 'early' — start tidak minta
         sebutkan menit agar LLM tidak halusinasi angka
  - FIX: urutan discard di dt_changed — cek was_completed SEBELUM discard
         agar _logged_scan tidak di-reset kalau event sudah selesai
  - FIX: scan tidak muncul lagi setelah event completed + jadwal berubah
"""

from __future__ import annotations

import math
import threading
from datetime import datetime, timedelta
from typing import Callable

import pytz
import requests

from tools.calendar_tool import CalendarTool


# ── Konfigurasi default ────────────────────────────────────────────
DEFAULT_CONFIG: dict = {
    # Seberapa sering scan kalender untuk debug + spawn watcher (detik)
    "check_interval_sec": 3600,  # 1 jam

    # Notifikasi dini: berapa menit sebelum event
    "notify_before_min": 30,

    # Notifikasi kedua: berapa menit sebelum event
    "notify_before_min_2": 10,

    # Toleransi window saat event "tepat mulai" (menit)
    "notify_at_start_window_min": 2,

    # Timezone user
    "timezone": "Asia/Jakarta",

    # Prefix pesan kalender
    "calendar_prefix": "📅 ",

    # Prefix pesan timer
    "timer_prefix": "⏱️ ",
}


# ── Fallback message formatter ─────────────────────────────────────

def _fmt_reminder(title: str, minutes_left: int) -> str:
    if minutes_left <= 1:
        return f"Hei! Event '{title}' dimulai sekarang."
    return f"Pengingat: '{title}' akan dimulai dalam {minutes_left} menit."


def _fmt_start(title: str) -> str:
    return f"Waktunya '{title}' — sudah dimulai!"


# ══════════════════════════════════════════════════════════════════
# NotifProactive
# ══════════════════════════════════════════════════════════════════

class NotifProactive:
    """
    Unified proactive engine: kalender + timer.

    Parameters
    ----------
    calendar   : CalendarTool
    on_trigger : Callable[[str], None]
    llm_config : dict  — model_server, model, api_key
    config     : dict | None  — override DEFAULT_CONFIG
    debug      : bool
    """

    def __init__(
        self,
        calendar  : CalendarTool,
        on_trigger: Callable[[str], None],
        llm_config: dict,
        config    : dict | None = None,
        debug     : bool = False,
    ):
        self._calendar   = calendar
        self._on_trigger = on_trigger
        self._llm_config = llm_config
        self._debug      = debug

        self._cfg = {**DEFAULT_CONFIG, **(config or {})}
        self._tz  = pytz.timezone(self._cfg["timezone"])

        # {event_id: set[notif_type]}  notif_type ∈ {"early","early10","start"}
        self._notified      : dict[str, set[str]] = {}
        self._notified_lock = threading.Lock()

        # set of event_id yang sedang punya watcher thread aktif
        self._watched_events: set[str] = set()
        self._watched_lock  = threading.Lock()

        # {event_id: dt_str} — untuk deteksi perubahan jadwal
        self._event_datetimes: dict[str, str] = {}
        self._event_dt_lock   = threading.Lock()

        # set of cancel_key = "event_id:dt_str" — watcher lama wajib cek ini
        self._cancelled_watchers: set[str] = set()
        self._cancelled_lock     = threading.Lock()

        # set of event_id yang sudah selesai semua notifnya (start sudah fire)
        self._completed_events: set[str] = set()
        self._completed_lock   = threading.Lock()

        # set of event_id yang sudah pernah di-log di scan
        self._logged_scan     : set[str] = set()
        self._logged_scan_lock = threading.Lock()

        self._last_reset_day: str = ""
        self._cal_thread    : threading.Thread | None = None
        self._stop_event    = threading.Event()

    # ── Lifecycle ──────────────────────────────────────────────────

    def start(self):
        if self._cal_thread and self._cal_thread.is_alive():
            return
        self._stop_event.clear()
        self._cal_thread = threading.Thread(
            target=self._calendar_loop,
            name="NotifProactive-Scanner",
            daemon=True,
        )
        self._cal_thread.start()
        if self._debug:
            print(
                f"  [NotifPro] started "
                f"(scan interval={self._cfg['check_interval_sec']}s)",
                flush=True,
            )

    def stop(self, timeout: float = 5.0):
        self._stop_event.set()
        if self._cal_thread:
            self._cal_thread.join(timeout=timeout)
        if self._debug:
            print("  [NotifPro] stopped.", flush=True)

    # ── Timer callback ─────────────────────────────────────────────

    def on_timer_done(self, message: str):
        prefix = self._cfg.get("timer_prefix", "")
        if self._debug:
            print(f"  [NotifPro] timer done: {message[:60]}", flush=True)
        try:
            self._on_trigger(prefix + message)
        except Exception as e:
            print(f"  [NotifPro] on_trigger (timer) error: {e}", flush=True)

    # ── Manual check ───────────────────────────────────────────────

    def check_calendar_now(self):
        """Scan sinkron, bisa dipanggil dari command 'calendar'."""
        self._check_events()

    # ── Scanner loop ───────────────────────────────────────────────

    def _calendar_loop(self):
        self._check_events()
        interval = self._cfg["check_interval_sec"]
        while not self._stop_event.wait(timeout=interval):
            self._check_events()

    def _now(self) -> datetime:
        return datetime.now(self._tz)

    def _reset_if_new_day(self, today: str):
        if today == self._last_reset_day:
            return
        with self._notified_lock:
            self._notified.clear()
        with self._watched_lock:
            self._watched_events.clear()
        with self._event_dt_lock:
            self._event_datetimes.clear()
        with self._cancelled_lock:
            self._cancelled_watchers.clear()
        with self._completed_lock:
            self._completed_events.clear()
        with self._logged_scan_lock:
            self._logged_scan.clear()
        self._last_reset_day = today
        if self._debug:
            print(f"  [NotifPro] new day {today} — cache reset", flush=True)

    def _check_events(self):
        now   = self._now()
        today = now.strftime("%Y-%m-%d")
        self._reset_if_new_day(today)

        dates = [today]
        if now.hour >= 23:
            tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
            dates.append(tomorrow)

        for date_str in dates:
            result = self._calendar.get_events(date_str)
            events = result.get("events", [])

            if self._debug:
                # Kumpulkan event upcoming yang belum pernah di-log
                upcoming = []
                for event in events:
                    dt_str = event.get("datetime", "")
                    try:
                        event_dt = datetime.fromisoformat(dt_str)
                        if event_dt.tzinfo is None:
                            event_dt = self._tz.localize(event_dt)
                        delta_min = (event_dt - now).total_seconds() / 60
                        if delta_min >= -self._cfg["notify_at_start_window_min"]:
                            upcoming.append((event, delta_min))
                    except Exception:
                        pass

                # Hanya log event yang belum pernah di-log
                to_log = []
                for event, delta_min in upcoming:
                    eid = event.get("id", "")
                    with self._logged_scan_lock:
                        if eid not in self._logged_scan:
                            self._logged_scan.add(eid)
                            to_log.append((event, delta_min))

                if to_log:
                    print(
                        f"  [NotifPro] scan {date_str}: "
                        f"{len(upcoming)} upcoming / {len(events)} total",
                        flush=True,
                    )
                    for event, delta_min in to_log:
                        eid      = event.get("id", "")
                        title    = event.get("title", "?")
                        watched  = eid in self._watched_events
                        notified = self._notified.get(eid, set())
                        print(
                            f"  [NotifPro]   '{title}' "
                            f"delta={delta_min:+.1f}m "
                            f"watched={watched} "
                            f"notified={notified or '-'}",
                            flush=True,
                        )

            for event in events:
                self._maybe_spawn_watcher(event, now)

    # ── Watcher spawner ────────────────────────────────────────────

    def _maybe_spawn_watcher(self, event: dict, now: datetime):
        event_id = event.get("id", "")
        title    = event.get("title", "(tanpa judul)")
        dt_str   = event.get("datetime", "")

        if not dt_str or not event_id:
            return

        try:
            event_dt = datetime.fromisoformat(dt_str)
            if event_dt.tzinfo is None:
                event_dt = self._tz.localize(event_dt)
        except ValueError:
            return

        delta_min = (event_dt - now).total_seconds() / 60

        # Sudah lewat lebih dari window → tidak perlu watcher
        if delta_min < -self._cfg["notify_at_start_window_min"]:
            return

        # Sudah selesai semua notifnya → tidak perlu watcher
        with self._completed_lock:
            if event_id in self._completed_events:
                return

        # ── Deteksi perubahan datetime ─────────────────────────────
        with self._event_dt_lock:
            prev_dt_str = self._event_datetimes.get(event_id)
            dt_changed  = prev_dt_str is not None and prev_dt_str != dt_str
            self._event_datetimes[event_id] = dt_str

        if dt_changed:
            if self._debug:
                print(
                    f"  [NotifPro] '{title}' jadwal berubah "
                    f"{prev_dt_str} → {dt_str}, respawn watcher",
                    flush=True,
                )
            # Tandai watcher lama sebagai cancelled
            cancel_key = f"{event_id}:{prev_dt_str}"
            with self._cancelled_lock:
                self._cancelled_watchers.add(cancel_key)
            with self._watched_lock:
                self._watched_events.discard(event_id)
            with self._notified_lock:
                self._notified.pop(event_id, None)
            # Cek was_completed SEBELUM discard untuk hindari race condition
            with self._completed_lock:
                was_completed = event_id in self._completed_events
                self._completed_events.discard(event_id)
            # Reset log scan hanya jika belum completed
            if not was_completed:
                with self._logged_scan_lock:
                    self._logged_scan.discard(event_id)

        # Guard duplikat
        with self._watched_lock:
            if event_id in self._watched_events:
                return
            self._watched_events.add(event_id)

        if self._debug:
            print(
                f"  [NotifPro] spawn watcher '{title}' "
                f"delta={delta_min:+.1f}m",
                flush=True,
            )

        t = threading.Thread(
            target=self._event_watcher,
            args=(event_id, title, event_dt, dt_str),
            name=f"EventWatcher-{event_id[:8]}",
            daemon=True,
        )
        t.start()

    # ── Per-event watcher thread ───────────────────────────────────

    def _event_watcher(
        self,
        event_id: str,
        title   : str,
        event_dt: datetime,
        dt_str  : str,
    ):
        """
        Thread per event. Sleep presisi ke setiap waktu notif,
        cek cancel sebelum dan sesudah tidur, lalu cleanup.
        """
        cancel_key  = f"{event_id}:{dt_str}"
        early_min   = self._cfg["notify_before_min"]
        early10_min = self._cfg["notify_before_min_2"]
        window      = self._cfg["notify_at_start_window_min"]

        targets = [
            (early_min,   "early"),
            (early10_min, "early10"),
            (0,           "start"),
        ]

        for minutes_before, notif_type in targets:
            if self._stop_event.is_set():
                break

            # Cek cancel sebelum tidur
            with self._cancelled_lock:
                if cancel_key in self._cancelled_watchers:
                    if self._debug:
                        print(
                            f"  [NotifPro] watcher cancelled '{title}'",
                            flush=True,
                        )
                    return

            now        = self._now()
            fire_at    = event_dt - timedelta(minutes=minutes_before)
            sleep_secs = (fire_at - now).total_seconds()

            # Cek apakah waktu notif sudah lewat
            cutoff = -(window * 60) if notif_type == "start" else -60
            if sleep_secs < cutoff:
                if self._debug:
                    print(
                        f"  [NotifPro] skip [{notif_type}] '{title}' "
                        f"(lewat {-sleep_secs:.0f}s)",
                        flush=True,
                    )
                continue

            # Tidur sampai waktu fire
            if sleep_secs > 0:
                if self._debug:
                    print(
                        f"  [NotifPro] wait [{notif_type}] '{title}' "
                        f"{sleep_secs / 60:.1f} menit",
                        flush=True,
                    )
                self._stop_event.wait(timeout=sleep_secs)

            if self._stop_event.is_set():
                break

            # Cek cancel setelah tidur
            with self._cancelled_lock:
                if cancel_key in self._cancelled_watchers:
                    if self._debug:
                        print(
                            f"  [NotifPro] watcher cancelled (post-sleep) '{title}'",
                            flush=True,
                        )
                    return

            # Fire jika belum pernah di notif_type ini
            with self._notified_lock:
                already_fired = notif_type in self._notified.get(event_id, set())

            if not already_fired:
                delta_min = (event_dt - self._now()).total_seconds() / 60
                self._fire(
                    title,
                    math.ceil(max(0.0, delta_min)),
                    event_id,
                    notif_type,
                )

        # Cleanup
        with self._watched_lock:
            self._watched_events.discard(event_id)
        with self._cancelled_lock:
            self._cancelled_watchers.discard(cancel_key)

        # Tandai completed jika 'start' sudah di-fire
        with self._notified_lock:
            start_fired = "start" in self._notified.get(event_id, set())
        if start_fired:
            with self._completed_lock:
                self._completed_events.add(event_id)

        if self._debug:
            print(f"  [NotifPro] watcher done '{title}'", flush=True)

    # ── LLM message generator ──────────────────────────────────────

    def _llm_notif_message(self, title: str, minutes_left: int, notif_type: str) -> str:
        if notif_type == "start":
            situation = f"Event '{title}' tepat dimulai sekarang."
            prompt = (
                "Kamu adalah MEI, AI personal assistant yang ramah dan casual.\n"
                "Berikan satu kalimat pemberitahuan singkat dalam Bahasa Indonesia "
                "bahwa event sudah dimulai sekarang.\n"
                "Jangan sebutkan angka menit sama sekali.\n"
                "Langsung isi kalimatnya saja, tanpa preamble.\n\n"
                f"Situasi: {situation}"
            )
        else:
            situation = (
                f"Event '{title}' akan dimulai dalam "
                f"{minutes_left} menit lagi."
            )
            prompt = (
                "Kamu adalah MEI, AI personal assistant yang ramah dan casual.\n"
                "Berikan satu kalimat pengingat singkat dalam Bahasa Indonesia.\n"
                f"WAJIB sebutkan angka '{minutes_left} menit' secara eksplisit.\n"
                "Langsung isi kalimatnya saja, tanpa preamble.\n\n"
                f"Situasi: {situation}"
            )

        try:
            resp = requests.post(
                f"{self._llm_config['model_server']}/chat/completions",
                headers={
                    "Authorization": (
                        f"Bearer {self._llm_config.get('api_key', 'lm-studio')}"
                    )
                },
                json={
                    "model"      : self._llm_config["model"],
                    "messages"   : [{"role": "user", "content": prompt}],
                    "max_tokens" : 80,
                    "temperature": 0.7,
                },
                timeout=10,
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]["content"].strip()
            return msg.strip('"').strip("'")
        except Exception as e:
            if self._debug:
                print(f"  [NotifPro] LLM gagal, fallback: {e}", flush=True)
            return (
                _fmt_start(title)
                if notif_type == "start"
                else _fmt_reminder(title, minutes_left)
            )

    def _fire(self, title: str, minutes_left: int, event_id: str, notif_type: str):
        if self._debug:
            print(
                f"  [NotifPro] FIRE [{notif_type}] '{title}' "
                f"({minutes_left} menit)",
                flush=True,
            )
        with self._notified_lock:
            self._notified.setdefault(event_id, set()).add(notif_type)

        msg    = self._llm_notif_message(title, minutes_left, notif_type)
        prefix = self._cfg.get("calendar_prefix", "")
        try:
            self._on_trigger(prefix + msg)
        except Exception as e:
            print(f"  [NotifPro] on_trigger error: {e}", flush=True)