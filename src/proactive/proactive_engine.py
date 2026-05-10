"""
Proactive Engine
=================
State machine + background thread yang mengawasi inaktivitas user
dan memicu percakapan proaktif dari MEI.

State Flow:
                        ┌─────────────┐
                        │    IDLE     │  ← user aktif / session baru
                        └──────┬──────┘
                               │  inactivity_threshold tercapai
                               ▼
                        ┌─────────────┐
                        │  WAITING_1  │  ← trigger pertama dikirim
                        └──────┬──────┘
                               │  retry_interval tercapai (masih diam)
                               ▼
                        ┌─────────────┐
                        │  WAITING_2  │  ← trigger kedua dikirim
                        └──────┬──────┘
                               │  masih tidak ada respon
                               ▼
                        ┌─────────────┐
                        │  AFK_MODE   │  ← diam total selama afk_cooldown
                        └──────┬──────┘
                               │  user kirim pesan baru
                               ▼
                        ┌─────────────┐
                        │    IDLE     │  ← reset, mulai dari awal
                        └─────────────┘

Di mana pun ada user message masuk → langsung RESET ke IDLE.
"""

import threading
import time
import logging
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Callable, Optional

from .proactive_config import PROACTIVE_CONFIG
from .topic_suggester import TopicSuggester


logger = logging.getLogger("proactive_engine")


# ──────────────────────────────────────────────────────────
# STATE DEFINITION
# ──────────────────────────────────────────────────────────

class ProactiveState(Enum):
    IDLE       = auto()  # user aktif, tidak ada proaktif
    WAITING_1  = auto()  # sudah kirim trigger pertama, menunggu respon
    WAITING_2  = auto()  # sudah kirim trigger kedua, menunggu respon
    AFK_MODE   = auto()  # user dianggap AFK, engine berhenti sementara


# ──────────────────────────────────────────────────────────
# PROACTIVE ENGINE
# ──────────────────────────────────────────────────────────

class ProactiveEngine:
    """
    Background engine yang memantau inaktivitas dan memanggil
    callback saat ingin memulai percakapan.

    Usage:
        engine = ProactiveEngine(llm_config, on_trigger_callback)
        engine.start()

        # Saat user kirim pesan:
        engine.on_user_message(conversation_history)

        # Saat session selesai:
        engine.stop()
    """

    def __init__(
        self,
        llm_config: dict,
        on_trigger: Callable[[str], None],
        user_name: str = "user",
        config: dict = None,
    ):
        """
        Args:
            llm_config  : config LLM (model, model_server, api_key)
            on_trigger  : callback dipanggil dengan (message: str)
                          ketika engine ingin memulai percakapan.
                          Callback ini yang bertanggung jawab menampilkan
                          pesan di terminal dan/atau TTS.
            user_name   : nama user untuk personalisasi generate
            config      : override PROACTIVE_CONFIG (opsional)
        """
        self.cfg = {**PROACTIVE_CONFIG, **(config or {})}
        self.on_trigger = on_trigger
        self.user_name = user_name
        self.suggester = TopicSuggester(llm_config)

        # ─── State ───
        self._state = ProactiveState.IDLE
        self._state_lock = threading.Lock()

        # ─── Timestamps ───
        self._last_user_activity: datetime = datetime.now()
        self._last_trigger_time: Optional[datetime] = None
        self._afk_since: Optional[datetime] = None

        # ─── Context ───
        self._conversation_history: list[dict] = []
        self._session_start: datetime = datetime.now()
        self._message_count: int = 0
        self._trigger_count: int = 0   # berapa kali trigger dalam sesi inaktif ini

        # ─── Thread control ───
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._check_interval = 15  # cek state setiap 15 detik

        if self.cfg["debug_logging"]:
            logging.basicConfig(level=logging.DEBUG)

    # ──────────────────────────────────────────
    # LIFECYCLE
    # ──────────────────────────────────────────

    def start(self):
        """Mulai background monitoring thread."""
        if not self.cfg["enabled"]:
            logger.debug("ProactiveEngine disabled, tidak dijalankan.")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="proactive-monitor",
            daemon=True,
        )
        self._thread.start()
        logger.debug("ProactiveEngine started.")

    def stop(self):
        """Hentikan background thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        logger.debug("ProactiveEngine stopped.")

    # ──────────────────────────────────────────
    # PUBLIC HOOKS (dipanggil dari main loop)
    # ──────────────────────────────────────────

    def on_user_message(self, conversation_history: list[dict]):
        """
        Dipanggil SETIAP KALI ada pesan baru dari user.
        Mereset state ke IDLE dan update konteks.
        """
        with self._state_lock:
            prev_state = self._state
            self._state = ProactiveState.IDLE
            self._last_user_activity = datetime.now()
            self._conversation_history = conversation_history
            self._message_count += 1
            self._trigger_count = 0  # reset trigger counter

            if prev_state == ProactiveState.AFK_MODE:
                logger.debug(f"User kembali dari AFK. Reset ke IDLE.")
            else:
                logger.debug(f"User aktif. State IDLE. Messages: {self._message_count}")

    def get_state(self) -> ProactiveState:
        """Return state saat ini (untuk debug / display)."""
        with self._state_lock:
            return self._state

    def is_afk(self) -> bool:
        with self._state_lock:
            return self._state == ProactiveState.AFK_MODE

    # ──────────────────────────────────────────
    # BACKGROUND MONITOR LOOP
    # ──────────────────────────────────────────

    def _monitor_loop(self):
        """Loop utama yang berjalan di background thread."""
        while self._running:
            try:
                self._tick()
            except Exception as e:
                logger.error(f"Error di monitor loop: {e}")
            time.sleep(self._check_interval)

    def _tick(self):
        """Satu iterasi pengecekan state."""
        with self._state_lock:
            now = datetime.now()
            state = self._state
            inactive_sec = (now - self._last_user_activity).total_seconds()
            inactive_min = inactive_sec / 60

        # ─── Guard: minimal context sebelum proaktif ───
        if self._message_count < self.cfg["min_messages_for_proactive"]:
            return

        session_age_min = (datetime.now() - self._session_start).total_seconds() / 60
        if session_age_min < self.cfg["min_session_duration_minutes"]:
            return

        # ─── State transitions ───

        if state == ProactiveState.IDLE:
            threshold = self.cfg["inactivity_threshold_minutes"]
            if inactive_min >= threshold:
                logger.debug(f"Inaktif {inactive_min:.1f} menit. Kirim trigger pertama.")
                self._do_trigger(trigger_count=1)

        elif state == ProactiveState.WAITING_1:
            retry_min = self.cfg["retry_interval_minutes"]
            time_since_trigger = (
                (datetime.now() - self._last_trigger_time).total_seconds() / 60
                if self._last_trigger_time else 999
            )
            if time_since_trigger >= retry_min:
                logger.debug(f"Tidak ada respon setelah {retry_min} menit. Kirim retry.")
                self._do_trigger(trigger_count=2)

        elif state == ProactiveState.WAITING_2:
            retry_min = self.cfg["retry_interval_minutes"]
            time_since_trigger = (
                (datetime.now() - self._last_trigger_time).total_seconds() / 60
                if self._last_trigger_time else 999
            )
            if time_since_trigger >= retry_min:
                logger.debug("Masih tidak ada respon. Masuk AFK mode.")
                with self._state_lock:
                    self._state = ProactiveState.AFK_MODE
                    self._afk_since = datetime.now()

        elif state == ProactiveState.AFK_MODE:
            afk_cooldown = self.cfg["afk_cooldown_minutes"]
            afk_min = (
                (datetime.now() - self._afk_since).total_seconds() / 60
                if self._afk_since else 0
            )
            if afk_min >= afk_cooldown:
                # Setelah cooldown, reset ke IDLE
                # (user masih belum balik, tapi state direset
                #  agar siap trigger lagi kalau user kembali)
                logger.debug(f"AFK cooldown ({afk_cooldown} menit) selesai. Reset ke IDLE.")
                with self._state_lock:
                    self._state = ProactiveState.IDLE
                    self._trigger_count = 0

    # ──────────────────────────────────────────
    # TRIGGER ACTION
    # ──────────────────────────────────────────

    def _do_trigger(self, trigger_count: int):
        """Generate pesan dan panggil callback on_trigger."""
        # Update state SEBELUM generate (agar tidak double trigger)
        with self._state_lock:
            if trigger_count == 1:
                self._state = ProactiveState.WAITING_1
            else:
                self._state = ProactiveState.WAITING_2
            self._trigger_count = trigger_count
            self._last_trigger_time = datetime.now()
            history_snapshot = list(self._conversation_history)

        # Generate pesan (di luar lock agar tidak block state)
        try:
            raw_message = self.suggester.generate(
                conversation_history=history_snapshot,
                trigger_count=trigger_count,
                user_name=self.user_name,
            )

            # Apply template
            if trigger_count == 1:
                template = self.cfg["trigger_template_first"]
            else:
                template = self.cfg["trigger_template_retry"]

            final_message = template.format(generated_message=raw_message)

            # Panggil callback (biasanya menampilkan ke terminal + TTS)
            self.on_trigger(final_message)
            logger.debug(f"Trigger {trigger_count} dikirim: {final_message}")

        except Exception as e:
            logger.error(f"Error saat do_trigger: {e}")