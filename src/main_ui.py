"""
MEI — AI Personal Assistant
=============================
Version 4.5.9

Perubahan v4.5.9:
  - ADD: _store_yesterday_summary_to_chroma() — saat startup, daily summary
         kemarin otomatis dimasukkan ke ChromaDB tanpa filter/SLM.
         Daily summary sudah condensed LLM (<400 char) sehingga langsung aman.
         Marker file {date}.chroma_stored mencegah double insert saat restart.
  - MOD: run_agent() — panggil _store_yesterday_summary_to_chroma() setelah
         embedding warmup selesai.

Perubahan v4.5.8:
  - ADD: _stream_and_speak() sekarang menerima parameter on_token callback
         (Callable[[str], None] | None) — dipanggil per-delta saat streaming
  - MOD: _process_turn() meneruskan on_token ke _stream_and_speak()
  - MOD: _make_ui_stt_and_callback() — _ui_agent_cb sekarang menerima
         on_token: Callable[[str], None] | None sebagai parameter ketiga
  - MOD: MEIApp.agent_callback signature: (text, mode, on_token) → str
         UI harus expose method stream_token(delta) dan finalize_stream()
  - ADD: shared key '_ui_on_token' untuk forward callback dari UI thread
  - NOTE: Backward compatible — jika on_token=None, perilaku sama dengan v4.5.7

Perubahan v4.5.7:
  - ADD: CHROMA_TEST_QUERIES  — 10 query episodic trigger
  - ADD: CHROMA_JSONL_PATH    — output path mei_chroma.jsonl
  - ADD: _run_chroma_test()   — runner 10 query via agent asli,
         hasil disimpan ke mei_chroma.jsonl
  - ADD: shared keys '_last_chroma_triggered', '_last_chroma_hits',
         '_last_hit_contents', '_last_hit_relevances'
         di-set oleh _process_turn() untuk keperluan chroma test
  - ADD: command 'test chroma' di terminal loop + UI callback
  - MOD: banner menampilkan 'test chroma'

Perubahan v4.5.6:
  - ADD: _nuclear_trim_history()   → layer 2 trim (keep 2 msg)
  - ADD: _layer3_trim_history()    → layer 3 trim (kosong total)
  - ADD: build_lean_system_message() → system msg minimal untuk L3
  - MOD: _process_turn() — nested ContextExceededError sekarang punya
    4 layer:
      L1 aggressive trim (~⅓)  → auto-retry
      L2 nuclear trim (2 msg)  → auto-retry
      L3 empty + lean sys msg  → auto-retry
      L4 hard error (input terlalu panjang)
  - FIX: Setiap layer auto-retry dengan user_input yang sama,
    tidak minta user ketik ulang atau arahkan ke 'clear'
  - NOTE: system_message & profile_agent TIDAK pernah dipangkas
    di L1/L2. Hanya di L3 system msg di-lean (persona inti saja).

Perubahan v4.5.5:
  - ADD: gt_by_turn lookup dict dari GT_DATASET (turn_no → label_gt, imp, tool_gt)
  - ADD: chroma_triggered bool di _process_turn — tracking apakah ChromaDB
    di-query pada turn ini, dikirim ke log_turn
  - ADD: fact_extractor._on_fact_saved callback → turn_logger.on_fact_result
    agar fact_label_predicted + fact_saved tercatat di JSONL secara async
  - ADD: log_turn() sekarang terima label_gt, chroma_triggered, chroma_hits
  - ADD: fact_extractor._current_turn_no diset sebelum log_turn dipanggil
    agar callback tahu turn mana yang sedang diproses

Memory stack:
  [1] JSONL           — window 10 turn short-term
  [2] MEMORY.md       — profil manual user (READ ONLY untuk AI, user yang nulis)
  [3] PROFILE_AGENT.md — persona + rules MEI (konstan)
  [4] Daily notes     — hanya 1 [summary] TERAKHIR (max 400 char)
  [5] ChromaDB        — HANYA saat ada trigger keyword historis

Tools (src/tools/):
  - camera_capture        → ambil foto + analisis
  - internet_search       → cari informasi ke internet
  - create_event          → buat event di calendar
  - get_events            → ambil event berdasarkan tanggal
  - delete_event          → hapus event berdasarkan ID
  - set_timer             → countdown timer
  - list_timers           → tampilkan timer aktif
  - cancel_timer          → batalkan timer
  - schedule_notification → jadwalkan notifikasi
  - list_notifications    → tampilkan notifikasi terjadwal
  - cancel_notification   → batalkan notifikasi

I/O Modes:
  1. Text → Text        (chat biasa, default)
  2. Text → TTS         (ketik, dengarkan)
  3. STT  → TTS         (full voice loop)
  4. STT  → TTS + RVC   (full pipeline, voice conversion)
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import logging

logging.basicConfig(level=logging.WARNING)
for name in ["httpx", "httpcore", "qwen_agent_logger", "base", "oai", "urllib3",
             "qwen_agent", "openai", "openai._base_client", "httpx._client"]:
    logging.getLogger(name).setLevel(logging.WARNING)

import json
import queue
import re
import io
import wave
import sys
import time
import threading
import platform
import select as _select
import requests
from pathlib import Path
from dataclasses import dataclass, field
from enum import IntEnum
from datetime import date, timedelta, datetime
from typing import Optional, Callable

from qwen_agent.agents import Assistant

# ── Custom tools ───────────────────────────────────────────────────────
from tools.camera_capture import CameraCaptureTool
from tools.internet_search import InternetSearch
from tools import (
    calendar_instance,
    set_timer_tool,
    list_timer_tool,
    cancel_timer_tool,
    create_event_tool,
    get_events_tool,
    delete_event_tool,
    get_base_tools,
)

# ── Proactive ──────────────────────────────────────────────────────────
from proactive import ProactiveEngine, PROACTIVE_CONFIG
from proactive.notif_proactive import NotifProactive
from proactive.notification_engine import (
    NotificationEngine,
    ScheduleNotificationTool,
    ListNotificationsTool,
    CancelNotificationTool,
)

# ── Memory ─────────────────────────────────────────────────────────────
from memory.jsonl_memory import JSONLMemoryManager
from memory.long_term_memory import LongTermMemoryManager
from memory.fact_extractor import FactExtractor

# ── Config ─────────────────────────────────────────────────────────────
from config.agent_config import (
    OUTPUT_FILTERS, MEMORY_CONFIG, LLM_CONFIG, RVC_CONFIG,
    CLASSIFIER_CONFIG,
    set_device,
)

# ── Latency & Token Tracker ───────────────────────────────────────────
from debug.latency.debug_latency import (
    LatencyTracker,
    log_latency_report,
    token_logger,
)

# ── Turn Logger ───────────────────────────────────────────────────────
from debug.logging.mei_turn_logger import TurnLogger

# ── Voice ──────────────────────────────────────────────────────────────
from voice.stt import STTEngine, STTConfig
from voice.tts import IndonesianTTS
from voice.rvc import RVCConverter

# ── Desktop UI ────────────────────────────────────────────────────────
try:
    from mei_ui import MEIApp
    _UI_AVAILABLE = True
except ImportError:
    _UI_AVAILABLE = False
    print("  [MEI] mei_ui.py tidak ditemukan — UI dinonaktifkan, terminal only.")

import numpy as np
import sounddevice as sd
import pytz

token_logger.install()

_SUPPRESSED_PREFIXES = ('qwen_agent', 'qwen_agent_logger', 'base', 'oai',
                        'openai', 'httpx', 'httpcore', 'urllib3')

class _QwenContentFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name
        return not any(
            name == p or name.startswith(p + '.')
            for p in _SUPPRESSED_PREFIXES
        )

for _rh in logging.root.handlers:
    _rh.addFilter(_QwenContentFilter())


# ══════════════════════════════════════════════════════════════════
# PATHS & CONSTANTS
# ══════════════════════════════════════════════════════════════════

SRC_DIR  = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
STORAGE  = ROOT_DIR / "storage"

USER_ID = "rifki"

DEBUG_TOOL = False

# Test mode: jeda (detik) antar input GT otomatis
TEST_INPUT_INTERVAL_S = 1.5

SENTENCE_END   = re.compile(r'[.!?。！？]+\s*')
SOFT_BOUNDARY  = re.compile(r'[,;:]\s*')
MIN_SOFT_CHARS = 40

MAX_AGENT_MESSAGES = 7
EXIT_CHUNK_SIZE    = 20

MAX_HISTORY_MESSAGES     = 10
MAX_HISTORY_MESSAGES_CPU = 4

MAX_FUNCTION_RESULT_CHARS = 400

_STOP_SENTINEL = object()


# ══════════════════════════════════════════════════════════════════
# GT DATASET — 27 input test case
# turn_no: 1, 3, 5, ... (genap = respons LLM)
# ══════════════════════════════════════════════════════════════════

GT_DATASET: list[tuple[int, str, str, int, Optional[str]]] = [
    # (turn_no, input_text, label, importance, tool_gt)
    ( 1, "Hai, nama saya rifki, saya seorang backend developer.",                 "data_pribadi",  9, None),
    ( 3, "Senang banget punya asisten AI yang bisa diajak ngobrol gini.",                   "tidak_penting", 1, None),
    ( 5, "Oh ya, tolong cek jadwal kerja saya hari ini dong.",                              "pekerjaan",     8, "get_events"),
    ( 7, "Foto CV saya ini ya, biar kamu tahu background pengalaman saya.",                 "pengalaman",    6, "camera_capture"),
    ( 9, "Tolong buatin event 'Sprint Planning' untuk besok jam 9 pagi.",                   "project",       8, "create_event"),
    (11, "Cari info tentang Kong API Gateway dong, buat referensi project saya.",           "teknis",        7, "internet_search"),
    (13, "Cek jadwal saya minggu ini, saya biasanya prefer meeting pagi.",                  "preferensi",    7, "get_events"),
    (15, "Haha seru juga, set timer 5 menit dulu ah buat jeda sebentar.",                   "tidak_penting", 1, "set_timer"),
    (17, "Ngomong-ngomong, apa ya perbedaan REST API dan GraphQL?",                         "teknis",        7, "internet_search"),
    (19, "Nomor HP saya 0812-3456-7890, kalau perlu dihubungi.",                            "data_pribadi",  9, None),
    (21, "Saya sudah 4 tahun pengalaman di web development, fokusnya di backend.",          "pengalaman",    6, None),
    (23, "Tolong foto diagram arsitektur project yang ada di papan tulis ini.",             "project",       8, "camera_capture"),
    (25, "Set timer 25 menit buat sesi Pomodoro coding sekarang.",                          "lainnya",       5, "set_timer"),
    (27, "Cari tutorial cara setup Docker di Ubuntu 22.04 dong.",                           "teknis",        7, "internet_search"),
    (29, "Buatin jadwal meeting dengan klien, Kamis jam 2 siang.",                          "pekerjaan",     8, "create_event"),
    (31, "Ada event apa saja yang terjadwal minggu ini?",                                   "pekerjaan",     8, "get_events"),
    (33, "Bikin event standup harian rutin tiap Senin sampai Jumat jam 8 pagi.",            "preferensi",    7, "create_event"),
    (35, "Set timer 30 menit buat sesi review kode project API ini.",                       "project",       8, "set_timer"),
    (37, "Oke, ngerti sekarang. Makasih ya infonya.",                                       "tidak_penting", 1, None),
    (39, "Oh ya, email kantor saya rifki@devstudio.id.",                                     "data_pribadi",  9, None),
    (41, "Foto pesan error di terminal ini, tolong bantu analisis masalahnya.",             "lainnya",       5, "camera_capture"),
    (43, "Bikin reminder 'Demo ke Klien' untuk Jumat jam 3 sore.",                          "pekerjaan",     8, "create_event"),
    (45, "Set timer 10 menit buat istirahat sebentar dulu.",                                "lainnya",       5, "set_timer"),
    (47, "Cari cara implementasi JWT authentication di Node.js.",                           "teknis",        7, "internet_search"),
    (49, "Foto halaman portfolio project lama saya ini buat referensi.",                    "pengalaman",    6, "camera_capture"),
    (51, "Cek jadwal besok pagi, saya prefer mulai aktivitas dari jam 8.",                  "preferensi",    7, "get_events"),
    (53, "Sip, segitu dulu untuk sekarang. Nanti lanjut lagi.",                             "tidak_penting", 1, None),
]

# ── v4.5.5: GT lookup dict (turn_no → metadata) ─────────────────
gt_by_turn: dict[int, dict] = {
    turn_no: {"label_gt": label, "importance": imp, "tool_gt": tool}
    for turn_no, _, label, imp, tool in GT_DATASET
}


# ══════════════════════════════════════════════════════════════════
# CHROMA TEST QUERIES — v4.5.7
# ══════════════════════════════════════════════════════════════════

CHROMA_TEST_QUERIES = [
    # q01 — jadwal standup (tidak ada di memory.md)
    ('q01', 'Kamu masih ingat jadwal standup rutin saya itu hari apa dan jam berapa?',
     ['Senin', 'jam 8']),

    # q02 — teknologi environment (tidak ada di memory.md)
    ('q02', 'Kamu ingat teknologi apa yang saya pakai untuk environment development saya?',
     ['Docker', 'Ubuntu']),

    # q03 — metode coding (tidak ada di memory.md)
    ('q03', 'Saya pernah bilang soal metode kerja saya buat sesi coding. Apa itu?',
     ['Pomodoro', '25 menit']),

    # q04 — project (tidak ada di memory.md — bukan MEI/skripsi)
    ('q04', 'Saya pernah cerita soal project yang lagi saya kerjakan. Itu tentang apa?',
     ['Kong', 'API Gateway']),

    # q05 — jadwal meeting klien (tidak ada di memory.md)
    ('q05', 'Ingatkan saya, jadwal meeting rutin dengan klien itu hari apa dan jam berapa?',
     ['Kamis', 'jam 2']),

    # q06 — jadwal demo (tidak ada di memory.md)
    ('q06', 'Kamu masih ingat jadwal demo saya ke klien itu kapan?',
     ['Jumat', 'jam 3']),

    # q07 — pengalaman kerja (memory.md bilang Mahasiswa, bukan backend developer)
    ('q07', 'Berapa tahun pengalaman saya di web development yang sudah saya ceritakan dulu?',
     ['4 tahun', 'backend']),

    # q08 — preferensi meeting (tidak ada di memory.md)
    ('q08', 'Kamu ingat preferensi jam meeting saya? Saya biasanya prefer mulai jam berapa?',
     ['pagi', 'jam 8', 'jam 9']),

    # q09 — tempat kerja (memory.md bilang Mahasiswa, bukan DevStudio)
    ('q09', 'Saya pernah cerita soal pekerjaan dan tempat kerja saya yang sekarang. Apa itu?',
     ['DevStudio', 'backend developer']),

    # q10 — teknologi backend spesifik (tidak ada di memory.md)
    ('q10', 'Kamu ingat stack teknologi backend yang saya pakai di project saya?',
     ['Express', 'PostgreSQL']),
]

CHROMA_JSONL_PATH = SRC_DIR / 'debug' /'mei_chroma.jsonl'


# ══════════════════════════════════════════════════════════════════
# REALTIME TOOLS
# ══════════════════════════════════════════════════════════════════

REALTIME_TOOLS = {
    "camera_capture",
    "internet_search",
    "get_events",
    "create_event",
    "delete_event",
    "set_timer",
    "cancel_timer",
    "schedule_notification",
    "cancel_notification",
    "list_timers",
    "list_notifications",
}


# ══════════════════════════════════════════════════════════════════
# CUSTOM EXCEPTIONS
# ══════════════════════════════════════════════════════════════════

class ContextExceededError(Exception):
    pass


# ══════════════════════════════════════════════════════════════════
# INTERRUPT STATE
# ══════════════════════════════════════════════════════════════════

@dataclass
class InterruptState:
    event        : threading.Event = field(default_factory=threading.Event)
    pending_input: str             = ""
    by_user      : bool            = False

    def reset(self):
        self.event.clear()
        self.pending_input = ""
        self.by_user       = False

    def trigger(self, text: str = ""):
        self.pending_input = text.strip()
        self.by_user       = True
        self.event.set()


def _start_interrupt_listener(intr: InterruptState) -> threading.Thread:
    def _listener():
        try:
            if platform.system() == "Windows":
                import msvcrt
                typed = ""
                while not intr.event.is_set():
                    if msvcrt.kbhit():
                        ch = msvcrt.getwche()
                        if ch in ("\r", "\n"):
                            if not intr.event.is_set():
                                intr.trigger(typed)
                                _suffix = f"→ '{typed}'" if typed else "dihentikan."
                                print(f"\n  [↩] Interrupt: {_suffix}", flush=True)
                            break
                        elif ch == "\x08":
                            typed = typed[:-1]
                        else:
                            typed += ch
                    else:
                        time.sleep(0.05)
            else:
                while not intr.event.is_set():
                    r, _, _ = _select.select([sys.stdin], [], [], 0.1)
                    if r:
                        typed = sys.stdin.readline().rstrip("\n")
                        if not intr.event.is_set():
                            intr.trigger(typed)
                            _suffix = f"→ '{typed}'" if typed else "dihentikan."
                            print(f"\n  [↩] Interrupt: {_suffix}", flush=True)
                        break
        except Exception as exc:
            _dbg(f"InterruptListener error: {exc}")

    t = threading.Thread(target=_listener, name="InterruptListener", daemon=True)
    t.start()
    return t


# ══════════════════════════════════════════════════════════════════
# LOGGING HELPERS
# ══════════════════════════════════════════════════════════════════

def _log(msg: str):
    print(f"  [MEI] {msg}", flush=True)

def _dbg(msg: str):
    if DEBUG_TOOL:
        print(f"  [MEI][DBG] {msg}", flush=True)

def _warn(msg: str):
    print(f"  [MEI][WARN] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════
# I/O MODE
# ══════════════════════════════════════════════════════════════════

class IOMode(IntEnum):
    TEXT_TEXT   = 1
    TEXT_TTS    = 2
    STT_TTS     = 3
    STT_TTS_RVC = 4

IO_MODE_LABELS = {
    IOMode.TEXT_TEXT:   "Text → Text    (default)",
    IOMode.TEXT_TTS:    "Text → TTS",
    IOMode.STT_TTS:     "STT  → TTS",
    IOMode.STT_TTS_RVC: "STT  → TTS + RVC",
}

@dataclass
class IOState:
    mode   : IOMode = IOMode.TEXT_TEXT
    use_gpu: bool   = True

    @property
    def voice_in(self)  -> bool: return self.mode in (IOMode.STT_TTS, IOMode.STT_TTS_RVC)
    @property
    def voice_out(self) -> bool: return self.mode in (IOMode.TEXT_TTS, IOMode.STT_TTS, IOMode.STT_TTS_RVC)
    @property
    def use_rvc(self)   -> bool: return self.mode == IOMode.STT_TTS_RVC
    @property
    def device(self)    -> str:  return "cuda" if self.use_gpu else "cpu"

    def describe(self) -> str:
        hw = "GPU (CUDA)" if self.use_gpu else "CPU"
        return f"{IO_MODE_LABELS[self.mode]}  |  hw: {hw}"


# ══════════════════════════════════════════════════════════════════
# VOICE STATE WRAPPER
# ══════════════════════════════════════════════════════════════════

@dataclass
class VoiceState:
    stt     : object = None
    tts     : object = None
    rvc     : object = None
    voice_ok: bool   = False
    rvc_ok  : bool   = False


# ══════════════════════════════════════════════════════════════════
# RUNTIME CONTEXT
# ══════════════════════════════════════════════════════════════════

def get_runtime_context() -> str:
    tz  = pytz.timezone("Asia/Jakarta")
    now = datetime.now(tz)
    return (
        f"\nKONTEKS SAAT INI:\n"
        f"- Waktu sekarang: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- Zona waktu: Asia/Jakarta\n"
        f"- Lokasi user: Bandung, Indonesia\n"
    )


# ══════════════════════════════════════════════════════════════════
# DAILY SUMMARY
# ══════════════════════════════════════════════════════════════════

def _generate_daily_summary_llm(messages: list[dict], llm_config: dict) -> str:
    if not messages:
        return ""

    conv_lines: list[str] = []
    total = 0
    for m in messages[-30:]:
        role    = m.get("role", "?").upper()
        content = (m.get("content") or "")[:200].replace("\n", " ")
        line    = f"[{role}] {content}"
        total  += len(line) + 1
        if total > 3000:
            break
        conv_lines.append(line)

    conv_text = "\n".join(conv_lines)
    today     = date.today().isoformat()

    prompt = (
        "Buat ringkasan singkat percakapan berikut dalam Bahasa Indonesia.\n"
        f"Format output (HANYA ini, tanpa penjelasan lain):\n\n"
        f"## Daily Summary — {today}\n"
        "[2-4 kalimat ringkasan topik utama, keputusan, atau hal penting yang dibahas]\n\n"
        "---\n\n"
        f"Percakapan:\n{conv_text}"
    )
    try:
        resp = requests.post(
            f"{llm_config['model_server']}/chat/completions",
            headers={"Authorization": f"Bearer {llm_config.get('api_key', 'lm-studio')}"},
            json={
                "model"      : llm_config["model"],
                "messages"   : [{"role": "user", "content": prompt}],
                "max_tokens" : 300,
                "temperature": 0.3,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        _warn(f"Daily summary LLM gagal: {e}")
        user_msgs = [m["content"][:60] for m in messages if m.get("role") == "user"]
        topics    = " | ".join(user_msgs[:5])
        return f"## Daily Summary — {today}\nTopik: {topics}"


# ══════════════════════════════════════════════════════════════════
# EPISODIC TRIGGER
# ══════════════════════════════════════════════════════════════════

EPISODIC_TRIGGERS = [
    "ingat", "inget", "kemarin", "kemaren", "dulu", "waktu itu", "pernah",
    "sebelumnya", "sebelum ini", "lupa", "remind", "tadi", "minggu lalu",
    "bulan lalu", "sudah pernah", "udah pernah", "ceritanya", "katanya",
    "tempo hari", "beberapa waktu", "belum lama", "awal tadi",
    "kamu tau", "kamu tahu", "kau tau", "kau tahu",
    "aku pernah", "gue pernah", "saya pernah",
    "kita pernah", "kita udah", "kita sudah",
    "pernah bilang", "pernah ngomong", "pernah cerita",
    "udah bahas", "sudah bahas", "udah diskusi", "sudah diskusi",
    "yang kemarin", "yang tadi", "yang dulu",
    "balik lagi", "balik ke",
    "masih inget", "masih ingat",
    "lanjut dari", "lanjutin dari",
    "januari", "februari", "maret", "april", "mei", "juni",
    "juli", "agustus", "september", "oktober", "november", "desember",
]

_EPISODIC_PATTERNS: list[re.Pattern] = [
    re.compile(r'\d+\s*(jam|menit|hari|minggu|bulan|detik)\s*(lalu|yang lalu|yg lalu)', re.I),
    re.compile(r'\btadi\s+(pagi|siang|sore|malam|subuh)\b', re.I),
    re.compile(r'\bkemarin\w*\s*(pagi|siang|sore|malam|subuh)?\b', re.I),
    re.compile(r'(kamu|kau|mei|lo)\s*(masih\s*)?(inget|ingat|remember|tau|tahu)\s*(gak|nggak|tidak|dong|kan)?', re.I),
    re.compile(r'(masih\s*)?(inget|ingat)\s*(gak|nggak|tidak|dong|kan|ga)?', re.I),
    re.compile(r'(yang|yg)\s+(aku|gue|saya|gw)\s+(bilang|cerita|bahas|omongin|diskusi|kasih tau)', re.I),
    re.compile(r'gimana\s+\w*(revisi|progress|update|kelanjutan|hasilnya)', re.I),
    re.compile(r'\b(revisi|progress|update|kelanjutan)\b.{0,30}\b(gak|dong|kan|apa|gimana)\b', re.I),
    re.compile(r'\btanggal\s+\d{1,2}\b', re.I),
    re.compile(r'\b\d{1,2}\s+(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)\b', re.I),
    re.compile(r'\b(minggu|bulan|tahun|pekan)\s+(lalu|kemarin|sebelumnya)\b', re.I),
    re.compile(r'\b(waktu|pas|saat)\s+\w+\s*(lalu|kemarin|dulu|tadi)?\b', re.I),
    re.compile(r'apa\s+yang\s+(terjadi|dibahas|kita|aku|gue|sudah|udah)', re.I),
]

def _should_search_episodic(text: str) -> bool:
    low = text.lower()

    # debug
    print(f"  [DEBUG EPIS] text='{low[:60]}'", flush=True)
    matched_kw = [t for t in EPISODIC_TRIGGERS if t in low]
    print(f"  [DEBUG EPIS] matched_kw={matched_kw}", flush=True)

    if any(trigger in low for trigger in EPISODIC_TRIGGERS):
        _dbg(f"Episodic trigger: keyword match")
        return True
    for pattern in _EPISODIC_PATTERNS:
        if pattern.search(text):
            _dbg(f"Episodic trigger: pattern match [{pattern.pattern[:40]}]")
            return True
    return False


# ══════════════════════════════════════════════════════════════════
# MEMORY HELPERS
# ══════════════════════════════════════════════════════════════════

def _extract_core_memory(memory_md: str, max_chars: int = 800) -> str:
    KEEP_SECTIONS = {"Identitas", "Project Aktif"}
    lines    = memory_md.splitlines()
    result   = []
    in_keep  = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") or stripped.startswith("> "):
            result.append(line)
            continue
        if stripped == "---":
            if in_keep:
                result.append(line)
            continue
        if stripped.startswith("## "):
            section_name = stripped[3:].strip()
            in_keep = section_name in KEEP_SECTIONS
            if in_keep:
                result.append(line)
            continue
        if in_keep:
            result.append(line)

    out = "\n".join(result).strip()
    return out[:max_chars]


def _read_md(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""

def _load_profile_agent(storage: Path, use_gpu: bool) -> str:
    if use_gpu:
        return _read_md(storage / "memory" / "PROFILE_AGENT.md")
    cpu_profile = storage / "memory" / "PROFILE_AGENT_CPU.md"
    if cpu_profile.exists():
        _log("Profile agent: CPU mode (PROFILE_AGENT_CPU.md)")
        return _read_md(cpu_profile)
    _log("Profile agent: CPU mode tapi PROFILE_AGENT_CPU.md tidak ada, pakai default")
    return _read_md(storage / "memory" / "PROFILE_AGENT.md")


def _apply_cpu_token_limit(llm_config: dict, use_gpu: bool):
    cfg = llm_config.setdefault("generate_cfg", {})
    if use_gpu:
        cfg.pop("max_tokens", None)
    else:
        cfg["max_tokens"] = 900
    _log(f"Token limit: {'GPU (unlimited)' if use_gpu else 'CPU (max 900)'}")

def _get_max_history(use_gpu: bool) -> int:
    return MAX_HISTORY_MESSAGES if use_gpu else MAX_HISTORY_MESSAGES_CPU


# ══════════════════════════════════════════════════════════════════
# BUILD SYSTEM MESSAGE
# ══════════════════════════════════════════════════════════════════

def build_system_message(
    profile_agent : str,
    memory_md     : str,
    last_summary  : str,
    recent_msgs   : list[dict],
    episodic_hits : list[dict],
    last_tool_used: str | None = None,
    cpu_mode      : bool       = False,
) -> str:
    sep = "=" * 50

    mem_max_chars     = 400  if cpu_mode else 800
    recent_n          = 2    if cpu_mode else 5
    recent_content_ch = 80   if cpu_mode else 200

    msg = profile_agent.strip() or "Kamu adalah MEI, AI personal assistant Rifki."
    msg = get_runtime_context() + "\n\n" + msg

    if memory_md.strip():
        msg += f"\n\n{sep}\nPROFIL USER:\n{sep}\n{memory_md.strip()[:mem_max_chars]}"
    if last_summary.strip():
        msg += f"\n\n{sep}\nRINGKASAN HARI INI/KEMARIN:\n{sep}\n{last_summary.strip()}"
    if recent_msgs:
        filtered_recent = [
            m for m in recent_msgs
            if not m.get("content", "").startswith("[CATATAN SISTEM:")
        ][:recent_n]
        if filtered_recent:
            msg += f"\n\n{sep}\nPERCAKAPAN TERAKHIR:\n{sep}\n"
            for m in filtered_recent:
                msg += f"[{m.get('role','?').upper()}] {m.get('content','')[:recent_content_ch]}\n"
    if episodic_hits:
        msg += f"\n\n{sep}\nMEMORI EPISODIK RELEVAN:\n{sep}\n"
        for hit in episodic_hits:
            relevance = round(1 - hit["distance"], 2)
            msg += (
                f"- [{hit.get('label', hit['memory_type'])}] {hit['content']} "
                f"(tanggal: {hit['date']}, relevance: {relevance})\n"
            )

    if last_tool_used:
        msg += (
            f"\n\n{sep}\nREMINDER TOOL:\n{sep}\n"
            f"Turn sebelumnya menggunakan tool '{last_tool_used}'.\n"
            f"ATURAN WAJIB untuk turn ini:\n"
            f"  1. Jika user membuat request yang membutuhkan data real-time "
            f"(jadwal, cuaca, timer, foto, pencarian web, dll.), "
            f"WAJIB panggil tool yang sesuai — JANGAN jawab dari memory atau history.\n"
            f"  2. Tool '{last_tool_used}' kemungkinan besar dibutuhkan lagi. "
            f"Panggil dari awal, bukan dari hasil sebelumnya.\n"
            f"  3. Jangan asumsikan data dari history conversation masih valid "
            f"untuk request baru.\n"
        )

    msg += f"\n{sep}"
    return msg


# ── v4.5.6: Lean system message untuk Layer 3 ────────────────────
def build_lean_system_message(profile_agent: str) -> str:
    sep  = "=" * 50
    core = profile_agent.strip()[:600] or "Kamu adalah MEI, AI personal assistant Rifki."
    return get_runtime_context() + "\n\n" + core + f"\n{sep}"


# ══════════════════════════════════════════════════════════════════
# FUNCTION RESULT COMPRESSOR
# ══════════════════════════════════════════════════════════════════

def _compress_function_result(msg: dict) -> dict:
    if msg.get("role") != "function":
        return msg

    tool_name = msg.get("name", "")
    raw       = msg.get("content", "")

    if not raw.strip().startswith("{"):
        _dbg(f"Compress [{tool_name}]: plain string, truncating to {MAX_FUNCTION_RESULT_CHARS} chars")
        return {**msg, "content": raw[:MAX_FUNCTION_RESULT_CHARS]}

    try:
        result = json.loads(raw)
        status = result.get("status", "unknown")

        if status != "success":
            summary = {
                "status" : status,
                "message": result.get("message", "")[:80],
            }
        elif tool_name == "create_event":
            event = result.get("event", {})
            summary = {
                "status"  : "success",
                "title"   : event.get("title", ""),
                "event_id": event.get("id", "")[:8],
                "datetime": event.get("datetime", ""),
            }
        elif tool_name == "delete_event":
            summary = {
                "status"  : "success",
                "event_id": result.get("event_id", "")[:8],
            }
        elif tool_name == "get_events":
            events = result.get("events", [])
            summary = {
                "status"     : "success",
                "event_count": len(events),
                "titles"     : [e.get("title", "") for e in events[:5]],
            }
        elif tool_name == "set_timer":
            summary = {
                "status"  : "success",
                "timer_id": result.get("timer_id", "")[:8],
                "label"   : result.get("label", ""),
                "duration": result.get("duration_seconds", ""),
            }
        elif tool_name == "cancel_timer":
            summary = {
                "status"  : "success",
                "timer_id": result.get("timer_id", "")[:8],
            }
        elif tool_name in ("list_timers", "list_notifications"):
            items = result.get("timers", result.get("notifications", []))
            summary = {
                "status": "success",
                "count" : len(items),
            }
        elif tool_name == "schedule_notification":
            summary = {
                "status"  : "success",
                "notif_id": result.get("notif_id", "")[:8],
                "message" : result.get("message", "")[:60],
            }
        elif tool_name == "cancel_notification":
            summary = {
                "status"  : "success",
                "notif_id": result.get("notif_id", "")[:8],
            }
        elif tool_name == "internet_search":
            results = result.get("results", [])
            summary = {
                "status"      : "success",
                "result_count": len(results),
                "first_title" : results[0].get("title", "")[:60] if results else "",
            }
        elif tool_name == "camera_capture":
            summary = {
                "status"  : "success",
                "filepath": str(result.get("filepath", ""))[-40:],
            }
        else:
            summary = {
                "status" : status,
                "message": result.get("message", raw[:80]),
            }

        compressed = json.dumps(summary, ensure_ascii=False)
        _dbg(f"Compressed [{tool_name}]: {len(raw)} → {len(compressed)} chars")
        return {**msg, "content": compressed}

    except Exception as exc:
        _warn(f"_compress_function_result failed for '{tool_name}': {exc}")
        return {**msg, "content": raw[:MAX_FUNCTION_RESULT_CHARS]}


# ══════════════════════════════════════════════════════════════════
# CONVERSATION HISTORY HELPERS
# ══════════════════════════════════════════════════════════════════

def _trim_history(history: list[dict], max_messages: int = MAX_HISTORY_MESSAGES) -> list[dict]:
    if len(history) <= max_messages:
        return history
    start_idx = max(0, len(history) - max_messages)
    for i in range(start_idx, len(history)):
        if history[i].get("role") == "user":
            return history[i:]
    for i in range(start_idx - 1, -1, -1):
        if history[i].get("role") == "user":
            return history[i:]
    return []


def _aggressive_trim_history(history: list[dict]) -> list[dict]:
    target  = max(4, MAX_HISTORY_MESSAGES // 3)
    start_idx = max(0, len(history) - target)
    
    result = []
    for i in range(start_idx, len(history)):
        if history[i].get("role") == "user":
            result = history[i:]
            break
    if not result:
        for i in range(start_idx - 1, -1, -1):
            if history[i].get("role") == "user":
                result = history[i:]
                break
                
    _warn(f"Aggressive trim (L1): {len(history)} → {len(result)} messages")
    return result


def _nuclear_trim_history(history: list[dict]) -> list[dict]:
    KEEP = 2
    start_idx = max(0, len(history) - KEEP)
    
    result = []
    for i in range(start_idx, len(history)):
        if history[i].get("role") == "user":
            result = history[i:]
            break
    if not result:
        for i in range(start_idx - 1, -1, -1):
            if history[i].get("role") == "user":
                result = history[i:]
                break
                
    if result:
        _warn(f"Nuclear trim (L2): {len(history)} → {len(result)} messages")
        return result
        
    _warn(f"Nuclear trim (L2): {len(history)} → 0 messages (no user role found)")
    return []


def _layer3_trim_history(history: list[dict]) -> list[dict]:
    _warn(f"Layer3 trim (L3): {len(history)} → 0 messages (full reset)")
    return []


# ══════════════════════════════════════════════════════════════════
# BUILD MESSAGES TO SEND
# ══════════════════════════════════════════════════════════════════

def _build_messages_to_send(
    conv_history : list[dict],
    user_input   : str,
    last_tool    : str | None = None,
) -> list[dict]:
    if last_tool:
        _dbg(f"_build_messages_to_send: last_tool={last_tool} (reminder di system msg)")
    return conv_history + [{"role": "user", "content": user_input}]


# ══════════════════════════════════════════════════════════════════
# STALE RESPONSE DETECTION
# ══════════════════════════════════════════════════════════════════

_TIME_PATTERN   = re.compile(r'jam\s*(\d{1,2})', re.IGNORECASE)
_EVENT_KEYWORDS = re.compile(
    r'\b(meeting|rapat|ujian|kuliah|jogging|futsal|gym|kelas|webinar|'
    r'presentasi|makan|belajar|acara|kumpul|miting|bank|kampus|film|'
    r'timer|notifikasi|nonton|istirahat)\b',
    re.IGNORECASE,
)

def _is_response_stale(user_input: str, response: str) -> bool:
    input_times  = set(_TIME_PATTERN.findall(user_input.lower()))
    resp_times   = set(_TIME_PATTERN.findall(response.lower()))
    input_events = set(m.lower() for m in _EVENT_KEYWORDS.findall(user_input))
    resp_events  = set(m.lower() for m in _EVENT_KEYWORDS.findall(response))

    if input_times and resp_times and not input_times.intersection(resp_times):
        _warn(f"Stale? time mismatch: input={input_times} resp={resp_times}")
        return True
    if input_events and resp_events and not input_events.intersection(resp_events):
        _warn(f"Stale? event mismatch: input={input_events} resp={resp_events}")
        return True
    return False


# ══════════════════════════════════════════════════════════════════
# OUTPUT HELPERS
# ══════════════════════════════════════════════════════════════════

def remove_emojis(text: str) -> str:
    if not OUTPUT_FILTERS['remove_emoji']:
        return text
    return re.sub(
        "["
        "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF\U0001FA00-\U0001FAFF"
        "]+", "", text, flags=re.UNICODE,
    )


def format_response(response: str) -> str:
    response = remove_emojis(response)
    if OUTPUT_FILTERS["simplify_markdown_in_chitchat"]:
        response = re.sub(r"\*\*([^*]+)\*\*", r"\1", response)
        response = re.sub(r"__([^_]+)__",     r"\1", response)
        response = re.sub(r"#+\s+", "", response)
    max_nl   = OUTPUT_FILTERS["max_newlines"]
    response = re.sub(f"\n{{{max_nl+1},}}", "\n" * max_nl, response)
    if OUTPUT_FILTERS["trim_whitespace"]:
        response = response.strip()
    return response


def extract_tool_info(messages: list) -> dict:
    info = {"tool_used": None, "filepath": None, "memory_updated": False}
    for msg in messages:
        if msg.get("role") != "function":
            continue
        tool_name = msg.get("name", "")
        info["tool_used"] = tool_name
        try:
            result = json.loads(msg.get("content", "{}"))
            if result.get("status") == "success":
                if tool_name in ("word_document_manager",):
                    for part in result.get("message", "").split():
                        if part.endswith((".docx", ".doc")):
                            info["filepath"] = part
                            break
                elif tool_name == "camera_capture":
                    info["filepath"] = result.get("filepath")
        except Exception:
            pass
    return info


def debug_messages(all_messages: list):
    if not DEBUG_TOOL:
        return
    roles = [m.get("role") for m in all_messages]
    _dbg(f"message roles: {roles}")
    for msg in all_messages:
        role = msg.get("role")
        if role == "function":
            _dbg(
                f"TOOL EXECUTED  name={msg.get('name')} | "
                f"result={str(msg.get('content', ''))[:120]}"
            )
        if role == "assistant":
            content = msg.get("content", "")
            fc      = msg.get("function_call") or msg.get("tool_calls")
            if fc:
                _dbg(f"FUNCTION_CALL field: {str(fc)[:200]}")
            if isinstance(content, str):
                _dbg(f"ASSISTANT RAW: {content[:200]}")
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        _dbg(f"ASSISTANT BLOCK type={item.get('type')} | {str(item)[:150]}")


# ══════════════════════════════════════════════════════════════════
# VOICE HELPERS
# ══════════════════════════════════════════════════════════════════

def _init_voice(device: str = "cuda") -> tuple:
    print(f"\nInitializing voice (device={device})...")
    stt = tts = rvc = None
    voice_ok = rvc_ok = False

    try:
        stt_config         = STTConfig()
        stt_config.DEVICE  = device
        stt_config.USE_GPU = (device == "cuda")
        stt = STTEngine(config=stt_config)
        print(f"  STT ready (device={device})")
    except Exception as e:
        print(f"  STT gagal: {e}")

    try:
        tts_path = (
            ROOT_DIR / "models"
            / "id_ID-news_tts-medium"
            / "id_ID-news_tts-medium.onnx"
        )
        if tts_path.exists():
            tts      = IndonesianTTS(str(tts_path), device=device)
            voice_ok = True
            print(f"  TTS ready ({device})")
        else:
            print("  TTS model tidak ditemukan")
    except Exception as e:
        print(f"  TTS gagal: {e}")

    try:
        rvc    = RVCConverter(RVC_CONFIG)
        rvc_ok = True
        print(f"  RVC ready ({RVC_CONFIG.model_name})")
    except ConnectionError:
        print("  RVC WebUI tidak aktif")
    except Exception as e:
        print(f"  RVC gagal: {e}")

    return stt, tts, rvc, voice_ok, rvc_ok


def _play_wav_bytes(wav_bytes: bytes, intr: "InterruptState | None" = None):
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sr     = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    sd.play(audio, samplerate=sr)
    if intr:
        while sd.get_stream().active:
            if intr.event.is_set():
                sd.stop()
                return
            time.sleep(0.05)
    else:
        sd.wait()


def _synthesize_and_play(
    text      : str,
    vs        : VoiceState,
    io_state  : IOState,
    audio_lock: threading.Lock,
    intr      : "InterruptState | None" = None,
    tracker   : "LatencyTracker | None" = None,
) -> float:
    text = text.strip()
    if not text or not vs.voice_ok or not vs.tts:
        return 0.0
    if intr and intr.event.is_set():
        return 0.0
    t0 = time.perf_counter()
    try:
        wav_bytes, _ = vs.tts.get_audio_bytes(text)
        tts_ms = (time.perf_counter() - t0) * 1000

        with audio_lock:
            if intr and intr.event.is_set():
                return 0.0
            t1 = time.perf_counter()
            if io_state.use_rvc and vs.rvc:
                vs.rvc.speak_bytes(wav_bytes)
            else:
                _play_wav_bytes(wav_bytes, intr)
            playback_ms = (time.perf_counter() - t1) * 1000

        elapsed_ms = (time.perf_counter() - t0) * 1000
        _dbg(f"TTS chunk: synthesis={tts_ms:.0f}ms playback={playback_ms:.0f}ms total={elapsed_ms:.0f}ms")
    except Exception as e:
        print(f"\n[TTS/RVC error] {e}", flush=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000

    if tracker is not None:
        tracker.mark_tts_chunk(text, elapsed_ms)
    return elapsed_ms


# ══════════════════════════════════════════════════════════════════
# PROACTIVE AUDIO WORKER
# ══════════════════════════════════════════════════════════════════

def _start_proactive_speaker(
    vs        : VoiceState,
    io_state  : IOState,
    audio_lock: threading.Lock,
) -> tuple[queue.Queue, threading.Thread]:
    q = queue.Queue()

    def _worker():
        while True:
            item = q.get()
            if item is _STOP_SENTINEL:
                q.task_done()
                break
            try:
                if vs.voice_ok and vs.tts and io_state.voice_out:
                    _synthesize_and_play(item, vs, io_state, audio_lock)
            except Exception as e:
                print(f"  [ProactiveSpeaker] error: {e}", flush=True)
            finally:
                q.task_done()

    t = threading.Thread(target=_worker, name="ProactiveSpeaker", daemon=True)
    t.start()
    return q, t


# ══════════════════════════════════════════════════════════════════
# CHUNK SPLITTER
# ══════════════════════════════════════════════════════════════════

def _split_chunks(buffer: str) -> tuple[list[str], str]:
    chunks    = []
    remaining = buffer
    while remaining:
        m = SENTENCE_END.search(remaining)
        if m:
            chunks.append(remaining[:m.end()])
            remaining = remaining[m.end():]
            continue
        if len(remaining) >= MIN_SOFT_CHARS:
            m = SOFT_BOUNDARY.search(remaining)
            if m:
                chunks.append(remaining[:m.end()])
                remaining = remaining[m.end():]
                continue
        break
    return chunks, remaining


# ══════════════════════════════════════════════════════════════════
# STREAMING RESPONSE + TTS PIPELINE
# v4.5.8: tambah parameter on_token untuk UI streaming
# ══════════════════════════════════════════════════════════════════

def _extract_text_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def _stream_and_speak(
    bot         : Assistant,
    user_input  : str,
    conv_history: list[dict],
    io_state    : IOState,
    vs          : VoiceState,
    audio_lock  : threading.Lock,
    last_tool   : str | None                  = None,
    intr        : "InterruptState | None"     = None,
    tracker     : "LatencyTracker | None"     = None,
    on_token    : Callable[[str], None] | None = None,  # ← v4.5.8: UI streaming callback
) -> tuple[str, list]:
    """
    Stream respons dari LLM, kirim delta ke:
      - terminal (print)
      - on_token(delta) jika tersedia → UI
      - TTS jika voice_out aktif
    """
    full_response = ""
    all_messages : list[dict] = []
    text_buffer  = ""
    speak_mode   = io_state.voice_out and vs.voice_ok and vs.tts

    print("MEI: ", end="", flush=True)

    messages_to_send = _build_messages_to_send(conv_history, user_input, last_tool)

    try:
        for responses in bot.run(messages=messages_to_send):
            if intr and intr.event.is_set():
                _dbg("Stream interrupted by user")
                all_messages = responses or all_messages
                break

            if not responses:
                continue

            if len(responses) > MAX_AGENT_MESSAGES:
                _warn(
                    f"Tool loop terdeteksi ({len(responses)} messages > "
                    f"MAX_AGENT_MESSAGES={MAX_AGENT_MESSAGES}), menghentikan."
                )
                all_messages = responses
                for msg in reversed(responses):
                    if msg.get("role") == "assistant":
                        candidate = _extract_text_content(msg.get("content"))
                        if candidate.strip():
                            full_response = candidate
                            break
                if not full_response.strip():
                    full_response = (
                        "Maaf, terjadi masalah saat mencari informasi. "
                        "Coba tanyakan dengan lebih spesifik."
                    )
                print(full_response, flush=True)
                # Kirim ke UI jika ada on_token
                if on_token is not None:
                    try:
                        on_token(full_response)
                    except Exception as e:
                        _dbg(f"on_token error (tool loop fallback): {e}")
                break

            all_messages = responses
            last = responses[-1]
            if last.get("role") not in ("assistant", None):
                continue

            text = _extract_text_content(last.get("content"))
            if not text or len(text) <= len(full_response):
                continue

            delta         = text[len(full_response):]
            full_response = text

            if tracker is not None:
                tracker.mark_llm_first_token()

            # Terminal output
            print(delta, end="", flush=True)

            # ── v4.5.8: kirim delta ke UI ──────────────────────────
            if on_token is not None:
                try:
                    on_token(delta)
                except Exception as e:
                    _dbg(f"on_token error: {e}")

            if not speak_mode:
                continue

            text_buffer += delta
            chunks, text_buffer = _split_chunks(text_buffer)
            for chunk in chunks:
                if intr and intr.event.is_set():
                    text_buffer = ""
                    break
                _synthesize_and_play(chunk, vs, io_state, audio_lock, intr, tracker)

    except Exception as e:
        err_str = str(e).lower()
        if "context" in err_str and (
            "exceed" in err_str or "length" in err_str or "too long" in err_str
        ):
            print()
            raise ContextExceededError(str(e))
        print(f"\n[stream error] {e}")

    print()

    if speak_mode and text_buffer.strip() and not (intr and intr.event.is_set()):
        _synthesize_and_play(text_buffer, vs, io_state, audio_lock, intr, tracker)

    return full_response, all_messages


# ══════════════════════════════════════════════════════════════════
# BANNER & PARSE MODE
# ══════════════════════════════════════════════════════════════════

def _print_banner(lt_mem, vs: VoiceState, io_state: IOState):
    s = lt_mem.stats()
    print("\n" + "=" * 60)
    print("MEI — Personal Assistant v4.5.9")
    print("=" * 60)
    print(f"  User        : {USER_ID}")
    print(f"  Short-term  : JSONL window {MEMORY_CONFIG['window_size']} turn")
    print(f"  Long-term   : MEMORY.md (manual) + PROFILE_AGENT.md + ChromaDB")
    print(f"  Episodic DB : {s['total_memories']} memories")
    print(f"  Voice       : {'STT+TTS ready (streaming)' if vs.voice_ok else 'disabled'}")
    if vs.rvc_ok:
        print(f"  RVC         : {RVC_CONFIG.model_name}")
    print(f"  I/O mode    : {io_state.describe()}")
    print(f"  Debug tool  : {'ON' if DEBUG_TOOL else 'OFF'}")
    print(f"  Desktop UI  : {'aktif (tkinter) + STREAMING' if _UI_AVAILABLE else 'tidak tersedia'}")
    print(f"  NotifPro    : kalender + timer terpadu (1 thread)")
    print(f"  Audio       : thread-safe lock + proactive queue")
    print(f"  History max : {MAX_HISTORY_MESSAGES} messages (compressed)")
    print(f"  Extract every: {MEMORY_CONFIG['extract_every_n_msgs']} turns + on-exit safety net")
    print(f"  Interrupt   : Enter (kosong=stop / teks+Enter=stop+input baru)")
    print(f"  Episodic    : keyword ({len(EPISODIC_TRIGGERS)}) + regex ({len(_EPISODIC_PATTERNS)}) patterns")
    print(f"  Turn logger : {SRC_DIR / 'debug' /'mei_turns.jsonl'}")
    print(f"  Chroma log  : {CHROMA_JSONL_PATH}")
    print(f"  Ctx overflow: L1 aggressive → L2 nuclear → L3 empty+lean → L4 hard error")
    print(f"  UI Streaming: ON (token-by-token via on_token callback)")
    print()
    print("Commands:")
    print("  exit / quit / keluar     — keluar")
    print("  memory                   — stats memori")
    print("  clear                    — hapus short-term")
    print("  proactive                — status proactive engine")
    print("  calendar                 — events hari ini + cek notif sekarang")
    print("  mode                     — tampilkan mode aktif")
    print("  mode 1/2/3/4             — ganti I/O mode")
    print("  gpu / cpu                — ganti hardware target")
    print("  debug on/off             — toggle debug tool logging")
    print("  test                     — jalankan 27 GT input otomatis (semua)")
    print("  test N                   — mulai dari GT input ke-N (1-27)")
    print("  test info                — tampilkan daftar GT dataset")
    print("  test chroma              — jalankan 10 query ChromaDB → mei_chroma.jsonl")
    print("=" * 60 + "\n")


def _parse_mode_cmd(cmd: str) -> Optional[IOMode]:
    table = {
        "mode 1"    : IOMode.TEXT_TEXT,
        "mode text" : IOMode.TEXT_TEXT,
        "mode tt"   : IOMode.TEXT_TEXT,
        "mode 2"    : IOMode.TEXT_TTS,
        "mode tts"  : IOMode.TEXT_TTS,
        "mode 3"    : IOMode.STT_TTS,
        "mode stt"  : IOMode.STT_TTS,
        "mode voice": IOMode.STT_TTS,
        "mode 4"    : IOMode.STT_TTS_RVC,
        "mode rvc"  : IOMode.STT_TTS_RVC,
        "mode full" : IOMode.STT_TTS_RVC,
        "voice"     : IOMode.STT_TTS,
        "voice tts" : IOMode.STT_TTS,
        "voice rvc" : IOMode.STT_TTS_RVC,
    }
    return table.get(cmd)


# ══════════════════════════════════════════════════════════════════
# EXIT HANDLER
# ══════════════════════════════════════════════════════════════════

def _handle_exit(
    jsonl_mem        : JSONLMemoryManager,
    lt_mem           : LongTermMemoryManager,
    msg_count        : int,
    all_session_msgs : list[dict],
    llm_config       : dict,
    proactive_engine,
    fact_extractor   : FactExtractor,
    proactive_q      : queue.Queue,
    turn_logger      : Optional[TurnLogger] = None,
):
    proactive_q.put(_STOP_SENTINEL)

    if msg_count > 0 and MEMORY_CONFIG.get("daily_summary_on_exit", True):
        print("\nMembuat daily summary...")
        summary = _generate_daily_summary_llm(all_session_msgs, llm_config)
        if summary:
            lt_mem.append_daily_and_maybe_compress(
                summary, tag="summary", llm_cfg=LLM_CONFIG
            )
            print(f"  Daily summary tersimpan ({len(summary)} chars)")
            _dbg(f"Summary preview: {summary[:200]}")

    fact_extractor.submit_remaining(all_session_msgs)

    if turn_logger is not None:
        try:
            turn_logger.log_session_end()
            _log(f"Turn log session_end tersimpan → {turn_logger.log_path}")
        except Exception as e:
            _warn(f"turn_logger.log_session_end() gagal: {e}")

    print("  Menunggu fact extraction selesai (maks 90 detik)...")
    try:
        fact_extractor._queue.join()
        print("  Fact extraction selesai.")
    except Exception as e:
        _warn(f"queue.join() error: {e}")

    proactive_engine.stop()
    fact_extractor.stop(timeout=90)


# ══════════════════════════════════════════════════════════════════
# PROCESS SINGLE TURN
# v4.5.8: tambah on_token parameter untuk UI streaming
# ══════════════════════════════════════════════════════════════════

def _process_turn(
    user_input      : str,
    shared          : dict,
    bot             : Assistant,
    io_state        : IOState,
    vs              : VoiceState,
    audio_lock      : threading.Lock,
    jsonl_mem       : JSONLMemoryManager,
    lt_mem          : LongTermMemoryManager,
    fact_extractor  : FactExtractor,
    profile_agent   : str,
    agent_lock      : threading.Lock,
    intr            : "InterruptState | None"      = None,
    tracker         : "LatencyTracker | None"      = None,
    on_token        : Callable[[str], None] | None = None,  # ← v4.5.8
) -> str:
    with agent_lock:
        conv_history            = shared["conversation_history"]
        all_session_msgs        = shared["all_session_msgs"]
        last_used_realtime_tool = shared["last_used_realtime_tool"]
        _last_turn_had_error    = shared["_last_turn_had_error"]

        token_logger.reset()
        if tracker is None:
            tracker = LatencyTracker()
            tracker.mark_turn_start()

        memory_md          = lt_mem.read_memory_md()
        memory_md_filtered = _extract_core_memory(memory_md)
        recent_msgs        = jsonl_mem.get_recent_messages(
            USER_ID, n=MEMORY_CONFIG["max_recent_messages"]
        )
        last_summary = lt_mem.read_last_daily_summary()

        episodic_hits    = []
        chroma_triggered = False
        tracker.mark_pre_step("memory_read")
        if _should_search_episodic(user_input):
            chroma_triggered = True
            _log(f"ChromaDB: trigger terdeteksi, mencari...")
            top_k    = 1 if not io_state.use_gpu else MEMORY_CONFIG["top_k"]
            raw_hits = lt_mem.search(
                query          = user_input,
                n_results      = top_k,
                # min_importance = 6,
            )
            episodic_hits = raw_hits[:MEMORY_CONFIG["max_longterm_memories"]]
            if episodic_hits:
                _log(f"ChromaDB: {len(episodic_hits)} hasil diinjek ke prompt:")
                for i, hit in enumerate(episodic_hits, 1):
                    label     = hit.get("label", hit.get("memory_type", "?"))
                    relevance = round(1 - hit["distance"], 2)
                    preview   = hit["content"][:80].replace("\n", " ")
                    _log(f"  [{i}] label={label} relevance={relevance} | {preview}")
            else:
                _log("ChromaDB: tidak ada hasil relevan, prompt tanpa episodik")
        else:
            _log("ChromaDB: tidak ada trigger, skip")
        tracker.mark_pre_step_done("memory_read")

        # ── v4.5.7: expose chroma result ke shared ────────────────
        shared['_last_chroma_triggered'] = chroma_triggered
        # TAMBAH INI
        print(f"  [DEBUG TURN] chroma_triggered={chroma_triggered} hits={len(episodic_hits)}", flush=True)

        shared['_last_chroma_hits']      = len(episodic_hits)
        shared['_last_hit_contents']     = [h['content'] for h in episodic_hits]
        shared['_last_hit_relevances']   = [round(1 - h['distance'], 4) for h in episodic_hits]

        tracker.mark_pre_step("build_system_msg")
        bot.system_message = build_system_message(
            profile_agent  = profile_agent,
            memory_md      = memory_md_filtered,
            last_summary   = last_summary,
            recent_msgs    = recent_msgs,
            episodic_hits  = episodic_hits,
            last_tool_used = last_used_realtime_tool,
            cpu_mode       = not io_state.use_gpu,
        )
        tracker.mark_pre_step_done("build_system_msg")
        tracker.mark_preprocess_done()

        # ── Helper: bungkus _stream_and_speak dengan on_token ─────
        def _call_stream(ch, tool=None):
            return _stream_and_speak(
                bot,
                user_input,
                ch,
                io_state,
                vs,
                audio_lock,
                last_tool = tool,
                intr      = intr,
                tracker   = tracker,
                on_token  = on_token,   # ← v4.5.8: forward ke stream
            )

        # ── LLM call dengan 4-layer context overflow handling ─────
        try:
            tracker.mark_llm_start()
            full_response, all_messages = _call_stream(
                conv_history, last_used_realtime_tool
            )
            shared["_last_turn_had_error"] = False

        except ContextExceededError as ce:
            # ── LAYER 1: aggressive trim (~⅓ history) ─────────────
            _warn(f"Context exceeded [L1]: {ce}")
            shared["conversation_history"] = _aggressive_trim_history(conv_history)
            conv_history                   = shared["conversation_history"]
            shared["_last_turn_had_error"] = True
            shared["last_used_realtime_tool"] = None

            try:
                tracker.mark_llm_start()
                full_response, all_messages = _call_stream(conv_history)
                shared["_last_turn_had_error"] = False

            except ContextExceededError as ce2:
                # ── LAYER 2: nuclear trim (keep 2 msg) ────────────
                _warn(f"Context exceeded [L2]: {ce2}")
                shared["conversation_history"] = _nuclear_trim_history(conv_history)
                conv_history                   = shared["conversation_history"]
                shared["last_used_realtime_tool"] = None

                try:
                    tracker.mark_llm_start()
                    full_response, all_messages = _call_stream(conv_history)
                    shared["_last_turn_had_error"] = False

                except ContextExceededError as ce3:
                    # ── LAYER 3: empty history + lean system msg ───
                    _warn(f"Context exceeded [L3]: {ce3}")
                    shared["conversation_history"] = _layer3_trim_history(conv_history)
                    conv_history                   = shared["conversation_history"]
                    shared["last_used_realtime_tool"] = None

                    bot.system_message = build_lean_system_message(profile_agent)
                    _log("L3: system message diganti ke lean mode (persona inti saja)")

                    try:
                        tracker.mark_llm_start()
                        full_response, all_messages = _call_stream(conv_history)
                        shared["_last_turn_had_error"] = False

                    except ContextExceededError:
                        # ── LAYER 4: hard fallback ─────────────────
                        _warn("Context exceeded [L4]: user_input kemungkinan terlalu panjang")
                        full_response = (
                            "Maaf, pesanmu terlalu panjang untuk diproses. "
                            "Coba sampaikan dengan lebih singkat ya."
                        )
                        all_messages = []
                        print(f"\nMEI: {full_response}", flush=True)
                        # Kirim fallback ke UI
                        if on_token is not None:
                            try:
                                on_token(full_response)
                            except Exception:
                                pass

        if intr and intr.by_user:
            _dbg("Turn di-interrupt — skip history update")
            return ""

        jsonl_mem.add_message(USER_ID, "user", user_input)

        if not full_response.strip():
            full_response = "(MEI tidak memberikan respons. Coba lagi.)"
            print(full_response)

        debug_messages(all_messages)

        tracker.mark_llm_done()
        tracker.set_full_response(full_response)
        tool_info      = extract_tool_info(all_messages)
        clean_response = format_response(full_response)
        tool_used      = tool_info.get("tool_used")
        tracker.mark_tool_used(tool_used)
        record = tracker.finalize()

        if io_state.voice_in and io_state.voice_out:
            _mode_lbl = "STT -> TTS"
        elif io_state.voice_out:
            _mode_lbl = "Text -> TTS"
        else:
            _mode_lbl = "Text -> Text"
        log_latency_report(record, _mode_lbl)
        token_logger.advance_turn()

        _log(f"Tool: {tool_used or 'tidak ada'}")

        if _is_response_stale(user_input, clean_response):
            if shared["_last_turn_had_error"]:
                _warn("Stale response after error — replacing with fallback")
                clean_response = (
                    "Sepertinya jawaban tadi tidak nyambung. "
                    "Coba ulangi permintaanmu ya."
                )
            else:
                _warn("Possible stale response (normal turn, not replacing)")

        if tool_info.get("tool_used") in REALTIME_TOOLS:
            shared["last_used_realtime_tool"] = tool_info["tool_used"]
        else:
            shared["last_used_realtime_tool"] = None

        if io_state.use_gpu:
            new_history_entries = [
                _compress_function_result(m)
                for m in all_messages
                if m.get("role") != "user"
            ]
            conv_history.append({"role": "user", "content": user_input})
            conv_history.extend(new_history_entries)
        else:
            conv_history.append({"role": "user", "content": user_input})
            for m in reversed(all_messages):
                if m.get("role") == "assistant":
                    text = _extract_text_content(m.get("content", ""))
                    if text.strip():
                        conv_history.append({
                            "role"   : "assistant",
                            "content": text[:100],
                        })
                        break

        shared["conversation_history"] = _trim_history(
            conv_history, _get_max_history(io_state.use_gpu)
        )

        _dbg(
            f"History updated: {len(shared['conversation_history'])} messages "
            f"(tool_used={tool_info.get('tool_used')})"
        )

        shared["all_session_msgs"].append({"role": "user",      "content": user_input})
        shared["all_session_msgs"].append({"role": "assistant", "content": clean_response})

        jsonl_mem.add_message(USER_ID, "assistant", clean_response)
        shared["msg_count"] += 1

        # ── v4.5.5: Log turn ke JSONL ─────────────────────────────
        if shared.get("turn_logger") is not None:
            try:
                _turn_no_log = shared["msg_count"] * 2 - 1
                _gt_info     = gt_by_turn.get(_turn_no_log, {})

                fact_extractor._current_turn_no = _turn_no_log

                shared["turn_logger"].log_turn(
                    record             = record,
                    turn_no            = _turn_no_log,
                    user_input         = user_input,
                    assistant_response = clean_response,
                    io_mode            = _mode_lbl,
                    use_gpu            = io_state.use_gpu,
                    label_gt           = _gt_info.get("label_gt", ""),
                    chroma_triggered   = chroma_triggered,
                    chroma_hits        = len(episodic_hits),
                )
            except Exception as e:
                _warn(f"turn_logger.log_turn() gagal: {e}")

        n_ext = MEMORY_CONFIG["extract_every_n_msgs"]
        if shared["msg_count"] % n_ext == 0:
            fact_extractor.submit_periodic(
                shared["all_session_msgs"], n_turns=shared["msg_count"]
            )

        return clean_response


# ══════════════════════════════════════════════════════════════════
# CHROMA TEST RUNNER — v4.5.7
# ══════════════════════════════════════════════════════════════════

def _run_chroma_test(
    shared           : dict,
    bot_ref          : list,
    io_state         : IOState,
    vs               : VoiceState,
    audio_lock       : threading.Lock,
    jsonl_mem        : JSONLMemoryManager,
    lt_mem           : LongTermMemoryManager,
    fact_extractor   : FactExtractor,
    profile_agent_ref: list,
    agent_lock       : threading.Lock,
    stop_event       : threading.Event,
    app              = None,
):
    total = len(CHROMA_TEST_QUERIES)
    print(f"\n{'='*60}")
    print(f"  CHROMA TEST — {total} query episodic recall")
    print(f"  Output → {CHROMA_JSONL_PATH}")
    print(f"{'='*60}\n")

    results = []

    for idx, (qid, query, expected_keywords) in enumerate(CHROMA_TEST_QUERIES):
        if stop_event.is_set():
            break

        print(f"\n{'─'*60}")
        print(f"  [{qid}] {query}")
        print(f"  Expected kw : {expected_keywords}")
        print(f"{'─'*60}")

        _saved_history    = shared['conversation_history'][:]
        _saved_tool       = shared['last_used_realtime_tool']
        _saved_error      = shared['_last_turn_had_error']
        shared['conversation_history']    = []
        shared['last_used_realtime_tool'] = None
        shared['_last_turn_had_error']    = False

        tracker = LatencyTracker()
        tracker.mark_turn_start()

        intr = InterruptState()

        response = _process_turn(
            user_input     = query,
            shared         = shared,
            bot            = bot_ref[0],
            io_state       = io_state,
            vs             = vs,
            audio_lock     = audio_lock,
            jsonl_mem      = jsonl_mem,
            lt_mem         = lt_mem,
            fact_extractor = fact_extractor,
            profile_agent  = profile_agent_ref[0],
            agent_lock     = agent_lock,
            intr           = intr,
            tracker        = tracker,
            on_token       = None,  # chroma test: tidak perlu UI streaming
        )

        record = {
            'qid'              : qid,
            'query'            : query,
            'expected_keywords': expected_keywords,
            'agent_response'   : response,
            'chroma_triggered' : shared.get('_last_chroma_triggered', False),
            'chroma_hits'      : shared.get('_last_chroma_hits', 0),
            'hit_contents'     : shared.get('_last_hit_contents', []),
            'hit_relevances'   : shared.get('_last_hit_relevances', []),
            'timestamp'        : datetime.now().isoformat(timespec='seconds'),
        }
        results.append(record)

        kw_found = [kw for kw in expected_keywords if kw.lower() in response.lower()]
        print(f"  chroma_triggered : {record['chroma_triggered']}")
        print(f"  chroma_hits      : {record['chroma_hits']}")
        print(f"  kw_found         : {kw_found} / {expected_keywords}")
        print(f"  response         : {response[:120]}")

        shared['conversation_history']    = _saved_history
        shared['last_used_realtime_tool'] = _saved_tool
        shared['_last_turn_had_error']    = _saved_error

        if app is not None and response:
            app.add_mei_message(f"[ChromaTest {qid}] {response[:80]}")

        if idx < total - 1 and not stop_event.is_set():
            time.sleep(1.0)

    with open(CHROMA_JSONL_PATH, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    n_triggered = sum(1 for r in results if r['chroma_triggered'])
    n_has_hits  = sum(1 for r in results if r['chroma_hits'] > 0)

    print(f"\n{'='*60}")
    print(f"  ✓ Chroma test selesai: {len(results)} query")
    print(f"  ✓ Triggered   : {n_triggered}/{total}")
    print(f"  ✓ Got hits    : {n_has_hits}/{total}")
    print(f"  ✓ Disimpan → {CHROMA_JSONL_PATH}")
    print(f"  → Buka notebook Section 10b untuk evaluasi Judge LLM")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════
# UI STT + AGENT CALLBACK FACTORY
# v4.5.8: _ui_agent_cb sekarang mendukung on_token streaming
# ══════════════════════════════════════════════════════════════════

def _make_ui_stt_and_callback(
    app,
    vs               : VoiceState,
    shared           : dict,
    bot_ref          : list,
    bot_gpu          : Assistant,
    bot_cpu          : Assistant,
    cpu_tools        : list,
    io_state         : IOState,
    audio_lock       : threading.Lock,
    jsonl_mem        : JSONLMemoryManager,
    lt_mem           : LongTermMemoryManager,
    fact_extractor   : FactExtractor,
    agent_lock       : threading.Lock,
    profile_agent_ref: list,
    stop_event       : threading.Event,
):
    _stt_last_asr_done: list[float] = [0.0]
    _stt_turn_start: list[float]    = [0.0]
    _stt_processing = threading.Event()
    _stt_ui_stop_event = threading.Event()

    # ── STT loop ──────────────────────────────────────────────────
    def _stt_loop_fn(on_text):
        _stt_ui_stop_event.clear()
        _tts_last_done: list[float] = [0.0]
        TTS_COOLDOWN_MS = 2000

        def _is_safe_to_record() -> bool:
            if app._busy:
                return False
            if audio_lock.locked():
                _tts_last_done[0] = time.perf_counter() * 1000.0
                return False
            if _tts_last_done[0] > 0.0:
                since_ms = time.perf_counter() * 1000.0 - _tts_last_done[0]
                if since_ms < TTS_COOLDOWN_MS:
                    return False
                _tts_last_done[0] = 0.0
            return True

        def _loop():
            while not _stt_ui_stop_event.is_set():
                if not _is_safe_to_record():
                    time.sleep(0.05)
                    continue
                if not (vs.stt and vs.voice_ok):
                    time.sleep(0.5)
                    continue
                try:
                    result = vs.stt.process_voice_input()
                    if not result:
                        continue
                    if not result.get("text"):
                        continue
                    if not result.get("should_respond", True):
                        continue
                    if _stt_ui_stop_event.is_set() or app._busy:
                        continue
                    if audio_lock.locked():
                        _tts_last_done[0] = time.perf_counter() * 1000.0
                        _dbg("STT result dibuang: TTS mulai lagi saat rekam")
                        continue
                    _stt_turn_start[0]    = result.get("t_silence_start_ms", 0.0)
                    _stt_last_asr_done[0] = result.get("t_asr_done_ms", 0.0)
                    if _stt_turn_start[0] > 0.0 and _stt_last_asr_done[0] > 0.0:
                        _dbg(
                            f"ASR latency: "
                            f"{_stt_last_asr_done[0] - _stt_turn_start[0]:.1f} ms"
                            f" | text='{result['text'][:40]}'"
                        )
                    on_text(result["text"])
                except Exception as e:
                    _warn(f"UISTTLoop error: {e}")
                    time.sleep(0.5)
        _loop()

    def _stt_stop_fn():
        _stt_ui_stop_event.set()

    # ── Helper: switch hardware dari UI ───────────────────────────
    def _ui_switch_gpu() -> str:
        if io_state.use_gpu:
            return "Sudah pakai GPU."
        io_state.use_gpu = True
        fact_extractor._classifier = None
        fact_extractor._device     = io_state.device
        set_device(io_state.device)
        _apply_cpu_token_limit(LLM_CONFIG, True)
        bot_ref[0] = bot_gpu
        print("  [UI] Hardware: GPU (CUDA) — reinit voice + embedding...", flush=True)
        vs.stt, vs.tts, vs.rvc, vs.voice_ok, vs.rvc_ok = _init_voice(device=io_state.device)
        lt_mem.reinit_embedding(device=io_state.device)
        profile_agent_ref[0] = _load_profile_agent(STORAGE, True)
        _log("Bot: GPU (all tools)")
        if io_state.voice_out and not vs.voice_ok:
            io_state.mode = IOMode.TEXT_TEXT
            return "GPU aktif. TTS gagal, fallback ke mode 1."
        return f"GPU aktif. Mode: {io_state.describe()}"

    def _ui_switch_cpu() -> str:
        if not io_state.use_gpu:
            return "Sudah pakai CPU."
        io_state.use_gpu = False
        fact_extractor._classifier = None
        fact_extractor._device     = io_state.device
        set_device(io_state.device)
        _apply_cpu_token_limit(LLM_CONFIG, False)
        bot_ref[0] = bot_cpu
        shared["conversation_history"] = _trim_history(
            shared["conversation_history"], MAX_HISTORY_MESSAGES_CPU
        )
        print("  [UI] Hardware: CPU — reinit voice + embedding...", flush=True)
        vs.stt, vs.tts, vs.rvc, vs.voice_ok, vs.rvc_ok = _init_voice(device=io_state.device)
        lt_mem.reinit_embedding(device=io_state.device)
        profile_agent_ref[0] = _load_profile_agent(STORAGE, False)
        _log(f"Bot: CPU ({len(cpu_tools)} tools)")
        if io_state.voice_out and not vs.voice_ok:
            io_state.mode = IOMode.TEXT_TEXT
            return f"CPU aktif ({len(cpu_tools)} tools). TTS gagal, fallback ke mode 1."
        return f"CPU aktif ({len(cpu_tools)} tools). Mode: {io_state.describe()}"

    # ── Agent callback — v4.5.8: tambah on_token parameter ───────
    def _ui_agent_cb(
        user_text : str,
        mode      : str,
        on_token  : Callable[[str], None] | None = None,  # ← v4.5.8
    ) -> str:
        """
        Dipanggil oleh MEIApp saat user mengirim pesan.

        Parameter:
            user_text : teks dari user
            mode      : "1"/"2"/"3"/"4" — I/O mode string
            on_token  : callback (delta: str) → None, dipanggil per token streaming.
                        Jika None, tidak ada streaming ke UI (kompatibel v4.5.7).

        Returns:
            str : full response (setelah streaming selesai)
        """
        mode_map = {
            "1": IOMode.TEXT_TEXT,
            "2": IOMode.TEXT_TTS,
            "3": IOMode.STT_TTS,
            "4": IOMode.STT_TTS_RVC,
        }
        io_state.mode = mode_map.get(mode, IOMode.TEXT_TEXT)

        print(f"\n[UI] {USER_ID}: {user_text}", flush=True)
        cmd = user_text.lower().strip()

        if cmd in ("gpu", "full gpu", "cuda"):
            return _ui_switch_gpu()

        if cmd in ("cpu", "full cpu"):
            return _ui_switch_cpu()

        if cmd == "mode":
            return f"I/O mode aktif: {io_state.describe()}"

        new_mode = _parse_mode_cmd(cmd)
        if new_mode is not None:
            needs_voice = new_mode in (IOMode.TEXT_TTS, IOMode.STT_TTS, IOMode.STT_TTS_RVC)
            needs_rvc   = new_mode == IOMode.STT_TTS_RVC
            if needs_voice and not vs.voice_ok:
                return "Voice tidak tersedia. Mode tidak diubah."
            if needs_rvc and not vs.rvc_ok:
                return "RVC tidak tersedia. Mode tidak diubah."
            io_state.mode = new_mode
            return f"I/O mode: {io_state.describe()}"

        if cmd == "clear":
            jsonl_mem.clear(USER_ID)
            shared["conversation_history"].clear()
            shared["all_session_msgs"].clear()
            shared["msg_count"]               = 0
            shared["last_used_realtime_tool"] = None
            shared["_last_turn_had_error"]    = False
            shared["_pending_input"]          = ""
            fact_extractor._last_submitted_idx = 0
            if vs.stt:
                vs.stt.clear_conversation_history()
            return "Short-term memory dihapus."

        if cmd == "test info":
            lines = ["GT Dataset (27 input):"]
            for i, (tn, inp, lbl, imp, tool) in enumerate(GT_DATASET, 1):
                lines.append(f"  [{i:>2}] Turn {tn:>2} [{lbl:13}] imp={imp} | {inp[:50]}")
            return "\n".join(lines)

        # ── test chroma via UI ────────────────────────────────────
        if cmd == "test chroma":
            if shared.get("_test_running"):
                return "Test sedang berjalan. Tunggu selesai atau restart."

            def _chroma_thread():
                shared["_test_running"] = True
                try:
                    _run_chroma_test(
                        shared            = shared,
                        bot_ref           = bot_ref,
                        io_state          = io_state,
                        vs                = vs,
                        audio_lock        = audio_lock,
                        jsonl_mem         = jsonl_mem,
                        lt_mem            = lt_mem,
                        fact_extractor    = fact_extractor,
                        profile_agent_ref = profile_agent_ref,
                        agent_lock        = agent_lock,
                        stop_event        = stop_event,
                        app               = app,
                    )
                except Exception as e:
                    _warn(f"Chroma test error: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    shared["_test_running"] = False

            t = threading.Thread(target=_chroma_thread, name="ChromaTest", daemon=True)
            t.start()
            return (
                f"Chroma test dimulai — {len(CHROMA_TEST_QUERIES)} query. "
                f"Progress di terminal. Hasil → mei_chroma.jsonl"
            )

        if cmd == "test" or (cmd.startswith("test ") and cmd not in ("test info", "test chroma")):
            if shared.get("_test_running"):
                return "Test sedang berjalan. Tunggu selesai atau restart."
            parts   = cmd.split()
            start_n = 0
            if len(parts) > 1:
                try:
                    start_n = max(0, int(parts[1]) - 1)
                except ValueError:
                    return "Penggunaan: test [N]  (N = nomor input 1-27)"

            def _test_thread():
                shared["_test_running"] = True
                try:
                    _run_test_mode(
                        start_idx        = start_n,
                        shared           = shared,
                        bot_ref          = bot_ref,
                        io_state         = io_state,
                        vs               = vs,
                        audio_lock       = audio_lock,
                        jsonl_mem        = jsonl_mem,
                        lt_mem           = lt_mem,
                        fact_extractor   = fact_extractor,
                        profile_agent_ref= profile_agent_ref,
                        agent_lock       = agent_lock,
                        stop_event       = stop_event,
                        app              = app,
                    )
                except Exception as e:
                    _warn(f"Test mode error: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    shared["_test_running"] = False
                    print("\n  [TEST] Thread selesai.", flush=True)

            t = threading.Thread(target=_test_thread, name="TestMode", daemon=True)
            t.start()
            remaining = len(GT_DATASET) - start_n
            return (
                f"Test mode dimulai — {remaining} input "
                f"(mulai dari #{start_n + 1}). "
                f"Progress di terminal."
            )

        if cmd in ("memory", "mem"):
            s  = jsonl_mem.stats(USER_ID)
            lt = lt_mem.stats()
            return (
                f"Short-term: {s['total_turns']} turns | "
                f"Episodic: {lt['total_memories']} memories | "
                f"Session msgs: {len(shared['all_session_msgs'])} | "
                f"History: {len(shared['conversation_history'])}/{MAX_HISTORY_MESSAGES}"
            )

        # ── Normal LLM turn (dengan UI streaming) ─────────────────
        tracker = LatencyTracker()
        turn_ts = _stt_turn_start[0]
        asr_ts  = _stt_last_asr_done[0]
        _stt_turn_start[0]    = 0.0
        _stt_last_asr_done[0] = 0.0

        if turn_ts > 0.0 and asr_ts > 0.0:
            tracker._rec.t_turn_start = turn_ts
            tracker._rec.t_stt_done   = asr_ts
            tracker._rec.had_stt      = True
            _dbg(f"ASR latency inject: {asr_ts - turn_ts:.1f} ms")
        else:
            tracker.mark_turn_start()

        app.reset_interrupt()
        intr       = InterruptState()
        intr.event = app.get_interrupt_event()

        try:
            response = _process_turn(
                user_input     = user_text,
                shared         = shared,
                bot            = bot_ref[0],
                io_state       = io_state,
                vs             = vs,
                audio_lock     = audio_lock,
                jsonl_mem      = jsonl_mem,
                lt_mem         = lt_mem,
                fact_extractor = fact_extractor,
                profile_agent  = profile_agent_ref[0],
                agent_lock     = agent_lock,
                intr           = intr,
                tracker        = tracker,
                on_token       = on_token,   # ← v4.5.8: forward ke _process_turn
            )
            return response or "(tidak ada respons)"
        finally:
            _dbg("_stt_processing cleared — STT loop resumed")

    return _stt_loop_fn, _stt_stop_fn, _ui_agent_cb


# ══════════════════════════════════════════════════════════════════
# TEST MODE RUNNER
# ══════════════════════════════════════════════════════════════════

def _run_test_mode(
    start_idx      : int,
    shared         : dict,
    bot_ref        : list,
    io_state       : IOState,
    vs             : VoiceState,
    audio_lock     : threading.Lock,
    jsonl_mem      : JSONLMemoryManager,
    lt_mem         : LongTermMemoryManager,
    fact_extractor : FactExtractor,
    profile_agent_ref: list,
    agent_lock     : threading.Lock,
    stop_event     : threading.Event,
    app            = None,
):
    total = len(GT_DATASET)
    print(f"\n{'='*60}")
    print(f"  TEST MODE  — {total - start_idx} input ({start_idx+1}–{total})")
    print(f"  Interval antar input: {TEST_INPUT_INTERVAL_S}s")
    print(f"  Interrupt: Enter untuk hentikan")
    print(f"{'='*60}\n")

    for idx in range(start_idx, total):
        if stop_event.is_set():
            print("\n  [TEST] Stop event — test dihentikan.", flush=True)
            break

        turn_no, text, label, imp, tool_gt = GT_DATASET[idx]
        pos = idx + 1

        print(
            f"\n{'─'*60}\n"
            f"  [TEST {pos:>2}/{total}]  Turn {turn_no}  "
            f"[{label}] imp={imp}  tool_gt={tool_gt or '—'}\n"
            f"  Input: {text}\n"
            f"{'─'*60}",
            flush=True,
        )

        intr         = InterruptState()
        _intr_thread = _start_interrupt_listener(intr)

        tracker = LatencyTracker()
        tracker.mark_turn_start()

        clean_response = _process_turn(
            user_input     = text,
            shared         = shared,
            bot            = bot_ref[0],
            io_state       = io_state,
            vs             = vs,
            audio_lock     = audio_lock,
            jsonl_mem      = jsonl_mem,
            lt_mem         = lt_mem,
            fact_extractor = fact_extractor,
            profile_agent  = profile_agent_ref[0],
            agent_lock     = agent_lock,
            intr           = intr,
            tracker        = tracker,
            on_token       = None,  # test mode: tidak perlu UI streaming
        )

        intr.event.set()

        if app is not None and clean_response:
            app.add_mei_message(clean_response)

        if intr.by_user:
            pending = intr.pending_input
            print(f"\n  [TEST] Interrupt oleh user — test dihentikan.", flush=True)
            if pending:
                shared["_pending_input"] = pending
            break

        if idx < total - 1 and not stop_event.is_set():
            print(
                f"  [TEST] {pos}/{total} selesai — jeda {TEST_INPUT_INTERVAL_S}s...",
                flush=True,
            )
            time.sleep(TEST_INPUT_INTERVAL_S)

    print(f"\n{'='*60}\n  TEST MODE selesai.\n{'='*60}\n", flush=True)


# ══════════════════════════════════════════════════════════════════
# DAILY SUMMARY → CHROMADB (v4.5.9)
# ══════════════════════════════════════════════════════════════════

def _store_yesterday_summary_to_chroma(lt_mem: LongTermMemoryManager):
    """
    Dipanggil saat startup. Cek daily summary kemarin, kalau ada dan
    belum tersimpan di ChromaDB → langsung insert tanpa filter/SLM.

    Daily summary sudah condensed oleh LLM (<400 char) sehingga tidak
    perlu filter tambahan — langsung aman dimasukkan ke ChromaDB.

    Marker file {date}.chroma_stored mencegah double insert saat
    MEI direstart berkali-kali di hari yang sama.
    """
    yesterday = date.today() - timedelta(days=1)

    # Cek marker — sudah pernah diinsert sebelumnya?
    marker_path = lt_mem.daily_dir / f"{yesterday.isoformat()}.chroma_stored"
    if marker_path.exists():
        _log(f"Daily summary {yesterday} sudah ada di ChromaDB, skip.")
        return

    summary_text = lt_mem.read_last_daily_summary()
    if not summary_text.strip():
        _log(f"Tidak ada daily summary untuk {yesterday}, skip.")
        return
    
    clean_summary = re.sub(
        r'^-\s+\d{2}:\d{2}\s+\[summary\]\s*', '', 
        summary_text.strip(), 
        flags=re.MULTILINE
    ).strip()

    _log(f"Menyimpan daily summary {yesterday} ke ChromaDB...")
    try:
        lt_mem.add_memory(
            content     = f"[daily_summary {yesterday.isoformat()}] {clean_summary}",
            memory_type = "episodic",
            importance  = 7,
            metadata    = {
                "label"  : "daily_summary",
                "date"   : yesterday.isoformat(),
                "source" : "daily_auto",
            },
        )
        # Tandai sudah diproses
        marker_path.touch()
        _log(f"Daily summary {yesterday} berhasil masuk ChromaDB ({len(summary_text)} chars).")
    except Exception as e:
        _warn(f"Gagal menyimpan daily summary ke ChromaDB: {e}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def run_agent():
    global DEBUG_TOOL

    io_state   = IOState(mode=IOMode.TEXT_TEXT, use_gpu=True)
    vs         = VoiceState()
    audio_lock = threading.Lock()
    _apply_cpu_token_limit(LLM_CONFIG, io_state.use_gpu)

    # ── Memory ────────────────────────────────────────────────────
    jsonl_mem = JSONLMemoryManager(
        sessions_dir = MEMORY_CONFIG["sessions_dir"],
        window_size  = MEMORY_CONFIG["window_size"],
    )
    lt_mem = LongTermMemoryManager(
        storage_dir     = MEMORY_CONFIG["storage_dir"],
        embedding_model = MEMORY_CONFIG["embedding_model"],
        collection_name = MEMORY_CONFIG["collection_name"],
        top_k           = MEMORY_CONFIG["top_k"],
        device          = io_state.device,
    )

    print("  Warming up embedding model...", flush=True)
    try:
        _ = lt_mem._ef._model()
        stats = lt_mem.stats()
        print(f"  Embedding ready | ChromaDB: {stats['total_memories']} memories", flush=True)
    except Exception as e:
        print(f"  Embedding warmup error: {e}", flush=True)

    # ── v4.5.9: simpan daily summary kemarin ke ChromaDB ─────────
    _store_yesterday_summary_to_chroma(lt_mem)

    _startup_summary = lt_mem.read_last_daily_summary()

    # ── v4.5.5: Turn Logger ───────────────────────────────────────
    turn_logger = TurnLogger(SRC_DIR / "debug" / "mei_turns.jsonl")
    _log(f"Turn logger: {turn_logger.log_path}  session={turn_logger.session_id}")

    # ── v4.5.5: on_fact_saved callback ───────────────────────────
    def _on_fact_saved_cb(turn_no: int, label: str, saved: bool):
        turn_logger.on_fact_result(turn_no, label, saved)

    fact_extractor = FactExtractor(
        lt_mem           = lt_mem,
        llm_config       = LLM_CONFIG,
        classifier_model = CLASSIFIER_CONFIG["model"],
        user_id          = USER_ID,
        device           = io_state.device,
        debug            = DEBUG_TOOL,
        on_fact_saved    = _on_fact_saved_cb,
    )
    fact_extractor.start()

    _startup_summary = lt_mem.read_last_daily_summary()
    if _startup_summary:
        print(f"\n  [Ringkasan Terakhir]\n{_startup_summary}\n")

    # ── Voice ─────────────────────────────────────────────────────
    vs.stt, vs.tts, vs.rvc, vs.voice_ok, vs.rvc_ok = _init_voice(
        device=io_state.device
    )

    # ── Proactive audio queue ─────────────────────────────────────
    proactive_q, _proactive_thread = _start_proactive_speaker(
        vs, io_state, audio_lock
    )

    # ── Camera & Search ───────────────────────────────────────────
    camera_capture = CameraCaptureTool(
        capture_dir = STORAGE / "captures",
        llm_cfg     = LLM_CONFIG,
    )
    internet_search = InternetSearch()

    # ── NotifProactive ────────────────────────────────────────────
    notif_pro = NotifProactive(
        calendar   = calendar_instance,
        on_trigger = lambda msg: None,
        llm_config = LLM_CONFIG,
        config     = {
            "check_interval_sec"        : 60,
            "notify_before_min"         : 30,
            "notify_before_min_2"       : 10,
            "notify_at_start_window_min": 2,
            "calendar_prefix"           : "📅 ",
            "timer_prefix"              : "⏱️ ",
        },
        debug = DEBUG_TOOL,
    )
    set_timer_tool._on_done = notif_pro.on_timer_done

    # ── NotificationEngine ────────────────────────────────────────
    notif_engine = NotificationEngine(on_trigger=lambda msg: None, debug=DEBUG_TOOL)
    notif_engine.start()
    schedule_notification = ScheduleNotificationTool(engine=notif_engine)
    list_notifications    = ListNotificationsTool(engine=notif_engine)
    cancel_notification   = CancelNotificationTool(engine=notif_engine)

    all_tools = get_base_tools(camera_capture, internet_search) + [
        schedule_notification,
        list_notifications,
        cancel_notification,
    ]

    _CPU_TOOL_NAMES = {"camera_capture", "internet_search", "create_event", "get_events", "set_timer"}
    cpu_tools = [t for t in all_tools if getattr(t, "name", "") in _CPU_TOOL_NAMES]

    bot_gpu = Assistant(
        llm           = LLM_CONFIG,
        name          = "MEI",
        function_list = all_tools,
        system_message= "",
    )
    bot_cpu = Assistant(
        llm           = LLM_CONFIG,
        name          = "MEI",
        function_list = cpu_tools,
        system_message= "",
    )

    bot_ref = [bot_gpu if io_state.use_gpu else bot_cpu]
    _log(f"Bot init: {'GPU (all tools)' if io_state.use_gpu else f'CPU ({len(cpu_tools)} tools)'}")

    proactive_engine = ProactiveEngine(
        llm_config = LLM_CONFIG,
        on_trigger = lambda msg: None,
        user_name  = USER_ID,
    )

    profile_agent = _load_profile_agent(STORAGE, io_state.use_gpu)
    profile_agent_ref = [profile_agent]

    # ── Shared mutable state ──────────────────────────────────────
    shared = {
        "conversation_history"   : [],
        "all_session_msgs"       : [],
        "msg_count"              : 0,
        "last_used_realtime_tool": None,
        "_last_turn_had_error"   : False,
        "_pending_input"         : "",
        "turn_logger"            : turn_logger,
        "_test_running"          : False,
        # ── v4.5.7: chroma test state ─────────────────────────────
        "_last_chroma_triggered" : False,
        "_last_chroma_hits"      : 0,
        "_last_hit_contents"     : [],
        "_last_hit_relevances"   : [],
    }

    agent_lock  = threading.Lock()
    _stop_event = threading.Event()

    # ══════════════════════════════════════════════════════════════
    # DESKTOP UI SETUP
    # ══════════════════════════════════════════════════════════════

    app: "MEIApp | None" = None

    if _UI_AVAILABLE:
        app = MEIApp(
            agent_callback = lambda t, m, cb=None: "(inisialisasi...)",
            stt_loop_fn    = None,
            stt_stop_fn    = None,
        )

        _stt_loop_fn, _stt_stop_fn, _ui_agent_cb = _make_ui_stt_and_callback(
            app              = app,
            vs               = vs,
            shared           = shared,
            bot_ref          = bot_ref,
            bot_gpu          = bot_gpu,
            bot_cpu          = bot_cpu,
            cpu_tools        = cpu_tools,
            io_state         = io_state,
            audio_lock       = audio_lock,
            jsonl_mem        = jsonl_mem,
            lt_mem           = lt_mem,
            fact_extractor   = fact_extractor,
            agent_lock       = agent_lock,
            profile_agent_ref= profile_agent_ref,
            stop_event       = _stop_event,
        )

        app._agent_cb    = _ui_agent_cb
        app._stt_loop_fn = _stt_loop_fn if (vs.voice_ok and vs.stt) else None
        app._stt_stop_fn = _stt_stop_fn

    # ══════════════════════════════════════════════════════════════
    # PROACTIVE TRIGGER
    # ══════════════════════════════════════════════════════════════

    def on_proactive_trigger(message: str):
        print(
            f"\n{PROACTIVE_CONFIG['display_prefix']}{message}"
            f"{PROACTIVE_CONFIG['display_suffix']}",
            flush=True,
        )
        print(f"{USER_ID}: ", end="", flush=True)

        if io_state.voice_out and vs.voice_ok and vs.tts:
            _dbg(f"Proactive enqueue TTS: {message[:50]}")
            proactive_q.put(message)

        if app is not None:
            app.push_notif(message, tag="info")
            app.add_mei_message(message)

    notif_pro._on_trigger        = on_proactive_trigger
    notif_engine._on_trigger     = on_proactive_trigger
    proactive_engine._on_trigger = on_proactive_trigger

    proactive_engine.start()
    notif_pro.start()

    _print_banner(lt_mem, vs, io_state)
    print("ProactiveEngine  : active")
    print("NotifProactive   : active (kalender + timer)")
    print("ProactiveSpeaker : active (audio queue thread)")
    if app is not None:
        print("Desktop UI       : active (tkinter — streaming ON)")
    print()

    ctx = jsonl_mem.get_context(USER_ID)
    if ctx.current_topic:
        print(f"Melanjutkan — topik terakhir: {ctx.current_topic}\n")

    _dbg(f"bot tools: {list(bot_ref[0].function_map.keys())}")
    _dbg(f"bot llm type: {type(bot_ref[0].llm).__name__}")

    # ══════════════════════════════════════════════════════════════
    # TERMINAL CONVERSATION LOOP
    # ══════════════════════════════════════════════════════════════

    def _terminal_loop():
        global DEBUG_TOOL

        while not _stop_event.is_set():
            try:
                user_input    = None
                _turn_tracker = None

                if shared["_pending_input"]:
                    user_input                 = shared["_pending_input"]
                    shared["_pending_input"]   = ""
                    print(f"{USER_ID}: {user_input}")
                    _turn_tracker = LatencyTracker()
                    _turn_tracker.mark_turn_start()
                elif io_state.voice_in and vs.voice_ok and vs.stt:
                    print("\nVoice mode — Enter untuk rekam / ketik langsung")
                    typed = input().strip()
                    if typed:
                        user_input    = typed
                        _turn_tracker = LatencyTracker()
                        _turn_tracker.mark_turn_start()
                    else:
                        _turn_tracker = LatencyTracker()
                        result = vs.stt.process_voice_input()
                        if result:
                            user_input = result["text"]
                            t_start = result.get("t_silence_start_ms", 0.0)
                            t_done  = result.get("t_asr_done_ms", 0.0)
                            if t_start > 0.0 and t_done > 0.0:
                                _turn_tracker._rec.t_turn_start = t_start
                                _turn_tracker._rec.t_stt_done   = t_done
                                _turn_tracker._rec.had_stt      = True
                            else:
                                _turn_tracker.mark_turn_start()
                            if not result.get("should_respond", True):
                                continue
                        else:
                            continue
                else:
                    user_input = input(f"{USER_ID}: ").strip()
                    if not user_input:
                        continue
                    _turn_tracker = LatencyTracker()
                    _turn_tracker.mark_turn_start()

                if not user_input:
                    continue

                cmd = user_input.lower().strip()

                # ── EXIT ──────────────────────────────────────────
                if cmd in ("exit", "quit", "keluar"):
                    notif_pro.stop()
                    _handle_exit(
                        jsonl_mem, lt_mem, shared["msg_count"],
                        shared["all_session_msgs"], LLM_CONFIG,
                        proactive_engine, fact_extractor, proactive_q,
                        turn_logger = turn_logger,
                    )
                    print("\nSampai jumpa!")
                    _stop_event.set()
                    if app is not None:
                        try:
                            app.root.after(0, app.root.destroy)
                        except Exception:
                            pass
                    break

                # ── MEMORY ────────────────────────────────────────
                if cmd == "memory":
                    s  = jsonl_mem.stats(USER_ID)
                    c  = jsonl_mem.get_context(USER_ID)
                    lt = lt_mem.stats()
                    print(f"\nMemory Stats:")
                    print(f"  Short-term  : {s['total_turns']} turns, window {s['window_turns']}/{s['window_size']}")
                    print(f"  Episodic DB : {lt['total_memories']} memories (ChromaDB)")
                    print(f"  MEMORY.md   : {lt_mem.memory_md_path} (manual)")
                    print(f"  Daily dir   : {lt_mem.daily_dir}")
                    if c.current_topic:
                        print(f"  Last topic  : {c.current_topic}")
                    print(f"  Session msgs: {len(shared['all_session_msgs'])}")
                    print(f"  History msgs: {len(shared['conversation_history'])} / {MAX_HISTORY_MESSAGES}")
                    print(f"  Extracted until idx: {fact_extractor._last_submitted_idx}")
                    print(f"  Turn log    : {turn_logger.log_path}")
                    print()
                    continue

                # ── CLEAR ─────────────────────────────────────────
                if cmd == "clear":
                    jsonl_mem.clear(USER_ID)
                    shared["conversation_history"].clear()
                    shared["all_session_msgs"].clear()
                    shared["msg_count"]               = 0
                    shared["last_used_realtime_tool"] = None
                    shared["_last_turn_had_error"]    = False
                    shared["_pending_input"]          = ""
                    fact_extractor._last_submitted_idx = 0
                    if vs.stt:
                        vs.stt.clear_conversation_history()
                    print("Short-term memory dihapus.\n")
                    continue

                # ── PROACTIVE ─────────────────────────────────────
                if cmd == "proactive":
                    state = proactive_engine.get_state()
                    print(f"\nProactiveEngine : {state.name}")
                    if proactive_engine.is_afk():
                        print("  (AFK mode aktif)")
                    print()
                    continue

                # ── CALENDAR ──────────────────────────────────────
                if cmd == "calendar":
                    today_str = date.today().isoformat()
                    result    = calendar_instance.get_events(today_str)
                    events    = result.get("events", [])
                    print(f"\nEvents hari ini ({today_str}):")
                    if events:
                        for ev in events:
                            t_label = ev.get("datetime", "")
                            t_label = t_label[11:16] if len(t_label) >= 16 else t_label
                            print(f"  [{t_label}] {ev['title']}  (id: {ev['id'][:8]}...)")
                    else:
                        print("  Tidak ada event.")
                    notif_pro.check_calendar_now()
                    print()
                    continue

                # ── MODE ──────────────────────────────────────────
                if cmd == "mode":
                    print(f"\nI/O mode aktif: {io_state.describe()}")
                    print("Ganti dengan: mode 1 / mode 2 / mode 3 / mode 4\n")
                    continue

                # ── DEBUG ─────────────────────────────────────────
                if cmd == "debug on":
                    DEBUG_TOOL                = True
                    fact_extractor._debug     = True
                    notif_pro._debug          = True
                    print("  Debug tool logging: ON\n")
                    continue

                if cmd == "debug off":
                    DEBUG_TOOL                = False
                    fact_extractor._debug     = False
                    notif_pro._debug          = False
                    print("  Debug tool logging: OFF\n")
                    continue

                # ── MODE SWITCH ───────────────────────────────────
                new_mode = _parse_mode_cmd(cmd)
                if new_mode is not None:
                    needs_voice = new_mode in (IOMode.TEXT_TTS, IOMode.STT_TTS, IOMode.STT_TTS_RVC)
                    needs_rvc   = new_mode == IOMode.STT_TTS_RVC
                    if needs_voice and not vs.voice_ok:
                        print("  Voice tidak tersedia. Mode tidak diubah.\n")
                        continue
                    if needs_rvc and not vs.rvc_ok:
                        print("  RVC tidak tersedia. Mode tidak diubah.\n")
                        continue
                    io_state.mode = new_mode
                    print(f"\nI/O mode: {io_state.describe()}\n")
                    continue

                # ── GPU ───────────────────────────────────────────
                if cmd in ("gpu", "full gpu", "cuda"):
                    if io_state.use_gpu:
                        print("  Sudah pakai GPU.\n")
                        continue
                    io_state.use_gpu = True
                    fact_extractor._classifier = None
                    fact_extractor._device     = io_state.device
                    set_device(io_state.device)
                    _apply_cpu_token_limit(LLM_CONFIG, True)
                    bot_ref[0] = bot_gpu
                    print("  Hardware: GPU (CUDA) — reinit voice + embedding...\n")
                    vs.stt, vs.tts, vs.rvc, vs.voice_ok, vs.rvc_ok = _init_voice(device=io_state.device)
                    lt_mem.reinit_embedding(device=io_state.device)
                    profile_agent = _load_profile_agent(STORAGE, True)
                    profile_agent_ref[0] = profile_agent
                    _log("Bot: GPU (all tools)")
                    if io_state.voice_out and not vs.voice_ok:
                        io_state.mode = IOMode.TEXT_TEXT
                        print("  TTS gagal, fallback ke mode 1.\n")
                    continue

                # ── CPU ───────────────────────────────────────────
                if cmd in ("cpu", "full cpu"):
                    if not io_state.use_gpu:
                        print("  Sudah pakai CPU.\n")
                        continue
                    io_state.use_gpu = False
                    fact_extractor._classifier = None
                    fact_extractor._device     = io_state.device
                    set_device(io_state.device)
                    _apply_cpu_token_limit(LLM_CONFIG, False)
                    bot_ref[0] = bot_cpu
                    shared["conversation_history"] = _trim_history(
                        shared["conversation_history"], MAX_HISTORY_MESSAGES_CPU
                    )
                    print("  Hardware: CPU — reinit voice + embedding...\n")
                    vs.stt, vs.tts, vs.rvc, vs.voice_ok, vs.rvc_ok = _init_voice(device=io_state.device)
                    lt_mem.reinit_embedding(device=io_state.device)
                    profile_agent = _load_profile_agent(STORAGE, False)
                    profile_agent_ref[0] = profile_agent
                    _log(f"Bot: CPU ({len(cpu_tools)} tools)")
                    if io_state.voice_out and not vs.voice_ok:
                        io_state.mode = IOMode.TEXT_TEXT
                        print("  TTS gagal, fallback ke mode 1.\n")
                    continue

                # ── EPISODIC ──────────────────────────────────────
                if cmd.startswith("episodic"):
                    parts = cmd.split(maxsplit=1)
                    query = parts[1] if len(parts) > 1 else "skripsi"
                    bulan_map = {
                        "januari":1,  "februari":2, "maret":3,    "april":4,
                        "mei":5,      "juni":6,     "juli":7,     "agustus":8,
                        "september":9,"oktober":10, "november":11,"desember":12,
                    }
                    import datetime as _dt
                    year  = _dt.date.today().year
                    month = _dt.date.today().month
                    for nama, nomor in bulan_map.items():
                        if nama in query:
                            month = nomor
                            query = query.replace(nama, "").strip()
                            break
                    hits = lt_mem.search_episodic_by_period(query, year, month)
                    if hits:
                        print(f"\nEpisodic memories ({year}-{month:02d}) — '{query}':")
                        for h in hits:
                            label = h.get("label", h.get("memory_type", "?"))
                            print(f"  [{h['date']}] label={label} | {h['content'][:120]}")
                    else:
                        print(f"  Tidak ada episodik untuk '{query}' di {year}-{month:02d}")
                    print()
                    continue

                # ── TEST INFO ─────────────────────────────────────
                if cmd == "test info":
                    print(f"\nGT Dataset — {len(GT_DATASET)} input:")
                    for i, (tn, inp, lbl, imp, tool) in enumerate(GT_DATASET, 1):
                        print(f"  [{i:>2}] Turn {tn:>2} [{lbl:13}] imp={imp} "
                              f"tool={tool or '—':23} | {inp[:50]}")
                    print()
                    continue

                # ── TEST CHROMA ───────────────────────────────────
                if cmd == "test chroma":
                    if shared["_test_running"]:
                        print("  Test sudah berjalan.\n")
                        continue
                    shared["_test_running"] = True
                    _run_chroma_test(
                        shared            = shared,
                        bot_ref           = bot_ref,
                        io_state          = io_state,
                        vs                = vs,
                        audio_lock        = audio_lock,
                        jsonl_mem         = jsonl_mem,
                        lt_mem            = lt_mem,
                        fact_extractor    = fact_extractor,
                        profile_agent_ref = profile_agent_ref,
                        agent_lock        = agent_lock,
                        stop_event        = _stop_event,
                        app               = app,
                    )
                    shared["_test_running"] = False
                    continue

                # ── TEST MODE ─────────────────────────────────────
                if cmd == "test" or (cmd.startswith("test ") and cmd != "test info"):
                    if shared["_test_running"]:
                        print("  Test sudah berjalan.\n")
                        continue
                    parts    = cmd.split()
                    start_n  = 0
                    if len(parts) > 1:
                        try:
                            start_n = max(0, int(parts[1]) - 1)
                        except ValueError:
                            print("  Penggunaan: test [N]  (N = 1-27)\n")
                            continue
                    shared["_test_running"] = True
                    _run_test_mode(
                        start_idx        = start_n,
                        shared           = shared,
                        bot_ref          = bot_ref,
                        io_state         = io_state,
                        vs               = vs,
                        audio_lock       = audio_lock,
                        jsonl_mem        = jsonl_mem,
                        lt_mem           = lt_mem,
                        fact_extractor   = fact_extractor,
                        profile_agent_ref= profile_agent_ref,
                        agent_lock       = agent_lock,
                        stop_event       = _stop_event,
                        app              = app,
                    )
                    shared["_test_running"] = False
                    continue

                # ── NORMAL TURN (terminal — tanpa UI on_token) ────
                proactive_engine.on_user_message(shared["conversation_history"])

                intr         = InterruptState()
                _intr_thread = _start_interrupt_listener(intr)

                clean_response = _process_turn(
                    user_input     = user_input,
                    shared         = shared,
                    bot            = bot_ref[0],
                    io_state       = io_state,
                    vs             = vs,
                    audio_lock     = audio_lock,
                    jsonl_mem      = jsonl_mem,
                    lt_mem         = lt_mem,
                    fact_extractor = fact_extractor,
                    profile_agent  = profile_agent_ref[0],
                    agent_lock     = agent_lock,
                    intr           = intr,
                    tracker        = _turn_tracker,
                    on_token       = None,  # terminal: tidak perlu UI callback
                )

                intr.event.set()

                if intr.pending_input:
                    shared["_pending_input"] = intr.pending_input
                    _dbg(f"Pending input setelah interrupt: '{shared['_pending_input']}'")

                if app is not None and clean_response:
                    app.add_mei_message(clean_response)

            except KeyboardInterrupt:
                notif_pro.stop()
                _handle_exit(
                    jsonl_mem, lt_mem, shared["msg_count"],
                    shared["all_session_msgs"], LLM_CONFIG,
                    proactive_engine, fact_extractor, proactive_q,
                    turn_logger = turn_logger,
                )
                print("\nBye!")
                _stop_event.set()
                if app is not None:
                    try:
                        app.root.after(0, app.root.destroy)
                    except Exception:
                        pass
                break

            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()

    # ══════════════════════════════════════════════════════════════
    # START TERMINAL THREAD + RUN UI / MAIN LOOP
    # ══════════════════════════════════════════════════════════════

    terminal_thread = threading.Thread(
        target  = _terminal_loop,
        name    = "TerminalLoop",
        daemon  = True,
    )
    terminal_thread.start()

    if app is not None:
        def _on_window_close():
            print("\n[UI] Window ditutup — menjalankan cleanup...", flush=True)
            _stop_event.set()
            notif_pro.stop()
            _handle_exit(
                jsonl_mem, lt_mem, shared["msg_count"],
                shared["all_session_msgs"], LLM_CONFIG,
                proactive_engine, fact_extractor, proactive_q,
                turn_logger = turn_logger,
            )
            print("Sampai jumpa!")
            app.root.destroy()

        app.root.protocol("WM_DELETE_WINDOW", _on_window_close)
        app.set_online(True, "LM Studio connected")
        app.run()
        _stop_event.set()

    else:
        try:
            terminal_thread.join()
        except KeyboardInterrupt:
            _stop_event.set()


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        run_agent()
    except Exception as e:
        print(f"\nFatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)