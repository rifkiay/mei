"""
debug_latency.py — Latency & Token Tracker for MEI Pipeline
=============================================================
Mengukur latensi per-komponen untuk pipeline:
  • Text → Text   : TTFT, TPS, total generation, e2e
  • Text → Voice  : + TTFC, per-chunk TTS latency, e2e-to-audio
  • Voice → Voice : + ASR latency

Token counting via QwenTokenLogger:
  • input  = "ALL tokens: N" dari log qwen-agent (kumulatif konteks)
  • output = selisih ALL tokens antar turn (exact), fallback ~1 tok/3 char

Tool tracking:
  • tool_used  : nama tool yang dipanggil (None = tidak ada)
  • Ditampilkan di section TOOLS pada laporan latency

FIX v2:
  • log_latency_report: output dikumpulkan dulu ke buffer, baru satu
    sys.stdout.write() atomik → tidak ada interleaving dari thread lain.
  • ASR section: ditampilkan saat mode_label mengandung "STT", bukan
    hanya saat rec.had_stt=True (yang hanya True kalau mark_stt_done
    dipanggil dari path rekam mic, bukan path ketik langsung).
  • Emoji/non-ASCII di chunk text: dihapus dari preview agar lebar
    kolom ASCII-art tidak meleset.

Usage (lihat main.py):
  from debug_latency import LatencyTracker, LatencyRecord, log_latency_report, token_logger

  token_logger.install()          # sekali di startup

  token_logger.reset()            # awal tiap turn
  tracker = LatencyTracker()
  tracker.mark_turn_start()
  ...
  tracker.mark_tool_used(tool_info.get("tool_used"))   # setelah extract_tool_info
  record = tracker.finalize()
  log_latency_report(record, "Text -> Text")
  token_logger.advance_turn()     # akhir tiap turn
"""

from __future__ import annotations

import io as _io
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════════
# QWEN TOKEN LOGGER
# ══════════════════════════════════════════════════════════════════

class QwenTokenLogger:
    """
    Menangkap token usage ASLI dari LM Studio via dua jalur:

    [Jalur 1 - UTAMA] stream_options usage chunk
        qwen-agent log berisi JSON chunk akhir stream:
            {"usage": {"prompt_tokens": N, "completion_tokens": M, ...}}
        Aktifkan dengan menambah di agent_config.py:
            LLM_CONFIG = {
                ...
                'generate_cfg': {
                    'stream_options': {'include_usage': True}
                },
            }

    [Jalur 2 - FALLBACK] LM Studio print_timing
        Baris seperti: "prompt eval time = 1809 ms / 3252 tokens"

    [Jalur 3 - LAST RESORT] estimasi karakter: len(response) // 3
    """

    _USAGE_BLOCK_RE = re.compile(r'"usage"\s*:\s*(\{[^{}]*\})', re.DOTALL)
    _F_PROMPT       = re.compile(r'"prompt_tokens"\s*:\s*(\d+)')
    _F_COMPLETION   = re.compile(r'"completion_tokens"\s*:\s*(\d+)')
    _PROMPT_EVAL_RE = re.compile(
        r'prompt\s+eval\s+time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens',
        re.IGNORECASE,
    )
    _EVAL_RE = re.compile(
        r'^\s*eval\s+time\s*=\s*[\d.]+\s*ms\s*/\s*(\d+)\s*tokens',
        re.IGNORECASE | re.MULTILINE,
    )

    def __init__(self):
        self._prompt_tokens:     Optional[int] = None
        self._completion_tokens: Optional[int] = None
        self._handler:           Optional[logging.Handler] = None
        self._installed:         bool = False
        self._prev_all_tokens:   Optional[int] = None

    def install(self):
        if self._installed:
            return

        _self = self

        class _UsageHandler(logging.Handler):
            def emit(self, record):
                msg = record.getMessage()

                ub = QwenTokenLogger._USAGE_BLOCK_RE.search(msg)
                if ub:
                    block = ub.group(1)
                    mp = QwenTokenLogger._F_PROMPT.search(block)
                    mc = QwenTokenLogger._F_COMPLETION.search(block)
                    if mp:
                        _self._prompt_tokens = int(mp.group(1))
                    if mc:
                        _self._completion_tokens = int(mc.group(1))
                    if mp or mc:
                        return

                mp = QwenTokenLogger._PROMPT_EVAL_RE.search(msg)
                if mp and _self._prompt_tokens is None:
                    _self._prompt_tokens = int(mp.group(1))
                me = QwenTokenLogger._EVAL_RE.search(msg)
                if me and _self._completion_tokens is None:
                    _self._completion_tokens = int(me.group(1))

        handler = _UsageHandler()
        handler.setLevel(logging.DEBUG)

        _TARGET_LOGGERS = ('qwen_agent', 'qwen_agent_logger', 'base', 'oai')
        for name in _TARGET_LOGGERS:
            lg = logging.getLogger(name)
            lg.handlers.clear()
            lg.addHandler(handler)
            lg.setLevel(logging.DEBUG)
            lg.propagate = False

        for lg_name, lg_obj in logging.root.manager.loggerDict.items():
            if not isinstance(lg_obj, logging.Logger):
                continue
            if any(lg_name.startswith(t + '.') for t in _TARGET_LOGGERS):
                lg_obj.handlers.clear()
                lg_obj.propagate = True

        self._handler   = handler
        self._installed = True

    def reset(self):
        self._prompt_tokens     = None
        self._completion_tokens = None

    def advance_turn(self):
        self._prev_all_tokens = self._prompt_tokens
        self.reset()

    def get_input_tokens(self) -> Optional[int]:
        return self._prompt_tokens

    def get_output_tokens(self, full_response: str = "") -> tuple[int, bool]:
        if self._completion_tokens is not None:
            return self._completion_tokens, False
        if full_response:
            return max(1, len(full_response) // 3), True
        return 0, True


token_logger = QwenTokenLogger()


# ══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class TtsChunkInfo:
    text:         str    # preview teks chunk (maks 45 char, ASCII only)
    synthesis_ms: float
    seq:          int


@dataclass
class LatencyRecord:
    t_turn_start:      float           = 0.0
    t_stt_done:        Optional[float] = None
    t_preprocess_done: float           = 0.0
    t_llm_start:       float           = 0.0
    t_llm_first_token: Optional[float] = None
    t_llm_done:        float           = 0.0
    t_first_tts_done:  Optional[float] = None

    pre_steps:  dict[str, float] = field(default_factory=dict)
    tts_chunks: list[TtsChunkInfo] = field(default_factory=list)

    input_tokens:            int  = 0
    output_tokens:           int  = 0
    output_tokens_estimated: bool = False

    tool_used: Optional[str] = None

    had_stt: bool = False
    had_tts: bool = False

    @property
    def stt_latency_ms(self) -> Optional[float]:
        if self.had_stt and self.t_stt_done:
            return self.t_stt_done - self.t_turn_start
        return None

    @property
    def preprocess_ms(self) -> float:
        base = self.t_stt_done if (self.had_stt and self.t_stt_done) else self.t_turn_start
        return max(0.0, self.t_preprocess_done - base)

    @property
    def ttft_ms(self) -> Optional[float]:
        if self.t_llm_first_token:
            return self.t_llm_first_token - self.t_llm_start
        return None

    @property
    def total_llm_ms(self) -> float:
        return max(0.0, self.t_llm_done - self.t_llm_start)

    @property
    def ttfc_ms(self) -> Optional[float]:
        if self.t_llm_first_token and self.t_first_tts_done:
            return self.t_first_tts_done - self.t_llm_first_token
        return None

    @property
    def tps(self) -> Optional[float]:
        sec = self.total_llm_ms / 1000.0
        if sec > 0 and self.output_tokens > 0:
            return self.output_tokens / sec
        return None

    @property
    def e2e_ms(self) -> float:
        if self.had_tts and self.t_first_tts_done:
            first_synth = self.tts_chunks[0].synthesis_ms if self.tts_chunks else 0.0
            return self.t_first_tts_done - first_synth - self.t_turn_start
        return max(0.0, self.t_llm_done - self.t_turn_start)

    @property
    def total_tts_ms(self) -> float:
        return sum(c.synthesis_ms for c in self.tts_chunks)


# ══════════════════════════════════════════════════════════════════
# LATENCY TRACKER
# ══════════════════════════════════════════════════════════════════

# Regex untuk strip emoji & non-ASCII dari preview chunk
_NON_ASCII_RE = re.compile(r'[^\x00-\x7F]')


class LatencyTracker:
    def __init__(self):
        self._rec             = LatencyRecord()
        self._pre_step_start: dict[str, float] = {}
        self._tts_seq         = 0
        self._full_response   = ""

    @staticmethod
    def _now_ms() -> float:
        return time.perf_counter() * 1000.0

    def mark_turn_start(self):
        self._rec.t_turn_start = self._now_ms()

    def mark_stt_done(self):
        """
        Panggil setelah STT menghasilkan teks (path rekam mic).
        had_stt=True → seksi ASR ditampilkan dengan nilai terukur.
        Kalau user ketik langsung di STT mode, JANGAN panggil ini —
        seksi ASR tetap muncul di laporan tapi nilainya '--'.
        """
        self._rec.t_stt_done = self._now_ms()
        self._rec.had_stt    = True

    def mark_pre_step(self, name: str):
        self._pre_step_start[name] = self._now_ms()

    def mark_pre_step_done(self, name: str):
        if name in self._pre_step_start:
            self._rec.pre_steps[name] = self._now_ms() - self._pre_step_start.pop(name)

    def mark_preprocess_done(self):
        self._rec.t_preprocess_done = self._now_ms()

    def mark_llm_start(self):
        self._rec.t_llm_start = self._now_ms()

    def mark_llm_first_token(self):
        if self._rec.t_llm_first_token is None:
            self._rec.t_llm_first_token = self._now_ms()

    def mark_llm_done(self):
        self._rec.t_llm_done = self._now_ms()

    def mark_tts_chunk(self, chunk_text: str, synthesis_ms: float):
        """
        Catat satu chunk TTS.
        Emoji dan karakter non-ASCII di-strip dari preview agar
        lebar kolom ASCII-art tidak meleset.
        """
        self._tts_seq += 1
        now = self._now_ms()

        # Hapus non-ASCII (emoji, dsb.) sebelum dijadikan preview
        clean = _NON_ASCII_RE.sub("", chunk_text).strip()
        if not clean and chunk_text.strip():
            # Teks ada tapi semua non-ASCII — beri keterangan jumlah karakter
            clean = f"[{len(chunk_text.strip())} non-ASCII chars]"
        preview = (clean[:45] + "...") if len(clean) > 45 else clean

        self._rec.tts_chunks.append(TtsChunkInfo(
            text         = preview,
            synthesis_ms = synthesis_ms,
            seq          = self._tts_seq,
        ))

        if self._rec.t_first_tts_done is None:
            self._rec.t_first_tts_done = now

        self._rec.had_tts = True

    def mark_tool_used(self, tool_name: Optional[str]):
        self._rec.tool_used = tool_name

    def set_full_response(self, text: str):
        self._full_response = text

    def finalize(self) -> LatencyRecord:
        self._rec.input_tokens = token_logger.get_input_tokens() or 0
        out_tok, estimated = token_logger.get_output_tokens(self._full_response)
        self._rec.output_tokens           = out_tok
        self._rec.output_tokens_estimated = estimated
        return self._rec


# ══════════════════════════════════════════════════════════════════
# REPORT PRINTER
# ══════════════════════════════════════════════════════════════════

_TOOL_LABELS: dict[str, str] = {
    "camera_capture"        : "camera_capture       (kamera + analisis LLM)",
    "internet_search"       : "internet_search      (pencarian web)",
    "get_events"            : "get_events           (ambil event kalender)",
    "create_event"          : "create_event         (buat event kalender)",
    "delete_event"          : "delete_event         (hapus event kalender)",
    "set_timer"             : "set_timer            (buat countdown timer)",
    "list_timers"           : "list_timers          (daftar timer aktif)",
    "cancel_timer"          : "cancel_timer         (batalkan timer)",
    "schedule_notification" : "schedule_notification(jadwalkan notifikasi)",
    "list_notifications"    : "list_notifications   (daftar notifikasi)",
    "cancel_notification"   : "cancel_notification  (batalkan notifikasi)",
}


def log_latency_report(rec: LatencyRecord, mode_label: str = ""):
    """
    Cetak laporan latency ke stdout secara ATOMIK:
    seluruh output dikumpulkan ke buffer dulu, baru satu sys.stdout.write()
    → tidak ada interleaving dari thread lain (proactive engine, dll.).

    Seksi ASR ditampilkan saat mode_label mengandung "STT"
    (bukan hanya saat rec.had_stt=True), agar selalu terlihat.
    Kalau user ketik langsung (tidak rekam), nilai latency jadi '--'.
    """

    INNER = 58
    W     = INNER + 6

    buf = _io.StringIO()

    def _out(line: str = ""):
        buf.write(line + "\n")

    def _top():
        _out("  +" + "-" * (W - 2) + "+")

    def _mid():
        _out("  |" + "-" * (W - 2) + "|")

    def _bot():
        _out("  +" + "-" * (W - 2) + "+")

    def _row(text: str = "", right: str = ""):
        # Normalisasi unicode ke ASCII agar len() akurat di box
        text  = _NON_ASCII_RE.sub("?", text)
        right = _NON_ASCII_RE.sub("?", right)

        if right:
            gap  = INNER - len(text) - len(right)
            line = text + (" " * max(1, gap)) + right
        else:
            line = text

        if len(line) > INNER:
            line = line[:INNER - 3] + "..."

        _out(f"  |  {line:<{INNER}}  |")

    def _fmt_ms(val: Optional[float], unit: str = "ms") -> str:
        if val is None:
            return "    --"
        if val < 1000:
            return f"{val:>8.1f} {unit}"
        else:
            return f"{val/1000:>8.2f} s "

    def _fmt_int(val: int) -> str:
        return f"{val:>6,}"

    # ── Header ────────────────────────────────────────────────────
    _top()
    title = f"LATENCY REPORT  |  {mode_label}" if mode_label else "LATENCY REPORT"
    _row(title)

    # ── TOOLS ────────────────────────────────────────────────────
    _mid()
    _row("TOOLS")
    if rec.tool_used:
        label = _TOOL_LABELS.get(rec.tool_used, rec.tool_used)
        _row("  Digunakan", f"  {label}")
    else:
        _row("  Digunakan", "  tidak ada  (plain chat)")

    # ── ASR ───────────────────────────────────────────────────────
    # Tampilkan seksi ASR kalau mode mengandung "STT", terukur atau tidak.
    # had_stt=True + t_stt_done set → nilai terukur
    # had_stt=False (user ketik di STT mode) → nilai '--'
    is_stt_mode = "STT" in mode_label
    if is_stt_mode or rec.had_stt:
        _mid()
        _row("ASR  (audio selesai -> teks keluar)")
        if rec.stt_latency_ms is not None:
            _row("  Latency", _fmt_ms(rec.stt_latency_ms))
        else:
            # User ketik langsung di STT mode — tidak ada rekaman
            _row("  Latency", "    --   (ketik langsung)")

    # ── Pre-processing ────────────────────────────────────────────
    _mid()
    _row("PRE-PROCESSING", f"[{rec.preprocess_ms:,.1f} ms]")
    if rec.pre_steps:
        max_name = max(len(k) for k in rec.pre_steps)
        for name, dur in rec.pre_steps.items():
            _row(f"  {name:<{max_name}}", _fmt_ms(dur))
    else:
        _row("  (no steps recorded)")

    # ── LLM Generation ───────────────────────────────────────────
    _mid()
    _row("LLM  GENERATION")
    _row("  TTFT   (start -> 1st token)",  _fmt_ms(rec.ttft_ms))
    _row("  Total  (start -> done)",       _fmt_ms(rec.total_llm_ms))
    if rec.tps is not None:
        _row("  Throughput", f"{rec.tps:>7.1f} tok/s")
    else:
        _row("  Throughput", "    -- tok/s")

    # ── Prompt processing ─────────────────────────────────────────
    if rec.ttft_ms is not None:
        _mid()
        _row("PROMPT  PROCESSING")
        _row("  Prefill  (prompt eval)", _fmt_ms(rec.ttft_ms))
        gen_only = rec.total_llm_ms - rec.ttft_ms
        _row("  Decode   (token gen)",  _fmt_ms(gen_only if gen_only > 0 else None))

    # ── TTS ───────────────────────────────────────────────────────
    if rec.had_tts:
        _mid()
        _row("TTS  PIPELINE")
        _row("  TTFC  (1st token -> 1st audio)", _fmt_ms(rec.ttfc_ms))
        for c in rec.tts_chunks:
            _row(f'  Chunk {c.seq}  "{c.text}"')
            _row("    synthesis", _fmt_ms(c.synthesis_ms))
        _row("  Total TTS synthesis", _fmt_ms(rec.total_tts_ms))

    # ── Tokens ───────────────────────────────────────────────────
    _mid()
    _row("TOKENS")
    inp_str = _fmt_int(rec.input_tokens) if rec.input_tokens else "    --"
    _row("  Input  (context size)", inp_str)
    out_label = (
        f"{_fmt_int(rec.output_tokens)}  (~est)"
        if rec.output_tokens_estimated
        else _fmt_int(rec.output_tokens)
    )
    _row("  Output", out_label)

    # ── End-to-end ────────────────────────────────────────────────
    _mid()
    e2e_note = "turn start -> 1st audio" if rec.had_tts else "turn start -> done"
    _row(f"END-TO-END  ({e2e_note})", _fmt_ms(rec.e2e_ms))

    _bot()
    _out()   # baris kosong setelah box

    # ── Satu write atomik — tidak ada thread lain yang bisa nyusup ──
    sys.stdout.write(buf.getvalue())
    sys.stdout.flush()