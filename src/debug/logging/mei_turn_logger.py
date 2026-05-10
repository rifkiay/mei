"""
mei_turn_logger.py — Auto-logger setiap turn MEI ke JSONL
==========================================================
Versi v4.5.6 — patch dari v4.5.5.

Perubahan v4.5.6:
  - ADD: field baru di log_turn():
      assistant_response — jawaban LLM (clean_response dari _process_turn)

Perubahan v4.5.5:
  - ADD: field baru di log_turn():
      label_gt        — ground truth label dari GT_DATASET
      chroma_triggered — apakah ChromaDB query dipanggil turn ini
      chroma_hits     — jumlah hit dari ChromaDB query
  - ADD: field yang diisi async oleh on_fact_result():
      fact_label_predicted — label yang diprediksi FactExtractor
      fact_saved           — apakah fakta akhirnya disimpan ke ChromaDB
  - ADD: on_fact_result(turn_no, label, saved) — dipanggil dari callback
    fact_extractor._on_fact_saved, update entry yang sudah ditulis
    dengan cara append "fact_patch" line (patch-style, tidak rewrite).
  - ADD: _pending_fact dict untuk buffer hasil async (keadaan di mana
    log_turn belum dipanggil saat callback sudah tiba, atau sebaliknya).

Format tiap baris "__type__": "turn":
{
  "__type__": "turn",
  "session_id": str,
  "timestamp": str,
  "turn_no": int,
  "user_input": str,
  "assistant_response": str,  # ← v4.5.6
  "io_mode": str,
  "use_gpu": bool,

  "label_gt": str,            # ← v4.5.5 (kosong "" jika turn bukan GT)
  "chroma_triggered": bool,   # ← v4.5.5
  "chroma_hits": int,         # ← v4.5.5

  "raw_timestamps": { ... },
  "metrics": { ... },
  "pre_steps": { ... },
  "tts_chunks": [ ... ],
  "tokens": { ... },
  "tool_used": str|null,
  "had_stt": bool,
  "had_tts": bool
}

Format baris "__type__": "fact_patch" (async, setelah fact_extractor selesai):
{
  "__type__": "fact_patch",
  "session_id": str,
  "timestamp": str,
  "turn_no": int,
  "fact_label_predicted": str,
  "fact_saved": bool
}
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from debug.latency.debug_latency import LatencyRecord


class TurnLogger:
    """
    Append-only JSONL logger. Thread-safe (lock per append).

    Penggunaan di main.py (v4.5.6):

        turn_logger = TurnLogger(SRC_DIR / "mei_turns.jsonl")

        # Sambungkan callback FactExtractor
        def _on_fact_saved_cb(turn_no: int, label: str, saved: bool):
            turn_logger.on_fact_result(turn_no, label, saved)
        fact_extractor._on_fact_saved = _on_fact_saved_cb

        shared["turn_logger"] = turn_logger

    Di akhir _process_turn:

        if shared.get("turn_logger") is not None:
            _turn_no_log = shared["msg_count"] * 2 - 1
            _gt_info     = gt_by_turn.get(_turn_no_log, {})
            fact_extractor._current_turn_no = _turn_no_log

            shared["turn_logger"].log_turn(
                record              = record,
                turn_no             = _turn_no_log,
                user_input          = user_input,
                assistant_response  = clean_response,   # ← v4.5.6
                io_mode             = _mode_lbl,
                use_gpu             = io_state.use_gpu,
                label_gt            = _gt_info.get("label_gt", ""),
                chroma_triggered    = chroma_triggered,
                chroma_hits         = len(episodic_hits),
            )
    """

    def __init__(self, log_path: Path, session_id: Optional[str] = None):
        self.log_path   = Path(log_path)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock           = threading.Lock()
        self._pending_fact   : dict[int, dict] = {}   # turn_no → {label, saved}

        self._write_marker("session_start")

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def log_turn(
        self,
        record              : "LatencyRecord",
        turn_no             : int,
        user_input          : str,
        assistant_response  : str  = "",   # ← v4.5.6
        io_mode             : str  = "",
        use_gpu             : bool = True,
        # ── v4.5.5 fields ────────────────────────────────────────
        label_gt            : str  = "",
        chroma_triggered    : bool = False,
        chroma_hits         : int  = 0,
    ) -> None:
        """Tulis satu baris JSONL untuk turn ini."""
        entry = {
            "__type__"          : "turn",
            "session_id"        : self.session_id,
            "timestamp"         : datetime.now().isoformat(),
            "turn_no"           : turn_no,
            "user_input"        : user_input,
            "assistant_response": assistant_response,   # ← v4.5.6
            "io_mode"           : io_mode,
            "use_gpu"           : use_gpu,

            # ── v4.5.5 ────────────────────────────────────────────
            "label_gt"        : label_gt,
            "chroma_triggered": chroma_triggered,
            "chroma_hits"     : chroma_hits,

            # ── Raw perf_counter timestamps (ms, absolut) ─────────
            "raw_timestamps": {
                "t_turn_start"     : record.t_turn_start,
                "t_stt_done"       : record.t_stt_done,
                "t_preprocess_done": record.t_preprocess_done,
                "t_llm_start"      : record.t_llm_start,
                "t_llm_first_token": record.t_llm_first_token,
                "t_llm_done"       : record.t_llm_done,
                "t_first_tts_done" : record.t_first_tts_done,
            },

            # ── Computed metrics (semua @property LatencyRecord) ──
            "metrics": {
                "stt_latency_ms": record.stt_latency_ms,
                "preprocess_ms" : record.preprocess_ms,
                "ttft_ms"       : record.ttft_ms,
                "total_llm_ms"  : record.total_llm_ms,
                "ttfc_ms"       : record.ttfc_ms,
                "tps"           : record.tps,
                "e2e_ms"        : record.e2e_ms,
                "total_tts_ms"  : record.total_tts_ms,
            },

            # ── Pre-processing breakdown (nama step → durasi ms) ──
            "pre_steps": dict(record.pre_steps),

            # ── TTS chunks (semua, tidak dipotong) ────────────────
            "tts_chunks": [
                {
                    "seq"         : c.seq,
                    "text"        : c.text,
                    "synthesis_ms": c.synthesis_ms,
                }
                for c in record.tts_chunks
            ],

            # ── Token counts ──────────────────────────────────────
            "tokens": {
                "input"           : record.input_tokens,
                "output"          : record.output_tokens,
                "output_estimated": record.output_tokens_estimated,
            },

            # ── Tool & flags ──────────────────────────────────────
            "tool_used": record.tool_used,
            "had_stt"  : record.had_stt,
            "had_tts"  : record.had_tts,
        }
        self._append(entry)

        # Flush pending fact result jika sudah tiba sebelum log_turn selesai
        # (race condition: callback lebih cepat dari log_turn)
        with self._lock:
            pending = self._pending_fact.pop(turn_no, None)
        if pending is not None:
            self._write_fact_patch(
                turn_no = turn_no,
                label   = pending["label"],
                saved   = pending["saved"],
            )

    def on_fact_result(self, turn_no: int, label: str, saved: bool) -> None:
        """
        Dipanggil (async) dari FactExtractor._fire_callback setelah _process selesai.
        Menulis baris "fact_patch" ke JSONL — bisa datang kapan saja setelah log_turn.

        Jika log_turn untuk turn_no ini BELUM dipanggil (race condition sangat jarang
        terjadi karena fact_extractor berjalan di background thread), simpan ke
        _pending_fact — akan di-flush di dalam log_turn.
        """
        # Cek apakah turn ini sudah ada di file (log_turn sudah dipanggil).
        # Karena kita append-only, kita tidak tahu pasti — pendekatan sederhana:
        # langsung tulis fact_patch. Notebook bisa join berdasarkan turn_no + session_id.
        self._write_fact_patch(turn_no, label, saved)

    def log_session_end(self) -> None:
        self._write_marker("session_end")

    # ──────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────

    def _write_fact_patch(self, turn_no: int, label: str, saved: bool) -> None:
        """Tulis baris fact_patch untuk satu turn."""
        self._append({
            "__type__"            : "fact_patch",
            "session_id"          : self.session_id,
            "timestamp"           : datetime.now().isoformat(),
            "turn_no"             : turn_no,
            "fact_label_predicted": label,
            "fact_saved"          : saved,
        })

    def _write_marker(self, marker_type: str) -> None:
        self._append({
            "__type__"  : marker_type,
            "session_id": self.session_id,
            "timestamp" : datetime.now().isoformat(),
        })

    def _append(self, obj: dict) -> None:
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")