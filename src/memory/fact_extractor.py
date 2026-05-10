"""
Async Fact Extractor — MEI v6.2.4
==================================
Pipeline 4-step yang dioptimalkan untuk SLM kecil (Qwen 3 1.7B):

  [1] Rule-based filter     — eliminasi basa-basi SEBELUM kena model apapun
  [1.5] Recall question filter — eliminasi pertanyaan recall/ingat SEBELUM classifier
  [2] Classifier            — label via cosine similarity (ST) atau NLI pipeline
  [3] SLM binary + teks     — HANYA untuk label ambigu
                              Label jelas (data_pribadi/project) langsung SAVE
                              Label jelas tidak penting langsung DROP
  [4] Python routing        — berdasarkan label + should_save → simpan / drop

Perubahan dari v6.2.3 (PATCH v6.2.4):
  - FIX: Prefix [USER] tidak lagi ikut tersimpan ke ChromaDB.
    Akar masalah: _format_user_only() menghasilkan string dengan prefix
    "[USER] ..." yang kemudian langsung dipakai sebagai summary di
    _save_episodic(). Akibatnya isi ChromaDB menjadi:
      "[2026-05-09] [USER] Hai, nama saya rifki..."
    bukannya:
      "[2026-05-09] Hai, nama saya rifki..."
  - ADD: _extract_user_texts() — helper baru yang mengambil konten pesan
    user tanpa prefix role. Digunakan khusus untuk isi summary ChromaDB.
  - MOD: Jalur 2 (direct save) & Jalur 3 (SLM/ambigu): ganti
    _format_user_only(messages) → _extract_user_texts(messages) untuk
    pembentukan summary yang akan disimpan ke ChromaDB.
  - MOD: _save_episodic() fallback: hapus hardcode "[USER]" prefix
    agar konsisten dengan perubahan di atas.
  - NOTE: _format_user_only() tetap dipertahankan — masih dipakai
    oleh conv untuk classifier Step [2] yang butuh label role.

Perubahan dari v6.2.2 (PATCH v6.2.3):
  - FIX: Summary episodic sekarang hanya menyimpan pesan USER, bukan ASSISTANT
  - FIX: Jalur 3 (SLM/ambigu): abaikan parsed["summary"] dari SLM
  - ADD: _format_user_only()

Perubahan dari v6.2.1 (PATCH v4.5.5 → v6.2.2):
  - ADD: _RECALL_QUESTION_PATTERNS
  - ADD: _is_recall_question()
  - ADD: Step [1.5] di _process()

Perubahan dari v6.2.0 (PATCH v4.5.5):
  - ADD: on_fact_saved callback(turn_no, label, saved)
  - ADD: _current_turn_no
  - ADD: _fire_callback()
  - ADD: on_fact_saved parameter di __init__

Labels yang didukung:
   data_pribadi | pekerjaan | project | teknis | preferensi |
   pengalaman   | lainnya   | tidak_penting

Importance mapping (Python, bukan SLM):
  data_pribadi → 9 | project/pekerjaan → 8 | teknis/preferensi → 7
  pengalaman → 6   | pertanyaan/lainnya → 5
  chat_biasa → 3   | basa-basi → 1
"""

import re
import queue
import threading
import time
import requests
from datetime import date
from typing import Optional, Callable


# ── Logger ─────────────────────────────────────────────────────────

def _log(msg: str, level: str = "INFO", debug: bool = False):
    if level == "DEBUG" and not debug:
        return
    prefix = {
        "INFO" : "  [FactExtractor]",
        "DEBUG": "  [FactExtractor][DBG]",
        "WARN" : "  [FactExtractor][WARN]",
        "ERROR": "  [FactExtractor][ERR]",
    }.get(level, "  [FactExtractor]")
    print(f"{prefix} {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════
# STEP [1] — RULE-BASED FILTER
# ══════════════════════════════════════════════════════════════════

_SKIP_PATTERNS = re.compile(
    r"^("
    r"(ok|oke|oks|okay)[!.,\s]*|"
    r"(iya|ya|yep|yup)[!.,\s]*|"
    r"(haha|hihi|wkwk|hehe|lol)+[!.,\s]*|"
    r"(sip|siap|noted)[!.,\s]*|"
    r"(makasih|thanks|terima kasih|thank you|thx|ty)[!.,\s]*|"
    r"(sama-sama|no problem|np)[!.,\s]*|"
    r"(halo|hai|hi|hello|hey)[!.,\s]*|"
    r"(selamat pagi|selamat siang|selamat sore|selamat malam)[!.,\s]*|"
    r"(bye|sampai jumpa|dah|da)[!.,\s]*|"
    r"(bagus|keren|mantap|nice|good|great)[!.,\s]*|"
    r"(paham|ngerti|mengerti)[!.,\s]*"
    r")$",
    re.IGNORECASE,
)

_MIN_CONTENT_CHARS = 15


# ══════════════════════════════════════════════════════════════════
# STEP [1.5] — RECALL QUESTION FILTER
# ══════════════════════════════════════════════════════════════════

_RECALL_QUESTION_PATTERNS = re.compile(
    r"^("
    r"ingatkan|ingetin|"
    r"kamu ingat|kamu masih|kamu tau|kamu tahu|"
    r"masih ingat|masih inget|"
    r"lo ingat|lo inget|lo masih|"
    r"apa (itu|tadi|yang)|"
    r"kapan (itu|tadi)|"
    r"hari apa|jam berapa|"
    r"gimana (tadi|itu|hasilnya|progressnya)|"
    r"siapa yang|di mana|berapa (lama|kali|tahun)|"
    r"saya pernah (bilang|cerita|ngomong|bahas)|"
    r"gue pernah (bilang|cerita|ngomong|bahas)|"
    r"aku pernah (bilang|cerita|ngomong|bahas)|"
    r"pernah (bilang|cerita|ngomong|bahas)|"
    r"kamu (masih )?(inget|ingat|remember)"
    r")",
    re.IGNORECASE
)


def _is_recall_question(messages: list[dict]) -> bool:
    """
    Deteksi apakah pesan terakhir dari user adalah pertanyaan recall
    (menanyakan kembali informasi lama) bukan deklarasi fakta baru.
    """
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return False
    last = (user_msgs[-1].get("content") or "").strip()
    return bool(_RECALL_QUESTION_PATTERNS.match(last))


# ── Label routing ──────────────────────────────────────────────────

_NO_SAVE_LABELS     = {"tidak_penting"}
_DIRECT_SAVE_LABELS = {"data_pribadi", "project", "pekerjaan", "preferensi"}
_AMBIGUOUS_LABELS   = {"teknis", "pengalaman", "lainnya"}


def _rule_based_filter(messages: list[dict]) -> bool:
    contents = [
        (msg, (msg.get("content") or "").strip())
        for msg in messages
        if (msg.get("content") or "").strip()
    ]
    if not contents:
        return True

    total_chars = sum(len(c) for _, c in contents)
    if total_chars < _MIN_CONTENT_CHARS:
        return True

    non_system = [c for msg, c in contents if msg.get("role") != "system"]
    if non_system and all(_SKIP_PATTERNS.match(c) for c in non_system):
        return True

    return False


# ══════════════════════════════════════════════════════════════════
# STEP [2] — CLASSIFIER
# ══════════════════════════════════════════════════════════════════

_LABEL_CANDIDATES = [
    "nama pengguna, nomor HP, email, alamat, atau data identitas personal",
    "jadwal kerja, meeting rutin, reminder pekerjaan, atau aktivitas profesional sehari-hari — bukan proyek spesifik",
    "proyek software spesifik yang sedang dikerjakan: sprint planning, arsitektur sistem, code review project sendiri",
    "pertanyaan atau diskusi teknis umum: Docker, JWT, REST API, GraphQL, Node.js — bukan tentang proyek sendiri",
    "preferensi jadwal, kebiasaan kerja, atau cara kerja yang disukai pengguna. Contoh: prefer meeting pagi, standup rutin",
    "riwayat karir, pengalaman kerja masa lalu, atau portfolio — bukan proyek yang sedang berjalan sekarang",
    "hal random, aktivitas non-kerja, pertanyaan singkat, atau topik yang tidak ada informasi personal berguna",
    "sapaan, basa-basi, tawa, acknowledgement, atau penutup percakapan tanpa informasi apapun",
]

_CANDIDATE_TO_LABEL = {
    "nama pengguna, nomor HP, email, alamat, atau data identitas personal"                                                                      : "data_pribadi",
    "jadwal kerja, meeting rutin, reminder pekerjaan, atau aktivitas profesional sehari-hari — bukan proyek spesifik"                           : "pekerjaan",
    "proyek software spesifik yang sedang dikerjakan: sprint planning, arsitektur sistem, code review project sendiri"                          : "project",
    "pertanyaan atau diskusi teknis umum: Docker, JWT, REST API, GraphQL, Node.js — bukan tentang proyek sendiri"                              : "teknis",
    "preferensi jadwal, kebiasaan kerja, atau cara kerja yang disukai pengguna. Contoh: prefer meeting pagi, standup rutin"                    : "preferensi",
    "riwayat karir, pengalaman kerja masa lalu, atau portfolio — bukan proyek yang sedang berjalan sekarang"                                   : "pengalaman",
    "hal random, aktivitas non-kerja, pertanyaan singkat, atau topik yang tidak ada informasi personal berguna"                                : "lainnya",
    "sapaan, basa-basi, tawa, acknowledgement, atau penutup percakapan tanpa informasi apapun"                                                 : "tidak_penting",
}

_CLASSIFIER_CACHE: dict = {}
_ST_LABEL_CACHE  : dict = {}


def _get_nli_classifier(model_name: str, device: str):
    key = (model_name, device)
    if key not in _CLASSIFIER_CACHE:
        from transformers import pipeline
        _log(f"Loading NLI classifier: {model_name} | device={device}")
        device_id = 0 if device == "cuda" else -1
        _CLASSIFIER_CACHE[key] = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device_id,
        )
        _log("NLI Classifier ready")
    return _CLASSIFIER_CACHE[key]


def _classify_with_nli(text: str, classifier) -> str:
    try:
        result        = classifier(text, candidate_labels=_LABEL_CANDIDATES, multi_label=False)
        top_candidate = result["labels"][0]
        return _CANDIDATE_TO_LABEL.get(top_candidate, "lainnya")
    except Exception as e:
        _log(f"NLI Classifier error: {e}", level="WARN")
        return "lainnya"


def _classify_with_st(text: str, st_model) -> str:
    import numpy as np

    model_key = id(st_model)
    if model_key not in _ST_LABEL_CACHE:
        _ST_LABEL_CACHE[model_key] = st_model.encode(
            _LABEL_CANDIDATES, convert_to_numpy=True, show_progress_bar=False
        )
    label_embeddings = _ST_LABEL_CACHE[model_key]

    try:
        text_emb  = st_model.encode([text], convert_to_numpy=True)[0]
        norms     = np.linalg.norm(label_embeddings, axis=1) * np.linalg.norm(text_emb)
        scores    = label_embeddings @ text_emb / (norms + 1e-8)
        top_label = _LABEL_CANDIDATES[int(np.argmax(scores))]
        return _CANDIDATE_TO_LABEL.get(top_label, "lainnya")
    except Exception as e:
        _log(f"ST classifier error: {e}", level="WARN")
        return "lainnya"


# ══════════════════════════════════════════════════════════════════
# STEP [3] — SLM BINARY (hanya untuk label ambigu)
# ══════════════════════════════════════════════════════════════════

_IMPORTANCE_PROMPT = """\
Kamu membantu sistem memori AI personal assistant bernama MEI.

Percakapan:
{conversation}

Kategori: {label}

Apakah percakapan ini perlu disimpan ke memori jangka panjang?
Jawab "yes" jika ada informasi berguna untuk masa depan, "no" jika tidak.

Jawab dengan format ini (3 baris, tidak ada yang lain):
SAVE: yes atau no
SUMMARY: satu kalimat ringkasan isi percakapan (kosong jika no)
NOTE: satu kalimat catatan harian natural (kosong jika no)
"""


def _parse_slm_response(raw: str) -> dict:
    result = {"should_save": False, "summary": "", "daily_note": ""}
    _EMPTY = {"", "kosong", "-", "none", "n/a", "tidak ada", "empty"}

    for line in raw.splitlines():
        line = line.strip()
        low  = line.lower()
        if low.startswith("save:"):
            val = line[5:].strip().lower()
            result["should_save"] = val.startswith("y") or val.startswith("iya")
        elif low.startswith("summary:"):
            val = line[8:].strip().strip("[]()\"'")
            if val.lower() not in _EMPTY:
                result["summary"] = val
        elif low.startswith("note:"):
            val = line[5:].strip().strip("[]()\"'")
            if val.lower() not in _EMPTY:
                result["daily_note"] = val
    return result


# ══════════════════════════════════════════════════════════════════
# IMPORTANCE MAP
# ══════════════════════════════════════════════════════════════════

_LABEL_IMPORTANCE = {
    "data_pribadi" : 9,
    "project"      : 8,
    "pekerjaan"    : 8,
    "teknis"       : 7,
    "preferensi"   : 7,
    "pengalaman"   : 6,
    "lainnya"      : 5,
    "tidak_penting": 1,
}


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

MAX_CONV_CHARS = 2500


def _format_conversation(messages: list[dict], max_chars: int = MAX_CONV_CHARS) -> str:
    lines = []
    total = 0
    for msg in reversed(messages[-20:]):
        if msg.get("role") == "function":
            continue
        role    = msg.get("role", "?").upper()
        content = (msg.get("content") or "")[:300].replace("\n", " ")
        line    = f"[{role}] {content}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(reversed(lines))


def _format_user_only(messages: list[dict], max_chars: int = MAX_CONV_CHARS) -> str:
    """
    Sama seperti _format_conversation tapi HANYA mengambil pesan role 'user'.
    Masih dipakai oleh classifier Step [2] yang butuh label role [USER].
    JANGAN gunakan output fungsi ini sebagai isi summary ChromaDB —
    gunakan _extract_user_texts() untuk itu.
    """
    lines = []
    total = 0
    for msg in reversed(messages[-20:]):
        if msg.get("role") != "user":
            continue
        content = (msg.get("content") or "")[:300].replace("\n", " ")
        line    = f"[USER] {content}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(reversed(lines))


def _extract_user_texts(messages: list[dict], max_chars: int = MAX_CONV_CHARS) -> str:
    """
    Ambil konten pesan user saja, TANPA prefix role [USER].

    Digunakan khusus untuk membentuk summary yang akan disimpan ke ChromaDB,
    agar isi memori bersih dari label role dan hanya berisi teks asli user.

    Contoh output:
      "Hai, nama saya rifki ainul yaqin, saya seorang backend developer."

    Bukan:
      "[USER] Hai, nama saya rifki ainul yaqin, saya seorang backend developer."

    Berbeda dari _format_user_only() yang sengaja menyertakan prefix [USER]
    untuk keperluan classifier yang butuh konteks role.
    """
    parts = []
    total = 0
    for msg in messages[-20:]:
        if msg.get("role") != "user":
            continue
        content = (msg.get("content") or "")[:300].replace("\n", " ").strip()
        if not content:
            continue
        if total + len(content) + 3 > max_chars:
            break
        parts.append(content)
        total += len(content) + 3  # 3 = len(" | ")
    return " | ".join(parts)


def _make_fingerprint(messages: list[dict]) -> str:
    parts = []
    for m in messages[-5:]:
        content = (m.get("content") or "")[:80]
        parts.append(f"{m.get('role','?')}:{content}")
    return "|".join(parts)


# ══════════════════════════════════════════════════════════════════
# FACT EXTRACTOR
# ══════════════════════════════════════════════════════════════════

class FactExtractor:
    """
    Async fact extractor untuk MEI — pipeline 4-step.

    classifier_model:
      "st"          → reuse ST embedding ChromaDB (default, hemat VRAM)
      nama HF model → load NLI zero-shot pipeline

    on_fact_saved (v4.5.5):
      Callback opsional dengan signature: callback(turn_no: int, label: str, saved: bool)
      Dipanggil setiap kali _process selesai (termasuk kasus drop).
      Digunakan oleh TurnLogger untuk mencatat fact_label_predicted dan fact_saved.
    """

    def __init__(
        self,
        lt_mem,
        llm_config       : dict,
        classifier_model : str                              = "st",
        user_id          : str                              = "rifki",
        device           : str                              = "cuda",
        debug            : bool                             = False,
        on_fact_saved    : Optional[Callable]               = None,
    ):
        self._lt_mem           = lt_mem
        self._llm_cfg          = llm_config
        self._classifier_model = classifier_model
        self._user_id          = user_id
        self._device           = device
        self._debug            = debug

        self._last_extracted_content : str = ""
        self._last_submitted_idx     : int = 0
        self._classifier                   = None

        # ── callback fields ───────────────────────────────────────
        self._on_fact_saved   : Optional[Callable] = on_fact_saved
        self._current_turn_no : int                = 0

        self._queue  : queue.Queue      = queue.Queue(maxsize=100)
        self._thread : threading.Thread = threading.Thread(
            target=self._worker, daemon=True, name="FactExtractor"
        )
        self._running = False

    # ── Lifecycle ──────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread.start()
        _log("Started (background thread)")

    def stop(self, timeout: float = 90.0):
        self._running = False
        self._queue.put(None)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            _log("Warning: thread did not stop cleanly", level="WARN")
        else:
            _log("Stopped")

    def _get_classifier(self):
        if self._classifier is None:
            if self._classifier_model == "st":
                self._classifier = self._lt_mem._ef._model()
                _log("Classifier: ST cosine similarity (reuse ChromaDB model)")
            else:
                self._classifier = _get_nli_classifier(self._classifier_model, self._device)
        return self._classifier

    # ── Public API ─────────────────────────────────────────────────

    def submit(self, messages: list[dict]):
        if not messages:
            return
        if _rule_based_filter(messages):
            _log("Rule-based: skip", level="DEBUG", debug=self._debug)
            return
        fingerprint = _make_fingerprint(messages)
        if fingerprint == self._last_extracted_content:
            _log("Duplicate batch, skipping", level="DEBUG", debug=self._debug)
            return
        try:
            self._queue.put_nowait(list(messages))
            self._last_extracted_content = fingerprint
            _log(f"Batch queued ({len(messages)} msgs)", level="DEBUG", debug=self._debug)
        except queue.Full:
            _log("Queue full, skipping batch", level="WARN")

    def submit_periodic(self, all_session_msgs: list[dict], n_turns: int):
        new_msgs = all_session_msgs[self._last_submitted_idx:]
        if not new_msgs:
            _log("No new messages for periodic extraction", level="DEBUG", debug=self._debug)
            return
        _log(
            f"Periodic: {len(new_msgs)} new msgs "
            f"(idx {self._last_submitted_idx}→{len(all_session_msgs)}, turn={n_turns})"
        )
        self.submit(new_msgs)
        self._last_submitted_idx = len(all_session_msgs)

    def submit_remaining(self, all_session_msgs: list[dict]):
        remaining = all_session_msgs[self._last_submitted_idx:]
        if not remaining:
            _log("No remaining messages for on-exit extraction")
            return
        _log(
            f"On-exit safety net: {len(remaining)} remaining msgs "
            f"(idx {self._last_submitted_idx}→{len(all_session_msgs)})"
        )
        self.submit(remaining)
        self._last_submitted_idx = len(all_session_msgs)

    # ── Worker Loop ────────────────────────────────────────────────

    def _worker(self):
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
                if item is None:
                    self._queue.task_done()
                    break
                try:
                    self._process(item)
                except Exception as e:
                    import traceback
                    _log(f"Process error: {e}", level="ERROR")
                    if self._debug:
                        traceback.print_exc()
                finally:
                    self._queue.task_done()
            except queue.Empty:
                continue

    # ── Core Process ───────────────────────────────────────────────

    def _process(self, messages: list[dict]):
        t0 = time.perf_counter()

        conv = _format_conversation(messages, MAX_CONV_CHARS)
        if not conv.strip():
            _log("Empty conversation, skipping", level="DEBUG", debug=self._debug)
            self._fire_callback("", False)
            return

        # ── Step [1.5]: recall question filter ────────────────────
        if _is_recall_question(messages):
            _log("Recall question detected → skip save (step 1.5)")
            self._fire_callback("tidak_penting", False)
            return

        # ── Step [2]: classifier → label ──────────────────────────
        classifier = self._get_classifier()
        if self._classifier_model == "st":
            label = _classify_with_st(conv, classifier)
        else:
            label = _classify_with_nli(conv, classifier)

        _log(f"Step[2] label={label}")

        importance = _LABEL_IMPORTANCE.get(label, 6)

        # Jalur 1: jelas tidak penting → DROP
        if label in _NO_SAVE_LABELS:
            _log(f"Label '{label}' → drop (no SLM)")
            self._fire_callback(label, False)
            return

        # Jalur 2: jelas penting → SAVE langsung (tanpa SLM)
        if label in _DIRECT_SAVE_LABELS:
            _log(f"Label '{label}' imp={importance} → direct save (no SLM)")

            # FIX v6.2.4: pakai _extract_user_texts() — tanpa prefix [USER]
            summary    = _extract_user_texts(messages)[:150]
            daily_note = f"[{label}] {summary[:80]}"

            self._save_daily(daily_note)
            self._save_episodic(label, summary, messages)
            elapsed = (time.perf_counter() - t0) * 1000
            _log(f"Done {elapsed:.0f}ms | label={label} | saved=True (direct)")
            self._fire_callback(label, True)
            return

        # Jalur 3: ambigu → tanya SLM
        _log(f"Label '{label}' → SLM check", level="DEBUG", debug=self._debug)
        raw_response = self._call_slm(conv, label)
        if not raw_response:
            _log("SLM returned empty response", level="WARN")
            self._fire_callback(label, False)
            return

        _log(f"SLM raw:\n{raw_response}", level="DEBUG", debug=self._debug)

        parsed      = _parse_slm_response(raw_response)
        should_save = parsed["should_save"]
        daily_note  = parsed["daily_note"]

        # FIX v6.2.4: pakai _extract_user_texts() — tanpa prefix [USER]
        # SLM hanya dipakai untuk should_save dan daily_note, bukan summary
        summary = _extract_user_texts(messages)[:150]

        if daily_note:
            self._save_daily(daily_note)
        if should_save:
            self._save_episodic(label, summary, messages)
        else:
            _log(f"SLM: not saved (label={label})", level="DEBUG", debug=self._debug)

        elapsed = (time.perf_counter() - t0) * 1000
        _log(f"Done {elapsed:.0f}ms | label={label} | saved={should_save}")
        self._fire_callback(label, should_save)

    # ── Callback ───────────────────────────────────────────────────

    def _fire_callback(self, label: str, saved: bool):
        if self._on_fact_saved and self._current_turn_no > 0:
            try:
                self._on_fact_saved(self._current_turn_no, label, saved)
            except Exception as e:
                _log(f"on_fact_saved callback error: {e}", level="WARN")

    # ── SLM Call ───────────────────────────────────────────────────

    def _call_slm(self, conversation: str, label: str) -> str:
        prompt = (
            _IMPORTANCE_PROMPT
            .replace("{conversation}", conversation)
            .replace("{label}", label)
        )
        for attempt in range(2):
            try:
                resp = requests.post(
                    f"{self._llm_cfg['model_server']}/chat/completions",
                    headers={"Authorization": f"Bearer {self._llm_cfg.get('api_key', 'lm-studio')}"},
                    json={
                        "model"      : self._llm_cfg["model"],
                        "messages"   : [{"role": "user", "content": prompt}],
                        "max_tokens" : 120,
                        "temperature": 0.1,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                _log(f"SLM call failed (attempt {attempt + 1}): {e}", level="ERROR")
        return ""

    # ── Save Daily ─────────────────────────────────────────────────

    def _save_daily(self, note: str):
        try:
            self._lt_mem.append_daily_and_maybe_compress(
                text=note, tag="auto", llm_cfg=self._llm_cfg,
            )
            _log(f"Daily note saved: {note[:70]}")
        except Exception as e:
            _log(f"Failed to save daily note: {e}", level="ERROR")

    # ── Save Episodic ──────────────────────────────────────────────

    def _save_episodic(self, label: str, summary: str, raw_msgs: list[dict]):
        if not summary:
            _log("Summary kosong, pakai fallback snippet", level="WARN")
            parts = []
            for msg in raw_msgs[-4:]:
                if msg.get("role") != "user":
                    continue
                # FIX v6.2.4: fallback tanpa prefix [USER]
                content = (msg.get("content") or "")[:100].replace("\n", " ").strip()
                if content:
                    parts.append(content)
            if not parts:
                _log("Tidak ada konten untuk fallback, skip ChromaDB", level="WARN")
                return
            summary = " | ".join(parts)

        today      = date.today().isoformat()
        importance = _LABEL_IMPORTANCE.get(label, 6)
        content    = f"[{today}] {summary}"

        self._lt_mem.add_memory(
            content     = content,
            memory_type = "episodic",
            importance  = importance,
            metadata    = {
                "label"    : label,
                "msg_count": len(raw_msgs),
                "user_id"  : self._user_id,
            },
        )
        _log(f"Episodic saved (label={label}, imp={importance}): {summary[:70]}")