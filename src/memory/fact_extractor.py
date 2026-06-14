"""
Async Fact Extractor — MEI v6.4.0
==================================
Pipeline 4-step yang dioptimalkan untuk LLM lokal (Qwen3-VL 4B / SLM):

  [1]   Rule-based filter     — eliminasi basa-basi SEBELUM kena model apapun
  [1.5] Recall question filter — eliminasi pertanyaan recall/ingat
  [1.6] Utility command filter — eliminasi perintah utilitas one-shot
        set timer, cari berita, cek waktu, ambil gambar, dll.
  [2]   Classifier            — label + confidence via cosine similarity (ST) atau NLI
  [3]   SLM binary + teks     — HANYA untuk label ambigu ATAU direct-save confidence rendah
                                Label jelas langsung SAVE (jika confidence ≥ 0.45)
                                Label jelas tidak penting langsung DROP
  [4]   Python routing        — berdasarkan label + should_save → simpan / drop

Perubahan dari v6.3.2 (v6.4.0):
  - FIX COVERAGE: CPU path Jalur 2 (direct save) — sebelumnya langsung simpan
    _extract_user_texts() mentah. Sekarang di-route ke SLM untuk di-summarize
    menjadi kalimat dense third-person sebelum disimpan ke ChromaDB.
    Impact: kualitas teks ChromaDB lebih informatif → retrieval lebih akurat.

  - FIX COVERAGE GPU: _SLM_DIRECT_PROMPT diperkuat dengan 3 few-shot examples
    konkret (input → SAVE/LABEL/SUMMARY/NOTE) sehingga Qwen3-VL 4B lebih
    konsisten dalam format output dan keputusan SAVE/DROP.

  - FIX: max_tokens _call_slm_direct naik dari 150 → 200 untuk beri ruang
    SLM menulis SUMMARY yang lebih informatif.

  - ADD: _SLM_SUMMARIZE_PROMPT — prompt baru khusus untuk CPU Jalur 2.
    SLM diminta tulis ulang teks user menjadi kalimat padat third-person
    (bukan extract mentah). Fallback ke raw text jika SLM gagal/timeout.

  - ADD: _call_slm_summarize() — helper untuk CPU summarize path.
    Timeout 45 detik, max_tokens 100, temperature 0.1.

  - KEEP: Semua logika v6.3.x lainnya tidak berubah (rule filter, NLI path,
    recall filter, utility filter, GPU path branching, callback, worker).
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
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return False
    last = (user_msgs[-1].get("content") or "").strip()
    return bool(_RECALL_QUESTION_PATTERNS.match(last))


# ══════════════════════════════════════════════════════════════════
# STEP [1.6] — UTILITY COMMAND FILTER (v6.3.0)
# ══════════════════════════════════════════════════════════════════

_UTILITY_PATTERNS = re.compile(
    r"^("
    # Timer
    r"(tolong\s+|coba\s+)?(set|buat|mulai|pasang|tambah)\s+timer|"
    r"(batalkan|cancel|stop)\s+timer|"
    r"(list|tampilkan|cek)\s+timer|"
    # Berita / search
    r"(carikan|cari|ambil|tampilkan|kasih|kasih tau|cek)\s+berita|"
    r"ada\s+berita\s+(apa|terbaru|hari ini)|"
    r"berita\s+(hari ini|terbaru|terkini)|"
    r"(tolong\s+)?cari(kan)?\s+(berita|jadwal|lokasi|tempat|restoran|cuaca|gambar|foto)|"
    # Waktu / tanggal
    r"(jam|pukul)\s+berapa(\s+sekarang)?[\?]*|"
    r"sekarang\s+(jam|pukul)\s+berapa[\?]*|"
    r"(tanggal|hari)\s+(berapa|apa|ini)[\?]*|"
    r"tanggal[\?]*$|"
    # Kamera
    r"(tolong\s+)?(ambil|foto|capture|analisa|analisis)\s+(gambar|foto|kamera|image)|"
    r"(tolong\s+)?ambil\s+gambar|"
    # Kalender utilitas
    r"(cek|tampilkan|lihat)\s+(jadwal|event|kalender|agenda)|"
    r"(ada\s+)?(event|jadwal)\s+(apa|hari ini|besok|minggu ini)|"
    # Cuaca
    r"cuaca\s+(hari ini|sekarang|di\s+\w+)|"
    r"(bagaimana|gimana)\s+cuaca|"
    # Notifikasi
    r"(tolong\s+)?(set|buat|jadwalkan)\s+notifikasi|"
    r"(batalkan|cancel)\s+notifikasi"
    r")",
    re.IGNORECASE,
)


def _is_utility_command(messages: list[dict]) -> bool:
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return False
    last = (user_msgs[-1].get("content") or "").strip()
    return bool(_UTILITY_PATTERNS.match(last))


# ── Label routing ──────────────────────────────────────────────────

_NO_SAVE_LABELS     = {"tidak_penting", "lainnya"}
_DIRECT_SAVE_LABELS = {"data_pribadi", "project", "pekerjaan", "preferensi"}
_AMBIGUOUS_LABELS   = {"teknis", "pengalaman"}


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
    (
        "jadwal kerja terjadwal, meeting rutin berulang, atau pengingat tugas "
        "profesional spesifik — BUKAN query informasi umum, BUKAN perintah "
        "satu-kali seperti cek berita atau set timer"
    ),
    "proyek software spesifik yang sedang dikerjakan: sprint planning, arsitektur sistem, code review project sendiri",
    (
        "pertanyaan atau diskusi teknis umum mendalam: Docker, JWT, REST API, "
        "GraphQL, Node.js — diskusi panjang, bukan pencarian satu-kali"
    ),
    (
        "preferensi jadwal, kebiasaan kerja, atau cara kerja yang disukai "
        "pengguna yang dinyatakan secara eksplisit — BUKAN perintah one-shot"
    ),
    "riwayat karir, pengalaman kerja masa lalu, atau portfolio — bukan proyek yang sedang berjalan sekarang",
    "hal random, aktivitas non-kerja, pertanyaan singkat umum, atau topik yang tidak ada informasi personal berguna",
    (
        "sapaan, basa-basi, tawa, acknowledgement, atau penutup percakapan; "
        "juga perintah satu-kali seperti set timer, cari berita, cek waktu, "
        "ambil foto, atau query informasi tanpa konteks personal"
    ),
]

_CANDIDATE_TO_LABEL = {
    "nama pengguna, nomor HP, email, alamat, atau data identitas personal": "data_pribadi",
    (
        "jadwal kerja terjadwal, meeting rutin berulang, atau pengingat tugas "
        "profesional spesifik — BUKAN query informasi umum, BUKAN perintah "
        "satu-kali seperti cek berita atau set timer"
    ): "pekerjaan",
    "proyek software spesifik yang sedang dikerjakan: sprint planning, arsitektur sistem, code review project sendiri": "project",
    (
        "pertanyaan atau diskusi teknis umum mendalam: Docker, JWT, REST API, "
        "GraphQL, Node.js — diskusi panjang, bukan pencarian satu-kali"
    ): "teknis",
    (
        "preferensi jadwal, kebiasaan kerja, atau cara kerja yang disukai "
        "pengguna yang dinyatakan secara eksplisit — BUKAN perintah one-shot"
    ): "preferensi",
    "riwayat karir, pengalaman kerja masa lalu, atau portfolio — bukan proyek yang sedang berjalan sekarang": "pengalaman",
    "hal random, aktivitas non-kerja, pertanyaan singkat umum, atau topik yang tidak ada informasi personal berguna": "lainnya",
    (
        "sapaan, basa-basi, tawa, acknowledgement, atau penutup percakapan; "
        "juga perintah satu-kali seperti set timer, cari berita, cek waktu, "
        "ambil foto, atau query informasi tanpa konteks personal"
    ): "tidak_penting",
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


def _classify_with_nli(text: str, classifier) -> tuple[str, float]:
    """Return (label, confidence). confidence = skor probabilitas top label."""
    try:
        result        = classifier(text, candidate_labels=_LABEL_CANDIDATES, multi_label=False)
        top_candidate = result["labels"][0]
        confidence    = float(result["scores"][0])
        return _CANDIDATE_TO_LABEL.get(top_candidate, "lainnya"), confidence
    except Exception as e:
        _log(f"NLI Classifier error: {e}", level="WARN")
        return "lainnya", 0.0


def _classify_with_st(text: str, st_model) -> tuple[str, float]:
    """Return (label, confidence). confidence = cosine similarity tertinggi."""
    import numpy as np

    model_key = id(st_model)
    if model_key not in _ST_LABEL_CACHE:
        _ST_LABEL_CACHE[model_key] = st_model.encode(
            _LABEL_CANDIDATES, convert_to_numpy=True, show_progress_bar=False
        )
    label_embeddings = _ST_LABEL_CACHE[model_key]

    try:
        text_emb   = st_model.encode([text], convert_to_numpy=True)[0]
        norms      = np.linalg.norm(label_embeddings, axis=1) * np.linalg.norm(text_emb)
        scores     = label_embeddings @ text_emb / (norms + 1e-8)
        top_idx    = int(np.argmax(scores))
        top_label  = _LABEL_CANDIDATES[top_idx]
        confidence = float(scores[top_idx])
        return _CANDIDATE_TO_LABEL.get(top_label, "lainnya"), confidence
    except Exception as e:
        _log(f"ST classifier error: {e}", level="WARN")
        return "lainnya", 0.0


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
# GPU DIRECT PATH — prompt + helpers (v6.3.2, diperkuat v6.4.0)
# ══════════════════════════════════════════════════════════════════

# v6.4.0: Diperkuat dengan 3 few-shot examples konkret
# Sebelumnya hanya ada 2 contoh bagus/buruk summary yang terlalu singkat.
# Sekarang ada 3 contoh lengkap input → output 4 baris agar Qwen3-VL 4B
# lebih konsisten dalam format dan keputusan SAVE/DROP.
_SLM_DIRECT_PROMPT = """Kamu adalah sistem memori AI personal assistant bernama MEI.
Tugasmu: analisis pesan user, putuskan apakah perlu disimpan ke memori jangka panjang,
dan kalau ya — tulis ulang menjadi kalimat informatif padat third-person.

Percakapan user:
{conversation}

Aturan keputusan:
- SIMPAN jika ada fakta personal: nama, kontak, preferensi, pengalaman, project aktif, info pekerjaan/teknologi
- JANGAN SIMPAN jika: basa-basi, recall question, perintah utilitas one-shot (timer, cari berita, cek jadwal, ambil foto)

Jika SAVE=yes, tulis ulang sebagai kalimat padat third-person:
- Gunakan nama asli user jika diketahui, atau sebut "pengguna"
- Sertakan fakta spesifik (nama teknologi, angka, waktu, nama project)
- Maksimal 1-2 kalimat
- JANGAN gunakan "[USER]" sebagai subjek

CONTOH 1 — data pribadi (nama & kontak):
Input: [USER] Hai, nama saya Budi, saya kerja sebagai data engineer di Jakarta.
SAVE: yes
LABEL: data_pribadi
SUMMARY: Budi adalah data engineer yang bekerja di Jakarta.
NOTE: Budi memperkenalkan diri sebagai data engineer di Jakarta.

CONTOH 2 — utilitas one-shot (tidak disimpan):
Input: [USER] Cari berita teknologi terbaru dong.
SAVE: no
LABEL: tidak_penting
SUMMARY:
NOTE:

CONTOH 3 — preferensi kerja:
Input: [USER] Gue lebih suka kerja dari rumah, produktivitas lebih tinggi kalau tidak di kantor.
SAVE: yes
LABEL: preferensi
SUMMARY: Pengguna lebih produktif saat bekerja dari rumah dibanding di kantor.
NOTE: Pengguna menyebutkan preferensi WFH karena lebih produktif.

CONTOH 4 — diskusi teknis mendalam:
Input: [USER] Gimana cara optimasi query PostgreSQL yang lambat? Saya pakai index tapi masih lelet.
SAVE: yes
LABEL: teknis
SUMMARY: Pengguna mempelajari optimasi query PostgreSQL yang lambat meskipun sudah menggunakan index.
NOTE: Pengguna mendiskusikan masalah performa query PostgreSQL.

CONTOH 5 — project aktif:
Input: [USER] Project saya sekarang namanya PaymentHub, pakai microservice architecture.
SAVE: yes
LABEL: project
SUMMARY: Pengguna sedang mengerjakan project PaymentHub dengan arsitektur microservice.
NOTE: Pengguna memperkenalkan project aktif bernama PaymentHub.

CONTOH 6 — pengalaman kerja:
Input: [USER] Saya pernah 3 tahun kerja di startup fintech sebelum pindah ke perusahaan sekarang.
SAVE: yes
LABEL: pengalaman
SUMMARY: Pengguna memiliki pengalaman 3 tahun di startup fintech sebelum posisi saat ini.
NOTE: Pengguna menceritakan riwayat karir di startup fintech.

CONTOH 7 — perintah kamera (tidak disimpan):
Input: [USER] Foto layar laptop saya buat dokumentasi.
SAVE: no
LABEL: tidak_penting
SUMMARY:
NOTE:

Label: data_pribadi | pekerjaan | project | teknis | preferensi | pengalaman | lainnya | tidak_penting

Jawab HANYA dengan format ini (4 baris, tidak ada yang lain):
SAVE: yes atau no
LABEL: salah satu label di atas
SUMMARY: kalimat ringkasan (kosong jika no)
NOTE: catatan harian singkat (kosong jika no)
"""


# ══════════════════════════════════════════════════════════════════
# CPU SUMMARIZE PATH — prompt baru v6.4.0
# ══════════════════════════════════════════════════════════════════

# Digunakan di Jalur 2 (direct save) CPU path.
# Sebelumnya Jalur 2 langsung simpan _extract_user_texts() mentah → noise.
# Sekarang SLM diminta tulis ulang jadi kalimat padat sebelum masuk ChromaDB.
_SLM_SUMMARIZE_PROMPT = """Tulis ulang teks berikut menjadi 1-2 kalimat padat third-person
untuk disimpan ke memori jangka panjang AI assistant.

Aturan:
- Gunakan nama asli user jika disebutkan, atau gunakan kata "pengguna"
- JANGAN gunakan "[USER]" sebagai subjek
- Sertakan fakta spesifik: nama, angka, teknologi, waktu, nama project
- Buang filler word, sapaan, dan kata tidak penting
- Maksimal 150 karakter

Contoh input  : Nama saya Andi, saya frontend developer pakai React dan TypeScript.
Contoh output : Andi adalah frontend developer yang menggunakan React dan TypeScript.

Contoh input  : Saya prefer kerja mulai jam 7 pagi, lebih fokus sebelum orang lain online.
Contoh output : Pengguna prefer mulai kerja jam 7 pagi karena lebih fokus sebelum jam ramai.

Teks yang akan diproses:
{text}

Tulis ulang (langsung tanpa penjelasan):"""


def _init_gpu_cache() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

_GPU_AVAILABLE: bool = _init_gpu_cache()


def _detect_gpu() -> bool:
    """Return True jika CUDA tersedia. Hasil di-cache — tidak berubah di runtime."""
    return _GPU_AVAILABLE


def _parse_slm_direct_response(raw: str) -> dict:
    """Parse output GPU SLM direct path (4 baris: SAVE/LABEL/SUMMARY/NOTE)."""
    result = {"should_save": False, "label": "", "summary": "", "daily_note": ""}
    _EMPTY = {"", "kosong", "-", "none", "n/a", "tidak ada", "empty"}

    for line in raw.splitlines():
        line = line.strip()
        low  = line.lower()
        if low.startswith("save:"):
            val = line[5:].strip().lower()
            result["should_save"] = val.startswith("y") or val.startswith("iya")
        elif low.startswith("label:"):
            val = line[6:].strip().lower().replace(" ", "_")
            if val in _LABEL_IMPORTANCE:
                result["label"] = val
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

    Digunakan sebagai FALLBACK jika SLM summarize gagal.
    Output ini adalah teks mentah — lebih baik dihindari sebagai isi ChromaDB
    karena mengandung noise. Gunakan _call_slm_summarize() untuk versi bersihnya.
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
        total += len(content) + 3
    return " | ".join(parts)


def _make_fingerprint(messages: list[dict]) -> str:
    parts = []
    for m in messages[-5:]:
        content = (m.get("content") or "")[:80]
        parts.append(f"{m.get('role','?')}:{content}")
    return "|".join(parts)


# Threshold confidence minimum untuk direct save tanpa SLM check.
# Di bawah nilai ini → redirect ke SLM meskipun label sudah di _DIRECT_SAVE_LABELS.
_DIRECT_SAVE_MIN_CONFIDENCE = 0.60
_SLM_MIN_CONFIDENCE         = 0.50


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
        self._use_gpu_path     = _detect_gpu()

        if self._use_gpu_path:
            _log("GPU detected — menggunakan SLM direct path (skip cosine)")
        else:
            _log("GPU tidak tersedia — menggunakan CPU path (cosine/NLI classifier)")

        self._lock             = threading.Lock()
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

    def set_use_gpu(self, use_gpu: bool, device: Optional[str] = None):
        with self._lock:
            self._use_gpu_path = use_gpu and _detect_gpu()
            self._classifier   = None
            if device is not None:
                self._device = device
        _log(
            f"GPU path: {'ON (SLM direct)' if self._use_gpu_path else 'OFF (cosine/NLI classifier)'}"
            + (f" | device={self._device}" if device else "")
        )

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
        with self._lock:
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

        conv_user = _format_user_only(messages, MAX_CONV_CHARS)
        if not conv_user.strip():
            _log("Empty conversation, skipping", level="DEBUG", debug=self._debug)
            self._fire_callback("", False)
            return

        # ── Step [1.5]: recall question filter ────────────────────
        if _is_recall_question(messages):
            _log("Recall question detected → skip save (step 1.5)")
            self._fire_callback("tidak_penting", False)
            return

        # ── Step [1.6]: utility command filter ───────────────────
        if _is_utility_command(messages):
            _log("Utility command detected → skip save (step 1.6)")
            self._fire_callback("tidak_penting", False)
            return

        # ── Step [2]: GPU path → SLM direct, CPU path → cosine/NLI ─
        if self._use_gpu_path:
            conv_full    = _format_conversation(messages, MAX_CONV_CHARS)
            raw_response = self._call_slm_direct(conv_full)
            if not raw_response:
                _log("GPU SLM returned empty response", level="WARN")
                self._fire_callback("", False)
                return

            _log(f"GPU SLM raw:\n{raw_response}", level="DEBUG", debug=self._debug)

            parsed      = _parse_slm_direct_response(raw_response)
            should_save = parsed["should_save"]
            label       = parsed["label"] or "lainnya"
            summary     = parsed["summary"]
            daily_note  = parsed["daily_note"]

            _log(f"GPU Step[2] label={label} save={should_save}")

            if daily_note:
                self._save_daily(daily_note)
            if should_save and summary:
                self._save_episodic(label, summary, messages)
            elif should_save and not summary:
                _log("GPU SLM: SAVE=yes tapi summary kosong, fallback ke raw text", level="WARN")
                fallback = _extract_user_texts(messages)[:150]
                self._save_episodic(label, fallback, messages)
            else:
                _log(f"GPU SLM: not saved (label={label})", level="DEBUG", debug=self._debug)

            elapsed = (time.perf_counter() - t0) * 1000
            _log(f"Done {elapsed:.0f}ms | label={label} | saved={should_save} (GPU direct)")
            self._fire_callback(label, should_save)
            return

        # ── CPU path: cosine/NLI classifier → label + confidence ──
        classifier = self._get_classifier()
        if classifier is None:
            _log("Classifier gagal load (model None) → fallback SLM check", level="WARN")
            conv_full    = _format_conversation(messages, MAX_CONV_CHARS)
            raw_response = self._call_slm(conv_full, "lainnya")
            if not raw_response:
                self._fire_callback("lainnya", False)
                return
            parsed      = _parse_slm_response(raw_response)
            should_save = parsed["should_save"]
            daily_note  = parsed["daily_note"]
            summary     = _extract_user_texts(messages)[:150]
            if daily_note:
                self._save_daily(daily_note)
            if should_save:
                self._save_episodic("lainnya", summary, messages)
            self._fire_callback("lainnya", should_save)
            return

        if self._classifier_model == "st":
            label, confidence = _classify_with_st(conv_user, classifier)
        else:
            label, confidence = _classify_with_nli(conv_user, classifier)

        _log(f"CPU Step[2] label={label} confidence={confidence:.3f}")

        # Jalur 0: confidence terlalu rendah → DROP global
        if confidence <= _SLM_MIN_CONFIDENCE:
            _log(
                f"Label '{label}' conf={confidence:.3f} <= {_SLM_MIN_CONFIDENCE} "
                f"→ drop global (terlalu ambigu)"
            )
            self._fire_callback(label, False)
            return

        # Jalur 1: jelas tidak penting → DROP
        if label in _NO_SAVE_LABELS:
            _log(f"Label '{label}' → drop (no SLM)")
            self._fire_callback(label, False)
            return

        # ── Jalur 2: jelas penting + confidence cukup → SLM summarize dulu lalu SAVE ──
        # v6.4.0: sebelumnya langsung simpan _extract_user_texts() mentah.
        # Sekarang SLM diminta tulis ulang jadi kalimat padat sebelum masuk ChromaDB.
        # Ini meningkatkan kualitas teks → retrieval lebih akurat, noise berkurang.
        if label in _DIRECT_SAVE_LABELS:
            if confidence >= _DIRECT_SAVE_MIN_CONFIDENCE:
                _log(f"Label '{label}' conf={confidence:.3f} → SLM summarize then save")
                raw_text   = _extract_user_texts(messages)
                summary    = self._call_slm_summarize(raw_text)
                daily_note = f"[{label}] {summary[:80]}"
                self._save_daily(daily_note)
                self._save_episodic(label, summary, messages)
                elapsed = (time.perf_counter() - t0) * 1000
                _log(f"Done {elapsed:.0f}ms | label={label} | saved=True (summarized)")
                self._fire_callback(label, True)
                return
            else:
                _log(
                    f"Label '{label}' conf={confidence:.3f} < {_DIRECT_SAVE_MIN_CONFIDENCE} "
                    f"→ redirect ke SLM check"
                )

        # Jalur 3: ambigu ATAU confidence rendah → tanya SLM
        _log(f"Label '{label}' → SLM check", level="DEBUG", debug=self._debug)
        conv_full    = _format_conversation(messages, MAX_CONV_CHARS)
        raw_response = self._call_slm(conv_full, label)
        if not raw_response:
            _log("SLM returned empty response", level="WARN")
            self._fire_callback(label, False)
            return

        _log(f"SLM raw:\n{raw_response}", level="DEBUG", debug=self._debug)

        parsed      = _parse_slm_response(raw_response)
        should_save = parsed["should_save"]
        daily_note  = parsed["daily_note"]
        # Jalur 3 juga pakai SLM summarize jika perlu simpan
        if should_save:
            raw_text = _extract_user_texts(messages)
            summary  = self._call_slm_summarize(raw_text)
        else:
            summary = ""

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

    # ── SLM Direct Call (GPU path) ────────────────────────────────

    def _call_slm_direct(self, conversation: str) -> str:
        """GPU path: SLM classify + summarize sekaligus, tanpa label hint."""
        prompt = _SLM_DIRECT_PROMPT.replace("{conversation}", conversation)
        for attempt in range(2):
            try:
                resp = requests.post(
                    f"{self._llm_cfg['model_server']}/chat/completions",
                    headers={"Authorization": f"Bearer {self._llm_cfg.get('api_key', 'lm-studio')}"},
                    json={
                        "model"      : self._llm_cfg["model"],
                        "messages"   : [{"role": "user", "content": prompt}],
                        "max_tokens" : 200,    # v6.4.0: naik dari 150 → 200
                        "temperature": 0.1,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                _log(f"SLM direct call failed (attempt {attempt + 1}): {e}", level="ERROR")
        return ""

    # ── SLM Summarize Call (CPU Jalur 2 & 3, v6.4.0) ─────────────

    def _call_slm_summarize(self, raw_text: str) -> str:
        """
        CPU path: minta SLM tulis ulang raw_text menjadi kalimat padat third-person
        sebelum disimpan ke ChromaDB.

        Fallback ke raw_text[:150] jika SLM gagal atau timeout.
        Timeout pendek (45 detik) karena ini blocking call di background thread.
        """
        if not raw_text.strip():
            return raw_text[:150]

        prompt = _SLM_SUMMARIZE_PROMPT.replace("{text}", raw_text[:500])
        try:
            resp = requests.post(
                f"{self._llm_cfg['model_server']}/chat/completions",
                headers={"Authorization": f"Bearer {self._llm_cfg.get('api_key', 'lm-studio')}"},
                json={
                    "model"      : self._llm_cfg["model"],
                    "messages"   : [{"role": "user", "content": prompt}],
                    "max_tokens" : 100,
                    "temperature": 0.1,
                },
                timeout=45,
            )
            resp.raise_for_status()
            result = resp.json()["choices"][0]["message"]["content"].strip()
            # Buang tanda kutip jika model membungkus output
            result = result.strip("\"'")
            if len(result) > 10:
                _log(f"SLM summarize: '{result[:80]}'")
                return result[:200]
        except Exception as e:
            _log(f"SLM summarize gagal ({e}), fallback ke raw text", level="WARN")

        return raw_text[:150]

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
        summary = summary.strip()
        summary = re.sub(r"^\[?USER\]?\s*", "", summary)  # ← tambah ini
        if not summary:
            _log("Summary kosong, pakai fallback snippet", level="WARN")
            parts = []
            for msg in raw_msgs[-4:]:
                if msg.get("role") != "user":
                    continue
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