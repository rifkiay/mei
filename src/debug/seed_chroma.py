"""
seed_chroma.py — Seeder data episodik untuk validasi Chroma Test
================================================================
Script ini mengisi ChromaDB dengan fakta-fakta yang dibutuhkan oleh
CHROMA_TEST_QUERIES (10 query) di main_ui.py.

Jalankan SEKALI sebelum 'test chroma' untuk memastikan retrieval pipeline
bisa diuji secara independen dari ingestion pipeline (FactExtractor).

Fakta yang di-seed:
  q01 → standup Senin jam 8
  q02 → Docker + Ubuntu
  q03 → Pomodoro 25 menit
  q04 → project Kong API Gateway
  q05 → meeting klien Kamis jam 2
  q06 → demo klien Jumat jam 3
  q07 → 4 tahun pengalaman backend
  q08 → preferensi meeting jam 8-9 pagi
  q09 → kerja di DevStudio sebagai backend developer
  q10 → stack Express + PostgreSQL

Usage:
  cd e:/skripsi pendekatan openclaw/agent_qwen/src
  python seed_chroma.py
"""

import sys
import os

os.chdir(r'e:/skripsi pendekatan openclaw/agent_qwen/src')
sys.path.insert(0, r'e:/skripsi pendekatan openclaw/agent_qwen/src')

from datetime import date
from config.agent_config import MEMORY_CONFIG
from memory.long_term_memory import LongTermMemoryManager

# ── Init LongTermMemoryManager (sama persis dengan main_ui.py) ──────
em = MEMORY_CONFIG['embedding_model']
cn = MEMORY_CONFIG['collection_name']
sd = MEMORY_CONFIG['storage_dir']

print("=" * 60)
print("  ChromaDB Seeder — MEI episodic memory")
print("=" * 60)
print(f"  embedding_model : {em}")
print(f"  collection_name : {cn}")
print(f"  storage_dir     : {sd}")
print()

lt_mem = LongTermMemoryManager(
    storage_dir     = sd,
    embedding_model = em,
    collection_name = cn,
    top_k           = 5,
    device          = 'cpu',
)

count_before = lt_mem.count()
print(f"  Count sebelum seed : {count_before}")
print()

# ── Data seed — match persis dengan CHROMA_TEST_QUERIES ─────────────
# Format content: "[YYYY-MM-DD] <fakta>"
# Label & importance harus konsisten dengan _LABEL_IMPORTANCE di fact_extractor.py

SEED_DATE = "2026-05-01"  # tanggal "percakapan seeding" simulasi

SEED_DATA = [
    # q01 — jadwal standup
    {
        "content"    : f"[{SEED_DATE}] Rifki punya standup rutin setiap hari Senin jam 8 pagi bersama tim engineering.",
        "memory_type": "episodic",
        "importance" : 8,
        "metadata"   : {"label": "pekerjaan", "user_id": "rifki", "seed": True},
    },
    # q02 — environment development
    {
        "content"    : f"[{SEED_DATE}] Environment development Rifki menggunakan Docker dan Ubuntu sebagai OS utama.",
        "memory_type": "episodic",
        "importance" : 7,
        "metadata"   : {"label": "teknis", "user_id": "rifki", "seed": True},
    },
    # q03 — metode coding Pomodoro
    {
        "content"    : f"[{SEED_DATE}] Rifki menggunakan metode Pomodoro untuk sesi coding: 25 menit fokus, 5 menit istirahat.",
        "memory_type": "episodic",
        "importance" : 7,
        "metadata"   : {"label": "preferensi", "user_id": "rifki", "seed": True},
    },
    # q04 — project Kong API Gateway
    {
        "content"    : f"[{SEED_DATE}] Rifki sedang mengerjakan project integrasi Kong API Gateway untuk manajemen microservices.",
        "memory_type": "episodic",
        "importance" : 8,
        "metadata"   : {"label": "project", "user_id": "rifki", "seed": True},
    },
    # q05 — meeting klien Kamis jam 2
    {
        "content"    : f"[{SEED_DATE}] Jadwal meeting rutin dengan klien adalah setiap Kamis jam 2 siang.",
        "memory_type": "episodic",
        "importance" : 8,
        "metadata"   : {"label": "pekerjaan", "user_id": "rifki", "seed": True},
    },
    # q06 — demo klien Jumat jam 3
    {
        "content"    : f"[{SEED_DATE}] Rifki ada jadwal demo ke klien setiap Jumat jam 3 sore untuk review sprint.",
        "memory_type": "episodic",
        "importance" : 8,
        "metadata"   : {"label": "pekerjaan", "user_id": "rifki", "seed": True},
    },
    # q07 — pengalaman 4 tahun backend
    {
        "content"    : f"[{SEED_DATE}] Rifki punya pengalaman 4 tahun di web development, dengan fokus utama sebagai backend developer.",
        "memory_type": "episodic",
        "importance" : 6,
        "metadata"   : {"label": "pengalaman", "user_id": "rifki", "seed": True},
    },
    # q08 — preferensi meeting jam 8-9 pagi
    {
        "content"    : f"[{SEED_DATE}] Rifki prefer meeting di pagi hari, idealnya mulai jam 8 atau jam 9, sebelum masuk sesi coding.",
        "memory_type": "episodic",
        "importance" : 7,
        "metadata"   : {"label": "preferensi", "user_id": "rifki", "seed": True},
    },
    # q09 — kerja di DevStudio
    {
        "content"    : f"[{SEED_DATE}] Rifki saat ini bekerja di DevStudio sebagai backend developer, sudah 2 tahun di sana.",
        "memory_type": "episodic",
        "importance" : 9,
        "metadata"   : {"label": "data_pribadi", "user_id": "rifki", "seed": True},
    },
    # q10 — stack Express + PostgreSQL
    {
        "content"    : f"[{SEED_DATE}] Stack teknologi backend yang dipakai Rifki di project-nya: Express.js dan PostgreSQL.",
        "memory_type": "episodic",
        "importance" : 7,
        "metadata"   : {"label": "teknis", "user_id": "rifki", "seed": True},
    },
]

# ── Insert ───────────────────────────────────────────────────────────
print("  Inserting seed data...")
for i, item in enumerate(SEED_DATA, 1):
    lt_mem.add_memory(
        content     = item["content"],
        memory_type = item["memory_type"],
        importance  = item["importance"],
        metadata    = item["metadata"],
    )
    label = item["metadata"]["label"]
    print(f"  [{i:02d}] label={label:<12} | {item['content'][:70]}")

count_after = lt_mem.count()
print()
print(f"  Count setelah seed : {count_after} (+{count_after - count_before})")
print()

# ── Verifikasi retrieval untuk semua 10 query ────────────────────────
print("=" * 60)
print("  Verifikasi retrieval (quick check)")
print("=" * 60)

TEST_QUERIES = [
    ("q01", "Kamu masih ingat jadwal standup rutin saya itu hari apa dan jam berapa?",   ["Senin", "jam 8"]),
    ("q02", "Kamu ingat teknologi apa yang saya pakai untuk environment development saya?", ["Docker", "Ubuntu"]),
    ("q03", "Saya pernah bilang soal metode kerja saya buat sesi coding. Apa itu?",       ["Pomodoro", "25 menit"]),
    ("q04", "Saya pernah cerita soal project yang lagi saya kerjakan. Itu tentang apa?",  ["Kong", "API Gateway"]),
    ("q05", "Ingatkan saya, jadwal meeting rutin dengan klien itu hari apa dan jam berapa?", ["Kamis", "jam 2"]),
    ("q06", "Kamu masih ingat jadwal demo saya ke klien itu kapan?",                      ["Jumat", "jam 3"]),
    ("q07", "Berapa tahun pengalaman saya di web development yang sudah saya ceritakan dulu?", ["4 tahun", "backend"]),
    ("q08", "Kamu ingat preferensi jam meeting saya? Saya biasanya prefer mulai jam berapa?", ["pagi", "jam 8", "jam 9"]),
    ("q09", "Saya pernah cerita soal pekerjaan dan tempat kerja saya yang sekarang. Apa itu?", ["DevStudio", "backend developer"]),
    ("q10", "Kamu ingat stack teknologi backend yang saya pakai di project saya?",         ["Express", "PostgreSQL"]),
]

passed = 0
for qid, query, expected_kws in TEST_QUERIES:
    hits = lt_mem.search(query=query, n_results=3)
    top_content = hits[0]["content"] if hits else ""
    top_relevance = round(1 - hits[0]["distance"], 3) if hits else 0.0

    kw_found = [kw for kw in expected_kws if kw.lower() in top_content.lower()]
    ok = len(kw_found) > 0

    status = "[PASS]" if ok else "[FAIL]"
    if ok:
        passed += 1

    print(f"  [{qid}] {status}")
    print(f"         query   : {query[:65]}")
    print(f"         top hit : rel={top_relevance} | {top_content[:70]}")
    print(f"         kw found: {kw_found} / {expected_kws}")
    print()

print("=" * 60)
print(f"  Hasil: {passed}/{len(TEST_QUERIES)} query berhasil retrieve")
if passed == len(TEST_QUERIES):
    print("  [OK] Semua query berhasil! Sekarang jalankan 'test chroma' di main_ui.")
else:
    print("  [ERROR] Ada query yang gagal — cek konten seed atau embedding model.")
print("=" * 60)
