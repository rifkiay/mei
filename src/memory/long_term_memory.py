"""
Long-Term Memory Manager — MEI v4.2.0
=======================================
Perubahan dari v4.1.0:

  - ADD: BM25 hybrid retrieval di search()
    Sebelumnya search() hanya cosine similarity via ChromaDB.
    Sekarang hasil cosine di-fuse dengan BM25 (keyword exact match)
    menggunakan Reciprocal Rank Fusion (RRF).
    
    Impact: query dengan keyword spesifik (nama project, nama teknologi,
    nama orang) yang mungkin miss di cosine sekarang lebih mudah ter-retrieve.
    Noise dokumen yang relevan secara semantik tapi tidak punya keyword
    penting akan turun rankingnya.

    Library: rank_bm25 (pure Python, ringan, tidak butuh GPU).
    Install: pip install rank_bm25
    Fallback: jika rank_bm25 tidak tersedia, otomatis fallback ke
    cosine-only seperti v4.1.0 (tidak ada breaking change).

  - ADD: _bm25_tokenize() — tokenizer sederhana untuk BM25
    Lowercase + split + buang stopword Bahasa Indonesia yang umum.
    Tidak pakai NLTK/spaCy agar tetap ringan.

  - MOD: search() — parameter baru `use_bm25` (default True)
    Kalau False: perilaku identik dengan v4.1.0 (cosine + MMR only).
    Kalau True: fetch lebih banyak kandidat cosine, fuse dengan BM25,
    baru di-MMR.

  - KEEP: Semua logika v4.1.0 lainnya tidak berubah.
    (daily notes, compress, add_memory, stats, search_episodic_by_period)

  - KEEP: MMR tetap berjalan setelah BM25 fusion sebagai final
    diversification step.
"""

import re
import requests
from datetime import datetime, date, timedelta
from pathlib import Path

import chromadb
from chromadb.config import Settings


# ── Singleton embedding model ──────────────────────────────────────

_EMBEDDING_MODEL_CACHE: dict = {}


def _get_or_load_sentence_transformer(model_name: str, device: str):
    key = (model_name, device)
    if key not in _EMBEDDING_MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        print(f"  [LTM] Loading embedding model: {model_name} | device={device}")
        _EMBEDDING_MODEL_CACHE[key] = SentenceTransformer(model_name, device=device)
        print(f"  [LTM] Embedding model ready")
    return _EMBEDDING_MODEL_CACHE[key]


class _CachedEmbeddingFunction:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device     = device

    def name(self) -> str:
        return "sentence_transformer"

    def _model(self):
        return _get_or_load_sentence_transformer(self.model_name, self.device)

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = self._model().encode(
            input, convert_to_numpy=True, show_progress_bar=False,
        )
        return embeddings.tolist()

    # ChromaDB 1.5.5+ compatibility
    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)


DAILY_COMPRESS_EVERY = 3

_RE_AUTO_LINE    = re.compile(r"^\s*-\s+\d{2}:\d{2}\s+\[auto\]")
_RE_SUMMARY_LINE = re.compile(r"^\s*-\s+\d{2}:\d{2}\s+\[summary\]")


# ══════════════════════════════════════════════════════════════════
# BM25 HELPERS (v4.2.0)
# ══════════════════════════════════════════════════════════════════

# Stopword Bahasa Indonesia yang umum muncul di daily conversation.
# Tidak perlu lengkap — cukup buang kata yang tidak informatif untuk retrieval.
_BM25_STOPWORDS = {
    "dan", "di", "ke", "yang", "dengan", "untuk", "dari", "pada", "adalah",
    "ini", "itu", "ada", "tidak", "ya", "atau", "juga", "sudah", "saya",
    "aku", "gue", "gw", "kamu", "kau", "lo", "kami", "kita", "mereka",
    "dia", "beliau", "si", "sang", "para", "akan", "bisa", "bisa", "jadi",
    "kalau", "karena", "tapi", "namun", "mau", "ingin", "perlu", "harus",
    "the", "a", "an", "is", "in", "of", "to", "for", "and", "or", "that",
    "ini", "itu", "nya", "lah", "pun", "pula", "sih", "nih", "deh", "dong",
    "user", "rifki",  # nama user — terlalu umum di semua dokumen
}


def _bm25_tokenize(text: str) -> list[str]:
    """
    Tokenizer sederhana untuk BM25.
    Lowercase → split whitespace/punctuation → buang stopword → buang token pendek.
    
    Sengaja dibuat ringan tanpa NLTK/spaCy agar tidak menambah dependency.
    """
    text   = text.lower()
    tokens = re.findall(r"[a-z0-9][a-z0-9\-_\.]*[a-z0-9]|[a-z0-9]", text)
    return [t for t in tokens if t not in _BM25_STOPWORDS and len(t) >= 2]


def _bm25_score_docs(query: str, docs: list[str]) -> list[float]:
    """
    Hitung BM25 score untuk tiap dokumen terhadap query.
    Return list scores dengan panjang sama seperti docs.
    
    Fallback ke semua-nol jika rank_bm25 tidak tersedia.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return [0.0] * len(docs)

    tokenized_docs = [_bm25_tokenize(d) for d in docs]
    tokenized_query = _bm25_tokenize(query)

    if not tokenized_query or not any(tokenized_docs):
        return [0.0] * len(docs)

    bm25   = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenized_query)
    return scores.tolist()


def _reciprocal_rank_fusion(
    cosine_distances : list[float],
    bm25_scores      : list[float],
    k                : int   = 60,
    bm25_weight      : float = 0.3,
) -> list[float]:
    """
    Reciprocal Rank Fusion (RRF) untuk gabung cosine similarity + BM25.

    Formula per dokumen:
        rrf_score = (1 / (rank_cosine + k)) + bm25_weight * (1 / (rank_bm25 + k))

    Parameter:
        k           — konstanta smoothing RRF. Default 60 sesuai paper asli.
        bm25_weight — bobot relatif BM25 vs cosine. Default 0.3 (70% cosine,
                      30% BM25) karena cosine embedding lebih kaya semantik,
                      BM25 hanya sebagai keyword booster.
    
    Return: list RRF scores (lebih tinggi = lebih baik).
    """
    n = len(cosine_distances)
    if n == 0:
        return []

    # Rank cosine: distance kecil = lebih relevan = rank rendah (dimulai 1)
    cosine_order = sorted(range(n), key=lambda i: cosine_distances[i])
    cosine_rank  = [0] * n
    for rank, idx in enumerate(cosine_order):
        cosine_rank[idx] = rank + 1

    # Rank BM25: score besar = lebih relevan = rank rendah
    bm25_order = sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)
    bm25_rank  = [0] * n
    for rank, idx in enumerate(bm25_order):
        bm25_rank[idx] = rank + 1

    # Fusi
    rrf_scores = [
        (1.0 / (cosine_rank[i] + k))
        + bm25_weight * (1.0 / (bm25_rank[i] + k))
        for i in range(n)
    ]
    return rrf_scores


class LongTermMemoryManager:

    def __init__(
        self,
        storage_dir    : str = "./../storage/memory",
        embedding_model: str = "default",
        collection_name: str = "rifki_episodic",
        top_k          : int = 5,
        device         : str = "cuda",
    ):
        self.storage_dir      = Path(storage_dir)
        self.memory_md_path   = self.storage_dir / "MEMORY.md"
        self.daily_dir        = self.storage_dir / "daily"
        self.chroma_dir       = self.storage_dir / "chroma"
        self.top_k            = top_k
        self.device           = device
        self._embedding_model = embedding_model
        self._collection_name = collection_name

        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        self._chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._ef         = self._make_ef(embedding_model, device)
        self._collection = self._chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

        # Cek apakah rank_bm25 tersedia — log sekali saat init
        try:
            import rank_bm25  # noqa
            self._bm25_available = True
            print("  [LTM] BM25 hybrid: rank_bm25 tersedia ✓")
        except ImportError:
            self._bm25_available = False
            print("  [LTM] BM25 hybrid: rank_bm25 tidak tersedia, fallback cosine-only")
            print("  [LTM] Install dengan: pip install rank_bm25")

    # ──────────────────────────────────────
    # EMBEDDING
    # ──────────────────────────────────────

    @staticmethod
    def _make_ef(model_name: str, device: str):
        if model_name == "default":
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            if device == "cuda":
                print("  [LTM] 'default' embedding tidak support CUDA.")
            return DefaultEmbeddingFunction()
        return _CachedEmbeddingFunction(model_name=model_name, device=device)

    def reinit_embedding(self, device: str):
        if self.device == device:
            return
        self.device  = device
        self._ef     = self._make_ef(self._embedding_model, device)
        self._collection = self._chroma_client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"  [LTM] Embedding re-init: device={device}")
        _clear_gpu_cache()

    # ──────────────────────────────────────
    # 1. MEMORY.md (READ ONLY untuk AI)
    # ──────────────────────────────────────

    def read_memory_md(self) -> str:
        if not self.memory_md_path.exists():
            return ""
        return self.memory_md_path.read_text(encoding="utf-8")

    # ──────────────────────────────────────
    # 2. DAILY NOTES
    # ──────────────────────────────────────

    def _daily_path(self, for_date: date = None) -> Path:
        return self.daily_dir / f"{(for_date or date.today()).isoformat()}.md"

    def append_daily(self, content: str, tag: str = ""):
        p             = self._daily_path()
        header_needed = not p.exists()
        with open(p, "a", encoding="utf-8") as f:
            if header_needed:
                f.write(f"# Daily Notes — {date.today().isoformat()}\n\n")
            tag_str = f"[{tag}] " if tag else ""
            f.write(f"- {datetime.now().strftime('%H:%M')} {tag_str}{content.strip()}\n")

    def read_daily(self, for_date: date = None) -> str:
        p = self._daily_path(for_date)
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def read_recent_daily(self, days: int = 1) -> str:
        result = []
        for i in range(days):
            text = self.read_daily(date.today() - timedelta(days=i))
            if text:
                result.append(text)
        return "\n\n".join(result)

    def read_last_daily_summary(self, max_chars: int = 400) -> str:
        _RE_SUMMARY_START = re.compile(r'^\s*-\s+\d{2}:\d{2}\s+\[summary\]')
        _RE_NEW_ENTRY     = re.compile(r'^\s*-\s+\d{2}:\d{2}')

        for delta_days in (0, 1):
            text = self.read_daily(date.today() - timedelta(days=delta_days))
            if not text:
                continue

            summary_blocks: list[list[str]] = []
            current_block: list[str] | None = None

            for line in text.splitlines():
                if _RE_SUMMARY_START.match(line):
                    if current_block is not None:
                        summary_blocks.append(current_block)
                    current_block = [line]
                elif current_block is not None:
                    if _RE_NEW_ENTRY.match(line):
                        summary_blocks.append(current_block)
                        current_block = None
                    else:
                        current_block.append(line)

            if current_block is not None:
                summary_blocks.append(current_block)

            if summary_blocks:
                last = "\n".join(summary_blocks[-1]).strip()
                return last[:max_chars]

        return ""

    def write_daily_summary(self, summary: str):
        self.append_daily(summary, tag="summary")

    # ──────────────────────────────────────
    # 2b. DAILY COMPRESS
    # ──────────────────────────────────────

    def append_daily_and_maybe_compress(
        self,
        text   : str,
        tag    : str       = "auto",
        llm_cfg: dict|None = None,
    ) -> None:
        self.append_daily(text, tag=tag)

        if not llm_cfg:
            return

        daily_path = self._daily_path()
        if not daily_path.exists():
            return

        content   = daily_path.read_text(encoding="utf-8")
        all_lines = content.splitlines()

        # ── Kompres [auto] setiap DAILY_COMPRESS_EVERY entri ────────────
        auto_lines = [l for l in all_lines if _RE_AUTO_LINE.match(l)]
        if len(auto_lines) >= DAILY_COMPRESS_EVERY:
            raw_text = "\n".join(auto_lines)
            prompt = (
                "Ringkas entri-entri log percakapan berikut menjadi 2-3 kalimat "
                "padat dalam Bahasa Indonesia. Pertahankan topik utama dan keputusan "
                "penting. Jangan tambahkan penjelasan atau header, langsung "
                "ringkasan saja.\n\n"
                f"{raw_text}"
            )
            try:
                resp = requests.post(
                    f"{llm_cfg['model_server']}/chat/completions",
                    headers={"Authorization": f"Bearer {llm_cfg.get('api_key', 'lm-studio')}"},
                    json={
                        "model"      : llm_cfg["model"],
                        "messages"   : [{"role": "user", "content": prompt}],
                        "max_tokens" : 200,
                        "temperature": 0.2,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                summary = resp.json()["choices"][0]["message"]["content"].strip()
                now = datetime.now().strftime("%H:%M")
                summary_line = f"- {now} [summary] {summary}"
                new_lines = []
                inserted  = False
                for line in all_lines:
                    if _RE_AUTO_LINE.match(line):
                        if not inserted:
                            new_lines.append(summary_line)
                            inserted = True
                    else:
                        new_lines.append(line)
                daily_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                print(f"  [LTM] Daily compressed: {len(auto_lines)} entri [auto] → 1 [summary]")
                content   = daily_path.read_text(encoding="utf-8")
                all_lines = content.splitlines()
            except Exception as e:
                print(f"  [LTM][WARN] Daily compress LLM gagal: {e}")

        # ── Kompres [summary] setiap SUMMARY_COMPRESS_EVERY ─────────────
        SUMMARY_COMPRESS_EVERY = 3
        _RE_NEW_ENTRY = re.compile(r'^\s*-\s+\d{2}:\d{2}')

        summary_blocks  = []
        summary_indices = set()
        current_block   = None

        for i, line in enumerate(all_lines):
            if _RE_SUMMARY_LINE.match(line):
                if current_block is not None:
                    summary_blocks.append(current_block)
                current_block = {"lines": [line], "idx": [i]}
            elif current_block is not None:
                if _RE_NEW_ENTRY.match(line):
                    summary_blocks.append(current_block)
                    current_block = None
                else:
                    current_block["lines"].append(line)
                    current_block["idx"].append(i)

        if current_block is not None:
            summary_blocks.append(current_block)

        for block in summary_blocks:
            summary_indices.update(block["idx"])

        if len(summary_blocks) >= SUMMARY_COMPRESS_EVERY:
            raw_text = "\n\n".join(
                "\n".join(b["lines"]).strip() for b in summary_blocks
            )
            prompt = (
                "Gabungkan ringkasan-ringkasan percakapan berikut menjadi 1 "
                "ringkasan padat 2-3 kalimat dalam Bahasa Indonesia. "
                "Pertahankan semua poin penting dari semua sumber. "
                "Langsung tulis ringkasan saja tanpa header.\n\n"
                f"{raw_text}"
            )
            try:
                resp = requests.post(
                    f"{llm_cfg['model_server']}/chat/completions",
                    headers={"Authorization": f"Bearer {llm_cfg.get('api_key', 'lm-studio')}"},
                    json={
                        "model"      : llm_cfg["model"],
                        "messages"   : [{"role": "user", "content": prompt}],
                        "max_tokens" : 150,
                        "temperature": 0.2,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                merged = resp.json()["choices"][0]["message"]["content"].strip()
                new_lines = [l for i, l in enumerate(all_lines) if i not in summary_indices]
                daily_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                self.append_daily(merged, tag="summary")
                print(f"  [LTM] Summary compressed: {len(summary_blocks)} [summary] → 1")
            except Exception as e:
                print(f"  [LTM][WARN] Summary compress LLM gagal: {e}")

    # ──────────────────────────────────────
    # 3. CHROMADB
    # ──────────────────────────────────────

    def add_memory(
        self,
        content    : str,
        memory_type: str  = "fact",
        importance : int  = 5,
        metadata   : dict = None,
    ):
        import hashlib
        uid = (
            datetime.now().strftime("%Y%m%d%H%M%S%f")
            + "_" + hashlib.md5(content.encode()).hexdigest()[:8]
        )
        meta = {
            "memory_type": memory_type,
            "importance" : importance,
            "date"       : date.today().isoformat(),
            "timestamp"  : datetime.now().isoformat(timespec="seconds"),
        }
        if metadata:
            meta.update(metadata)
        self._collection.add(ids=[uid], documents=[content], metadatas=[meta])

    def search(
        self,
        query         : str,
        n_results     : int   = None,
        memory_type   : str   = None,
        min_importance: int   = None,
        label         : str   = None,
        date_from     : str   = None,
        date_to       : str   = None,
        use_mmr       : bool  = True,
        mmr_lambda    : float = 0.6,
        use_bm25      : bool  = True,   # v4.2.0: BM25 hybrid toggle
        bm25_weight   : float = 0.3,    # v4.2.0: bobot BM25 vs cosine di RRF
    ) -> list[dict]:
        """
        Semantic search dengan BM25 hybrid + MMR re-ranking (v4.2.0).

        Pipeline:
          1. ChromaDB cosine fetch top (n * fetch_multiplier) kandidat
          2. Hitung BM25 score untuk kandidat yang sama
          3. Fusi via Reciprocal Rank Fusion (RRF) → sorted by combined score
          4. MMR re-rank untuk diversifikasi dan anti-noise duplikat
          5. Return top-n

        Parameter baru v4.2.0:
          use_bm25    — aktifkan BM25 hybrid (default True). False = perilaku v4.1.0.
          bm25_weight — proporsi BM25 di RRF. 0.3 = 70% cosine, 30% BM25.
                        Naikkan ke 0.5 jika keyword exact match lebih penting.

        Backward compatible: kalau rank_bm25 tidak terinstall, otomatis
        fallback ke cosine-only tanpa error.
        """
        n     = n_results or self.top_k
        total = self._collection.count()
        if total == 0:
            return []

        conditions = []
        if memory_type:
            conditions.append({"memory_type": {"$eq": memory_type}})
        if min_importance is not None:
            conditions.append({"importance": {"$gte": min_importance}})
        if label:
            conditions.append({"label": {"$eq": label}})
        if date_from:
            conditions.append({"date": {"$gte": date_from}})
        if date_to:
            conditions.append({"date": {"$lte": date_to}})

        where = None
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        # ── Tentukan fetch_k ──────────────────────────────────────────
        # Ambil lebih banyak untuk BM25+MMR punya cukup kandidat.
        # GPU path (dipanggil dari main_ui.py dengan n_results = top_k * 4)
        # sudah pass n_results besar — tidak double-multiply.
        # CPU path juga sudah pass n_results dari main_ui.py.
        # Jika dipanggil langsung (e.g. test), multiply di sini.
        if use_bm25 and self._bm25_available:
            fetch_multiplier = 4 if use_mmr else 2
        elif use_mmr:
            fetch_multiplier = 3
        else:
            fetch_multiplier = 1

        fetch_k = min(n * fetch_multiplier, total)

        kwargs = {
            "query_texts" : [query],
            "n_results"   : fetch_k,
            "include"     : ["documents", "metadatas", "distances", "embeddings"],
        }
        if where:
            kwargs["where"] = where

        try:
            results = self._collection.query(**kwargs)
        except Exception:
            return []

        docs       = results["documents"][0]
        metas      = results["metadatas"][0]
        distances  = results["distances"][0]
        embeddings = results.get("embeddings", [[]])[0]

        if not docs:
            return []

        # ── BM25 + RRF fusion (v4.2.0) ───────────────────────────────
        # Hanya jalan kalau use_bm25=True AND rank_bm25 tersedia.
        # Jika tidak tersedia: lanjut ke MMR dengan urutan cosine biasa.
        if use_bm25 and self._bm25_available:
            bm25_scores = _bm25_score_docs(query, docs)
            rrf_scores  = _reciprocal_rank_fusion(
                cosine_distances = distances,
                bm25_scores      = bm25_scores,
                bm25_weight      = bm25_weight,
            )
            # Sort by RRF score descending
            sorted_idx = sorted(range(len(docs)), key=lambda i: rrf_scores[i], reverse=True)
            docs       = [docs[i]       for i in sorted_idx]
            metas      = [metas[i]      for i in sorted_idx]
            distances  = [distances[i]  for i in sorted_idx]
            embeddings = [embeddings[i] for i in sorted_idx] if embeddings is not None and len(embeddings) > 0 else embeddings

        # ── MMR re-ranking ────────────────────────────────────────────
        # Pilih n dokumen yang paling relevan DAN paling beragam.
        if use_mmr and embeddings is not None and len(embeddings) > 0 and len(docs) > n:
            import numpy as np

            def cos_sim(a, b):
                a, b = np.array(a), np.array(b)
                denom = (np.linalg.norm(a) * np.linalg.norm(b))
                return float(np.dot(a, b) / denom) if denom > 0 else 0.0

            relevances   = [1 - d for d in distances]
            selected_idx = []
            remaining    = list(range(len(docs)))

            while len(selected_idx) < n and remaining:
                if not selected_idx:
                    best = max(remaining, key=lambda i: relevances[i])
                else:
                    def mmr_score(i):
                        rel = relevances[i]
                        max_sim = max(
                            cos_sim(embeddings[i], embeddings[j])
                            for j in selected_idx
                        )
                        return mmr_lambda * rel - (1 - mmr_lambda) * max_sim
                    best = max(remaining, key=mmr_score)

                selected_idx.append(best)
                remaining.remove(best)

            docs      = [docs[i]      for i in selected_idx]
            metas     = [metas[i]     for i in selected_idx]
            distances = [distances[i] for i in selected_idx]
        else:
            docs      = docs[:n]
            metas     = metas[:n]
            distances = distances[:n]

        return [
            {
                "content"    : doc,
                "memory_type": meta.get("memory_type", "fact"),
                "label"      : meta.get("label", meta.get("memory_type", "episodic")),
                "importance" : meta.get("importance", 5),
                "date"       : meta.get("date", ""),
                "distance"   : round(dist, 4),
            }
            for doc, meta, dist in zip(docs, metas, distances)
        ]

    def search_episodic_by_period(
        self,
        query    : str,
        year     : int,
        month    : int,
        n_results: int = 5,
    ) -> list[dict]:
        import calendar
        last_day = calendar.monthrange(year, month)[1]
        return self.search(
            query       = query,
            n_results   = n_results,
            memory_type = "episodic",
            date_from   = f"{year:04d}-{month:02d}-01",
            date_to     = f"{year:04d}-{month:02d}-{last_day:02d}",
        )

    # ──────────────────────────────────────
    # STATS
    # ──────────────────────────────────────

    def count(self) -> int:
        return self._collection.count()

    def stats(self) -> dict:
        return {
            "total_memories": self.count(),
            "top_k"         : self.top_k,
            "chroma_dir"    : str(self.chroma_dir),
            "device"        : self.device,
            "bm25_available": self._bm25_available,
        }


# ── Utils ──────────────────────────────────────────────────────────

def _clear_gpu_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass