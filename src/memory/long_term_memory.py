"""
Long-Term Memory Manager — MEI v4.1.0
=======================================
Perubahan dari v4.0.0:
  - CHANGED: search() sekarang return field 'label' dari metadata
    (diisi oleh FactExtractor v6.0, untuk tampilan di system prompt dan episodic command)
  - search() tetap backward-compatible: kalau metadata tidak punya 'label',
    fallback ke memory_type
  - Tidak ada perubahan lain — semua logika daily notes, ChromaDB, compress tetap sama
"""

import requests
import re
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

        if not self.memory_md_path.exists():
            self.memory_md_path.write_text(_DEFAULT_MEMORY_MD, encoding="utf-8")

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
        n_results     : int  = None,
        memory_type   : str  = None,
        min_importance: int  = None,
        label         : str  = None,   # filter by label (dari FactExtractor v6)
        date_from     : str  = None,
        date_to       : str  = None,
    ) -> list[dict]:
        """
        Semantic search dengan optional filter.
        Return dict sekarang include field 'label' — fallback ke memory_type
        jika metadata tidak punya label (data lama sebelum v6).
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

        kwargs = {
            "query_texts": [query],
            "n_results"  : min(n, total),
        }
        if where:
            kwargs["where"] = where

        try:
            results = self._collection.query(**kwargs)
        except Exception:
            return []

        return [
            {
                "content"    : doc,
                "memory_type": meta.get("memory_type", "fact"),
                # 'label' dari FactExtractor v6, fallback ke memory_type untuk data lama
                "label"      : meta.get("label", meta.get("memory_type", "episodic")),
                "importance" : meta.get("importance", 5),
                "date"       : meta.get("date", ""),
                "distance"   : round(dist, 4),
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
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
        }


# ── Utils ──────────────────────────────────────────────────────────

def _clear_gpu_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ── Default MEMORY.md template ─────────────────────────────────────

_DEFAULT_MEMORY_MD = """\
# User Profile
---

## Identitas
- [2026-01-01] nama: Rifki
- [2026-01-01] lokasi: Bandung, Jawa Barat
- [2026-01-01] pekerjaan: Mahasiswa

---

## Notes / Memory (knowledge pribadi)

---

## Project Aktif

---
"""