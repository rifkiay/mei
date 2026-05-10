"""
Topic Suggester
================
Generate pesan proaktif yang relevan dengan konteks percakapan sebelumnya.
Menggunakan LLM (via endpoint yang sama dengan agent) untuk membuat
kalimat opening yang natural dan personal.
"""

import random
import requests
import json
from typing import Optional

from .proactive_config import PROACTIVE_CONFIG


class TopicSuggester:
    """
    Menghasilkan kalimat pembuka percakapan yang context-aware.

    Strategy:
    1. Ambil N pesan terakhir sebagai konteks
    2. Kirim ke LLM dengan prompt khusus untuk generate 1 kalimat pendek
    3. Fallback ke fallback_topics jika LLM gagal / timeout
    """

    def __init__(self, llm_config: dict):
        """
        Args:
            llm_config: dict config LLM yang sama dengan agent utama.
                        Butuh: 'model', 'model_server', 'api_key'
        """
        self.model = llm_config.get("model", "")
        self.server = llm_config.get("model_server", "").rstrip("/")
        self.api_key = llm_config.get("api_key", "")
        self.timeout = 8  # detik — harus cepat agar tidak nge-block loop

    # ──────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────

    def generate(
        self,
        conversation_history: list[dict],
        trigger_count: int = 1,
        user_name: str = "kamu",
    ) -> str:
        """
        Generate satu kalimat proaktif berdasarkan konteks percakapan.

        Args:
            conversation_history : list pesan {'role': ..., 'content': ...}
            trigger_count        : berapa kali sudah trigger (1 = pertama, 2 = retry)
            user_name            : nama user untuk personalisasi

        Returns:
            String kalimat proaktif yang siap ditampilkan
        """
        context_window = PROACTIVE_CONFIG["context_window_messages"]
        recent = conversation_history[-context_window:] if conversation_history else []

        if len(recent) < 2:
            return self._fallback()

        try:
            message = self._call_llm(recent, trigger_count, user_name)
            if message:
                return message
        except Exception:
            pass

        return self._fallback()

    # ──────────────────────────────────────────
    # PRIVATE HELPERS
    # ──────────────────────────────────────────

    def _call_llm(
        self,
        recent_messages: list[dict],
        trigger_count: int,
        user_name: str,
    ) -> Optional[str]:
        """Panggil LLM untuk generate 1 kalimat proaktif."""

        # Bangun ringkasan konteks untuk prompt
        context_lines = []
        for msg in recent_messages:
            role = "User" if msg["role"] == "user" else "MEI"
            content = msg["content"][:200]  # batasi panjang
            context_lines.append(f"{role}: {content}")
        context_str = "\n".join(context_lines)

        # Bedakan instruksi antara trigger pertama vs retry
        if trigger_count == 1:
            instruction = (
                "User tiba-tiba diam. Buat 1 kalimat pendek yang natural untuk "
                "memancing obrolan lanjutan berdasarkan topik yang tadi dibahas. "
                "Jangan terlalu formal. Gunakan Bahasa Indonesia. Jangan pakai emoji. "
                "Maksimal 20 kata."
            )
        else:
            instruction = (
                "User masih belum merespon. Buat 1 kalimat pendek yang lebih santai "
                "dan tidak memaksa, seperti teman yang mengingatkan pelan-pelan. "
                "Gunakan Bahasa Indonesia. Jangan pakai emoji. Maksimal 15 kata."
            )

        prompt = f"""Berikut adalah percakapan terakhir antara MEI (AI assistant) dan {user_name}:

---
{context_str}
---

{instruction}

HANYA tulis kalimatnya saja, tanpa penjelasan apapun."""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 80,
            "temperature": 0.8,
            "top_p": 0.9,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        resp = requests.post(
            f"{self.server}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        if resp.status_code == 200:
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            # Bersihkan kalau ada quotes atau prefix yang tidak perlu
            text = text.strip('"\'')
            if text:
                return text

        return None

    def _fallback(self) -> str:
        """Pilih satu topik fallback secara acak."""
        return random.choice(PROACTIVE_CONFIG["fallback_topics"])