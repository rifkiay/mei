"""
src/tools/camera_capture.py
============================
Tool untuk buka kamera, ambil foto, dan opsional analisis dengan LLM vision.

Cara kerja (gaya OpenClaw):
  1. Tool menerima mode + prompt dari LLM
  2. Buka kamera default via OpenCV
  3. Ambil satu frame, simpan ke storage/captures/
  4. Kalau mode="analyze": encode gambar ke base64, kirim ke LLM vision
  5. Return filepath + hasil analisis (jika ada)

Didokumentasikan di: storage/TOOLS.md

Dependencies: pip install opencv-python
"""

import json
import base64
import re
from datetime import datetime
from pathlib import Path
from typing import Union

from qwen_agent.tools.base import BaseTool, register_tool


_DEFAULT_CAPTURE_DIR = Path("./../storage/captures")


@register_tool("camera_capture")
class CameraCaptureTool(BaseTool):
    """
    Buka kamera, ambil foto, dan opsional analisis dengan LLM vision.
    """

    name        = "camera_capture"
    description = (
        "Buka kamera dan ambil foto. "
        "Gunakan saat user eksplisit minta foto, gambar, atau analisis visual. "
        "Mode 'capture' hanya simpan foto. "
        "Mode 'analyze' ambil foto lalu analisis isinya dengan deskripsi visual."
    )

    parameters = [
        {
            "name"       : "mode",
            "type"       : "string",
            "description": (
                "'capture' = ambil foto dan simpan (default). "
                "'analyze' = ambil foto lalu deskripsikan isinya."
            ),
            "required"   : False,
        },
        {
            "name"       : "prompt",
            "type"       : "string",
            "description": (
                "Pertanyaan atau instruksi untuk analisis gambar. "
                "Hanya relevan jika mode='analyze'. "
                "Contoh: 'Apa yang ada di meja ini?'"
            ),
            "required"   : False,
        },
    ]

    def __init__(self, capture_dir: Union[str, Path] = None, llm_cfg: dict = None):
        super().__init__()
        self.capture_dir = Path(capture_dir) if capture_dir else _DEFAULT_CAPTURE_DIR
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.llm_cfg = llm_cfg   # opsional, untuk mode analyze

    # ── Entry point ───────────────────────────────────────────────

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {}

        mode   = params.get("mode", "capture").strip().lower()
        prompt = params.get("prompt", "Deskripsikan apa yang ada di gambar ini.").strip()

        # Ambil foto
        filepath, error = self._capture_photo()
        if error:
            return self._error(error)

        result = {
            "status"  : "success",
            "filepath": str(filepath),
            "message" : f"Foto disimpan di {filepath}",
        }

        # Analisis jika diminta
        if mode == "analyze":
            analysis = self._analyze_image(filepath, prompt)
            result["analysis"] = analysis
            result["message"]  = analysis

        return json.dumps(result, ensure_ascii=False)

    # ── Internal: capture ─────────────────────────────────────────

    def _capture_photo(self) -> tuple[Path, str]:
        """
        Buka kamera via OpenCV, ambil satu frame, simpan ke file.
        Return: (filepath, error_string_or_None)
        """
        try:
            import cv2
        except ImportError:
            return None, "OpenCV tidak terinstall. Jalankan: pip install opencv-python"

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return None, "Kamera tidak bisa dibuka. Pastikan kamera tersambung."

        try:
            # Baca beberapa frame agar kamera sempat auto-expose
            for _ in range(5):
                ret, frame = cap.read()

            if not ret or frame is None:
                return None, "Gagal mengambil frame dari kamera."

            # Simpan ke file
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filepath  = self.capture_dir / f"{timestamp}.jpg"
            cv2.imwrite(str(filepath), frame)

            return filepath, None

        finally:
            cap.release()
            # Tutup semua window OpenCV jika ada
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    # ── Internal: analyze ─────────────────────────────────────────

    def _analyze_image(self, filepath: Path, prompt: str) -> str:
        """
        Encode gambar ke base64 lalu kirim ke LLM vision.
        Kalau llm_cfg tidak ada, return deskripsi file saja.
        """
        if not filepath.exists():
            return "File gambar tidak ditemukan."

        # Encode ke base64
        with open(filepath, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Kalau ada LLM config, kirim ke vision model
        if self.llm_cfg:
            try:
                import requests

                payload = {
                    "model"  : self.llm_cfg.get("model", "qwen/qwen3-vl-4b"),
                    "messages": [
                        {
                            "role"   : "user",
                            "content": [
                                {
                                    "type"     : "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{img_b64}"
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    "max_tokens": 500,
                }

                headers  = {"Authorization": f"Bearer {self.llm_cfg.get('api_key', '')}"}
                response = requests.post(
                    f"{self.llm_cfg.get('model_server', '')}/chat/completions",
                    json    = payload,
                    headers = headers,
                    timeout = 30,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]

            except Exception as e:
                return f"Foto berhasil diambil di {filepath}. (Analisis gagal: {e})"

        # Fallback tanpa LLM
        size_kb = filepath.stat().st_size // 1024
        return (
            f"Foto berhasil diambil dan disimpan di {filepath} ({size_kb} KB). "
            f"LLM vision tidak dikonfigurasi — analisis tidak tersedia."
        )

    def _error(self, msg: str) -> str:
        return json.dumps({"status": "error", "message": msg}, ensure_ascii=False)