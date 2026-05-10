"""
Modul RVC (Retrieval-based Voice Conversion)
=============================================
Konversi audio dari Piper TTS ke suara target via RVC WebUI Gradio API.
Tidak pakai gradio_client (rawan error ws://) — langsung HTTP REST.

Prasyarat:
  - RVC WebUI jalan: cd rvc && venv\\Scripts\\python infer-web.py
  - Model .pth ada di  : rvc/assets/weights/NekoyamaSena.pth
  - Index ada di       : rvc/assets/weights/NekoyamaSena/*.index
  - hubert_base.pt     : rvc/assets/hubert/hubert_base.pt
  - rmvpe.pt           : rvc/assets/rmvpe/rmvpe.pt

Letakkan di: E:\\skripsi\\agent_qwen\\src\\voice\\rvc.py
"""

import io
import os
import time
import wave
import base64
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)


class RVCConfig:
    """Konfigurasi RVC WebUI."""

    def __init__(
        self,
        host: str = "http://127.0.0.1:7865",
        model_name: str = "zetaTest",
        index_path: str = "added_IVF462_Flat_nprobe_1_zetaTest_v2",
        pitch: int = 8,
        f0_method: str = "rmvpe",
        index_rate: float = 0.75,
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        timeout: int = 300,
    ):
        self.host         = host.rstrip("/")
        self.model_name   = model_name
        self.index_path   = index_path
        self.pitch        = pitch
        self.f0_method    = f0_method
        self.index_rate   = index_rate
        self.filter_radius = filter_radius
        self.resample_sr  = resample_sr
        self.rms_mix_rate = rms_mix_rate
        self.protect      = protect
        self.timeout      = timeout


class RVCConverter:
    """
    Voice converter menggunakan RVC WebUI Gradio API.

    Contoh penggunaan::

        from voice.rvc import RVCConverter, RVCConfig

        cfg = RVCConfig(
            host="http://127.0.0.1:7865",
            model_name="NekoyamaSena",
            index_path=r"E:\\...\\NekoyamaSena.index",
            pitch=0,
        )
        rvc = RVCConverter(cfg)

        # Dari WAV bytes (hasil Piper TTS)
        out_sr, out_audio = rvc.convert_bytes(wav_bytes)

        # Langsung speak (TTS bytes → RVC → speaker)
        metrics = rvc.speak_bytes(wav_bytes)
    """

    def __init__(self, config: RVCConfig):
        self.cfg = config
        self._model_loaded: Optional[str] = None   # nama model yang sudah di-load
        logger.info(
            f"RVCConverter init | host={config.host} | model={config.model_name} | "
            f"f0={config.f0_method} | pitch={config.pitch:+d}st"
        )
        self._check_server()

    # ──────────────────────────────────────────────
    #  PUBLIC API
    # ──────────────────────────────────────────────

    def convert_bytes(self, wav_bytes: bytes) -> tuple[int, np.ndarray]:
        """
        Konversi WAV bytes → audio numpy array dengan suara RVC.

        Args:
            wav_bytes: Raw WAV bytes (output dari Piper get_audio_bytes)

        Returns:
            (sample_rate: int, audio: np.ndarray float32)
        """
        tmp_in = self._write_temp_wav(wav_bytes)
        try:
            return self._convert_file(tmp_in)
        finally:
            _safe_remove(tmp_in)

    def convert_file(self, wav_path: str) -> tuple[int, np.ndarray]:
        """
        Konversi file WAV langsung.

        Args:
            wav_path: Path ke file WAV input

        Returns:
            (sample_rate: int, audio: np.ndarray float32)
        """
        return self._convert_file(wav_path)

    def speak_bytes(self, wav_bytes: bytes) -> dict:
        """
        TTS bytes → RVC → langsung putar ke speaker.

        Args:
            wav_bytes: WAV bytes dari Piper TTS

        Returns:
            dict metrik latensi
        """
        t0 = time.perf_counter()
        sr, audio = self.convert_bytes(wav_bytes)
        t_convert = time.perf_counter() - t0

        t1 = time.perf_counter()
        self._play(audio, sr)
        t_play = time.perf_counter() - t1

        duration = len(audio) / sr
        metrics = {
            "convert_ms"      : round(t_convert * 1000, 1),
            "play_ms"         : round(t_play * 1000, 1),
            "audio_duration_s": round(duration, 3),
            "sample_rate"     : sr,
            "rtf_convert"     : round(t_convert / duration, 4) if duration else 0,
        }
        self._log_metrics(metrics)
        return metrics

    def speak_file(self, wav_path: str) -> dict:
        """Konversi file WAV lalu langsung putar."""
        t0 = time.perf_counter()
        sr, audio = self.convert_file(wav_path)
        t_convert = time.perf_counter() - t0
        self._play(audio, sr)
        duration = len(audio) / sr
        return {
            "convert_ms"      : round(t_convert * 1000, 1),
            "audio_duration_s": round(duration, 3),
            "sample_rate"     : sr,
        }

    def save_bytes(self, wav_bytes: bytes, output_path: str) -> dict:
        """
        Konversi WAV bytes → simpan ke file (tanpa playback).

        Args:
            wav_bytes   : WAV bytes dari Piper
            output_path : Path output .wav

        Returns:
            dict metrik
        """
        t0 = time.perf_counter()
        sr, audio = self.convert_bytes(wav_bytes)
        sf.write(output_path, audio, sr)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"Saved RVC output: {output_path} | {elapsed:.0f} ms")
        return {
            "convert_ms"      : round(elapsed, 1),
            "audio_duration_s": round(len(audio) / sr, 3),
            "output_path"     : output_path,
        }

    def is_server_up(self) -> bool:
        """Cek apakah RVC WebUI sedang berjalan."""
        try:
            r = requests.get(f"{self.cfg.host}/", timeout=3)
            return r.status_code < 500
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Ambil daftar model yang tersedia di RVC WebUI."""
        result = self._post("/infer_refresh", [])
        raw = result[0] if result else {}
        if isinstance(raw, dict):
            return raw.get("choices", [])
        return raw if isinstance(raw, list) else []

    # ──────────────────────────────────────────────
    #  PRIVATE
    # ──────────────────────────────────────────────

    def _check_server(self):
        if not self.is_server_up():
            raise ConnectionError(
                f"RVC WebUI tidak bisa diakses di {self.cfg.host}. "
                "Pastikan sudah dijalankan: python infer-web.py"
            )
        logger.info(f"RVC WebUI OK: {self.cfg.host}")

    def _ensure_model_loaded(self):
        """Load model ke RVC hanya jika belum di-load (atau model berbeda)."""
        models = self.list_models()
        logger.debug(f"Model tersedia: {models}")

        # Cari nama yang cocok (partial match, case-insensitive)
        matched = self.cfg.model_name
        for m in models:
            if self.cfg.model_name.lower() in str(m).lower():
                matched = m
                break

        if matched == self._model_loaded:
            logger.debug(f"Model sudah ter-load: {matched}")
            return

        logger.info(f"Loading RVC model: {matched}")
        self._post("/infer_change_voice", [
            matched,
            self.cfg.protect,
            self.cfg.protect,
        ])
        self._model_loaded = matched
        logger.info(f"Model loaded: {matched}")

    def _convert_file(self, input_path: str) -> tuple[int, np.ndarray]:
        """Core conversion: kirim path ke RVC, terima audio output."""
        self._ensure_model_loaded()

        logger.info(
            f"RVC convert | input={os.path.basename(input_path)} | "
            f"f0={self.cfg.f0_method} | pitch={self.cfg.pitch:+d}st"
        )
        t0 = time.perf_counter()

        result = self._post("/infer_convert", [
            0,                      # speaker id
            input_path,             # path audio (Textbox — RVC baca langsung)
            self.cfg.pitch,         # transpose semitone
            None,                   # f0_curve_file (opsional)
            self.cfg.f0_method,     # pitch extraction algorithm
            self.cfg.index_path,    # feature index path (Textbox)
            "",                     # autodetect_index dropdown (kosong)
            self.cfg.index_rate,    # index rate
            self.cfg.filter_radius, # filter radius
            self.cfg.resample_sr,   # resample sr
            self.cfg.rms_mix_rate,  # rms mix rate
            self.cfg.protect,       # protect
        ])

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"RVC inference selesai: {elapsed:.0f} ms")

        info         = result[0] if len(result) > 0 else ""
        audio_result = result[1] if len(result) > 1 else result[0]

        if info:
            logger.debug(f"RVC info: {info}")

        return self._parse_audio_result(audio_result)

    def _parse_audio_result(self, audio_result) -> tuple[int, np.ndarray]:
        """Parse output Gradio → (sr, audio np.float32)."""
        tmp_out = None

        if isinstance(audio_result, dict):
            if "name" in audio_result or audio_result.get("is_file"):
                output_path = audio_result.get("name") or audio_result.get("path")
            elif "data" in audio_result:
                b64 = audio_result["data"]
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                raw = base64.b64decode(b64)
                tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_out.write(raw)
                tmp_out.close()
                output_path = tmp_out.name
            else:
                raise RuntimeError(f"Format audio_result tidak dikenal: {audio_result}")
        else:
            output_path = audio_result

        try:
            audio, sr = sf.read(output_path)
            audio = audio.astype(np.float32)
            logger.info(f"Output audio: {len(audio)/sr:.2f}s | sr={sr}Hz")
            return sr, audio
        finally:
            if tmp_out:
                _safe_remove(output_path)

    def _post(self, api_name: str, data: list) -> list:
        """POST ke Gradio REST endpoint."""
        url = f"{self.cfg.host}/run{api_name}"
        logger.debug(f"POST {url}")
        resp = requests.post(
            url, json={"data": data}, timeout=self.cfg.timeout
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Gradio API error {resp.status_code}: {resp.text[:300]}"
            )
        return resp.json().get("data", [])

    # SESUDAH — di rvc.py
    @staticmethod
    def _write_temp_wav(wav_bytes: bytes, target_sr: int = 16000) -> str:
        from math import gcd
        from scipy.signal import resample_poly

        buf = io.BytesIO(wav_bytes)
        audio, src_sr = sf.read(buf, dtype="float32")

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if src_sr != target_sr:
            g     = gcd(src_sr, target_sr)
            audio = resample_poly(audio, target_sr // g, src_sr // g)

        # ← normalisasi dihapus, biarkan level asli dari Piper

        audio_int16 = (audio * 32767).astype(np.int16)
        out_buf = io.BytesIO()
        with wave.open(out_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(target_sr)
            wf.writeframes(audio_int16.tobytes())

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(out_buf.getvalue())
        tmp.close()
        return tmp.name

    @staticmethod
    def _play(audio: np.ndarray, sr: int):
        sd.play(audio, samplerate=sr)
        sd.wait()

    @staticmethod
    def _log_metrics(m: dict):
        logger.info("=== RVC LATENSI ===")
        logger.info(f"  Convert   : {m['convert_ms']} ms")
        logger.info(f"  Durasi    : {m['audio_duration_s']} s")
        logger.info(f"  RTF       : {m['rtf_convert']}  "
                    f"({'✓ real-time' if m['rtf_convert'] < 1 else '✗ lambat'})")
        logger.info("===================")


# ──────────────────────────────────────────────
#  HELPER
# ──────────────────────────────────────────────

def _safe_remove(path: str):
    try:
        os.unlink(path)
    except Exception:
        pass


def create_rvc_from_env() -> RVCConverter:
    """
    Factory: buat RVCConverter dari environment variables.
    Berguna agar config tidak hardcode.

    Env vars (semua opsional, ada default):
      RVC_HOST, RVC_MODEL, RVC_INDEX, RVC_PITCH, RVC_F0
    """
    cfg = RVCConfig(
        host        = os.getenv("RVC_HOST",    "http://127.0.0.1:7865"),
        model_name  = os.getenv("RVC_MODEL",   "zetaTest"),
        index_path  = os.getenv("RVC_INDEX",   "added_IVF462_Flat_nprobe_1_zetaTest_v2"),
        pitch       = int(os.getenv("RVC_PITCH", "8")),
        f0_method   = os.getenv("RVC_F0",      "rmvpe"),
    )
    return RVCConverter(cfg)