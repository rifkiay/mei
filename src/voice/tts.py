"""
Modul TTS (Text-to-Speech) menggunakan Piper untuk Bahasa Indonesia
Model: id_ID-news_tts-medium
Audio langsung diputar ke speaker — tidak menyimpan file.
"""

import io
import wave
import time
import logging
from typing import Generator

import numpy as np

try:
    from piper import PiperVoice
except ImportError:
    raise ImportError(
        "Piper TTS tidak terinstal. Silakan install dengan: pip install piper-tts"
    )

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndonesianTTS:
    """
    Text-to-Speech Bahasa Indonesia menggunakan Piper.
    Audio langsung diputar ke speaker tanpa menyimpan file.

    Args:
        model_path : Path ke file model .onnx
        device     : "cuda" | "cpu"  (default "cpu")
                     Diteruskan ke PiperVoice.load(use_cuda=...)
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError(
                "sounddevice diperlukan untuk playback. "
                "Install dengan: pip install sounddevice"
            )

        from pathlib import Path
        self.model_path = Path(model_path)
        self.device     = device

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

        use_cuda = device == "cuda"
        logger.info(
            f"Loading model: {self.model_path.name} | "
            f"device={device} | use_cuda={use_cuda}"
        )
        t0 = time.perf_counter()
        self.voice = PiperVoice.load(str(self.model_path), use_cuda=use_cuda)
        load_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Model dimuat dalam {load_ms:.1f} ms | "
            f"sample_rate={self.voice.config.sample_rate} Hz"
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def speak(self, text: str) -> dict:
        """
        Sintesis teks dan langsung putar ke speaker secara real-time
        (streaming — chunk pertama diputar sebelum sintesis selesai).

        Args:
            text: Teks Bahasa Indonesia yang akan diucapkan

        Returns:
            dict: Metrik latensi
        """
        if not text.strip():
            raise ValueError("Teks tidak boleh kosong")

        logger.info(f"Speak: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")

        t_start       = time.perf_counter()
        t_first_chunk = None
        total_samples = 0
        chunk_count   = 0

        stream = sd.OutputStream(
            samplerate=self.voice.config.sample_rate,
            channels=1,
            dtype="int16",
        )
        stream.start()

        try:
            for chunk in self.voice.synthesize(text):
                audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)

                if t_first_chunk is None:
                    t_first_chunk = time.perf_counter() - t_start
                    logger.info(f"▶  Latensi chunk pertama : {t_first_chunk * 1000:.1f} ms")

                stream.write(audio)
                total_samples += len(audio)
                chunk_count   += 1

            stream.stop()
        finally:
            stream.close()

        t_total        = time.perf_counter() - t_start
        audio_duration = total_samples / self.voice.config.sample_rate
        rtf = (t_first_chunk or t_total) / audio_duration if audio_duration else 0

        metrics = {
            "text_length"      : len(text),
            "first_chunk_ms"   : round((t_first_chunk or 0) * 1000, 1),
            "total_ms"         : round(t_total * 1000, 1),
            "audio_duration_s" : round(audio_duration, 3),
            "rtf"              : round(rtf, 4),
            "chunk_count"      : chunk_count,
            "sample_rate"      : self.voice.config.sample_rate,
            "device"           : self.device,
        }

        self._log_metrics(metrics)
        return metrics

    def speak_from_file(self, text_file: str) -> dict:
        """
        Baca teks dari file .txt lalu langsung ucapkan.

        Args:
            text_file: Path ke file .txt

        Returns:
            dict: Metrik latensi
        """
        with open(text_file, encoding="utf-8") as f:
            text = f.read().strip()
        logger.info(f"Membaca dari file: {text_file} ({len(text)} karakter)")
        return self.speak(text)

    def get_audio_bytes(self, text: str) -> tuple:
        """
        Sintesis ke memori (bytes WAV) TANPA menyimpan file dan TANPA playback.
        Berguna untuk pipeline lain (RVC, WebSocket, API, dll).

        Args:
            text: Teks Bahasa Indonesia

        Returns:
            (wav_bytes: bytes, sample_rate: int)
        """
        t0  = time.perf_counter()
        buf = io.BytesIO()

        with wave.open(buf, "wb") as wf:
            self.voice.synthesize_wav(text, wf)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        wav_bytes  = buf.getvalue()
        logger.info(
            f"get_audio_bytes: {len(wav_bytes) / 1024:.1f} KB | "
            f"{elapsed_ms:.1f} ms | device={self.device}"
        )
        return wav_bytes, self.voice.config.sample_rate

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    @staticmethod
    def _log_metrics(m: dict):
        logger.info("=== LATENSI ===")
        logger.info(f"  Panjang teks     : {m['text_length']} karakter")
        logger.info(f"  Latensi chunk-1  : {m['first_chunk_ms']} ms")
        logger.info(f"  Durasi audio     : {m['audio_duration_s']} s")
        logger.info(f"  Total waktu      : {m['total_ms']} ms")
        logger.info(f"  RTF              : {m['rtf']}  ({'✓ real-time' if m['rtf'] < 1 else '✗ lambat'})")
        logger.info(f"  Jumlah chunk     : {m['chunk_count']}")
        logger.info(f"  Device           : {m['device']}")
        logger.info("===============")