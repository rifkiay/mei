"""
STT MODULE: Whisper.cpp
=============================
Changelog:
  - REMOVED: parameter `k` dari VoiceIntentDetector (KNN tidak dipakai,
    detect_intent sudah pakai max-similarity langsung)
  - FIX: t_silence_start_ms sekarang di-track dari dalam record_audio()
    sehingga latency ASR di main.py bisa diukur dengan benar
    (sebelumnya selisih ~0ms karena kedua timestamp diambil berurutan)
  - FIX: process_voice_input() mengembalikan t_silence_start_ms di dict
  - CLEANUP: self.k dihapus, tidak ada referensi KNN tersisa
"""

import sounddevice as sd
import scipy.io.wavfile as wav
import subprocess
import os
import numpy as np
import time
import re
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent.parent



# ========================================
# WHISPER HALLUCINATION FILTER
# ========================================
# Whisper dikenal menghasilkan teks-teks ini pada audio kosong/senyap.
# Kalau output Whisper (stripped lowercase) ada di sini, hasil dibuang.
_WHISPER_HALLUCINATIONS: set = {
    # Bahasa Indonesia
    "terima kasih", "sampai jumpa", "selamat tinggal",
    "ya", "oke", "iya", "baik", "halo", "hai",
    "hmm", "eh", "oh", "uh", "mm",
    # English (Whisper kadang output ini meski mode -l id)
    "thank you", "thanks", "okay", "ok", "yes", "no",
    "bye", "goodbye", "hello", "hi",
    "subtitles by", "subtitle by", "transcribed by",
    "you", ".", "",
}


def _is_whisper_hallucination(text: str) -> bool:
    """True kalau teks adalah halusinasi Whisper pada audio kosong/pendek."""
    cleaned = text.strip().lower().rstrip(".")
    if cleaned in _WHISPER_HALLUCINATIONS:
        return True
    if text.strip().lower() in _WHISPER_HALLUCINATIONS:
        return True
    # Terlalu pendek
    if len(cleaned) < 3:
        return True
    return False


# ========================================
# CONFIGURATION
# ========================================
class STTConfig:
    """Konfigurasi untuk STT Engine"""

    # Audio settings
    SAMPLE_RATE  = 16000
    CHUNK_SIZE_MS = 100

    # SILENCE_THRESHOLD: amplitudo rata-rata di bawah ini = senyap.
    # 0.01 terlalu sensitif untuk virtual mic NVIDIA Broadcast.
    # Naikkan kalau masih false-positive, turunkan kalau speech tidak terdeteksi.
    SILENCE_THRESHOLD = 0.02

    SILENCE_DURATION    = 1.0
    MAX_RECORD_DURATION = 30

    # MIN_SPEECH_DURATION: rekaman dengan total speech < ini (detik) dibuang
    # tanpa dikirim ke Whisper — cegah klik/transien pendek memicu halusinasi.
    MIN_SPEECH_DURATION = 0.8

    # Whisper.cpp paths
    WHISPER_CLI     = BASE_DIR / "whisper.cpp" / "build" / "bin" / "Release" / "whisper-cli.exe"
    WHISPER_MODEL   = BASE_DIR / "whisper.cpp" / "models" / "ggml-large-v3-turbo-q8_0.bin"
    TEMP_AUDIO_FILE = "temp_recording.wav"

    THRESHOLD = 0.48

    # Model — k DIHAPUS, tidak relevan untuk max-similarity approach
    MODEL_NAME : str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    # Hardware
    DEVICE  : str  = "cuda"
    USE_GPU : bool = True


# ========================================
# VOICE INTENT DETECTOR
# ========================================
class VoiceIntentDetector:
    """
    Intent detector untuk voice menggunakan max cosine similarity.
    Parameter `k` dihapus — tidak dipakai sejak migrasi dari KNN.
    """

    def __init__(
        self,
        model_name    : str            = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        examples_path : Optional[Path] = None,
        device        : str            = "cpu",
        threshold     : float          = 0.48,
    ):
        # k TIDAK ADA — dihapus karena detect_intent tidak pakai KNN
        self.threshold = threshold

        if examples_path is None:
            examples_path = Path(__file__).resolve().parent / "intent_voice.json"
        self.examples_path = Path(examples_path)

        print(f"Loading intent detection model: {model_name}...")
        self.model = SentenceTransformer(str(model_name), device=device)
        print("✅ Intent detector ready")

        self.intent = self._load_examples(self.examples_path)
        self._create_intent_embeddings()

    # ----------------------------------------------------------
    # LOAD DATASET
    # ----------------------------------------------------------

    def _load_examples(self, path: Path) -> Dict[str, List[str]]:
        import json
        if not path.exists():
            raise FileNotFoundError(
                f"Intent examples tidak ditemukan: {path}\n"
                f"Pastikan intent_voice.json ada di: {path.parent}"
            )
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        total = sum(len(v) for v in data.values())
        print(f"   Loaded {len(data)} intents, {total} examples dari {path.name}")
        return data

    def reload_examples(self, path: Optional[Path] = None):
        """Reload dataset dari JSON tanpa restart."""
        self.intent = self._load_examples(path or self.examples_path)
        self._create_intent_embeddings()
        print("✅ Dataset reloaded")

    # ----------------------------------------------------------
    # BUILD EMBEDDINGS
    # ----------------------------------------------------------

    def _create_intent_embeddings(self):
        self.intent_embeddings: Dict[str, Dict] = {}
        for intent, examples in self.intent.items():
            embeddings = self.model.encode(examples)
            self.intent_embeddings[intent] = {
                "examples"  : examples,
                "embeddings": embeddings,
            }

    # ----------------------------------------------------------
    # DETECT INTENT  (max cosine similarity, bukan KNN)
    # ----------------------------------------------------------

    def detect_intent(self, user_input: str, margin: float = 0.05) -> Tuple[str, float]:
        input_emb = self.model.encode(self._preprocess(user_input))

        best_intent, best_score = None, -1.0
        for intent, data in self.intent_embeddings.items():
            max_sim = max(
                self._cosine_similarity(input_emb, emb)
                for emb in data["embeddings"]
            )
            if max_sim > best_score:
                best_score, best_intent = max_sim, intent

        if best_score < self.threshold:
            return "default", best_score
        return best_intent, best_score

    # ----------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------

    def _preprocess(self, text: str) -> str:
        text   = text.lower().strip()
        tokens = text.split()
        if tokens and all(len(t) == 1 for t in tokens):
            text = "".join(tokens)
        text = re.sub(r"\s+", " ", text)
        return text

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ========================================
# STT ENGINE (WHISPER.CPP)
# ========================================
class STTEngine:
    """
    Main STT Engine: Whisper.cpp untuk transkripsi + VoiceIntentDetector untuk intent.

    Perubahan latency tracking:
      record_audio() sekarang mengembalikan tuple (audio, t_silence_start_ms).
      t_silence_start_ms = timestamp (perf_counter * 1000) saat silence pertama
      terdeteksi setelah speech → ini adalah "user selesai bicara", yang dipakai
      sebagai t_turn_start di LatencyTracker supaya latency ASR terukur benar.
    """

    def __init__(self, config: STTConfig = None):
        self.config          = config or STTConfig()
        self.intent_detector = VoiceIntentDetector(
            model_name = self.config.MODEL_NAME,
            device     = self.config.DEVICE,
            threshold  = self.config.THRESHOLD,
            # k tidak diteruskan — sudah dihapus dari VoiceIntentDetector
        )
        self.conversation_context: List[str] = []
        self._validate_whisper_setup()

    def _validate_whisper_setup(self):
        if not os.path.exists(self.config.WHISPER_CLI):
            print(f"⚠️  Warning: whisper-cli tidak ditemukan di {self.config.WHISPER_CLI}")
        if not os.path.exists(self.config.WHISPER_MODEL):
            print(f"⚠️  Warning: Model tidak ditemukan di {self.config.WHISPER_MODEL}")

    # ----------------------------------------------------------
    # AUDIO
    # ----------------------------------------------------------

    def _is_silence(self, audio_chunk: np.ndarray) -> bool:
        return np.abs(audio_chunk).mean() < self.config.SILENCE_THRESHOLD

    def record_audio(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Rekam audio dari mic sampai silence terdeteksi.

        Returns:
            (audio_array, t_silence_start_ms)
            - audio_array        : numpy array float32, atau None jika gagal
                                   None juga dikembalikan kalau speech terlalu
                                   pendek (< MIN_SPEECH_DURATION) — ini
                                   mencegah klik mic memicu halusinasi Whisper.
            - t_silence_start_ms : timestamp (ms) saat user selesai bicara.
        """
        print("🎙️  Recording... (berbicara sekarang)")

        audio_buffer       = []
        silence_counter    = 0.0
        speech_duration    = 0.0   # total detik di mana ada suara
        chunk_size         = int(self.config.SAMPLE_RATE * self.config.CHUNK_SIZE_MS / 1000)
        chunk_sec          = self.config.CHUNK_SIZE_MS / 1000.0
        t_silence_start_ms = 0.0

        try:
            with sd.InputStream(
                samplerate=self.config.SAMPLE_RATE,
                channels=1,
                dtype="float32",
            ) as stream:
                start_time      = time.time()
                speech_detected = False

                while True:
                    chunk, _ = stream.read(chunk_size)
                    audio_buffer.append(chunk)
                    is_sil = self._is_silence(chunk)

                    if not speech_detected and not is_sil:
                        speech_detected = True
                        print("🗣️  Speech detected!")

                    if speech_detected:
                        if is_sil:
                            if silence_counter == 0.0:
                                t_silence_start_ms = time.perf_counter() * 1000.0
                            silence_counter += chunk_sec
                        else:
                            speech_duration += chunk_sec
                            silence_counter    = 0.0
                            t_silence_start_ms = 0.0

                        if silence_counter >= self.config.SILENCE_DURATION:
                            print("⏸️  Silence detected, processing...")
                            break

                    if time.time() - start_time > self.config.MAX_RECORD_DURATION:
                        print("⏱️  Max duration reached")
                        break

            # Buang rekaman yang speech-nya terlalu pendek — hampir pasti
            # hanya klik mic atau transien, dan Whisper akan halusinasi.
            if speech_duration < self.config.MIN_SPEECH_DURATION:
                print(f"⚠️  Speech terlalu pendek ({speech_duration:.2f}s < "
                      f"{self.config.MIN_SPEECH_DURATION}s), dibuang.")
                return None, 0.0

            audio = np.concatenate(audio_buffer, axis=0).flatten()
            return audio, t_silence_start_ms

        except KeyboardInterrupt:
            print("\n⏹️  Recording stopped by user")
            return None, 0.0

    # ----------------------------------------------------------
    # TRANSCRIBE
    # ----------------------------------------------------------

    def transcribe_audio(self, audio: np.ndarray) -> Optional[str]:
        try:
            audio_int16 = (audio * 32767).astype(np.int16)
            wav.write(self.config.TEMP_AUDIO_FILE, self.config.SAMPLE_RATE, audio_int16)

            print("🔄 Transcribing...")

            cmd = [
                str(self.config.WHISPER_CLI),
                "-m", str(self.config.WHISPER_MODEL),
                "-f", self.config.TEMP_AUDIO_FILE,
                "-l", "id",
                "-nt",
            ]
            if not self.config.USE_GPU:
                cmd.append("--no-gpu")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            if result.returncode != 0:
                print(f"❌ Whisper error (code {result.returncode})")
                print(f"STDERR: {result.stderr}")
                return None

            transcription = result.stdout.strip()
            for line in reversed(transcription.split("\n")):
                line = line.strip()
                if line and not line.startswith("["):
                    return line

            return transcription if transcription else None

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None
        finally:
            if os.path.exists(self.config.TEMP_AUDIO_FILE):
                try:
                    os.remove(self.config.TEMP_AUDIO_FILE)
                except Exception:
                    pass

    # ----------------------------------------------------------
    # INTENT
    # ----------------------------------------------------------

    def classify_intent(self, text: str) -> Dict:
        try:
            label, score = self.intent_detector.detect_intent(text)
            return {"label": label, "confidence": score}
        except Exception as e:
            print(f"⚠️  Intent classification error: {e}")
            return {"label": "default", "confidence": 0.0}

    # ----------------------------------------------------------
    # MAIN
    # ----------------------------------------------------------

    def process_voice_input(self) -> Optional[Dict]:
        """
        Record → Transcribe → Classify Intent

        Returns dict dengan field:
            text              : str   — hasil transkripsi
            label             : str   — intent label
            confidence        : float — skor similarity
            should_respond    : bool  — apakah AI harus merespons
            t_silence_start_ms: float — timestamp (ms) saat user selesai bicara
                                        → dipakai sebagai t_turn_start di LatencyTracker
                                        → 0.0 jika tidak ada speech terdeteksi
        """
        try:
            audio, t_silence_start_ms = self.record_audio()
            if audio is None:
                return None

            # Catat timestamp tepat setelah Whisper selesai (= t_stt_done)
            text = self.transcribe_audio(audio)
            t_asr_done_ms = time.perf_counter() * 1000.0

            if not text or not text.strip():
                print("❌ No speech detected")
                return None

            print(f"📝 Transcribed: '{text}'")

            # Filter halusinasi Whisper — buang sebelum intent classification
            if _is_whisper_hallucination(text):
                print(f"⚠️  Whisper hallucination dideteksi, dibuang: '{text}'")
                return None

            result = self.classify_intent(text)

            print(f"\n{'='*50}")
            print(f"🏷️  Label      : {result['label']}")
            print(f"💯 Confidence : {result['confidence']:.2%}")
            print(f"{'='*50}\n")

            self.conversation_context.append(text)

            label = result["label"]
            if label == "DIRECTED":
                print("🤖 AI should respond")
                should_respond = True
            elif label == "SELF-TALK":
                print("🙊 AI listening silently...")
                should_respond = False
            elif label == "INTERRUPT":
                print("✋ Interrupt detected - AI stopping")
                should_respond = False
            else:
                print("❓ Uncertain - AI waiting for clarification")
                should_respond = False

            return {
                "text"              : text,
                "label"             : label,
                "confidence"        : result["confidence"],
                "should_respond"    : should_respond,
                # ── Latency timestamps ──────────────────────────────
                # t_silence_start_ms : user selesai bicara → t_turn_start
                # t_asr_done_ms      : whisper selesai transkripsi → t_stt_done
                "t_silence_start_ms": t_silence_start_ms,
                "t_asr_done_ms"     : t_asr_done_ms,
            }

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_conversation_history(self) -> List[str]:
        return self.conversation_context.copy()

    def clear_conversation_history(self):
        self.conversation_context.clear()


# ========================================
# CONVENIENCE FUNCTIONS
# ========================================
def create_stt_engine(
    whisper_cli_path : Optional[str] = None,
    model_path       : Optional[str] = None,
    model_name       : str  = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    device           : str  = "cuda",
    use_gpu          : bool = True,
) -> STTEngine:
    """
    Factory: buat STTEngine dengan custom paths + hardware target.
    Parameter `k` dihapus — tidak relevan untuk max-similarity intent detection.
    """
    config = STTConfig()
    if whisper_cli_path:
        config.WHISPER_CLI   = whisper_cli_path
    if model_path:
        config.WHISPER_MODEL = model_path
    config.MODEL_NAME = model_name
    config.DEVICE     = device
    config.USE_GPU    = use_gpu
    return STTEngine(config)