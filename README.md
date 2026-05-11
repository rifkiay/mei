# mei

AI agent dengan kemampuan speech-to-text, text-to-speech, dan voice conversion.

---

## Struktur Folder

```
mei/
├── models/
│   └── id_ID-news_tts-medium/     # Model Piper TTS bahasa Indonesia
├── rvc/                            # RVC WebUI (clone langsung)
│   ├── configs/
│   ├── infer/
│   ├── tools/
│   ├── infer-web.py
│   ├── gui_v1.py
│   └── ...
├── whisper.cpp/                    # whisper.cpp (clone langsung)
│   ├── build/
│   ├── models/
│   ├── src/
│   └── ...
└── requirements.txt
```

---

### `models/` — Piper TTS

Berisi model text-to-speech menggunakan [Piper](https://rhasspy.github.io/piper-samples/).

Saat ini tersedia:

* `id_ID-news_tts-medium` — suara bahasa Indonesia, kualitas medium

Setiap folder model berisi:

```
id_ID-news_tts-medium/
├── id_ID-news_tts-medium.onnx
└── id_ID-news_tts-medium.onnx.json
```

**Download model lain:** [https://rhasspy.github.io/piper-samples/](https://rhasspy.github.io/piper-samples/)

---

### `rvc/` — Retrieval-based Voice Conversion

RVC WebUI di-clone langsung ke folder ini.

**Setup awal:**

bash

```bash
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI rvc
cd rvc
pip install -r requirements.txt
```

Jalankan WebUI untuk training / inferensi:

bash

```bash
cd rvc
python infer-web.py    # WebUI lengkap
# atau
python gui_v1.py       # GUI desktop
```

Referensi: [https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

---

### `whisper.cpp/` — Speech-to-Text

whisper.cpp di-clone langsung ke folder ini, lalu di-build dari source.

**Setup awal:**

bash

```bash
git clone https://github.com/ggml-org/whisper.cpp whisper.cpp
cd whisper.cpp
cmake -B build
cmake --build build --config Release

# Download model (taruh di whisper.cpp/models/)
bash ./models/download-ggml-model.sh medium
```

Model tersedia: `tiny`, `base`, `small`, `medium`, `large`

Referensi: [https://github.com/ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp)

---

## Instalasi Dependencies

**Catatan:** Beberapa package dalam `requirements.txt` mungkin tidak digunakan secara langsung oleh project ini. Serta mungkin ada package yang hilang dalam proses fillter packagenya.

bash

```bash
# Buat virtual environment
uv venv .venv --python 3.11.3

# Aktifkan
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# Install torch CUDA dulu (sebelum yang lain)
uv pip install --python .venv\Scripts\python.exe torch==2.10.0+cu130 torchaudio==2.10.0+cu130 torchvision==0.25.0+cu130 --index-url https://download.pytorch.org/whl/cu130

# Install sisanya
uv pip install --python .venv\Scripts\python.exe -r requirements.txt

# Kalau error pkg_resources
uv pip install --python .venv\Scripts\python.exe librosa --upgrade
```

---

## Requirements

Lihat `requirements.txt` untuk daftar lengkap dependencies.
LM Studio.
Rekomendasi gunakan GPU RTX 2060 12 GB jika ingin full menggunakan semua fitur.
