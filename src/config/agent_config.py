"""
Agent Configuration - Personal Edition
======================================
Short-term : JSONL file (storage/sessions/rifki.jsonl)
Long-term  : MEMORY.md + daily notes + ChromaDB HNSW embedded
"""
from voice.rvc import RVCConfig

# ==========================================
# HARDWARE TARGET
# ==========================================

HARDWARE_CONFIG = {
    'device': 'cuda',
}

def set_device(device: str):
    """Dipanggil dari main.py saat user ketik 'gpu' atau 'cpu'."""
    if HARDWARE_CONFIG['device'] == device:
        return  # tidak perlu reset kalau device sama
    HARDWARE_CONFIG['device'] = device
    # intent detector akan reload otomatis di get_intent_detector()
    # karena _intent_detector_device tidak match — tidak perlu set None manual

def get_device() -> str:
    return HARDWARE_CONFIG['device']

# ==========================================
# MEMORY SYSTEM
# ==========================================

MEMORY_CONFIG = {
    # Short-term: JSONL
    'sessions_dir' : './../storage/memory/sessions',
    'window_size'  : 10,

    # Long-term: MEMORY.md + daily + ChromaDB
    'storage_dir'     : './../storage/memory',
    'collection_name' : 'rifki_episodic',
    'embedding_model' : 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'top_k'           : 5,

    # Trigger
    'max_recent_messages'  : 5,
    'max_longterm_memories': 5,
    'extract_every_n_msgs' : 1,              # chroma 
    'daily_summary_on_exit': True,
}

# ==========================================
# LLM CONFIGURATION
# ==========================================

LLM_CONFIG = {
    'model'       : 'qwen/qwen3-vl-4b',
    'model_server': 'http://127.0.0.1:1234/v1',
    'api_key'     : 'lm-studio',
    'generate_cfg': {                              # ← tambah ini
        'stream_options': {'include_usage': True}  # LM Studio kirim usage di akhir stream
    },
}


# ==========================================
# CLASSIFIER CONFIGURATION
# ==========================================
# Classifier dipakai oleh FactExtractor untuk menentukan label percakapan
# sebelum diputuskan perlu disimpan ke memori atau tidak.
#
# Cara kerja:
#   "st" → ST cosine similarity
#          Reuse model sentence-transformers/paraphrase-multilingual-mpnet-base-v2
#          yang sudah di-load oleh ChromaDB — tidak ada model baru yang di-load.
#          Teks percakapan di-encode jadi vektor, lalu dicari kandidat label
#          yang paling mirip via cosine similarity.
#          Lebih stabil untuk teks Indonesia pendek, hemat VRAM.
#
#   nama HuggingFace model → NLI zero-shot pipeline
#          Load model NLI baru via transformers.pipeline().
#          Cocok jika ingin akurasi lebih tinggi dan VRAM tidak jadi masalah.
#          Contoh: "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
#
CLASSIFIER_CONFIG = {
    # "model": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",  # reuse paraphrase-multilingual-mpnet-base-v2 dari ChromaDB
    "model": "st",  # reuse paraphrase-multilingual-mpnet-base-v2 dari ChromaDB
}

# ==========================================
# RVC CONFIG
# ==========================================

RVC_CONFIG = RVCConfig(
    host          = "http://127.0.0.1:7865",
    model_name    = "model_name",
    index_path    = r"E:\skripsi pendekatan openclaw\agent_qwen\rvc\assets\weights\model_name.index",
    pitch         = 7,
    f0_method     = "rmvpe",
    index_rate    = 0.85,
    resample_sr   = 0,
    rms_mix_rate  = 0.1,
    filter_radius = 5,
    protect       = 0.33,
)

# ==========================================
# OUTPUT FILTERS
# ==========================================

OUTPUT_FILTERS = {
    'remove_emoji'                 : True,
    'simplify_markdown_in_chitchat': True,
    'max_newlines'                 : 2,
    'trim_whitespace'              : True,
}