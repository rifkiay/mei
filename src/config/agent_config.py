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
    'top_k'           : 10,

    # Trigger
    'max_recent_messages'  : 5,
    'max_longterm_memories': 5,              # jumlah yang di ambil dari chroma ke sytem   
    'extract_every_n_msgs' : 1,              # chroma extract
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
CLASSIFIER_CONFIG = {
    "model": "st",  # reuse paraphrase-multilingual-mpnet-base-v2 dari ChromaDB (cosine similiarty)
}

# ==========================================
# RVC CONFIG
# ==========================================

RVC_CONFIG = RVCConfig(
    host          = "http://127.0.0.1:7865",
    model_name    = "zetaTest",
    index_path    = r"E:\skripsi pendekatan openclaw\agent_qwen\rvc\assets\weights\added_IVF462_Flat_nprobe_1_zetaTest_v2.index",
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