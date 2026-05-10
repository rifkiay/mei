"""
Proactive Engine Configuration
================================
Semua setting untuk behavior proaktif MEI.
Sesuaikan nilai ini untuk mengatur seberapa aktif MEI memulai percakapan.
"""

PROACTIVE_CONFIG = {

    # ────────────────────────────────────────────
    # TIMING SETTINGS
    # ────────────────────────────────────────────

    # Berapa menit tidak ada aktivitas sebelum trigger pertama
    # Default: 3 menit
    "inactivity_threshold_minutes": 30,

    # Jeda antara trigger pertama dan trigger kedua (jika tidak direspon)
    # Default: 2 menit
    "retry_interval_minutes": 0.3,

    # Setelah masuk AFK mode, berapa lama MEI tidak akan trigger lagi
    # Default: 30 menit
    "afk_cooldown_minutes": 2,

    # ────────────────────────────────────────────
    # TRIGGER LIMITS
    # ────────────────────────────────────────────

    # Maksimum berapa kali proaktif trigger sebelum masuk AFK mode
    # Recommended: 2 (trigger 1x, retry 1x, lalu diam)
    "max_triggers_before_afk": 2,

    # ────────────────────────────────────────────
    # CONTEXT REQUIREMENTS
    # ────────────────────────────────────────────

    # Minimal jumlah pesan dalam session sebelum MEI boleh proaktif
    # Mencegah MEI langsung proaktif di awal sesi
    "min_messages_for_proactive": 2,  # cukup 2 pesan

    # Minimal durasi sesi (menit) sebelum proaktif diizinkan
    "min_session_duration_minutes": 1,

    # ────────────────────────────────────────────
    # CONTEXT WINDOW
    # ────────────────────────────────────────────

    # Berapa pesan terakhir yang dipertimbangkan untuk generate topik
    "context_window_messages": 8,

    # ────────────────────────────────────────────
    # TOPIC GENERATION
    # ────────────────────────────────────────────

    # Topik fallback jika tidak ada konteks yang cukup
    # MEI akan pilih satu secara acak
    "fallback_topics": [
        "Ngomong-ngomong, ada hal yang lagi kepikiran belakangan ini?",
        "Eh, ada project atau rencana seru yang mau dikerjain?",
        "Kalau lagi santai, biasanya ngapain?",
        "Ada topik atau hal baru yang lagi dipelajari sekarang?",
        "Gimana harinya sejauh ini?",
    ],

    # Template untuk trigger pertama (opening)
    "trigger_template_first": (
        "Masih di sini? {generated_message}"
    ),

    # Template untuk retry / trigger kedua (lebih casual)
    "trigger_template_retry": (
        "{generated_message}"
    ),

    # ────────────────────────────────────────────
    # DISPLAY SETTINGS
    # ────────────────────────────────────────────

    # Prefix yang ditampilkan sebelum pesan proaktif di terminal
    "display_prefix": "\n💬 MEI: ",

    # Suffix setelah pesan proaktif
    "display_suffix": "\n",

    # ────────────────────────────────────────────
    # FEATURE FLAGS
    # ────────────────────────────────────────────

    # Aktifkan/matikan proactive engine secara keseluruhan
    "enabled": True,

    # Apakah pesan proaktif juga di-speak via TTS (jika voice mode aktif)
    "voice_output_enabled": True,

    # Log debug ke terminal
    "debug_logging": False,
}