"""
MEI Desktop UI
==============
Version 4.5.8  (+ UI streaming support)

File terpisah — jalankan bersamaan dengan main.py via:
    python mei_ui.py

Atau integrasikan ke main.py dengan:
    from mei_ui import MEIApp
    app = MEIApp(agent_callback=your_callback)
    app.run()

Fitur:
  - Chat bubble (user kanan, MEI kiri)
  - Notif bar maks 3 notifikasi di atas chat
  - Mode bar: Text / TTS / STT / STT+RVC
  - Toggle mic — kalau aktif langsung dengerin tanpa tekan Enter
  - Send button → berubah jadi STOP (interrupt) saat MEI sedang balas
  - STT mic: kalau aktif, auto-kirim hasil transkripsi ke agent
  - [v4.5.8] Streaming token: bubble MEI update real-time per delta
"""

import tkinter as tk
from tkinter import ttk
import threading
import queue
import time
from datetime import datetime
from typing import Callable, Optional


# ══════════════════════════════════════════════════════════════════
# THEME
# ══════════════════════════════════════════════════════════════════

COLORS = {
    "bg"           : "#0f0f13",
    "bg_panel"     : "#16161d",
    "bg_input"     : "#1e1e28",
    "bg_bubble_mei": "#1e2a3a",
    "bg_bubble_usr": "#1a3a2a",
    "bg_notif"     : "#1a1a24",
    "bg_notif_item": "#22222e",
    "accent"       : "#4af0a8",
    "accent2"      : "#4ab8f0",
    "accent_warn"  : "#f0c84a",
    "text"         : "#e8e8f0",
    "text_dim"     : "#6a6a7a",
    "text_mei"     : "#a8d8ff",
    "text_usr"     : "#a8ffd8",
    "border"       : "#2a2a38",
    "mic_on"       : "#f04a4a",
    "mic_off"      : "#3a3a4a",
    "btn_send"     : "#2a4a3a",
    "btn_hover"    : "#3a5a4a",
    "btn_stop"     : "#4a1a1a",
    "btn_stop_hov" : "#5a2a2a",
    "btn_stop_fg"  : "#f08080",
}

FONTS = {
    "chat"      : ("Consolas", 11),
    "chat_name" : ("Consolas", 9),
    "input"     : ("Consolas", 11),
    "notif"     : ("Consolas", 10),
    "btn"       : ("Consolas", 10),
    "mode"      : ("Consolas", 9),
    "title"     : ("Consolas", 13, "bold"),
    "timestamp" : ("Consolas", 8),
}

MAX_NOTIF = 3


# ══════════════════════════════════════════════════════════════════
# DUMMY AGENT (untuk standalone testing)
# ══════════════════════════════════════════════════════════════════

def _dummy_agent(user_text: str, mode: str, on_token: Callable = None) -> str:
    """
    Dummy agent untuk testing standalone.
    Simulasi streaming dengan kirim token satu per satu.
    """
    words = f"[dummy] kamu bilang: '{user_text}' | mode: {mode}".split()
    full  = ""
    for i, word in enumerate(words):
        delta = ("" if i == 0 else " ") + word
        full += delta
        if on_token:
            on_token(delta)
        time.sleep(0.08)
    return full


# ══════════════════════════════════════════════════════════════════
# NOTIF BAR
# ══════════════════════════════════════════════════════════════════

class NotifBar(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=COLORS["bg_notif"], **kwargs)
        self._notifs: list[dict] = []
        self._widgets: list[tk.Frame] = []
        self._rebuild()

    def push(self, message: str, tag: str = "info"):
        now = datetime.now().strftime("%H:%M")
        if len(self._notifs) >= MAX_NOTIF:
            self._notifs.pop(0)
        self._notifs.append({"msg": message, "time": now, "tag": tag})
        self._rebuild()

    def dismiss(self, index: int):
        if 0 <= index < len(self._notifs):
            self._notifs.pop(index)
            self._rebuild()

    def _rebuild(self):
        for w in self._widgets:
            w.destroy()
        self._widgets.clear()

        if not self._notifs:
            ph = tk.Label(
                self,
                text="— tidak ada notifikasi —",
                bg=COLORS["bg_notif"],
                fg=COLORS["text_dim"],
                font=FONTS["notif"],
                pady=6,
            )
            ph.pack(fill="x", padx=12)
            self._widgets.append(ph)
            return

        for i, notif in enumerate(self._notifs):
            color = {
                "info"   : COLORS["accent2"],
                "warn"   : COLORS["accent_warn"],
                "success": COLORS["accent"],
                "error"  : COLORS["mic_on"],
            }.get(notif.get("tag", "info"), COLORS["accent2"])

            row = tk.Frame(self, bg=COLORS["bg_notif_item"], pady=2)
            row.pack(fill="x", padx=8, pady=2)

            tk.Label(
                row, text="●", bg=COLORS["bg_notif_item"],
                fg=color, font=FONTS["notif"],
            ).pack(side="left", padx=(8, 4))

            tk.Label(
                row,
                text=notif["msg"][:80] + ("…" if len(notif["msg"]) > 80 else ""),
                bg=COLORS["bg_notif_item"],
                fg=COLORS["text"],
                font=FONTS["notif"],
                anchor="w",
            ).pack(side="left", fill="x", expand=True)

            tk.Label(
                row, text=notif["time"],
                bg=COLORS["bg_notif_item"],
                fg=COLORS["text_dim"],
                font=FONTS["timestamp"],
            ).pack(side="left", padx=4)

            idx = i
            close_btn = tk.Label(
                row, text="✕",
                bg=COLORS["bg_notif_item"],
                fg=COLORS["text_dim"],
                font=FONTS["timestamp"],
                cursor="hand2",
            )
            close_btn.pack(side="right", padx=(4, 8))
            close_btn.bind("<Button-1>", lambda e, i=idx: self.dismiss(i))
            self._widgets.append(row)


# ══════════════════════════════════════════════════════════════════
# MODE BAR
# ══════════════════════════════════════════════════════════════════

class ModeBar(tk.Frame):
    MODES = [
        ("Text",    "1"),
        ("TTS",     "2"),
        ("STT",     "3"),
        ("STT+RVC", "4"),
    ]

    def __init__(self, parent, on_mode_change: Callable, on_mic_toggle: Callable, **kwargs):
        super().__init__(parent, bg=COLORS["bg_panel"], **kwargs)
        self._on_mode_change = on_mode_change
        self._on_mic_toggle  = on_mic_toggle
        self._current_mode   = "1"
        self._mic_active     = False
        self._btns: dict[str, tk.Label] = {}
        self._build()

    def _build(self):
        tk.Label(
            self, text="MODE:",
            bg=COLORS["bg_panel"], fg=COLORS["text_dim"],
            font=FONTS["mode"],
        ).pack(side="left", padx=(12, 6))

        for label, code in self.MODES:
            btn = tk.Label(
                self, text=label,
                bg=COLORS["bg_panel"],
                fg=COLORS["text_dim"],
                font=FONTS["mode"],
                padx=10, pady=4,
                cursor="hand2",
                relief="flat",
            )
            btn.pack(side="left", padx=2)
            btn.bind("<Button-1>", lambda e, c=code: self._select_mode(c))
            btn.bind("<Enter>", lambda e, b=btn: b.config(fg=COLORS["text"]))
            btn.bind("<Leave>", lambda e, b=btn, c=code: b.config(
                fg=COLORS["accent"] if c == self._current_mode else COLORS["text_dim"]
            ))
            self._btns[code] = btn

        self._update_mode_colors()

        tk.Label(
            self, text="│",
            bg=COLORS["bg_panel"], fg=COLORS["border"],
            font=FONTS["mode"],
        ).pack(side="left", padx=8)

        self._mic_label = tk.Label(
            self, text="🎙 MIC: OFF",
            bg=COLORS["bg_panel"],
            fg=COLORS["mic_off"],
            font=FONTS["mode"],
            padx=10, pady=4,
            cursor="hand2",
        )
        self._mic_label.pack(side="left", padx=2)
        self._mic_label.bind("<Button-1>", lambda e: self._toggle_mic())

        self._status = tk.Label(
            self, text="",
            bg=COLORS["bg_panel"],
            fg=COLORS["text_dim"],
            font=FONTS["mode"],
        )
        self._status.pack(side="right", padx=12)

    def _select_mode(self, code: str):
        self._current_mode = code
        self._update_mode_colors()
        self._on_mode_change(code)

    def _update_mode_colors(self):
        for code, btn in self._btns.items():
            if code == self._current_mode:
                btn.config(fg=COLORS["accent"], font=(*FONTS["mode"][:2], "bold"))
            else:
                btn.config(fg=COLORS["text_dim"], font=FONTS["mode"])

    def _toggle_mic(self):
        self._mic_active = not self._mic_active
        if self._mic_active:
            self._mic_label.config(text="🎙 MIC: ON", fg=COLORS["mic_on"])
        else:
            self._mic_label.config(text="🎙 MIC: OFF", fg=COLORS["mic_off"])
        self._on_mic_toggle(self._mic_active)

    def set_status(self, text: str, color: str = None):
        self._status.config(text=text, fg=color or COLORS["text_dim"])

    @property
    def current_mode(self) -> str:
        return self._current_mode

    @property
    def mic_active(self) -> bool:
        return self._mic_active


# ══════════════════════════════════════════════════════════════════
# CHAT AREA
# ══════════════════════════════════════════════════════════════════

class ChatArea(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=COLORS["bg"], **kwargs)
        self._build()

    def _build(self):
        self._canvas = tk.Canvas(
            self, bg=COLORS["bg"],
            highlightthickness=0, bd=0,
        )
        scrollbar = tk.Scrollbar(
            self, orient="vertical",
            command=self._canvas.yview,
            bg=COLORS["bg"],
            troughcolor=COLORS["bg_panel"],
            width=6,
        )
        self._canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self._inner = tk.Frame(self._canvas, bg=COLORS["bg"])
        self._canvas_window = self._canvas.create_window(
            (0, 0), window=self._inner, anchor="nw"
        )
        self._inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_inner_configure(self, event):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self._canvas.itemconfig(self._canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def add_message(self, role: str, text: str, timestamp: str = None):
        ts       = timestamp or datetime.now().strftime("%H:%M")
        is_user  = role == "user"
        bubble_color = COLORS["bg_bubble_usr"] if is_user else COLORS["bg_bubble_mei"]
        text_color   = COLORS["text_usr"]       if is_user else COLORS["text_mei"]
        name_text    = "Rifki" if is_user else "MEI"
        name_color   = COLORS["accent"]  if is_user else COLORS["accent2"]

        row = tk.Frame(self._inner, bg=COLORS["bg"], pady=4)
        row.pack(fill="x", padx=8)

        bubble = tk.Frame(row, bg=bubble_color, padx=12, pady=8)

        if is_user:
            tk.Frame(row, bg=COLORS["bg"]).pack(side="left", fill="x", expand=True)
            bubble.pack(side="right")
        else:
            bubble.pack(side="left")
            tk.Frame(row, bg=COLORS["bg"]).pack(side="right", fill="x", expand=True)

        header = tk.Frame(bubble, bg=bubble_color)
        header.pack(fill="x")
        tk.Label(header, text=name_text, bg=bubble_color, fg=name_color,
                 font=FONTS["chat_name"]).pack(side="left")
        tk.Label(header, text=ts, bg=bubble_color, fg=COLORS["text_dim"],
                 font=FONTS["timestamp"]).pack(side="right", padx=(8, 0))

        tk.Label(
            bubble, text=text,
            bg=bubble_color, fg=text_color,
            font=FONTS["chat"],
            wraplength=420, justify="left", anchor="w",
        ).pack(fill="x", pady=(4, 0))

        self.after(50, self._scroll_bottom)

    def add_system(self, text: str):
        row = tk.Frame(self._inner, bg=COLORS["bg"], pady=2)
        row.pack(fill="x")
        tk.Label(
            row, text=f"— {text} —",
            bg=COLORS["bg"], fg=COLORS["text_dim"],
            font=(*FONTS["chat"][:1], FONTS["chat"][1] - 1, "italic"),
        ).pack()
        self.after(50, self._scroll_bottom)

    def set_typing(self, active: bool):
        if active:
            if not hasattr(self, "_typing_row"):
                self._typing_row = tk.Frame(self._inner, bg=COLORS["bg"], pady=4)
                self._typing_row.pack(fill="x", padx=8)
                self._typing_label = tk.Label(
                    self._typing_row,
                    text="MEI sedang mengetik...",
                    bg=COLORS["bg"], fg=COLORS["text_dim"],
                    font=(*FONTS["chat"][:1], FONTS["chat"][1] - 1, "italic"),
                )
                self._typing_label.pack(side="left")
            self.after(50, self._scroll_bottom)
        else:
            if hasattr(self, "_typing_row"):
                self._typing_row.destroy()
                del self._typing_row, self._typing_label

    def _scroll_bottom(self):
        self._canvas.yview_moveto(1.0)


# ══════════════════════════════════════════════════════════════════
# INPUT BAR
# ══════════════════════════════════════════════════════════════════

class InputBar(tk.Frame):
    """
    Area input teks + tombol Send / STOP.

    Saat agent sedang merespons (_interrupt_mode=True):
      - Tombol berubah menjadi "✕ STOP" (merah)
      - Klik tombol atau tekan Enter memanggil on_interrupt()
    """

    def __init__(self, parent, on_send: Callable,
                 on_interrupt: Optional[Callable] = None, **kwargs):
        super().__init__(parent, bg=COLORS["bg_input"], pady=8, **kwargs)
        self._on_send       = on_send
        self._on_interrupt  = on_interrupt
        self._interrupt_mode = False
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=1)

        self._entry = tk.Text(
            self, height=2,
            bg=COLORS["bg_input"], fg=COLORS["text"],
            insertbackground=COLORS["accent"],
            font=FONTS["input"],
            relief="flat", wrap="word",
            padx=12, pady=8, bd=0,
        )
        self._entry.grid(row=0, column=0, sticky="ew", padx=(12, 4))
        self._entry.bind("<Return>",       self._on_enter)
        self._entry.bind("<Shift-Return>", self._on_shift_enter)

        self._send_btn = tk.Label(
            self,
            text="SEND ▶",
            bg=COLORS["btn_send"],
            fg=COLORS["accent"],
            font=FONTS["btn"],
            padx=14, pady=12,
            cursor="hand2",
            width=8,
        )
        self._send_btn.grid(row=0, column=1, sticky="ns", padx=(0, 12))
        self._send_btn.bind("<Button-1>", lambda e: self._handle_btn_click())
        self._send_btn.bind("<Enter>",    self._on_btn_hover_enter)
        self._send_btn.bind("<Leave>",    self._on_btn_hover_leave)

        tk.Label(
            self,
            text="Enter kirim · Shift+Enter baris baru · Enter saat MEI balas = interrupt",
            bg=COLORS["bg_input"], fg=COLORS["text_dim"],
            font=FONTS["timestamp"],
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=14, pady=(0, 2))

    # ── Event handlers ─────────────────────────────────────────────

    def _on_enter(self, event):
        self._handle_btn_click()
        return "break"

    def _on_shift_enter(self, event):
        return None  # biarkan newline masuk

    def _handle_btn_click(self):
        """Dispatch ke interrupt atau send tergantung mode."""
        if self._interrupt_mode:
            if self._on_interrupt:
                self._on_interrupt()
        else:
            self._fire_send()

    def _fire_send(self):
        text = self._entry.get("1.0", "end-1c").strip()
        if text:
            self._entry.delete("1.0", "end")
            self._on_send(text)

    # ── Hover styling (aware of mode) ─────────────────────────────

    def _on_btn_hover_enter(self, e):
        if self._interrupt_mode:
            self._send_btn.config(bg=COLORS["btn_stop_hov"])
        else:
            self._send_btn.config(bg=COLORS["btn_hover"])

    def _on_btn_hover_leave(self, e):
        if self._interrupt_mode:
            self._send_btn.config(bg=COLORS["btn_stop"])
        else:
            self._send_btn.config(bg=COLORS["btn_send"])

    # ── Public API ─────────────────────────────────────────────────

    def set_interrupt_mode(self, active: bool):
        """
        True  → tampilkan tombol "✕ STOP" (merah), Enter = interrupt.
        False → kembali ke "SEND ▶" (hijau).
        """
        self._interrupt_mode = active
        if active:
            self._send_btn.config(
                text="✕ STOP",
                bg=COLORS["btn_stop"],
                fg=COLORS["btn_stop_fg"],
            )
        else:
            self._send_btn.config(
                text="SEND ▶",
                bg=COLORS["btn_send"],
                fg=COLORS["accent"],
            )

    def set_enabled(self, enabled: bool):
        """Enable/disable text input (bukan tombol Stop)."""
        state = "normal" if enabled else "disabled"
        self._entry.config(state=state)

    def focus(self):
        self._entry.focus_set()


# ══════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════

class MEIApp:
    """
    Aplikasi utama MEI Desktop UI.

    Parameters
    ----------
    agent_callback : callable(user_text: str, mode: str, on_token: callable) -> str
        Dipanggil di background thread saat user kirim pesan.
        on_token(delta: str) dipanggil per token streaming — forward ke UI.

    stt_loop_fn : callable(on_text: callable) -> None
        Dipanggil saat mic diaktifkan.

    stt_stop_fn : callable() -> None
        Dipanggil saat mic dinonaktifkan.
    """

    def __init__(
        self,
        agent_callback: Callable = None,
        stt_loop_fn   : Optional[Callable] = None,
        stt_stop_fn   : Optional[Callable] = None,
    ):
        self._agent_cb    = agent_callback or _dummy_agent
        self._stt_loop_fn = stt_loop_fn
        self._stt_stop_fn = stt_stop_fn

        self._busy            = False
        self._msg_queue       = queue.Queue()
        self._interrupt_event = threading.Event()

        # ── v4.5.8: Streaming state ───────────────────────────────
        self._streaming_active = False
        self._streaming_bubble = None   # tk.Label — teks yang terus diupdate
        self._streaming_row    = None   # tk.Frame — row bubble
        self._streaming_text   = ""     # akumulasi teks streaming

        self._build_window()
        self._start_queue_processor()

    # ── Build Window ───────────────────────────────────────────────

    def _build_window(self):
        self.root = tk.Tk()
        self.root.title("MEI — Personal Assistant")
        self.root.geometry("620x720")
        self.root.minsize(480, 520)
        self.root.configure(bg=COLORS["bg"])
        self.root.resizable(True, True)

        # Title bar
        title_bar = tk.Frame(self.root, bg=COLORS["bg_panel"], pady=10)
        title_bar.pack(fill="x")
        tk.Label(
            title_bar, text="MEI",
            bg=COLORS["bg_panel"], fg=COLORS["accent"],
            font=FONTS["title"],
        ).pack(side="left", padx=16)
        self._conn_label = tk.Label(
            title_bar, text="● offline",
            bg=COLORS["bg_panel"], fg=COLORS["text_dim"],
            font=FONTS["mode"],
        )
        self._conn_label.pack(side="right", padx=16)

        tk.Frame(self.root, bg=COLORS["border"], height=1).pack(fill="x")

        self.notif_bar = NotifBar(self.root)
        self.notif_bar.pack(fill="x")

        tk.Frame(self.root, bg=COLORS["border"], height=1).pack(fill="x")

        self.mode_bar = ModeBar(
            self.root,
            on_mode_change=self._on_mode_change,
            on_mic_toggle=self._on_mic_toggle,
        )
        self.mode_bar.pack(fill="x", pady=(0, 1))

        tk.Frame(self.root, bg=COLORS["border"], height=1).pack(fill="x")

        self.chat = ChatArea(self.root)
        self.chat.pack(fill="both", expand=True)

        tk.Frame(self.root, bg=COLORS["border"], height=1).pack(fill="x")

        self.input_bar = InputBar(
            self.root,
            on_send=self._on_send,
            on_interrupt=self._on_interrupt_clicked,
        )
        self.input_bar.pack(fill="x")

        # Welcome message
        self.chat.add_system("MEI siap digunakan")
        self.chat.add_message("mei", "Halo! Ada yang bisa aku bantu?")
        self.input_bar.focus()

    # ── Queue Processor ────────────────────────────────────────────

    def _start_queue_processor(self):
        def _poll():
            while True:
                try:
                    cmd, data = self._msg_queue.get_nowait()
                    if cmd == "add_message":
                        self.chat.add_message(**data)
                    elif cmd == "add_system":
                        self.chat.add_system(data)
                    elif cmd == "set_typing":
                        self.chat.set_typing(data)
                    elif cmd == "push_notif":
                        self.notif_bar.push(**data)
                    elif cmd == "set_status":
                        self.mode_bar.set_status(**data)
                    elif cmd == "set_input_enabled":
                        self.input_bar.set_enabled(data)
                        if data:
                            self.input_bar.focus()
                    elif cmd == "set_interrupt_mode":
                        self.input_bar.set_interrupt_mode(data)
                    elif cmd == "set_conn":
                        online, label = data
                        self._conn_label.config(
                            text=f"● {label}",
                            fg=COLORS["accent"] if online else COLORS["text_dim"],
                        )
                    elif cmd == "inject_user_input":
                        self._on_send(data)
                except queue.Empty:
                    break
            self.root.after(80, _poll)

        self.root.after(80, _poll)

    def _ui(self, cmd, data):
        """Thread-safe UI update via queue."""
        self._msg_queue.put((cmd, data))

    # ── Interrupt ──────────────────────────────────────────────────

    def _on_interrupt_clicked(self):
        """Dipanggil saat tombol STOP diklik atau Enter ditekan saat busy."""
        self._interrupt_event.set()
        self._ui("set_status", {"text": "↩ interrupted", "color": COLORS["mic_on"]})
        self._ui("set_interrupt_mode", False)

    def get_interrupt_event(self) -> threading.Event:
        return self._interrupt_event

    def reset_interrupt(self):
        self._interrupt_event.clear()

    # ── Send ───────────────────────────────────────────────────────

    def _on_send(self, text: str):
        if self._busy:
            return
        self._busy = True
        self.chat.add_message("user", text)
        self._ui("set_input_enabled", False)
        self._ui("set_interrupt_mode", True)
        self._ui("set_typing", True)
        self._ui("set_status", {"text": "MEI sedang berpikir...", "color": COLORS["accent_warn"]})

        mode = self.mode_bar.current_mode
        self.reset_interrupt()

        def _run():
            try:
                # ── v4.5.8: on_token callback → thread-safe ke main thread ──
                def _on_token(delta: str):
                    self.root.after(0, lambda d=delta: self._stream_token(d))

                response = self._agent_cb(text, mode, on_token=_on_token)

                # Setelah stream selesai, finalize bubble
                self.root.after(0, self._finalize_stream)
                self._ui("set_status", {"text": "siap", "color": COLORS["accent"]})

            except Exception as e:
                self.root.after(0, self._finalize_stream)
                self._ui("set_typing", False)
                self._ui("add_message", {"role": "mei", "text": f"[error] {e}"})
                self._ui("set_status", {"text": "error", "color": COLORS["mic_on"]})
            finally:
                self._busy = False
                self._ui("set_interrupt_mode", False)
                self._ui("set_input_enabled", True)

        threading.Thread(target=_run, daemon=True).start()

    # ── v4.5.8: Streaming token methods ───────────────────────────

    def _stream_token(self, delta: str):
        """
        Dipanggil di main thread per delta token dari LLM.
        Token pertama: sembunyikan 'mengetik...' dan buat bubble baru.
        Token berikutnya: append ke label yang sama.
        Harus dipanggil via root.after() — TIDAK dari thread lain langsung.
        """
        inner = self.chat._inner

        if not self._streaming_active:
            # Token pertama — sembunyikan typing indicator, buat bubble
            self.chat.set_typing(False)
            self._streaming_active = True
            self._streaming_text   = ""

            # Buat row dan bubble MEI
            row = tk.Frame(inner, bg=COLORS["bg"], pady=4)
            row.pack(fill="x", padx=8)
            self._streaming_row = row

            bubble = tk.Frame(row, bg=COLORS["bg_bubble_mei"], padx=12, pady=8)
            bubble.pack(side="left")
            tk.Frame(row, bg=COLORS["bg"]).pack(side="right", fill="x", expand=True)

            # Header: nama + timestamp
            ts = datetime.now().strftime("%H:%M")
            header = tk.Frame(bubble, bg=COLORS["bg_bubble_mei"])
            header.pack(fill="x")
            tk.Label(
                header, text="MEI",
                bg=COLORS["bg_bubble_mei"], fg=COLORS["accent2"],
                font=FONTS["chat_name"],
            ).pack(side="left")
            tk.Label(
                header, text=ts,
                bg=COLORS["bg_bubble_mei"], fg=COLORS["text_dim"],
                font=FONTS["timestamp"],
            ).pack(side="right", padx=(8, 0))

            # Label teks — ini yang akan diupdate setiap token
            self._streaming_bubble = tk.Label(
                bubble, text="",
                bg=COLORS["bg_bubble_mei"], fg=COLORS["text_mei"],
                font=FONTS["chat"],
                wraplength=420, justify="left", anchor="w",
            )
            self._streaming_bubble.pack(fill="x", pady=(4, 0))

        # Append delta dan update label
        self._streaming_text += delta
        self._streaming_bubble.config(text=self._streaming_text)
        self.chat._scroll_bottom()

    def _finalize_stream(self):
        """
        Dipanggil di main thread setelah streaming selesai (atau error/interrupt).
        Reset semua streaming state. Bubble yang sudah dibuat tetap tampil.
        Harus dipanggil via root.after() — TIDAK dari thread lain langsung.
        """
        # Kalau streaming belum pernah dimulai sama sekali (misal pure command
        # yang return sebelum ada token), pastikan typing indicator hilang
        if not self._streaming_active:
            self.chat.set_typing(False)

        self._streaming_active = False
        self._streaming_bubble = None
        self._streaming_row    = None
        self._streaming_text   = ""

    # ── Mode ───────────────────────────────────────────────────────

    def _on_mode_change(self, mode: str):
        labels = {"1": "Text", "2": "TTS", "3": "STT", "4": "STT+RVC"}
        self.chat.add_system(f"Mode: {labels.get(mode, mode)}")
        self._ui("set_status", {"text": f"mode {labels.get(mode, mode)}", "color": COLORS["text_dim"]})

        if self.mode_bar.mic_active:
            self._stop_stt()
            if mode in ("3", "4"):
                self._start_stt()

    # ── Mic / STT ──────────────────────────────────────────────────

    def _on_mic_toggle(self, active: bool):
        if active:
            self.chat.add_system("Mic aktif — mendengarkan...")
            self._ui("set_status", {"text": "🎙 mendengarkan", "color": COLORS["mic_on"]})
            self._start_stt()
        else:
            self.chat.add_system("Mic nonaktif")
            self._ui("set_status", {"text": "mic off", "color": COLORS["text_dim"]})
            self._stop_stt()

    def _start_stt(self):
        if self._stt_loop_fn:
            def _on_stt_text(text: str):
                self._ui("inject_user_input", text)
            threading.Thread(
                target=self._stt_loop_fn,
                args=(_on_stt_text,),
                daemon=True,
                name="UISTTLoop",
            ).start()

    def _stop_stt(self):
        if self._stt_stop_fn:
            self._stt_stop_fn()

    # ── Public API ─────────────────────────────────────────────────

    def push_notif(self, message: str, tag: str = "info"):
        """Thread-safe: push notifikasi proaktif."""
        self._ui("push_notif", {"message": message, "tag": tag})

    def add_mei_message(self, text: str):
        """Thread-safe: tambah pesan MEI dari proactive engine."""
        self._ui("add_message", {"role": "mei", "text": text})

    def send_as_user(self, text: str):
        """Thread-safe: kirim teks seolah user mengetik."""
        self._ui("inject_user_input", text)

    def set_online(self, online: bool, label: str = None):
        """Update status koneksi di title bar."""
        self._ui("set_conn", (online, label or ("online" if online else "offline")))

    def run(self):
        """Jalankan main loop tkinter."""
        self.root.mainloop()


# ══════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = MEIApp(agent_callback=_dummy_agent)

    def _demo_notif():
        time.sleep(2)
        app.push_notif("Briefing Tim Fotografi — 30 menit lagi", tag="warn")
        time.sleep(3)
        app.push_notif("Timer pomodoro selesai!", tag="success")
        time.sleep(3)
        app.push_notif("Jadwal jogging pagi jam 05:30", tag="info")

    threading.Thread(target=_demo_notif, daemon=True).start()
    app.set_online(True, "LM Studio connected")
    app.run()