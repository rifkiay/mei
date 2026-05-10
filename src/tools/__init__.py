"""
src/tools/__init__.py
=====================
Registry semua custom tools MEI.

SetTimerTool diinstansiasi di sini dengan placeholder callback.
Callback nyata (notif_pro.on_timer_done) di-inject via
`set_timer_tool.rebind(callback)` setelah NotifProactive dibuat di main.py.
"""

from .camera_capture import CameraCaptureTool
from .internet_search import InternetSearch
from .calendar_tool import (
    CalendarTool,
    CreateEventTool,
    GetEventsTool,
    DeleteEventTool,
)
from .timer_tool import SetTimerTool, ListTimersTool, CancelTimerTool


# ── Placeholder callback ───────────────────────────────────────────
# Dipakai saat SetTimerTool diinstansiasi sebelum NotifProactive ada.
# Akan di-replace lewat set_timer_tool.rebind() di main.py.
def _placeholder_timer_done(message: str):
    print(f"  [Timer] {message}", flush=True)


# ── Shared instances ───────────────────────────────────────────────
calendar_instance = CalendarTool()

create_event_tool = CreateEventTool(calendar_instance)
get_events_tool   = GetEventsTool(calendar_instance)
delete_event_tool = DeleteEventTool(calendar_instance)

set_timer_tool    = SetTimerTool(on_timer_done=_placeholder_timer_done)
list_timer_tool   = ListTimersTool()
cancel_timer_tool = CancelTimerTool()


def get_base_tools(camera_capture, internet_search) -> list:
    """
    Kembalikan list tools yang siap dipakai oleh Assistant.
    camera_capture dan internet_search diinject dari main.py
    karena butuh path / config khusus.
    """
    return [
        camera_capture,
        internet_search,
        create_event_tool,
        get_events_tool,
        delete_event_tool,
        set_timer_tool,
        list_timer_tool,
        cancel_timer_tool,
    ]


__all__ = [
    # Classes
    "CameraCaptureTool",
    "InternetSearch",
    "CalendarTool",
    "CreateEventTool",
    "GetEventsTool",
    "DeleteEventTool",
    "SetTimerTool",
    "ListTimersTool",
    "CancelTimerTool",
    # Shared instances
    "calendar_instance",
    "create_event_tool",
    "get_events_tool",
    "delete_event_tool",
    "set_timer_tool",
    "list_timer_tool",
    "cancel_timer_tool",
    # Helpers
    "get_base_tools",
]