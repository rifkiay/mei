from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

from qwen_agent.tools.base import BaseTool, register_tool


# root project/storage/event/calendar.json
ROOT_DIR = Path(__file__).resolve().parents[2]
EVENT_DIR = ROOT_DIR / "storage" / "event"
EVENT_FILE = EVENT_DIR / "calendar.json"


class CalendarTool:
    def __init__(self, filepath: Union[str, Path] = EVENT_FILE):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.events = self.load()

    def load(self):
        if not self.filepath.exists():
            return []

        try:
            with self.filepath.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

    def save(self):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with self.filepath.open("w", encoding="utf-8") as f:
            json.dump(self.events, f, indent=2, ensure_ascii=False)

    def create_event(self, title: str, datetime_str: str):
        self.events = self.load() 
        # Validasi format datetime ISO 8601
        try:
            datetime.fromisoformat(datetime_str)
        except ValueError:
            return {
                "status": "error",
                "message": "datetime harus format ISO 8601, contoh: 2026-04-16T09:00:00"
            }

        event = {
            "id": str(uuid.uuid4()),
            "title": title,
            "datetime": datetime_str
        }

        self.events.append(event)
        self.save()

        return {
            "status": "success",
            "message": "event dibuat",
            "event": event
        }

    def get_events(self, date: str):
        self.events = self.load()
        results = [
            e for e in self.events
            if e.get("datetime", "").startswith(date)
        ]
        results.sort(key=lambda x: x["datetime"])

        return {
            "status": "success",
            "date": date,
            "events": results
        }

    def delete_event(self, event_id: str):
        self.events = self.load()
        before = len(self.events)
        self.events = [e for e in self.events if e.get("id") != event_id]

        if len(self.events) == before:
            return {
                "status": "error",
                "message": f"Event {event_id} tidak ditemukan"
            }

        self.save()

        return {
            "status": "success",
            "message": f"Event {event_id} dihapus"
        }


def _parse_params(params: Union[str, dict]) -> dict:
    if isinstance(params, str):
        try:
            return json.loads(params)
        except json.JSONDecodeError:
            return {}
    return params or {}


@register_tool("create_event")
class CreateEventTool(BaseTool):
    name = "create_event"
    description = "Membuat event di calendar"

    parameters = [
        {
            "name": "title",
            "type": "string",
            "description": "Judul event",
            "required": True,
        },
        {
            "name": "datetime",
            "type": "string",
            "description": "Datetime ISO 8601, contoh: 2026-04-16T09:00:00",
            "required": True,
        },
    ]

    def __init__(self, calendar: CalendarTool = None):
        super().__init__()
        self.calendar = calendar or CalendarTool()

    def call(self, params: Union[str, dict], **kwargs):
        params = _parse_params(params)

        if "title" not in params or "datetime" not in params:
            return json.dumps(
                {
                    "status": "error",
                    "message": "Parameter title dan datetime wajib diisi"
                },
                ensure_ascii=False
            )

        result = self.calendar.create_event(
            params["title"],
            params["datetime"]
        )
        return json.dumps(result, ensure_ascii=False)


@register_tool("get_events")
class GetEventsTool(BaseTool):
    name = "get_events"
    description = "Mengambil event berdasarkan tanggal (YYYY-MM-DD)"

    parameters = [
        {
            "name": "date",
            "type": "string",
            "description": "Tanggal format YYYY-MM-DD",
            "required": True,
        }
    ]

    def __init__(self, calendar: CalendarTool = None):
        super().__init__()
        self.calendar = calendar or CalendarTool()

    def call(self, params: Union[str, dict], **kwargs):
        params = _parse_params(params)

        if "date" not in params:
            return json.dumps(
                {
                    "status": "error",
                    "message": "Parameter date wajib diisi"
                },
                ensure_ascii=False
            )

        result = self.calendar.get_events(params["date"])
        return json.dumps(result, ensure_ascii=False)


@register_tool("delete_event")
class DeleteEventTool(BaseTool):
    name = "delete_event"
    description = "Menghapus event berdasarkan ID"

    parameters = [
        {
            "name": "event_id",
            "type": "string",
            "description": "ID event yang akan dihapus",
            "required": True,
        }
    ]

    def __init__(self, calendar: CalendarTool = None):
        super().__init__()
        self.calendar = calendar or CalendarTool()

    def call(self, params: Union[str, dict], **kwargs):
        params = _parse_params(params)

        if "event_id" not in params:
            return json.dumps(
                {
                    "status": "error",
                    "message": "Parameter event_id wajib diisi"
                },
                ensure_ascii=False
            )

        result = self.calendar.delete_event(params["event_id"])
        return json.dumps(result, ensure_ascii=False)