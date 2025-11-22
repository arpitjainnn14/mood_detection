import json
import os
from typing import Any


class Settings:
    """Simple JSON-backed settings for the application."""

    def __init__(self, path: str = 'settings.json') -> None:
        self.path = path
        self._data = {}
        self._defaults = {
            'theme': 'dark',
            'camera_index': 0,
            'detection_interval': 30,
            'show_fps': True,
            'save_screenshots': True,
            'emotion_smoothing': 2,  # Reduced from 3 for faster processing
            'min_face_size': 40,  # Increased for better performance
            'detection_quality': 'performance',  # Changed from 'balanced' for speed
        }
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}
        # merge defaults
        for k, v in self._defaults.items():
            self._data.setdefault(k, v)

    def save(self) -> None:
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=2)
        except Exception:
            pass

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self.save()

    def reset(self) -> None:
        self._data = dict(self._defaults)
        self.save()