"""Speech-to-text: faster-whisper (small, int8, CPU).

The model takes a few seconds to load, so preload() warms it in a background
thread at server startup; transcribe_bytes() blocks until it's ready. Audio
arrives as whatever container the browser's MediaRecorder produced (webm/opus
usually) — PyAV inside faster-whisper decodes it from the temp file.
"""

import tempfile
import threading
from pathlib import Path

MODEL_SIZE = "small"


class Transcriber:
    def __init__(self, model_size: str = MODEL_SIZE):
        self.model_size = model_size
        self._model = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel  # deferred: heavy import

                self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self._model

    def preload(self) -> None:
        threading.Thread(target=self._ensure_loaded, daemon=True).start()

    def transcribe_bytes(self, data: bytes, suffix: str = ".webm") -> str:
        """Blocking — call via asyncio.to_thread."""
        model = self._ensure_loaded()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            path = f.name
        try:
            segments, _info = model.transcribe(path, vad_filter=True)
            return " ".join(seg.text.strip() for seg in segments).strip()
        finally:
            Path(path).unlink(missing_ok=True)
