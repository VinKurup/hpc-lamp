"""CursorAttention — dev scaffold driver.

The browser reports cursor position and whether it hovers on/near the lamp;
this driver turns those events into AttentionSignals. Swapping in
CameraAttention (M2) is a one-line change in server.py.
"""

from .attention import AttentionSignal


class CursorAttention:
    def __init__(self, stale_after: float = 1.0):
        self.stale_after = stale_after  # no browser event for this long = nobody there
        self._x: float | None = None
        self._y: float | None = None
        self._over = False
        self._at: float | None = None

    def update(self, x: float, y: float, over_lamp: bool, now: float) -> None:
        """Called by the server on every cursor message from the viz."""
        self._x, self._y, self._over, self._at = x, y, over_lamp, now

    def poll(self, now: float) -> AttentionSignal:
        fresh = self._at is not None and now - self._at <= self.stale_after
        if not fresh:
            return AttentionSignal(engaged=False, target=None, confidence=0.0, timestamp=now)
        return AttentionSignal(
            engaged=self._over,
            target=(self._x, self._y),
            confidence=1.0,
            timestamp=now,
        )
