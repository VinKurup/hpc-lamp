"""Rolling-window smoothing over raw AttentionSignals, applied before the FSM.

Near pass-through for the noise-free cursor driver; earns its keep against
camera jitter in M2.
"""

from collections import deque

from .attention import AttentionSignal


class SignalSmoother:
    def __init__(self, window: float = 0.5):
        self.window = window
        self._buf: deque[AttentionSignal] = deque()

    def push(self, sig: AttentionSignal) -> AttentionSignal:
        self._buf.append(sig)
        while self._buf and sig.timestamp - self._buf[0].timestamp > self.window:
            self._buf.popleft()
        present = [s for s in self._buf if s.confidence > 0]
        if not present:
            return AttentionSignal(engaged=False, target=None, confidence=0.0, timestamp=sig.timestamp)
        engaged_frac = sum(s.engaged for s in present) / len(present)
        target = next((s.target for s in reversed(present) if s.target is not None), None)
        confidence = sum(s.confidence for s in present) / len(self._buf)
        return AttentionSignal(
            engaged=engaged_frac >= 0.5,
            target=target,
            confidence=confidence,
            timestamp=sig.timestamp,
        )
