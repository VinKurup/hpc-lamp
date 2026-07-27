"""Interface 1 — AttentionSignal.

Everything upstream of the engagement FSM produces this; the FSM consumes
nothing else. Drivers (cursor in M1, camera in M2) are interchangeable.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class AttentionSignal:
    engaged: bool                        # is the user attending to the lamp
    target: tuple[float, float] | None   # normalized (x, y) of the user's face/attention
    confidence: float
    timestamp: float


class AttentionDriver(Protocol):
    """A source of AttentionSignals: CursorAttention (M1), CameraAttention (M2)."""

    def poll(self, now: float) -> AttentionSignal: ...
