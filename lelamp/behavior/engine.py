"""Behavior engine: maps FSM state (+ attention target) to LampCommands.

One-shot behaviors (greet, droop, attention-seek) run as small time-based
scripts; steady ENGAGED tracking emits a fresh pose every tick. The renderer
does the easing, so commands here are just targets.
"""

from .commands import LampCommand
from .fsm import State

# The arm's bend plane faces world -X, but the viz camera sits ~59° around
# from that; this yaw offset makes base_yaw's neutral point AT the viewer.
FACE_VIEWER_YAW = 1.0

# head_pitch carries a ~90° forward bend (Luxo-style) so the shade/beam faces
# outward from the arm, not along it: total tilt = pitch sum + head_pitch.
REST = {"base_yaw": FACE_VIEWER_YAW, "shoulder_pitch": 0.55, "elbow_pitch": -1.1,
        "wrist_pitch": 0.55, "head_yaw": 0.0, "head_pitch": 1.15}
PERK = {"base_yaw": FACE_VIEWER_YAW, "shoulder_pitch": 0.1, "elbow_pitch": -0.25,
        "wrist_pitch": 0.15, "head_yaw": 0.0, "head_pitch": 1.25}
DROOP = {"base_yaw": FACE_VIEWER_YAW, "shoulder_pitch": 0.95, "elbow_pitch": -2.0,
         "wrist_pitch": 1.35, "head_yaw": 0.0, "head_pitch": 1.9}
LISTEN = {"base_yaw": FACE_VIEWER_YAW, "shoulder_pitch": 0.3, "elbow_pitch": -0.55,
          "wrist_pitch": 0.35, "head_yaw": 0.25, "head_pitch": 1.05}  # attentive head-cock

WARM = "#fff2cc"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _greet_script(now: float) -> list[tuple[float, LampCommand]]:
    return [
        (now, LampCommand(joints=PERK, light={"intensity": 1.0, "color": WARM}, sound="chirp")),
        (now + 0.5, LampCommand(joints={**PERK, "head_yaw": 0.35})),
        (now + 0.85, LampCommand(joints={**PERK, "head_yaw": -0.35})),
        (now + 1.2, LampCommand(joints=PERK)),
    ]


def _seek_script(now: float) -> list[tuple[float, LampCommand]]:
    return [
        (now, LampCommand(joints=PERK, light={"intensity": 1.0, "color": WARM}, sound="chirp")),
        (now + 0.5, LampCommand(light={"intensity": 0.15})),
        (now + 0.9, LampCommand(light={"intensity": 1.0}, sound="chirp")),
        (now + 1.3, LampCommand(light={"intensity": 0.15})),
        (now + 1.7, LampCommand(light={"intensity": 1.0})),
        (now + 2.2, LampCommand(light={"intensity": 0.5})),
    ]


class BehaviorEngine:
    def __init__(self):
        self._script: list[tuple[float, LampCommand]] = []
        self._listening = False

    def set_listening(self, on: bool, state: State, now: float) -> None:
        """Push-to-talk: hold an attentive pose and suppress tracking while
        the user speaks; re-assert the current state's pose on release."""
        self._listening = on
        if on:
            self._script = [(now, LampCommand(joints=LISTEN, light={"intensity": 0.9, "color": "#cfe8ff"}))]
        elif state is State.DISENGAGED:
            self._script = [(now, LampCommand(joints=DROOP, light={"intensity": 0.12, "color": WARM}))]
        elif state is State.IDLE:
            self._script = [(now, LampCommand(joints=REST, light={"intensity": 0.3, "color": WARM}))]
        else:
            self._script = []  # ENGAGED/SEEKING: tracking or seek script resumes

    def step(self, state: State, transition, sig, now: float) -> LampCommand | None:
        if transition:
            _, new = transition
            if new is State.ENGAGED:
                self._script = _greet_script(now)
            elif new is State.SEEKING:
                self._script = _seek_script(now)
            elif new is State.DISENGAGED:
                self._script = [(now, LampCommand(joints=DROOP, light={"intensity": 0.12, "color": WARM}))]
            elif new is State.IDLE:
                self._script = [(now, LampCommand(joints=REST, light={"intensity": 0.3, "color": WARM}))]

        # Fire at most one due script step per tick (steps are >=0.3s apart).
        if self._script and self._script[0][0] <= now:
            return self._script.pop(0)[1]

        if self._listening:
            return None  # hold the listening pose; no tracking

        if state is State.ENGAGED and not self._script and sig.target is not None:
            return self._track(sig.target)
        return None

    def _track(self, target: tuple[float, float]) -> LampCommand:
        x, y = target  # normalized screen coords, (0,0) = top-left
        # Under the -X viz camera, increasing base_yaw turns the lamp toward
        # viewer-right, so dx must be positive when the target is right of
        # center — the lamp swings toward the side you moved to.
        dx = x - 0.5
        # Layered gaze: the head absorbs the yaw error first (fast, light
        # segment); the base only turns for the residual beyond the head's
        # range. Small target shifts move just the head, like eyes-then-torso.
        yaw_total = dx * 2.3
        head_yaw = _clamp(yaw_total, -0.5, 0.5)
        base_residual = _clamp(yaw_total - head_yaw, -1.1, 1.1)
        # Keep a visible coil while tracking — a straight arm reads as a post,
        # not a creature. Head pitch is centered on a face at ~0.6 frame height.
        return LampCommand(
            joints={
                "base_yaw": FACE_VIEWER_YAW + base_residual,
                "shoulder_pitch": 0.45,
                "elbow_pitch": -0.85,
                "wrist_pitch": 0.5,
                "head_yaw": head_yaw,
                "head_pitch": _clamp(1.45 + (y - 0.6) * 0.9, 0.9, 2.0),
            },
            light={"intensity": 1.0, "color": WARM},
        )
