"""Interface 2 — LampCommand.

The behavior engine emits pose targets, light states, and sound cues; the
renderer executes them with easing and knows nothing about why.
"""

from dataclasses import dataclass

# The lamp's 6 DOF, in radians.
JOINTS = (
    "base_yaw",
    "shoulder_pitch",
    "elbow_pitch",
    "wrist_pitch",
    "head_yaw",
    "head_pitch",
)


@dataclass
class LampCommand:
    joints: dict[str, float] | None = None  # keys from JOINTS, radians
    light: dict | None = None               # {"intensity": 0..1, "color": "#rrggbb"}
    sound: str | None = None                # cue name, e.g. "chirp"

    def to_message(self) -> dict:
        msg: dict = {"type": "command"}
        if self.joints is not None:
            msg["joints"] = self.joints
        if self.light is not None:
            msg["light"] = self.light
        if self.sound is not None:
            msg["sound"] = self.sound
        return msg
