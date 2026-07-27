"""Engagement FSM: IDLE -> ENGAGED -> DISENGAGED -> SEEKING.

Hysteresis everywhere, never instant transitions. Thresholds are constructor
params so M2 can re-tune them against real faces.
"""

from enum import Enum

from lelamp.perception.attention import AttentionSignal


class State(Enum):
    IDLE = "idle"                # nobody there
    ENGAGED = "engaged"          # user attending to the lamp
    DISENGAGED = "disengaged"    # user present but looked away
    SEEKING = "seeking"          # actively trying to win attention back


class EngagementFSM:
    def __init__(
        self,
        engage_after: float = 1.0,       # sustained attention before ENGAGED
        disengage_after: float = 3.0,    # sustained looking-away before DISENGAGED
        seek_after: float = 10.0,        # time in DISENGAGED before SEEKING
        seek_duration: float = 2.5,      # length of one attention-seeking attempt
        max_seek_attempts: int = 2,      # back off (IDLE) after this many failures
        presence_timeout: float = 5.0,   # no signal for this long = user gone
    ):
        self.engage_after = engage_after
        self.disengage_after = disengage_after
        self.seek_after = seek_after
        self.seek_duration = seek_duration
        self.max_seek_attempts = max_seek_attempts
        self.presence_timeout = presence_timeout

        self.state = State.IDLE
        self._engaged_since: float | None = None
        self._away_since: float | None = None
        self._disengaged_at: float | None = None
        self._seek_started: float | None = None
        self._seek_attempts = 0
        self._last_present: float | None = None

    def step(self, sig: AttentionSignal, now: float):
        """Returns (state, transition). transition is (old, new) or None."""
        prev = self.state
        present = sig.confidence > 0
        if present:
            self._last_present = now

        # Track continuous spans of engagement / looking-away.
        if present and sig.engaged:
            if self._engaged_since is None:
                self._engaged_since = now
            self._away_since = None
        else:
            if self._away_since is None:
                self._away_since = now
            self._engaged_since = None

        gone = self._last_present is None or now - self._last_present > self.presence_timeout

        if gone:
            self.state = State.IDLE
            self._seek_attempts = 0
            self._seek_started = None
            self._disengaged_at = None
        elif self.state in (State.IDLE, State.DISENGAGED):
            if self._engaged_since is not None and now - self._engaged_since >= self.engage_after:
                self.state = State.ENGAGED
                self._seek_attempts = 0
            elif (
                self.state is State.DISENGAGED
                and self._seek_attempts < self.max_seek_attempts
                and now - self._disengaged_at >= self.seek_after
            ):
                self.state = State.SEEKING
                self._seek_started = now
        elif self.state is State.ENGAGED:
            if self._away_since is not None and now - self._away_since >= self.disengage_after:
                self.state = State.DISENGAGED
                self._disengaged_at = now
        elif self.state is State.SEEKING:
            # Seeking worked: engage on a shorter fuse.
            if self._engaged_since is not None and now - self._engaged_since >= self.engage_after * 0.5:
                self.state = State.ENGAGED
                self._seek_attempts = 0
            elif now - self._seek_started >= self.seek_duration:
                self._seek_attempts += 1
                if self._seek_attempts >= self.max_seek_attempts:
                    self.state = State.IDLE  # give up, back off
                else:
                    self.state = State.DISENGAGED
                    self._disengaged_at = now

        transition = None if self.state is prev else (prev, self.state)
        return self.state, transition
