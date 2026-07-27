"""FSM hysteresis test: simulates the M1 demo scenario at 30Hz.

Run: .venv/bin/python tests/test_fsm.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lelamp.behavior.fsm import EngagementFSM, State
from lelamp.perception.attention import AttentionSignal

DT = 1 / 30


def feed(fsm, t, duration, engaged, present=True):
    """Feed constant signals for `duration`; return (end_time, transitions)."""
    transitions = []
    end = t + duration
    while t < end:
        sig = AttentionSignal(
            engaged=engaged and present,
            target=(0.5, 0.3) if present else None,
            confidence=1.0 if present else 0.0,
            timestamp=t,
        )
        _, tr = fsm.step(sig, t)
        if tr:
            transitions.append(tr)
        t += DT
    return t, transitions


def test_demo_scenario():
    fsm = EngagementFSM()
    t = 0.0

    # Hover on the lamp: ENGAGED after ~1s, not instantly.
    t, trs = feed(fsm, t, 0.5, engaged=True)
    assert trs == [], f"engaged too early: {trs}"
    t, trs = feed(fsm, t, 1.0, engaged=True)
    assert trs == [(State.IDLE, State.ENGAGED)], trs

    # Brief glance away (< 3s) must NOT disengage.
    t, trs = feed(fsm, t, 2.0, engaged=False)
    assert trs == [], f"disengaged too early: {trs}"
    t, trs = feed(fsm, t, 1.5, engaged=True)
    assert trs == [], trs

    # Sustained looking-away: DISENGAGED after ~3s.
    t, trs = feed(fsm, t, 3.5, engaged=False)
    assert trs == [(State.ENGAGED, State.DISENGAGED)], trs

    # Stay away: SEEKING fires after ~10s, attempt lasts ~2.5s, then back.
    t, trs = feed(fsm, t, 13.5, engaged=False)
    assert trs == [
        (State.DISENGAGED, State.SEEKING),
        (State.SEEKING, State.DISENGAGED),
    ], trs

    # Second failed attempt: give up and back off to IDLE.
    t, trs = feed(fsm, t, 13.5, engaged=False)
    assert trs == [
        (State.DISENGAGED, State.SEEKING),
        (State.SEEKING, State.IDLE),
    ], trs

    # Backed off: no third seek even after a long wait.
    t, trs = feed(fsm, t, 15.0, engaged=False)
    assert trs == [], f"sought after backoff: {trs}"

    # Re-engaging still works from IDLE.
    t, trs = feed(fsm, t, 1.5, engaged=True)
    assert trs == [(State.IDLE, State.ENGAGED)], trs


def test_seeking_success_engages():
    fsm = EngagementFSM()
    t = 0.0
    t, _ = feed(fsm, t, 1.5, engaged=True)           # ENGAGED
    t, _ = feed(fsm, t, 3.5, engaged=False)          # DISENGAGED
    t, trs = feed(fsm, t, 10.2, engaged=False)       # SEEKING fires
    assert trs == [(State.DISENGAGED, State.SEEKING)], trs
    t, trs = feed(fsm, t, 0.8, engaged=True)         # user responds: fast engage
    assert trs == [(State.SEEKING, State.ENGAGED)], trs


def test_presence_lost_resets_to_idle():
    fsm = EngagementFSM()
    t = 0.0
    t, _ = feed(fsm, t, 1.5, engaged=True)
    t, trs = feed(fsm, t, 6.0, engaged=False, present=False)
    # Away-timer fires before the presence timeout: droop first, then settle to IDLE.
    assert trs[-1] == (State.DISENGAGED, State.IDLE), trs
    assert fsm.state is State.IDLE


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all FSM tests passed")
