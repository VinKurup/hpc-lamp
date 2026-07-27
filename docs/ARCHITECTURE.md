# LeLamp — Architecture & Design Writeup

A 6-DOF desk lamp (rendered in three.js) that detects when you're paying
attention, reacts with expressive motion/light/sound, remembers objects it
sees, and answers questions about them by voice. Everything perceptual runs
locally at 30fps; only the conversational path touches the cloud.

## System overview

```
                      ┌────────────────────── Python (server.py, 30Hz tick) ─────────────────────┐
 webcam ──frames──> CameraAttention ──AttentionSignal──> Smoother ──> Engagement FSM ──state──┐  │
    │                 (MediaPipe head pose)                (rolling window)   (hysteresis)     │  │
    │                                                                                          ▼  │
    │                                                                                 BehaviorEngine
    └──frames──> SceneObserver ──detections──> MemoryStore (SQLite + thumbnails)        │  (poses,
                  (YOLOv8n @ ~5s)                    ▲                                  │   light,
                                                     │ query_memory tool                │   sound)
 mic (browser) ──audio──> faster-whisper ──text──> LLM (Claude via OpenRouter) ──text──>│──> say
                                                                                        │
                                                                              LampCommand over WS
                                                                                        ▼
                                                                        three.js lamp (browser)
                                                                        springs, no logic
```

Two narrow interfaces carry everything:

- **AttentionSignal** `{engaged, target, confidence, timestamp}` — everything
  upstream of the FSM produces this; the FSM consumes nothing else. Drivers
  are interchangeable: `CursorAttention` (dev scaffold) and `CameraAttention`
  swap with a CLI flag (`--camera`).
- **LampCommand** `{joints, light, sound}` — the behavior engine emits
  targets; the renderer executes them with critically damped springs and
  knows nothing about why. All behavior semantics live server-side.

## Engagement pipeline

**Proxy definition.** "Looking at the lamp" is ill-posed when the lamp lives
on a monitor, so engagement = *face detected AND head roughly facing the
screen* (|yaw| ≤ 25°, pitch in [−30°, +20°], from MediaPipe's facial
transformation matrix). This assumption is stated rather than pretending to
true gaze tracking.

**FSM.** `IDLE → ENGAGED → DISENGAGED → SEEKING`, with hysteresis on every
edge: engage after 1.0s sustained attention, disengage after 3.0s sustained
looking-away, seek after 10s disengaged, give up (back to IDLE) after 2
failed seek attempts. A rolling-window majority vote smooths the raw signal
before the FSM sees it. Instant transitions are never allowed — the smoothing
+ hysteresis combination is what prevents flicker at the attention boundary.

**Behavior.** One-shot behaviors (greet with head-wiggle + chirp, droop,
attention-seek light-flash) run as small timed scripts; steady ENGAGED
tracking emits a fresh pose every tick. Two animation decisions matter most
for "creature, not CNC":

- *Layered gaze*: head yaw absorbs tracking error first (fast, ±0.5 rad);
  the base recruits only for the residual — eyes-then-torso, not
  whole-body swivel.
- *Spring easing*: per-joint critically damped springs (head stiff ~130,
  base soft ~30) make response speed proportional to target speed — quick
  human motion produces a quick head-flick with the body settling behind.

## Memory pipeline

Every ~5s (or within 3s of a scene change, detected by downscaled frame
diff), one camera frame goes through YOLOv8n. Detections above 0.55
confidence upsert into SQLite with the dedupe policy: **same label whose
center is within 0.15 (normalized) of a record seen in the last 10 minutes =
same object** — update position/`times_seen`/thumbnail instead of inserting.
`person` is never stored; the person belongs to the attention pipeline.

Storage is deliberately honest: every detection is recorded, including
one-shot detector ghosts. Filtering happens at *query* time (`min_seen=2`),
so answers exclude ghosts without falsifying the observation log.

## Conversation pipeline

Push-to-talk in the browser (MediaRecorder) → audio blob over the existing
WebSocket → faster-whisper (small, int8, CPU, worker thread) → Claude via
OpenRouter with one tool, `query_memory(label)` → short spoken-style answer →
macOS `say`. The lamp holds an attentive "listening" pose with a cool light
tint while the mic is open.

The tool returns *pre-humanized* sightings — `"2 minutes ago"`, `"to your
left"` — computed deterministically in Python (including the mirror flip from
camera frame to user frame). The LLM composes prose; it never interprets
coordinates. This keeps spatial correctness testable without an LLM in the
loop.

## Design decisions & tradeoffs

**Rule-based FSM over learned engagement classifier.** Four states and four
thresholds are debuggable, tunable live, and need zero training data. A
classifier is the "touch weights" path: it would need labeled data we'd have
to collect anyway (see eval harness) and would turn every tuning question
into a retraining question. At this scale the FSM wins outright.

**Cursor scaffold first, camera second.** The behavior library, FSM, and
renderer were built and tuned against a noise-free cursor driver, so
perception noise never confounded behavior bugs. The swap to camera was one
flag, as designed. The scaffold also *lied* twice, instructively: cursor
input is noise-free (thresholds tuned on it don't survive real faces), and on
macOS, background windows receive no mouse-move events — which looked exactly
like an FSM bug until reproduced synthetically. Verifying against the
deterministic driver before blaming the pipeline became the debugging pattern
for the whole project.

**Web viz over native/pygame.** three.js gave real 6-DOF articulation,
shadows, and an emissive light cone for roughly the same effort a flat pygame
sketch would have cost, at the price of a WebSocket hop (~ms, measured).
Staging turned out to matter as much as rigging: the camera must sit on the
side the joints bend toward (three-quarter *front* view), the head carries a
~90° Luxo bend so the beam faces outward, and a yaw offset makes "neutral"
mean "facing the viewer." All of that lives in the behavior engine's pose
constants — the renderer stays dumb.

**Local perception, cloud conversation.** The 30fps loop (MediaPipe) and the
~5s loop (YOLO) must be local; the conversational path tolerates seconds of
latency, so it's the only cloud dependency — and it's behind a 30s timeout
(the SDK default of 600s turns any hang into a ten-minute frozen "…").

**YOLO local now, vision-LLM labels later.** COCO-80 covers cups/phones/
laptops but not "keys" and can't distinguish "the blue mug." The planned
hybrid keeps YOLO in the loop for cadence and adds open-vocabulary labels via
a vision-LLM pass. Region-based identity also cannot survive large moves —
a cup teleported across the desk is a new record by design. Appearance-based
re-ID is out of scope and stated as a limitation, not papered over with a
looser dedupe radius (which would merge genuinely distinct objects).

**Browser mic over PortAudio.** MediaRecorder → blob over the WebSocket
avoids native audio dependencies entirely and makes mic permission a browser
prompt. PyAV inside faster-whisper decodes whatever container arrives.

## Evaluation

**Engagement quality** (`lelamp/eval/record.py`, `replay.py`): the recorder
cues the subject through a script (`engaged:5,away:5,...`) with spoken
prompts, so ground-truth labels come from the schedule — no hand annotation.
Replay feeds recorded frames through the *identical* landmarker → signal →
smoother → FSM code and reports per-clip and aggregate precision/recall/F1,
raw-signal toggle count, and FSM flips vs expected (flips > expected =
flicker). Frames within 1.5s of a segment boundary are excluded — hysteresis
is *supposed* to lag there; boundary lag is a design constant, not an error.
Recommended clip set: plain, glasses, sitting at an angle, lights dimmed,
plus one absence clip.

Results (fill in after recording the clip set):

| clip set | P | R | F1 | fsm flips / expected |
|---|---|---|---|---|
| _pending_ | | | | |

**Latency** (`/latency` endpoint, rolling 2000 samples): p50/p95/max per
stage — `frame_to_decision` (capture → FSM verdict), `decision_to_command_sent`
(transition → WS broadcast; renderer adds ≤1 frame + spring rise),
`stt`, and `question_to_answer_start` (audio received → speech starts).

Results (fill in from a live session):

| stage | p50 | p95 |
|---|---|---|
| _pending_ | | |

## Known limitations

- Engagement is a head-pose proxy, not gaze; eyes-only glances don't count.
- Object identity is label+region; large moves fracture identity, and "keys"
  awaits open-vocabulary labels.
- TTS is macOS `say` — functional, not charming.
- Single user assumed (num_faces=1, nearest-face-wins).

## Bonus direction (if time allows)

Interruption awareness: VAD on the mic while the user is speaking suppresses
attention-seeking. Reuses the existing audio path and reads as social
judgment — the cheapest standout in the plan, and still true.
