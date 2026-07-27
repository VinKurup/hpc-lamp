# Demo video — shot list

One continuous screen recording (~90s) with the webcam feed of you visible if
possible (QuickTime: File → New Screen Recording; drag a small QuickTime
camera window into a corner so graders see cause and effect).

Prep (before recording):
1. Fresh memory so the recall answer is crisp: `rm -f memory.db && rm -rf thumbnails`
2. `source .venv/bin/activate && python server.py --camera`
3. Open `localhost:8000`, reload once, click the page once (enables chirp audio).
4. Put a mug and your phone in clear camera view, wait ~15s so they're remembered.

Shots (the 4-step challenge scenario):

1. **Greet + track (0:00–0:20)** — Sit down, look at the screen. Lamp perks
   up, chirps, wiggles, then tracks as you lean left and right. Lean far once
   so the base visibly recruits after the head.
2. **Disengage (0:20–0:35)** — Turn away to a book/second screen. ~3s later
   the lamp droops and dims. Narrate nothing; let it read.
3. **Attention-seek (0:35–0:55)** — Keep ignoring it. At ~10s of disengagement
   it perks, flashes, chirps twice, then gives up. Glance back on the second
   flash if you want the "seek succeeds" beat instead.
4. **Voice recall (0:55–1:30)** — Hold the 🎤 button: "Where is my mug?"
   Release. Lamp does the listening pose, transcript appears, spoken answer
   names position + when last seen. Bonus second question: "When did you last
   see my phone?"

Optional closer: move the mug somewhere else in view, wait ~10s, ask again —
shows memory updating.
