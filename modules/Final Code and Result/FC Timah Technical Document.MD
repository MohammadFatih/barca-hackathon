# Our approach

## Overview

The challenge asks for one label per frame — `H`, `A`, `D` or `X` — across the full 199,319
frames of the match. We split the problem into two independent stages: **dead-ball detection**
(is the ball in play at all) and **possession decoding** (if it is, which team has it). The two
are composed so that possession decoding never runs across a stretch the first stage has marked
dead.

That split mirrors the challenge's own framing: dead-ball ownership is deliberately not asked
for, so the first stage only has to answer alive-or-dead, and the second stage only has to answer
home-or-away on the frames where that matters.

---

## Pipeline

The pipeline runs in three steps:

1. **Dead-ball mask.** Two independent signals are combined (logical OR): the provided Smart
   Tagging, read by `dead_ball.py`, and a second signal read directly off the tracking itself
   (`possession.stoppage_mask` — long stretches with no ball and players walking rather than
   playing).
2. **Possession decoding.** Everywhere the mask from step 1 says the ball is alive,
   `possession.py` decodes home/away from the tracking (player positions and ball position),
   one uninterrupted stretch of live play at a time.
3. **Label assembly.** The two outputs are merged frame by frame: `D` wherever step 1 marked the
   frame dead, otherwise `H` or `A` from step 2's decode (or `X` where neither has a call).

![Technical Diagram](Technical%20Diagram.png)

Pipeline Architecture Diagram

## Part 1 — dead-ball detection

### Method

The Smart Tagging XML tags five set-piece types (Throw-in, Corner Kick, Goal Kick, Free Kick,
and Kick Off) as intervals. Checked against the public ground truth, the tagged interval runs well
past the moment the ball is actually back in play — a throw-in is live again almost as soon as it
lands, long before the tag ends. We keep only the first `trim_frames` frames of each tagged
interval as the true dead-ball stretch; everything after that is treated as live.

`trim_frames` is the one tunable parameter of this stage.

### Choosing `trim_frames`

A value of 70 frames (2.8 s) was found first by hand, against the 5-minute public ground truth.
We then ran an automated search over `trim_frames = 1 until 500`, scored on the actual ranking metric
(`spellF1`) with the **full** pipeline — dead-ball feeding directly into possession decoding —
rather than scoring dead-ball accuracy on its own. That distinction mattered: scored in
isolation, a much larger `trim_frames` looked best, but that result didn't hold once real
possession decoding was layered on top, which is why the combined-pipeline search is the one we
trust. The search ran as a coarse pass (step 5) over the full range, then a fine pass (step 1)
around the coarse peak.

| Submission | `trim_frames` | macroF1 | spellF1 (ranking metric) |
|---|---|---|---|
| Hand-tuned guess (manual, pre-pipeline) | 70 | n/a | 0.4471 |
| Automated two-phase grid search (this submission) | 69 | 0.7766 | 0.4828 |

The automated search converged on `trim_frames = 69` — within one frame of the original
hand-tuned guess, which confirms that guess was already close to optimal once scored against the
real, combined pipeline rather than against dead-ball detection in isolation. (Scored in
isolation, a substantially larger `trim_frames` looked best, but that result did not survive
contact with the actual possession decoder — see Limitations.)

---

## Part 2 — possession decoding

### Evidence

Three sources, each expressed as a log-odds in favour of the home team:

- **Ball position** — the home/away log-odds comes from which team's nearest tracked player is
  closer to the ball, scaled by that margin and faded toward zero as the ball moves out of
  anyone's reach. A loose ball far from both teams is treated as ambiguous rather than assigned,
  in line with the challenge's own definition of *controlled* possession.
- **Player formation** — a logistic regression over collective features of the two teams' shape
  (width, forward velocity, mean speed, territory, how deep the back line sits), built so that
  swapping the teams flips the sign. It is trained on this match's own pseudo-labels (frames
  where the ball sits at one team's feet long enough to be unambiguous) with the match split into
  time blocks so every prediction is out-of-fold. This is the only evidence available on the 41%
  of frames where the ball itself is not tracked.
- **Persistence** — a fixed cost is charged for every change of team, set from an estimate of how
  often the ball actually changes hands, so no single ambiguous frame can flip the call by itself.

### Decoding

The three signals are combined per frame and decoded with the Viterbi algorithm, independently
within each uninterrupted stretch of live play — decoding restarts at every dead-ball boundary
from Part 1, since nothing about who had the ball before a stoppage says anything about who
restarts it.

### What was *not* tuned on this match

Three constants — how long a clear hand-over must persist before it's believed, and how much
weight the formation model gets on frames with and without a tracked ball — were calibrated on
two other, publicly available matches with real event-derived possession labels, not on the
provided match. The provided match carries no possession ground truth outside its 5-minute public
excerpt, and the rules are explicit that labels must come from an algorithm, not from tuning
against hand-inspected answers. The sample matches' clean tracking was first degraded (ball
smoothing, synthetic player drop-out) to match the noise profile measured — label-free — on the
provided match's own tracking, so the calibration transfers rather than assuming clean data.

---

## Validation

All comparisons above the full-match leaderboard use the 5-minute public ground truth (17
possessions across both halves), scored with the organisers' own evaluator exactly as the
leaderboard will score the full match: `spellF1` (mean possession-spell F1 over IoU 0.3–0.7) as
the ranking metric, `macroF1` over `{H, A, D}` as the first tie-break.

Final result (`trim_frames = 69`): **macroF1 = 0.7766**, **spellF1 = 0.4828**. The decoder found
12 possession spells against the ground truth's 17 — an under-segmentation that is a known
property of the persistence term (a sticky decoder merges some short, genuine hand-overs rather
than fragmenting long ones).

---

## Limitations

- The ball is untracked on 41% of graded frames; those frames are decided by the formation model
  alone, which is weaker evidence than a visible ball.
- Smart Tagging is itself automatically inferred, not ground truth, and any mistagged or
  mistimed set piece is inherited by the dead-ball mask.
- Only `trim_frames` was tuned against this match. The three possession-model constants were left
  at their externally calibrated values rather than re-optimised here, to avoid overfitting three
  parameters to a 17-possession excerpt.
- 17 possessions is a small sample: score differences below roughly 0.04 `spellF1` on the public
  ground truth are within the metric's own noise band, not a meaningful difference between
  settings.

---

## Repository layout

```
modules/deadball/     dead_ball.py            -- Part 1
modules/possession/   possession.py           -- Part 2 core (HMM/Viterbi decode)
                      players_model.py        -- formation-based evidence
                      atd_arrays.py           -- raw tracking -> numpy arrays, cached
                      metrica_sample.py       -- external calibration data (other matches)
                      calibrate_possession.py -- one-off calibration of Part 2's constants
starter/              load_smart_tagging.py, baseline.py, ...  (provided)
evaluation/           possession_metrics.py, run_evaluation.py, gt/  (provided)
```

## Reproducing the submission

1. Place the provided match data under `data/tactical/` and `data/broadcast/` as named in
   `DATA.md`.
2. Run `Final_running.py` — it sweeps `trim_frames`, prints the ranked results,
   and writes the final submission CSV for the best value found.
3. Score it with `python evaluation/run_evaluation.py` against the public ground truth to
   reproduce the numbers above.
