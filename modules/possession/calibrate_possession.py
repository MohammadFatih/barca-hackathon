"""Where CONTROL_SECONDS, KAPPA and KAPPA_GAP in possession.py come from.

    python calibrate_possession.py          # about a minute; downloads 127 MB on first run

The three constants decide what counts as a possession, and that cannot be learned from the
provided match: it has no labels we may tune on (RULES.md). They are calibrated on two other
matches instead, Metrica's open sample games, whose event data gives possession labels by an
algorithm (metrica_sample.py). Nothing here reads the ground truth of the provided match.

Steps
-----
1. Labels for the sample games, under two readings of "controlled possession" (every touch
   changes possession / a one-touch spell does not). The calibration averages the two.
2. The sample tracking is corrected tracking. It is degraded until it measures like the
   automatic tracking of the provided match, on two label-free statistics of that match: the
   ball's acceleration (the automatic ball is smoothed) and the number of players tracked per
   frame (some are missing).
3. Grid over control time and kappa. Criterion: mean spellF1 (the challenge's ranking metric)
   over 2 games x 2 readings on the degraded tracking, with dead ball taken from the labels.
   Among settings within 0.01 of the best, the smallest kappa, then the best score.
4. kappa_gap: stretches of live play are hidden, with the lengths of the provided match's own
   in-play ball gaps, and decoded without a ball. The smallest kappa_gap within 0.005 of the
   best is taken.

The script prints the chosen values and whether they equal the defaults in possession.py.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import atd_arrays as aa  # noqa: E402
import metrica_sample as ms  # noqa: E402
import players_model  # noqa: E402
import possession as ps  # noqa: E402
import possession_metrics as pm  # noqa: E402  (evaluation/, put on the path by metrica_sample)

FPS = ps.FPS
SEED = 0
GRID_CONTROL = (0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)      # seconds
GRID_KAPPA = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)
GRID_KAPPA_GAP = (0.01, 0.02, 0.05, 0.1, 0.2)
SMOOTH_CANDIDATES = (0, 3, 6, 9, 12, 15, 18)                  # half-widths in frames
READINGS = ("labels", "labels_controlled")
PLATEAU = 0.01
PLATEAU_GAP = 0.005
GAP_REPEATS, GAPS_PER_REPEAT = 3, 40


def alive_of(labels):
    return (labels == pm.H) | (labels == pm.A)


def ball_acceleration(ball, mask):
    """Magnitude of the ball's acceleration in m/s^2, by central differences, on mask frames."""
    velocity = (ball[2:] - ball[:-2]) * (FPS / 2.0)
    acceleration = np.hypot(*(velocity[2:] - velocity[:-2]).T) * (FPS / 2.0)
    return acceleration[mask[2:-2] & ~np.isnan(acceleration)]


def spell_f1(team, labels, blocks):
    """The challenge's spellF1 of a team call against labels, with the labels' own dead ball."""
    runs = [(a, b + 1) for a, b in blocks]
    codes = labels.copy()
    alive = alive_of(labels)
    codes[alive] = np.where(team[alive] == ps.HOME, pm.H, np.where(team[alive] == ps.AWAY, pm.A, pm.X))
    return pm.spell_scores(labels, pm.fill_abstentions(codes, runs), runs)["mean_f1"]


def degradation(sample_games, S_match, pos_match):
    """(ball smoothing half-width, player drop rate) that make the sample measure like the match."""
    live = S_match["graded"] & S_match["usable"]
    match_acc = np.percentile(ball_acceleration(S_match["ball"], live), [90, 99])
    match_tracked = (~np.isnan(pos_match[S_match["graded"]][:, :, 0])).sum(axis=1).mean()

    mismatch = []
    for half_width in SMOOTH_CANDIDATES:
        distance = []
        for G in sample_games:
            acc = np.percentile(ball_acceleration(ms.smooth_ball(G["ball"], half_width),
                                                  alive_of(G["labels"])), [90, 99])
            distance.append(np.abs(np.log(acc / match_acc)).sum())
        mismatch.append(np.mean(distance))
    ball_smooth = SMOOTH_CANDIDATES[int(np.argmin(mismatch))]

    sample_tracked = np.mean([(~np.isnan(G["pos"][alive_of(G["labels"])][:, :, 0])).sum(axis=1).mean()
                              for G in sample_games])
    return ball_smooth, float(1 - match_tracked / sample_tracked)


def prepare(G, ball_smooth, drop_rate):
    """Decoder inputs for a degraded sample game, with the players-only model trained on it
    exactly as on the provided match (pseudo-labels from its own ball, out-of-fold)."""
    D = ms.degrade(G, ball_half_width=ball_smooth, drop_rate=drop_rate, seed=SEED)
    S = ps.signals_from_arrays(D["pos"], D["team"], D["ball"], D["blocks"], D["pitch"])
    S["logit_players"], _ = players_model.players_logit(D["pos"], D["team"], S, D["pitch"][0])
    return S


def synthetic_gaps(S, dead, lengths, n_gaps, rng, margin=FPS):
    """Hide n_gaps stretches of live play, each with `margin` frames of real ball on both sides."""
    free = S["graded"] & ~dead & S["usable"]
    hide = np.zeros(S["n_frames"], dtype=bool)
    for length in rng.choice(lengths, size=n_gaps):
        runs = np.array(ps.half_runs(free, S["blocks"]))
        room = runs[:, 1] - runs[:, 0] - length - 2 * margin
        fits = room > 0
        if not fits.any():
            continue
        r = rng.choice(np.flatnonzero(fits), p=room[fits] / room[fits].sum())
        start = runs[r, 0] + margin + rng.integers(room[r])
        hide[start:start + length] = True
        free[start - margin:start + length + margin] = False
    return hide


def main():
    print("provided match: tracking statistics (no labels) ...")
    S_match = ps.load_signals()
    pos_match = aa.load(verbose=False)["pos"]
    sample_games = [ms.load(name) for name in ms.GAMES]
    for G in sample_games:
        print("  {}: {} possessions (every touch), {} (controlled)".format(
            G["name"], *(len(pm.possession_spells(G[key], [(a, b + 1) for a, b in G["blocks"]]))
                         for key in READINGS)))

    ball_smooth, drop_rate = degradation(sample_games, S_match, pos_match)
    print("degradation: ball smoothed +/- {} frames, {:.1f}% of player-frames dropped".format(
        ball_smooth, 100 * drop_rate))
    prepared = [prepare(G, ball_smooth, drop_rate) for G in sample_games]

    print("grid: control time x kappa ...")
    surface = np.zeros((len(GRID_CONTROL), len(GRID_KAPPA)))
    for i, control in enumerate(GRID_CONTROL):
        for j, kappa in enumerate(GRID_KAPPA):
            scores = []
            for G, S in zip(sample_games, prepared):
                team = ps.decode_hmm(S, ~alive_of(G["labels"]), control_seconds=control,
                                     kappa=kappa, kappa_gap=kappa)
                scores += [spell_f1(team, G[key], G["blocks"]) for key in READINGS]
            surface[i, j] = np.mean(scores)
    print("  mean spellF1    kappa: " + "  ".join("{:>5}".format(k) for k in GRID_KAPPA))
    for control, row in zip(GRID_CONTROL, surface):
        print("  control {:>4} s        ".format(control) + "  ".join("{:.3f}".format(v) for v in row))

    plateau = [(GRID_KAPPA[j], -surface[i, j], GRID_CONTROL[i])
               for i in range(len(GRID_CONTROL)) for j in range(len(GRID_KAPPA))
               if surface[i, j] >= surface.max() - PLATEAU]
    kappa, negative_score, control = min(plateau)       # smallest kappa, then the best score
    print("best {:.3f}; {} settings within {} of it; chosen control {} s, kappa {} ({:.3f})".format(
        surface.max(), len(plateau), PLATEAU, control, kappa, -negative_score))

    print("frames without a ball: kappa_gap ...")
    starts, stops = aa.true_runs(S_match["graded"] & (S_match["gap_kind"] == 1))
    lengths = stops - starts
    rng = np.random.default_rng(SEED)
    per_reading = {key: {k: [] for k in GRID_KAPPA_GAP} for key in READINGS}
    for G, S in zip(sample_games, prepared):
        dead = ~alive_of(G["labels"])
        for _ in range(GAP_REPEATS):
            hide = synthetic_gaps(S, dead, lengths, GAPS_PER_REPEAT, rng)
            for kappa_gap in GRID_KAPPA_GAP:
                team = ps.decode_hmm(S, dead, control_seconds=control, kappa=kappa,
                                     kappa_gap=kappa_gap, hide=hide)
                for key in READINGS:
                    per_reading[key][kappa_gap].append(spell_f1(team, G[key], G["blocks"]))
    gap_scores = [np.mean([np.mean(per_reading[key][k]) for key in READINGS]) for k in GRID_KAPPA_GAP]
    for k, value in zip(GRID_KAPPA_GAP, gap_scores):
        print("  kappa_gap {:>4}: spellF1 {:.3f}".format(k, value))
    kappa_gap = next(k for k, value in zip(GRID_KAPPA_GAP, gap_scores)
                     if value >= max(gap_scores) - PLATEAU_GAP)

    print("\ncalibrated: CONTROL_SECONDS = {}, KAPPA = {}, KAPPA_GAP = {}".format(control, kappa, kappa_gap))
    same = (control, kappa, kappa_gap) == (ps.CONTROL_SECONDS, ps.KAPPA, ps.KAPPA_GAP)
    print("possession.py defaults: {}, {}, {}  ->  {}".format(
        ps.CONTROL_SECONDS, ps.KAPPA, ps.KAPPA_GAP, "same" if same else "DIFFERENT"))
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
