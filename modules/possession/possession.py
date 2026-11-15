"""Part 2: which team has the ball on every frame, including frames where the ball is not seen.

    import possession as ps
    S = ps.load_signals()                          # everything derived from the tracking
    dead = part1_dead | ps.stoppage_mask(S)        # bool [T]: True where the ball is not in play
    team = ps.decode_hmm(S, dead)                  # [T] 0 = home, 1 = away, -1 = no call
    labels = ps.to_labels(team, dead, S)           # [T] 'H' / 'A' / 'D' / 'X', ready to write

or in one call, `labels = ps.possession_labels(part1_dead)`.

What part 2 answers is "if the ball is alive, which team has it". Which frames are dead is part
1's answer, passed in as `dead`. The decoder restarts at every dead-ball stretch, because the
possession before a stoppage says nothing about who restarts, so `decode_hmm` must be run again
whenever the dead mask changes.

The model
---------
Three sources of evidence, all as log-odds for the home team (positive = home):

    ball      (dA - dH) / TAU, clipped at CLIP, faded when the ball is in nobody's reach. dH and
              dA are the distances from the ball to the nearest tracked home and away player.
    players   a logistic regression on the shape of the two teams (players_model.py), trained on
              pseudo-labels from this same match and predicted out-of-fold. It never needs the
              ball, so it is the evidence on frames where the ball is not seen.
    time      possession persists: a change of team costs SWITCH_COST.

Per frame, e(t) = w_ball * ball(t) + kappa * players(t). On each stretch of live play the
decoder finds the home/away sequence that maximises  sum_t +-e(t)/2  -  SWITCH_COST * (number of
changes), exactly, with the Viterbi algorithm. A frame without a ball is not a special case: it
has no ball term, so its two ends, the players and the cost of a change decide it.

Where the constants come from
-----------------------------
None was chosen on the public ground truth of the provided match.

    TAU, CLIP, CONTROL_R, FLIGHT_SCALE   physical scales: a metre, a player's reach
    Q_SWITCH                             hand-overs per frame of visible ball, measured on the
                                         provided match without labels
    CONTROL_SECONDS, KAPPA, KAPPA_GAP    calibrated on two other matches (Metrica's open sample
                                         games, possession derived from their event data). The
                                         script is calibrate_possession.py; it prints these values.

numpy only. Needs the match data under data/ (see DATA.md) and nothing else.
"""

import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import atd_arrays as aa  # noqa: E402  (puts the repository root on the path)
import games  # noqa: E402
import players_model  # noqa: E402

FPS = 25
HOME, AWAY, NOBODY = 0, 1, -1

CONTROL_R = 1.5      # m. A ball within this of a player is at that player's feet.
TAU = 1.0            # m. A margin of TAU between the two nearest opponents is one unit of log-odds.
CLIP = 3.0           # cap on the per-frame ball log-odds, so one frame is never certain.
FLIGHT_SCALE = 1.5   # m. Ball evidence fades by 1/e for each FLIGHT_SCALE beyond CONTROL_R.
SHORT_GAP = 25       # frames. Ball gaps of up to 1 s are interpolated; the tracking has none
                     # between 1 s and 5 s, so a longer gap is a different event.
OUT_MARGIN = 0.5     # m beyond a pitch line before a seen ball counts as out of the pitch.

Q_SWITCH = 0.0018    # per-frame probability of a hand-over.
SWITCH_COST = float(np.log((1 - Q_SWITCH) / Q_SWITCH))      # 6.32 nats

# Calibrated on other games, see calibrate_possession.py.
CONTROL_SECONDS = 0.75   # clear control for this long pays for one change of possession
KAPPA = 0.05             # weight of one frame of players evidence, on frames with a ball
KAPPA_GAP = 0.02         # the same on frames without a ball


def ball_weight(control_seconds=CONTROL_SECONDS):
    """Weight of one frame of ball evidence.

    Set so that `control_seconds` of a ball clearly at one team's feet (log-odds CLIP on every
    frame) add up to SWITCH_COST, the price of one change of possession. So a lasting hand-over
    needs control_seconds of clear control, and a possession that is given back needs twice
    that, because it pays for two changes.
    """
    return SWITCH_COST / (CLIP * control_seconds * FPS)


# -- signals -----------------------------------------------------------------------------------

def match_blocks(game=games.GAME):
    """[(first, last)] frame range of each graded half of the provided match, both inclusive."""
    return [(first, last) for _, first, last in games.half_video_blocks(game)]


def half_runs(mask, blocks=None):
    """[(start, stop)] maximal runs of True, half-open, never spanning the half-time break.

    blocks is [(first, last)] per half. Default: the two graded halves of the provided match.
    """
    out = []
    for first, last in (match_blocks() if blocks is None else blocks):
        starts, stops = aa.true_runs(mask[first:last + 1])
        out += [(int(a) + first, int(b) + first) for a, b in zip(starts, stops)]
    return out


def interpolate_short_gaps(ball, max_gap=SHORT_GAP, blocks=None):
    """Ball with gaps of up to max_gap frames filled linearly. Long gaps stay NaN."""
    out = ball.copy()
    for start, stop in half_runs(np.isnan(ball[:, 0]), blocks):
        if stop - start > max_gap or start == 0 or stop >= len(ball):
            continue
        left, right = ball[start - 1], ball[stop]
        if np.isnan(left[0]) or np.isnan(right[0]):
            continue                                   # the gap touches the edge of a half
        weight = (np.arange(start, stop) - (start - 1)) / float(stop - (start - 1))
        out[start:stop] = (1 - weight)[:, None] * left + weight[:, None] * right
    return out


def ball_evidence(d_home, d_away, flight_scale=FLIGHT_SCALE):
    """Per-frame log-odds(home) from the ball. 0 where there is no ball.

    The sign comes from which team's nearest player is closer, the size from how much closer.
    A ball in nobody's reach says little about possession (CHALLENGE.md: a loose ball stays with
    the last team), so the evidence fades with the distance to the nearest player.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d_min = np.fmin(d_home, d_away)
        margin = np.clip((d_away - d_home) / TAU, -CLIP, CLIP)
        fade = np.exp(-np.maximum(0.0, d_min - CONTROL_R) / flight_scale)
    return np.nan_to_num(margin * fade, nan=0.0)


def signals_from_arrays(pos, team, ball, blocks, pitch, logit_players=None):
    """The decoder's inputs from raw arrays, for any match.

    pos [T, S, 2] and ball [T, 2] in metres with NaN where not tracked, team [S] of 0 / 1,
    blocks [(first, last)] per half. logit_players is the players-only evidence if there is one.
    """
    length, width = pitch
    n_frames = len(pos)
    graded = np.zeros(n_frames, dtype=bool)
    for first, last in blocks:
        graded[first:last + 1] = True

    ball = interpolate_short_gaps(ball, blocks=blocks)
    seen = ~np.isnan(ball[:, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                # frames with no ball, or no player of a team
        distance = np.hypot(pos[:, :, 0] - ball[:, None, 0], pos[:, :, 1] - ball[:, None, 1])
        d_home = np.nanmin(np.where(team == HOME, distance, np.nan), axis=1)
        d_away = np.nanmin(np.where(team == AWAY, distance, np.nan), axis=1)
    usable = seen & ~np.isnan(d_home) & ~np.isnan(d_away)
    nearest = np.where(usable, np.where(d_home <= d_away, HOME, AWAY), NOBODY).astype(np.int8)
    out_of_pitch = seen & ((ball[:, 0] < -OUT_MARGIN) | (ball[:, 0] > length + OUT_MARGIN)
                           | (ball[:, 1] < -OUT_MARGIN) | (ball[:, 1] > width + OUT_MARGIN))
    if logit_players is None:
        logit_players = np.zeros(n_frames)
    return {
        "n_frames": n_frames, "blocks": list(blocks), "graded": graded,
        "ball": ball, "seen": seen, "usable": usable, "d_home": d_home, "d_away": d_away,
        "d_min": np.fmin(d_home, d_away), "nearest": nearest,
        "e_ball": ball_evidence(d_home, d_away),
        "logit_players": np.nan_to_num(np.asarray(logit_players, dtype=np.float64)),
        "out_of_pitch": out_of_pitch,
    }


def otsu(values):
    """The cut that splits values into two groups with the smallest within-group variance."""
    v = np.sort(np.asarray(values, dtype=float))
    best, cut = np.inf, np.nan
    for i in range(2, len(v) - 1):
        within = i * v[:i].var() + (len(v) - i) * v[i:].var()
        if within < best:
            best, cut = within, (v[i - 1] + v[i]) / 2
    return cut


def gap_kinds(seen, player_speed, blocks):
    """int8 [T]: 0 ball available, 1 in-play gap, 2 stoppage-like gap. Also returns the cut.

    A gap is a run of more than SHORT_GAP frames without a ball. The tracker bridges a lost ball
    by itself for about 5 s, so a gap means it could not find the ball for longer than that.
    Most gaps are stoppages: the ball left the pitch or play was stopped, and the players slow
    to a walk. The two kinds are split on the mean player speed inside the gap by Otsu's
    threshold, which takes the cut from the shape of the data alone. No label is involved.
    """
    kind = np.zeros(len(seen), dtype=np.int8)
    gaps = [(a, b) for a, b in half_runs(~seen, blocks) if b - a > SHORT_GAP]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        speed = np.array([np.nanmean(player_speed[a:b]) for a, b in gaps])
    cut = otsu(speed)
    for (a, b), value in zip(gaps, speed):
        kind[a:b] = 2 if value < cut else 1
    return kind, cut


def load_signals(game=games.GAME):
    """Everything the decoder needs for the provided match, as [T] arrays indexed by video frame.

    Computed from the raw tracking each time (about half a minute). The only cache is the parsed
    tracking itself, under data/cache/, which atd_arrays writes.
    """
    arrays = aa.load(game, verbose=False)
    pos, team = arrays["pos"].astype(np.float64), arrays["team"]
    S = signals_from_arrays(pos, team, arrays["ball"].astype(np.float64), match_blocks(game),
                            arrays["pitch"])
    logit, info = players_model.players_logit(pos, team, S, float(arrays["pitch"][0]))
    S["logit_players"] = np.nan_to_num(logit)
    S["players_model"] = info
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                # frames with no tracked player at all
        S["player_speed"] = np.nanmean(arrays["spd"], axis=1)
    S["gap_kind"], S["gap_speed_cut"] = gap_kinds(S["seen"], S["player_speed"], S["blocks"])
    return S


# -- dead ball that the tracking itself shows --------------------------------------------------

def stoppage_mask(S):
    """bool [T]: stoppages that can be read off the tracking, to be OR-ed with part 1's mask.

    True inside a stoppage-like ball gap (the tracker lost the ball for more than 5 s and the
    players are walking), and where the ball is seen outside the pitch. This is 40% of the graded
    frames of the provided match. It cannot see a stoppage during which the ball stays visible
    inside the pitch (a free kick being set up, a keeper holding the ball): those are part 1's.
    """
    return S["graded"] & ((S["gap_kind"] == 2) | S["out_of_pitch"])


# -- the decoder -------------------------------------------------------------------------------

def viterbi_two_state(evidence, switch_cost):
    """Most likely home/away path for per-frame log-odds(home), with a cost for each change.

    Offline, so a change is placed where the evidence turns, not some frames later as an online
    rule would. Returns int8, HOME or AWAY, for every frame.
    """
    n = len(evidence)
    if n == 0:
        return np.zeros(0, np.int8)
    half_e = (0.5 * np.asarray(evidence, dtype=np.float64)).tolist()
    from_other_home = np.zeros(n, dtype=bool)          # best path into HOME at t came from AWAY
    from_other_away = np.zeros(n, dtype=bool)
    score_home, score_away = half_e[0], -half_e[0]
    for t in range(1, n):
        switch_to_home = score_away - switch_cost > score_home
        switch_to_away = score_home - switch_cost > score_away
        new_home = (score_away - switch_cost if switch_to_home else score_home) + half_e[t]
        new_away = (score_home - switch_cost if switch_to_away else score_away) - half_e[t]
        from_other_home[t], from_other_away[t] = switch_to_home, switch_to_away
        score_home, score_away = new_home, new_away

    path = np.empty(n, dtype=np.int8)
    state = HOME if score_home >= score_away else AWAY
    for t in range(n - 1, -1, -1):
        path[t] = state
        if (from_other_home[t] if state == HOME else from_other_away[t]):
            state = AWAY if state == HOME else HOME
    return path


def hmm_evidence(S, control_seconds=CONTROL_SECONDS, kappa=KAPPA, kappa_gap=KAPPA_GAP, hide=None):
    """Per-frame log-odds(home): weighted ball evidence plus weighted players evidence.

    kappa weights the players on frames with a ball, kappa_gap on frames without. `hide` is a
    boolean mask of frames whose ball is to be ignored; calibrate_possession.py uses it to test
    the handling of frames without a ball.
    """
    has_ball = S["usable"] if hide is None else S["usable"] & ~hide
    return (np.where(has_ball, ball_weight(control_seconds) * S["e_ball"], 0.0)
            + np.where(has_ball, kappa, kappa_gap) * S["logit_players"])


def decode_hmm(S, dead, control_seconds=CONTROL_SECONDS, kappa=KAPPA, kappa_gap=KAPPA_GAP,
               switch_cost=SWITCH_COST, hide=None):
    """[T] int8: HOME / AWAY on live graded frames, NOBODY where dead or ungraded.

    Ball + players + persistence, decoded separately on each stretch of live play, so that no
    possession runs through a stoppage.
    """
    dead = np.asarray(dead, dtype=bool)
    if len(dead) != S["n_frames"]:
        raise ValueError("dead mask has {} frames, the match has {}".format(len(dead), S["n_frames"]))
    evidence = hmm_evidence(S, control_seconds, kappa, kappa_gap, hide)
    team = np.full(S["n_frames"], NOBODY, dtype=np.int8)
    for start, stop in half_runs(S["graded"] & ~dead, S["blocks"]):
        team[start:stop] = viterbi_two_state(evidence[start:stop], switch_cost)
    return team


# -- labels ------------------------------------------------------------------------------------

def to_labels(team, dead, S):
    """[T] array of 'H' / 'A' / 'D' / 'X'. X outside the graded halves and where there is no call."""
    labels = np.full(S["n_frames"], "X", dtype="<U1")
    labels[team == HOME] = "H"
    labels[team == AWAY] = "A"
    labels[S["graded"] & np.asarray(dead, dtype=bool)] = "D"
    return labels


def possession_labels(part1_dead=None, S=None):
    """The whole of part 2 in one call: [T] labels, the dead mask used, and the signals.

    part1_dead is part 1's boolean mask over video frames, or None to use only the stoppages the
    tracking shows. Either way the mask the decoder gets is part1_dead OR stoppage_mask(S).
    """
    S = load_signals() if S is None else S
    dead = stoppage_mask(S)
    if part1_dead is not None:
        part1_dead = np.asarray(part1_dead, dtype=bool)
        if len(part1_dead) != S["n_frames"]:
            raise ValueError("part 1 mask has {} frames, the match has {}".format(
                len(part1_dead), S["n_frames"]))
        dead = (dead | part1_dead) & S["graded"]
    return to_labels(decode_hmm(S, dead), dead, S), dead, S
