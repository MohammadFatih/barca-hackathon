"""Metrica's open sample games as calibration data: tracking, and possession labels from events.

    import metrica_sample as ms
    game = ms.load("Sample_Game_1")        # downloads on first use, then reads data/external/
    game["pos"], game["ball"]              # [T, 28, 2], [T, 2] in metres, NaN where absent
    game["labels"]                         # [T] int8, possession_metrics codes H / A / D

Source: https://github.com/metrica-sports/sample-data (tracking and event data of other matches,
published by Metrica for research). RULES.md allows data about other games, hand-labelled or
not. Nothing here touches the provided match.

How the labels are derived from the event file, so that they mean what the challenge means:

  team   the team of the latest on-ball event (pass, shot, recovery, set piece, ball lost, ball
         out). A lost ball stays with the team that lost it until the other team's RECOVERY,
         which is CHALLENGE.md's rule: a loose ball stays with the last team, and possession
         changes when the other team gains control. Duels (CHALLENGE) change nothing by themselves.
  dead   from the end of the last event before a restart (set piece, goal kick) to the restart;
         from an event that stops play (ball out, shot out or goal, foul, offside) to the next
         on-ball event; and wherever the corrected tracking has no ball for more than 2 s.

The files are data, read with pandas and never executed. They live under data/, gitignored.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from atd_arrays import ROOT  # noqa: E402  (also puts the repository root on the path)

if os.path.join(ROOT, "evaluation") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import games  # noqa: E402
import possession_metrics as pm  # noqa: E402

URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/{game}/{name}"
GAMES = ("Sample_Game_1", "Sample_Game_2")
FILES = ("RawEventsData", "RawTrackingData_Home_Team", "RawTrackingData_Away_Team")
PITCH = (105.0, 68.0)          # metres, as documented with the sample data
FPS = 25
HOME, AWAY = 0, 1

ON_BALL = ("PASS", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST", "BALL OUT")
NO_TIMING = ("CARD",)          # logged during the stoppage, not at the moment play stopped
MIN_DEAD_GAP = 50              # frames (2 s). A longer stretch without a ball is a stoppage.


def data_dir(game):
    return os.path.join(games.DATA_DIR, "external", "metrica_sample", game)


def file_path(game, name):
    return os.path.join(data_dir(game), "{}_{}.csv".format(game, name))


def ensure(game, verbose=True):
    """Download the three CSVs of one game if they are not there yet."""
    if not os.path.isdir(data_dir(game)):
        os.makedirs(data_dir(game))
    for name in FILES:
        path = file_path(game, name)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            continue
        url = URL.format(game=game, name=os.path.basename(path))
        if verbose:
            print("downloading {}".format(url))
        from urllib.request import urlretrieve
        urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)


def read_tracking(game):
    """pos [T, S, 2] and ball [T, 2] in metres, period [T], team [S]. Row t is Frame t + 1."""
    tables = [pd.read_csv(file_path(game, "RawTrackingData_{}_Team".format(side)),
                          skiprows=3, header=None).to_numpy(dtype=np.float64)
              for side in ("Home", "Away")]
    home, away = tables
    if len(home) != len(away):
        raise ValueError("{}: home and away tracking differ in length".format(game))
    period = home[:, 0].astype(np.int8)
    length, width = PITCH

    def players(table):                   # columns: period, frame, time, x1, y1, ..., ball x, y
        xy = table[:, 3:-2]
        return np.stack([xy[:, 0::2] * length, xy[:, 1::2] * width], axis=2)

    pos_home, pos_away = players(home), players(away)
    pos = np.concatenate([pos_home, pos_away], axis=1)
    team = np.array([HOME] * pos_home.shape[1] + [AWAY] * pos_away.shape[1], dtype=np.int8)
    ball = np.stack([home[:, -2] * length, home[:, -1] * width], axis=1)
    return pos, ball, period, team


def read_events(game):
    events = pd.read_csv(file_path(game, "RawEventsData"))
    events["Subtype"] = events["Subtype"].fillna("")
    return events


def stops_play(row):
    """True for an event after which the ball is out of play until the next on-ball event."""
    kind, subtype = row["Type"], row["Subtype"]
    return (kind == "BALL OUT" or kind == "FAULT RECEIVED"
            or (kind == "SHOT" and ("OUT" in subtype or "GOAL" in subtype))
            or (kind == "BALL LOST" and ("OFFSIDE" in subtype or "END HALF" in subtype)))


def one_touch_events(rows):
    """Indices of on-ball events that belong to a one-touch spell.

    A spell is a run of on-ball events by one team with nothing by the other team in between. It
    is one-touch when all its events start on the same frame: the team met the ball once and lost
    it again (a blocked pass, a deflection, a header straight back). CHALLENGE.md says that is
    not a change of possession. Restarts are never one-touch: a set piece has a clear possessor.
    """
    on_ball = [i for i, row in enumerate(rows) if row["Type"] in ON_BALL]
    out, spell = set(), []

    def close(spell):
        if not spell:
            return
        frames = {rows[i]["Start Frame"] for i in spell}
        restart = any(rows[i]["Type"] == "SET PIECE" or "GOAL KICK" in rows[i]["Subtype"]
                      for i in spell)
        if len(frames) == 1 and not restart:
            out.update(spell)

    for i in on_ball:
        same = spell and rows[spell[-1]]["Team"] == rows[i]["Team"] \
            and rows[spell[-1]]["Period"] == rows[i]["Period"]
        if not same:
            close(spell)
            spell = []
        spell.append(i)
    close(spell)
    return out


def possession_labels(events, period, ball=None, controlled=False):
    """[T] int8 of pm.H / pm.A / pm.D from the event file. See the module docstring.

    controlled=False: every on-ball event gives its team the ball ("every touch").
    controlled=True: one-touch spells are ignored, so the previous team keeps the ball.

    With `ball`, a stretch of more than MIN_DEAD_GAP frames without a ball is dead as well. The
    sample tracking is corrected tracking, and it carries no ball while play is stopped: its
    ball gaps begin and end on the frame of the event that stops and restarts play. That is the
    challenge's own definition of D ("where the corrected tracking marks the ball dead"), and it
    catches the restarts the event file does not log as a set piece.
    """
    n_frames = len(period)
    owner = np.full(n_frames, -1, dtype=np.int8)
    dead = np.zeros(n_frames, dtype=bool)

    rows = events.to_dict("records")
    ignored = one_touch_events(rows) if controlled else set()
    previous = None                        # the last event with a usable time
    stopped_at = None                      # frame after a play-stopping event, until play resumes
    for index, row in enumerate(rows):
        start = int(row["Start Frame"]) - 1
        kind, subtype = row["Type"], row["Subtype"]
        on_ball = kind in ON_BALL and 0 <= start < n_frames
        restart = kind == "SET PIECE" or "GOAL KICK" in subtype
        if restart and previous is not None and previous["Period"] == row["Period"]:
            stopped = int(max(previous["Start Frame"], previous["End Frame"])) - 1
            dead[max(stopped, 0):start] = True
        if on_ball and stopped_at is not None and stopped_at[1] == row["Period"]:
            dead[stopped_at[0]:start] = True
        if on_ball:
            if index not in ignored:
                owner[start] = HOME if row["Team"] == "Home" else AWAY
            stopped_at = None
        if stops_play(row):
            stopped_at = (int(max(row["Start Frame"], row["End Frame"])), row["Period"])
        if kind not in NO_TIMING:
            previous = row

    if ball is not None:
        missing = np.isnan(ball[:, 0])
        edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if stop - start > MIN_DEAD_GAP:
                dead[start:stop] = True

    labels = np.full(n_frames, pm.X, dtype=np.int8)
    for number in np.unique(period):
        frames = np.flatnonzero(period == number)
        marks = owner[frames]
        known = np.flatnonzero(marks >= 0)
        if not len(known):
            continue
        carry = np.maximum.accumulate(np.where(marks >= 0, np.arange(len(frames)), -1))
        carry[carry < 0] = known[0]
        # After the last event of a half nothing is known: leave those frames ungraded.
        last_event = int(events.loc[events["Period"] == number, "End Frame"].max()) - 1
        graded = frames <= max(last_event, frames[known[-1]])
        labels[frames[graded]] = np.where(marks[carry][graded] == HOME, pm.H, pm.A)
    labels[dead & (labels != pm.X)] = pm.D
    return labels


def load(game, verbose=True):
    """One sample game: tracking in metres, halves, and event-derived possession labels."""
    ensure(game, verbose)
    pos, ball, period, team = read_tracking(game)
    events = read_events(game)
    labels = possession_labels(events, period, ball)
    blocks = []
    for number in np.unique(period):
        frames = np.flatnonzero((period == number) & (labels != pm.X))
        blocks.append((int(frames[0]), int(frames[-1])))
    return {"name": game, "pos": pos, "ball": ball, "team": team, "period": period,
            "labels": labels,                       # every touch changes possession
            "labels_controlled": possession_labels(events, period, ball, controlled=True),
            "blocks": blocks, "pitch": PITCH}


# -- making clean tracking look like the automatic tracking of the provided match --------------

def smooth_ball(ball, half_width):
    """Centred moving average of the ball over +/- half_width frames, within each visible run."""
    if half_width <= 0:
        return ball.copy()
    out = np.full_like(ball, np.nan)
    seen = ~np.isnan(ball[:, 0])
    edges = np.diff(np.concatenate(([0], seen.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        segment = ball[start:stop]
        total = np.concatenate((np.zeros((1, 2)), np.cumsum(segment, axis=0)))
        i = np.arange(len(segment))
        lo, hi = np.maximum(0, i - half_width), np.minimum(len(segment), i + half_width + 1)
        out[start:stop] = (total[hi] - total[lo]) / (hi - lo)[:, None]
    return out


def drop_players(pos, rate, mean_seconds, rng):
    """Copy of pos with players missing in runs: a fraction `rate` of player-frames, in
    stretches of exponential length with mean `mean_seconds`, as a tracker loses a track."""
    out = pos.copy()
    n_frames, n_slots = pos.shape[:2]
    if rate <= 0:
        return out
    mean_frames = mean_seconds * FPS
    for slot in range(n_slots):
        t = int(rng.exponential(mean_frames * (1 - rate) / rate))
        while t < n_frames:
            lost = max(1, int(rng.exponential(mean_frames)))
            out[t:t + lost, slot] = np.nan
            t += lost + max(1, int(rng.exponential(mean_frames * (1 - rate) / rate)))
    return out


def degrade(game, ball_half_width, drop_rate, drop_seconds=5.0, seed=0):
    """A copy of a game whose tracking has the two defects of the automatic tracking measured in
    the provided match: a smoothed ball, and players missing from the tracking in runs."""
    rng = np.random.default_rng(seed)
    out = dict(game)
    out["ball"] = smooth_ball(game["ball"], ball_half_width)
    out["pos"] = drop_players(game["pos"], drop_rate, drop_seconds, rng)
    return out
