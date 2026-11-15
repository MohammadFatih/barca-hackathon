"""Part 2, stage 0: the raw ATD as numpy arrays in metres, cached under data/cache/.

    import atd_arrays as aa
    A = aa.load()            # first call parses the raw file (~1 min), later calls read the cache
    A["pos"]                 # [T, S, 2] player positions in metres, NaN where not tracked
    A["spd"]                 # [T, S]    player speeds in m/s, as the ATD reports them
    A["ball"]                # [T, 2]    ball position in metres, NaN where not tracked
    A["team"]                # [S]       0 = home, 1 = away, for each track slot

T is the number of tactical video frames (199,319) and row t is video frame t, so the arrays line
up with a submission row for row. S is the number of track slots (46): tracklets, not people.

The cache lives under data/, which is gitignored. It is frame-level data derived from the match,
so it must never be committed -- see RULES.md, "Do not publish derived data".

numpy only, plus the repository's own games.py and starter/baseline.py, so it runs in the same
environment as the scorer.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def repo_root():
    """The repository root: the nearest folder above this file that holds games.py.

    Found by walking up, so this module works from whatever folder of the repository it is
    placed in.
    """
    folder = HERE
    while not os.path.isfile(os.path.join(folder, "games.py")):
        parent = os.path.dirname(folder)
        if parent == folder:
            raise ImportError("games.py not found above {}: is this inside the repository?"
                              .format(HERE))
        folder = parent
    return folder


ROOT = repo_root()
for _path in (ROOT, os.path.join(ROOT, "starter")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import games  # noqa: E402
from baseline import team_of_each_track, team_names  # noqa: E402

# Bump when the array layout changes, so an old cache is never read as a new one.
CACHE_VERSION = 1

# Raw ATD line: "<frame_id>:<x,y,speed>;<x,y,speed>;...:<ball x,y,...>". See starter/baseline.py.
FIELD_SEP = ":"
GROUP_SEP = ";"
VALUE_SEP = ","


def cache_dir():
    """data/cache/, or $MTAH_DATA_DIR/cache/. Inside the gitignored data folder on purpose."""
    return os.path.join(games.DATA_DIR, "cache")


def _parse_group(group):
    """(x, y, speed) from one 'x,y,speed' group, tolerating a bare 'NaN'."""
    values = group.split(VALUE_SEP)
    out = [np.nan, np.nan, np.nan]
    for index, value in enumerate(values[:3]):
        out[index] = float(value)
    return out


def parse_raw(raw_path, n_frames, n_slots, pitch):
    """Stream the raw ATD into arrays. Positions are scaled from 0-1 to metres.

    The fast path converts a whole line of player groups in one numpy call. A line whose groups
    are not all 'x,y,speed' (for example a bare 'NaN') falls back to parsing group by group, so an
    unexpected encoding of a missing player costs speed, not correctness.
    """
    length, width = pitch
    pos = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    spd = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    has_record = np.zeros(n_frames, dtype=bool)

    with open(raw_path) as handle:
        for line_no, line in enumerate(handle, 1):
            parts = line.rstrip("\r\n").split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # ATD frame_id counts from 1, video frames from 0
            if not 0 <= frame < n_frames:
                continue

            groups = parts[1].split(GROUP_SEP)
            if len(groups) != n_slots:
                raise ValueError("{} line {}: {} player groups, metadata has {} slots".format(
                    raw_path, line_no, len(groups), n_slots))
            values = np.array(parts[1].replace(GROUP_SEP, VALUE_SEP).split(VALUE_SEP),
                              dtype=np.float64)
            if values.size == n_slots * 3:
                values = values.reshape(n_slots, 3)
            else:
                values = np.array([_parse_group(group) for group in groups], dtype=np.float64)

            pos[frame, :, 0] = values[:, 0] * length
            pos[frame, :, 1] = values[:, 1] * width
            spd[frame] = values[:, 2]

            ball_values = parts[2].strip().split(VALUE_SEP)
            if len(ball_values) >= 2:
                ball[frame] = (float(ball_values[0]) * length, float(ball_values[1]) * width)
            has_record[frame] = True

    return pos, spd, ball, has_record


def short_path(path):
    """Path relative to the repository for printing; absolute when it is on another drive."""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:  # Windows: MTAH_DATA_DIR on a different drive from the repository
        return path


def cache_path(game=games.GAME):
    return os.path.join(cache_dir(), "{}_tactical_arrays_v{}.npz".format(game, CACHE_VERSION))


def load(game=games.GAME, refresh=False, verbose=True):
    """The tactical ATD as a dict of arrays. Parses once, then reads the cache.

    The cache is rebuilt when it is older than the raw file, or when refresh=True.
    """
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game)
    path = cache_path(game)

    if (not refresh and os.path.isfile(path)
            and os.path.getmtime(path) >= os.path.getmtime(raw_path)):
        with np.load(path) as stored:
            arrays = {key: stored[key] for key in stored.files}
        if verbose:
            print("read cache {}".format(short_path(path)))
        return arrays

    team = team_of_each_track(meta_path)
    n_frames = games.source_frames("tactical", game)
    pitch = games.GAMES[game]["pitch"]
    if verbose:
        home, away = team_names(meta_path)
        print("teams: H = {}, A = {}; {} track slots".format(home, away, len(team)))
        print("parsing {} ...".format(os.path.basename(raw_path)))

    started = time.time()
    pos, spd, ball, has_record = parse_raw(raw_path, n_frames, len(team), pitch)
    arrays = {"pos": pos, "spd": spd, "ball": ball, "has_record": has_record,
              "team": team, "pitch": np.array(pitch, dtype=np.float64)}

    if not os.path.isdir(cache_dir()):
        os.makedirs(cache_dir())
    np.savez_compressed(path, **arrays)
    if verbose:
        print("  {:.0f} s; cached to {}".format(time.time() - started, short_path(path)))
    return arrays


def true_runs(mask):
    """(starts, stops) of every maximal run of True, half-open: mask[start:stop] is all True.

    Run-length encoding -- the tool for turning a per-frame boolean (ball missing, ball alive,
    label == H) into intervals.
    """
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    steps = np.diff(padded.astype(np.int8))
    return np.flatnonzero(steps == 1), np.flatnonzero(steps == -1)


def graded_half(n_frames, game=games.GAME):
    """int8 array over video frames: 1 or 2 inside a graded half, 0 elsewhere."""
    half = np.zeros(n_frames, dtype=np.int8)
    for number, first, last in games.half_video_blocks(game):
        half[first:last + 1] = number
    return half


if __name__ == "__main__":
    arrays = load(refresh="--refresh" in sys.argv)
    for key, value in arrays.items():
        print("  {:<11} {} {}".format(key, value.dtype, value.shape))
