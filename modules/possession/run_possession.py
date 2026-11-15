"""Write a possession submission from the tracking, with or without part 1's dead-ball mask.

    python run_possession.py --out evaluation/results/part2.csv
    python run_possession.py --dead-mask shared_code/dead_ball_mask.csv --out evaluation/results/part2.csv

--dead-mask is a CSV with a header and two columns, frame and is_dead (True / False), one row per
video frame, as part 1 writes it. Without it, only the stoppages the tracking itself shows are
dead (possession.stoppage_mask), which under-calls D.

Paths are relative to the repository root. The file is a complete submission: 199,319 rows of
H / A / D / X. Check it with evaluation/validate_submission.py and score it with
evaluation/run_evaluation.py.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import possession as ps  # noqa: E402
from atd_arrays import ROOT  # noqa: E402
from baseline import write_submission  # noqa: E402  (starter/baseline.py)


def read_dead_mask(path, n_frames):
    """bool [n_frames] from a frame,is_dead CSV. Rows must be the video frames in order."""
    table = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=str)
    if table.shape != (n_frames, 2):
        raise ValueError("{}: expected {} rows of frame,is_dead, found shape {}".format(
            path, n_frames, table.shape))
    if not (table[:, 0].astype(int) == np.arange(n_frames)).all():
        raise ValueError("{}: the frame column is not 0 .. {} in order".format(path, n_frames - 1))
    values = np.char.lower(np.char.strip(table[:, 1]))
    unknown = set(np.unique(values)) - {"true", "false", "1", "0"}
    if unknown:
        raise ValueError("{}: is_dead has values other than True / False: {}".format(
            path, sorted(unknown)))
    return (values == "true") | (values == "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dead-mask", help="part 1's frame,is_dead CSV")
    parser.add_argument("--out", required=True, help="where to write the submission CSV")
    args = parser.parse_args()

    S = ps.load_signals()
    part1 = None
    if args.dead_mask:
        part1 = read_dead_mask(os.path.join(ROOT, args.dead_mask), S["n_frames"])
    labels, dead, _ = ps.possession_labels(part1, S)

    graded = S["graded"]
    own = ps.stoppage_mask(S)
    print("players-only model: {:.1f}% out-of-fold accuracy on {} pseudo-labels".format(
        100 * S["players_model"]["accuracy"], S["players_model"]["pseudo_labels"]))
    print("dead frames: {} from the tracking{}, {} in total ({:.1f}% of graded frames)".format(
        int(own.sum()),
        "" if part1 is None else ", {} more from part 1".format(int((dead & ~own).sum())),
        int(dead.sum()), 100.0 * dead.sum() / graded.sum()))
    print("labels: " + "  ".join("{} {}".format(k, int((labels == k).sum())) for k in "HADX"))
    path = write_submission(os.path.join(ROOT, args.out), labels)
    print("wrote {}".format(os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()
