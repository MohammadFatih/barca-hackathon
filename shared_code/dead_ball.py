"""Dead-ball detection using Smart Tagging SET PIECES intervals.

Approach: use only the first TRIM_FRAMES of each SET PIECES interval (Throw-in, Corner
Kick, Goal Kick, Free Kick, Kick Off) as the true dead-ball stretch. The full tagged
interval overshoots into real possession time -- tested empirically against the public
5-minute ground truth, best result at trim=70 frames (2.8s): spellF1 0.4471 vs baseline
0.4242, possessions matched exactly (17 vs gt 17).

    from dead_ball import detect_dead_ball
    from starter.load_smart_tagging import load as load_tagging

    events = load_tagging()
    set_pieces = [e for e in events if e["code"] == "SET PIECES"]
    is_dead = detect_dead_ball(set_pieces, n_frames=199319)
    # is_dead[i] == True  ->  frame i should be labelled "D"
"""

import numpy as np

TRIM_FRAMES = 70  # 2.8s at 25fps -- best macroF1 in the 60-95 frame plateau


def detect_dead_ball(set_pieces, n_frames, trim_frames=TRIM_FRAMES):
    """Boolean array, length n_frames. True where the frame is dead-ball.

    set_pieces: list of dicts with start_frame/end_frame, filtered to code=="SET PIECES"
                (see starter/load_smart_tagging.py).
    n_frames:   total frames in the match (199319 for BAR-ATH).
    trim_frames: how many frames from the start of each tagged interval count as dead.
    """
    is_dead = np.zeros(n_frames, dtype=bool)
    for sp in set_pieces:
        start = max(0, sp["start_frame"])
        end = min(n_frames, start + trim_frames, sp["end_frame"])
        is_dead[start:end] = True
    return is_dead
