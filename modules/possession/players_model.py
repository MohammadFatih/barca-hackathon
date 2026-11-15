"""The players-only model: which team has the ball, from the players alone. Runs on any match.

    logit, info = players_model.players_logit(pos, team, S, pitch_length)

Which team has the ball, read from the positions of the players alone. Trained on pseudo-labels
that an algorithm produces from the same match (the ball at one team's feet for a while), with
time-blocked cross-validation, so every prediction is out-of-fold. No human label is involved.

The idea: the team with the ball spreads out and moves forward, the team without it squeezes in
and drops off. Every feature is collective, because track slots are not people, and is built so
that swapping the two teams flips its sign. The model has no intercept, so it cannot learn
"the home team usually has the ball". numpy only.
"""

import warnings

import numpy as np

HOME, AWAY = 0, 1
FPS = 25
CONTROL_R = 1.5            # m   ball within reach of a player's feet
DWELL = 12                 # frames of unbroken control on each side before a frame is a pseudo-label
VEL_K = 5                  # frames: player velocity is a central difference over +/- 0.2 s
MAX_PLAYER_SPEED = 12.0    # m/s: faster than any sprint, so the track slot changed identity
SMOOTH = (12, 75)          # frames: features are averaged over +/- 0.5 s and +/- 3 s
TREND = 50                 # frames: a trend is the change from 2 s before to 2 s after
BLOCKS_PER_HALF = 5        # time blocks for cross-validation
L2 = 10.0                  # ridge penalty of the logistic regression
TEMPERATURES = np.linspace(0.3, 1.5, 49)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def moving_average(x, w):
    """Centred mean over +/- w frames, ignoring NaN. NaN where the whole window is NaN."""
    ok = ~np.isnan(x)
    total = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    count = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - w), np.minimum(len(x), i + w + 1)
    n = count[hi] - count[lo]
    return np.where(n > 0, (total[hi] - total[lo]) / np.maximum(n, 1), np.nan)


def windowed_all(mask, w):
    """True where mask holds on every frame from t - w to t + w."""
    total = np.concatenate(([0], np.cumsum(mask)))
    i = np.arange(len(mask))
    lo, hi = np.maximum(0, i - w), np.minimum(len(mask), i + w + 1)
    return (total[hi] - total[lo]) == (2 * w + 1)


def fit_logistic(Z, y, l2):
    """Weights of a logistic regression without intercept, by Newton's method."""
    w = np.zeros(Z.shape[1])
    for _ in range(50):
        p = sigmoid(Z @ w)
        gradient = Z.T @ (p - y) + l2 * w
        hessian = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.eye(len(w))
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def log_loss(logit, y):
    p = np.clip(sigmoid(logit), 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def features(pos, team, blocks, length):
    """[T, 14] collective features. Swapping the two teams flips the sign of every one."""
    n_frames = len(pos)
    is_home, is_away = team == HOME, team == AWAY
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        home_x = np.nanmean(pos[:, is_home, 0], axis=1)
        away_x = np.nanmean(pos[:, is_away, 0], axis=1)

    direction = np.zeros(n_frames)       # +1: home attacks towards larger x. 0 outside the halves
    for first, last in blocks:
        block = slice(first, last + 1)
        direction[block] = 1.0 if np.nanmean(home_x[block] - away_x[block]) < 0 else -1.0

    vel = np.full_like(pos, np.nan)
    vel[VEL_K:-VEL_K] = (pos[2 * VEL_K:] - pos[:-2 * VEL_K]) * (FPS / (2.0 * VEL_K))
    speed = np.hypot(vel[..., 0], vel[..., 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        too_fast = speed > MAX_PLAYER_SPEED
    vel[too_fast] = np.nan
    speed[too_fast] = np.nan

    flip = direction[:, None] < 0
    u = np.where(flip, length - pos[..., 0], pos[..., 0])    # metres from the home goal line
    v_u = vel[..., 0] * direction[:, None]
    y = pos[..., 1]

    def per_team(values, members, func):
        return func(np.where(members[None, :], values, np.nan), axis=1)

    def back_line(depth, members):
        d = np.where(members[None, :] & ~np.isnan(depth), depth, np.inf)
        second = np.partition(d, 1, axis=1)[:, 1]
        return np.where(np.isfinite(second), second, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = {
            "width": per_team(y, is_home, np.nanstd) - per_team(y, is_away, np.nanstd),
            "v_home": per_team(v_u, is_home, np.nanmean),
            "v_away": per_team(v_u, is_away, np.nanmean),
            "speed": per_team(speed, is_home, np.nanmean) - per_team(speed, is_away, np.nanmean),
            "territory": np.nanmean(u, axis=1) - length / 2.0,
            "back_line": back_line(u, is_home) - back_line(length - u, is_away),
        }

    columns = []
    for values in base.values():
        for w in SMOOTH:
            columns.append(moving_average(values, w))
    for name in ("territory", "back_line"):
        smooth = moving_average(base[name], SMOOTH[0])
        trend = np.full(n_frames, np.nan)
        trend[TREND:-TREND] = smooth[2 * TREND:] - smooth[:-2 * TREND]
        columns.append(trend)
    return np.stack(columns, axis=1)


def players_logit(pos, team, S, length):
    """Calibrated out-of-fold log-odds(home has the ball) on every frame, 0 where there is none.

    S is the dict of possession.signals_from_arrays for the same match: it supplies the halves
    and the ball-at-feet frames the pseudo-labels are made from.
    """
    blocks, graded = S["blocks"], S["graded"]
    n_frames = len(pos)
    X = features(pos, team, blocks, length)
    valid = graded & ~np.isnan(X).any(axis=1)
    Z = np.nan_to_num(X / X[valid].std(axis=0))    # scaled, not centred: a swap must flip the sign

    controlled = S["usable"] & (S["d_min"] < CONTROL_R)
    label_home = windowed_all(controlled & (S["nearest"] == HOME), DWELL)
    label_away = windowed_all(controlled & (S["nearest"] == AWAY), DWELL)
    labelled = (label_home | label_away) & valid
    target = label_home.astype(float)

    fold = np.full(n_frames, -1)
    for index, (first, last) in enumerate(blocks):
        n = last - first + 1
        fold[first:last + 1] = index * BLOCKS_PER_HALF + np.minimum(
            BLOCKS_PER_HALF - 1, np.arange(n) * BLOCKS_PER_HALF // n)

    oof = np.zeros(n_frames)
    for k in range(len(blocks) * BLOCKS_PER_HALF):
        train = labelled & (fold != k)
        test = valid & (fold == k)
        oof[test] = Z[test] @ fit_logistic(Z[train], target[train], L2)

    temperature = TEMPERATURES[np.argmin([log_loss(t * oof[labelled], target[labelled])
                                          for t in TEMPERATURES])]
    info = {
        "pseudo_labels": int(labelled.sum()),
        "home_share": float(target[labelled].mean()),
        "accuracy": float(((oof[labelled] > 0) == (target[labelled] == 1)).mean()),
        "temperature": float(temperature),
        "features_valid": float(valid[graded].mean()),
    }
    return temperature * oof, info
