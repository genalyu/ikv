"""Versioned normalization for absolute EEF position / column-rot6d / gripper.

Fit on TRAIN episodes only. Rotation stays in its physical [-1,1] range;
position spans cover the observed training workspace with a small margin.
No clipping: extrapolation remains visible to audits.
"""
import numpy as np

ROTATION_IDS = [*range(3, 9), *range(13, 19)]
POSITION_IDS = [0, 1, 2, 10, 11, 12]
GRIPPER_IDS = [9, 19]
VERSION = "absolute-eef20-native-rotation-workspace-v1"


def fit_eef20_bounds(actions):
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 20 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Finite nonempty [N,20] absolute EEF actions required")
    if np.abs(a[:, ROTATION_IDS]).max() > 1.0001:
        raise ValueError("Rotation components outside physical [-1,1] range")
    if a[:, GRIPPER_IDS].min() < -1e-4 or a[:, GRIPPER_IDS].max() > 1.0001:
        raise ValueError("Grippers must be converted to [0,1] before fitting")
    lo, hi = np.full(20, -1.), np.full(20, 1.)
    for i in POSITION_IDS:
        low, high = float(a[:, i].min()), float(a[:, i].max())
        margin = max(.01, .05 * (high - low))
        lo[i], hi[i] = low - margin, high + margin
    lo[GRIPPER_IDS] = 0.
    return dict(version=VERSION, q01=lo.tolist(), q99=hi.tolist(),
                fitting_split="train", clipping=False,
                rotation_ids=ROTATION_IDS, position_ids=POSITION_IDS,
                gripper_ids=GRIPPER_IDS,
                note="Bounds stored in legacy q01/q99 fields; these are physical/workspace bounds, not percentiles")
