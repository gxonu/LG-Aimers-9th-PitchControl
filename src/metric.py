"""Competition metric: Brier Skill Score (DACON 236743, LG Aimers 9기).

Score = max(0, 100000 * (1 - Brier / (r*(1-r))))
  Brier = mean((p - y)^2)
  r     = base rate (mean of y) of the eval set  (constant prediction of r -> score 0)

A constant base-rate prediction scores exactly 0, so probability CALIBRATION is
the primary lever, not ranking (unlike AUC).
"""
from __future__ import annotations

import numpy as np


def brier_score(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def brier_skill_score(y, p, r: float | None = None) -> float:
    """Brier Skill Score, scaled x100000 with a max(0, .) floor.

    y : true 0/1 labels of the eval set
    p : predicted probabilities in [0, 1]
    r : reference base rate. Defaults to mean(y) of the eval set (the official
        definition uses the *eval set* mean, which is non-public for the real
        test but equals mean(y) for any local holdout).
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if r is None:
        r = float(np.mean(y))
    ref = r * (1.0 - r)
    if ref <= 0:
        return 0.0
    bs = brier_score(y, p)
    return float(max(0.0, 100000.0 * (1.0 - bs / ref)))
