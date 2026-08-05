from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


EPS = 1e-6


@dataclass(frozen=True)
class RaliV2Result:
    y_r: float
    r: float
    p: float
    n: float | None
    c_p: float
    c_n: float | None
    branch: str


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def target_extremity(score: float) -> float:
    score = clamp(score, 0.0, 1.0)
    return max(score, 1.0 - score)


def rali_v2_clip(
    p: float,
    n: float | None,
    *,
    video_label: int,
    alpha: float,
    tau: float,
    r_min: float,
) -> RaliV2Result:
    p = clamp(p, 0.0, 1.0)
    c_p = target_extremity(p)

    if n is None:
        if int(video_label) == 0:
            return RaliV2Result(y_r=0.0, r=1.0, p=p, n=None, c_p=c_p, c_n=None, branch="missing_normal_zero")
        return RaliV2Result(y_r=p, r=c_p, p=p, n=None, c_p=c_p, c_n=None, branch="missing_abnormal_pseudo")

    n = clamp(n, 0.0, 1.0)
    c_n = target_extremity(n)
    w_p = alpha * c_p
    w_n = (1.0 - alpha) * c_n
    agreement = 1.0 - abs(p - n)

    if abs(p - n) <= tau:
        y_r = (w_p * p + w_n * n) / (w_p + w_n + EPS)
        branch = "agreement_weighted_mean"
    elif w_p >= w_n:
        y_r = p
        branch = "conflict_use_pseudo"
    else:
        y_r = n
        branch = "conflict_use_mllm"

    r = clamp(agreement * (w_p + w_n), r_min, 1.0)
    return RaliV2Result(y_r=clamp(y_r, 0.0, 1.0), r=r, p=p, n=n, c_p=c_p, c_n=c_n, branch=branch)


def quantiles(values: Iterable[float]) -> dict[str, float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {}

    def q(frac: float) -> float:
        pos = frac * (len(vals) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(vals) - 1)
        w = pos - lo
        return vals[lo] * (1.0 - w) + vals[hi] * w

    return {
        "min": vals[0],
        "q05": q(0.05),
        "q25": q(0.25),
        "median": q(0.50),
        "q75": q(0.75),
        "q95": q(0.95),
        "max": vals[-1],
        "mean": sum(vals) / len(vals),
    }

