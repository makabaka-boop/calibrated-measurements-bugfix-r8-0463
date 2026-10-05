"""Reference-calibrated segment lengths from one triangulation result."""

import math
import numpy as np
from triangulation import triangulate_all

MIN_DISTANCE = 1e-12  # 参考段/目标段重建距离下限，低于此值视为零长度段


def _pair(value, size):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("segment requires two original point indices")
    if any(type(i) is not int or not 0 <= i < size for i in value):
        raise ValueError("point index outside original input")
    if value[0] == value[1]:
        raise ValueError("segment endpoints must differ")
    return tuple(value)


def _point(index, row):
    if row["status"] != "ok":
        raise ValueError(
            f"segment references rejected point {index}: "
            f"{row.get('code', '?')}: {row.get('reason', '?')}"
        )
    x = np.asarray(row["X"], dtype=float)
    c = np.asarray(row["cov"], dtype=float)
    if x.shape != (3,) or c.shape != (3, 3):
        raise ValueError("invalid point evidence")
    if not np.isfinite(x).all() or not np.isfinite(c).all():
        raise ValueError("nonfinite point evidence")
    if not np.allclose(c, c.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance is not symmetric")
    if np.linalg.eigvalsh(c).min() < -1e-12:
        raise ValueError("covariance is not positive semidefinite")
    return x, c


def _finite_distance(vector, label):
    distance = float(np.linalg.norm(vector))
    if not (math.isfinite(distance) and distance > MIN_DISTANCE):
        raise ValueError(f"{label} distance is zero")
    return distance


def calibrated_lengths(rows, reference, known_length, segments):
    """Independent point estimates; shared endpoints refer to the same variable.

    The scale reference is exact; its reconstructed distance is uncertain.
    This returns local first-order uncertainty, not a global confidence bound.
    """
    if isinstance(known_length, bool) or not isinstance(known_length, (int, float)):
        raise ValueError("known_length must be finite and positive")
    if not math.isfinite(known_length) or known_length <= 0:
        raise ValueError("known_length must be finite and positive")
    if not isinstance(segments, list) or not 1 <= len(segments) <= 8:
        raise ValueError("one to eight segments required")
    if not isinstance(rows, list) or not rows:
        raise ValueError("triangulation evidence must be a non-empty list")
    # 始终按原始对应点编号索引：被拒绝的点留在原位置，引用它即整次失败。
    indexed = {i: row for i, row in enumerate(rows)}
    a, b = _pair(reference, len(rows))
    pairs = [_pair(s, len(rows)) for s in segments]
    needed = {a, b} | {i for s in pairs for i in s}
    points = {i: _point(i, indexed[i]) for i in needed}
    vector = points[b][0] - points[a][0]
    distance = _finite_distance(vector, "reference")
    unit = vector / distance
    output = []
    for c, d in pairs:
        target = points[d][0] - points[c][0]
        size = _finite_distance(target, "target")
        if c in (a, b) and d in (a, b):
            # 参考段测自己：已知长度是精确值，比例分子分母完全抵消，
            # 长度与方差精确（端点交换不影响结果）。
            output.append(
                {
                    "points": [c, d],
                    "length": float(known_length),
                    "variance": 0.0,
                    "stddev": 0.0,
                    "gradient": {str(a): [0.0, 0.0, 0.0], str(b): [0.0, 0.0, 0.0]},
                }
            )
            continue
        length = known_length * size / distance
        # 溢出/无效值在下方 isfinite 检查中统一拒绝，此处不逐次告警
        with np.errstate(over="ignore", invalid="ignore"):
            target_gradient = known_length / distance * target / size
            scale_gradient = known_length * size / distance**2 * unit
            # 同一原始编号的端点是同一随机变量：先把梯度贡献按编号求和，
            # 再对每个点做一次二次型，不得把共用端点当成不同点重复累加。
            gradient = {}
            for i, g in (
                (c, -target_gradient),
                (d, target_gradient),
                (a, scale_gradient),
                (b, -scale_gradient),
            ):
                gradient[i] = gradient[i] + g if i in gradient else g.copy()
            variance = sum(float(g @ points[i][1] @ g) for i, g in gradient.items())
        if not math.isfinite(length) or not math.isfinite(variance):
            raise ValueError("calibrated result exceeds numeric range")
        if variance < -1e-12:
            raise ValueError("negative variance")
        output.append(
            {
                "points": [c, d],
                "length": length,
                "variance": max(0.0, variance),
                "stddev": math.sqrt(max(0.0, variance)),
                "gradient": {str(i): g.tolist() for i, g in gradient.items()},
            }
        )
    return {
        "reference": [a, b],
        "known_length": known_length,
        "scale": known_length / distance,
        "segments": output,
    }


def measure_request(data):
    if not isinstance(data, dict):
        raise ValueError("request must be an object")
    for key in ("reference", "known_length", "segments"):
        if key not in data:
            raise ValueError(f"missing field: {key}")
    rows = triangulate_all(
        {k: data[k + "1"] for k in ("K", "R", "t")},
        {k: data[k + "2"] for k in ("K", "R", "t")},
        data["pairs"],
        data["max_px_err"],
        data["sigma"],
    )
    try:
        measurement = calibrated_lengths(
            rows, data["reference"], data["known_length"], data["segments"]
        )
    except (ValueError, KeyError, TypeError) as exc:
        # 整次量测失败：仍返回同一次三角化的证据
        # （原始编号 + 每个被拒绝点的原始拒绝原因）。
        return {"ok": False, "error": str(exc), "results": rows}
    return {"ok": True, "results": rows, "measurement": measurement}
