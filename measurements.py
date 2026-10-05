"""Reference-calibrated segment lengths from one triangulation result."""

import math
import numpy as np
from triangulation import triangulate_all


def _pair(value, size):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("segment requires two original point indices")
    if any(type(i) is not int or not 0 <= i < size for i in value):
        raise ValueError("point index outside original input")
    if value[0] == value[1]:
        raise ValueError("segment endpoints must differ")
    return tuple(value)


def _point(row):
    if row["status"] != "ok":
        raise ValueError("segment references rejected triangulation")
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
    indexed = {
        i: row for i, row in enumerate(row for row in rows if row["status"] == "ok")
    }
    a, b = _pair(reference, len(rows))
    pairs = [_pair(s, len(rows)) for s in segments]
    needed = {a, b} | {i for s in pairs for i in s}
    points = {i: _point(indexed[i]) for i in needed}
    vector = points[b][0] - points[a][0]
    distance = float(np.linalg.norm(vector))
    if distance <= 1e-12:
        raise ValueError("reference distance is zero")
    unit = vector / distance
    output = []
    for c, d in pairs:
        target = points[d][0] - points[c][0]
        size = float(np.linalg.norm(target))
        if size <= 1e-12:
            raise ValueError("target distance is zero")
        length = known_length * size / distance
        # Sum contributions before covariance propagation: an endpoint can
        # belong to both the numerator and the denominator.
        target_gradient = known_length / distance * target / size
        scale_gradient = known_length * size / distance**2 * unit
        gradient = {
            c: -target_gradient,
            d: target_gradient,
            a: scale_gradient,
            b: -scale_gradient,
        }
        variance = sum(
            float(g @ points[i][1] @ g)
            for i, g in [
                (c, -target_gradient),
                (d, target_gradient),
                (a, scale_gradient),
                (b, -scale_gradient),
            ]
        )
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
    rows = triangulate_all(
        {k: data[k + "1"] for k in ("K", "R", "t")},
        {k: data[k + "2"] for k in ("K", "R", "t")},
        data["pairs"],
        data["max_px_err"],
        data["sigma"],
    )
    measurement = calibrated_lengths(
        rows, data["reference"], data["known_length"], data["segments"]
    )
    return {"ok": True, "results": rows, "measurement": measurement}
