"""Reference-calibrated segment lengths from one triangulation result.

不变量：
  * 端点编号始终是【原始对应点编号】（triangulate_all 返回行的 ``index``），
    绝不按通过/拒绝重新编号；
  * 已知物理参考长度按精确值处理：参考段量自己必须得到该长度与【精确】零方差；
  * 多个线段引用同一端点时，该端点是同一个随机变量——所有线段的长度
    组成联合向量一次性传播协方差，因此联合协方差一般非对角；
  * 交换任一线段（含参考段）的端点顺序不改变长度与方差；
  * 引用被三角化拒绝的点、零长度段或出现无法表示的数值时，【整次量测失败】，
    并保留三角化各行的原始拒绝 code/reason 作为证据。

返回的是解处的一阶（局部）不确定度，不是全局置信保证。
"""

import math

import numpy as np

from triangulation import triangulate_all

ZERO_EPS = 1e-12  # 重建距离不大于该值即判为零长度段


class MeasurementFailure(Exception):
    """整次量测失败；携带稳定 code、中文说明与（可能的）三角化证据。"""

    def __init__(self, code, message, results=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.results = results


# ---------------------------------------------------------------- 请求校验


def _pair(value, size, what):
    if not isinstance(value, list) or len(value) != 2:
        raise MeasurementFailure(
            "INVALID_REQUEST", f"{what}必须是两个原始点编号组成的数组"
        )
    if any(type(i) is not int or not 0 <= i < size for i in value):
        raise MeasurementFailure(
            "INVALID_REQUEST", f"{what}点编号超出原始对应点范围（0..{size - 1}）"
        )
    if value[0] == value[1]:
        raise MeasurementFailure(
            "ZERO_LENGTH", f"{what}两端引用同一原始点 {value[0]}，长度为零"
        )
    return tuple(value)


def _rows_by_index(rows):
    """按三角化行自带的原始 index 建表，拒绝任何错位/重号。"""
    if not isinstance(rows, (list, tuple)):
        raise MeasurementFailure("INVALID_REQUEST", "三角化证据格式非法")
    table = {}
    for row in rows:
        if not isinstance(row, dict) or "index" not in row:
            raise MeasurementFailure("INVALID_REQUEST", "三角化证据缺少原始编号")
        i = row["index"]
        if type(i) is not int or i in table or not 0 <= i < len(rows):
            raise MeasurementFailure("INVALID_REQUEST", "三角化证据原始编号不一致")
        table[i] = row
    if set(table) != set(range(len(rows))):
        raise MeasurementFailure("INVALID_REQUEST", "三角化证据原始编号不连续")
    return table


def _point(table, i):
    """取原始编号 i 的空间点与局部协方差；被拒绝则整次量测失败。"""
    row = table[i]
    if row.get("status") != "ok":
        raise MeasurementFailure(
            "REJECTED_POINT",
            f"原始点 {i} 在三角化中已被拒绝：[{row.get('code')}] {row.get('reason')}",
        )
    x = np.asarray(row["X"], dtype=float)
    c = np.asarray(row["cov"], dtype=float)
    if x.shape != (3,) or c.shape != (3, 3):
        raise MeasurementFailure("INVALID_REQUEST", f"原始点 {i} 的量测证据形状非法")
    if not np.isfinite(x).all() or not np.isfinite(c).all():
        raise MeasurementFailure("NUMERIC_RANGE", f"原始点 {i} 的证据含非有限数值")
    if not np.allclose(c, c.T, rtol=1e-10, atol=1e-12):
        raise MeasurementFailure("INVALID_REQUEST", f"原始点 {i} 的协方差不对称")
    if np.linalg.eigvalsh(c).min() < -1e-12:
        raise MeasurementFailure("INVALID_REQUEST", f"原始点 {i} 的协方差非半正定")
    return x, c


# ---------------------------------------------------------------- 核心


def calibrated_lengths(rows, reference, known_length, segments):
    """见模块 docstring。失败抛 MeasurementFailure，成功返回证据 dict。"""
    if isinstance(known_length, bool) or not isinstance(known_length, (int, float)):
        raise MeasurementFailure(
            "INVALID_REQUEST", "known_length 必须是有限正数（物理精确值）"
        )
    known_length = float(known_length)
    if not math.isfinite(known_length) or known_length <= 0:
        raise MeasurementFailure(
            "INVALID_REQUEST", "known_length 必须是有限正数（物理精确值）"
        )
    if not isinstance(segments, list) or not 1 <= len(segments) <= 8:
        raise MeasurementFailure("INVALID_REQUEST", "segments 必须是 1～8 个线段")

    table = _rows_by_index(rows)
    a, b = _pair(reference, len(rows), "参考段 reference")
    pairs = [_pair(s, len(rows), "线段 segments") for s in segments]

    # 只收集真正被引用的原始点；每个原始点在此只出现一次（同一随机变量）
    needed = sorted({a, b} | {i for c, d in pairs for i in (c, d)})
    points = {i: _point(table, i) for i in needed}

    ref_vec = points[b][0] - points[a][0]
    ref_dist = float(np.linalg.norm(ref_vec))
    if not math.isfinite(ref_dist) or ref_dist <= ZERO_EPS:
        raise MeasurementFailure(
            "ZERO_LENGTH", f"参考段端点 {a}、{b} 的重建距离为零，无法标定尺度"
        )
    ref_unit = ref_vec / ref_dist

    # 梯度矩阵：行=线段（请求顺序），列块=被引用原始点（编号升序，顺序固定）。
    # 端点在同一线段里重复出现（参考段量自己）时必须先【合并】该点的系数，
    # 不能拆成若干 g^T C g 相加——否则抵消项不会精确归零。
    pos = {i: k for k, i in enumerate(needed)}
    grad = np.zeros((len(pairs), len(needed), 3))
    lengths = np.zeros(len(pairs))
    is_self = [False] * len(pairs)

    for r, (c, d) in enumerate(pairs):
        if {c, d} == {a, b}:
            # 参考段量自己：精确参考长度，对任何重建坐标的局部导数恒为零
            lengths[r] = known_length
            is_self[r] = True
            continue

        tgt_vec = points[d][0] - points[c][0]
        tgt_dist = float(np.linalg.norm(tgt_vec))
        if not math.isfinite(tgt_dist) or tgt_dist <= ZERO_EPS:
            raise MeasurementFailure(
                "ZERO_LENGTH",
                f"线段端点 {c}、{d} 的重建距离为零（原始对应点 {c}、{d}）",
            )
        lengths[r] = known_length * tgt_dist / ref_dist

        # L = L0 * |Xd-Xc| / |Xb-Xa|，对各原始点的局部行向量：
        tgt_g = known_length / ref_dist * (tgt_vec / tgt_dist)
        ref_g = known_length * tgt_dist / ref_dist**2 * ref_unit
        merged = {c: -tgt_g, d: tgt_g, a: ref_g, b: -ref_g}
        for i, g in merged.items():  # 同一端点的多份贡献先求和
            grad[r, pos[i]] += g

    if not np.isfinite(lengths).all() or not np.isfinite(grad).all():
        raise MeasurementFailure("NUMERIC_RANGE", "标定结果超出可表示的数值范围")

    # 联合协方差：不同原始点的三角化噪声相互独立（跨点块为零），
    # 同一原始点的三个坐标共用其 3x3 局部协方差——共享端点由此自然耦合。
    joint = np.zeros((len(pairs), len(pairs)))
    for i, (_, cov_i) in points.items():
        gi = grad[:, pos[i], :]
        joint += gi @ cov_i @ gi.T
    joint = 0.5 * (joint + joint.T)  # 消去浮点层面的微小不对称

    scale = abs(joint).max()
    eig_min = float(np.linalg.eigvalsh(joint).min())
    if not np.isfinite(joint).all() or eig_min < -1e-9 * max(1.0, scale):
        raise MeasurementFailure("NUMERIC_RANGE", "传播后方差无法表示或非半正定")
    variances = np.clip(np.diag(joint), 0.0, None)

    out_segments = []
    for r, (c, d) in enumerate(pairs):
        if is_self[r]:
            gradient_out = {
                str(a): [0.0, 0.0, 0.0],
                str(b): [0.0, 0.0, 0.0],
            }
        else:
            # dict 键去重：同一端点出现多次时输出的是上面已合并的导数
            gradient_out = {
                str(i): grad[r, pos[i]].tolist()
                for i in (c, d, a, b)
            }
        v = float(variances[r])
        out_segments.append(
            {
                "points": [c, d],  # 保留请求中的原始编号与书写顺序
                "length": float(lengths[r]),
                "variance": v,
                "stddev": math.sqrt(v),
                "gradient": gradient_out,
            }
        )

    return {
        "reference": [a, b],
        "known_length": known_length,
        "scale": known_length / ref_dist,
        "points": needed,  # 联合协方差的行/列顺序
        "segments": out_segments,
        "covariance": {"points": needed, "matrix": joint.tolist()},
    }


# ---------------------------------------------------------------- 请求入口


def measure_request(data):
    """校验请求 -> 三角化（仅一次）-> 量测。

    相机/三角化参数非法时抛 TriangulationRejected（由 server 转成带 code 的 400）；
    量测层面失败时抛 MeasurementFailure，并挂上同次三角化证据 results。
    """
    if not isinstance(data, dict):
        raise MeasurementFailure("INVALID_REQUEST", "请求体必须是 JSON 对象")
    for key in ("reference", "known_length", "segments"):
        if key not in data:
            raise MeasurementFailure("INVALID_REQUEST", f"缺少字段：{key}")

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
    except MeasurementFailure as exc:
        # 整次量测失败：保留同次三角化的全部证据与原始拒绝原因
        exc.results = rows
        raise
    return {"ok": True, "results": rows, "measurement": measurement}
