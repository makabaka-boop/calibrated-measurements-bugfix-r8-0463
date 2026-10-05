"""标定量测单元 + 接口测试：

1. 端点编号始终为原始对应点编号（前面有点被拒绝时不得错位）；
2. 已知参考长度按精确值处理，参考段测自己返回该长度及零方差；
3. 多线段共用端点 = 同一随机变量（梯度先按编号求和再传播）；
4. 交换端点顺序不改变长度和方差；
5. 引用被拒绝的点、零长度段、无法表示的数值 -> 整次量测明确失败，
   且失败响应保留同一次三角化的证据与原始拒绝原因；
6. /api/measurements 实际请求/响应行为。
"""

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from measurements import calibrated_lengths, measure_request  # noqa: E402
from server import app  # noqa: E402
from triangulation import project, triangulate_all  # noqa: E402

# ---------------------------------------------------------------- 相机与真值


def look_at(C, target, up=(0.0, 1.0, 0.0)):
    z = np.asarray(target, float) - np.asarray(C, float)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.array([x, y, z])


K1 = [[900.0, 0.0, 480.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]]
R1 = np.eye(3)
t1 = np.zeros(3)
C2 = np.array([1.5, 0.2, 0.5])
R2 = look_at(C2, [0.0, 0.0, 4.5])
t2 = -R2 @ C2
K2 = [[920.0, 0.0, 480.0], [0.0, 910.0, 360.0], [0.0, 0.0, 1.0]]
CAM1 = {"K": K1, "R": R1.tolist(), "t": t1.tolist()}
CAM2 = {"K": K2, "R": R2.tolist(), "t": t2.tolist()}

POINTS3D = [
    [-1.2, -0.8, 3.5],
    [0.8, -0.6, 4.0],
    [-0.5, 0.9, 5.0],
    [1.3, 0.7, 5.5],
]


def make_pairs(noise_sigma=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pairs = []
    for X in POINTS3D:
        m1 = project(np.asarray(K1), R1, t1, np.asarray(X))
        m2 = project(np.asarray(K2), R2, t2, np.asarray(X))
        if noise_sigma > 0:
            m1 = m1 + rng.normal(0, noise_sigma, 2)
            m2 = m2 + rng.normal(0, noise_sigma, 2)
        pairs.append({"m1": m1.tolist(), "m2": m2.tolist()})
    return pairs


def make_request(**overrides):
    req = {
        "K1": K1,
        "R1": R1.tolist(),
        "t1": t1.tolist(),
        "K2": K2,
        "R2": R2.tolist(),
        "t2": t2.tolist(),
        "pairs": make_pairs(),
        "max_px_err": 2.0,
        "sigma": 0.5,
        "reference": [0, 1],
        "known_length": 2.5,
        "segments": [[1, 2], [2, 3]],
    }
    req.update(overrides)
    return req


# ---------------------------------------------------------------- 合成证据行


def ok_row(X, var=1e-4):
    """构造一个 status=ok 的三角化证据行（对角协方差）。"""
    return {
        "status": "ok",
        "X": list(map(float, X)),
        "cov": (np.eye(3) * var).tolist(),
        "reproj1": [0.0, 0.0],
        "reproj2": [0.0, 0.0],
        "err1": 0.0,
        "err2": 0.0,
        "ray_angle_deg": 30.0,
        "iterations": 3,
    }


def rejected_row(index, code="PIXEL_ERROR", reason="重投影误差超限：视图1 60.0 px"):
    return {"index": index, "status": "rejected", "code": code, "reason": reason}


# ---------------------------------------------------------------- 1. 原始编号


def test_indices_are_original_not_compacted():
    """前面有点被拒绝时，后续线段必须引用正确的空间点。"""
    rows = [
        rejected_row(0),
        ok_row([0.0, 0.0, 0.0]),
        ok_row([3.0, 0.0, 0.0]),
        ok_row([0.0, 4.0, 0.0]),
    ]
    out = calibrated_lengths(rows, [1, 2], 6.0, [[2, 3], [1, 3]])
    # 参考段 |X2-X1| = 3 -> 比例尺 2；|X3-X2| = 5 -> 10；|X3-X1| = 4 -> 8
    assert out["scale"] == pytest.approx(2.0)
    assert out["segments"][0]["length"] == pytest.approx(10.0)
    assert out["segments"][1]["length"] == pytest.approx(8.0)


def test_rejected_point_reference_fails_loudly():
    """引用被拒绝的点必须整次失败，不得静默换用其他空间点。"""
    rows = [rejected_row(0), ok_row([0.0, 0.0, 0.0]), ok_row([3.0, 0.0, 0.0])]
    with pytest.raises(ValueError, match="rejected point 0"):
        calibrated_lengths(rows, [1, 2], 6.0, [[0, 2]])
    with pytest.raises(ValueError, match="PIXEL_ERROR"):
        calibrated_lengths(rows, [1, 2], 6.0, [[0, 2]])
    with pytest.raises(ValueError, match="重投影误差超限"):
        calibrated_lengths(rows, [1, 2], 6.0, [[0, 2]])
    # 参考段引用被拒绝的点同样失败
    with pytest.raises(ValueError, match="rejected point 0"):
        calibrated_lengths(rows, [0, 2], 6.0, [[1, 2]])


def test_last_index_after_rejection_not_keyerror():
    """引用最后一个原始编号（旧实现会 KeyError）必须给出明确失败或正确结果。"""
    rows = [rejected_row(0), ok_row([0.0, 0.0, 0.0]), ok_row([3.0, 0.0, 0.0])]
    out = calibrated_lengths(rows, [1, 2], 3.0, [[1, 2], [2, 1]])
    assert out["segments"][0]["length"] == pytest.approx(3.0)
    with pytest.raises(ValueError, match="outside original input"):
        calibrated_lengths(rows, [1, 2], 3.0, [[1, 3]])


# ---------------------------------------------------------------- 2. 参考段测自己


def test_reference_measures_itself_exactly():
    rows = [
        ok_row([0.0, 0.0, 0.0]),
        ok_row([3.0, 4.0, 0.0]),
        ok_row([1.0, 1.0, 1.0]),
    ]
    for seg in ([0, 1], [1, 0]):  # 两种端点顺序
        out = calibrated_lengths(rows, [0, 1], 7.5, [seg])
        s = out["segments"][0]
        assert s["length"] == 7.5  # 精确等于已知长度
        assert s["variance"] == 0.0
        assert s["stddev"] == 0.0
        assert all(all(v == 0.0 for v in g) for g in s["gradient"].values())


# ---------------------------------------------------------------- 3. 共用端点 = 同一随机变量


def test_shared_endpoint_gradient_accumulated():
    """共用端点的梯度贡献必须先求和再做二次型（不能当成不同点）。"""
    c0, c1, c2 = 1e-4, 2e-4, 4e-4
    rows = [
        ok_row([0.0, 0.0, 0.0], var=c0),
        ok_row([1.0, 0.0, 0.0], var=c1),
        ok_row([2.0, 1.0, 0.0], var=c2),
    ]
    L = 2.0
    out = calibrated_lengths(rows, [0, 1], L, [[1, 2]])
    s = out["segments"][0]
    # distance=1, unit=(1,0,0)；target=(1,1,0)，size=sqrt(2)
    tg = np.array([np.sqrt(2.0), np.sqrt(2.0), 0.0])  # L/d * target/size
    sg = np.array([2.0 * np.sqrt(2.0), 0.0, 0.0])  # L*size/d^2 * unit
    g0, g1, g2 = sg, -(tg + sg), tg
    assert np.allclose(s["gradient"]["0"], g0, atol=1e-12)
    assert np.allclose(s["gradient"]["1"], g1, atol=1e-12)
    assert np.allclose(s["gradient"]["2"], g2, atol=1e-12)
    expected_var = c0 * (g0 @ g0) + c1 * (g1 @ g1) + c2 * (g2 @ g2)
    assert s["variance"] == pytest.approx(expected_var, rel=1e-12)
    # 与“把共用端点当不同点”的错误结果区分：错误值会丢掉交叉项
    wrong_var = c0 * (g0 @ g0) + c1 * (tg @ tg + sg @ sg) + c2 * (g2 @ g2)
    assert abs(s["variance"] - wrong_var) > 1e-6


def test_multiple_segments_share_endpoint_consistently():
    """多条线段引用同一端点：共用端点是同一随机变量，分段/合并不改变结果。"""
    rows = [
        ok_row([0.0, 0.0, 0.0], var=1e-4),
        ok_row([1.0, 0.0, 0.0], var=2e-4),
        ok_row([1.0, 1.0, 0.0], var=3e-4),
        ok_row([0.0, 1.0, 0.0], var=4e-4),
    ]
    together = calibrated_lengths(rows, [0, 1], 2.0, [[1, 2], [1, 3]])
    alone_a = calibrated_lengths(rows, [0, 1], 2.0, [[1, 2]])
    alone_b = calibrated_lengths(rows, [0, 1], 2.0, [[1, 3]])
    for joint, single in zip(together["segments"],
                             (alone_a["segments"][0], alone_b["segments"][0])):
        assert joint["length"] == single["length"]
        assert joint["variance"] == single["variance"]
        assert joint["gradient"] == single["gradient"]
    # 共用端点 1 的梯度 = 该段方向项与参考比例项之和（同一变量的全部贡献）
    g1 = together["segments"][0]["gradient"]["1"]
    tg = np.array([0.0, 2.0, 0.0])  # L/d * (X2-X1)/|X2-X1|
    sg = np.array([2.0, 0.0, 0.0])  # L*|X2-X1|/d^2 * (X1-X0)/|X1-X0|
    assert np.allclose(g1, -(tg + sg), atol=1e-12)


# ---------------------------------------------------------------- 4. 端点顺序不变性


def test_endpoint_swap_invariant():
    rows = [
        ok_row([0.3, -0.2, 0.1], var=1e-4),
        ok_row([1.4, 0.5, -0.3], var=2e-4),
        ok_row([-0.6, 0.9, 0.8], var=3e-4),
    ]
    base = calibrated_lengths(rows, [0, 1], 2.5, [[1, 2], [0, 2]])
    swapped_seg = calibrated_lengths(rows, [0, 1], 2.5, [[2, 1], [2, 0]])
    swapped_ref = calibrated_lengths(rows, [1, 0], 2.5, [[1, 2], [0, 2]])
    for other in (swapped_seg, swapped_ref):
        assert other["scale"] == pytest.approx(base["scale"])
        for s0, s1 in zip(base["segments"], other["segments"]):
            assert s0["length"] == pytest.approx(s1["length"], rel=1e-15)
            assert s0["variance"] == pytest.approx(s1["variance"], rel=1e-12)


# ---------------------------------------------------------------- 5. 明确失败


def test_zero_length_segments_fail():
    rows = [
        ok_row([1.0, 1.0, 1.0]),
        ok_row([2.0, 2.0, 2.0]),
        ok_row([1.0, 1.0, 1.0]),  # 与点 0 重建坐标相同
    ]
    with pytest.raises(ValueError, match="target distance is zero"):
        calibrated_lengths(rows, [0, 1], 2.0, [[0, 2]])
    with pytest.raises(ValueError, match="reference distance is zero"):
        calibrated_lengths(rows, [0, 2], 2.0, [[0, 1]])
    with pytest.raises(ValueError, match="endpoints must differ"):
        calibrated_lengths(rows, [0, 1], 2.0, [[1, 1]])


def test_unrepresentable_values_fail():
    rows = [ok_row([0.0, 0.0, 0.0]), ok_row([2.0, 0.0, 0.0]), ok_row([0.0, 3.0, 0.0])]
    for bad in (0.0, -1.0, float("nan"), float("inf"), True, "2.0", None):
        with pytest.raises(ValueError, match="known_length"):
            calibrated_lengths(rows, [0, 1], bad, [[1, 2]])
    # 溢出到 inf 的长度
    with pytest.raises(ValueError, match="numeric range"):
        calibrated_lengths(rows, [0, 1], 1e308, [[1, 2]])
    # 非有限证据
    bad_rows = [ok_row([0.0, 0.0, 0.0]), ok_row([2.0, 0.0, 0.0])]
    bad_rows[1]["cov"][0][0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        calibrated_lengths(bad_rows, [0, 1], 2.0, [[0, 1]])


def test_index_and_count_validation():
    rows = [ok_row([0.0, 0.0, 0.0]), ok_row([2.0, 0.0, 0.0]), ok_row([0.0, 3.0, 0.0])]
    for bad_seg in ([0, 3], [-1, 2], [0, 1.0], [0, True], [0, "1"], [0], [0, 1, 2]):
        with pytest.raises(ValueError):
            calibrated_lengths(rows, [0, 1], 2.0, [bad_seg])
    with pytest.raises(ValueError, match="one to eight"):
        calibrated_lengths(rows, [0, 1], 2.0, [])
    with pytest.raises(ValueError, match="one to eight"):
        calibrated_lengths(rows, [0, 1], 2.0, [[0, 2]] * 9)


# ---------------------------------------------------------------- 6. 同次重建 + 证据保留


def test_measure_request_success_matches_rows():
    """量测值必须由返回的同一次三角化结果复现。"""
    req = make_request()
    out = measure_request(req)
    assert out["ok"] is True
    rows = out["results"]
    assert [r["index"] for r in rows] == list(range(len(rows)))
    m = out["measurement"]
    X = [np.asarray(r["X"]) for r in rows]
    distance = np.linalg.norm(X[1] - X[0])
    assert m["scale"] == pytest.approx(req["known_length"] / distance)
    for seg, (c, d) in zip(m["segments"], req["segments"]):
        size = np.linalg.norm(X[d] - X[c])
        assert seg["length"] == pytest.approx(req["known_length"] * size / distance)
        assert seg["points"] == [c, d]
    # 无噪声投影：长度应还原真值比例
    truth = [np.asarray(p) for p in POINTS3D]
    true_scale = req["known_length"] / np.linalg.norm(truth[1] - truth[0])
    assert m["segments"][0]["length"] == pytest.approx(
        true_scale * np.linalg.norm(truth[2] - truth[1]), rel=1e-6
    )


def test_measure_request_failure_keeps_evidence():
    """引用被拒绝的点：整次失败，但保留同次三角化证据与原始拒绝原因。"""
    pairs = make_pairs()
    pairs[0]["m1"][0] += 60.0  # 让第 0 对被 PIXEL_ERROR 拒绝
    req = make_request(pairs=pairs, max_px_err=1.0, segments=[[0, 2]])
    out = measure_request(req)
    assert out["ok"] is False
    assert "rejected point 0" in out["error"]
    assert "PIXEL_ERROR" in out["error"]
    assert "measurement" not in out  # 整次失败：没有部分量测结果
    rows = out["results"]
    assert rows[0]["status"] == "rejected"
    assert rows[0]["code"] == "PIXEL_ERROR"
    assert "重投影误差超限" in rows[0]["reason"]  # 原始拒绝原因保留
    assert all(r["status"] == "ok" for r in rows[1:])


def test_measure_request_missing_fields():
    for key in ("reference", "known_length", "segments"):
        req = make_request()
        del req[key]
        with pytest.raises(ValueError, match=key):
            measure_request(req)
    with pytest.raises(ValueError, match="object"):
        measure_request(None)


# ---------------------------------------------------------------- 7. HTTP 接口


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_api_success(client):
    resp = client.post("/api/measurements", json=make_request())
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert [r["index"] for r in data["results"]] == list(range(4))
    assert data["measurement"]["reference"] == [0, 1]
    assert len(data["measurement"]["segments"]) == 2
    # 响应可被严格 JSON 解析（无 NaN/Infinity 等无法表示的数值）
    json.loads(resp.get_data(as_text=True), parse_constant=lambda x: 1 / 0)


def test_api_failure_keeps_evidence(client):
    pairs = make_pairs()
    pairs[0]["m1"][0] += 60.0
    req = make_request(pairs=pairs, max_px_err=1.0, segments=[[0, 2], [2, 3]])
    resp = client.post("/api/measurements", json=req)
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["ok"] is False
    assert "rejected point 0" in data["error"]
    assert data["results"][0]["code"] == "PIXEL_ERROR"
    assert "重投影误差超限" in data["results"][0]["reason"]


def test_api_malformed_requests(client):
    resp = client.post("/api/measurements", data="not json{",
                       content_type="application/json")
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False

    resp = client.post("/api/measurements", json={"pairs": []})
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False

    bad = make_request()
    bad["K1"] = [row[:] for row in K1]  # 复制后再改，避免污染模块级常量
    bad["K1"][0][0] = -900.0  # 焦距为负 -> 相机校验拒绝
    resp = client.post("/api/measurements", json=bad)
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["ok"] is False
    assert data["code"] == "BAD_CAMERA"
    assert "焦距" in data["error"]


def test_api_reference_self_and_swap(client):
    req = make_request(segments=[[0, 1], [1, 0], [2, 3], [3, 2]])
    data = client.post("/api/measurements", json=req).get_json()
    segs = data["measurement"]["segments"]
    assert segs[0]["length"] == req["known_length"]
    assert segs[0]["variance"] == 0.0
    assert segs[1]["length"] == segs[0]["length"]
    assert segs[1]["variance"] == 0.0
    assert segs[3]["length"] == pytest.approx(segs[2]["length"])
    assert segs[3]["variance"] == pytest.approx(segs[2]["variance"])
