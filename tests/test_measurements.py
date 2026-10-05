"""标定量测单元测试：

1. 参考段量自己 -> 精确已知长度、精确零方差/零标准差；
2. 多线段共用端点 -> 同一随机变量（联合协方差非对角，且与蒙特卡洛一致的结构）；
3. 交换参考段/线段端点顺序 -> 长度、方差完全不变；
4. 前面的对应点被拒绝 -> 后续编号仍指向正确空间点，不重排；
5. 引用被拒绝的点 -> 整次量测失败，保留原始拒绝 code/reason 与三角化证据；
6. 零长度段（同点引用）、编号越界、非法 known_length、非有限数值 -> 整次失败；
7. 全部输出编号都是原始编号，且与同次三角化结果一一对应。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from measurements import MeasurementFailure, calibrated_lengths, measure_request  # noqa
from triangulation import project, triangulate_all  # noqa

K1 = [[900.0, 0.0, 480.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]]
R1 = np.eye(3)
t1 = np.zeros(3)


def look_at(C, target, up=(0.0, 1.0, 0.0)):
    z = np.asarray(target, float) - np.asarray(C, float)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.array([x, y, z])


C2 = np.array([1.5, 0.2, 0.5])
R2 = look_at(C2, [0.0, 0.0, 4.5])
t2 = -R2 @ C2
K2 = [[920.0, 0.0, 480.0], [0.0, 910.0, 360.0], [0.0, 0.0, 1.0]]

POINTS = [
    [-1.2, -0.8, 3.5],
    [0.8, -0.6, 4.0],
    [-0.5, 0.9, 5.0],
    [1.3, 0.7, 5.5],
]
CAM1 = {"K": K1, "R": R1.tolist(), "t": t1.tolist()}
CAM2 = {"K": K2, "R": R2.tolist(), "t": t2.tolist()}
TRUTH = [np.asarray(X, float) for X in POINTS]


@pytest.fixture
def setup():
    pairs = [
        {
            "m1": project(np.asarray(K1), R1, t1, X).tolist(),
            "m2": project(np.asarray(K2), R2, t2, X).tolist(),
        }
        for X in TRUTH
    ]
    rows = triangulate_all(CAM1, CAM2, pairs, 2.0, 1.0)
    assert all(r["status"] == "ok" for r in rows)
    L01 = float(np.linalg.norm(TRUTH[0] - TRUTH[1]))
    return pairs, rows, L01


def request_body(pairs, **over):
    body = {
        "K1": K1,
        "R1": R1.tolist(),
        "t1": t1.tolist(),
        "K2": K2,
        "R2": R2.tolist(),
        "t2": t2.tolist(),
        "pairs": pairs,
        "max_px_err": 2.0,
        "sigma": 1.0,
        "reference": [0, 1],
        "known_length": 1.0,
        "segments": [[0, 1]],
    }
    body.update(over)
    return body


def truth_len(i, j):
    return float(np.linalg.norm(TRUTH[i] - TRUTH[j]))


# ------------------------------------------------ 1. 参考段量自己


def test_reference_measures_itself_exact(setup):
    _, rows, L01 = setup
    out = calibrated_lengths(rows, [0, 1], L01, [[0, 1]])
    seg = out["segments"][0]
    assert seg["points"] == [0, 1]
    assert seg["length"] == L01  # 精确值，不只是近似
    assert seg["variance"] == 0.0
    assert seg["stddev"] == 0.0
    assert all(v == [0.0, 0.0, 0.0] for v in seg["gradient"].values())
    cov = np.asarray(out["covariance"]["matrix"])
    assert cov.shape == (1, 1) and cov[0, 0] == 0.0


def test_reference_self_with_shared_extra_segment(setup):
    _, rows, L01 = setup
    out = calibrated_lengths(rows, [0, 1], L01, [[0, 1], [1, 2], [0, 2]])
    variances = [s["variance"] for s in out["segments"]]
    assert variances[0] == 0.0
    assert variances[1] > 0 and variances[2] > 0
    assert out["segments"][0]["length"] == L01
    # 其他线段长度符合真值（参考尺度精确）
    for s in out["segments"][1:]:
        assert s["length"] == pytest.approx(truth_len(*s["points"]), abs=1e-8)


# ------------------------------------------------ 2. 共享端点 = 同一随机变量


def test_shared_endpoint_joint_covariance(setup):
    _, rows, L01 = setup
    out = calibrated_lengths(
        rows, [0, 1], L01, [[0, 1], [1, 2], [2, 3], [0, 2]]
    )
    cov = np.asarray(out["covariance"]["matrix"])
    assert out["covariance"]["points"] == [0, 1, 2, 3]
    assert cov.shape == (4, 4)
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() >= 0
    # 对角线即各线段方差
    assert np.allclose(np.diag(cov), [s["variance"] for s in out["segments"]])
    # 共享端点的线段（1-2 与 2-3 共享点 2 等）必须非对角耦合
    assert cov[1, 2] != 0.0
    assert cov[1, 3] != 0.0  # 1-2 与 0-2 共享点 2
    assert cov[2, 3] != 0.0
    # 参考段行/列恒为零（它是精确值）
    assert np.all(cov[0] == 0.0) and np.all(cov[:, 0] == 0.0)


def test_joint_covariance_matches_monte_carlo(setup):
    pairs, _, L01 = setup
    sig = 0.5
    body = request_body(pairs, max_px_err=5.0, sigma=sig, known_length=L01,
                        segments=[[0, 1], [1, 2], [2, 3], [0, 2]])
    pred = np.asarray(measure_request(body)["measurement"]["covariance"]["matrix"])

    rng = np.random.default_rng(11)
    samples = []
    for _ in range(2500):
        pp = [
            {
                "m1": (np.asarray(p["m1"]) + rng.normal(0, sig, 2)).tolist(),
                "m2": (np.asarray(p["m2"]) + rng.normal(0, sig, 2)).tolist(),
            }
            for p in pairs
        ]
        try:
            rows = measure_request({**body, "pairs": pp})["results"]
            mm = calibrated_lengths(rows, [0, 1], L01, body["segments"])
            samples.append([s["length"] for s in mm["segments"]])
        except MeasurementFailure:
            pass
    samples = np.asarray(samples)
    assert len(samples) > 1000
    emp = np.cov(samples.T)
    # 参考段在每次采样中都精确返回已知长度（仅余机器精度的舍入抖动）
    assert np.var(samples[:, 0]) < 1e-20
    # 一阶传播与经验协方差结构一致（对角与非对角，20% 容差）
    for i in range(1, 4):
        for j in range(1, 4):
            assert emp[i, j] == pytest.approx(pred[i, j], rel=0.25, abs=3e-5)
    assert np.sign(emp[1, 2]) == np.sign(pred[1, 2])


# ------------------------------------------------ 3. 交换端点顺序不变


def test_endpoint_order_invariance(setup):
    _, rows, L01 = setup
    segs = [[0, 1], [1, 2], [2, 3], [0, 2]]
    out1 = calibrated_lengths(rows, [0, 1], L01, segs)
    out2 = calibrated_lengths(rows, [1, 0], L01, [[1, 0], [2, 1], [3, 2], [2, 0]])
    for s1, s2 in zip(out1["segments"], out2["segments"]):
        assert s1["length"] == s2["length"]  # 期望位级一致
        assert s1["variance"] == s2["variance"]
        assert s1["stddev"] == s2["stddev"]
    cov1 = np.asarray(out1["covariance"]["matrix"])
    cov2 = np.asarray(out2["covariance"]["matrix"])
    assert np.array_equal(np.diag(cov1), np.diag(cov2))
    # 线段集合相同（顺序相同），整矩阵也应一致
    assert np.allclose(cov1, cov2, atol=1e-18)
    assert out1["scale"] == pytest.approx(out2["scale"])


def test_reversed_reference_in_other_segment(setup):
    _, rows, L01 = setup
    a = calibrated_lengths(rows, [0, 1], L01, [[1, 3]])
    b = calibrated_lengths(rows, [1, 0], L01, [[3, 1]])
    assert a["segments"][0]["length"] == b["segments"][0]["length"]
    assert a["segments"][0]["variance"] == b["segments"][0]["variance"]


# ------------------------------------------------ 4/5. 拒绝点与原始编号


def test_indices_not_reindexed_after_rejection(setup):
    pairs, _, L01 = setup
    bad = [dict(p) for p in pairs]
    bad[1]["m1"][0] += 90.0  # 让原始点 1 被 PIXEL_ERROR 拒绝
    rows = triangulate_all(CAM1, CAM2, bad, 2.0, 1.0)
    assert rows[0]["status"] == "ok"
    assert rows[1]["status"] == "rejected" and rows[1]["code"] == "PIXEL_ERROR"
    assert rows[2]["status"] == "ok" and rows[3]["status"] == "ok"
    # 后续编号 2、3 必须仍指向它们自己的空间点（旧实现会错位到 1、2）
    out = calibrated_lengths(rows, [2, 3], truth_len(2, 3), [[2, 0], [3, 0]])
    assert out["segments"][0]["points"] == [2, 0]
    assert out["segments"][0]["length"] == pytest.approx(truth_len(2, 0), abs=1e-7)
    assert out["segments"][1]["length"] == pytest.approx(truth_len(3, 0), abs=1e-7)


def test_reference_to_rejected_point_fails_entire_measurement(setup):
    pairs, _, L01 = setup
    bad = [dict(p) for p in pairs]
    bad[1]["m1"][0] += 90.0
    rows = triangulate_all(CAM1, CAM2, bad, 2.0, 1.0)
    with pytest.raises(MeasurementFailure) as exc:
        calibrated_lengths(rows, [1, 2], L01, [[2, 3]])
    assert exc.value.code == "REJECTED_POINT"
    assert "1" in exc.value.message and "PIXEL_ERROR" in exc.value.message
    # 不返回任何线段结果：调用方拿不到“部分成功”
    assert not hasattr(exc.value, "segments")


def test_segment_to_rejected_point_fails_with_original_reason(setup):
    pairs, _, L01 = setup
    bad = [dict(p) for p in pairs]
    bad[3]["m1"][0] += 90.0
    body = request_body(bad, reference=[0, 1], known_length=L01,
                        segments=[[0, 1], [2, 3]])
    with pytest.raises(MeasurementFailure) as exc:
        measure_request(body)
    e = exc.value
    assert e.code == "REJECTED_POINT"
    assert e.results is not None
    rejected = [r for r in e.results if r["status"] == "rejected"]
    assert len(rejected) == 1
    assert rejected[0]["index"] == 3
    assert rejected[0]["code"] == "PIXEL_ERROR"  # 原始三角化拒绝原因原样保留
    assert "重投影误差" in rejected[0]["reason"]


# ------------------------------------------------ 6. 非法输入整次失败


def test_zero_length_and_bad_indices(setup):
    _, rows, L01 = setup
    cases = [
        ("SAME_POINT_SEG", [2, 2], None),
        ("RANGE_REF", [0, 9], None),
        ("NEG_REF", [-1, 0], None),
        ("BOOL_REF", True, None),
    ]
    for label, ref, _ in cases:
        if label == "SAME_POINT_SEG":
            with pytest.raises(MeasurementFailure) as exc:
                calibrated_lengths(rows, ref, L01, [[0, 1]])
            assert exc.value.code == "ZERO_LENGTH"
        else:
            with pytest.raises(MeasurementFailure) as exc:
                calibrated_lengths(rows, ref, L01, [[0, 1]])
            assert exc.value.code == "INVALID_REQUEST"

    with pytest.raises(MeasurementFailure) as exc:
        calibrated_lengths(rows, [0, 1], L01, [[2, 2]])
    assert exc.value.code == "ZERO_LENGTH"
    with pytest.raises(MeasurementFailure) as exc:
        calibrated_lengths(rows, [0, 1], L01, [])
    assert exc.value.code == "INVALID_REQUEST"
    with pytest.raises(MeasurementFailure) as exc:
        calibrated_lengths(rows, [0, 1], L01, [[0, 1]] * 9)
    assert exc.value.code == "INVALID_REQUEST"


@pytest.mark.parametrize("bad_len", [0.0, -1.0, float("inf"), float("nan"), True, "1.0"])
def test_bad_known_length(setup, bad_len):
    _, rows, L01 = setup
    with pytest.raises(MeasurementFailure) as exc:
        calibrated_lengths(rows, [0, 1], bad_len, [[0, 1]])
    assert exc.value.code == "INVALID_REQUEST"


def test_measure_request_field_validation():
    with pytest.raises(MeasurementFailure) as exc:
        measure_request({"reference": [0, 1]})
    assert exc.value.code == "INVALID_REQUEST"
    with pytest.raises(MeasurementFailure) as exc:
        measure_request("not-an-object")
    assert exc.value.code == "INVALID_REQUEST"


# ------------------------------------------------ 7. 编号与证据一致性


def test_output_indices_match_original_rows(setup):
    pairs, rows, L01 = setup
    body = request_body(pairs, reference=[3, 0], known_length=truth_len(3, 0),
                        segments=[[3, 0], [0, 2]])
    result = measure_request(body)
    assert result["ok"] is True
    for req_row, out_row in zip(rows, result["results"]):
        assert req_row["index"] == out_row["index"]
        assert req_row["X"] == out_row["X"]
        assert req_row["cov"] == out_row["cov"]
    m = result["measurement"]
    assert m["reference"] == [3, 0]
    assert [s["points"] for s in m["segments"]] == [[3, 0], [0, 2]]
    # 参考段自量精确，尺度与编号无关
    assert m["segments"][0]["length"] == truth_len(3, 0)
    assert m["segments"][0]["variance"] == 0.0
    assert m["segments"][1]["length"] == pytest.approx(truth_len(0, 2), abs=1e-8)


def test_triangulation_called_once_per_request(setup, monkeypatch):
    pairs, _, L01 = setup
    import measurements

    calls = []
    orig = measurements.triangulate_all

    def counting(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(measurements, "triangulate_all", counting)
    body = request_body(pairs, known_length=L01,
                        segments=[[0, 1], [1, 2], [2, 3]])
    measurements.measure_request(body)
    assert len(calls) == 1
