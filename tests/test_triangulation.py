"""三角化核心单元测试：

1. 已知三维点经两相机独立投影 -> 三角化应还原；
2. 投影加轻微噪声 -> 结果应接近真值且误差在限内；
3. 近共线射线（夹角距 0/180 度 < 0.5 度）-> 拒绝；
4. 反向相机（解在相机后方）-> 拒绝。
另含相机校验、误差上限、不收敛、秩不足与“同次求解”一致性测试。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triangulation import (  # noqa: E402
    TriangulationRejected,
    dlt_triangulate,
    project,
    reprojection_jacobian,
    triangulate_all,
)


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

POINTS3D = [
    [-1.2, -0.8, 3.5],
    [0.8, -0.6, 4.0],
    [-0.5, 0.9, 5.0],
    [1.3, 0.7, 5.5],
    [0.0, 0.0, 4.2],
    [-1.5, 0.3, 6.0],
    [0.4, -1.1, 3.8],
    [1.0, 1.2, 4.6],
]

CAM1 = {"K": K1, "R": R1.tolist(), "t": t1.tolist()}
CAM2 = {"K": K2, "R": R2.tolist(), "t": t2.tolist()}


def make_pairs(noise_sigma=0.0, seed=0):
    """已知三维点经两相机各自独立投影得到对应点。"""
    rng = np.random.default_rng(seed)
    pairs, truth = [], []
    for X in POINTS3D:
        m1 = project(np.asarray(K1), R1, t1, np.asarray(X))
        m2 = project(np.asarray(K2), R2, t2, np.asarray(X))
        if noise_sigma > 0:
            m1 = m1 + rng.normal(0, noise_sigma, 2)
            m2 = m2 + rng.normal(0, noise_sigma, 2)
        pairs.append({"m1": m1.tolist(), "m2": m2.tolist()})
        truth.append(np.asarray(X))
    return pairs, truth


# ------------------------------------------------ 1. 精确已知点


def test_exact_known_points():
    pairs, truth = make_pairs()
    results = triangulate_all(CAM1, CAM2, pairs, max_px_err=2.0, sigma=1.0)
    assert all(r["status"] == "ok" for r in results)
    for res, X_true in zip(results, truth):
        assert np.allclose(res["X"], X_true, atol=1e-6)
        assert res["err1"] < 1e-6 and res["err2"] < 1e-6
        cov = np.asarray(res["cov"])
        assert cov.shape == (3, 3)
        assert np.allclose(cov, cov.T, atol=1e-12)  # 对称
        assert np.all(np.diag(cov) > 0)  # 正定对角
        assert 0.5 <= res["ray_angle_deg"] <= 179.5


# ------------------------------------------------ 2. 轻微噪声


def test_slight_noise():
    pairs, truth = make_pairs(noise_sigma=0.4, seed=42)
    results = triangulate_all(CAM1, CAM2, pairs, max_px_err=3.0, sigma=0.5)
    for res, X_true in zip(results, truth):
        assert res["status"] == "ok"
        assert np.linalg.norm(np.asarray(res["X"]) - X_true) < 0.05
        assert res["err1"] <= 3.0 and res["err2"] <= 3.0


# ------------------------------------------------ 3. 近共线射线


def test_near_parallel_rays_rejected():
    # 基线 1cm、深度 10m：夹角约 0.057 度 < 0.5 度
    cam2_near = {"K": K2, "R": R1.tolist(), "t": [-0.01, 0.0, 0.0]}
    X = np.array([0.0, 0.0, 10.0])
    m1 = project(np.asarray(K1), R1, t1, X)
    m2 = project(np.asarray(K2), R1, np.array([-0.01, 0.0, 0.0]), X)
    pairs = [{"m1": m1.tolist(), "m2": m2.tolist()}]
    results = triangulate_all(CAM1, cam2_near, pairs, max_px_err=2.0, sigma=1.0)
    assert results[0]["status"] == "rejected"
    assert results[0]["code"] == "RAY_ANGLE"


def test_near_opposite_rays_rejected():
    # 两相机同中心、光轴几乎相反：夹角距 180 度 < 0.5 度
    ang = np.pi - 1e-4
    R_opp = np.array(
        [[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]]
    )
    cam2_opp = {"K": K2, "R": R_opp.tolist(), "t": [0.0, 0.0, 0.0]}
    pairs = [{"m1": [480.0, 360.0], "m2": [480.0, 360.0]}]
    results = triangulate_all(CAM1, cam2_opp, pairs, max_px_err=2.0, sigma=1.0)
    assert results[0]["status"] == "rejected"
    assert results[0]["code"] == "RAY_ANGLE"


# ------------------------------------------------ 4. 反向相机（解在相机后方）


def test_reversed_camera_behind_rejected():
    # 相机2 朝向 -z（绕 y 转 180 度），中心在 (1,0,0)
    R_rev = np.diag([-1.0, 1.0, -1.0])
    C = np.array([1.0, 0.0, 0.0])
    cam2_rev = {"K": K2, "R": R_rev.tolist(), "t": (-R_rev @ C).tolist()}
    # 该对应关系的唯一最小二乘交点为 (0,0,-45)：在相机1后方、相机2前方
    pairs = [{"m1": [480.0, 360.0], "m2": [500.0, 360.0]}]
    results = triangulate_all(CAM1, cam2_rev, pairs, max_px_err=2.0, sigma=1.0)
    assert results[0]["status"] == "rejected"
    assert results[0]["code"] == "BEHIND_CAMERA"
    assert "相机1" in results[0]["reason"]


# ------------------------------------------------ 相机校验


def test_camera_validation():
    bad_K = {
        "K": [[-900, 0, 480], [0, 900, 360], [0, 0, 1]],
        "R": R1.tolist(),
        "t": t1.tolist(),
    }
    with pytest.raises(TriangulationRejected, match="焦距"):
        triangulate_all(bad_K, CAM2, [{"m1": [1, 2], "m2": [3, 4]}], 2.0, 1.0)

    bad_K2 = {
        "K": [[900, 0, 480], [1, 900, 360], [0, 0, 1]],
        "R": R1.tolist(),
        "t": t1.tolist(),
    }
    with pytest.raises(TriangulationRejected, match="上三角"):
        triangulate_all(bad_K2, CAM2, [{"m1": [1, 2], "m2": [3, 4]}], 2.0, 1.0)

    bad_R = {"K": K1, "R": np.diag([1, 1, -1]).tolist(), "t": t1.tolist()}
    with pytest.raises(TriangulationRejected, match="det"):
        triangulate_all(bad_R, CAM2, [{"m1": [1, 2], "m2": [3, 4]}], 2.0, 1.0)

    bad_R2 = {"K": K1, "R": (2 * R1).tolist(), "t": t1.tolist()}
    with pytest.raises(TriangulationRejected, match="不正交"):
        triangulate_all(bad_R2, CAM2, [{"m1": [1, 2], "m2": [3, 4]}], 2.0, 1.0)

    bad_t = {"K": K1, "R": R1.tolist(), "t": [[0], [0], [0]]}
    with pytest.raises(TriangulationRejected, match="形状"):
        triangulate_all(bad_t, CAM2, [{"m1": [1, 2], "m2": [3, 4]}], 2.0, 1.0)


# ------------------------------------------------ 误差上限 / 点数上限 / 不收敛 / 秩不足


def test_pixel_error_threshold():
    pairs, _ = make_pairs()
    pairs[0]["m1"][0] += 60.0  # 视图1 人为引入 60px 误差（拟合吸收后仍超限）
    results = triangulate_all(CAM1, CAM2, pairs, max_px_err=1.0, sigma=1.0)
    assert results[0]["status"] == "rejected"
    assert results[0]["code"] == "PIXEL_ERROR"
    assert all(r["status"] == "ok" for r in results[1:])


def test_max_pairs_limit():
    pairs, _ = make_pairs()
    too_many = (pairs * 3)[:21]
    with pytest.raises(TriangulationRejected, match="20"):
        triangulate_all(CAM1, CAM2, too_many, 2.0, 1.0)


def test_not_converged():
    pairs, _ = make_pairs()
    results = triangulate_all(CAM1, CAM2, pairs[:1], 2.0, 1.0, max_iter=0)
    assert results[0]["status"] == "rejected"
    assert results[0]["code"] == "NOT_CONVERGED"


def test_rank_deficient_dlt():
    # 两个相同的投影矩阵 + 相同像素 -> DLT 方程秩 2
    P = np.asarray(K1) @ np.hstack([R1, t1.reshape(3, 1)])
    x = np.array([100.0, 200.0, 1.0])
    with pytest.raises(TriangulationRejected) as exc:
        dlt_triangulate(P, P, x, x)
    assert exc.value.code == "RANK_DEFICIENT"


# ------------------------------------------------ 同次求解一致性


def test_results_come_from_same_solve():
    pairs, _ = make_pairs(noise_sigma=0.4, seed=7)
    sigma = 0.7
    results = triangulate_all(CAM1, CAM2, pairs, max_px_err=3.0, sigma=sigma)
    K1a, K2a = np.asarray(K1), np.asarray(K2)
    for res, p in zip(results, pairs):
        assert res["status"] == "ok"
        X = np.asarray(res["X"])
        # 重投影与误差必须由返回的 X 复现
        rp1 = project(K1a, R1, t1, X)
        rp2 = project(K2a, R2, t2, X)
        assert np.allclose(res["reproj1"], rp1, atol=1e-9)
        assert np.allclose(res["reproj2"], rp2, atol=1e-9)
        assert res["err1"] == pytest.approx(np.linalg.norm(rp1 - p["m1"]), abs=1e-9)
        assert res["err2"] == pytest.approx(np.linalg.norm(rp2 - p["m2"]), abs=1e-9)
        # 协方差必须由同一点的雅可比复现：sigma^2 (J^T J)^{-1}
        J = np.vstack(
            [
                reprojection_jacobian(K1a, R1, t1, X),
                reprojection_jacobian(K2a, R2, t2, X),
            ]
        )
        cov_expected = sigma**2 * np.linalg.inv(J.T @ J)
        assert np.allclose(res["cov"], cov_expected, rtol=1e-9, atol=1e-15)
