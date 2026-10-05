"""双视图三角化核心算法。

约定（固定，不可配置）：
    X_camera = R · X_world + t
    像素投影  x ~ K · X_camera ，K 为上三角内参（图片已去畸变）。

流程：校验相机矩阵 -> 归一化 DLT 求初值 -> 自写阻尼最小二乘
（解析重投影雅可比 + Levenberg 阻尼 + 下降检查，有限次迭代）->
误差/几何校验 -> 输出局部协方差 sigma^2 (J^T J)^{-1}。

只允许使用 NumPy 的矩阵分解/线性求解；不调用任何现成三角化
或非线性优化函数。
"""

import numpy as np

MAX_PAIRS = 20  # 最多 20 对对应点
RAY_ANGLE_MIN_DEG = 0.5  # 射线夹角距 0 度或 180 度小于该值即拒绝
DEFAULT_MAX_ITER = 50  # 最小二乘有限迭代上限


class TriangulationRejected(Exception):
    """携带具体拒绝原因（code + 中文描述）。"""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------- 相机校验


def _as_array(name, value, shape, code="BAD_CAMERA"):
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise TriangulationRejected(code, f"{name} 不是数值数组")
    if arr.shape != shape:
        raise TriangulationRejected(
            code, f"{name} 形状应为 {shape}，实际为 {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise TriangulationRejected(code, f"{name} 含有非有限数值")
    return arr


def validate_intrinsics(K):
    """校验并归一化内参 K（上三角、焦距为正、K(3,3)=1）。"""
    K = _as_array("K", K, (3, 3))
    if abs(K[2, 2]) < 1e-12:
        raise TriangulationRejected("BAD_CAMERA", "K(3,3) 不能为 0")
    K = K / K[2, 2]
    if max(abs(K[1, 0]), abs(K[2, 0]), abs(K[2, 1])) > 1e-9:
        raise TriangulationRejected(
            "BAD_CAMERA", "K 必须是上三角矩阵（无畸变针孔模型）"
        )
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise TriangulationRejected(
            "BAD_CAMERA", f"K 的焦距必须为正（fx={K[0,0]}, fy={K[1,1]}）"
        )
    return K


def validate_rotation(R):
    """校验旋转矩阵：正交且 det=+1（拒绝反射/镜像）。"""
    R = _as_array("R", R, (3, 3))
    ortho_err = np.linalg.norm(R.T @ R - np.eye(3), ord="fro")
    if ortho_err > 1e-6:
        raise TriangulationRejected(
            "BAD_CAMERA", f"R 不正交：||R^T R - I||_F = {ortho_err:.3e}"
        )
    det = float(np.linalg.det(R))
    if abs(det - 1.0) > 1e-6:
        raise TriangulationRejected("BAD_CAMERA", f"det(R) 必须为 +1，实际为 {det:.6f}")
    return R


def validate_translation(t):
    return _as_array("t", t, (3,))


def validate_camera(K, R, t, tag):
    try:
        return (validate_intrinsics(K), validate_rotation(R), validate_translation(t))
    except TriangulationRejected as e:
        raise TriangulationRejected(e.code, f"相机{tag}：{e.message}")


# ---------------------------------------------------------------- 投影与雅可比


def camera_center(R, t):
    """X_camera = R X_world + t  =>  C = -R^T t"""
    return -R.T @ t


def project(K, R, t, X):
    """投影到像素坐标。z<=0 时仍按公式计算（由调用方判定是否在相机后方）。"""
    Xc = R @ X + t
    if abs(Xc[2]) < 1e-12:
        raise TriangulationRejected("DEGENERATE", "点位于相机主平面上，无法投影")
    uv = K @ Xc
    return uv[:2] / uv[2]


def reprojection_jacobian(K, R, t, X):
    """像素 (u,v) 对世界点 X 的解析雅可比（2x3）。

    u = (K00 x + K01 y + K02 z)/z,  v = (K11 y + K12 z)/z,  (x,y,z)^T = R X + t
    """
    Xc = R @ X + t
    x, y, z = Xc
    if abs(z) < 1e-12:
        raise TriangulationRejected("DEGENERATE", "点位于相机主平面上，雅可比无定义")
    u = (K[0, 0] * x + K[0, 1] * y + K[0, 2] * z) / z
    v = (K[1, 1] * y + K[1, 2] * z) / z
    d_uv_d_Xc = np.array(
        [
            [K[0, 0] / z, K[0, 1] / z, (K[0, 2] - u) / z],
            [0.0, K[1, 1] / z, (K[1, 2] - v) / z],
        ]
    )
    return d_uv_d_Xc @ R


# ---------------------------------------------------------------- 归一化 DLT


def normalizing_transform(pts):
    """Hartley 归一化相似变换：质心移到原点、平均距离缩放到 sqrt(2)。

    单点（或点重合）时退化为纯平移，缩放取 1。
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    centroid = pts.mean(axis=0)
    if len(pts) > 1:
        mean_dist = np.hypot(*(pts - centroid).T).mean()
    else:
        mean_dist = 0.0
    s = np.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0
    return np.array(
        [
            [s, 0.0, -s * centroid[0]],
            [0.0, s, -s * centroid[1]],
            [0.0, 0.0, 1.0],
        ]
    )


def dlt_triangulate(P1n, P2n, x1n, x2n):
    """归一化坐标下的 4x4 DLT。返回世界系齐次点（归一化只作用于方程，

    解仍在原世界坐标系中）。秩不足时拒绝。"""
    A = np.array(
        [
            x1n[0] * P1n[2] - x1n[2] * P1n[0],
            x1n[1] * P1n[2] - x1n[2] * P1n[1],
            x2n[0] * P2n[2] - x2n[2] * P2n[0],
            x2n[1] * P2n[2] - x2n[2] * P2n[1],
        ]
    )
    _, S, Vt = np.linalg.svd(A)
    if S[0] <= 0 or S[2] <= S[0] * 1e-12:
        raise TriangulationRejected(
            "RANK_DEFICIENT",
            f"DLT 矩阵秩不足（奇异值比 sigma3/sigma1 = "
            f"{S[2] / S[0] if S[0] > 0 else 0.0:.2e}），对应点约束退化",
        )
    Xh = Vt[3]
    if abs(Xh[3]) < 1e-12:
        raise TriangulationRejected("RANK_DEFICIENT", "DLT 解为无穷远点，三角化退化")
    return Xh[:3] / Xh[3]


# ---------------------------------------------------------------- 射线夹角


def ray_angle_deg(K1, R1, t1, K2, R2, t2, m1, m2):
    """两条反投影射线在世界系中的夹角（度）。"""
    C1 = camera_center(R1, t1)
    C2 = camera_center(R2, t2)
    d1 = R1.T @ (np.linalg.inv(K1) @ np.array([m1[0], m1[1], 1.0]))
    d2 = R2.T @ (np.linalg.inv(K2) @ np.array([m2[0], m2[1], 1.0]))
    d1 /= np.linalg.norm(d1)
    d2 /= np.linalg.norm(d2)
    cosang = float(np.clip(d1 @ d2, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


# ---------------------------------------------------------------- 阻尼最小二乘


def refine_least_squares(K1, R1, t1, K2, R2, t2, m1, m2, X0, max_iter=DEFAULT_MAX_ITER):
    """对 sum_i ||proj_i(X) - m_i||^2 做阻尼高斯-牛顿最小化。

    解析雅可比 + Levenberg 阻尼 + 下降检查；有限次迭代。
    返回 (X, 迭代次数, 最终残差向量 r, 最终雅可比 J)。
    """
    cams = ((K1, R1, t1), (K2, R2, t2))
    meas = (np.asarray(m1, float), np.asarray(m2, float))

    def residual_and_jac(X):
        r = np.empty(4)
        J = np.empty((4, 3))
        for k, (K, R, t) in enumerate(cams):
            p = project(K, R, t, X)
            r[2 * k : 2 * k + 2] = p - meas[k]
            J[2 * k : 2 * k + 2] = reprojection_jacobian(K, R, t, X)
        return r, J

    X = np.asarray(X0, dtype=float).copy()
    r, J = residual_and_jac(X)
    cost = float(r @ r)
    lam = 1e-3
    converged = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        g = J.T @ r
        H = J.T @ J
        if np.linalg.norm(g, ord=np.inf) < 1e-10:
            converged = True
            break
        diag = np.diag(H).copy()
        diag[diag <= 0] = 1e-12

        step_accepted = False
        for _ in range(30):  # 阻尼内循环：只做有限次尝试
            A = H + lam * np.diag(diag)
            try:
                delta = np.linalg.solve(A, -g)
            except np.linalg.LinAlgError:
                raise TriangulationRejected(
                    "RANK_DEFICIENT", "阻尼正规方程奇异，无法求解步长"
                )
            X_new = X + delta
            try:
                r_new, J_new = residual_and_jac(X_new)
                cost_new = float(r_new @ r_new)
            except TriangulationRejected:
                cost_new = np.inf  # 主平面附近视为不可接受步
            if cost_new < cost:  # 下降检查
                X, r, J, cost = X_new, r_new, J_new, cost_new
                lam = max(lam / 10.0, 1e-15)
                step_accepted = True
                break
            lam *= 10.0
            if lam > 1e13:
                break

        if not step_accepted:
            # 阻尼已极大仍找不到下降步：若梯度已小则视为收敛，否则未收敛
            if np.linalg.norm(J.T @ r, ord=np.inf) < 1e-8:
                converged = True
            break
        if np.linalg.norm(delta) < 1e-12 * (1.0 + np.linalg.norm(X)):
            converged = True
            break
        if abs(float(g @ delta)) < 1e-14:
            converged = True
            break

    if not converged:
        raise TriangulationRejected(
            "NOT_CONVERGED",
            f"最小二乘在 {max_iter} 次有限迭代内未收敛（当前残差 "
            f"{np.sqrt(cost):.3f} px）",
        )
    return X, n_iter, r, J


# ---------------------------------------------------------------- 单对三角化


def triangulate_pair(
    K1,
    R1,
    t1,
    K2,
    R2,
    t2,
    m1,
    m2,
    max_px_err,
    sigma,
    P1n=None,
    P2n=None,
    T1=None,
    T2=None,
    max_iter=DEFAULT_MAX_ITER,
):
    """三角化一对对应点。成功返回 dict，失败抛 TriangulationRejected。"""
    m1 = _as_array("m1", m1, (2,), code="BAD_INPUT")
    m2 = _as_array("m2", m2, (2,), code="BAD_INPUT")

    # 1) 射线夹角：距 0 度或 180 度小于阈值即病态
    angle = ray_angle_deg(K1, R1, t1, K2, R2, t2, m1, m2)
    if angle < RAY_ANGLE_MIN_DEG or angle > 180.0 - RAY_ANGLE_MIN_DEG:
        raise TriangulationRejected(
            "RAY_ANGLE",
            f"两射线夹角 {angle:.3f} 度，距 0/180 度小于 "
            f"{RAY_ANGLE_MIN_DEG} 度，三角化病态",
        )

    # 2) 归一化 DLT 初值
    if P1n is None or P2n is None or T1 is None or T2 is None:
        T1 = normalizing_transform([m1])
        T2 = normalizing_transform([m2])
        P1 = K1 @ np.hstack([R1, t1.reshape(3, 1)])
        P2 = K2 @ np.hstack([R2, t2.reshape(3, 1)])
        P1n, P2n = T1 @ P1, T2 @ P2
    x1n = T1 @ np.array([m1[0], m1[1], 1.0])
    x2n = T2 @ np.array([m2[0], m2[1], 1.0])
    X0 = dlt_triangulate(P1n, P2n, x1n, x2n)

    # 3) 阻尼最小二乘精化（解析雅可比 + 下降检查，有限次）
    X, n_iter, r, J = refine_least_squares(
        K1, R1, t1, K2, R2, t2, m1, m2, X0, max_iter=max_iter
    )

    # 4) 点必须在两个相机前方
    z1 = float((R1 @ X + t1)[2])
    z2 = float((R2 @ X + t2)[2])
    behind = []
    if z1 <= 0:
        behind.append(f"相机1（深度 {z1:.4f}）")
    if z2 <= 0:
        behind.append(f"相机2（深度 {z2:.4f}）")
    if behind:
        raise TriangulationRejected(
            "BEHIND_CAMERA", "三角化结果位于相机后方：" + "、".join(behind)
        )

    # 5) 重投影与误差（与协方差同一次求解的 X、J）
    reproj1 = project(K1, R1, t1, X)
    reproj2 = project(K2, R2, t2, X)
    err1 = float(np.linalg.norm(reproj1 - m1))
    err2 = float(np.linalg.norm(reproj2 - m2))
    if err1 > max_px_err or err2 > max_px_err:
        raise TriangulationRejected(
            "PIXEL_ERROR",
            f"重投影误差超限：视图1 {err1:.3f} px、视图2 {err2:.3f} px "
            f"> 上限 {max_px_err:.3f} px",
        )

    # 6) 局部协方差 sigma^2 (J^T J)^{-1}（J 为本次求解最终雅可比）
    H = J.T @ J
    if np.linalg.cond(H) > 1e12:
        raise TriangulationRejected(
            "RANK_DEFICIENT", "J^T J 病态（秩不足），局部协方差不可信"
        )
    cov = sigma**2 * np.linalg.solve(H, np.eye(3))

    return {
        "status": "ok",
        "X": X.tolist(),
        "reproj1": reproj1.tolist(),
        "reproj2": reproj2.tolist(),
        "err1": err1,
        "err2": err2,
        "cov": cov.tolist(),
        "ray_angle_deg": angle,
        "iterations": int(n_iter),
    }


# ---------------------------------------------------------------- 批量入口


def triangulate_all(cam1, cam2, pairs, max_px_err, sigma, max_iter=DEFAULT_MAX_ITER):
    """校验相机与参数后逐对三角化。相机/参数非法时抛 TriangulationRejected。"""
    K1, R1, t1 = validate_camera(cam1["K"], cam1["R"], cam1["t"], "1")
    K2, R2, t2 = validate_camera(cam2["K"], cam2["R"], cam2["t"], "2")

    if not np.isfinite(max_px_err) or max_px_err <= 0:
        raise TriangulationRejected("BAD_PARAM", "像素误差上限必须为正数")
    if not np.isfinite(sigma) or sigma <= 0:
        raise TriangulationRejected("BAD_PARAM", "噪声 sigma 必须为正数")
    if not isinstance(pairs, (list, tuple)) or not (1 <= len(pairs) <= MAX_PAIRS):
        raise TriangulationRejected(
            "BAD_PARAM",
            f"对应点对数须在 1..{MAX_PAIRS} 之间，实际为 "
            f"{len(pairs) if isinstance(pairs, (list, tuple)) else '非法'}",
        )

    # 归一化 DLT 的相似变换用各视图全部对应点估计（Hartley 归一化）
    pts1, pts2 = [], []
    for p in pairs:
        if not isinstance(p, dict) or "m1" not in p or "m2" not in p:
            raise TriangulationRejected("BAD_PARAM", "每对对应点必须是含 m1、m2 的对象")
        pts1.append(_as_array("m1", p["m1"], (2,), code="BAD_PARAM"))
        pts2.append(_as_array("m2", p["m2"], (2,), code="BAD_PARAM"))
    T1 = normalizing_transform(np.array(pts1))
    T2 = normalizing_transform(np.array(pts2))
    P1 = K1 @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K2 @ np.hstack([R2, t2.reshape(3, 1)])
    P1n, P2n = T1 @ P1, T2 @ P2

    results = []
    for i, p in enumerate(pairs):
        try:
            res = triangulate_pair(
                K1,
                R1,
                t1,
                K2,
                R2,
                t2,
                p["m1"],
                p["m2"],
                max_px_err,
                sigma,
                P1n=P1n,
                P2n=P2n,
                T1=T1,
                T2=T2,
                max_iter=max_iter,
            )
            res["index"] = i
            results.append(res)
        except TriangulationRejected as e:
            results.append(
                {
                    "index": i,
                    "status": "rejected",
                    "code": e.code,
                    "reason": e.message,
                }
            )
    return results
