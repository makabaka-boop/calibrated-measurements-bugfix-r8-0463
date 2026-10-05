"""生成演示数据：两张无畸变合成视图 + cameras.json（K/R/t 与真值三维点）。

约定 X_camera = R · X_world + t。相机1位于世界原点、朝向 +z；
相机2位于 (1.5, 0.2, 0.5)，朝向场景中心。
"""

import json
import os

import numpy as np
from PIL import Image, ImageDraw

W, H = 960, 720
HERE = os.path.dirname(os.path.abspath(__file__))

K1 = [[900.0, 0.0, 480.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]]
R1 = np.eye(3)
t1 = np.zeros(3)

# 真值三维点（世界系，位于两相机前方）
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


def look_at(C, target, up_hint=(0.0, 1.0, 0.0)):
    """返回朝向 target 的旋转 R（行向量为相机 x/y/z 轴在世界系中的方向）。"""
    z = np.asarray(target, float) - np.asarray(C, float)
    z /= np.linalg.norm(z)
    x = np.cross(up_hint, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.array([x, y, z])


C2 = np.array([1.5, 0.2, 0.5])
R2 = look_at(C2, target=[0.0, 0.0, 4.5])
t2 = -R2 @ C2
K2 = [[920.0, 0.0, 480.0], [0.0, 910.0, 360.0], [0.0, 0.0, 1.0]]


def project(K, R, t, X):
    Xc = np.asarray(R) @ np.asarray(X) + np.asarray(t)
    uv = np.asarray(K) @ Xc
    return uv[:2] / uv[2]


def render(K, R, t, path, seed):
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (W, H), (245, 246, 248))
    dr = ImageDraw.Draw(img)
    for gx in range(0, W, 48):  # 背景网格，便于肉眼检查叠画
        dr.line([(gx, 0), (gx, H)], fill=(225, 228, 232))
    for gy in range(0, H, 48):
        dr.line([(0, gy), (W, gy)], fill=(225, 228, 232))
    for _ in range(150):  # 随机噪点纹理
        x, y = rng.integers(0, W), rng.integers(0, H)
        dr.point((x, y), fill=(180, 185, 190))
    for i, X in enumerate(POINTS3D):
        u, v = project(K, R, t, X)
        dr.ellipse(
            [u - 7, v - 7, u + 7, v + 7],
            fill=(200, 30, 30),
            outline=(255, 255, 255),
            width=2,
        )
        dr.text((u + 9, v - 12), str(i), fill=(20, 20, 20))
    img.save(path)


def main():
    render(K1, R1, t1, os.path.join(HERE, "view1.png"), seed=1)
    render(K2, R2, t2, os.path.join(HERE, "view2.png"), seed=2)
    cams = {
        "image_size": [W, H],
        "K1": K1,
        "R1": R1.tolist(),
        "t1": t1.tolist(),
        "K2": K2,
        "R2": R2.tolist(),
        "t2": t2.tolist(),
        "points3d": POINTS3D,
        "note": "X_camera = R * X_world + t；图片已去畸变",
    }
    with open(os.path.join(HERE, "cameras.json"), "w", encoding="utf-8") as f:
        json.dump(cams, f, ensure_ascii=False, indent=2)
    print("demo 数据已生成：view1.png / view2.png / cameras.json")


if __name__ == "__main__":
    main()
