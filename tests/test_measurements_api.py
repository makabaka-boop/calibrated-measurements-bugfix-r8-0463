"""/api/measurements 接口测试：实际请求体、返回结构、失败证据与编号。"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app  # noqa
from triangulation import project  # noqa

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
TRUTH = [np.asarray(X, float) for X in POINTS]


@pytest.fixture
def client():
    app.testing = True
    return app.test_client()


@pytest.fixture
def body():
    pairs = [
        {
            "m1": project(np.asarray(K1), R1, t1, X).tolist(),
            "m2": project(np.asarray(K2), R2, t2, X).tolist(),
        }
        for X in TRUTH
    ]
    L01 = float(np.linalg.norm(TRUTH[0] - TRUTH[1]))
    return {
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
        "known_length": L01,
        "segments": [[0, 1], [1, 2], [2, 3]],
    }


def test_api_success_structure(client, body):
    resp = client.post("/api/measurements", json=body)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert len(data["results"]) == 4
    assert all(r["status"] == "ok" for r in data["results"])
    m = data["measurement"]
    assert m["reference"] == [0, 1]
    assert [s["points"] for s in m["segments"]] == [[0, 1], [1, 2], [2, 3]]
    self_seg = m["segments"][0]
    assert self_seg["length"] == body["known_length"]
    assert self_seg["variance"] == 0.0
    assert len(m["covariance"]["matrix"]) == 3


def test_api_rejected_reference_returns_evidence(client, body):
    body["pairs"][1]["m1"][0] += 90.0  # 原始点 1 将被 PIXEL_ERROR 拒绝
    body["reference"] = [1, 2]
    resp = client.post("/api/measurements", json=body)
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["ok"] is False
    assert data["code"] == "REJECTED_POINT"
    assert "1" in data["error"]
    # 同次三角化证据随失败返回，原始拒绝原因完整保留
    results = data["results"]
    assert results is not None and len(results) == 4
    rejected = [r for r in results if r["status"] == "rejected"]
    assert len(rejected) == 1
    assert rejected[0]["index"] == 1
    assert rejected[0]["code"] == "PIXEL_ERROR"
    assert rejected[0]["reason"]


def test_api_rejected_segment_same_failure(client, body):
    body["pairs"][2]["m1"][0] += 90.0
    resp = client.post("/api/measurements", json=body)
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["code"] == "REJECTED_POINT"
    assert data["results"][2]["code"] == "PIXEL_ERROR"
    assert "measurement" not in data  # 整次失败，不产生部分结果


def test_api_zero_length_segment(client, body):
    body["segments"] = [[1, 1]]
    resp = client.post("/api/measurements", json=body)
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["code"] == "ZERO_LENGTH"
    # 三角化本身成功：证据仍随失败返回
    assert all(r["status"] == "ok" for r in data["results"])


def test_api_bad_known_length(client, body):
    body["known_length"] = 0
    resp = client.post("/api/measurements", json=body)
    assert resp.status_code == 400
    assert resp.get_json()["code"] == "INVALID_REQUEST"


def test_api_index_out_of_range_after_rejection(client, body):
    # 点 0 被拒绝；引用编号 3（旧实现会错位成第 3 个通过点或 500）
    body["pairs"][0]["m1"][0] += 90.0
    body["reference"] = [1, 3]
    body["segments"] = [[2, 3]]
    resp = client.post("/api/measurements", json=body)
    assert resp.status_code == 200
    data = resp.get_json()
    statuses = {r["index"]: r["status"] for r in data["results"]}
    assert statuses == {0: "rejected", 1: "ok", 2: "ok", 3: "ok"}
    seg = data["measurement"]["segments"][0]
    assert seg["points"] == [2, 3]
    truth_23 = float(np.linalg.norm(TRUTH[2] - TRUTH[3]))
    # 点 0 的坏像素同时改变 Hartley 归一化，尺度有轻微偏差属正常
    assert seg["length"] == pytest.approx(truth_23, rel=0.02)


def test_api_bad_camera_still_has_code(client, body):
    body["K1"] = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    resp = client.post("/api/measurements", json=body)
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["ok"] is False and data["code"] == "BAD_CAMERA"


def test_api_non_json_body(client):
    resp = client.post(
        "/api/measurements",
        data="not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False
