"""标定量测页面的浏览器流程测试：

成功流程：输入合法量测请求 -> 断言页面实际发出的请求体就是输入原文 ->
断言页面证据（三角化表/量测表）对应原始编号与同次重建结果 ->
断言下载内容与后端响应原文逐字节一致。
失败流程：引用被拒绝的对应点 -> 整次量测失败、页面展示原始拒绝原因、
证据表仍给出同次三角化结果、不提供下载。
"""

import json
import os
import urllib.request

import numpy as np
import pytest

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

CHROME_LIB_DIRS = "/tmp/chromelibs/root/usr/lib/aarch64-linux-gnu:/tmp/chromelibs/root/lib/aarch64-linux-gnu"


def project(K, R, t, X):
    Xc = np.asarray(R) @ np.asarray(X) + np.asarray(t)
    uv = np.asarray(K) @ Xc
    return uv[:2] / uv[2]


def build_request(cams, pairs):
    return {
        "K1": cams["K1"],
        "R1": cams["R1"],
        "t1": cams["t1"],
        "K2": cams["K2"],
        "R2": cams["R2"],
        "t2": cams["t2"],
        "pairs": pairs,
        "max_px_err": 2.0,
        "sigma": 0.5,
        "reference": [0, 1],
        "known_length": 2.5,
        "segments": [[1, 2], [2, 3], [0, 1]],  # 末段为参考段测自己
    }


def exact_pairs(cams, count=4):
    return [
        {
            "m1": project(cams["K1"], cams["R1"], cams["t1"], X).tolist(),
            "m2": project(cams["K2"], cams["R2"], cams["t2"], X).tolist(),
        }
        for X in cams["points3d"][:count]
    ]


@pytest.fixture()
def browser_page(server):
    if os.path.isdir("/tmp/chromelibs/root"):
        os.environ["LD_LIBRARY_PATH"] = (
            CHROME_LIB_DIRS + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        )
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(server + "/measurements")
        yield page, server
        browser.close()


def test_measurements_page_success_and_export(browser_page):
    page, server = browser_page
    cams = json.load(urllib.request.urlopen(server + "/demo/cameras.json"))
    req_obj = build_request(cams, exact_pairs(cams))
    body = json.dumps(req_obj, indent=2)
    page.fill("#input", body)

    captured = []
    page.on("request", lambda r: captured.append(r) if "api/measurements" in r.url else None)
    page.click("#run")
    page.wait_for_function("window.measurement && window.measurement.lastResponse !== null")

    # 1) 实际请求：POST /api/measurements，请求体逐字节等于输入原文
    assert len(captured) == 1
    sent = captured[0]
    assert sent.method == "POST"
    assert sent.post_data == body
    assert page.evaluate("window.measurement.lastRequest") == body

    # 2) 量测结果与同次重建一致（无噪声投影 -> 还原真值比例）
    resp = page.evaluate("window.measurement.lastResponse")
    assert resp["ok"] is True
    assert [r["index"] for r in resp["results"]] == [0, 1, 2, 3]
    truth = [np.asarray(p) for p in cams["points3d"]]
    scale = req_obj["known_length"] / np.linalg.norm(truth[1] - truth[0])
    segs = resp["measurement"]["segments"]
    assert segs[0]["length"] == pytest.approx(
        scale * np.linalg.norm(truth[2] - truth[1]), rel=1e-6
    )
    assert segs[1]["length"] == pytest.approx(
        scale * np.linalg.norm(truth[3] - truth[2]), rel=1e-6
    )
    # 参考段测自己：精确已知长度 + 零方差
    assert segs[2]["points"] == [0, 1]
    assert segs[2]["length"] == req_obj["known_length"]
    assert segs[2]["variance"] == 0.0

    # 3) 页面证据：状态、三角化表（原始编号）、量测表
    assert page.locator("#status").inner_text() == "量测成功"
    tri_rows = page.locator("#triangulationTable tbody tr")
    assert tri_rows.count() == 4
    for i in range(4):
        assert tri_rows.nth(i).locator("td").nth(0).inner_text() == str(i)
        assert tri_rows.nth(i).locator("td").nth(1).inner_text() == "ok"
    seg_rows = page.locator("#segmentTable tbody tr")
    assert seg_rows.count() == 3
    assert seg_rows.nth(2).locator("td").nth(0).inner_text() == "[0, 1]"
    assert "参考段（原始编号）=[0, 1]" in page.locator("#measurementSummary").inner_text()

    # 4) 导出：下载内容 = 后端响应原文 = 页面 <pre> 证据
    raw = page.evaluate("window.measurement.rawResponse")
    assert page.locator("#out").inner_text() == raw
    assert json.loads(raw)["measurement"]["segments"][2]["length"] == 2.5
    assert page.locator("#download").is_visible()
    downloaded = page.evaluate(
        """async () => {
            const a = document.getElementById("download");
            const r = await fetch(a.href);
            return await r.text();
        }"""
    )
    assert downloaded == raw
    assert page.evaluate("document.getElementById('download').download") == "measurement.json"


def test_measurements_page_failure_keeps_evidence(browser_page):
    page, server = browser_page
    cams = json.load(urllib.request.urlopen(server + "/demo/cameras.json"))
    pairs = exact_pairs(cams)
    pairs[0]["m1"][0] += 60.0  # 第 0 对将被 PIXEL_ERROR 拒绝
    req_obj = build_request(cams, pairs)
    req_obj["max_px_err"] = 1.0
    req_obj["segments"] = [[0, 2]]  # 引用被拒绝的点 -> 整次失败
    page.fill("#input", json.dumps(req_obj))

    page.click("#run")
    page.wait_for_function("window.measurement && window.measurement.lastResponse !== null")

    resp = page.evaluate("window.measurement.lastResponse")
    assert resp["ok"] is False
    assert "rejected point 0" in resp["error"]
    assert "PIXEL_ERROR" in resp["error"]
    assert "measurement" not in resp  # 整次失败，无部分量测
    # 同次三角化证据保留：原始编号 + 原始拒绝原因
    assert resp["results"][0]["status"] == "rejected"
    assert resp["results"][0]["code"] == "PIXEL_ERROR"
    assert "重投影误差超限" in resp["results"][0]["reason"]
    assert all(r["status"] == "ok" for r in resp["results"][1:])

    # 页面展示失败与原始拒绝原因，证据表仍对应原始编号，不提供下载
    assert "量测失败" in page.locator("#status").inner_text()
    tri_rows = page.locator("#triangulationTable tbody tr")
    assert tri_rows.count() == 4
    row0 = tri_rows.nth(0)
    assert row0.locator("td").nth(0).inner_text() == "0"
    assert row0.locator("td").nth(1).inner_text() == "rejected"
    assert "PIXEL_ERROR" in row0.locator("td").nth(6).inner_text()
    assert "重投影误差超限" in row0.locator("td").nth(6).inner_text()
    assert tri_rows.nth(1).locator("td").nth(1).inner_text() == "ok"
    assert page.locator("#segmentTable").count() == 0
    assert page.locator("#download").is_hidden()
