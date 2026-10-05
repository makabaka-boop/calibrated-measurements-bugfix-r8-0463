"""浏览器流程测试（一次完整流程）：

加载页面 -> 缩放视图1、平移视图2 -> 在两张图上点击对应标记点 ->
断言记录的是【原图像素坐标】而非 CSS 坐标 -> 三角化 ->
断言后端返回的原图坐标（重投影/空间点）正确且叠画位置正确 ->
编辑对应点后旧结果作废。
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


def test_browser_flow(server):
    if os.path.isdir("/tmp/chromelibs/root"):
        os.environ["LD_LIBRARY_PATH"] = (
            CHROME_LIB_DIRS + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        )

    cams = json.load(urllib.request.urlopen(server + "/demo/cameras.json"))
    expected = [  # 每个真值三维点在两视图中的期望原图像素坐标
        (
            project(cams["K1"], cams["R1"], cams["t1"], X),
            project(cams["K2"], cams["R2"], cams["t2"], X),
        )
        for X in cams["points3d"]
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 1100})
        page.goto(server)
        page.wait_for_function("window.app && window.app.ready === true")

        box1 = page.locator("#canvas1").bounding_box()
        box2 = page.locator("#canvas2").bounding_box()

        # 缩放视图1（滚轮）、平移视图2（拖动）：制造非平凡视图变换
        page.mouse.move(box1["x"] + 320, box1["y"] + 240)
        page.mouse.wheel(0, -200)
        page.mouse.move(box2["x"] + 300, box2["y"] + 220)
        page.mouse.down()
        page.mouse.move(box2["x"] + 340, box2["y"] + 245, steps=5)
        page.mouse.up()
        scale1 = page.evaluate("window.app.views[0].scale")
        assert abs(scale1 - 2 / 3) > 0.05  # 确认视图变换确实被改变

        def to_screen(v, m):
            return page.evaluate(
                "([v,x,y]) => window.app.imgToScreen(v,x,y)",
                [v, float(m[0]), float(m[1])],
            )

        def click_at_image_point(v, m, button="left"):
            # 画布内容盒原点（视口坐标）+ 图像点的屏幕坐标 -> 点击
            origin = page.evaluate(
                """(v) => { const c = window.app.views[v].canvas;
                            const r = c.getBoundingClientRect();
                            return [r.left + c.clientLeft, r.top + c.clientTop]; }""",
                v,
            )
            s = to_screen(v, m)
            page.mouse.click(origin[0] + s[0], origin[1] + s[1], button=button)

        def inside(s):
            return 20 < s[0] < 620 and 20 < s[1] < 460

        chosen = []
        for i, (m1, m2) in enumerate(expected):
            if inside(to_screen(0, m1)) and inside(to_screen(1, m2)):
                chosen.append(i)
            if len(chosen) == 3:
                break
        assert len(chosen) == 3, "可视区域内标记点不足"

        for i in chosen:
            click_at_image_point(0, expected[i][0])
            click_at_image_point(1, expected[i][1])

        # 关键断言：记录的是原图像素坐标（经逆变换还原），不是 CSS 坐标。
        # 浏览器把鼠标坐标截断到整数 CSS 像素，量化误差 <= 1 CSS px / scale。
        rec1 = page.evaluate("window.app.views[0].points.map(p => [p.x, p.y])")
        rec2 = page.evaluate("window.app.views[1].points.map(p => [p.x, p.y])")
        for k, i in enumerate(chosen):
            assert abs(rec1[k][0] - expected[i][0][0]) < 1.6
            assert abs(rec1[k][1] - expected[i][0][1]) < 1.6
            assert abs(rec2[k][0] - expected[i][1][0]) < 1.6
            assert abs(rec2[k][1] - expected[i][1][1]) < 1.6

        page.fill("#maxPxErr", "5.0")  # 容纳浏览器点击量化噪声
        page.click("#btnTriangulate")
        page.wait_for_function("window.app.getResults() !== null")
        results = page.evaluate("window.app.getResults()")
        assert len(results) == 3
        for k, i in enumerate(chosen):
            r = results[k]
            assert r["status"] == "ok", r.get("reason")
            Xt = cams["points3d"][i]
            assert all(abs(r["X"][j] - Xt[j]) < 0.1 for j in range(3))
            assert abs(r["reproj1"][0] - expected[i][0][0]) < 3.0
            assert abs(r["reproj1"][1] - expected[i][0][1]) < 3.0
            assert abs(r["reproj2"][0] - expected[i][1][0]) < 3.0
            assert abs(r["reproj2"][1] - expected[i][1][1]) < 3.0

        # 后端返回的原图坐标必须经同一视图变换叠画到正确屏幕位置
        max_dev = page.evaluate(
            """() => {
            const app = window.app;
            const res = app.getResults();
            let d = 0;
            for (const o of app.getDrawnOverlays()) {
                const r = res[o.index];
                const rp = o.view === 0 ? r.reproj1 : r.reproj2;
                const [ex, ey] = app.imgToScreen(o.view, rp[0], rp[1]);
                d = Math.max(d, Math.hypot(ex - o.sx, ey - o.sy));
            }
            return d;
        }"""
        )
        assert max_dev < 1e-6
        assert page.locator("#resultTable tbody tr").count() == 3
        page.screenshot(path="/tmp/browser_flow.png")

        # 编辑对应点（右键删除）-> 旧结果作废
        p0 = page.evaluate("window.app.views[0].points[0]")
        click_at_image_point(0, [p0["x"], p0["y"]], button="right")
        assert page.evaluate("window.app.getResults()") is None
        assert page.evaluate("window.app.views[0].points.length") == 2

        browser.close()
