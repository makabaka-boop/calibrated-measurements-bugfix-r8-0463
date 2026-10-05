"""标定量测页浏览器测试：

成功流程：
  预填请求 -> 量测 -> 页面表格证据、状态、下载文件三者编号与数值一致，
  参考段自量为精确已知长度、零方差；下载 = 实际请求 + 同次返回。

失败流程：
  让某个原始对应点被三角化拒绝 -> 整次量测明确失败（状态/下载均标记），
  三角化证据表保留该点的原始编号与拒绝 code/reason；
  未被拒绝的后续编号仍指向正确的空间点（编号不重排）。
"""

import json
import os

import pytest

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

CHROME_LIB_DIRS = "/tmp/chromelibs/root/usr/lib/aarch64-linux-gnu:/tmp/chromelibs/root/lib/aarch64-linux-gnu"


@pytest.fixture
def page_ctx(server):
    if os.path.isdir("/tmp/chromelibs/root"):
        os.environ["LD_LIBRARY_PATH"] = (
            CHROME_LIB_DIRS + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        )
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page()
        page.goto(server + "/measurements")
        page.wait_for_function(
            "window.measureApp && window.measureApp.ready === true"
        )
        yield page, server
        browser.close()


def blob_content(page, anchor="#mDownload"):
    """真正读回下载链接（blob URL）里的文件内容，验证导出本身。"""
    href = page.get_attribute(anchor, "href")
    assert href.startswith("blob:")
    return page.evaluate(
        """async (href) => {
            const buf = await (await fetch(href)).arrayBuffer();
            return new TextDecoder().decode(buf);
        }""",
        href,
    )


def test_measurement_success_page_and_download(page_ctx):
    page, server = page_ctx
    request = json.loads(page.input_value("#input"))
    assert request["reference"] == [0, 1]
    assert [0, 1] in request["segments"]  # 演示请求包含“参考段量自己”

    page.click("#run")
    page.wait_for_selector("#mStatus.ok")
    status = page.text_content("#mStatus")
    assert "量测成功" in status
    assert "方差 0" in status

    evidence = page.evaluate("() => window.measureApp.getLastEvidence()")
    # 下载内容 = 实际请求 + 同次返回
    assert evidence["request"] == request
    resp = evidence["response"]
    assert resp["ok"] is True
    m = resp["measurement"]
    assert all(r["status"] == "ok" for r in resp["results"])

    # 下载文件内容逐字节等价于页面内存中的证据对象
    downloaded = json.loads(blob_content(page))
    assert downloaded["request"] == request
    assert downloaded["response"] == resp
    assert "exported_at" in downloaded

    # 页面线段表的编号与数值和返回 JSON 完全对应
    rows = page.locator("#mEvidence table").nth(1).locator("tbody tr")
    assert rows.count() == len(m["segments"])
    for k in range(rows.count()):
        cells = rows.nth(k).locator("td")
        label = (
            cells.nth(0)
            .text_content()
            .replace(" ", "")
            .replace("–", "-")
            .strip()
        )
        i, j = m["segments"][k]["points"]
        assert label == f"{i}-{j}"
        assert float(cells.nth(1).text_content()) == pytest.approx(
            m["segments"][k]["length"], abs=1e-5
        )
        assert float(cells.nth(3).text_content()) == pytest.approx(
            m["segments"][k]["variance"], rel=1e-3
        )

    # 参考段自量：页面（6 位小数显示）与下载（精确）都是已知长度、零方差
    self_cells = rows.first.locator("td")
    assert float(self_cells.nth(1).text_content()) == pytest.approx(
        request["known_length"], abs=1e-6
    )
    assert float(self_cells.nth(3).text_content()) == 0.0
    self_seg = m["segments"][0]
    assert self_seg["points"] == [0, 1]
    assert self_seg["length"] == request["known_length"]
    assert self_seg["variance"] == 0.0

    # 三角化证据表按原始编号 0..3 排列
    tri_rows = page.locator("#mEvidence table").first.locator("tbody tr")
    assert tri_rows.count() == 4
    for i in range(4):
        assert (
            tri_rows.nth(i).locator("td").first.text_content().strip() == str(i)
        )

    # 原始返回 JSON 区域与下载/内存一致
    assert json.loads(page.text_content("#mRaw")) == resp


def test_measurement_failure_keeps_evidence_and_indices(page_ctx):
    page, server = page_ctx
    good_request = json.loads(page.input_value("#input"))

    # 破坏原始点 2，但线段先不引用它：编号不重排，量测成功
    request = json.loads(json.dumps(good_request))
    request["pairs"][2]["m1"][0] += 90.0
    request["segments"] = [[0, 1], [1, 3]]
    page.fill("#input", json.dumps(request))
    page.click("#run")
    page.wait_for_selector("#mStatus.ok")
    evidence = page.evaluate("() => window.measureApp.getLastEvidence()")
    statuses = {
        r["index"]: r["status"] for r in evidence["response"]["results"]
    }
    assert statuses == {0: "ok", 1: "ok", 2: "rejected", 3: "ok"}
    seg13 = evidence["response"]["measurement"]["segments"][1]
    assert seg13["points"] == [1, 3]  # 3 没有被错位成“第 3 个通过点”

    # reference 引用坏点 2 -> 整次量测明确失败
    request["reference"] = [2, 3]
    page.fill("#input", json.dumps(request))
    page.click("#run")
    page.wait_for_selector("#mStatus.bad")
    status = page.text_content("#mStatus")
    assert "整次量测失败" in status and "REJECTED_POINT" in status

    evidence = page.evaluate("() => window.measureApp.getLastEvidence()")
    resp = evidence["response"]
    assert resp["ok"] is False
    assert resp["code"] == "REJECTED_POINT"
    assert evidence["request"] == request  # 失败证据也对应实际请求
    rej = [r for r in resp["results"] if r["status"] == "rejected"]
    assert len(rej) == 1 and rej[0]["index"] == 2
    assert rej[0]["code"] == "PIXEL_ERROR" and rej[0]["reason"]
    assert "measurement" not in resp  # 不产生部分量测结果

    # 下载文件同样是失败证据（请求 + 带拒绝原因的同次结果）
    downloaded = json.loads(blob_content(page))
    assert downloaded["request"] == request
    assert downloaded["response"] == resp

    # 页面三角化证据表：原始编号 2 一行显示拒绝 code 与原因
    tri_rows = page.locator("#mEvidence table").first.locator("tbody tr")
    assert tri_rows.count() == 4
    bad_cells = tri_rows.nth(2).locator("td")
    assert bad_cells.nth(0).text_content().strip() == "2"
    assert "PIXEL_ERROR" in bad_cells.nth(1).text_content()
    assert "重投影误差" in bad_cells.nth(5).text_content()
    assert page.is_visible("#mDownload")

    # 重新发起一次成功请求：证据被替换为新的同次结果，不复用失败那轮
    page.fill("#input", json.dumps(good_request))
    page.click("#run")
    page.wait_for_selector("#mStatus.ok")
    fresh = page.evaluate("() => window.measureApp.getLastEvidence()")
    assert fresh["request"] == good_request
    assert fresh["response"]["ok"] is True
    assert all(r["status"] == "ok" for r in fresh["response"]["results"])
    assert json.loads(blob_content(page))["response"] == fresh["response"]


def test_endpoint_order_swap_gives_identical_displayed_result(page_ctx):
    page, server = page_ctx
    page.click("#run")
    page.wait_for_selector("#mStatus.ok")
    first = page.evaluate("() => window.measureApp.getLastEvidence()")["response"]

    req = json.loads(page.input_value("#input"))
    req["reference"] = [req["reference"][1], req["reference"][0]]
    req["segments"] = [list(reversed(s)) for s in req["segments"]]
    page.fill("#input", json.dumps(req))
    page.click("#run")
    page.wait_for_selector("#mStatus.ok")
    second = page.evaluate("() => window.measureApp.getLastEvidence()")["response"]

    # 书写顺序保留（端点编号按请求展示），但长度与方差不变
    for s1, s2 in zip(
        first["measurement"]["segments"], second["measurement"]["segments"]
    ):
        assert s1["points"] == list(reversed(s2["points"]))
        assert s1["length"] == s2["length"]
        assert s1["variance"] == s2["variance"]
        assert s1["stddev"] == s2["stddev"]
    page.wait_for_selector("#mStatus.ok")
    assert "方差 0" in page.text_content("#mStatus")
