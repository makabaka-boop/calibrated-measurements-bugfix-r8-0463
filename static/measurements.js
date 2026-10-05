"use strict";
/* 标定距离量测前端。
 * 不变量：发送的请求体就是输入框原文；页面证据与下载内容就是后端响应原文；
 * 所有端点编号均为原始对应点编号，量测与证据来自同一次重建。
 */

const els = {
  input: document.getElementById("input"),
  run: document.getElementById("run"),
  status: document.getElementById("status"),
  evidence: document.getElementById("evidence"),
  out: document.getElementById("out"),
  download: document.getElementById("download"),
};

// 供测试/自检：最后一次实际发送的请求体与收到的响应
window.measurement = { lastRequest: null, lastResponse: null, rawResponse: null };

function fmt(x, digits = 6) {
  return typeof x === "number" && Number.isFinite(x) ? x.toFixed(digits) : String(x);
}

function table(parent, id, headers, rows) {
  const t = document.createElement("table");
  t.id = id;
  t.border = "1";
  t.cellPadding = "4";
  const head = t.createTHead().insertRow();
  for (const h of headers) head.insertCell().textContent = h;
  const body = t.createTBody();
  for (const r of rows) {
    const tr = body.insertRow();
    for (const cell of r) tr.insertCell().textContent = cell;
  }
  parent.appendChild(t);
  return t;
}

function renderEvidence(data) {
  els.evidence.innerHTML = "";
  if (Array.isArray(data.results)) {
    // 三角化证据：每个原始编号的状态、空间点、误差与原始拒绝原因
    table(
      els.evidence,
      "triangulationTable",
      ["编号", "状态", "空间点 X", "误差1(px)", "误差2(px)", "射线夹角(°)", "拒绝原因"],
      data.results.map((r) =>
        r.status === "ok"
          ? [
              r.index,
              "ok",
              `(${r.X.map((v) => fmt(v, 4)).join(", ")})`,
              fmt(r.err1, 4),
              fmt(r.err2, 4),
              fmt(r.ray_angle_deg, 3),
              "",
            ]
          : [r.index, "rejected", "", "", "", "", `${r.code}: ${r.reason}`],
      ),
    );
  }
  const m = data.measurement;
  if (m) {
    const div = document.createElement("div");
    div.id = "measurementSummary";
    div.textContent =
      `参考段（原始编号）=[${m.reference.join(", ")}]，已知长度=${m.known_length}，` +
      `比例尺=${fmt(m.scale)}`;
    els.evidence.appendChild(div);
    table(
      els.evidence,
      "segmentTable",
      ["端点（原始编号）", "长度", "标准差", "方差", "局部导数"],
      m.segments.map((s) => [
        `[${s.points.join(", ")}]`,
        fmt(s.length),
        fmt(s.stddev),
        fmt(s.variance, 9),
        Object.entries(s.gradient)
          .map(([i, g]) => `#${i}: (${g.map((v) => fmt(v, 4)).join(", ")})`)
          .join("  "),
      ]),
    );
  }
}

els.run.onclick = async () => {
  const body = els.input.value; // 实际请求体 = 输入框原文，不做任何改写
  els.status.textContent = "量测中…";
  els.evidence.innerHTML = "";
  els.out.textContent = "";
  els.download.hidden = true;
  try {
    const response = await fetch("/api/measurements", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
    const text = await response.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      throw new Error(`服务返回非 JSON（HTTP ${response.status}）：${text.slice(0, 200)}`);
    }
    window.measurement = { lastRequest: body, lastResponse: data, rawResponse: text };
    els.status.textContent = data.ok
      ? "量测成功"
      : `量测失败：${data.error || "未知错误"}`;
    els.status.style.color = data.ok ? "#1a9e4b" : "#c0392b";
    renderEvidence(data);
    els.out.textContent = text; // 页面证据 = 后端响应原文
    if (response.ok && data.ok) {
      // 下载内容 = 同一次响应原文（与页面证据、量测结果完全一致）
      els.download.href = URL.createObjectURL(
        new Blob([text], { type: "application/json" }),
      );
      els.download.download = "measurement.json";
      els.download.hidden = false;
    }
  } catch (e) {
    els.status.textContent = `量测失败：${e.message}`;
    els.status.style.color = "#c0392b";
  }
};
