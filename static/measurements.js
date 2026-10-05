"use strict";
/* 标定距离量测前端。
 * 不变量：
 *  - 页面展示、量测结果与下载文件始终引用【原始对应点编号】；
 *  - 下载内容 = 实际发出的请求 + 同一次返回（成功或失败都可下载）；
 *  - 任何一次点击都重新发起请求，不与上一次三角化结果混用。
 */

const inputEl = document.getElementById("input");
const statusEl = document.getElementById("mStatus");
const rawEl = document.getElementById("mRaw");
const evidenceEl = document.getElementById("mEvidence");
const downloadEl = document.getElementById("mDownload");

let lastBlobUrl = null;
let lastEvidence = null; // { request, response } 供自检/下载

function fmt(x, d = 5) {
  const n = Number(x);
  return Number.isFinite(n) ? n.toFixed(d) : String(x);
}

function project(K, R, t, X) {
  const z = R[2][0] * X[0] + R[2][1] * X[1] + R[2][2] * X[2] + t[2];
  const dot = (row) =>
    row[0] * X[0] + row[1] * X[1] + row[2] * X[2] + row[3];
  const P = [
    [
      K[0][0] * R[0][0] + K[0][1] * R[1][0] + K[0][2] * R[2][0],
      K[0][0] * R[0][1] + K[0][1] * R[1][1] + K[0][2] * R[2][1],
      K[0][0] * R[0][2] + K[0][1] * R[1][2] + K[0][2] * R[2][2],
      K[0][0] * t[0] + K[0][1] * t[1] + K[0][2] * t[2],
    ],
    [
      K[1][1] * R[1][0] + K[1][2] * R[2][0],
      K[1][1] * R[1][1] + K[1][2] * R[2][1],
      K[1][1] * R[1][2] + K[1][2] * R[2][2],
      K[1][1] * t[1] + K[1][2] * t[2],
    ],
  ];
  return [dot(P[0]) / z, dot(P[1]) / z];
}

async function buildDemoRequest() {
  const resp = await fetch("/demo/cameras.json");
  if (!resp.ok) throw new Error("HTTP " + resp.status);
  const cams = await resp.json();
  const pairs = cams.points3d.slice(0, 4).map((X) => ({
    m1: project(cams.K1, cams.R1, cams.t1, X),
    m2: project(cams.K2, cams.R2, cams.t2, X),
  }));
  const p0 = cams.points3d[0];
  const p1 = cams.points3d[1];
  const known = Math.hypot(
    p0[0] - p1[0],
    p0[1] - p1[1],
    p0[2] - p1[2],
  );
  return {
    K1: cams.K1,
    R1: cams.R1,
    t1: cams.t1,
    K2: cams.K2,
    R2: cams.R2,
    t2: cams.t2,
    max_px_err: 2.0,
    sigma: 1.0,
    pairs,
    reference: [0, 1],
    known_length: Number(known.toFixed(9)),
    segments: [
      [0, 1], // 参考段量自己：长度应精确等于 known_length、方差为 0
      [1, 2],
      [2, 3],
      [0, 2],
    ],
  };
}

// ---------------- 证据渲染（全部使用原始编号） ----------------

function triangulationTable(rows) {
  const head =
    "<tr><th>原始编号</th><th>状态</th><th>空间点 X</th>" +
    "<th>误差1 (px)</th><th>误差2 (px)</th><th>拒绝原因（保留）</th></tr>";
  const body = rows
    .map((r) => {
      if (r.status !== "ok") {
        return `<tr class="rejected"><td>${r.index}</td><td>拒绝 [${r.code ?? ""}]</td>
          <td>—</td><td>—</td><td>—</td><td>${r.reason ?? ""}</td></tr>`;
      }
      const x = `(${r.X.map((v) => fmt(v)).join(", ")})`;
      return `<tr><td>${r.index}</td><td>成功</td><td>${x}</td>
        <td>${fmt(r.err1, 3)}</td><td>${fmt(r.err2, 3)}</td><td></td></tr>`;
    })
    .join("");
  return `<div><h3 style="font-size:14px;">同次三角化证据（原始对应点编号）</h3>
          <table><thead>${head}</thead><tbody>${body}</tbody></table></div>`;
}

function measurementTable(m) {
  const head =
    "<tr><th>线段（原始编号）</th><th>长度</th><th>标准差</th><th>方差</th><th>备注</th></tr>";
  const body = m.segments
    .map((s) => {
      const isRef =
        (s.points[0] === m.reference[0] && s.points[1] === m.reference[1]) ||
        (s.points[1] === m.reference[0] && s.points[0] === m.reference[1]);
      return `<tr><td>${s.points[0]} – ${s.points[1]}</td>
        <td>${fmt(s.length, 6)}</td><td>${fmt(s.stddev, 4)}</td>
        <td>${Number(s.variance).toExponential(3)}</td>
        <td>${isRef ? "参考段（精确值）" : ""}</td></tr>`;
    })
    .join("");
  const cov = m.covariance;
  const labels = cov.points.map((i) => "#" + i).join(" ");
  const matrix = cov.matrix
    .map((row) => row.map((v) => Number(v).toExponential(2)).join("  "))
    .join("\n");
  return `<div><h3 style="font-size:14px;">量测结果（端点为原始编号）</h3>
      <table><thead>${head}</thead><tbody>${body}</tbody></table>
      <p class="m-note">参考段 ${m.reference[0]}–${m.reference[1]}，
      已知长度 ${fmt(m.known_length, 6)}（精确值），尺度因子 ${fmt(m.scale, 6)}</p>
      <p class="m-note">联合长度协方差，端点顺序（${labels}）——
      共享端点使非对角项一般不为零：</p>
      <pre style="font-size:11px; margin:0;">${matrix}</pre>
    </div>`;
}

function renderEvidence(request, data) {
  evidenceEl.innerHTML = "";
  if (Array.isArray(data.results)) {
    const wrap = document.createElement("div");
    wrap.innerHTML = triangulationTable(data.results);
    evidenceEl.appendChild(wrap.firstElementChild);
  }
  if (data.ok && data.measurement) {
    const wrap = document.createElement("div");
    wrap.innerHTML = measurementTable(data.measurement);
    evidenceEl.appendChild(wrap.firstElementChild);
  }
}

// ---------------- 下载：始终是“本次请求 + 同次返回” ----------------

function offerDownload(request, data) {
  const payload = {
    request,
    response: data,
    exported_at: new Date().toISOString(),
  };
  lastEvidence = payload;
  if (lastBlobUrl) URL.revokeObjectURL(lastBlobUrl);
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  lastBlobUrl = URL.createObjectURL(blob);
  downloadEl.href = lastBlobUrl;
  downloadEl.style.display = "inline-block";
}

// ---------------- 主流程 ----------------

document.getElementById("run").onclick = async () => {
  statusEl.textContent = "量测中…";
  statusEl.className = "";
  evidenceEl.innerHTML = "";
  downloadEl.style.display = "none";

  let request;
  try {
    request = JSON.parse(inputEl.value);
  } catch (e) {
    statusEl.textContent = "请求 JSON 解析失败：" + e.message;
    statusEl.className = "bad";
    return;
  }

  let data;
  try {
    const response = await fetch("/api/measurements", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    data = await response.json();
  } catch (e) {
    statusEl.textContent = "请求失败：" + e.message;
    statusEl.className = "bad";
    return;
  }

  rawEl.textContent = JSON.stringify(data, null, 2);
  renderEvidence(request, data);
  offerDownload(request, data); // 成功/失败都可下载同次证据

  if (data.ok) {
    const m = data.measurement;
    const self = m.segments.find(
      (s) =>
        (s.points[0] === m.reference[0] && s.points[1] === m.reference[1]) ||
        (s.points[1] === m.reference[0] && s.points[0] === m.reference[1]),
    );
    statusEl.textContent =
      `量测成功：${m.segments.length} 条线段（原始编号）；` +
      (self
        ? `参考段自量长度 ${fmt(self.length, 8)}、方差 ${self.variance}`
        : "（本次未让参考段量自己）");
    statusEl.className = "ok";
  } else {
    statusEl.textContent = `整次量测失败 [${data.code ?? "ERROR"}]：${data.error}`;
    statusEl.className = "bad";
  }
};

// 测试/自检钩子（只读，不改变量测语义）
window.measureApp = {
  ready: false,
  getLastEvidence: () => lastEvidence,
  renderEvidence,
  run: () => document.getElementById("run").click(),
};

(async function init() {
  try {
    const demo = await buildDemoRequest();
    inputEl.value = JSON.stringify(demo, null, 2);
  } catch (e) {
    inputEl.value = "";
    statusEl.textContent = "演示数据载入失败：" + e.message;
  }
  window.measureApp.ready = true;
})();
