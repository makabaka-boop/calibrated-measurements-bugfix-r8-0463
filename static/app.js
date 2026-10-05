"use strict";
/* 双视角教学量测台前端。
 * 关键不变量：所有量测点都保存在【原图像素坐标】中，并携带图片身份（视图下标）。
 * 屏幕/CSS 坐标只用于显示与交互，经过视图逆变换 screenToImg 还原后才成为量测。
 */

const CSS_W = 640,
  CSS_H = 480;
const MAX_PAIRS = 20;

// ---------------- 视图状态 ----------------
const views = [0, 1].map((i) => ({
  idx: i, // 图片身份：0 = 视图1，1 = 视图2
  img: null,
  scale: 1,
  ox: 0,
  oy: 0, // 原图像素 -> CSS 像素：screen = scale * img + offset
  points: [], // [{x, y}] 原图像素坐标
  canvas: document.getElementById("canvas" + (i + 1)),
  ctx: null,
}));
views.forEach((v) => {
  v.ctx = v.canvas.getContext("2d");
});

let results = null; // 最近一次后端返回；任何编辑即作废
let drawnOverlays = []; // 最近一次叠画的屏幕位置（供自检/测试）

function imgToScreen(v, x, y) {
  return [v.scale * x + v.ox, v.scale * y + v.oy];
}
function screenToImg(v, sx, sy) {
  return [(sx - v.ox) / v.scale, (sy - v.oy) / v.scale];
}

function setupCanvas(v) {
  const dpr = window.devicePixelRatio || 1;
  v.canvas.width = CSS_W * dpr;
  v.canvas.height = CSS_H * dpr;
  v.canvas.style.width = CSS_W + "px";
  v.canvas.style.height = CSS_H + "px";
  v.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function fitView(v) {
  if (!v.img) {
    v.scale = 1;
    v.ox = 0;
    v.oy = 0;
    return;
  }
  v.scale = Math.min(CSS_W / v.img.width, CSS_H / v.img.height);
  v.ox = (CSS_W - v.img.width * v.scale) / 2;
  v.oy = (CSS_H - v.img.height * v.scale) / 2;
}

// ---------------- 绘制 ----------------
const VIEW_COLORS = ["#e67e00", "#1a9e4b"]; // 视图1 橙，视图2 绿

function drawView(v) {
  const ctx = v.ctx;
  ctx.clearRect(0, 0, CSS_W, CSS_H);
  if (v.img)
    ctx.drawImage(
      v.img,
      v.ox,
      v.oy,
      v.img.width * v.scale,
      v.img.height * v.scale,
    );

  // 量测点（原图像素 -> 屏幕）
  v.points.forEach((p, i) => {
    const [sx, sy] = imgToScreen(v, p.x, p.y);
    ctx.beginPath();
    ctx.arc(sx, sy, 5, 0, 2 * Math.PI);
    ctx.fillStyle = VIEW_COLORS[v.idx];
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "#fff";
    ctx.stroke();
    ctx.fillStyle = "#111";
    ctx.font = "11px sans-serif";
    ctx.fillText(String(i), sx + 7, sy - 6);
  });

  // 重投影叠画（后端返回的原图像素坐标 -> 屏幕）
  if (results) {
    drawnOverlays = drawnOverlays.filter((o) => o.view !== v.idx);
    results.forEach((res) => {
      if (res.status !== "ok") return;
      const rp = v.idx === 0 ? res.reproj1 : res.reproj2;
      const [sx, sy] = imgToScreen(v, rp[0], rp[1]);
      drawnOverlays.push({ view: v.idx, index: res.index, sx, sy });
      ctx.strokeStyle = "#00b7c2";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(sx - 6, sy);
      ctx.lineTo(sx + 6, sy);
      ctx.moveTo(sx, sy - 6);
      ctx.lineTo(sx, sy + 6);
      ctx.stroke();
      const meas = v.points[res.index];
      if (meas) {
        // 量测点与重投影之间的误差连线
        const [mx, my] = imgToScreen(v, meas.x, meas.y);
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(mx, my);
        ctx.lineTo(sx, sy);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    });
  }
  document.getElementById("count" + (v.idx + 1)).textContent = v.points.length;
}

function redrawAll() {
  views.forEach(drawView);
}

// ---------------- 结果作废 ----------------
function invalidateResults() {
  if (results) {
    results = null;
    drawnOverlays = [];
    document.querySelector("#resultTable tbody").innerHTML = "";
    document.getElementById("resultNote").textContent =
      "（已编辑对应点/参数，旧结果作废）";
    clearScene3D();
  }
}

// ---------------- 交互 ----------------
function canvasPos(v, e) {
  // offsetX/offsetY 相对画布内边距盒原点，与 2D 上下文的绘图坐标系一致；
  // 回退路径显式扣除边框（getBoundingClientRect 是边框盒）。
  if (typeof e.offsetX === "number") return [e.offsetX, e.offsetY];
  const rect = v.canvas.getBoundingClientRect();
  return [
    e.clientX - rect.left - v.canvas.clientLeft,
    e.clientY - rect.top - v.canvas.clientTop,
  ];
}

views.forEach((v) => {
  setupCanvas(v);
  let downPos = null,
    dragging = false;

  v.canvas.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    downPos = canvasPos(v, e);
    dragging = false;
  });
  v.canvas.addEventListener("mousemove", (e) => {
    if (!downPos) return;
    const pos = canvasPos(v, e);
    const dx = pos[0] - downPos[0],
      dy = pos[1] - downPos[1];
    if (!dragging && Math.hypot(dx, dy) > 4) dragging = true;
    if (dragging) {
      v.ox += dx;
      v.oy += dy;
      downPos = pos;
      drawView(v);
    }
  });
  v.canvas.addEventListener("mouseup", (e) => {
    if (e.button !== 0 || !downPos) return;
    const wasDragging = dragging;
    downPos = null;
    dragging = false;
    if (wasDragging) return;
    // 单击加点：CSS 坐标 -> 原图像素坐标
    const [sx, sy] = canvasPos(v, e);
    const [ix, iy] = screenToImg(v, sx, sy);
    if (v.points.length >= MAX_PAIRS) {
      setStatus(`每视图最多 ${MAX_PAIRS} 个点`);
      return;
    }
    if (
      v.img &&
      (ix < 0 || iy < 0 || ix >= v.img.width || iy >= v.img.height)
    ) {
      setStatus("点击落在图片外，未记录");
      return;
    }
    v.points.push({ x: ix, y: iy }); // 携带图片身份 v.idx（存于对应视图）
    invalidateResults();
    drawView(v);
    updatePairStatus();
  });
  v.canvas.addEventListener("mouseleave", () => {
    downPos = null;
    dragging = false;
  });

  v.canvas.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    const [sx, sy] = canvasPos(v, e);
    let best = -1,
      bestD = 20; // 20 CSS px 以内
    v.points.forEach((p, i) => {
      const [px, py] = imgToScreen(v, p.x, p.y);
      const d = Math.hypot(px - sx, py - sy);
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    });
    if (best >= 0) {
      v.points.splice(best, 1);
      invalidateResults();
      drawView(v);
      updatePairStatus();
    }
  });

  v.canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const [sx, sy] = canvasPos(v, e);
      const factor = Math.pow(1.0015, -e.deltaY);
      const newScale = Math.min(60, Math.max(0.05, v.scale * factor));
      const k = newScale / v.scale; // 光标下的图像点保持不动
      v.ox = sx - (sx - v.ox) * k;
      v.oy = sy - (sy - v.oy) * k;
      v.scale = newScale;
      drawView(v);
    },
    { passive: false },
  );
});

document.querySelectorAll(".reset").forEach((btn) => {
  btn.addEventListener("click", () => {
    const v = views[Number(btn.dataset.view)];
    fitView(v);
    drawView(v);
  });
});

document.getElementById("btnClear").addEventListener("click", () => {
  views.forEach((v) => {
    v.points = [];
    drawView(v);
  });
  invalidateResults();
  updatePairStatus();
});

["file1", "file2"].forEach((id, i) => {
  document.getElementById(id).addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const img = new Image();
    img.onload = () => {
      views[i].img = img;
      views[i].points = [];
      fitView(views[i]);
      invalidateResults();
      drawView(views[i]);
      updatePairStatus();
    };
    img.src = URL.createObjectURL(f);
  });
});

["K1", "R1", "t1", "K2", "R2", "t2"].forEach((id) => {
  document.getElementById(id).addEventListener("input", invalidateResults);
});

function setStatus(msg) {
  document.getElementById("statusLine").textContent = msg;
}

function updatePairStatus() {
  const n1 = views[0].points.length,
    n2 = views[1].points.length;
  const n = Math.min(n1, n2);
  setStatus(
    `视图1: ${n1} 点，视图2: ${n2} 点，可配对 ${n}/${MAX_PAIRS}` +
      (n1 !== n2 ? "（两视图点数需相等才能三角化）" : ""),
  );
}

// ---------------- 三角化 ----------------
function parseParams() {
  const out = {};
  for (const id of ["K1", "R1", "t1", "K2", "R2", "t2"]) {
    out[id] = JSON.parse(document.getElementById(id).value);
  }
  out.max_px_err = parseFloat(document.getElementById("maxPxErr").value);
  out.sigma = parseFloat(document.getElementById("sigma").value);
  return out;
}

document
  .getElementById("btnTriangulate")
  .addEventListener("click", async () => {
    const n1 = views[0].points.length,
      n2 = views[1].points.length;
    if (n1 !== n2 || n1 < 1) {
      setStatus("两视图点数必须相等且至少 1 对");
      return;
    }
    let params;
    try {
      params = parseParams();
    } catch (e) {
      setStatus("相机参数 JSON 解析失败：" + e.message);
      return;
    }

    const pairs = views[0].points.map((p, i) => ({
      m1: [p.x, p.y], // 原图像素坐标 + 图片身份（m1=视图1）
      m2: [views[1].points[i].x, views[1].points[i].y],
    }));

    setStatus("三角化中…");
    let resp;
    try {
      resp = await fetch("/api/triangulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...params, pairs }),
      });
    } catch (e) {
      setStatus("请求失败：" + e.message);
      return;
    }
    const data = await resp.json();
    if (!data.ok) {
      setStatus("后端拒绝：" + data.error);
      return;
    }
    results = data.results;
    setStatus(
      `完成：${results.filter((r) => r.status === "ok").length}/${results.length} 对通过`,
    );
    renderResults();
    redrawAll();
  });

// ---------------- 结果展示 ----------------
function fmt(x, d = 4) {
  return Number(x).toFixed(d);
}

function renderResults() {
  const tbody = document.querySelector("#resultTable tbody");
  tbody.innerHTML = "";
  document.getElementById("resultNote").textContent =
    `（${new Date().toLocaleTimeString()} 求解，重投影/误差/协方差来自同一次最小二乘解）`;
  results.forEach((res) => {
    const tr = document.createElement("tr");
    if (res.status !== "ok") {
      tr.className = "rejected";
      tr.innerHTML = `<td>${res.index}</td><td>拒绝 [${res.code}]</td>
                      <td colspan="5">${res.reason}</td>`;
    } else {
      const covId = `cov-${res.index}`;
      tr.innerHTML = `<td>${res.index}</td><td>成功</td>
         <td>(${res.X.map((v) => fmt(v)).join(", ")})</td>
         <td>${fmt(res.err1, 3)}</td><td>${fmt(res.err2, 3)}</td>
         <td>${fmt(res.ray_angle_deg, 2)}</td>
         <td><button data-cov="${covId}">查看</button>
             <pre id="${covId}" hidden>${formatCov(res.cov)}</pre></td>`;
    }
    tbody.appendChild(tr);
  });
  tbody.querySelectorAll("button[data-cov]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const pre = document.getElementById(btn.dataset.cov);
      pre.hidden = !pre.hidden;
    });
  });
  updateScene3D();
}

function formatCov(cov) {
  return cov
    .map((row) => row.map((v) => v.toExponential(3)).join("  "))
    .join("\n");
}

// ---------------- 三维示意（自写正交轨道视图） ----------------
const scene = {
  yaw: 0.6,
  pitch: -0.4,
  zoom: 1,
  canvas: document.getElementById("scene3d"),
};
const sctx = scene.canvas.getContext("2d");

function cameraCenters() {
  // C = -R^T t，在浏览器端仅用于显示
  const centers = [];
  for (const [rId, tId] of [
    ["R1", "t1"],
    ["R2", "t2"],
  ]) {
    try {
      const R = JSON.parse(document.getElementById(rId).value);
      const t = JSON.parse(document.getElementById(tId).value);
      centers.push([
        -(R[0][0] * t[0] + R[1][0] * t[1] + R[2][0] * t[2]),
        -(R[0][1] * t[0] + R[1][1] * t[1] + R[2][1] * t[2]),
        -(R[0][2] * t[0] + R[1][2] * t[1] + R[2][2] * t[2]),
      ]);
    } catch (e) {
      centers.push(null);
    }
  }
  return centers;
}

function sceneProject(p, center, scale) {
  const cy = Math.cos(scene.yaw),
    sy = Math.sin(scene.yaw);
  const cp = Math.cos(scene.pitch),
    sp = Math.sin(scene.pitch);
  const x = p[0] - center[0],
    y = p[1] - center[1],
    z = p[2] - center[2];
  const x1 = cy * x + sy * z,
    z1 = -sy * x + cy * z;
  const y2 = cp * y - sp * z1;
  return [
    scene.canvas.width / 2 + scale * x1 * scene.zoom,
    scene.canvas.height / 2 - scale * y2 * scene.zoom,
  ];
}

function clearScene3D() {
  sctx.clearRect(0, 0, scene.canvas.width, scene.canvas.height);
}

function updateScene3D() {
  clearScene3D();
  if (!results) return;
  const pts = results.filter((r) => r.status === "ok").map((r) => r.X);
  const cams = cameraCenters().filter(Boolean);
  const all = pts.concat(cams);
  if (!all.length) return;
  const center = [0, 1, 2].map(
    (k) => all.reduce((s, p) => s + p[k], 0) / all.length,
  );
  let maxR = 1e-6;
  all.forEach((p) => {
    maxR = Math.max(
      maxR,
      Math.hypot(p[0] - center[0], p[1] - center[1], p[2] - center[2]),
    );
  });
  const scale =
    (0.42 * Math.min(scene.canvas.width, scene.canvas.height)) / maxR;

  // 世界坐标轴
  const O = sceneProject(center, center, scale);
  const axes = [
    [[1, 0, 0], "#d00", "X"],
    [[0, 1, 0], "#0a0", "Y"],
    [[0, 0, 1], "#00d", "Z"],
  ];
  axes.forEach(([dir, color, label]) => {
    const end = [
      center[0] + dir[0] * maxR * 0.5,
      center[1] + dir[1] * maxR * 0.5,
      center[2] + dir[2] * maxR * 0.5,
    ];
    const q = sceneProject(end, center, scale);
    sctx.strokeStyle = color;
    sctx.beginPath();
    sctx.moveTo(O[0], O[1]);
    sctx.lineTo(q[0], q[1]);
    sctx.stroke();
    sctx.fillStyle = color;
    sctx.fillText(label, q[0] + 3, q[1] + 3);
  });
  cams.forEach((C, i) => {
    const q = sceneProject(C, center, scale);
    sctx.fillStyle = VIEW_COLORS[i];
    sctx.fillRect(q[0] - 5, q[1] - 5, 10, 10);
    sctx.fillStyle = "#111";
    sctx.fillText("C" + (i + 1), q[0] + 7, q[1] - 7);
  });
  results.forEach((res) => {
    if (res.status !== "ok") return;
    const q = sceneProject(res.X, center, scale);
    sctx.beginPath();
    sctx.arc(q[0], q[1], 4, 0, 2 * Math.PI);
    sctx.fillStyle = "#00b7c2";
    sctx.fill();
    sctx.fillStyle = "#111";
    sctx.fillText(String(res.index), q[0] + 6, q[1] - 5);
  });
}

(function enableSceneOrbit() {
  let last = null;
  scene.canvas.addEventListener("mousedown", (e) => {
    last = [e.clientX, e.clientY];
  });
  window.addEventListener("mousemove", (e) => {
    if (!last) return;
    scene.yaw += (e.clientX - last[0]) * 0.01;
    scene.pitch += (e.clientY - last[1]) * 0.01;
    last = [e.clientX, e.clientY];
    updateScene3D();
  });
  window.addEventListener("mouseup", () => {
    last = null;
  });
  scene.canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      scene.zoom *= Math.pow(1.0015, -e.deltaY);
      updateScene3D();
    },
    { passive: false },
  );
})();

// ---------------- 初始化：载入演示数据 ----------------
async function loadDemo() {
  try {
    const resp = await fetch("/demo/cameras.json");
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const cams = await resp.json();
    for (const id of ["K1", "R1", "t1", "K2", "R2", "t2"]) {
      const val = cams[id];
      document.getElementById(id).value = JSON.stringify(val, (k, v) =>
        typeof v === "number" ? Math.round(v * 1e9) / 1e9 : v,
      );
    }
    await Promise.all(
      [0, 1].map(
        (i) =>
          new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
              views[i].img = img;
              fitView(views[i]);
              drawView(views[i]);
              resolve();
            };
            img.onerror = () => resolve();
            img.src = `/demo/view${i + 1}.png`;
          }),
      ),
    );
    setStatus("演示数据已载入：在两张图上按相同顺序点击对应标记点");
  } catch (e) {
    setStatus("演示数据载入失败（可手动上传图片并填写参数）：" + e.message);
  }
  updatePairStatus();
  window.app.ready = true;
}

// 测试/调试钩子（不改变量测语义：坐标均为原图像素）
window.app = {
  ready: false,
  views,
  imgToScreen: (vi, x, y) => imgToScreen(views[vi], x, y),
  screenToImg: (vi, sx, sy) => screenToImg(views[vi], sx, sy),
  getResults: () => results,
  getDrawnOverlays: () => drawnOverlays,
};

loadDemo();
