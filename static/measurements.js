document.getElementById("run").onclick = async () => {
  const out = document.getElementById("out");
  out.textContent = "量测中";
  try {
    const response = await fetch("/api/measurements", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: document.getElementById("input").value,
    });
    const data = await response.json();
    out.textContent = JSON.stringify(data, null, 2);
    if (response.ok) {
      const a = document.createElement("a");
      a.textContent = "下载本次证据";
      a.href = URL.createObjectURL(
        new Blob([JSON.stringify(data)], { type: "application/json" }),
      );
      a.download = "measurement.json";
      out.append(a);
    }
  } catch (e) {
    out.textContent = e.message;
  }
};
