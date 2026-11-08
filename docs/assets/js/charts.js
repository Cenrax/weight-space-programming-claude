/* Tiny inline-SVG chart kit (no dependencies).
   Specs (from docs/data/<exp_id>.json → "charts": [...]):
     bar:  {"type":"bar","title","subtitle","unit","categories":[...],
            "series":[{"name","values":[...]}], "caption"}
     line: {"type":"line","title","subtitle","x_label","y_label","x":[...],
            "series":[{"name","values":[...]}], "y_min", "y_max",
            "reference_lines":[{"y","label"}], "caption"}
   Up to 4 series, in fixed order: --s1 blue, --s2 teal, --s3 purple, --s4 olive
   (validated with the dataviz palette validator in both themes). In a
   single-series bar chart, categories that are baselines ("PPO …", "random",
   "zero net", "heuristic", "baseline") are drawn in neutral grey (--s-ref) so the
   model's own attempts stand out; the category label still names each bar.
   The rendered <figure class="fig chart"> carries a caption below the plot:
   "Figure n. <title>. <caption>" (app.js writes the number). */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const SERIES = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"];
  const REF = "var(--s-ref)";
  const isRefCat = (c) => /^(ppo|random|zero|heuristic|hand[- ]?coded|baseline|unedited|before)\b/i.test(String(c).trim());
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function el(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs || {}) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function fmt(v) {
    if (v == null || Number.isNaN(v)) return "—";
    const a = Math.abs(v);
    const s = a >= 10000 ? (v / 1000).toFixed(a >= 100000 ? 0 : 1) + "K"
      : a >= 100 ? Math.round(v).toLocaleString()
      : a >= 10 ? (Math.round(v * 10) / 10).toString()
      : (Math.round(v * 100) / 100).toString();
    return s.replace(/^-/, "−"); // true minus sign
  }
  function niceTicks(min, max, count) {
    if (min === max) { max = min + 1; }
    const span = max - min;
    const step0 = Math.pow(10, Math.floor(Math.log10(span / count)));
    const err = (span / count) / step0;
    const step = step0 * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
    const lo = Math.floor(min / step) * step, hi = Math.ceil(max / step) * step;
    const out = [];
    for (let v = lo; v <= hi + step / 2; v += step) out.push(Math.round(v / step) * step);
    return out;
  }
  // bar with a small rounded data end, square at the baseline
  function barPath(x0, x1, y, h) {
    const r = Math.min(2, Math.abs(x1 - x0), h / 2);
    if (x1 >= x0) return `M${x0},${y}H${x1 - r}Q${x1},${y} ${x1},${y + r}V${y + h - r}Q${x1},${y + h} ${x1 - r},${y + h}H${x0}Z`;
    return `M${x0},${y}H${x1 + r}Q${x1},${y} ${x1},${y + r}V${y + h - r}Q${x1},${y + h} ${x1 + r},${y + h}H${x0}Z`;
  }
  const textW = (s, px) => String(s).length * px * 0.55; // rough sans advance

  function legend(spec, kind) {
    const S = (spec.series || []).slice(0, 4);
    const items = S.map((s, i) => ({ name: s.name, color: SERIES[i] }));
    if (kind === "bar" && S.length === 1 && (spec.categories || []).some(isRefCat)) items.push({ name: "Baselines", color: REF });
    if (items.length < 2) return "";
    return `<ul class="legend">${items.map((it) =>
      `<li><span class="sw${kind === "line" ? " line" : ""}" style="background:${it.color}"></span>${esc(it.name)}</li>`).join("")}</ul>`;
  }
  function dataTable(spec) {
    const isLine = spec.type === "line";
    const rowsKeys = isLine ? spec.x : spec.categories;
    const head = `<tr><th scope="col">${esc(isLine ? (spec.x_label || "x") : "")}</th>${spec.series.map((s) => `<th scope="col" class="num">${esc(s.name)}</th>`).join("")}</tr>`;
    const body = rowsKeys.map((k, i) => `<tr><th scope="row">${esc(k)}</th>${spec.series.map((s) => `<td class="num">${esc(fmt(s.values[i]))}</td>`).join("")}</tr>`).join("");
    return `<details class="x"><summary>Show data</summary><div class="table-wrap" tabindex="0" role="region" aria-label="Chart data"><table><thead>${head}</thead><tbody>${body}</tbody></table></div></details>`;
  }

  function tipHTML(title, rows) {
    return `<div class="t">${esc(title)}</div>` + rows.map((r) =>
      `<div class="r"><span><span class="sw" style="background:${r.color}"></span>${esc(r.name)}</span><b>${esc(fmt(r.value))}</b></div>`).join("");
  }
  function placeTip(fig, tip, x, y) {
    const fw = fig.clientWidth, tw = tip.offsetWidth, th = tip.offsetHeight;
    let left = x + 14; if (left + tw > fw) left = x - tw - 14; if (left < 0) left = 0;
    let top = y - th - 10; if (top < 0) top = y + 14;
    tip.style.left = left + "px"; tip.style.top = top + "px";
  }

  /* ---------------- horizontal grouped bars ---------------- */
  function drawBar(holder, spec, fig, tip) {
    const W = Math.max(260, holder.clientWidth);
    const S = spec.series.slice(0, 4);
    const single = S.length === 1;
    const all = S.flatMap((s) => s.values).filter((v) => v != null);
    const lo = Math.min(0, ...all), hi = Math.max(0, ...all);
    const ticks = niceTicks(lo, hi, W < 420 ? 3 : 5);
    const dmin = ticks[0], dmax = ticks[ticks.length - 1];
    const negOnly = hi <= 0 && lo < 0;
    // room for value labels on whichever side bars grow
    const valW = Math.max(...all.map((v) => textW(fmt(v), 11.5))) + 8;
    const padL = lo < 0 ? Math.max(valW, 30) : 2;
    const padR = hi > 0 ? Math.max(valW, 30) : 12;
    const catH = 17, barH = single ? 13 : 10, gap = 2, groupGap = single ? 8 : 14;
    const groupH = catH + S.length * barH + (S.length - 1) * gap;
    const H = spec.categories.length * (groupH + groupGap) + 18;
    const x = (v) => padL + ((v - dmin) / (dmax - dmin)) * (W - padL - padR);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": (spec.title || "Bar chart") + (spec.unit ? ` (${spec.unit})` : "") });
    ticks.forEach((t) => {
      el("line", { class: "grid", x1: x(t), x2: x(t), y1: 0, y2: H - 18 }, svg);
      const tx = el("text", { x: x(t), y: H - 4, "text-anchor": "middle" }, svg); tx.textContent = fmt(t);
    });
    el("line", { class: "axis", x1: x(0), x2: x(0), y1: 0, y2: H - 18 }, svg);
    spec.categories.forEach((cat, ci) => {
      const gy = ci * (groupH + groupGap);
      // category label sits above its bars, on the side away from the bars for all-negative charts
      const ct = el("text", { class: "cat", x: negOnly ? x(0) - 2 : x(0) + (lo < 0 ? 4 : 2), y: gy + 12, "text-anchor": negOnly ? "end" : "start" }, svg);
      ct.textContent = cat;
      const hit = el("rect", { x: 0, y: gy, width: W, height: groupH + groupGap / 2, fill: "transparent" }, svg);
      S.forEach((s, si) => {
        const v = s.values[ci]; if (v == null) return;
        const y = gy + catH + si * (barH + gap);
        const color = single && isRefCat(cat) ? REF : SERIES[si];
        el("path", { d: barPath(x(0), x(v), y, barH), fill: color }, svg);
        const t = el("text", { class: "val", x: v >= 0 ? x(v) + 4 : x(v) - 4, y: y + barH - 2, "text-anchor": v >= 0 ? "start" : "end" }, svg);
        t.textContent = fmt(v);
      });
      const show = (ev) => {
        tip.innerHTML = tipHTML(cat, S.map((s, si) => ({ name: s.name, value: s.values[ci], color: single && isRefCat(cat) ? REF : SERIES[si] }))) + (spec.unit && !S.some((s) => s.name === spec.unit) ? `<div class="r">${esc(spec.unit)}</div>` : "");
        tip.classList.add("on");
        const r = fig.getBoundingClientRect();
        placeTip(fig, tip, ev.clientX - r.left, ev.clientY - r.top);
      };
      hit.addEventListener("pointermove", show);
      hit.addEventListener("pointerleave", () => tip.classList.remove("on"));
    });
    holder.replaceChildren(svg);
  }

  /* ---------------- line chart ---------------- */
  function drawLine(holder, spec, fig, tip) {
    const W = Math.max(260, holder.clientWidth);
    const H = Math.round(Math.min(320, Math.max(210, W * 0.48)));
    const S = spec.series.slice(0, 4);
    const xs = spec.x.map(Number);
    const all = S.flatMap((s) => s.values).filter((v) => v != null).concat((spec.reference_lines || []).map((r) => r.y));
    const yt = niceTicks(spec.y_min ?? Math.min(...all), spec.y_max ?? Math.max(...all), 4);
    const ymin = spec.y_min ?? yt[0], ymax = spec.y_max ?? yt[yt.length - 1];
    const endLabels = W >= 480;
    const m = { l: 40, r: endLabels ? 104 : 14, t: 10, b: 34 };
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const X = (v) => m.l + ((v - xmin) / (xmax - xmin || 1)) * (W - m.l - m.r);
    const Y = (v) => m.t + (1 - (v - ymin) / (ymax - ymin || 1)) * (H - m.t - m.b);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": spec.title || "Line chart" });
    yt.filter((t) => t >= ymin && t <= ymax).forEach((t) => {
      el("line", { class: "grid", x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }, svg);
      const tx = el("text", { x: m.l - 6, y: Y(t) + 4, "text-anchor": "end" }, svg); tx.textContent = fmt(t);
    });
    xs.forEach((v, i) => {
      const tx = el("text", { x: X(v), y: H - m.b + 16, "text-anchor": "middle" }, svg); tx.textContent = spec.x[i];
    });
    if (spec.x_label) { const t = el("text", { x: (m.l + W - m.r) / 2, y: H - 2, "text-anchor": "middle" }, svg); t.textContent = spec.x_label; }
    el("line", { class: "axis", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }, svg);
    (spec.reference_lines || []).forEach((r) => {
      el("line", { class: "ref", x1: m.l, x2: W - m.r, y1: Y(r.y), y2: Y(r.y) }, svg);
      const t = el("text", { x: m.l + 4, y: Y(r.y) - 5 }, svg); t.textContent = r.label || "";
    });
    const cross = el("line", { class: "cross", x1: 0, x2: 0, y1: m.t, y2: H - m.b, opacity: 0 }, svg);
    const ends = [];
    S.forEach((s, si) => {
      const pts = s.values.map((v, i) => (v == null ? null : [X(xs[i]), Y(v)]));
      let d = "", pen = false;
      pts.forEach((p) => { if (!p) { pen = false; return; } d += (pen ? "L" : "M") + p[0].toFixed(1) + "," + p[1].toFixed(1); pen = true; });
      el("path", { d, fill: "none", stroke: SERIES[si], "stroke-width": 1.75, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
      pts.forEach((p) => p && el("circle", { cx: p[0], cy: p[1], r: 3.5, fill: SERIES[si], stroke: "var(--bg)", "stroke-width": 1.5 }, svg));
      const last = [...pts].reverse().find(Boolean);
      const lastV = [...s.values].reverse().find((v) => v != null);
      if (last) ends.push({ y: last[1], x: last[0], text: `${s.name} ${fmt(lastV)}` });
    });
    // direct end labels only when they don't collide (legend + tooltip carry identity otherwise)
    if (endLabels) {
      const sorted = [...ends].sort((a, b) => a.y - b.y);
      const ok = sorted.every((e, i) => i === 0 || e.y - sorted[i - 1].y >= 14);
      if (ok) ends.forEach((e) => { const t = el("text", { class: "val", x: e.x + 8, y: e.y + 4 }, svg); t.textContent = e.text; });
    }
    const hit = el("rect", { x: m.l - 10, y: 0, width: W - m.l - m.r + 20, height: H - m.b, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", (ev) => {
      const r = svg.getBoundingClientRect();
      const px = (ev.clientX - r.left) * (W / r.width);
      let bi = 0; xs.forEach((v, i) => { if (Math.abs(X(v) - px) < Math.abs(X(xs[bi]) - px)) bi = i; });
      cross.setAttribute("x1", X(xs[bi])); cross.setAttribute("x2", X(xs[bi])); cross.setAttribute("opacity", 1);
      tip.innerHTML = tipHTML(`${spec.x_label ? spec.x_label + ": " : ""}${spec.x[bi]}`, S.map((s, si) => ({ name: s.name, value: s.values[bi], color: SERIES[si] })));
      tip.classList.add("on");
      const fr = fig.getBoundingClientRect();
      placeTip(fig, tip, (X(xs[bi]) * r.width) / W + (r.left - fr.left), ev.clientY - fr.top);
    });
    hit.addEventListener("pointerleave", () => { tip.classList.remove("on"); cross.setAttribute("opacity", 0); });
    holder.replaceChildren(svg);
  }

  function render(spec) {
    const fig = document.createElement("figure");
    fig.className = "fig chart";
    const kind = spec.type === "line" ? "line" : "bar";
    const axis = spec.subtitle || spec.y_label || spec.unit;
    const title = spec.title ? `<span class="fig-title">${esc(String(spec.title).replace(/\.?$/, "."))}</span> ` : "";
    fig.innerHTML =
      legend(spec, kind) +
      (axis ? `<p class="axis-label">${esc(axis)}</p>` : "") +
      `<div class="plot"></div><div class="tip" role="presentation"></div>` +
      `<figcaption><span class="fig-label"></span>${title}${spec.caption ? esc(spec.caption) : ""}</figcaption>` +
      dataTable(spec);
    const holder = fig.querySelector(".plot"), tip = fig.querySelector(".tip");
    const draw = () => (kind === "line" ? drawLine : drawBar)(holder, spec, fig, tip);
    let lastW = 0;
    const ro = new ResizeObserver(() => { const w = holder.clientWidth; if (w && Math.abs(w - lastW) > 4) { lastW = w; draw(); } });
    ro.observe(holder);
    return fig;
  }

  window.WSPCharts = { render };
})();
