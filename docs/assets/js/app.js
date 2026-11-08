/* Page behaviour + data-driven experiment rendering. No dependencies.
   Every <div class="mount" data-exp="exp1"> is filled from data/exp1.json.
   Schema (PLAN.md "Blog data"), plus OPTIONAL extensions marked (+):
   {
     "exp_id", "title", "status": "pending"|"running"|"done",
     "summary": "paragraphs separated by blank lines; `code`, **bold**, *em*, [link](url)",
     "table":  {"columns":[...], "rows":[[...]], "caption"(+)},
     "runs":   [{"label","answer","correct","confidence","reasoning_excerpt","prompt_excerpt",
                 "group"(+), "score"(+), "tokens"(+), "weights_excerpt"(+), "log_path"(+)}],
     "media":  [{"type":"gif"|"png","src","caption", "alt"(+), "group"(+), "pixel"(+)}],
     "charts"(+): [see assets/js/charts.js]
   }
   Runs are grouped into tabs by run.group, else by the part of run.label before " · ".

   Article apparatus handled here:
   - Figures and tables are numbered in document order once every mount has
     rendered: any <figure class="fig"> gets "Figure n.", any <figure class="tbl">
     gets "Table n." written into its .fig-label.
   - Sidenotes: <span class="sn">text</span> inside a paragraph becomes a numbered
     margin note on wide screens and an entry in the #notes list everywhere.
   - Citations: <span class="cite" data-refs="elhage olsson">[fallback]</span>
     becomes [10, 11] linking to #ref-elhage / #ref-olsson, with back-links. */
(function () {
  "use strict";
  const root = document.documentElement;
  root.classList.add("js");
  const REPO = "https://github.com/Cenrax/weight-space-programming-claude";
  const LOGS = REPO + "/tree/main/logs";
  const DATA = root.getAttribute("data-data-base") || "data/";
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
  };

  /* very small, escape-first inline formatter */
  function inline(s) {
    let t = esc(s);
    t = t.replace(/`([^`]+)`/g, "<code>$1</code>");
    t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    t = t.replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1<em>$2</em>");
    t = t.replace(/\[([^\]]+)\]\(((?:https?:\/\/|\.{0,2}\/|#)[^)\s]*|[\w.\/-]+\.[a-z]{2,5}[^)\s]*)\)/g, '<a href="$2">$1</a>');
    return t;
  }
  const paras = (s) => String(s || "").split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean)
    .map((p) => `<p>${inline(p).replace(/\n/g, "<br>")}</p>`).join("");
  const LABEL = '<span class="fig-label"></span>';

  /* ---------------- theme ---------------- */
  function initTheme() {
    const btn = document.querySelector(".theme-toggle");
    if (!btn) return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const current = () => root.getAttribute("data-theme") || (mq.matches ? "dark" : "light");
    const txt = btn.querySelector(".label");
    const sync = () => {
      const c = current();
      btn.setAttribute("aria-pressed", String(c === "dark"));
      btn.setAttribute("aria-label", c === "dark" ? "Switch to light theme" : "Switch to dark theme");
      if (txt) txt.textContent = c === "dark" ? "Light" : "Dark";
    };
    btn.addEventListener("click", () => {
      const next = current() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next); store.set("wsp-theme", next); sync();
    });
    mq.addEventListener && mq.addEventListener("change", sync);
    sync();
  }

  /* ---------------- sticky contents rail (very wide screens only; CSS hides it otherwise) ---------------- */
  function initRail() {
    const src = document.querySelector(".contents > ol");
    if (!src) return;
    const rail = document.createElement("nav");
    rail.className = "toc-rail"; rail.setAttribute("aria-label", "Contents (sidebar)");
    const ol = document.createElement("ol");
    [...src.children].forEach((li) => { const a = li.querySelector(":scope > a"); if (a) { const n = document.createElement("li"); n.appendChild(a.cloneNode(true)); ol.appendChild(n); } });
    rail.appendChild(ol);
    document.body.appendChild(rail);
    const links = [...ol.querySelectorAll("a")];
    const targets = links.map((a) => document.getElementById(a.getAttribute("href").slice(1))).filter(Boolean);
    let raf = 0;
    const upd = () => {
      raf = 0;
      if (getComputedStyle(rail).display === "none") return;
      let cur = null;
      targets.forEach((t) => { if (t.getBoundingClientRect().top < innerHeight * 0.3) cur = t; });
      links.forEach((a) => (cur && a.getAttribute("href") === "#" + cur.id ? a.setAttribute("aria-current", "true") : a.removeAttribute("aria-current")));
    };
    addEventListener("scroll", () => { if (!raf) raf = requestAnimationFrame(upd); }, { passive: true });
    upd();
  }

  /* ---------------- sidenotes -> margin notes + Notes list ---------------- */
  function initSidenotes() {
    const notes = [...document.querySelectorAll("main .sn")];
    const sec = document.getElementById("notes");
    if (!notes.length) { if (sec) sec.hidden = true; return; }
    const list = sec && sec.querySelector("ol");
    notes.forEach((sn, i) => {
      const n = i + 1;
      const body = sn.innerHTML;
      const mark = document.createElement("sup");
      mark.className = "sn-ref";
      mark.innerHTML = `<a href="#fn-${n}" id="fnref-${n}" aria-label="Note ${n}">${n}</a>`;
      sn.parentNode.insertBefore(mark, sn);
      sn.id = "sn-" + n;
      sn.setAttribute("role", "note");
      sn.innerHTML = `<span class="sn-num">${n}</span>${body}`;
      if (list) {
        const li = document.createElement("li");
        li.id = "fn-" + n;
        li.innerHTML = `${body} <a class="backref" href="#fnref-${n}" aria-label="Back to text for note ${n}">↩</a>`;
        list.appendChild(li);
      }
    });
  }

  /* ---------------- numbered citations ---------------- */
  function initCitations() {
    const refs = [...document.querySelectorAll("ol.refs > li[id]")];
    const num = new Map(refs.map((li, i) => [li.id.replace(/^ref-/, ""), i + 1]));
    const back = new Map();
    document.querySelectorAll(".cite[data-refs]").forEach((c) => {
      const keys = c.dataset.refs.split(/[\s,]+/).filter((k) => num.has(k));
      if (!keys.length) return;
      const fallback = c.textContent.trim();
      const parts = keys.map((k) => {
        const occ = (back.get(k) || []).length + 1;
        const id = `cite-${k}-${occ}`;
        if (!back.has(k)) back.set(k, []);
        back.get(k).push(id);
        return `<a href="#ref-${k}" id="${id}">${num.get(k)}</a>`;
      });
      c.innerHTML = `[${parts.join(", ")}]`;
      if (fallback) c.title = fallback.replace(/^\[|\]$/g, "");
    });
    refs.forEach((li) => {
      const ids = back.get(li.id.replace(/^ref-/, ""));
      if (!ids) return;
      const letters = "abcdefghij";
      const html = ids.length === 1
        ? ` <a class="backref" href="#${ids[0]}" aria-label="Back to citation">↩</a>`
        : ` <span class="backref">↩ ${ids.map((id, i) => `<a href="#${id}" aria-label="Back to citation ${i + 1}">${letters[i] || i + 1}</a>`).join(" ")}</span>`;
      li.insertAdjacentHTML("beforeend", html);
    });
  }

  /* ---------------- figure / table numbering ---------------- */
  function numberFigures() {
    let f = 0, t = 0;
    document.querySelectorAll("main figure.fig, main figure.tbl").forEach((fig) => {
      const lab = fig.querySelector(":scope > figcaption .fig-label");
      if (!lab) return;
      const isTable = fig.classList.contains("tbl");
      const n = isTable ? ++t : ++f;
      const name = (isTable ? "Table " : "Figure ") + n;
      lab.textContent = name + ".";
      if (!fig.id) fig.id = (isTable ? "table-" : "figure-") + n;
    });
  }

  /* ---------------- copy buttons ---------------- */
  function initCopy() {
    document.querySelectorAll("[data-copy]").forEach((btn) => {
      const target = document.getElementById(btn.dataset.copy);
      const live = btn.parentElement.querySelector("[aria-live]");
      btn.addEventListener("click", async () => {
        const text = target.textContent.trim();
        let ok = false;
        try { await navigator.clipboard.writeText(text); ok = true; } catch (e) {
          const sel = getSelection(), range = document.createRange(); range.selectNodeContents(target); sel.removeAllRanges(); sel.addRange(range);
          try { ok = document.execCommand("copy"); } catch (e2) { ok = false; }
        }
        btn.textContent = ok ? "Copied" : "Select & copy";
        if (live) live.textContent = ok ? "Copied to clipboard" : "Copy failed; text selected";
        setTimeout(() => (btn.textContent = "Copy"), 1800);
      });
    });
  }

  /* ---------------- data loading ---------------- */
  const cache = new Map();
  function load(id) {
    if (!cache.has(id)) {
      cache.set(id, fetch(DATA + encodeURIComponent(id) + ".json", { cache: "no-cache" })
        .then((r) => (r.ok ? r.json() : { exp_id: id, status: "missing" }))
        .catch(() => ({ exp_id: id, status: "missing" })));
    }
    return cache.get(id);
  }
  const hasResults = (d) => d && (d.status === "done" || d.status === "running");

  const pendingHTML = (what) => `<p class="pending" role="status">${what || "Results for this experiment are pending."}</p>`;

  /* ---------- table ---------- */
  function cell(v) {
    if (v === true || v === "✓") return `<span class="tick" aria-label="correct">✓</span>`;
    if (v === false || v === "✗") return `<span class="cross" aria-label="incorrect">✗</span>`;
    if (v == null || v === "") return `<span aria-label="not applicable">—</span>`;
    if (typeof v === "number") return (Number.isInteger(v) ? v.toLocaleString() : String(v)).replace(/^-/, "−");
    return inline(String(v).replace(/(^|[\s(])-(?=\d)/g, "$1−")); // true minus signs
  }
  const isNumLike = (v) => typeof v === "number" || (typeof v === "string" && /\d/.test(v) &&
    /^\s*(?:[-+−]?[\d,.]+%?|\d+\s*\/\s*\d+)(?:\s*\([^)]{0,12}\))?(?:\s*\[[^\]]{0,8}\])?\s*$/.test(v));
  function tableHTML(t, fallbackCaption) {
    if (!t || !t.columns || !t.columns.length || !t.rows || !t.rows.length) return "";
    const blank = (v) => v == null || v === "" || v === "—" || v === "–";
    // a column is numeric when most of its filled cells are; prose cells in it (e.g. a control row) stay left-aligned
    const numCol = t.columns.map((_, ci) => {
      const vals = t.rows.map((r) => r[ci]).filter((v) => !blank(v));
      return ci > 0 && vals.length > 0 && vals.filter(isNumLike).length / vals.length >= 0.6;
    });
    const tdCls = (v, ci) => {
      const c = [];
      if (numCol[ci] && (isNumLike(v) || blank(v))) c.push("num");
      if (typeof v === "string" && v.length > 40) c.push("long");
      else if (typeof v === "string" && v.length <= 26) c.push("short");
      return c.length ? ` class="${c.join(" ")}"` : "";
    };
    const head = t.columns.map((c, ci) => `<th scope="col"${numCol[ci] ? ' class="num"' : ""}>${inline(c)}</th>`).join("");
    const body = t.rows.map((r) => `<tr>${t.columns.map((_, ci) => { const v = r[ci]; return ci === 0 ? `<th scope="row">${cell(v)}</th>` : `<td${tdCls(v, ci)}>${cell(v)}</td>`; }).join("")}</tr>`).join("");
    const cap = t.caption ? inline(t.caption) : esc(fallbackCaption || "");
    const wide = t.columns.length > 6 ? " wide" : "";
    return `<figure class="tbl${wide}"><figcaption>${LABEL}${cap}</figcaption>
      <div class="table-wrap" tabindex="0" role="region" aria-label="${esc(t.caption || fallbackCaption || "Results table")} (scrolls horizontally)">
      <table class="data" style="--cols:${t.columns.length}"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div></figure>`;
  }

  /* ---------- media ---------- */
  function mediaFigure(m) {
    const alt = m.alt || m.caption || "Experiment media";
    const isGif = m.type === "gif";
    const cls = [isGif ? "gif" : "png", m.pixel || isGif ? "pixel" : "smooth"].join(" ");
    return `<figure class="fig media ${isGif ? "gif" : "png"}"><img src="${esc(m.src)}" alt="${esc(alt)}" loading="lazy" decoding="async" class="${cls}">` +
      `<figcaption>${LABEL}${m.caption ? inline(m.caption) : ""}</figcaption></figure>`;
  }
  function mediaHTML(list) {
    if (!list || !list.length) return "";
    const gifs = list.filter((m) => m.type === "gif"), imgs = list.filter((m) => m.type !== "gif");
    let out = "";
    if (gifs.length) {
      const cls = gifs.length === 1 ? "single" : gifs.length === 2 ? "cols-2 wide" : "wide";
      out += `<div class="gallery ${cls}">${gifs.map(mediaFigure).join("")}</div>`;
    }
    if (imgs.length) out += `<div class="gallery single wide">${imgs.map(mediaFigure).join("")}</div>`;
    return out;
  }

  /* ---------- runs ---------- */
  const groupOf = (r) => r.group || String(r.label || "Run").split(/\s+·\s+/)[0];
  function confPct(c) { if (c == null || c === "" || Number.isNaN(+c)) return null; const n = +c; return Math.max(0, Math.min(100, n <= 1 ? n * 100 : n)); }
  function teaser(s) {
    const t = String(s || "").trim(); if (!t) return { lead: "", rest: false };
    const first = t.split(/\n/)[0];
    const lead = first.length > 240 ? first.slice(0, 240).replace(/\s+\S*$/, "") + "…" : first;
    return { lead, rest: t.length > lead.length + 2 };
  }
  const lines = (s) => String(s || "").split("\n").length;
  const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;

  function attemptHTML(r, i) {
    const v = r.correct === true ? `<span class="verdict ok">✓ correct</span>` : r.correct === false ? `<span class="verdict no">✗ incorrect</span>` : `<span class="verdict na">not scored</span>`;
    const cp = confPct(r.confidence);
    const meta = [v];
    if (cp != null) meta.push(`<span>confidence ${cp.toFixed(0)}%</span>`);
    if (r.score != null && r.score !== "") meta.push(`<span>score ${esc(r.score)}</span>`);
    if (r.tokens != null) meta.push(`<span>${esc(Number(r.tokens).toLocaleString())} output tokens</span>`);
    const tz = teaser(r.reasoning_excerpt);
    return `<li class="attempt">
      <div class="attempt-head"><span class="attempt-label">${esc(r.label || "Attempt " + (i + 1))}</span>${meta.join("")}</div>
      <p class="answer"><span class="k">Answer</span>${inline(r.answer ?? "—")}</p>
      ${tz.lead ? `<p class="teaser">${inline(tz.lead)}</p>` : ""}
      ${r.reasoning_excerpt && tz.rest ? `<details class="x"><summary>Show reasoning<span class="meta">${plural(lines(r.reasoning_excerpt), "line")}</span></summary><div class="reasoning">${paras(r.reasoning_excerpt.replace(/\n(?!\n)/g, "\n\n"))}</div></details>` : ""}
      ${r.weights_excerpt ? `<details class="x"><summary>Show weights<span class="meta">${plural(lines(r.weights_excerpt), "line")}</span></summary><pre class="box" tabindex="0" aria-label="Weights excerpt">${esc(r.weights_excerpt)}</pre></details>` : ""}
      ${r.prompt_excerpt ? `<details class="x"><summary>Show prompt<span class="meta">${plural(lines(r.prompt_excerpt), "line")}</span></summary><pre class="box" tabindex="0" aria-label="Prompt excerpt">${esc(r.prompt_excerpt)}</pre></details>` : ""}
      ${r.log_path ? `<p class="attempt-foot"><a href="${REPO}/tree/main/${esc(String(r.log_path).replace(/^\/+/, ""))}">Transcript and result.json</a></p>` : ""}
    </li>`;
  }

  let uidN = 0;
  function runsHTML(d) {
    const runs = d.runs || [];
    if (!runs.length) return "";
    const groups = [];
    const by = new Map();
    runs.forEach((r) => { const g = groupOf(r); if (!by.has(g)) { by.set(g, []); groups.push(g); } by.get(g).push(r); });
    const media = d.media || [];
    const uid = "r" + (++uidN);
    const tally = (rs) => { const sc = rs.filter((r) => r.correct === true || r.correct === false); return sc.length ? { ok: sc.filter((r) => r.correct).length, n: sc.length } : null; };
    const panel = (g) => {
      const rs = by.get(g), t = tally(rs);
      const gm = media.filter((m) => m.group === g);
      return `${t ? `<p class="panel-tally">${t.ok} of ${plural(t.n, "attempt")} correct.</p>` : ""}
        ${mediaHTML(gm)}
        <ol class="attempts">${rs.map((r, i) => attemptHTML(r, i)).join("")}</ol>`;
    };
    const head = `<p class="subhead" id="${uid}-lbl">Individual attempts</p>`;
    if (groups.length === 1) return `<section class="runs" aria-labelledby="${uid}-lbl">${head}${panel(groups[0])}</section>`;
    const tabs = groups.map((g, gi) => {
      const t = tally(by.get(g));
      return `<button class="tab" type="button" role="tab" id="${uid}-t${gi}" aria-controls="${uid}-p${gi}" aria-selected="${gi === 0}" tabindex="${gi === 0 ? 0 : -1}">${esc(g)}${t ? `<span class="tally">${t.ok}/${t.n}</span>` : ""}</button>`;
    }).join("");
    const panels = groups.map((g, gi) => `<div class="tabpanel" role="tabpanel" id="${uid}-p${gi}" aria-labelledby="${uid}-t${gi}" tabindex="0"${gi === 0 ? "" : " hidden"}>${panel(g)}</div>`).join("");
    return `<section class="runs" aria-labelledby="${uid}-lbl">${head}<div class="tablist" role="tablist" aria-label="Network">${tabs}</div>${panels}</section>`;
  }

  function wireTabs(scope) {
    scope.querySelectorAll('[role="tablist"]').forEach((list) => {
      const tabs = [...list.querySelectorAll('[role="tab"]')];
      const select = (t, focus) => {
        tabs.forEach((x) => { const on = x === t; x.setAttribute("aria-selected", on); x.tabIndex = on ? 0 : -1; document.getElementById(x.getAttribute("aria-controls")).hidden = !on; });
        if (focus) t.focus();
      };
      tabs.forEach((t, i) => {
        t.addEventListener("click", () => select(t, false));
        t.addEventListener("keydown", (e) => {
          let j = null;
          if (e.key === "ArrowRight") j = (i + 1) % tabs.length;
          else if (e.key === "ArrowLeft") j = (i - 1 + tabs.length) % tabs.length;
          else if (e.key === "Home") j = 0; else if (e.key === "End") j = tabs.length - 1;
          if (j != null) { e.preventDefault(); select(tabs[j], true); }
        });
      });
    });
  }

  function renderExp(mount, d) {
    const id = mount.dataset.exp;
    mount.dataset.state = d.status || "missing";
    if (!hasResults(d)) { mount.innerHTML = pendingHTML(); return; }
    const ungrouped = (d.media || []).filter((m) => !m.group);
    mount.innerHTML =
      (mount.hasAttribute("data-show-title") && d.title ? `<p class="question">${inline(d.title)}</p>` : "") +
      (d.status === "running" ? `<p class="status" role="status">Partial results; some runs are still in progress.</p>` : "") +
      (d.summary ? `<div class="mount-summary">${paras(d.summary)}</div>` : "") +
      `<div class="charts"></div>` +
      tableHTML(d.table, d.title ? `Results: ${d.title}.` : `Results for ${id}.`) +
      mediaHTML(ungrouped) +
      runsHTML(d) +
      `<p class="mount-foot">All prompts, transcripts and results for this experiment: <a href="${LOGS}/${encodeURIComponent(id)}">logs/${esc(id)}</a>.</p>`;
    const ch = mount.querySelector(".charts");
    (d.charts || []).forEach((spec) => { try { ch.appendChild(window.WSPCharts.render(spec)); } catch (e) { console.warn("chart failed", id, e); } });
    if (!ch.children.length) ch.remove();
    wireTabs(mount);
  }

  /* ---------- scoreboard (data/scoreboard.json) ----------
     {"status","summary","rows":[{"experiment","task","metric","astra","opus","higher_is_better","note"}]} */
  function renderScoreboard(mount, d) {
    if (!d || !(d.status === "done" || d.status === "running") || !(d.rows || []).length) {
      mount.innerHTML = pendingHTML("The comparison table is pending; it fills in one row per experiment as each one finishes.");
      return;
    }
    const num = (v) => (typeof v === "number" ? v : parseFloat(String(v).replace(/[,%]/g, "")));
    const rows = d.rows.map((r) => {
      const a = num(r.astra), o = num(r.opus);
      let wa = false, wo = false;
      if (!Number.isNaN(a) && !Number.isNaN(o) && r.higher_is_better != null && a !== o) { const hi = r.higher_is_better !== false; wa = hi ? a > o : a < o; wo = !wa; }
      return `<tr><th scope="row">${inline(r.experiment)}</th><td>${inline(r.task || "")}</td><td>${inline(r.metric || "")}</td>
        <td class="num${wa ? " win" : ""}">${cell(r.astra)}${wa ? '<span class="sr-only"> (better)</span>' : ""}</td>
        <td class="num${wo ? " win" : ""}">${cell(r.opus)}${wo ? '<span class="sr-only"> (better)</span>' : ""}</td>
        <td class="note">${inline(r.note || "")}</td></tr>`;
    }).join("");
    mount.innerHTML = (d.status === "running" ? `<p class="status" role="status">Partial; some experiments are still running.</p>` : "") +
      (d.summary ? `<div class="mount-summary">${paras(d.summary)}</div>` : "") +
      `<figure class="tbl wide"><figcaption>${LABEL}GPT-6 Astra (as reported in the original post) against Opus 5.5 (this replication). Bold marks the better score where the two are comparable.</figcaption>
      <div class="table-wrap" tabindex="0" role="region" aria-label="Scoreboard (scrolls horizontally)"><table class="data scoreboard" style="--cols:6"><thead><tr><th scope="col">Experiment</th><th scope="col">Task</th><th scope="col">Metric</th><th scope="col" class="num">GPT-6 Astra</th><th scope="col" class="num">Opus 5.5</th><th scope="col">Note</th></tr></thead><tbody>${rows}</tbody></table></div></figure>`;
  }

  /* ---------- failures, aggregated across loaded experiments ---------- */
  function renderFailures(mount, ids) {
    return Promise.all(ids.map(load)).then((all) => {
      const items = [];
      all.forEach((d) => { if (!hasResults(d)) return; (d.runs || []).forEach((r) => { if (r.correct === false) items.push({ d, r }); }); });
      if (!items.length) {
        mount.innerHTML = pendingHTML(all.some(hasResults) ? "No incorrect answers so far." : "Nothing has been published yet.");
        return;
      }
      mount.innerHTML = `<details class="x"><summary>Show all ${items.length} incorrect answers</summary><ul class="fail-list">${items.map(({ d, r }) => {
        const cp = confPct(r.confidence);
        return `<li><span class="where">${esc(d.exp_id)} · ${esc(r.label || "")}${cp != null ? ` · ${cp.toFixed(0)}% confident` : ""}</span>${inline(r.answer || "—")}${r.reasoning_excerpt ? `<span class="why">${inline(teaser(r.reasoning_excerpt).lead)}</span>` : ""}</li>`;
      }).join("")}</ul></details>`;
    });
  }

  function initData() {
    const mounts = [...document.querySelectorAll(".mount[data-exp]")];
    const jobs = mounts.map((m) => { m.setAttribute("aria-busy", "true"); return load(m.dataset.exp).then((d) => { renderExp(m, d); m.removeAttribute("aria-busy"); }); });
    document.querySelectorAll(".mount[data-scoreboard]").forEach((m) => jobs.push(load(m.dataset.scoreboard).then((d) => renderScoreboard(m, d))));
    document.querySelectorAll(".mount[data-failures]").forEach((m) => jobs.push(renderFailures(m, mounts.map((x) => x.dataset.exp))));
    return Promise.allSettled(jobs);
  }

  initTheme(); initSidenotes(); initCitations(); initRail(); initCopy();
  numberFigures();
  initData().then(() => {
    numberFigures();
    // re-apply a #figure-n / #table-n hash once numbered content exists
    if (location.hash && /^#(figure|table)-\d+$/.test(location.hash)) { const t = document.querySelector(location.hash); if (t) t.scrollIntoView(); }
  });
})();
