/* 确定性聚合执行器前端：原生 JS + Canvas，无外部依赖 */

const $ = (id) => document.getElementById(id);
let META = null;

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

function fmt(x, digits = 6) {
  if (x === null || x === undefined) return "—";
  if (Number.isNaN(x)) return "NaN";
  if (!Number.isFinite(x)) return String(x);
  if (x === 0) return "0";
  const ax = Math.abs(x);
  if (ax >= 1e6 || ax < 1e-4) return x.toExponential(6);
  return x.toFixed(digits);
}

function methodTag(m) {
  if (m.exact) return '<span class="tag exact">精确/顺序无关</span>';
  if (m.deterministic) return '<span class="tag det">逐位确定</span>';
  return '<span class="tag flaky">顺序敏感</span>';
}

async function init() {
  META = await fetchJSON("/api/meta");
  const dsSel = $("dataset");
  dsSel.innerHTML = META.datasets.map(d =>
    `<option value="${d.name}">${d.name} — ${d.description}（n=${d.n}）</option>`).join("");
  dsSel.onchange = renderMethodList;
  $("kind").onchange = renderMethodList;
  renderMethodList();
  document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => {
    document.querySelectorAll(".tabs button").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    $("tab-single").classList.toggle("hidden", b.dataset.tab !== "single");
    $("tab-sweep").classList.toggle("hidden", b.dataset.tab !== "sweep");
  });
  $("run").onclick = runSingle;
  $("sweep").onclick = runSweep;
  updateDsNote();
  dsSel.addEventListener("change", updateDsNote);
}

function updateDsNote() {
  const d = META.datasets.find(x => x.name === $("dataset").value);
  $("ds-note").innerHTML = d ? `量级范围 [${fmt(d.finite_min)}, ${fmt(d.finite_max)}]，
    空值 ${d.null_count} 个，定点 scale=${d.scale}<br><span style="color:var(--warn)">${d.note}</span>` : "";
}

function currentMethods() {
  return $("kind").value === "sum" ? META.sum_methods : META.var_methods;
}

function renderMethodList() {
  const ms = currentMethods();
  const list = $("methods");
  list.innerHTML = ms.map((m, i) => `
    <label><input type="radio" name="m" value="${m.key}" ${i === 0 ? "checked" : ""}>
      <span>${m.label} ${methodTag(m)}</span></label>`).join("");
  list.querySelectorAll("input").forEach(el => el.onchange = showMethodMeta);
  showMethodMeta();
}

function showMethodMeta() {
  const key = document.querySelector('input[name=m]:checked').value;
  const m = currentMethods().find(x => x.key === key);
  $("method-meta").innerHTML = `
    <div>💰 代价：${m.cost_note}</div>
    <div>✅ 保证：${m.guarantee_note}</div>
    <div>⚠️ 范围：${m.range_note}</div>`;
}

async function loadTruth() {
  const t = await fetchJSON(`/api/truth?dataset=${$("dataset").value}`);
  $("truth").innerHTML = `
    有效 n=<b>${t.n}</b><br>
    SUM = <b>${t.sum}</b><br>
    SUM→float64 = <b>${fmt(t.sum_f64, 12)}</b><br>
    MEAN = ${t.mean} → ${fmt(t.mean_f64, 12)}<br>
    VAR  = ${t.variance} → ${fmt(t.variance_f64, 12)}<br>
    STD  = ${t.stddev} → ${fmt(t.stddev_f64, 12)}`;
  return t;
}

async function runSingle() {
  $("spin").style.display = "block";
  try {
    const method = document.querySelector('input[name=m]:checked').value;
    const [truth] = await Promise.all([loadTruth()]);
    const res = await fetchJSON("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset: $("dataset").value,
        kind: $("kind").value,
        method,
        trials: Number($("trials").value),
        max_partitions: Number($("maxparts").value),
        threaded: $("threaded").checked,
      }),
    });
    renderResult(res, truth);
  } catch (e) {
    $("verdict").innerHTML = `<div class="err">请求失败：${e.message}</div>`;
  } finally {
    $("spin").style.display = "none";
  }
}

function renderResult(res, truth) {
  const d = res.distribution;
  const bitOk = d.distinct === 1 && d.error_count === 0;
  const truthOk = bitOk && d.results.length && d.results[0] === res.truth_f64;
  const kpiCls = bitOk ? "ok" : "bad";
  $("kpis").innerHTML = `
    <div class="card"><div class="v ${kpiCls}">${d.distinct}</div>
      <div class="k">不同 bit 模式数（${res.trials} 次运行）</div></div>
    <div class="card"><div class="v ${bitOk ? "ok" : "bad"}">${bitOk ? "是" : "否"}</div>
      <div class="k">逐位完全相同</div></div>
    <div class="card"><div class="v ${d.max_ulps !== null && d.max_ulps <= 1 ? "ok" : "warn"}">
      ${d.max_ulps === null ? "—" : fmt(d.max_ulps, 2)}</div>
      <div class="k">最大偏差（ULP）</div></div>
    <div class="card"><div class="v">${(res.elapsed_ms / 1000).toFixed(2)}s</div>
      <div class="k">总耗时（${res.threaded ? "线程池" : "顺序模拟"}）</div></div>`;

  let verdict;
  if (res.error_count > 0) {
    verdict = `<div class="err"><b>守卫触发 ${res.error_count} 次（设计如此，拒绝静默出错）：</b><br>` +
      res.errors.map(e => `#${e.trial}: ${e.error}`).join("<br>") + "</div>";
  } else if (bitOk && truthOk) {
    verdict = `<div class="okbox">✅ ${res.trials} 种随机切分/合并顺序下结果逐位相同，
      且等于十进制真值的最近 float64 —— 对账系统不会再报“数据被改”。</div>`;
  } else if (bitOk) {
    verdict = `<div class="okbox" style="border-color:var(--warn);color:var(--warn);background:rgba(255,180,84,.08)">
      ✅ 顺序无关已达成（每次相同），但与十进制真值存在输入侧固有偏差：
      float64 存入时就已舍入，求和算法无法恢复。要与真值逐位相同需定点/Decimal。</div>`;
  } else {
    verdict = `<div class="err">❌ 复现了生产事故：同一数据 ${res.trials} 次运行出现
      <b>${d.distinct}</b> 种不同结果，最大偏差 ${fmt(d.max_abs_err, 8)}
      （≈${fmt(d.max_ulps, 1)} ULP）。并行引擎每次切分/合并顺序不同，
      对账系统的逐位比对必然误报。</div>`;
  }
  $("verdict").innerHTML = verdict;
  if (res.mean_info) {
    const mi = res.mean_info;
    const miOk = mi.distinct === 1 && mi.sample === mi.truth_f64;
    $("verdict").innerHTML += `<div class="note" style="margin-top:8px">
      平均值（和/n）：${mi.distinct} 种结果，样本=${fmt(mi.sample, 10)}，
      真值=${fmt(mi.truth_f64, 10)}，
      <b style="color:${miOk ? "var(--ok)" : "var(--warn)"}">
      ${miOk ? "均值同样逐位确定" : "均值随求和结果跳动"}</b></div>`;
  }

  // bit 模式表
  const ulp = Math.ulp(res.truth_f64) || Number.MIN_VALUE;
  $("patterns").querySelector("tbody").innerHTML = d.bit_patterns.map(p => {
    const v = p.value === "nan" ? NaN : Number(p.value);
    const err = Number.isFinite(v) ? Math.abs(v - res.truth_f64) : null;
    return `<tr><td class="mono">${p.value}</td><td>${p.count}</td>
      <td class="mono">${err === null ? "—" : fmt(err, 10)}</td>
      <td class="mono">${err === null ? "—" : fmt(err / ulp, 2)}</td></tr>`;
  }).join("");

  drawChart(d.results, res.truth_f64);
  $("run-note").textContent =
    `每次运行分区数在 2~${document.querySelector("#maxparts").value} 间随机，合并顺序为随机排列；` +
    `样本：${JSON.stringify(res.orders_sample[0])}`;
}

function drawChart(results, truth) {
  const cv = $("chart");
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = 260;
  cv.width = W * dpr; cv.height = H * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);
  const pad = { l: 78, r: 20, t: 18, b: 34 };
  const finite = results.filter(Number.isFinite);
  if (!finite.length) {
    ctx.fillStyle = "#ff6b6b"; ctx.font = "13px sans-serif";
    ctx.fillText("全部运行返回 NaN/Inf（检查特殊值语义）", pad.l, H / 2);
    return;
  }
  let lo = Math.min(...finite, truth), hi = Math.max(...finite, truth);
  if (lo === hi) { lo -= 1; hi += 1; }
  const span = hi - lo;
  lo -= span * 0.1; hi += span * 0.1;
  const X = (i) => pad.l + i * (W - pad.l - pad.r) / Math.max(results.length - 1, 1);
  const Y = (v) => pad.t + (hi - v) * (H - pad.t - pad.b) / (hi - lo);

  // 网格 + y 轴标注
  ctx.strokeStyle = "#2a3645"; ctx.fillStyle = "#8595a7"; ctx.font = "11px monospace";
  for (let g = 0; g <= 4; g++) {
    const v = lo + (hi - lo) * g / 4, y = Y(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillText(fmt(v, 4), 6, y + 4);
  }
  // 真值红线
  ctx.strokeStyle = "#ff6b6b"; ctx.setLineDash([6, 5]);
  ctx.beginPath(); ctx.moveTo(pad.l, Y(truth)); ctx.lineTo(W - pad.r, Y(truth)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#ff6b6b"; ctx.fillText("truth", W - pad.r - 42, Y(truth) - 5);
  // 各次运行散点
  results.forEach((v, i) => {
    if (!Number.isFinite(v)) return;
    const distinct = results.some(x => x !== v);
    ctx.fillStyle = distinct ? "#ffb454" : "#4da3ff";
    ctx.beginPath(); ctx.arc(X(i), Y(v), 2.2, 0, Math.PI * 2); ctx.fill();
  });
  ctx.fillStyle = "#8595a7";
  ctx.fillText("试验序号 →", W - 90, H - 10);
}

async function runSweep() {
  $("spin").style.display = "block";
  try {
    const report = await fetchJSON("/api/sweep", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        trials: Number($("trials").value),
        threaded: $("threaded").checked,
      }),
    });
    renderSweep(report);
    document.querySelector('.tabs button[data-tab="sweep"]').click();
  } catch (e) {
    alert("全套对照失败：" + e.message);
  } finally {
    $("spin").style.display = "none";
  }
}

function renderSweep(rep) {
  const rows = rep.rows;
  const exact = rows.filter(r => r.error === null && r.bit_identical &&
    r.matches_truth_f64).length;
  const flaky = rows.filter(r => r.error === null && !r.bit_identical).length;
  const guards = rows.filter(r => r.error !== null).length;
  $("sweep-summary").innerHTML = `
    <div class="kpi" style="margin-bottom:6px">
      <div class="card"><div class="v ok">${exact}</div><div class="k">逐位相同且等于真值</div></div>
      <div class="card"><div class="v bad">${flaky}</div><div class="k">顺序敏感组合</div></div>
      <div class="card"><div class="v warn">${guards}</div><div class="k">守卫触发（溢出/越界）</div></div>
      <div class="card"><div class="v">${rep.trials}</div><div class="k">每组合随机顺序次数</div></div>
    </div>`;
  $("sweep-table").querySelector("tbody").innerHTML = rows.map(r => `
    <tr>
      <td>${r.dataset}</td><td>${r.method}</td><td>${r.kind}</td>
      <td>${r.distinct_results}</td>
      <td style="color:${r.bit_identical ? "var(--ok)" : "var(--bad)"}">${r.bit_identical ? "是" : "否"}</td>
      <td>${r.matches_truth_f64 === null ? "—" : r.matches_truth_f64 ? "✅" : "—"}</td>
      <td class="mono">${fmt(r.max_abs_err, 8)}</td>
      <td class="mono">${fmt(r.max_ulps, 2)}</td>
      <td>${r.time_ms === null ? "—" : r.time_ms.toFixed(1)}</td>
      <td>${r.speedup_vs_naive === null ? "—" : r.speedup_vs_naive.toFixed(2) + "x"}</td>
      <td>${r.peak_mem_kb === null ? "—" : r.peak_mem_kb.toFixed(1)}</td>
      <td style="max-width:260px;color:var(--warn);font-size:11px">${r.error || ""}</td>
    </tr>`).join("");
}

init().catch(e => document.body.insertAdjacentHTML("afterbegin",
  `<div class="err">初始化失败：${e.message}</div>`));
