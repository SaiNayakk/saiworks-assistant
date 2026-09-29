// Chat UI. All server text is inserted with textContent, never innerHTML.
const SAMPLES = {
  "glow-salon": ["How much is a men's haircut?", "keratin kitna ka hai?", "Do you do nail extensions?"],
  "spice-route": ["Are you open for lunch at 4 pm?", "Can I bring my own cake?", "Do you serve beer?"],
};
const $ = (id) => document.getElementById(id);
const log = $("log");

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function adminToken() {
  try { return sessionStorage.getItem("adminToken") || ""; } catch { return ""; }
}

function renderSamples() {
  const box = $("samples");
  box.replaceChildren();
  (SAMPLES[$("tenant").value] || []).forEach((s) => {
    const b = el("button", null, s);
    b.type = "button";
    b.onclick = () => { $("q").value = s; $("f").requestSubmit(); };
    box.append(b);
  });
}

async function loadTenants() {
  const res = await fetch("/api/tenants");
  const tenants = await res.json();
  tenants.forEach((t) => { const o = el("option", null, t.name); o.value = t.id; $("tenant").append(o); });
  renderSamples();
}

function debugText(d) {
  const rows = d.retrieved.map((h) =>
    `#${h.chunk_id} ${h.source} › ${h.heading}\n   rrf=${h.score.toFixed(4)} vec_rank=${h.vector_rank ?? "-"} sim=${h.vector_sim?.toFixed(3) ?? "-"} kw_rank=${h.keyword_rank ?? "-"}`);
  return [
    `timings ms: ${JSON.stringify(d.timings_ms)}`,
    `tokens: in=${d.usage.input_tokens} out=${d.usage.output_tokens}`,
    d.abstain_reason ? `${d.refused ? "refused" : "abstained"}: ${d.abstain_reason}` : "",
    rows.length ? "retrieved:" : "",
    ...rows,
  ].filter(Boolean).join("\n");
}

$("f").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("q").value.trim();
  if (!question) return;
  log.append(el("div", "msg me", question));
  $("q").value = "";
  $("send").disabled = true;
  const bubble = el("div", "msg", "…");
  log.append(bubble);
  bubble.scrollIntoView({ block: "end" });
  try {
    const headers = { "Content-Type": "application/json" };
    if ($("showDebug").checked && adminToken()) headers.Authorization = `Bearer ${adminToken()}`;
    const res = await fetch("/api/chat", { method: "POST", headers, body: JSON.stringify({ tenant: $("tenant").value, question }) });
    const d = await res.json();
    if (!res.ok) { bubble.textContent = d.error || "Something went wrong."; return; }
    bubble.textContent = d.answer;
    if (!d.answerable) bubble.classList.add("abstain");
    if (d.citations.length) {
      const c = el("div", "cites");
      d.citations.forEach((x) => c.append(el("span", "cite", `${x.source} › ${x.heading}`)));
      bubble.append(c);
    }
    if (d.debug) bubble.append(el("div", "debug", debugText(d.debug)));
  } catch {
    bubble.textContent = "Network error — please try again.";
  } finally {
    $("send").disabled = false;
    bubble.scrollIntoView({ block: "end" });
    $("q").focus();
  }
});

$("tenant").addEventListener("change", renderSamples);
$("showDebug").addEventListener("change", () => {
  if ($("showDebug").checked && !adminToken()) {
    const t = prompt("Owner debug needs the ADMIN_TOKEN:");
    if (!t) { $("showDebug").checked = false; return; }
    try { sessionStorage.setItem("adminToken", t); } catch {}
  }
});
loadTenants();
