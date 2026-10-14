/* Verge dashboard client.
 * Fetches review priorities (GeoJSON) + the recommendation report from the
 * FastAPI backend (backend/app/main.py) and renders them as a Leaflet map +
 * report viewer, plus the location-linked "Analyze a video" flow:
 * location -> upload -> real progress -> evidence replay -> metrics ->
 * site risk update -> PDF evidence report.
 * No build step — plain HTML/JS so it's trivial to open and inspect.
 */

const TIER_COLORS = {
  Critical: "#c1442e",
  High: "#dd8a2e",
  Moderate: "#d8b73d",
  Low: "#3f8f5c",
};
const SEVERITY_COLORS = { High: "#c1442e", Moderate: "#dd8a2e", Low: "#3f8f5c" };
const OBSERVED_COLOR = "#12463a";
const ALLOWED_VIDEO = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"];

let map;
let markersLayer;

const state = {
  features: [],
  selected: null,        // {site_id?, name, lat, lon, pinned, isNew}
  selectionLayer: null,
  pinMarker: null,
  lastUpdate: null,      // site_update from the latest location-linked analysis
  analysis: null,
  activeEvent: 0,
  player: null,
  polling: null,
  thresholds: { distance_m: 8, ttc_s: 3 },
};

/* ---- small helpers ---------------------------------------------------- */

function apiBase() {
  return document.getElementById("api-base").value.trim().replace(/\/$/, "");
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmtTime(t) {
  if (t === null || t === undefined || Number.isNaN(Number(t))) return "--:--";
  const total = Math.max(0, Number(t));
  const m = Math.floor(total / 60);
  const s = total - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(1).padStart(4, "0")}`;
}

function num(v, digits = 2) {
  return v === null || v === undefined ? "—" : Number(v).toFixed(digits);
}

function bullets(items) {
  if (!items || !items.length) return "";
  return `<ul class="bullets">${items.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`;
}

/* fetch() with a readable error for the three things that go wrong in a demo:
 * API not running, API returned an error, request timed out. */
async function api(path, opts = {}, timeoutMs = 0) {
  const ctrl = new AbortController();
  const timer = timeoutMs ? setTimeout(() => ctrl.abort(), timeoutMs) : null;
  let res;
  try {
    res = await fetch(`${apiBase()}${path}`, { ...opts, signal: ctrl.signal });
  } catch (err) {
    const e = new Error(
      err.name === "AbortError"
        ? `The API at ${apiBase()} did not respond in time.`
        : `Can't reach the Verge API at ${apiBase()}. Start it with start_dashboard.bat / ./start_dashboard.sh, or: uvicorn backend.app.main:app --port 8000`
    );
    e.offline = true;
    throw e;
  } finally {
    if (timer) clearTimeout(timer);
  }
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body && body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* not JSON */ }
    const e = new Error(detail);
    e.status = res.status;
    throw e;
  }
  return res;
}

function noticeHtml(kind, title, message, extra = "") {
  return `<div class="notice notice-${kind}"><strong>${esc(title)}</strong><p>${esc(message)}</p>${extra}</div>`;
}

/* ---- API status --------------------------------------------------------- */

async function checkApi() {
  const el = document.getElementById("api-status");
  try {
    const res = await api("/api/health", {}, 4000);
    const body = await res.json();
    if (body.thresholds) state.thresholds = body.thresholds;
    if (body.risk_tiers) setLegendRanges(body.risk_tiers);
    el.className = "api-status ok";
    el.querySelector(".api-label").textContent = "API connected";
    return true;
  } catch (_) {
    el.className = "api-status down";
    el.querySelector(".api-label").textContent = "API offline";
    return false;
  }
}

/* Legend tooltips show the actual score bands the backend uses (config.RISK_TIERS). */
function setLegendRanges(tiers) {
  const sorted = tiers.slice().sort((a, b) => a.min_score - b.min_score);
  sorted.forEach((t, i) => {
    const el = document.querySelector(`.legend [data-tier="${t.tier}"]`);
    if (!el) return;
    const next = sorted[i + 1];
    el.title = next
      ? (t.min_score > 0
        ? `Review priority ${t.min_score.toFixed(2)} to below ${next.min_score.toFixed(2)}`
        : `Review priority below ${next.min_score.toFixed(2)}`)
      : `Review priority ${t.min_score.toFixed(2)} and above`;
  });
}

/* ---- map ------------------------------------------------------------------ */

function initMap() {
  map = L.map("map").setView([18.5204, 73.8567], 12);
  const tiles = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19,
  }).addTo(map);
  let tileErrors = 0;
  tiles.on("tileerror", () => {
    tileErrors += 1;
    if (tileErrors === 3) {
      showMapHint("Map tiles unavailable (offline?). Hotspot positions and scores are still accurate.", "warn");
    }
  });
  markersLayer = L.layerGroup().addTo(map);

  map.on("click", (e) => {
    if (!isTabActive("analyze")) return;
    setPinnedLocation(e.latlng.lat, e.latlng.lng);
  });
  map.on("popupopen", (e) => {
    const btn = e.popup.getElement().querySelector("[data-analyze-site]");
    if (btn) btn.addEventListener("click", () => {
      selectSite(btn.dataset.analyzeSite);
      activateTab("analyze");
      map.closePopup();
    });
  });
}

function showMapHint(text, kind = "info") {
  const el = document.getElementById("map-hint");
  if (!text) { el.hidden = true; return; }
  el.hidden = false;
  el.className = `map-hint map-hint-${kind}`;
  el.textContent = text;
}

function warnIfOpenedAsFile() {
  if (location.protocol !== "file:") return;
  const banner = document.createElement("div");
  banner.className = "file-protocol-banner";
  banner.innerHTML = `
    <strong>Map tiles won't load like this.</strong>
    You opened this file directly (<code>file://…</code>), but OpenStreetMap's
    tile servers require pages to be served over HTTP. Run
    <code>start_dashboard.bat</code> (Windows) or
    <code>./start_dashboard.sh</code> (Mac/Linux) in the project folder,
    or see <code>README.md</code> — it takes 10 seconds to fix.
  `;
  document.body.prepend(banner);
}

function markerFor(feature) {
  const p = feature.properties;
  const [lon, lat] = feature.geometry.coordinates;
  const color = TIER_COLORS[p.risk_tier] || TIER_COLORS.Low;
  const radius = 8 + p.risk_score * 14;

  const critical = p.risk_tier === "Critical";
  const marker = L.circleMarker([lat, lon], {
    radius,
    color,
    weight: critical ? 3 : 2,
    fillColor: color,
    fillOpacity: critical ? 0.6 : 0.45,
  });
  marker.siteId = p.site_id;
  // top padding keeps an opened popup clear of the map's search box
  marker.bindPopup(popupHtml(p), { maxWidth: 320, autoPanPaddingTopLeft: [20, 84], autoPanPaddingBottomRight: [20, 20] });
  marker.on("click", () => {
    setAssistantContext(p.site_id);
    if (isTabActive("analyze")) selectSite(p.site_id, { keepPopup: true });
  });
  return marker;
}

/* Dashed outer ring = this site has AI-observed conflict evidence. */
function observedRingFor(feature) {
  const p = feature.properties;
  if (!p.near_miss_count) return null;
  const [lon, lat] = feature.geometry.coordinates;
  return L.circleMarker([lat, lon], {
    radius: 8 + p.risk_score * 14 + 5,
    color: OBSERVED_COLOR,
    weight: 1.5,
    dashArray: "3 3",
    fill: false,
    interactive: false,
  });
}

/* Map popup: title + risk badge, the key figures and the historical vs
 * AI-observed split, nothing more. Evidence, reasons and recommendations are
 * in View Analysis, the Recommendation Report and the PDF. */
function popupHtml(p) {
  return `
    <div class="popup">
      <div class="popup-head">
        <strong>#${p.rank} ${esc(p.name)}</strong>
        <span class="score-badge" style="background:${TIER_COLORS[p.risk_tier]}">${esc(p.risk_tier.toUpperCase())} · ${p.risk_score.toFixed(2)}</span>
      </div>
      ${popupFactsHtml(p)}
      ${scoreSplitHtml(p)}
    </div>
  `;
}

/* Risk level + the data behind it, straight from the hotspot record. */
function popupFactsHtml(p) {
  const noRecord = p.has_historical_record === false;
  const years = p.years && p.years !== "n/a" ? ` (${esc(p.years)})` : "";
  const crashes = (p.evidence || {}).crashes || 0;
  return `
    <dl class="popup-facts">
      <dt>Risk level</dt><dd><b class="tier-text" style="color:${TIER_COLORS[p.risk_tier]}">${esc(p.risk_tier.toUpperCase())}</b></dd>
      <dt>Risk score</dt><dd>${p.risk_score.toFixed(2)}</dd>
      <dt>Crashes${years}</dt><dd>${noRecord ? "no record" : p.accidents_count}</dd>
      <dt>Fatalities</dt><dd>${noRecord ? "—" : p.fatalities}</dd>
      <dt>AI near-misses</dt><dd>${(p.near_miss_count || 0) - crashes}</dd>
      ${crashes ? `<dt>AI-observed crashes</dt><dd class="crash-count">${crashes}</dd>` : ""}
    </dl>`;
}

/* "0.38 = 0.03 historical + 0.35 AI-observed" as a thin two-part bar. */
function scoreSplitHtml(p) {
  if (p.historical_component === undefined) return "";
  const h = Math.max(0, p.historical_component);
  const o = Math.max(0, p.observed_component);
  return `
    <div class="split" title="Review priority ${p.risk_score.toFixed(2)} = historical ${h.toFixed(2)} + AI-observed ${o.toFixed(2)}">
      <div class="split-bar">
        <span class="split-hist" style="width:${(h * 100).toFixed(1)}%"></span>
        <span class="split-obs" style="width:${(o * 100).toFixed(1)}%"></span>
      </div>
      <div class="split-legend"><span><i class="k-hist"></i>Historical ${h.toFixed(2)}</span><span><i class="k-obs"></i>AI-observed ${o.toFixed(2)}</span></div>
    </div>`;
}

function rankChangeFor(siteId) {
  const u = state.lastUpdate;
  if (!u) return "";
  const rc = (u.rank_changes || []).find((r) => r.site_id === siteId);
  if (!rc) return siteId === u.site_id ? `<span class="chip chip-updated">Updated</span>` : "";
  const up = rc.after < rc.before;
  return `<span class="chip ${up ? "chip-up" : "chip-down"}" title="Rank before the latest analysis: #${rc.before}">${up ? "▲" : "▼"} from #${rc.before}</span>`;
}

function renderHotspotList(features) {
  const list = document.getElementById("hotspot-list");
  const summary = document.getElementById("hotspot-summary");
  if (!features.length) {
    list.innerHTML = `<li class="empty-state">No hotspots yet — click "Re-run pipeline".</li>`;
    summary.innerHTML = "";
    return;
  }
  const conflicts = features.reduce((a, f) => a + (f.properties.near_miss_count || 0), 0);
  const uploads = features.reduce((a, f) => a + ((f.properties.evidence || {}).sources || {}).video_upload || 0, 0);
  summary.innerHTML = `
    <span><b>${features.length}</b> locations</span>
    <span><b>${conflicts}</b> AI-detected conflict events${uploads ? ` (${uploads} from uploaded video)` : ""}</span>
    <span class="muted">Historical figures: synthetic sample data</span>`;

  list.innerHTML = "";
  features
    .slice()
    .sort((a, b) => a.properties.rank - b.properties.rank)
    .forEach((f) => {
      const p = f.properties;
      const li = document.createElement("li");
      const isUpdated = state.lastUpdate && state.lastUpdate.site_id === p.site_id;
      const isSelected = state.selected && state.selected.site_id === p.site_id;
      li.className = `hotspot-card tier-${p.risk_tier}${isUpdated ? " is-updated" : ""}${isSelected ? " is-selected" : ""}${state.searchedSiteId === p.site_id ? " is-searched" : ""}`;
      li.dataset.siteId = p.site_id;
      const crashes = p.has_historical_record === false ? "no record" : p.accidents_count;
      li.innerHTML = `
        <div class="card-top">
          <h3>#${p.rank} ${esc(p.name)}</h3>
          ${rankChangeFor(p.site_id)}
        </div>
        <div class="meta">
          <span class="score-badge" style="background:${TIER_COLORS[p.risk_tier]}">${esc(p.risk_tier)} · ${p.risk_score.toFixed(2)}</span>
          <span>Crashes: ${crashes}</span>
          <span>Fatalities: ${p.has_historical_record === false ? "—" : p.fatalities}</span>
          <span class="${p.near_miss_count ? "has-obs" : ""}">AI conflicts: ${p.near_miss_count}</span>
          ${((p.evidence || {}).crashes || 0) ? `<span class="has-crash">AI crashes: ${p.evidence.crashes}</span>` : ""}
          ${p.pinned ? `<span class="chip chip-pin">Map pin</span>` : ""}
        </div>
        ${scoreSplitHtml(p)}
        <div class="card-actions">
          <button type="button" class="link-btn" data-analyze="${esc(p.site_id)}">Analyze video here</button>
        </div>
      `;
      li.addEventListener("click", (ev) => {
        if (ev.target.closest("[data-analyze]")) {
          selectSite(p.site_id);
          activateTab("analyze");
          return;
        }
        focusSite(p.site_id, true);
      });
      list.appendChild(li);
    });
}

function focusSite(siteId, openPopup) {
  const f = state.features.find((x) => x.properties.site_id === siteId);
  if (!f) return;
  setAssistantContext(siteId);
  const [lon, lat] = f.geometry.coordinates;
  map.setView([lat, lon], Math.max(map.getZoom(), 14));
  if (openPopup) {
    markersLayer.eachLayer((m) => { if (m.siteId === siteId) m.openPopup(); });
  }
}

async function loadHotspots() {
  const listEl = document.getElementById("hotspot-list");
  try {
    const res = await api("/api/hotspots", {}, 15000);
    const geojson = await res.json();
    const firstLoad = !state.features.length;
    state.features = geojson.features;
    markersLayer.clearLayers();
    geojson.features.slice().sort((a, b) => a.properties.risk_score - b.properties.risk_score).forEach((f) => {
      const ring = observedRingFor(f);
      if (ring) ring.addTo(markersLayer);
      markerFor(f).addTo(markersLayer);
    });
    if (firstLoad && geojson.features.length) {
      const pts = geojson.features.map((f) => [f.geometry.coordinates[1], f.geometry.coordinates[0]]);
      map.fitBounds(L.latLngBounds(pts).pad(0.3));
    }
    renderHotspotList(geojson.features);
    renderReport();
    fillSiteOptions();
    if (state.selected && state.selected.site_id) {
      // selection may have been removed (e.g. evidence cleared) or re-ranked
      const still = state.features.find((f) => f.properties.site_id === state.selected.site_id);
      if (still) selectSite(state.selected.site_id, { silent: true }); else clearSelection();
    }
    highlightUpdatedSite();
  } catch (err) {
    listEl.innerHTML = `<li class="empty-state">Couldn't load hotspots (${esc(err.message)}). Is the API running at ${esc(apiBase())}? Try "Re-run pipeline" or start it with:<br/><code>uvicorn backend.app.main:app --reload --port 8000</code></li>`;
  }
}

function highlightUpdatedSite() {
  const u = state.lastUpdate;
  if (!u || !u.after) return;
  markersLayer.eachLayer((m) => {
    if (m.siteId !== u.site_id) return;
    m.setStyle({ weight: 4, color: "#10231b" });
    const changed = u.delta !== null && u.delta !== undefined && Math.abs(u.delta) > 0.0005;
    const rank = u.before && u.before.rank !== u.after.rank ? `#${u.before.rank} → #${u.after.rank}` : `#${u.after.rank}`;
    const label = changed || (u.before && u.before.rank !== u.after.rank)
      ? `Updated ${rank} · ${u.delta >= 0 ? "+" : ""}${u.delta.toFixed(2)}`
      : "Evidence recorded · no change";
    m.bindTooltip(label, { permanent: true, direction: "top", className: "update-tip", offset: [0, -10] }).openTooltip();
  });
}

/* ---- report tab ------------------------------------------------------------ */

async function loadReport() {
  const el = document.getElementById("report-content");
  try {
    const res = await api("/api/report", {}, 15000);
    state.reportMd = await res.text();
    renderReport();
  } catch (err) {
    el.innerHTML = `<p class="empty-state">Couldn't load report (${esc(err.message)}).</p>`;
  }
}

/* Report tab: the report's title, meta line and disclaimer, then ONLY the
 * ranked table (same columns as the report) with a View Analysis action per
 * row. The detailed location briefs stay in the downloadable .md report and
 * open per hotspot in the analysis dialog. */
function renderReport() {
  const el = document.getElementById("report-content");
  if (!state.reportMd) return;
  const head = state.reportMd.split(/^## Ranked locations/m)[0];
  const rows = state.features.slice().sort((a, b) => a.properties.rank - b.properties.rank).map((f) => {
    const p = f.properties;
    const hist = p.has_historical_record !== false;
    return `<tr>
      <td>${p.rank}</td>
      <td class="rt-name">${esc(p.name)}</td>
      <td><span class="score-badge" style="background:${TIER_COLORS[p.risk_tier]}">${esc(p.risk_tier)} · ${p.risk_score.toFixed(2)}</span></td>
      <td>${hist ? p.accidents_count : "n/a"}</td>
      <td>${hist ? p.fatalities : "n/a"}</td>
      <td>${p.near_miss_count}</td>
      <td><button type="button" class="table-btn" data-view-analysis="${esc(p.site_id)}">View Analysis</button></td>
    </tr>`;
  }).join("");
  el.innerHTML = `${tinyMarkdownToHtml(head)}
    <h2>Ranked locations</h2>
    ${rows ? `<table class="rank-table">
      <thead><tr><th>#</th><th>Location</th><th>Priority</th><th>Crashes</th><th>Fatalities</th><th>AI conflicts</th><th></th></tr></thead>
      <tbody>${rows}</tbody></table>` : `<p class="empty-state">Loading hotspots…</p>`}`;
  el.querySelectorAll("[data-view-analysis]").forEach((btn) =>
    btn.addEventListener("click", () => openAnalysis(btn.dataset.viewAnalysis)));
}

/* ---- hotspot analysis dialog (one reusable dialog, filled per hotspot) ------ */

const TIER_WORDS = {
  en: { Low: "Low", Moderate: "Moderate", High: "High", Critical: "Critical" },
  hi: { Low: "कम", Moderate: "मध्यम", High: "उच्च", Critical: "गंभीर" },
  mr: { Low: "कमी", Moderate: "मध्यम", High: "उच्च", Critical: "गंभीर" },
};
/* The fixed set of review actions report.site_brief() can produce, translated. */
const REVIEW_I18N = {
  "Schedule a formal road-safety audit (geometry, signage, lighting)": {
    hi: "औपचारिक सड़क-सुरक्षा ऑडिट (ज्यामिति, संकेत, प्रकाश व्यवस्था) कराएँ",
    mr: "औपचारिक रस्ते-सुरक्षा ऑडिट (रचना, फलक, प्रकाशयोजना) करावे" },
  "Assess speed management on the approaches": {
    hi: "पहुँच मार्गों पर गति प्रबंधन का आकलन करें", mr: "जवळच्या मार्गांवरील वेग व्यवस्थापनाचे मूल्यांकन करावे" },
  "Inspect crossing visibility and pedestrian sight lines": {
    hi: "क्रॉसिंग की दृश्यता और पैदल यात्रियों की दृष्टि रेखाओं की जाँच करें", mr: "क्रॉसिंगची दृश्यता आणि पादचाऱ्यांची दृष्टिरेषा तपासावी" },
  "Review signal timing and pedestrian protection at the crossing": {
    hi: "क्रॉसिंग पर सिग्नल समय और पैदल यात्री सुरक्षा की समीक्षा करें", mr: "क्रॉसिंगवरील सिग्नलची वेळ आणि पादचारी सुरक्षेचा आढावा घ्यावा" },
  "Review junction priority and turning movements": {
    hi: "जंक्शन प्राथमिकता और मुड़ने वाले वाहनों की समीक्षा करें", mr: "जंक्शनवरील प्राधान्यक्रम आणि वळणाऱ्या वाहनांचा आढावा घ्यावा" },
  "Check approach speeds and stopping sight distance": {
    hi: "पहुँचने की गति और रुकने की दृष्टि दूरी जाँचें", mr: "जवळ येणारा वेग आणि थांबण्याचे दृष्टी-अंतर तपासावे" },
  "Include in the next routine road-safety review cycle": {
    hi: "अगले नियमित सड़क-सुरक्षा समीक्षा चक्र में शामिल करें", mr: "पुढील नियमित रस्ते-सुरक्षा आढाव्यात समाविष्ट करावे" },
};

function lowerFirst(s) { return s ? s.charAt(0).toLowerCase() + s.slice(1) : s; }

/* 3-5 sentence summary of one hotspot, generated only from its record and its
 * existing brief. Decision-support wording; never a crash prediction. */
function hotspotSummary(p, lang = "en") {
  // English: the backend's summary (hotspot_insight.hotspot_summary), the same
  // text the assistant and the site PDF use. Hindi / Marathi render the same
  // facts from the same record.
  if (lang === "en" && p.summary) return p.summary;
  const b = p.brief || { review: [] };
  const hist = p.has_historical_record !== false;
  const n = p.near_miss_count || 0;
  const nc = ((p.evidence || {}).crashes) || 0;
  const tier = TIER_WORDS[lang][p.risk_tier] || p.risk_tier;
  const score = p.risk_score.toFixed(2);
  const aiDriven = (p.observed_component || 0) > (p.historical_component || 0);
  const yearSep = { en: "–", hi: " से ", mr: " ते " }[lang];
  const years = (p.years || "").replace("-", yearSep);
  const review = (b.review || []).slice(0, 2);
  const s = [];
  if (lang === "hi") {
    s.push(hist
      ? `${p.name} को ${score} जोखिम स्कोर के साथ ${tier} समीक्षा प्राथमिकता में रखा गया है, जो मुख्य रूप से ${aiDriven ? "AI द्वारा देखे गए टकराव के साक्ष्य" : "पिछले दुर्घटना रिकॉर्ड"} पर आधारित है।`
      : `${p.name} का कोई पिछला दुर्घटना रिकॉर्ड नहीं है; इसकी ${tier} प्राथमिकता (स्कोर ${score}) AI द्वारा देखे गए टकराव के साक्ष्य से आती है।`);
    if (hist) s.push(`${years} के दौरान यहाँ ${p.accidents_count} दुर्घटनाएँ और ${p.fatalities} मौतें दर्ज हुईं।`);
    s.push(n ? `AI ने यहाँ ${n} टकराव की घटनाएँ देखीं${nc ? `, जिनमें ${nc} दुर्घटना (क्रैश) शामिल है` : ""}।`
             : "इस स्थान के लिए अभी तक AI द्वारा देखे गए टकराव का कोई साक्ष्य नहीं है।");
    const r = review.map((x) => (REVIEW_I18N[x] || {}).hi).filter(Boolean);
    s.push(r.length ? `मानव समीक्षा के लिए सुझाव: ${r.join("; ")}।` : "सड़क-सुरक्षा इंजीनियर द्वारा मानव समीक्षा की सिफारिश की जाती है।");
    s.push("यह निर्णय-सहायता है, दुर्घटना की भविष्यवाणी नहीं।");
  } else if (lang === "mr") {
    s.push(hist
      ? `${p.name} ला ${score} जोखीम गुणांसह ${tier} पुनरावलोकन प्राधान्य दिले आहे, जे मुख्यतः ${aiDriven ? "AI ने नोंदवलेल्या संघर्ष घटनांवर" : "मागील अपघात नोंदींवर"} आधारित आहे.`
      : `${p.name} साठी मागील अपघाताची नोंद नाही; याचे ${tier} प्राधान्य (गुण ${score}) AI ने नोंदवलेल्या संघर्ष घटनांवर आधारित आहे.`);
    if (hist) s.push(`${years} दरम्यान येथे ${p.accidents_count} अपघात आणि ${p.fatalities} मृत्यू नोंदवले गेले.`);
    s.push(n ? `AI ने येथे ${n} संघर्ष घटना नोंदवल्या${nc ? `, त्यात ${nc} अपघात (क्रॅश) समाविष्ट आहे` : ""}.`
             : "या ठिकाणासाठी अद्याप AI ने नोंदवलेला संघर्षाचा पुरावा उपलब्ध नाही.");
    const r = review.map((x) => (REVIEW_I18N[x] || {}).mr).filter(Boolean);
    s.push(r.length ? `मानवी पुनरावलोकनासाठी शिफारस: ${r.join("; ")}.` : "रस्ते-सुरक्षा अभियंत्याकडून मानवी पुनरावलोकनाची शिफारस केली जाते.");
    s.push("हे निर्णय-सहाय्य आहे, अपघाताचे भाकीत नाही.");
  } else {
    s.push(hist
      ? `${p.name} is classified as ${tier} review priority with a risk score of ${score}, driven mainly by ${aiDriven ? "AI-observed conflict evidence" : "historical crash evidence"}.`
      : `${p.name} has no historical crash record; its ${tier} priority (score ${score}) comes from AI-observed conflict evidence.`);
    if (hist) s.push(`The location recorded ${p.accidents_count} crash${p.accidents_count === 1 ? "" : "es"} and ${p.fatalities} fatalit${p.fatalities === 1 ? "y" : "ies"} during ${years}.`);
    s.push(n ? `The AI observed ${n} conflict event${n === 1 ? "" : "s"} here${nc ? `, including ${nc} crash${nc === 1 ? "" : "es"}` : ""}.`
             : "No AI-observed conflict evidence is currently available for this location.");
    s.push(review.length
      ? `For human review, Verge recommends: ${review.map(lowerFirst).join("; and ")}.`
      : "A review by a road-safety engineer is recommended.");
    s.push("This is decision support, not a prediction of crashes.");
  }
  return s.slice(0, 5).join(" ");
}

/* ---- Verge speech: ONE text-to-speech engine for every audio control ------------
 * Browser-native Web Speech API only (no cloud TTS). Voices are matched by
 * language code on this device; a language without a voice is shown as
 * unavailable and never played. Used by: assistant answers (Listen), the
 * View Analysis audio summary and the video-analysis audio summary. */

const LANG_NAMES = { en: "English", hi: "Hindi", mr: "Marathi" };
// preferred regional variants first, then any voice with the base language code
const VOICE_PREFS = { en: ["en-in", "en-gb", "en-us", "en"], hi: ["hi-in", "hi"], mr: ["mr-in", "mr"] };
const speechState = { token: 0, player: null, players: new Set() };

function voiceFor(lang) {
  if (!("speechSynthesis" in window)) return null;
  const vs = window.speechSynthesis.getVoices();
  const norm = (v) => (v.lang || "").toLowerCase().replace("_", "-");
  for (const code of VOICE_PREFS[lang] || []) {
    const m = vs.filter((v) => (code.includes("-") ? norm(v) === code : norm(v).split("-")[0] === code));
    if (m.length) return m.find((v) => v.localService) || m[0];
  }
  return null;
}

function stopSpeech() {
  speechState.token += 1;
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  if (speechState.player) { speechState.player._setMode("idle"); speechState.player = null; }
}

function voiceUnavailableText(lang) {
  return `${LANG_NAMES[lang]} — Voice unavailable on this device. Install one (Windows: Settings > Time & language > Speech) or choose another language.`;
}

/* Audio player: language select + Play / (Pause) / Stop.
 *   textFor(lang)  the exact text to speak for that language
 *   showText       also show the text that will be spoken (summaries)
 *   pause          offer Pause / Resume
 *   note(lang, v)  optional extra line under the voice status */
function mountAudioPlayer(root, { textFor, showText = false, pause = false, playLabel = "▶ Play", note = null }) {
  root.classList.add("vp");
  root.innerHTML = `
    <label class="vp-label">Audio language
      <select class="vp-lang">${Object.keys(LANG_NAMES).map((l) => `<option value="${l}">${LANG_NAMES[l]}</option>`).join("")}</select>
    </label>
    <div class="vp-ctl">
      <button type="button" class="table-btn" data-vp="play">${playLabel}</button>
      ${pause ? `<button type="button" class="table-btn" data-vp="pause" disabled>Pause</button>` : ""}
      <button type="button" class="table-btn" data-vp="stop" disabled>■ Stop</button>
    </div>
    <p class="hint vp-note"></p>
    ${showText ? `<p class="ad-spoken vp-text"></p>` : ""}`;
  const sel = root.querySelector(".vp-lang");
  const btn = (k) => root.querySelector(`[data-vp="${k}"]`);
  const player = root;

  player._setMode = (mode) => {
    btn("stop").disabled = mode === "idle";
    if (btn("pause")) {
      btn("pause").disabled = mode === "idle";
      btn("pause").textContent = mode === "paused" ? "Resume" : "Pause";
    }
    btn("play").textContent = mode === "idle" ? playLabel : "↻ Restart";
    root.classList.toggle("is-speaking", mode !== "idle");
  };

  player._refresh = () => {
    const synth = "speechSynthesis" in window;
    // option labels say which languages this device can actually speak
    [...sel.options].forEach((o) => {
      const ok = synth && !!voiceFor(o.value);
      o.textContent = ok ? LANG_NAMES[o.value] : `${LANG_NAMES[o.value]} — voice unavailable`;
    });
    const lang = sel.value;
    const v = synth ? voiceFor(lang) : null;
    const n = root.querySelector(".vp-note");
    if (!synth) n.textContent = "This browser has no built-in speech.";
    else if (!v) n.textContent = voiceUnavailableText(lang);
    else n.textContent = `Voice: ${v.name}${v.localService ? "" : " (online voice provided by the browser)"}.`
      + (note ? ` ${note(lang, v)}` : "");
    n.classList.toggle("ad-warn", !v);
    btn("play").disabled = !v;
    if (showText) root.querySelector(".vp-text").textContent = textFor(lang);
  };

  sel.addEventListener("change", () => { if (speechState.player === player) stopSpeech(); player._refresh(); });
  btn("stop").addEventListener("click", () => { if (speechState.player === player) stopSpeech(); });
  if (btn("pause")) btn("pause").addEventListener("click", () => {
    const syn = window.speechSynthesis;
    if (speechState.player !== player) return;
    if (syn.paused) { syn.resume(); player._setMode("speaking"); } else { syn.pause(); player._setMode("paused"); }
  });
  btn("play").addEventListener("click", () => {
    const v = voiceFor(sel.value);
    if (!v) { player._refresh(); return; }        // never a fake playback state
    stopSpeech();                                 // stop whatever else is playing first
    const token = speechState.token;
    const u = new SpeechSynthesisUtterance(textFor(sel.value));
    u.voice = v;
    u.lang = v.lang;
    u.rate = 0.95;
    u.onend = u.onerror = () => {
      if (token !== speechState.token) return;   // a newer playback already took over
      if (speechState.player === player) speechState.player = null;
      player._setMode("idle");
    };
    speechState.player = player;
    player._setMode("speaking");
    window.speechSynthesis.speak(u);
  });
  speechState.players.add(player);
  player._refresh();
  return player;
}

/* Voices often load after the page: refresh every mounted player when they do. */
function setupSpeech() {
  if (!("speechSynthesis" in window)) return;
  const refreshAll = () => {
    speechState.players.forEach((pl) => { if (pl.isConnected) pl._refresh(); else speechState.players.delete(pl); });
  };
  window.speechSynthesis.addEventListener("voiceschanged", refreshAll);
  window.speechSynthesis.getVoices();
}

function openAnalysis(siteId) {
  const f = state.features.find((x) => x.properties.site_id === siteId);
  const dlg = document.getElementById("analysis-dialog");
  if (!f || !dlg) return;
  stopSpeech();
  dlg.dataset.siteId = siteId;
  setAssistantContext(siteId);
  const p = f.properties;
  const b = p.brief || { evidence: [], key_factors: [], review: [] };
  const hist = p.has_historical_record !== false;
  const nc = ((p.evidence || {}).crashes) || 0;
  const evidence = [...b.evidence];
  evidence.push(`AI conflicts: ${p.near_miss_count}${nc ? ` (including ${nc} crash${nc === 1 ? "" : "es"})` : ""}`);
  dlg.querySelector(".ad-body").innerHTML = `
    <div class="ad-head">
      <div>
        <div class="ad-rank">Rank #${p.rank} of ${state.features.length}</div>
        <h3 id="ad-title">${esc(p.name)}</h3>
      </div>
      <span class="score-badge ad-badge" style="background:${TIER_COLORS[p.risk_tier]}">${esc(p.risk_tier.toUpperCase())} · ${p.risk_score.toFixed(2)}</span>
    </div>
    <div class="ad-grid">
      <section><div class="popup-label">Evidence</div>${bullets(evidence)}</section>
      <section><div class="popup-label">Why it was flagged</div>${bullets(b.key_factors.slice(0, 2))}
        <div class="popup-label">Recommended human review</div>${bullets(b.review.slice(0, 2))}</section>
    </div>
    <div class="popup-label">Risk breakdown</div>
    ${scoreSplitHtml(p)}
    <p class="hint">Historical ${(p.historical_component || 0).toFixed(2)} + AI-observed ${(p.observed_component || 0).toFixed(2)} = review priority ${p.risk_score.toFixed(2)} (relative across monitored sites, not a crash probability).${hist ? "" : " No historical crash record for this location."}</p>
    <div class="ad-actions">
      <button type="button" class="primary-btn ad-summarize" aria-expanded="false">Summarize</button>
      <button type="button" class="ghost-btn ad-ask">Ask Verge About This</button>
      <button type="button" class="ghost-btn ad-close">Close</button>
    </div>
    <div class="ad-report">
      <span class="popup-label">Site report</span>
      <button type="button" class="table-btn ad-pdf">Download PDF</button>
      <button type="button" class="table-btn ad-email">Email Report</button>
      <span class="hint ad-report-status"></span>
    </div>
    <div class="ad-summary" hidden>
      <div class="ad-choice" role="group" aria-label="Summary type">
        <button type="button" class="ghost-btn" data-summary="text">Text Summary</button>
        <button type="button" class="ghost-btn" data-summary="audio">Audio Summary</button>
      </div>
      <div class="ad-text" hidden></div>
      <div class="ad-audio" hidden></div>
    </div>`;
  const body = dlg.querySelector(".ad-body");
  const sum = body.querySelector(".ad-summary");
  body.querySelector(".ad-summarize").addEventListener("click", (e) => {
    sum.hidden = !sum.hidden;
    e.currentTarget.setAttribute("aria-expanded", String(!sum.hidden));
  });
  body.querySelector(".ad-close").addEventListener("click", () => dlg.close());
  body.querySelector(".ad-ask").addEventListener("click", () => { dlg.close(); openAssistant(p.site_id); });
  body.querySelector(".ad-pdf").addEventListener("click", async () => {
    const st = body.querySelector(".ad-report-status");
    st.textContent = "Generating PDF…";
    try {
      await downloadBlob(`/api/hotspots/${encodeURIComponent(p.site_id)}/report.pdf`, `verge-road-safety-report-${slugify(p.name)}.pdf`);
      st.textContent = "Site report downloaded.";
    } catch (err) { st.textContent = `Couldn't generate the PDF (${err.message}).`; }
  });
  body.querySelector(".ad-email").addEventListener("click", () => openEmailDialog({
    url: `/api/hotspots/${encodeURIComponent(p.site_id)}/email`,
    subject: `Verge Road Safety Report — ${p.name}`,
    attachment: `verge-road-safety-report-${slugify(p.name)}.pdf`,
  }));
  body.querySelectorAll("[data-summary]").forEach((btn) => btn.addEventListener("click", () => {
    const kind = btn.dataset.summary;
    body.querySelectorAll("[data-summary]").forEach((x) => x.classList.toggle("active", x === btn));
    const text = body.querySelector(".ad-text");
    const audio = body.querySelector(".ad-audio");
    text.hidden = kind !== "text";
    audio.hidden = kind !== "audio";
    if (kind === "text") text.textContent = hotspotSummary(p, "en");
    else { stopSpeech(); if (!audio.firstChild) mountAudioPlayer(audio, { textFor: (lang) => hotspotSummary(p, lang), showText: true, pause: true, playLabel: "▶ Play Summary" }); else audio._refresh(); }
  }));
  if (!dlg.open) dlg.showModal();
}

/* ---- Verge Road Safety Assistant -------------------------------------------------
 * Grounded, deterministic answers from the backend (/api/assistant): no language
 * model, only Verge's own hotspot data, the latest analysis and how Verge works.
 * The context (selected hotspot, latest video) follows what the user selects. */

const ASSIST_SUGGESTIONS = [
  "Why is this hotspot high risk?",
  "Compare the top hotspots",
  "Explain the latest video analysis",
  "What should authorities review?",
  "How does Verge detect conflicts?",
  "How is the risk score calculated?",
];

function setAssistantContext(siteId) {
  if (!siteId || !state.features.find((f) => f.properties.site_id === siteId)) return;
  state.assistantSiteId = siteId;
  renderAssistantContext();
}

function renderAssistantContext() {
  const el = document.getElementById("as-context");
  if (!el) return;
  const f = state.features.find((x) => x.properties.site_id === state.assistantSiteId);
  const chips = [];
  if (f) {
    const p = f.properties;
    chips.push(`<span class="as-chip"><i style="background:${TIER_COLORS[p.risk_tier]}"></i>${esc(p.name)}
      <button type="button" class="as-chip-x" data-clear-context title="Clear hotspot context" aria-label="Clear hotspot context">×</button></span>`);
  }
  if (state.analysis && state.analysis.analysis_id) {
    chips.push(`<span class="as-chip as-chip-video">Latest video analysis · ${esc(state.analysis.source_name || "clip")}</span>`);
  }
  el.innerHTML = chips.length
    ? `<span class="as-context-label">Context</span>${chips.join("")}`
    : `<span class="as-context-label">Context</span><span class="muted">All monitored hotspots · select a hotspot for location-specific questions</span>`;
  const x = el.querySelector("[data-clear-context]");
  if (x) x.addEventListener("click", () => { state.assistantSiteId = null; renderAssistantContext(); });
}

function openAssistant(siteId) {
  const panel = document.getElementById("assistant-panel");
  if (siteId) setAssistantContext(siteId);
  renderAssistantContext();
  panel.hidden = false;
  document.getElementById("btn-assistant").setAttribute("aria-expanded", "true");
  document.getElementById("as-input").focus();
}

function closeAssistant() {
  document.getElementById("assistant-panel").hidden = true;
  document.getElementById("btn-assistant").setAttribute("aria-expanded", "false");
}

function assistantBlocksHtml(blocks) {
  return blocks.map((b) => {
    if (b.type === "bullets") return `<ul>${b.items.map((i) => `<li>${esc(i)}</li>`).join("")}</ul>`;
    if (b.type === "table") {
      return `<table class="as-table"><thead><tr>${b.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
        <tbody>${b.rows.map((r) => `<tr>${r.map((c) => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
    }
    return `<p>${esc(b.text)}</p>`;
  }).join("");
}

function appendAssistantMsg(who, html) {
  const log = document.getElementById("as-log");
  const div = document.createElement("div");
  div.className = `as-msg as-${who}`;
  div.innerHTML = html;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

async function askAssistant(question) {
  const q = (question || "").trim();
  if (!q) return;
  appendAssistantMsg("user", `<p>${esc(q)}</p>`);
  const pending = appendAssistantMsg("bot",
    `<p class="as-typing">Verge is thinking<span class="as-dots"><i></i><i></i><i></i></span></p>`);
  const fab = document.getElementById("btn-assistant");
  fab.classList.add("is-thinking");
  const t0 = performance.now();
  // local answers are near-instant: a short pause keeps it conversational, but
  // never adds to an answer that already took longer
  const settle = () => new Promise((ok) => setTimeout(ok, Math.max(0, 1100 - (performance.now() - t0))));
  try {
    const res = await api("/api/assistant", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: q,
        site_id: state.assistantSiteId || null,
        analysis_id: (state.analysis && state.analysis.analysis_id) || null,
      }),
    }, 15000);
    const r = await res.json();
    await settle();
    pending.innerHTML = `${assistantBlocksHtml(r.blocks || [])}
      <div class="as-foot"><span class="as-source">${esc(r.source || "")}</span>
      <button type="button" class="as-listen" aria-expanded="false" title="Read this answer aloud">🔊 Listen</button></div>`;
    const speech = answerSpeechText(r.blocks);
    pending.querySelector(".as-listen").addEventListener("click", () => toggleListen(pending, speech));
  } catch (err) {
    await settle();
    pending.innerHTML = `<p class="error-text">Couldn't reach the Verge API (${esc(err.message)}).</p>`;
  } finally {
    fab.classList.remove("is-thinking");
  }
  document.getElementById("as-log").scrollTop = 1e9;
}

function setupAssistant() {
  document.getElementById("btn-assistant").addEventListener("click", () => {
    if (document.getElementById("assistant-panel").hidden) openAssistant(); else closeAssistant();
  });
  document.getElementById("as-close").addEventListener("click", closeAssistant);
  document.getElementById("as-suggest").innerHTML = ASSIST_SUGGESTIONS
    .map((s) => `<button type="button" class="as-suggestion">${esc(s)}</button>`).join("");
  document.getElementById("as-suggest").addEventListener("click", (e) => {
    const b = e.target.closest(".as-suggestion");
    if (b) askAssistant(b.textContent);
  });
  document.getElementById("as-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const input = document.getElementById("as-input");
    askAssistant(input.value);
    input.value = "";
  });
  renderAssistantContext();
}

/* ---- e-mail a report (the same PDF as the download, sent by the backend) ---------- */

function openEmailDialog({ url, subject, attachment, event = 0 }) {
  const dlg = document.getElementById("email-dialog");
  dlg.dataset.url = url;
  dlg.dataset.event = String(event);
  dlg.querySelector("#em-to").value = "";
  dlg.querySelector("#em-subject").value = subject;
  dlg.querySelector("#em-attachment").textContent = attachment;
  const status = dlg.querySelector("#em-status");
  const send = dlg.querySelector("#em-send");
  status.className = "em-status";
  status.textContent = "Checking email configuration…";
  send.disabled = true;
  dlg.querySelector("#em-form").hidden = false;
  dlg.querySelector("#em-done").hidden = true;
  if (!dlg.open) dlg.showModal();
  api("/api/email/status", {}, 5000).then((r) => r.json()).then((st) => {
    if (!st.configured) {
      status.className = "em-status em-warn";
      status.textContent = st.message || "Email delivery is not configured. Add SMTP credentials to enable direct sending.";
      send.disabled = true;
    } else if (st.mode === "outbox") {
      status.className = "em-status em-warn";
      status.textContent = "Test mode: the email (with the PDF attached) is saved to outputs/outbox, not sent.";
      send.disabled = false;
    } else {
      status.className = "em-status";
      status.textContent = `Sent from ${st.from} with the PDF attached.`;
      send.disabled = false;
    }
  }).catch(() => {
    status.className = "em-status em-warn";
    status.textContent = "Couldn't check the email configuration (is the API running?).";
  });
}

async function sendEmail(evt) {
  evt.preventDefault();
  const dlg = document.getElementById("email-dialog");
  const send = dlg.querySelector("#em-send");
  const status = dlg.querySelector("#em-status");
  const to = dlg.querySelector("#em-to").value.trim();
  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(to)) {
    status.className = "em-status em-warn";
    status.textContent = "Please enter one valid recipient email address.";
    return;
  }
  send.disabled = true;
  send.textContent = "Sending…";
  try {
    const res = await fetch(apiBase() + dlg.dataset.url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ to, subject: dlg.querySelector("#em-subject").value, event: Number(dlg.dataset.event) }),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || "Unable to send the report. Please check the email configuration and try again.");
    const done = dlg.querySelector("#em-done");
    done.innerHTML = body.status === "sent"
      ? `<div class="em-ok">✓ Report sent successfully</div>
         <dl class="em-facts"><dt>To</dt><dd>${esc(body.to)}</dd><dt>Attachment</dt><dd>${esc(body.attachment)}</dd></dl>`
      : `<div class="em-ok em-test">Saved in test mode (not sent)</div>
         <dl class="em-facts"><dt>To</dt><dd>${esc(body.to)}</dd><dt>Attachment</dt><dd>${esc(body.attachment)}</dd>
         <dt>Saved as</dt><dd><code>${esc(body.path || "")}</code></dd></dl>`;
    done.insertAdjacentHTML("beforeend", `<div class="ad-actions"><button type="button" class="ghost-btn" data-em-close>Close</button></div>`);
    done.querySelector("[data-em-close]").addEventListener("click", () => dlg.close());
    dlg.querySelector("#em-form").hidden = true;
    done.hidden = false;
  } catch (err) {
    status.className = "em-status em-warn";
    status.textContent = err.message;
  } finally {
    send.disabled = false;
    send.textContent = "Send Report";
  }
}

function setupEmailDialog() {
  const dlg = document.getElementById("email-dialog");
  dlg.querySelector("#em-form").addEventListener("submit", sendEmail);
  dlg.querySelector("#em-cancel").addEventListener("click", () => dlg.close());
}

function slugify(s) {
  return String(s || "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 60) || "report";
}

/* ---- speech helpers shared by every summary / answer ------------------------------ */

/* Summarize widget: [Summarize] -> [Text] [Audio]. Audio uses the shared
 * player and always speaks the same generated text. textFor(lang) returns the
 * summary for that language from real data. */
function mountSummarize(root, textFor) {
  root.innerHTML = `
    <button type="button" class="primary-btn sz-open" aria-expanded="false">Summarize</button>
    <div class="sz-panel" hidden>
      <div class="ad-choice" role="group" aria-label="Summary type">
        <button type="button" class="ghost-btn" data-sz="text">Text</button>
        <button type="button" class="ghost-btn" data-sz="audio">Audio</button>
      </div>
      <div class="ad-text sz-text" hidden></div>
      <div class="ad-audio sz-audio" hidden></div>
    </div>`;
  const panel = root.querySelector(".sz-panel");
  const audio = root.querySelector(".sz-audio");
  root.querySelector(".sz-open").addEventListener("click", (e) => {
    panel.hidden = !panel.hidden;
    e.currentTarget.setAttribute("aria-expanded", String(!panel.hidden));
  });
  root.querySelectorAll("[data-sz]").forEach((b) => b.addEventListener("click", () => {
    root.querySelectorAll("[data-sz]").forEach((x) => x.classList.toggle("active", x === b));
    const text = b.dataset.sz === "text";
    root.querySelector(".sz-text").hidden = !text;
    audio.hidden = text;
    if (text) root.querySelector(".sz-text").textContent = textFor("en");
    else { stopSpeech(); if (!audio.firstChild) mountAudioPlayer(audio, { textFor, showText: true, pause: true, playLabel: "▶ Play Summary" }); else audio._refresh(); }
  }));
}

/* ---- video event summary (from the displayed event's own measurements) ------------ */

function userLabel(label, cls, id) {
  const lab = label && label !== "vehicle" ? label : cls === "pedestrian" ? "pedestrian" : "vehicle";
  return `${lab.charAt(0).toUpperCase()}${lab.slice(1)} #${id}`;
}

function eventSummary(e, r, lang = "en") {
  const A = userLabel(e.a_label, e.a_cls, e.a_id);
  const B = userLabel(e.b_label, e.b_cls, e.b_id);
  const at = Number(e.peak_t || 0).toFixed(1);
  const loc = r.location && r.location.name;
  const crash = e.event_type === "crash";
  const close = e.indicator === "close_pass";
  const ttc = e.min_ttc_s !== null && e.min_ttc_s !== undefined ? Number(e.min_ttc_s).toFixed(2) : null;
  const dist = e.min_distance_m !== null && e.min_distance_m !== undefined ? Number(e.min_distance_m).toFixed(1) : null;
  const dur = e.zone_duration_s ? Number(e.zone_duration_s).toFixed(1) : null;
  const pet = e.pet_s !== null && e.pet_s !== undefined ? Number(e.pet_s).toFixed(2) : null;
  const upd = r.site_update && r.site_update.before && r.site_update.after ? r.site_update : null;
  const s = [];
  if (lang === "hi") {
    s.push(crash
      ? `विश्लेषित फुटेज${loc ? ` (${loc})` : ""} में क्लिप के ${at} सेकंड पर ${A} और ${B} के बीच टक्कर दर्ज हुई (${e.severity_label} गंभीरता)।`
      : `विश्लेषित फुटेज${loc ? ` (${loc})` : ""} में ${A} और ${B} के बीच संभावित टकराव दिखता है, जो क्लिप के ${at} सेकंड पर चरम पर था (${e.severity_label} गंभीरता)।`);
    if (crash) s.push(`संपर्क से पहले दूरी लगभग ${Math.round(e.approach_closing_ms || 0)} मीटर प्रति सेकंड की गति से घटी, और संपर्क के तुरंत बाद लगभग ${Math.round(e.impact_dv_ms || 0)} मीटर प्रति सेकंड का वेग परिवर्तन मापा गया।`);
    else if (close) s.push(`दोनों चलते हुए लगभग ${dist} मीटर की दूरी से निकले (क्लोज़ पास)।`);
    else s.push(`अनुमानित टकराव-समय (TTC) ${ttc ?? "—"} सेकंड तक पहुँचा${e.contact_phase ? ", और दोनों के फुटप्रिंट मिल गए (दूरी माप सीमा पर)" : dist !== null ? `, न्यूनतम दूरी लगभग ${dist} मीटर थी` : ""}।`);
    if (dur) s.push(`टकराव संकेतक लगभग ${dur} सेकंड तक बना रहा${pet ? `; पोस्ट-एनक्रोचमेंट समय ${pet} सेकंड था` : ""}।`);
    s.push(upd ? `इस साक्ष्य से ${upd.name} की समीक्षा प्राथमिकता ${upd.before.risk_score.toFixed(2)} से ${upd.after.risk_score.toFixed(2)} हो गई।`
               : "यह क्लिप किसी निगरानी स्थान से जुड़ी नहीं थी।");
    s.push("सड़क-सुरक्षा इंजीनियर को स्थल पर इसकी समीक्षा करनी चाहिए; यह निर्णय-सहायता है, दोष निर्धारण नहीं।");
  } else if (lang === "mr") {
    s.push(crash
      ? `विश्लेषित फुटेज${loc ? ` (${loc})` : ""} मध्ये क्लिपच्या ${at} सेकंदाला ${A} आणि ${B} यांच्यात धडक नोंदवली गेली (${e.severity_label} तीव्रता).`
      : `विश्लेषित फुटेज${loc ? ` (${loc})` : ""} मध्ये ${A} आणि ${B} यांच्यात संभाव्य संघर्ष दिसतो, जो क्लिपच्या ${at} सेकंदाला सर्वाधिक होता (${e.severity_label} तीव्रता).`);
    if (crash) s.push(`संपर्कापूर्वी अंतर सुमारे ${Math.round(e.approach_closing_ms || 0)} मीटर प्रति सेकंद वेगाने कमी झाले आणि संपर्कानंतर लगेच सुमारे ${Math.round(e.impact_dv_ms || 0)} मीटर प्रति सेकंद वेगबदल मोजला गेला.`);
    else if (close) s.push(`दोघे चालत असताना सुमारे ${dist} मीटर अंतरावरून गेले (क्लोज पास).`);
    else s.push(`अंदाजित धडक-वेळ (TTC) ${ttc ?? "—"} सेकंदांपर्यंत पोहोचली${e.contact_phase ? ", आणि दोघांचे फूटप्रिंट एकमेकांना भिडले (अंतर मापन मर्यादेवर)" : dist !== null ? `, किमान अंतर सुमारे ${dist} मीटर होते` : ""}.`);
    if (dur) s.push(`संघर्ष निर्देशक सुमारे ${dur} सेकंद टिकला${pet ? `; पोस्ट-एन्क्रोचमेंट वेळ ${pet} सेकंद होती` : ""}.`);
    s.push(upd ? `या पुराव्यामुळे ${upd.name} चे पुनरावलोकन प्राधान्य ${upd.before.risk_score.toFixed(2)} वरून ${upd.after.risk_score.toFixed(2)} झाले.`
               : "ही क्लिप कोणत्याही निरीक्षित ठिकाणाशी जोडलेली नव्हती.");
    s.push("रस्ते-सुरक्षा अभियंत्याने प्रत्यक्ष ठिकाणी याचा आढावा घ्यावा; हे निर्णय-सहाय्य आहे, दोषनिश्चिती नाही.");
  } else {
    s.push(crash
      ? `The analysed footage${loc ? ` at ${loc}` : ""} shows an observed collision between ${A} and ${B}, ${at} s into the clip (${e.severity_label} severity).`
      : `The analysed footage${loc ? ` at ${loc}` : ""} shows a potential conflict between ${A} and ${B}, peaking ${at} s into the clip (${e.severity_label} severity).`);
    if (crash) s.push(`The gap between them closed at about ${Math.round(e.approach_closing_ms || 0)} m/s, and straight after contact a velocity change of about ${Math.round(e.impact_dv_ms || 0)} m/s was measured.`);
    else if (close) s.push(`They passed each other at speed with a clearance of about ${dist} m (close pass).`);
    else s.push(`The estimated time-to-collision reached ${ttc ?? "an unmeasurable value"} s${e.contact_phase ? ", and their footprints met (separation at the measurement limit)" : dist !== null ? `, with a minimum separation of about ${dist} m` : ""}.`);
    if (dur) s.push(`The conflict indicator held for about ${dur} s${pet ? `, and the post-encroachment time was ${pet} s` : ""}.`);
    s.push(upd ? `This evidence moved ${upd.name}'s review priority from ${upd.before.risk_score.toFixed(2)} to ${upd.after.risk_score.toFixed(2)} (${upd.after.risk_tier}).`
               : "The clip was not linked to a monitored location, so no site score changed.");
    s.push("The event should be reviewed by a road-safety engineer on site; this is decision support, not a finding of fault.");
  }
  return s.slice(0, 5).join(" ");
}

/* ---- assistant: thinking indicator + Listen on every answer ------------------------- */

function answerSpeechText(blocks) {
  return (blocks || []).map((b) => {
    if (b.type === "bullets") return b.items.join(". ");
    if (b.type === "table") return b.rows.map((row) => row.map((c, i) => `${b.columns[i]} ${c}`).join(", ")).join(". ");
    return b.text;
  }).join(" ").replace(/↔/g, " and ").replace(/#(\d+)/g, "number $1");
}

/* Listen under an answer: opens the shared player for exactly that answer's
 * text. The answer itself stays as written; the chosen language picks the voice. */
function toggleListen(msg, text) {
  const btn = msg.querySelector(".as-listen");
  let box = msg.querySelector(".as-audio");
  if (box) {
    const open = box.hidden;
    box.hidden = !open;
    btn.setAttribute("aria-expanded", String(open));
    if (!open && speechState.player === box) stopSpeech();
    return;
  }
  box = document.createElement("div");
  box.className = "as-audio";
  msg.appendChild(box);
  mountAudioPlayer(box, {
    textFor: () => text,
    note: (lang) => (lang === "en" ? "" : `The answer is read as written, with a ${LANG_NAMES[lang]} voice.`),
  });
  btn.setAttribute("aria-expanded", "true");
  document.getElementById("as-log").scrollTop = 1e9;
}

/* ---- Visualize: ranked hotspots chart (current data only) ---------------------------- */

const VIZ_SERIES = { crashes: "#16925c", fatalities: "#7d5ba6" };  // validated pair (dataviz validator)

function vizTooltip(p) {
  const hist = p.has_historical_record !== false;
  return `<b>#${p.rank} ${esc(p.name)}</b><br>${esc(p.risk_tier)} · risk score ${p.risk_score.toFixed(2)}<br>`
    + `Crashes ${hist ? p.accidents_count : "no record"} · Fatalities ${hist ? p.fatalities : "—"} · AI conflicts ${p.near_miss_count}`
    + `<br><span class="muted">Click for View Analysis</span>`;
}

function renderViz(mode) {
  const box = document.getElementById("viz-chart");
  const sites = state.features.map((f) => f.properties).sort((a, b) => a.rank - b.rank);
  document.querySelectorAll("[data-viz]").forEach((b) => b.classList.toggle("active", b.dataset.viz === mode));
  const title = document.getElementById("viz-title");
  const sub = document.getElementById("viz-sub");
  let rows = "";
  let axis = "";
  let legend = "";
  if (mode === "risk") {
    title.textContent = "Monitored Hotspots — Review Priority";
    sub.textContent = "Current Verge risk scores (0–1). Bar colour = risk level.";
    const ticks = [0, 0.2, 0.4, 0.6, 0.8, 1];
    rows = sites.map((p) => `
      <button type="button" class="viz-row" data-site="${esc(p.site_id)}" data-tip="${esc(vizTooltip(p))}">
        <span class="viz-name">${p.rank}. ${esc(p.name)}</span>
        <span class="viz-track">${ticks.slice(1, -1).map((t) => `<i class="viz-grid" style="left:${t * 100}%"></i>`).join("")}
          <span class="viz-bar" style="width:${Math.max(0.4, p.risk_score * 100)}%;background:${TIER_COLORS[p.risk_tier]}"></span>
          <span class="viz-val" style="left:${p.risk_score * 100}%">${p.risk_score.toFixed(2)} · ${esc(p.risk_tier)}</span>
        </span>
      </button>`).join("");
    axis = `<div class="viz-axis"><span></span><span class="viz-ticks">${ticks.map((t) => `<i style="left:${t * 100}%">${t.toFixed(1)}</i>`).join("")}</span></div>
      <div class="viz-axis-label">Review priority (risk score)</div>`;
    legend = `<div class="viz-legend">${["Low", "Moderate", "High", "Critical"].map((t) => `<span><i style="background:${TIER_COLORS[t]}"></i>${t}</span>`).join("")}</div>`;
  } else {
    title.textContent = "Monitored Hotspots — Crashes & Fatalities";
    sub.textContent = "Historical records per location (count).";
    const hist = sites.filter((p) => p.has_historical_record !== false);
    const max = Math.max(1, ...hist.map((p) => p.accidents_count));
    const step = max > 20 ? 10 : 5;
    const top = Math.ceil(max / step) * step;
    const ticks = [];
    for (let t = 0; t <= top; t += step) ticks.push(t);
    rows = sites.map((p) => {
      const has = p.has_historical_record !== false;
      return `
      <button type="button" class="viz-row viz-row-2" data-site="${esc(p.site_id)}" data-tip="${esc(vizTooltip(p))}">
        <span class="viz-name">${p.rank}. ${esc(p.name)}</span>
        <span class="viz-track">${ticks.slice(1, -1).map((t) => `<i class="viz-grid" style="left:${(t / top) * 100}%"></i>`).join("")}
          ${has ? `
          <span class="viz-pair">
            <span class="viz-sub-row"><span class="viz-bar" style="width:${Math.max(0.4, (p.accidents_count / top) * 100)}%;background:${VIZ_SERIES.crashes}"></span><span class="viz-val2" style="left:calc(${(p.accidents_count / top) * 100}% + 5px)">${p.accidents_count}</span></span>
            <span class="viz-sub-row"><span class="viz-bar" style="width:${Math.max(0.4, (p.fatalities / top) * 100)}%;background:${VIZ_SERIES.fatalities}"></span><span class="viz-val2" style="left:calc(${(p.fatalities / top) * 100}% + 5px)">${p.fatalities}</span></span>
          </span>` : `<span class="viz-norecord">no historical record</span>`}
        </span>
      </button>`;
    }).join("");
    axis = `<div class="viz-axis"><span></span><span class="viz-ticks">${ticks.map((t) => `<i style="left:${(t / top) * 100}%">${t}</i>`).join("")}</span></div>
      <div class="viz-axis-label">Recorded count (historical data)</div>`;
    legend = `<div class="viz-legend"><span><i style="background:${VIZ_SERIES.crashes}"></i>Crashes</span><span><i style="background:${VIZ_SERIES.fatalities}"></i>Fatalities</span></div>`;
  }
  box.innerHTML = `${legend}<div class="viz-rows">${rows}</div>${axis}`;
  const tip = document.getElementById("viz-tip");
  box.querySelectorAll(".viz-row").forEach((row) => {
    row.addEventListener("mousemove", (ev) => {
      tip.innerHTML = row.dataset.tip;
      tip.hidden = false;
      const r = box.getBoundingClientRect();
      tip.style.left = `${Math.min(ev.clientX - r.left + 14, r.width - 240)}px`;
      tip.style.top = `${ev.clientY - r.top + 14}px`;
    });
    row.addEventListener("mouseleave", () => { tip.hidden = true; });
    row.addEventListener("click", () => {
      document.getElementById("viz-dialog").close();
      openAnalysis(row.dataset.site);
    });
  });
}

function setupViz() {
  const dlg = document.getElementById("viz-dialog");
  document.getElementById("btn-visualize").addEventListener("click", () => {
    if (!state.features.length) return;
    renderViz("risk");
    if (!dlg.open) dlg.showModal();
  });
  dlg.querySelectorAll("[data-viz]").forEach((b) => b.addEventListener("click", () => renderViz(b.dataset.viz)));
  dlg.querySelector("#viz-close").addEventListener("click", () => dlg.close());
  dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
}

function setupAnalysisDialog() {
  const dlg = document.getElementById("analysis-dialog");
  dlg.addEventListener("close", stopSpeech);
  dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });  // click on the backdrop
}

async function downloadBlob(path, fallbackName) {
  const res = await api(path, {}, 60000);
  const blob = await res.blob();
  const cd = res.headers.get("Content-Disposition") || "";
  const m = cd.match(/filename="?([^"]+)"?/);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = m ? m[1] : fallbackName;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

/* A deliberately small Markdown -> HTML converter — just enough for
 * report.py's output (headers, bold, italics, tables, blockquote, hr, lists,
 * inline code). Input is HTML-escaped first. Not a general-purpose parser. */
function tinyMarkdownToHtml(md) {
  const lines = md.split("\n");
  let html = "";
  let inTable = false;
  let inList = false;

  const inline = (s) =>
    esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[\s(])_([^_]+)_(?=$|[\s).,])/g, "$1<em>$2</em>");

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (/^\s*\|.*\|\s*$/.test(line)) {
      const cells = line.trim().slice(1, -1).split("|").map((c) => c.trim());
      if (/^-+$/.test(cells.join(""))) continue; // header separator row
      if (!inTable) {
        html += "<table><thead><tr>" + cells.map((c) => `<th>${inline(c)}</th>`).join("") + "</tr></thead><tbody>";
        inTable = true;
      } else {
        html += "<tr>" + cells.map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>";
      }
      continue;
    }
    if (inTable) {
      html += "</tbody></table>";
      inTable = false;
    }

    if (/^-\s+/.test(line)) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${inline(line.replace(/^-\s+/, ""))}</li>`;
      continue;
    }
    if (inList) {
      html += "</ul>";
      inList = false;
    }

    if (/^#### /.test(line)) html += `<h4>${inline(line.slice(5))}</h4>`;
    else if (/^### /.test(line)) html += `<h3>${inline(line.slice(4))}</h3>`;
    else if (/^## /.test(line)) html += `<h2>${inline(line.slice(3))}</h2>`;
    else if (/^# /.test(line)) html += `<h1>${inline(line.slice(2))}</h1>`;
    else if (/^\*\*[^*]+\*\*$/.test(line.trim())) html += `<h4 class="md-label">${esc(line.trim().slice(2, -2))}</h4>`;
    else if (/^_.+_$/.test(line.trim())) html += `<p class="md-meta">${esc(line.trim().slice(1, -1)).replace(/`([^`]+)`/g, "<code>$1</code>")}</p>`;
    else if (/^>\s?/.test(line)) html += `<blockquote>${inline(line.replace(/^>\s?/, ""))}</blockquote>`;
    else if (/^---$/.test(line)) html += "<hr/>";
    else if (line.trim() === "") html += "";
    else html += `<p>${inline(line)}</p>`;
  }
  if (inTable) html += "</tbody></table>";
  if (inList) html += "</ul>";
  return html;
}

/* ---- tabs ----------------------------------------------------------------------- */

function isTabActive(name) {
  const el = document.getElementById(`tab-${name}`);
  return el && el.classList.contains("active");
}

function activateTab(name) {
  document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab-content").forEach((c) => c.classList.toggle("active", c.id === `tab-${name}`));
  updateMapHint();
}

function updateMapHint() {
  const show = isTabActive("analyze") && !state.selected;
  showMapHint(show ? "Click a hotspot, or anywhere on the map to pin a new location for the video." : "");
}

function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => activateTab(btn.dataset.tab));
  });
}

async function refreshAll() {
  checkApi();
  await Promise.all([loadHotspots(), loadReport()]);
}

async function runPipeline() {
  const btn = document.getElementById("btn-run-pipeline");
  btn.disabled = true;
  btn.textContent = "Running…";
  try {
    await api("/api/pipeline/run", { method: "POST" }, 300000);
    await refreshAll();
  } catch (err) {
    alert(`Pipeline run failed: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Re-run pipeline";
  }
}

async function clearEvidence() {
  if (!confirm("Remove all evidence from uploaded videos and any map pins? Historical data and the demo pipeline result are kept.")) return;
  try {
    await api("/api/observations", { method: "DELETE" }, 15000);
    state.lastUpdate = null;
    document.getElementById("analyze-result").innerHTML = "";
    document.getElementById("analyze-progress").innerHTML = "";
    stopPlayer();
    await refreshAll();
  } catch (err) {
    alert(`Could not clear evidence: ${err.message}`);
  }
}

/* ---- location selection ------------------------------------------------------ */

function fillSiteOptions() {
  const dl = document.getElementById("site-options");
  dl.innerHTML = state.features
    .slice()
    .sort((a, b) => a.properties.name.localeCompare(b.properties.name))
    .map((f) => `<option value="${esc(f.properties.name)}"></option>`)
    .join("");
}

function clearSelectionLayers() {
  if (state.selectionLayer) { map.removeLayer(state.selectionLayer); state.selectionLayer = null; }
  if (state.pinMarker) { map.removeLayer(state.pinMarker); state.pinMarker = null; }
}

function clearSelection() {
  state.selected = null;
  clearSelectionLayers();
  renderSelection();
  renderHotspotList(state.features);
}

function selectSite(siteId, opts = {}) {
  const f = state.features.find((x) => x.properties.site_id === siteId);
  if (!f) return;
  const p = f.properties;
  const [lon, lat] = f.geometry.coordinates;
  clearSelectionLayers();
  state.selected = { site_id: p.site_id, name: p.name, lat, lon, pinned: !!p.pinned, isNew: false };
  setAssistantContext(p.site_id);
  state.selectionLayer = L.circleMarker([lat, lon], {
    radius: 8 + p.risk_score * 14 + 10, color: "#10231b", weight: 2, fill: false, interactive: false,
  }).addTo(map);
  renderSelection();
  if (!opts.silent) renderHotspotList(state.features);
  document.getElementById("loc-search").value = "";
  if (!opts.silent && !opts.keepPopup) map.setView([lat, lon], Math.max(map.getZoom(), 14));
}

function setPinnedLocation(lat, lon) {
  clearSelectionLayers();
  state.selected = { name: "", lat, lon, pinned: true, isNew: true };
  state.pinMarker = L.circleMarker([lat, lon], {
    radius: 9, color: "#10231b", weight: 2, dashArray: "4 3", fillColor: "#ffffff", fillOpacity: 0.9,
  }).addTo(map).bindTooltip("New location", { permanent: true, direction: "top", offset: [0, -8] });
  renderSelection();
  const input = document.getElementById("loc-name");
  if (input) input.focus();
}

function renderSelection() {
  updateMapHint();
  const box = document.getElementById("loc-selected");
  const picker = document.getElementById("loc-picker");
  const s = state.selected;
  if (!s) {
    box.hidden = true;
    picker.hidden = false;
    return;
  }
  box.hidden = false;
  picker.hidden = true;
  const coords = `${s.lat.toFixed(5)}, ${s.lon.toFixed(5)}`;
  if (s.isNew) {
    box.innerHTML = `
      <div class="loc-row">
        <div>
          <div class="loc-kicker">New map pin · no historical crash record</div>
          <input id="loc-name" type="text" maxlength="80" placeholder="Name this location (e.g. MG Road crossing)" />
          <div class="loc-coords">${coords}</div>
        </div>
        <button type="button" class="link-btn" id="loc-change">Change</button>
      </div>`;
  } else {
    const f = state.features.find((x) => x.properties.site_id === s.site_id);
    const p = f ? f.properties : null;
    box.innerHTML = `
      <div class="loc-row">
        <div>
          <div class="loc-kicker">${p && p.pinned ? "Map pin" : "Monitored location"}${p ? ` · rank #${p.rank} of ${state.features.length}` : ""}</div>
          <div class="loc-name">${esc(s.name)}</div>
          <div class="loc-coords">${coords}${p ? ` · <span class="score-badge" style="background:${TIER_COLORS[p.risk_tier]}">${esc(p.risk_tier)} · ${p.risk_score.toFixed(2)}</span>` : ""}</div>
        </div>
        <button type="button" class="link-btn" id="loc-change">Change</button>
      </div>`;
  }
  document.getElementById("loc-change").addEventListener("click", clearSelection);
}

function setupLocationSearch() {
  const input = document.getElementById("loc-search");
  const pick = () => {
    const q = input.value.trim();
    if (!q) return;
    const f = searchHotspots(q)[0];
    if (f) selectSite(f.properties.site_id);
  };
  input.addEventListener("change", pick);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); pick(); } });
}

/* ---- hotspot search (offline: matches the loaded hotspot catalogue only) ---- */

function normName(s) {
  return String(s).toLowerCase().normalize("NFKD").replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, " ").trim();
}

/* Levenshtein distance with an early exit once it exceeds `cap`. */
function editDistance(a, b, cap) {
  if (Math.abs(a.length - b.length) > cap) return cap + 1;
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    let rowMin = i;
    for (let j = 1; j <= b.length; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      rowMin = Math.min(rowMin, cur[j]);
    }
    if (rowMin > cap) return cap + 1;
    prev = cur;
  }
  return prev[b.length];
}

/* How well a query matches a site name (0 = no match): exact > prefix >
 * substring > every query word matched as a word prefix or with a small typo
 * (1 letter for words of 4-6 letters, 2 for longer words). */
function matchScore(query, name) {
  const q = normName(query);
  const n = normName(name);
  if (!q) return 0;
  if (n === q) return 100;
  if (n.startsWith(q)) return 80;
  if (n.includes(q)) return 60;
  const words = n.split(" ");
  let total = 0;
  for (const w of q.split(" ")) {
    const cap = w.length >= 7 ? 2 : 1;
    let best = 0;
    for (const nw of words) {
      if (nw === w) best = Math.max(best, 3);
      else if (w.length >= 2 && nw.startsWith(w)) best = Math.max(best, 2);
      else if (w.length >= 4 && editDistance(w, nw, cap) <= cap) best = Math.max(best, 1);
    }
    if (!best) return 0;  // every word of the query must match something
    total += best;
  }
  return 10 + total;
}

function searchHotspots(query, limit = 6) {
  return state.features
    .map((f) => ({ f, s: matchScore(query, f.properties.name) }))
    .filter((x) => x.s > 0)
    .sort((a, b) => b.s - a.s || a.f.properties.rank - b.f.properties.rank)
    .slice(0, limit)
    .map((x) => x.f);
}

/* Centre the map on a hotspot, emphasise its marker, open its details and
 * mark its card in the ranked list. */
function goToHotspot(siteId) {
  const f = state.features.find((x) => x.properties.site_id === siteId);
  if (!f) return;
  const p = f.properties;
  const [lon, lat] = f.geometry.coordinates;
  if (state.searchHalo) map.removeLayer(state.searchHalo);
  state.searchHalo = L.circleMarker([lat, lon], {
    radius: 8 + p.risk_score * 14 + 7, color: "#10231b", weight: 3, fill: false, interactive: false,
  }).addTo(map);
  state.searchedSiteId = siteId;
  setAssistantContext(siteId);
  map.setView([lat, lon], Math.max(map.getZoom(), 15), { animate: false });
  markersLayer.eachLayer((m) => { if (m.siteId === siteId) { m.bringToFront(); m.openPopup(); } });
  if (isTabActive("analyze")) selectSite(siteId, { keepPopup: true });
  markSearchedCard();
}

function markSearchedCard() {
  document.querySelectorAll("#hotspot-list .hotspot-card").forEach((li) => {
    const on = li.dataset.siteId === state.searchedSiteId;
    li.classList.toggle("is-searched", on);
    if (on && isTabActive("hotspots")) li.scrollIntoView({ block: "nearest", behavior: "smooth" });
  });
}

function setupHotspotSearch() {
  const input = document.getElementById("hotspot-search");
  const box = document.getElementById("hotspot-suggest");
  const msg = document.getElementById("hotspot-search-msg");
  let items = [];
  let active = -1;

  const close = () => { box.hidden = true; active = -1; };
  const showMsg = (text) => { msg.textContent = text; msg.hidden = !text; };
  const render = () => {
    const q = input.value.trim();
    showMsg("");
    if (!q) { items = []; close(); return; }
    items = searchHotspots(q);
    if (!items.length) {
      box.hidden = true;
      showMsg(state.features.length ? "No monitored hotspot found" : "Hotspots are still loading…");
      return;
    }
    box.innerHTML = items.map((f, i) => {
      const p = f.properties;
      return `<li role="option" data-i="${i}" class="${i === active ? "active" : ""}">
        <span class="dot" style="background:${TIER_COLORS[p.risk_tier]}"></span>
        <span class="sg-name">${esc(p.name)}</span>
        <span class="sg-tier">${esc(p.risk_tier)} · ${p.risk_score.toFixed(2)}</span></li>`;
    }).join("");
    box.hidden = false;
  };
  const choose = (f) => {
    if (!f) return;
    input.value = f.properties.name;
    close();
    showMsg("");
    goToHotspot(f.properties.site_id);
  };

  input.addEventListener("input", () => { active = -1; render(); });
  input.addEventListener("focus", () => { if (input.value.trim()) render(); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      if (!items.length) return;
      e.preventDefault();
      active = (active + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      render();
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (!items.length) render();
      if (items.length) choose(items[Math.max(0, active)]);
    } else if (e.key === "Escape") {
      close();
      showMsg("");
    }
  });
  box.addEventListener("mousedown", (e) => {
    const li = e.target.closest("li[data-i]");
    if (!li) return;
    e.preventDefault();  // keep focus in the input
    choose(items[Number(li.dataset.i)]);
  });
  input.addEventListener("blur", () => setTimeout(close, 150));
  // keep clicks/scrolls on the search box from reaching the map
  const wrap = input.closest(".map-search");
  L.DomEvent.disableClickPropagation(wrap);
  L.DomEvent.disableScrollPropagation(wrap);
}

/* ---- analyze: upload + real progress ---------------------------------------- */

function setupFileDrop() {
  const drop = document.getElementById("file-drop");
  const input = document.getElementById("video-file");
  const label = document.getElementById("file-drop-label");
  const show = () => {
    const f = input.files[0];
    label.textContent = f ? `${f.name} · ${(f.size / 1048576).toFixed(1)} MB` : "Choose a clip or drop it here";
    drop.classList.toggle("has-file", !!f);
  };
  input.addEventListener("change", show);
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
  drop.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length) { input.files = e.dataTransfer.files; show(); }
  });
}

function progressHtml(job, elapsedS) {
  const rows = (job.stages || []).map((s) => {
    let extra = "";
    if (s.state === "active" && s.total && s.done !== null) {
      extra = `<span class="stage-pct">${Math.min(100, Math.round((s.done / s.total) * 100))}%</span>`;
    }
    return `<li class="stage stage-${s.state}"><span class="stage-icon"></span><span>${esc(s.label)}</span>${extra}</li>`;
  }).join("");
  return `
    <div class="progress-card">
      <div class="section-label">Analyzing video <span class="muted">· ${elapsedS}s</span></div>
      <ol class="stages">${rows}</ol>
      <p class="hint">Each step is marked complete only when the backend finishes it.</p>
    </div>`;
}

async function analyzeVideo(evt) {
  evt.preventDefault();
  const fileInput = document.getElementById("video-file");
  const resultEl = document.getElementById("analyze-result");
  const progressEl = document.getElementById("analyze-progress");
  const btn = document.getElementById("btn-analyze");

  const file = fileInput.files[0];
  if (!file) {
    resultEl.innerHTML = noticeHtml("warn", "No video selected", "Choose a traffic clip to analyze first.");
    return;
  }
  const ext = (file.name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
  if (!ALLOWED_VIDEO.includes(ext)) {
    resultEl.innerHTML = noticeHtml("error", "Unsupported format", `"${file.name}" is not a supported video type. Use ${ALLOWED_VIDEO.join(", ")}.`);
    return;
  }
  if (file.size > 200 * 1024 * 1024) {
    resultEl.innerHTML = noticeHtml("error", "Video too large", "The demo accepts clips up to 200 MB. Trim the clip around the interaction and try again.");
    return;
  }

  const form = new FormData();
  form.append("file", file);
  form.append("detector", document.getElementById("detector").value);
  const s = state.selected;
  if (s && s.site_id) form.append("site_id", s.site_id);
  else if (s && s.isNew) {
    form.append("lat", s.lat);
    form.append("lon", s.lon);
    const nameInput = document.getElementById("loc-name");
    form.append("location_name", nameInput ? nameInput.value.trim() : "");
  }

  stopPlayer();
  btn.disabled = true;
  btn.textContent = "Analyzing…";
  resultEl.innerHTML = "";
  progressEl.innerHTML = `<div class="progress-card"><div class="section-label">Uploading video…</div><p class="hint">${esc(file.name)}</p></div>`;
  const started = Date.now();

  try {
    const res = await api("/api/analyses", { method: "POST", body: form }, 600000);
    const { analysis_id } = await res.json();
    const job = await pollJob(analysis_id, (j) => {
      progressEl.innerHTML = progressHtml(j, Math.round((Date.now() - started) / 1000));
    });
    progressEl.innerHTML = "";
    if (job.state === "error") {
      resultEl.innerHTML = noticeHtml("error", "Analysis could not be completed", job.error || "Unknown error.");
      return;
    }
    await showResult(job.result);
  } catch (err) {
    progressEl.innerHTML = "";
    resultEl.innerHTML = noticeHtml(
      "error",
      err.offline ? "Backend unavailable" : "Analysis failed",
      err.message
    );
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze video";
  }
}

function pollJob(analysisId, onUpdate) {
  return new Promise((resolve, reject) => {
    let failures = 0;
    const tick = async () => {
      try {
        const res = await api(`/api/analyses/${encodeURIComponent(analysisId)}/status`, {}, 10000);
        const job = await res.json();
        failures = 0;
        onUpdate(job);
        if (job.state === "done" || job.state === "error") { resolve(job); return; }
      } catch (err) {
        failures += 1;
        if (failures >= 5) { reject(err); return; }
      }
      state.polling = setTimeout(tick, 400);
    };
    tick();
  });
}

/* ---- analyze: result --------------------------------------------------------- */

async function showResult(result) {
  state.analysis = result;
  state.lastUpdate = result.site_update || null;
  const events = result.events || [];
  // Focus the most severe event first.
  state.activeEvent = events.length
    ? events.reduce((best, e) => ((e.peak_severity ?? e.severity) > (best.peak_severity ?? best.severity) ? e : best), events[0]).event_index
    : 0;

  if (result.location && result.location.site_id) {
    state.selected = { site_id: result.location.site_id, name: result.location.name, lat: result.location.lat, lon: result.location.lon, pinned: !result.location.has_historical_record, isNew: false };
  }
  renderResult();
  if (result.site_update) {
    await loadHotspots();
    focusSite(result.site_update.site_id, false);
    loadReport();
  }
  document.getElementById("analyze-result").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderResult() {
  const r = state.analysis;
  const el = document.getElementById("analyze-result");
  const events = r.events || [];
  stopPlayer();

  if (!events.length) {
    const noUsers = !r.road_users_detected;
    const th = r.thresholds || state.thresholds;
    const unstable = ((r.diagnostics || {}).conflicts || {}).camera_too_unstable;
    el.innerHTML = `
      <div class="result-head ${unstable ? "result-warn" : "result-safe"}">
        <div class="result-title">${unstable ? "Near-misses not assessed" : "No confirmed near-miss"}</div>
        <div class="result-sub">${noUsers
          ? "No road users were detected in this clip."
          : unstable
          ? "The camera moves or zooms too much in this clip to measure distances between road users. Use footage from a fixed camera."
          : `No conflict indicator exceeded the configured thresholds (within ${th.distance_m} m, TTC ≤ ${th.ttc_s} s, sustained over consecutive frames). This is a valid result.`}</div>
      </div>
      ${summaryHtml(r)}
      ${clipFactsHtml(r)}
      ${riskUpdateHtml(r)}
      ${fullClipHtml(r)}
      ${technicalHtml(r)}`;
    bindFullClip(el, r);
    return;
  }

  const e = events.find((x) => x.event_index === state.activeEvent) || events[0];
  const sev = e.severity_label || "Moderate";
  const nCrash = events.filter((x) => x.event_type === "crash").length;
  const nNear = events.length - nCrash;
  const headTitle = nCrash
    ? `Crash detected${nNear ? ` · ${nNear} near-miss${nNear > 1 ? "es" : ""}` : ""}`
    : "Near-miss detected";
  const isCrash = e.event_type === "crash";
  el.innerHTML = `
    <div class="result-head result-alert${nCrash ? " result-crash" : ""}" style="--sev:${SEVERITY_COLORS[sev] || "#c1442e"}">
      <div class="result-title">${headTitle}</div>
      <div class="result-sub">${isCrash ? "Crash" : "Near-miss"}: ${esc(e.pair_label)} · ${isCrash ? "impact" : "conflict peak"} at ${fmtTime(e.peak_t)} · ${esc(sev)} severity</div>
    </div>
    ${summaryHtml(r)}
    ${events.length > 1 ? eventPickerHtml(events) : ""}
    ${replayHtml(e)}
    ${metricsHtml(e)}
    <div class="two-col">
      <div><div class="section-label">What happened</div>${bullets(e.what_happened)}</div>
      <div><div class="section-label">Why flagged</div>${bullets(e.why_flagged)}</div>
    </div>
    <div class="event-summarize" id="event-summarize"></div>
    ${riskUpdateHtml(r)}
    ${reviewHtml(r, e)}
    <div class="actions">
      <button type="button" class="primary-btn" id="btn-pdf">Download PDF evidence report</button>
      <button type="button" class="ghost-btn" id="btn-email-pdf">Email Report</button>
      <span class="hint" id="pdf-status">1–2 page report for event ${e.event_index + 1}: evidence frame, metrics, observation, suggested review.</span>
    </div>
    ${fullClipHtml(r)}
    ${technicalHtml(r)}`;

  bindFullClip(el, r);
  el.querySelectorAll("[data-event]").forEach((b) => b.addEventListener("click", () => {
    state.activeEvent = Number(b.dataset.event);
    renderResult();
  }));
  document.getElementById("btn-pdf").addEventListener("click", () => downloadPdf(e));
  mountSummarize(document.getElementById("event-summarize"), (lang) => eventSummary(e, r, lang));
  document.getElementById("btn-email-pdf").addEventListener("click", () => {
    const label = (r.location && r.location.name) || r.source_name || `analysis ${r.analysis_id}`;
    openEmailDialog({
      url: `/api/analyses/${encodeURIComponent(r.analysis_id)}/email`,
      subject: `Verge Road Safety Report — ${label}`,
      attachment: `verge-road-safety-report-${slugify(label)}-event${e.event_index + 1}.pdf`,
      event: e.event_index,
    });
  });
  renderAssistantContext();
  if (e.evidence && e.evidence.frame_urls && e.evidence.frame_urls.length) {
    startPlayer(document.getElementById("replay"), e.evidence);
  }
}

function eventPickerHtml(events) {
  return `<div class="event-picker"><span class="section-label">${events.length} conflict events</span>${events.map((x) => `
    <button type="button" data-event="${x.event_index}" class="event-chip${x.event_index === state.activeEvent ? " active" : ""}">
      <i style="background:${SEVERITY_COLORS[x.severity_label] || "#6b7568"}"></i>${x.event_type === "crash" ? "<b>Crash</b> · " : ""}${fmtTime(x.peak_t)} · ${esc(x.pair_label || "")}
    </button>`).join("")}</div>`;
}

function replayHtml(e) {
  const ev = e.evidence;
  if (!ev || !ev.frame_urls || !ev.frame_urls.length) {
    return `<div class="replay-missing">No evidence replay was rendered for this event (replays are produced for the ${5} most severe events).</div>`;
  }
  const n = ev.frame_urls.length;
  const peakPct = n > 1 ? (ev.peak_index / (n - 1)) * 100 : 0;
  return `
    <div class="replay" id="replay">
      <div class="replay-head">
        <span class="section-label">AI conflict replay</span>
        <span class="replay-when">${e.event_type === "crash" ? "Crash (impact)" : "Near-miss detected"} at <b>${fmtTime(e.peak_t)}</b> · showing ${fmtTime(ev.start_t)} → ${fmtTime(ev.end_t)}</span>
      </div>
      <div class="replay-stage"><img alt="Annotated evidence frame" /><div class="replay-loading">Loading evidence frames…</div></div>
      <div class="replay-controls">
        <button type="button" data-act="play" class="ctl" title="Play / pause">Pause</button>
        <button type="button" data-act="prev" class="ctl" title="Previous frame">‹</button>
        <button type="button" data-act="next" class="ctl" title="Next frame">›</button>
        <div class="timeline">
          <input type="range" min="0" max="${n - 1}" value="0" step="1" aria-label="Replay position" />
          <span class="peak-mark" style="left:${peakPct.toFixed(2)}%" title="Conflict peak"></span>
        </div>
        <span class="replay-time">${fmtTime(ev.start_t)}</span>
        <button type="button" data-act="speed" class="ctl" title="Playback speed">1×</button>
        <button type="button" data-act="peak" class="ctl ctl-peak" title="Jump to the conflict peak">Peak</button>
      </div>
    </div>`;
}

function metricsHtml(e) {
  const cells = [
    e.event_type === "crash"
      ? ["Event", "Collision", `Observed crash: contact after a fast approach, then an impact (velocity change ${num(e.impact_dv_ms, 1)} m/s)`]
      : e.indicator === "close_pass"
      ? ["Indicator", "Close pass", "Proximity at speed: the pair passed within the close-pass clearance while moving (TTC does not apply to side-by-side passing)"]
      : ["TTC (min.)", e.min_ttc_s !== null && e.min_ttc_s !== undefined ? `${num(e.min_ttc_s)} s` : "n/a", "Lowest time-to-collision while the pair was on a collision course"],
    ["Minimum distance", e.contact_phase ? "≈ 0 m" : `${num(e.min_distance_m, 1)} m`, `Closest ${((state.analysis || {}).thresholds || {}).distance_kind || "separation"}, at ${fmtTime(e.min_distance_t)}`],
    ["Severity", `${esc(e.severity_label)}`, `Peak severity ${num(e.peak_severity)} on a 0–1 scale`],
    ["Event time", fmtTime(e.peak_t), "Conflict peak within the clip"],
  ];
  if (e.pet_s !== null && e.pet_s !== undefined) {
    cells.push(["PET", `${num(e.pet_s)} s`, "Post-encroachment time: gap between the two road users passing the same point"]);
  }
  cells.push(["Conflict duration", `${num(e.zone_duration_s, 1)} s`, "How long the conflict indicator held"]);
  return `<div class="metrics">${cells.map(([k, v, t]) => `
    <div class="metric${k === "Severity" ? " metric-sev" : ""}" title="${esc(t)}" ${k === "Severity" ? `style="--sev:${SEVERITY_COLORS[e.severity_label] || "#6b7568"}"` : ""}>
      <div class="metric-k">${esc(k)}</div><div class="metric-v">${v}</div>
    </div>`).join("")}</div>`;
}

function riskUpdateHtml(r) {
  const u = r.site_update;
  if (!u) {
    return `<div class="risk-update risk-none"><div class="section-label">Site risk</div>
      <p>This clip was not linked to a location, so the risk map was not changed. Choose a location in step 1 to add this evidence to a site.</p></div>`;
  }
  const b = u.before, a = u.after;
  if (!a || !b) return "";
  const dObs = a.observed_component - b.observed_component;
  const rankTxt = b.rank !== a.rank ? `#${b.rank} → #${a.rank}` : `#${a.rank} (unchanged)`;
  const changed = Math.abs(u.delta || 0) > 0.0005;
  return `
    <div class="risk-update">
      <div class="section-label">${changed ? "Risk priority updated" : "Site evidence recorded"} · ${esc(u.name)}</div>
      <table class="ledger">
        <tr><td>Historical evidence</td><td class="n">${a.historical_component.toFixed(2)}</td><td class="muted"></td></tr>
        <tr><td>AI-observed evidence</td><td class="n">${a.observed_component.toFixed(2)}</td><td class="muted">${dObs >= 0 ? "+" : ""}${dObs.toFixed(2)} from this video</td></tr>
        <tr class="total"><td>Updated review priority</td><td class="n">${a.risk_score.toFixed(2)}</td>
          <td><span class="score-badge" style="background:${TIER_COLORS[a.risk_tier]}">${esc(a.risk_tier)}</span> <span class="muted">was ${b.risk_score.toFixed(2)} ${esc(b.risk_tier)}</span></td></tr>
        <tr><td>Rank</td><td class="n">${rankTxt}</td><td class="muted">of ${a.of} locations</td></tr>
      </table>
      <p class="reason"><b>Reason:</b> ${esc(u.reason)}</p>
      ${(u.notes || []).map((n) => `<p class="hint">${esc(n)}</p>`).join("")}
      <p class="hint">${esc(u.scoring_note)} Historical crash data is kept separate and never modified.</p>
    </div>`;
}

function reviewHtml(r, e) {
  let items = r.site_after && r.site_after.brief ? r.site_after.brief.review : null;
  if (!items || !items.length) {
    items = e.kind === "vehicle_pedestrian"
      ? ["Review pedestrian crossing visibility and vehicle–pedestrian interaction at this location.", "Check approach speeds and whether crossing protection is adequate."]
      : ["Review junction priority, turning movements and approach speeds at this location."];
  }
  return `<div class="review"><div class="section-label">Suggested human review</div>${bullets(items)}
    <p class="hint">Suggestions for a road-safety engineer to confirm on site — not an automated decision.</p></div>`;
}

function clipFactsHtml(r) {
  return `<ul class="facts">
    <li><span>Clip</span><b>${num(r.duration_s, 1)} s</b></li>
    <li><span>Vehicles tracked</span><b>${r.vehicles_tracked}</b></li>
    <li><span>Pedestrians tracked</span><b>${r.pedestrians_tracked}</b></li>
  </ul>`;
}

const LABEL_NAMES = { person: "pedestrians", "two-wheeler": "two-wheelers (rider)", motorcycle: "motorcycles",
  car: "cars", bus: "buses", truck: "trucks", bicycle: "bicycles", vehicle: "vehicles", pedestrian: "pedestrians" };

/* One line saying which detector ran and what it tracked, plus any
 * reliability warnings from the backend's sanity checks. */
function summaryHtml(r) {
  const d = r.diagnostics || {};
  const det = r.detector === "object" ? "Object detector" : "Motion-based detector";
  const counts = Object.entries(r.road_users_by_class || {})
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `${v} ${LABEL_NAMES[k] || k}`)
    .join(", ");
  const secs = d.timings_s && d.timings_s.total_s ? ` · analysed in ${Math.round(d.timings_s.total_s)} s` : "";
  const warn = (d.warnings || []).length
    ? `<ul class="warn-list">${d.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : "";
  const note = d.detector_note ? `<p class="hint">${esc(d.detector_note)}</p>` : "";
  return `<p class="summary-line"><b>${det}</b> · ${counts ? `tracked ${esc(counts)}` : "no road users tracked"}${secs}</p>${note}${warn}`;
}

function fullClipHtml(r) {
  const pv = r.preview && r.preview.frame_urls && r.preview.frame_urls.length ? r.preview : null;
  const mp4 = r.annotated_video_url ? `<a href="${esc(apiBase() + r.annotated_video_url)}" download>Download annotated clip (.mp4)</a>` : "";
  if (!pv && !mp4) return "";
  const n = pv ? pv.frame_urls.length : 0;
  return `<details class="full-clip" id="full-clip"><summary>Full annotated clip</summary>
    ${pv ? `<div class="replay preview-player" id="preview-player">
      <div class="replay-stage"><img alt="Annotated clip preview" /><div class="replay-loading">Loading preview…</div></div>
      <div class="replay-controls">
        <button type="button" data-act="play" class="ctl">Pause</button>
        <button type="button" data-act="prev" class="ctl">‹</button>
        <button type="button" data-act="next" class="ctl">›</button>
        <div class="timeline"><input type="range" min="0" max="${n - 1}" value="0" step="1" aria-label="Preview position" /></div>
        <span class="replay-time">00:00.0</span>
        <button type="button" data-act="speed" class="ctl">1×</button>
      </div></div>
      <p class="hint">Preview at ${pv.fps} fps. ${mp4}</p>` : `<p class="hint">${mp4}</p>`}
  </details>`;
}

function bindFullClip(el, r) {
  const det = el.querySelector("#full-clip");
  if (!det || !r.preview) return;
  let started = false;
  det.addEventListener("toggle", () => {
    if (det.open && !started) {
      started = true;
      startPlayer(el.querySelector("#preview-player"), {
        frame_urls: r.preview.frame_urls, fps: r.preview.fps, start_t: 0, peak_index: -1,
      });
    }
  });
}

function technicalHtml(r) {
  const aside = (r.set_aside_events || []);
  const th = r.thresholds || {};
  const d = r.diagnostics || {};
  const ts = d.tracks || {};
  const cf = d.conflicts || {};
  const t = d.timings_s || {};
  const gm = cf.ground_model || {};
  const items = [
    `Detector: ${esc(d.detector_name || r.detector || "?")}`,
    `Video: ${esc(d.resolution || "?")} at ${d.source_fps ?? "?"} fps · ${d.frames_total ?? "?"} frames, detection on ${d.frames_analyzed ?? "?"} (${d.detect_fps ?? "?"} fps)`,
    `Thresholds: ${esc(th.distance_kind || "distance")} ≤ ${th.distance_m ?? "?"} m, TTC ≤ ${th.ttc_s ?? "?"} s` +
      (th.ttc_vv_s ? ` (vehicle–vehicle ≤ ${th.ttc_vv_s} s), collision course required, sustained ≥ ${th.min_duration_s} s` : "") +
      (th.pixels_per_meter ? ` · fixed demo scale ${th.pixels_per_meter} px/m` : ""),
  ];
  if (ts.raw_tracks !== undefined) {
    items.push(`Tracks: ${ts.confirmed_tracks} confirmed (of ${ts.raw_tracks} candidates; ${ts.unconfirmed_dropped} too short to confirm), at most ${ts.peak_simultaneous} visible at once`);
  }
  if (gm.mode) items.push(`Distance model: ${esc(gm.mode)} (estimated from road-user sizes)${d.camera && d.camera.moving_camera ? " · camera motion compensated" : ""}`);
  if (cf.potential_conflicts_not_confirmed !== undefined) {
    items.push(`${cf.potential_conflicts_not_confirmed} brief potential conflicts not confirmed (too short); ${cf.set_aside_no_clear_approach_or_overlap} set aside (no clear approach or boxes overlapping)`);
  }
  if (aside.length) items.push(`${aside.length} candidate pair(s) set aside: ${esc(aside[0].reason)}`);
  if (t.total_s) items.push(`Time: detection ${t.detect_s ?? "?"} s, tracking ${t.track_s ?? "?"} s, conflicts ${t.conflict_s ?? "?"} s, rendering ${t.render_s ?? "?"} s, total ${t.total_s} s`);
  items.push(`Analysis ID: <code>${esc(r.analysis_id || "")}</code> · source: ${esc(r.source_name || "")} (deleted after analysis)`);
  return `<details class="tech"><summary>Technical detail</summary>
    ${eventsTableHtml(r.events)}
    <ul class="tech-list">${items.map((x) => `<li>${x}</li>`).join("")}</ul></details>`;
}

function eventsTableHtml(events) {
  if (!events || !events.length) return "";
  const rows = events
    .slice()
    .sort((a, b) => (b.peak_severity ?? b.severity) - (a.peak_severity ?? a.severity))
    .map((e) => {
      const pair = esc(e.pair_label || (e.kind === "vehicle_pedestrian" ? "Vehicle ↔ Pedestrian" : "Vehicle ↔ Vehicle"));
      const ttc = e.min_ttc_s !== null && e.min_ttc_s !== undefined ? `${e.min_ttc_s.toFixed(2)} s` : "—";
      return `
        <tr>
          <td>${fmtTime(e.peak_t ?? e.t)}</td>
          <td>${pair} <span class="muted">#${e.a_id}/#${e.b_id}</span></td>
          <td>${(e.min_distance_m ?? e.distance_m).toFixed(1)} m</td>
          <td>${ttc}</td>
          <td>${e.pet_s !== null && e.pet_s !== undefined ? `${e.pet_s.toFixed(2)} s` : "—"}</td>
          <td>${esc(e.severity_label || "")} ${(e.peak_severity ?? e.severity).toFixed(2)}</td>
        </tr>`;
    })
    .join("");
  return `
    <table class="events-table">
      <thead><tr><th>Peak</th><th>Pair</th><th>Min dist.</th><th>TTC</th><th>PET</th><th>Severity</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

async function downloadPdf(e) {
  const btn = document.getElementById("btn-pdf");
  const status = document.getElementById("pdf-status");
  btn.disabled = true;
  btn.textContent = "Generating PDF…";
  try {
    await downloadBlob(e.report_pdf_url, `verge-evidence-event${e.event_index + 1}.pdf`);
    status.textContent = "Evidence report downloaded.";
  } catch (err) {
    status.innerHTML = `<span class="error-text">Report generation failed: ${esc(err.message)}</span>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Download PDF evidence report";
  }
}

/* ---- evidence replay player ----------------------------------------------------
 * The replay is a short JPEG frame sequence (rendered by the backend around
 * the conflict peak) rather than a video file: it plays in every browser with
 * no codec dependency and allows exact frame stepping. */

function stopPlayer() {
  (state.players || []).forEach((p) => p.stop());
  state.players = [];
}

function startPlayer(root, ev) {
  const img = root.querySelector(".replay-stage img");
  const loading = root.querySelector(".replay-loading");
  const range = root.querySelector('input[type="range"]');
  const timeEl = root.querySelector(".replay-time");
  const btnPlay = root.querySelector('[data-act="play"]');
  const btnSpeed = root.querySelector('[data-act="speed"]');
  const frames = ev.frame_urls.map((u) => apiBase() + u);
  const n = frames.length;
  const cache = frames.map((src) => { const im = new Image(); im.src = src; return im; });
  let idx = 0;
  let playing = true;
  let speed = 1;
  let timer = null;
  let holdUntil = 0;

  const show = (i) => {
    idx = (i + n) % n;
    img.src = frames[idx];
    range.value = idx;
    timeEl.textContent = fmtTime(ev.frame_times ? ev.frame_times[idx] : ev.start_t + idx / ev.fps);
    root.classList.toggle("at-peak", idx === ev.peak_index);
  };
  const step = () => {
    if (!playing) return;
    if (Date.now() < holdUntil) return;
    if (idx === n - 1) { holdUntil = Date.now() + 700; show(0); return; }
    show(idx + 1);
    if (idx === ev.peak_index) holdUntil = Date.now() + 500; // brief hold on the conflict peak
  };
  const restart = () => {
    clearInterval(timer);
    timer = setInterval(step, 1000 / (ev.fps * speed));
  };
  const setPlaying = (p) => { playing = p; btnPlay.textContent = p ? "Pause" : "Play"; };

  root.querySelector('[data-act="play"]').addEventListener("click", () => setPlaying(!playing));
  root.querySelector('[data-act="prev"]').addEventListener("click", () => { setPlaying(false); show(idx - 1); });
  root.querySelector('[data-act="next"]').addEventListener("click", () => { setPlaying(false); show(idx + 1); });
  const peakBtn = root.querySelector('[data-act="peak"]');
  if (peakBtn) peakBtn.addEventListener("click", () => { setPlaying(false); show(ev.peak_index); });
  btnSpeed.addEventListener("click", () => {
    speed = speed === 1 ? 0.5 : speed === 0.5 ? 0.25 : 1;
    btnSpeed.textContent = speed === 1 ? "1×" : speed === 0.5 ? "½×" : "¼×";
    restart();
  });
  range.addEventListener("input", () => { setPlaying(false); show(Number(range.value)); });

  let loaded = 0;
  const begin = () => { loading.hidden = true; show(0); restart(); };
  cache.forEach((im) => {
    const done = () => { loaded += 1; if (loaded === n) begin(); };
    if (im.complete) done(); else { im.onload = done; im.onerror = done; }
  });
  img.onerror = () => { loading.hidden = false; loading.textContent = "Evidence frames could not be loaded from the API."; };

  const player = { stop: () => clearInterval(timer) };
  state.players = (state.players || []).concat(player);
  return player;
}

/* ---- boot ------------------------------------------------------------------------ */

document.addEventListener("DOMContentLoaded", () => {
  // Optional ?api=http://host:port override, e.g. when port 8000 is taken.
  const apiParam = new URLSearchParams(location.search).get("api");
  if (apiParam) document.getElementById("api-base").value = apiParam;
  initMap();
  warnIfOpenedAsFile();
  setupTabs();
  setupLocationSearch();
  setupHotspotSearch();
  setupSpeech();
  setupAnalysisDialog();
  setupAssistant();
  setupEmailDialog();
  setupViz();
  setupFileDrop();
  document.getElementById("btn-refresh").addEventListener("click", refreshAll);
  document.getElementById("btn-run-pipeline").addEventListener("click", runPipeline);
  document.getElementById("analyze-form").addEventListener("submit", analyzeVideo);
  document.getElementById("btn-clear-evidence").addEventListener("click", clearEvidence);
  document.getElementById("btn-download-md").addEventListener("click", () =>
    downloadBlob("/api/report?download=1", "verge-review-priorities.md").catch((err) => alert(err.message)));
  document.getElementById("api-base").addEventListener("change", refreshAll);
  refreshAll();
  setInterval(checkApi, 15000);
});
