/* Verge dashboard client.
 * Fetches hotspots.geojson + report.md from the FastAPI backend
 * (backend/app/main.py) and renders them as a Leaflet map + report viewer.
 * No build step — plain HTML/JS so it's trivial to open and inspect.
 */

const TIER_COLORS = {
  Critical: "#c1442e",
  High: "#dd8a2e",
  Moderate: "#d8b73d",
  Low: "#3f8f5c",
};

let map;
let markersLayer;

function apiBase() {
  return document.getElementById("api-base").value.replace(/\/$/, "");
}

function initMap() {
  map = L.map("map").setView([18.5204, 73.8567], 12);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19,
  }).addTo(map);
  markersLayer = L.layerGroup().addTo(map);
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

  const marker = L.circleMarker([lat, lon], {
    radius,
    color,
    weight: 2,
    fillColor: color,
    fillOpacity: 0.45,
  });

  marker.bindPopup(popupHtml(p));
  return marker;
}

function popupHtml(p) {
  const factors = Object.entries(p.factors)
    .map(([k, v]) => `<li><code>${k}</code>: ${v}</li>`)
    .join("");
  return `
    <div class="popup">
      <strong>#${p.rank} ${p.name}</strong><br/>
      <span>Risk tier: <b>${p.risk_tier}</b> (score ${p.risk_score.toFixed(2)})</span>
      <ul style="margin:6px 0 0 16px;padding:0;font-size:12px;">
        <li>Historical accidents: ${p.accidents_count} (fatalities: ${p.fatalities})</li>
        <li>Detected near-misses: ${p.near_miss_count} (avg severity ${Number(p.avg_near_miss_severity).toFixed(2)})</li>
      </ul>
      <div style="font-size:11px;margin-top:6px;color:#6b7568;">Contributing factors:</div>
      <ul style="margin:2px 0 0 16px;padding:0;font-size:11px;">${factors}</ul>
    </div>
  `;
}

function renderHotspotList(features) {
  const list = document.getElementById("hotspot-list");
  if (!features.length) {
    list.innerHTML = `<li class="empty-state">No hotspots yet — click "Re-run pipeline".</li>`;
    return;
  }
  list.innerHTML = "";
  features
    .slice()
    .sort((a, b) => a.properties.rank - b.properties.rank)
    .forEach((f) => {
      const p = f.properties;
      const li = document.createElement("li");
      li.className = `hotspot-card tier-${p.risk_tier}`;
      li.innerHTML = `
        <h3>#${p.rank} ${p.name}</h3>
        <div class="meta">
          <span class="score-badge" style="background:${TIER_COLORS[p.risk_tier]}">${p.risk_tier} · ${p.risk_score.toFixed(2)}</span>
          <span>Accidents: ${p.accidents_count}</span>
          <span>Near-misses: ${p.near_miss_count}</span>
        </div>
      `;
      li.addEventListener("click", () => {
        const [lon, lat] = f.geometry.coordinates;
        map.setView([lat, lon], 15);
        markersLayer.eachLayer((m) => {
          if (m.getLatLng().lat === lat && m.getLatLng().lng === lon) m.openPopup();
        });
      });
      list.appendChild(li);
    });
}

async function loadHotspots() {
  const listEl = document.getElementById("hotspot-list");
  try {
    const res = await fetch(`${apiBase()}/api/hotspots`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const geojson = await res.json();
    markersLayer.clearLayers();
    geojson.features.forEach((f) => markerFor(f).addTo(markersLayer));
    if (geojson.features.length) {
      const group = L.featureGroup(markersLayer.getLayers());
      map.fitBounds(group.getBounds().pad(0.3));
    }
    renderHotspotList(geojson.features);
  } catch (err) {
    listEl.innerHTML = `<li class="empty-state">Couldn't load hotspots (${err.message}). Is the API running at ${apiBase()}? Try "Re-run pipeline" or start it with:<br/><code>uvicorn backend.app.main:app --reload --port 8000</code></li>`;
  }
}

async function loadReport() {
  const el = document.getElementById("report-content");
  try {
    const res = await fetch(`${apiBase()}/api/report`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const md = await res.text();
    el.innerHTML = tinyMarkdownToHtml(md);
  } catch (err) {
    el.innerHTML = `<p class="empty-state">Couldn't load report (${err.message}).</p>`;
  }
}

/* A deliberately small Markdown -> HTML converter — just enough for
 * report.py's output (headers, bold, tables, blockquote, hr, lists,
 * inline code). Not a general-purpose Markdown parser. */
function tinyMarkdownToHtml(md) {
  const lines = md.split("\n");
  let html = "";
  let inTable = false;
  let inList = false;

  const inline = (s) =>
    s
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

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
    if (inList && !/^-\s+/.test(line)) {
      html += "</ul>";
      inList = false;
    }

    if (/^#### /.test(line)) html += `<h4>${inline(line.slice(5))}</h4>`;
    else if (/^### /.test(line)) html += `<h3>${inline(line.slice(4))}</h3>`;
    else if (/^## /.test(line)) html += `<h2>${inline(line.slice(3))}</h2>`;
    else if (/^# /.test(line)) html += `<h1>${inline(line.slice(2))}</h1>`;
    else if (/^>\s?/.test(line)) html += `<blockquote>${inline(line.replace(/^>\s?/, ""))}</blockquote>`;
    else if (/^---$/.test(line)) html += "<hr/>";
    else if (line.trim() === "") html += "";
    else html += `<p>${inline(line)}</p>`;
  }
  if (inTable) html += "</tbody></table>";
  if (inList) html += "</ul>";
  return html;
}

function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    });
  });
}

async function refreshAll() {
  await Promise.all([loadHotspots(), loadReport()]);
}

async function runPipeline() {
  const btn = document.getElementById("btn-run-pipeline");
  btn.disabled = true;
  btn.textContent = "Running…";
  try {
    const res = await fetch(`${apiBase()}/api/pipeline/run`, { method: "POST" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    await refreshAll();
  } catch (err) {
    alert(`Pipeline run failed: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Re-run pipeline";
  }
}

/* ---- Analyze-a-video tab ---------------------------------------------
 * Uploads a clip to POST /api/analyze-video and renders the near-miss
 * verdict, the event table, and (if the backend produced one) the
 * annotated clip with tracked boxes + the flagged moment highlighted.
 */

function verdictBannerHtml(result) {
  const alertCls = result.near_miss_detected ? "verdict-alert" : "verdict-safe";
  const headline = result.near_miss_detected
    ? "\u26A0 Near-miss detected"
    : "\u2713 No near-miss detected";
  return `
    <div class="verdict-banner ${alertCls}">
      <strong>${headline}</strong>
      <p>${result.verdict}</p>
      <ul class="verdict-meta">
        <li>Duration: ${result.duration_s}s (${result.frames_processed} frames)</li>
        <li>Vehicles tracked: ${result.vehicles_tracked}</li>
        <li>Pedestrians tracked: ${result.pedestrians_tracked}</li>
        <li>Events flagged: ${result.event_count}</li>
      </ul>
    </div>
  `;
}

function eventsTableHtml(events) {
  if (!events || !events.length) return "";
  const rows = events
    .slice()
    .sort((a, b) => b.severity - a.severity)
    .map((e) => {
      const pair = e.kind === "vehicle_pedestrian" ? "Vehicle \u2194 Pedestrian" : "Vehicle \u2194 Vehicle";
      const ttc = e.ttc_s !== null && e.ttc_s !== undefined ? `${e.ttc_s.toFixed(1)}s` : "\u2014";
      return `
        <tr>
          <td>${e.t.toFixed(1)}s</td>
          <td>${pair}</td>
          <td>${e.distance_m.toFixed(1)} m</td>
          <td>${ttc}</td>
          <td>${Math.round(e.severity * 100)}%</td>
        </tr>
      `;
    })
    .join("");
  return `
    <table class="events-table">
      <thead><tr><th>Time</th><th>Pair</th><th>Distance</th><th>TTC</th><th>Severity</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

async function analyzeVideo(evt) {
  evt.preventDefault();
  const fileInput = document.getElementById("video-file");
  const resultEl = document.getElementById("analyze-result");
  const btn = document.getElementById("btn-analyze");
  if (!fileInput.files.length) return;

  const form = new FormData();
  form.append("file", fileInput.files[0]);

  btn.disabled = true;
  btn.textContent = "Analyzing…";
  resultEl.innerHTML = `<p class="empty-state">Running detection → tracking → conflict analysis on your clip — this can take a little while for longer videos…</p>`;

  try {
    const res = await fetch(`${apiBase()}/api/analyze-video`, { method: "POST", body: form });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${res.status}`);
    }
    const result = await res.json();

    let html = verdictBannerHtml(result) + eventsTableHtml(result.events);
    if (result.annotated_video_url) {
      html += `
        <h3 class="annotated-heading">Annotated clip</h3>
        <video class="annotated-video" controls src="${apiBase()}${result.annotated_video_url}"></video>
      `;
    }
    resultEl.innerHTML = html;
  } catch (err) {
    resultEl.innerHTML = `<p class="empty-state">Analysis failed (${err.message}). Is the API running at ${apiBase()}? See README.md.</p>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze video";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  warnIfOpenedAsFile();
  setupTabs();
  document.getElementById("btn-refresh").addEventListener("click", refreshAll);
  document.getElementById("btn-run-pipeline").addEventListener("click", runPipeline);
  document.getElementById("analyze-form").addEventListener("submit", analyzeVideo);
  refreshAll();
});
