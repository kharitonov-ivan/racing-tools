let analysis;
const state = { hoverValue: null, zoomAnchor: null, sector: 0, zoom: 1, xZoom: 1, range: null, axisMode: "distance" };
const channels = [
    { key: "speed", label: "Speed", min: 95, max: 215 },
    { key: "throttle", label: "Throttle", min: 0, max: 100 },
    { key: "brake", label: "Brake", min: 0, max: 100 },
    { key: "gear", label: "Gear", min: 1, max: 6 },
    { key: "rpm", label: "RPM", min: 4800, max: 7600 },
    { key: "steering", label: "Steering wheel angle", min: -85, max: 85 },
];
const centerCharts = [
    { key: "lineDelta", label: "Line distance", min: -3, max: 3 },
    { key: "timeDelta", label: "Time delta", min: -4.5, max: 0.4 },
];
const STORAGE_KEYS = {
    track: "analysis.trackPath",
    sessions: "analysis.sessionEntries",
};
const LEGACY_STORAGE_KEYS = {
    sessionA: "analysis.sessionAPath",
    sessionB: "analysis.sessionBPath",
};
const MAX_SESSION_ROWS = 6;

const map = document.getElementById("track-map");
const mapCtx = map.getContext("2d");
const chartsEl = document.getElementById("charts");
const centerChartsEl = document.getElementById("center-charts");
const mapLegend = document.getElementById("map-legend");
const compareLegend = document.getElementById("compare-legend");
const readout = document.getElementById("readout");
const sectorTable = document.getElementById("sector-table");
const sectorStrip = document.getElementById("sector-strip");
const titleEl = document.querySelector(".toolbar strong");
const loadButton = document.getElementById("load-data-btn");
const loadStatus = document.getElementById("load-status");
const trackPathInput = document.getElementById("track-path");
const sessionRowsEl = document.getElementById("session-rows");
const addSessionButton = document.getElementById("add-session-btn");
const chartCanvases = [];
const centerChartCanvases = [];
const workspace = document.querySelector(".workspace");
let activeResize = null;
let resizeFrame = 0;
analysis = makeDemoAnalysis();

void init();

async function init() {
    restorePathInputs();
    renderCharts();
    renderCenterCharts();
    bindControls();
    applyAnalysis(analysis, { status: "Demo data loaded. Start analysis_server.py to load real files." });
    await loadDefaults();
    await hydrateSessionRows();
}

function makeDemoAnalysis() {
    const laps = [
        buildDemoLap({
            slot: 1,
            id: 1,
            label: "Lap 1",
            color: "#ff4058",
            driver: "Ivan Kharitonov",
            source: "Demo",
            lapSeconds: 97.029,
            phase: 0,
            lineBias: 0,
            speedBias: -2,
            brakeBias: 0,
        }),
        buildDemoLap({
            slot: 2,
            id: 2,
            label: "Lap 2",
            color: "#6aa7ff",
            driver: "Matan Achituv",
            source: "Demo",
            lapSeconds: 92.988,
            phase: 0.042,
            lineBias: 0.028,
            speedBias: 2,
            brakeBias: -6,
        }),
    ];
    return {
        title: "Demo comparison analysis",
        trackPath: trackPathInput.value,
        trackPoints: laps[0].points.map(point => ({ x: point.x, y: point.y })),
        laps,
    };
}

function buildDemoLap(config) {
    const total = 4280;
    const sectors = [0, 0.13, 0.254, 0.422, 0.56, 0.736, 0.874, 1].map(value => value * total);
    const points = [];
    let elapsed = 0;
    for (let i = 0; i <= 720; i += 1) {
        const point = makeDemoPoint(i / 720, total, config);
        if (points.length) {
            const prev = points[points.length - 1];
            const metersPerSecond = Math.max(28, ((prev.speed + point.speed) * 0.5) / 3.6);
            elapsed += (point.distance - prev.distance) / metersPerSecond;
        }
        points.push({ ...point, time: elapsed });
    }
    const scale = config.lapSeconds / points[points.length - 1].time;
    points.forEach(point => { point.time *= scale; });
    return { ...config, total, time: formatLapTime(config.lapSeconds), sectors, points };
}

function makeDemoPoint(progress, total, config) {
    const angle = progress * Math.PI * 2;
    const phase = angle + config.phase * Math.PI * 2;
    const radius = 1.08 + 0.18 * Math.sin(3 * phase) + 0.12 * Math.cos(5 * phase);
    const lineOffset = config.lineBias * Math.sin(6 * angle - 0.2) + 0.012 * Math.cos(11 * angle + config.phase);
    const offsetScale = 1 + lineOffset;
    const x = offsetScale * (radius * Math.cos(angle) + 0.22 * Math.sin(2 * angle));
    const y = offsetScale * (0.72 * radius * Math.sin(angle) + 0.18 * Math.cos(4 * angle));
    const corner = Math.max(0, Math.sin(7 * phase - 0.78));
    const brake = clamp(Math.pow(Math.max(0, Math.sin(7 * phase + 0.52)), 8) * 100 + config.brakeBias, 0, 100);
    const speed = clamp(199 - 90 * corner + 9 * Math.sin(13 * phase) + config.speedBias, 96, 212);
    const throttle = clamp(101 - brake * 1.28 - corner * 31 + config.speedBias * 0.8, 0, 100);
    const gear = clamp(Math.round((speed - 78) / 24), 3, 6);
    const rpm = clamp(5200 + gear * 280 + speed * 4.8 + 160 * Math.sin(9 * phase), 4800, 7600);
    const steering = 58 * Math.sin(7 * phase - 0.22) + 18 * Math.sin(14 * phase + config.phase * 4);
    return { distance: progress * total, x, y, speed, throttle, brake, gear, rpm, steering };
}

function bindControls() {
    window.addEventListener("resize", resizeAll);
    document.getElementById("zoom-in").onclick = () => { state.zoom = Math.min(2.2, state.zoom + 0.15); drawAll(); };
    document.getElementById("zoom-out").onclick = () => { state.zoom = Math.max(0.75, state.zoom - 0.15); drawAll(); };
    document.getElementById("prev-btn").onclick = () => stepSector(-1);
    document.getElementById("next-btn").onclick = () => stepSector(1);
    document.getElementById("navigate-btn").onclick = resetSectorSelection;
    document.getElementById("view-select").onchange = event => setView(event.target.value);
    document.getElementById("axis-select").onchange = event => setAxisMode(event.target.value);
    document.getElementById("x-zoom-in-btn").onclick = () => zoomHorizontal(1);
    document.getElementById("x-zoom-out-btn").onclick = () => zoomHorizontal(-1);
    document.getElementById("x-zoom-reset-btn").onclick = resetHorizontalZoom;
    trackPathInput.addEventListener("change", onTrackPathChanged);
    document.querySelector('[data-pick-target="track"]').addEventListener("click", () => void onPickTrackPath());
    addSessionButton.addEventListener("click", () => addSessionRow());
    loadButton.onclick = () => void loadAnalysisData();
    bindSplitters();
    updateSessionControls();
}

function bindSplitters() {
    document.querySelectorAll(".splitter").forEach(splitter => splitter.addEventListener("mousedown", startResize));
    document.addEventListener("mousemove", resizeMove);
    document.addEventListener("mouseup", stopResize);
    window.addEventListener("blur", stopResize);
}

async function loadDefaults() {
    try {
        const response = await fetch("/api/analysis/defaults");
        const contentType = response.headers.get("content-type") || "";
        if (!response.ok || !contentType.includes("application/json")) return;
        const payload = await response.json();
        if (payload.trackPath && !trackPathInput.value.trim()) {
            trackPathInput.value = payload.trackPath;
            persistPathInputs();
        }
    } catch {
        // Static HTTP server mode has no API; keep demo data.
    }
}

async function onPickTrackPath() {
    try {
        const path = await pickPath("track", "directory");
        trackPathInput.value = path;
        persistPathInputs();
        setLoadStatus(`Selected ${path}`, false);
        await refreshAllSessionInfos({ silent: true });
    } catch (error) {
        setLoadStatus(error.message || "Failed to open picker.", true);
    }
}

function onTrackPathChanged() {
    persistPathInputs();
    void refreshAllSessionInfos({ silent: true });
}

async function pickPath(target, mode) {
    const payload = await postJson("/api/analysis/pick", { target, mode }, "Failed to open picker.");
    return payload.path || "";
}

async function postJson(url, body, fallbackMessage) {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || fallbackMessage);
        return payload;
    }
    const message = (await response.text()).trim();
    throw new Error(message || `${fallbackMessage} Start analysis_server.py to enable local loading.`);
}

function addSessionRow(savedEntry = {}) {
    if (sessionRows().length >= MAX_SESSION_ROWS) return null;
    const row = createSessionRow(savedEntry);
    sessionRowsEl.appendChild(row);
    updateSessionControls();
    persistPathInputs();
    return row;
}

function createSessionRow(savedEntry = {}) {
    const row = document.createElement("div");
    row.className = "session-row";
    row.dataset.preferredLapId = savedEntry.lapId == null ? "best" : String(savedEntry.lapId);
    row.innerHTML = `
        <div class="session-row-top">
            <div class="session-select-wrap">
                <input class="session-path" type="text" spellcheck="false" placeholder="Telemetry file or folder">
                <div class="session-meta session-driver"></div>
            </div>
            <div class="picker-button-group">
                <button type="button" class="picker-btn session-browse" data-mode="file">File</button>
                <button type="button" class="picker-btn session-browse" data-mode="directory">Folder</button>
            </div>
            <select class="session-lap" disabled></select>
            <button type="button" class="picker-btn session-remove">Remove</button>
        </div>
        <div class="session-row-bottom">
            <span class="session-meta session-status"></span>
            <button type="button" class="picker-btn session-refresh">Refresh laps</button>
        </div>`;
    sessionPathInput(row).value = savedEntry.path || "";
    resetSessionInfo(row, false);
    bindSessionRow(row);
    return row;
}

function bindSessionRow(row) {
    sessionPathInput(row).addEventListener("change", () => void onSessionPathEdited(row));
    sessionPathInput(row).addEventListener("keydown", event => {
        if (event.key !== "Enter") return;
        event.preventDefault();
        void refreshSessionInfo(row);
    });
    sessionLapSelect(row).addEventListener("change", () => {
        row.dataset.preferredLapId = sessionLapSelect(row).value || "best";
        persistPathInputs();
    });
    row.querySelectorAll(".session-browse").forEach(button => {
        button.addEventListener("click", () => void onPickSessionPath(row, button.dataset.mode));
    });
    row.querySelector(".session-refresh").addEventListener("click", () => void refreshSessionInfo(row));
    row.querySelector(".session-remove").addEventListener("click", () => removeSessionRow(row));
}

function removeSessionRow(row) {
    if (sessionRows().length === 1) {
        sessionPathInput(row).value = "";
        row.dataset.preferredLapId = "best";
        resetSessionInfo(row, false);
    } else {
        row.remove();
    }
    updateSessionControls();
    persistPathInputs();
}

function updateSessionControls() {
    const rows = sessionRows();
    addSessionButton.disabled = rows.length >= MAX_SESSION_ROWS;
    rows.forEach(row => {
        row.querySelector(".session-remove").disabled = rows.length === 1;
    });
}

async function hydrateSessionRows() {
    const rows = sessionRows();
    await Promise.all(rows.map(row => refreshSessionInfo(row, { silent: true, statusMessage: false })));
}

async function onSessionPathEdited(row) {
    row.dataset.preferredLapId = "best";
    resetSessionInfo(row, false);
    await refreshSessionInfo(row, { silent: true });
}

async function onPickSessionPath(row, mode) {
    try {
        const path = await pickPath("session", mode);
        sessionPathInput(row).value = path;
        row.dataset.preferredLapId = "best";
        resetSessionInfo(row, false);
        persistPathInputs();
        setLoadStatus(`Selected ${path}`, false);
        await refreshSessionInfo(row, { silent: true });
    } catch (error) {
        setLoadStatus(error.message || "Failed to open picker.", true);
    }
}

async function refreshAllSessionInfos(options = {}) {
    await Promise.all(sessionRows().map(row => refreshSessionInfo(row, options)));
}

async function refreshSessionInfo(row, options = {}) {
    const path = sessionPathInput(row).value.trim();
    if (!path) {
        row.dataset.preferredLapId = "best";
        resetSessionInfo(row, false);
        persistPathInputs();
        return;
    }

    const requestId = String((Number(row.dataset.requestId) || 0) + 1);
    row.dataset.requestId = requestId;
    setSessionBusy(row, true);
    setSessionStatus(row, "Inspecting laps…", false);
    if (options.statusMessage !== false) setLoadStatus(`Inspecting ${path}…`, false);

    try {
        const payload = await postJson(
            "/api/analysis/session-info",
            {
                trackPath: trackPathInput.value.trim(),
                sessionPath: path,
            },
            "Failed to inspect session.",
        );
        if (row.dataset.requestId !== requestId) return;
        applySessionInfo(row, payload, options.preferredLapId ?? row.dataset.preferredLapId);
        if (options.statusMessage !== false && !options.silent) setLoadStatus(`Loaded lap list from ${payload.path}`, false);
    } catch (error) {
        if (row.dataset.requestId !== requestId) return;
        resetSessionInfo(row, true, error.message || "Failed to inspect session.");
        if (!options.silent) setLoadStatus(error.message || "Failed to inspect session.", true);
    } finally {
        if (row.dataset.requestId === requestId) setSessionBusy(row, false);
        persistPathInputs();
    }
}

function applySessionInfo(row, info, preferredLapId) {
    row._sessionInfo = info;
    row.dataset.preferredLapId = preferredLapId == null ? "best" : String(preferredLapId);
    sessionDriverMeta(row).textContent = `${info.driver} · ${info.laps.length} lap${info.laps.length === 1 ? "" : "s"}`;
    populateLapSelect(row, info, row.dataset.preferredLapId);
    setSessionStatus(row, `${info.bestLabel} · ${info.path}`, false);
}

function populateLapSelect(row, info, preferredLapId) {
    const lapSelect = sessionLapSelect(row);
    lapSelect.innerHTML = "";
    if (!info) {
        lapSelect.add(new Option("Choose session first", ""));
        lapSelect.value = "";
        lapSelect.disabled = true;
        row.dataset.preferredLapId = "best";
        return;
    }
    lapSelect.add(new Option(`Best lap · ${info.bestTime}`, "best"));
    info.laps.forEach(lap => lapSelect.add(new Option(lap.label, String(lap.id))));
    const desired = preferredLapId == null ? "best" : String(preferredLapId);
    const allowed = Array.from(lapSelect.options).some(option => option.value === desired);
    lapSelect.value = allowed ? desired : "best";
    lapSelect.disabled = false;
    row.dataset.preferredLapId = lapSelect.value || "best";
}

function resetSessionInfo(row, isError, message = "Choose a telemetry file or folder, then refresh laps.") {
    row._sessionInfo = null;
    sessionDriverMeta(row).textContent = "";
    populateLapSelect(row, null);
    setSessionStatus(row, message, isError);
}

function setSessionBusy(row, busy) {
    row.querySelectorAll("button").forEach(button => {
        button.disabled = busy || (button.classList.contains("session-remove") && sessionRows().length === 1);
    });
    sessionLapSelect(row).disabled = busy || !row._sessionInfo;
}

function setSessionStatus(row, message, isError) {
    const status = row.querySelector(".session-status");
    status.textContent = message;
    status.classList.toggle("error", Boolean(isError));
}

async function loadAnalysisData() {
    const trackPath = trackPathInput.value.trim();
    const sessionEntries = collectSessionEntries();
    if (!trackPath || sessionEntries.length === 0) {
        setLoadStatus("Enter a track path and at least one telemetry session path.", true);
        return;
    }
    persistPathInputs();
    loadButton.disabled = true;
    setLoadStatus("Loading track and telemetry through racing_tools…", false);
    try {
        const payload = await postJson(
            "/api/analysis/load",
            { trackPath, sessionEntries },
            "Failed to load analysis data.",
        );
        const compareNote = payload.laps && payload.laps.length > 2 ? " · sectors and delta charts compare the first two laps" : "";
        applyAnalysis(normalizeAnalysis(payload), {
            status: `Loaded ${payload.laps.length} lap${payload.laps.length === 1 ? "" : "s"} from ${payload.title || trackPath}${compareNote}`,
        });
    } catch (error) {
        setLoadStatus(error.message || "Failed to load analysis data.", true);
    } finally {
        loadButton.disabled = false;
    }
}

function collectSessionEntries() {
    return sessionRows()
        .map(row => {
            const path = sessionPathInput(row).value.trim();
            if (!path) return null;
            const lapValue = sessionLapSelect(row).value || row.dataset.preferredLapId || "best";
            return {
                path,
                lapId: lapValue === "best" ? "best" : Number(lapValue),
            };
        })
        .filter(Boolean);
}

function applyAnalysis(nextAnalysis, options = {}) {
    analysis = nextAnalysis;
    titleEl.textContent = nextAnalysis.title || "Telemetry analysis";
    state.sector = 0;
    state.range = null;
    state.xZoom = 1;
    state.zoomAnchor = null;
    clearHover();
    renderLaps();
    renderSectors();
    resizeAll();
    if (options.status) setLoadStatus(options.status, false);
}

function normalizeAnalysis(payload) {
    return {
        title: payload.title || "Telemetry analysis",
        trackPath: payload.trackPath || trackPathInput.value.trim(),
        trackPoints: (payload.trackPoints || []).map(point => ({ x: Number(point.x), y: Number(point.y) })),
        laps: (payload.laps || []).map((lap, index) => ({
            slot: Number(lap.slot || index + 1),
            id: Number(lap.id),
            label: lap.label || `Lap ${Number(lap.id)}`,
            color: lap.color,
            driver: lap.driver || "Unknown",
            source: lap.source || "",
            lapSeconds: Number(lap.lapSeconds),
            time: lap.time || formatLapTime(Number(lap.lapSeconds)),
            total: Number(lap.total),
            sectors: (lap.sectors || []).map(Number),
            points: (lap.points || []).map(point => ({
                distance: Number(point.distance),
                time: Number(point.time),
                x: Number(point.x),
                y: Number(point.y),
                speed: Number(point.speed),
                throttle: Number(point.throttle),
                brake: Number(point.brake),
                gear: Number(point.gear),
                rpm: Number(point.rpm),
                steering: Number(point.steering),
            })),
        })),
    };
}

function renderLaps() {
    const baseLap = laps()[0];
    document.getElementById("lap-list").innerHTML = laps().map((lap, index) => {
        const delta = index === 0 ? "Reference" : formatDelta(lap.lapSeconds - baseLap.lapSeconds);
        return `
        <div class="lap-row">
            <span class="color-dot" style="background:${lap.color}"></span>
            <span class="lap-main">
                <span class="lap-topline"><span class="lap-time">${escapeHtml(lap.time)}</span><span class="lap-delta">${escapeHtml(delta)}</span></span>
                <span class="driver">${escapeHtml(lap.driver)}</span>
                <span class="lap-source">${escapeHtml(displayLapName(lap))} · ${escapeHtml(basenamePath(lap.source) || lap.source || "Demo")}</span>
            </span>
            <strong class="lap-index">${lap.slot || index + 1}</strong>
        </div>`;
    }).join("");
    mapLegend.innerHTML = laps().map(lap => `
        <span class="legend-row"><span class="color-dot" style="background:${lap.color}"></span>${escapeHtml(displayLapSummary(lap))} · ${escapeHtml(lap.driver)}</span>
    `).join("");
    const pair = comparisonLaps();
    compareLegend.innerHTML = pair.map(lap => `
        <span class="compare-pill"><span class="color-dot" style="background:${lap.color}"></span><strong>${escapeHtml(displayLapSummary(lap))}</strong><span>${escapeHtml(lap.driver)}</span></span>
    `).join("");
}

function renderSectors() {
    const count = sectorCount();
    const baseLap = laps()[0];
    const compareLap = comparisonLaps()[1] || null;
    if (!count || !baseLap) {
        sectorTable.innerHTML = "";
        sectorStrip.innerHTML = "";
        return;
    }

    const rows = [];
    const chips = [];
    const gridColumns = compareLap ? "auto minmax(76px,1fr) minmax(76px,1fr) auto" : "auto minmax(76px,1fr)";
    for (let index = 0; index < count; index += 1) {
        const baseTime = getSectorTime(baseLap, index);
        const active = Boolean(state.range && state.sector === index);
        if (compareLap) {
            const hasCompareSector = index + 1 < compareLap.sectors.length;
            const compareTime = hasCompareSector ? getSectorTime(compareLap, index) : null;
            const delta = compareTime === null ? null : compareTime - baseTime;
            const deltaClass = delta === null ? "" : delta <= 0 ? "better" : "worse";
            rows.push(`
        <div class="sector-row ${active ? "active" : ""}" data-sector="${index}" style="grid-template-columns:${gridColumns}">
            <span class="sector-label">S${index + 1}</span>
            <span class="sector-time">${formatSeconds(baseTime)}</span>
            <span class="sector-time">${compareTime === null ? "—" : formatSeconds(compareTime)}</span>
            <span class="sector-delta ${deltaClass}">${delta === null ? "—" : formatDelta(delta)}</span>
        </div>`);
            chips.push(`<button class="sector-chip ${active ? "active" : ""}" data-sector="${index}">S${index + 1}<small>${delta === null ? "—" : formatDelta(delta)}</small></button>`);
            continue;
        }
        rows.push(`
        <div class="sector-row ${active ? "active" : ""}" data-sector="${index}" style="grid-template-columns:${gridColumns}">
            <span class="sector-label">S${index + 1}</span>
            <span class="sector-time">${formatSeconds(baseTime)}</span>
        </div>`);
        chips.push(`<button class="sector-chip ${active ? "active" : ""}" data-sector="${index}">S${index + 1}<small>${formatSeconds(baseTime)}s</small></button>`);
    }

    sectorTable.innerHTML = compareLap ? `
        <div class="gap-header" style="grid-template-columns:${gridColumns}">
            <span></span>
            <span class="lap-head"><span class="color-dot" style="background:${baseLap.color}"></span>${escapeHtml(displayLapLabel(baseLap))}</span>
            <span class="lap-head"><span class="color-dot" style="background:${compareLap.color}"></span>${escapeHtml(displayLapLabel(compareLap))}</span>
            <span>Δ</span>
        </div>
        ${rows.join("")}` : `
        <div class="gap-header" style="grid-template-columns:${gridColumns}">
            <span></span>
            <span class="lap-head"><span class="color-dot" style="background:${baseLap.color}"></span>${escapeHtml(displayLapLabel(baseLap))}</span>
        </div>
        ${rows.join("")}`;
    sectorStrip.innerHTML = chips.join("");
    document.querySelectorAll("[data-sector]").forEach(element => {
        element.onclick = () => selectSector(Number(element.dataset.sector));
    });
}

function renderCharts() {
    chartCanvases.length = 0;
    chartsEl.innerHTML = "";
    channels.forEach(channel => createChartCard(chartsEl, chartCanvases, channel));
}

function renderCenterCharts() {
    centerChartCanvases.length = 0;
    centerChartsEl.innerHTML = "";
    centerCharts.forEach(chart => createChartCard(centerChartsEl, centerChartCanvases, chart));
}

function createChartCard(container, registry, chart) {
    const card = document.createElement("div");
    card.className = "chart-card";
    card.innerHTML = `
        <span class="chart-label">${chart.label}</span>
        <span class="chart-bound chart-bound-max"></span>
        <span class="chart-bound chart-bound-min"></span>
        <canvas></canvas>`;
    container.appendChild(card);
    const canvas = card.querySelector("canvas");
    canvas.addEventListener("mousemove", onChartHover);
    canvas.addEventListener("mouseleave", () => {
        clearHover();
        drawAll();
    });
    registry.push({
        canvas,
        chart,
        minLabel: card.querySelector(".chart-bound-min"),
        maxLabel: card.querySelector(".chart-bound-max"),
    });
}

function drawAll() {
    drawMap();
    chartCanvases.forEach(entry => drawOverlayChart(entry));
    centerChartCanvases.forEach(entry => drawDerivedChart(entry));
    updateReadout();
}

function drawMap() {
    const width = map.width;
    const height = map.height;
    mapCtx.clearRect(0, 0, width, height);
    mapCtx.fillStyle = "#080a0d";
    mapCtx.fillRect(0, 0, width, height);
    const toScreen = mapScale(width, height);
    drawTrackPath(mapCtx, trackPoints(), toScreen, "#526070", 8, 0.75);
    overlayLaps().forEach(lap => drawTrackPath(mapCtx, lap.points, toScreen, lap.color, 3, 0.9));
    if (state.range) laps().forEach(lap => drawHighlightedSegment(mapCtx, lap, toScreen, state.range));
    if (state.hoverValue === null) return;
    overlayLaps().forEach(lap => {
        const point = getHoverPoint(lap);
        if (point) drawMapMarker(mapCtx, toScreen(point), lap.color);
    });
}

function mapScale(width, height) {
    const allPoints = [...trackPoints(), ...laps().flatMap(lap => lap.points)];
    if (!allPoints.length) {
        return () => ({ x: width / 2, y: height / 2 });
    }
    const xs = allPoints.map(point => point.x);
    const ys = allPoints.map(point => point.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const scale = Math.min(width / (maxX - minX || 1), height / (maxY - minY || 1)) * 0.34 * state.zoom;
    return point => ({
        x: width / 2 + (point.x - (minX + maxX) / 2) * scale,
        y: height / 2 - (point.y - (minY + maxY) / 2) * scale,
    });
}

function drawTrackPath(ctx, points, toScreen, color, lineWidth, alpha) {
    if (!points.length) return;
    ctx.beginPath();
    points.forEach((point, index) => {
        const screen = toScreen(point);
        index ? ctx.lineTo(screen.x, screen.y) : ctx.moveTo(screen.x, screen.y);
    });
    ctx.strokeStyle = color;
    ctx.globalAlpha = alpha;
    ctx.lineWidth = lineWidth;
    ctx.lineJoin = "round";
    ctx.stroke();
    ctx.globalAlpha = 1;
}

function drawHighlightedSegment(ctx, lap, toScreen, range) {
    const points = lap.points.filter(point => point.distance >= range[0] && point.distance <= range[1]);
    if (!points.length) return;
    drawTrackPath(ctx, points, toScreen, "rgba(255,255,255,0.28)", 6, 1);
    drawTrackPath(ctx, points, toScreen, lap.color, 4, 1);
}

function drawMapMarker(ctx, point, color) {
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(point.x, point.y, 5.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.stroke();
}

function drawOverlayChart(entry) {
    const { canvas, chart } = entry;
    const ctx = canvas.getContext("2d");
    const range = plotRange();
    const valueRange = overlayChartValueRange(chart);
    updateChartBounds(entry, chart, valueRange);
    drawChartFrame(ctx, canvas.width, canvas.height);
    overlayLaps().forEach(lap => drawSeries(ctx, chart, lap, range, valueRange));
    drawCursor(ctx, canvas.width, canvas.height, range);
}

function drawSeries(ctx, chart, lap, range, valueRange) {
    const points = lap.points.filter(point => isPointVisible(point, range));
    if (!points.length) return;
    ctx.beginPath();
    points.forEach((point, index) => {
        const x = scaleAxis(getAxisValue(point), range, ctx.canvas.width);
        const y = scaleValue(point[chart.key], valueRange.min, valueRange.max, ctx.canvas.height);
        index ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.strokeStyle = lap.color;
    ctx.lineWidth = 2;
    ctx.stroke();
}

function drawDerivedChart(entry) {
    const { canvas, chart } = entry;
    const ctx = canvas.getContext("2d");
    const range = plotRange();
    const [baseLap, compareLap] = comparisonLaps();
    drawChartFrame(ctx, canvas.width, canvas.height);
    if (!baseLap || !compareLap) {
        updateChartBounds(entry, chart, null);
        drawEmptyMessage(ctx, canvas.width, canvas.height, "Load a second lap for delta");
        return;
    }
    const points = baseLap.points.filter(point => isPointVisible(point, range));
    const valueRange = derivedChartValueRange(chart, points);
    updateChartBounds(entry, chart, valueRange);
    drawReferenceLine(ctx, canvas.width, canvas.height, valueRange.min, valueRange.max, 0, "rgba(255,64,88,0.45)");
    if (!points.length) return;
    ctx.beginPath();
    points.forEach((point, index) => {
        const pointIndex = nearestIndexByKey(baseLap, point.distance, "distance");
        const value = chart.key === "timeDelta" ? getTimeDelta(point) : getLineDelta(pointIndex, point);
        const x = scaleAxis(getAxisValue(point), range, canvas.width);
        const y = scaleValue(value, valueRange.min, valueRange.max, canvas.height);
        index ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.strokeStyle = compareLap.color;
    ctx.lineWidth = 2;
    ctx.stroke();
    drawCursor(ctx, canvas.width, canvas.height, range);
}

function overlayChartValueRange(chart) {
    const range = plotRange();
    const values = overlayLaps().flatMap(lap => lap.points.filter(point => isPointVisible(point, range)).map(point => point[chart.key]));
    return computeValueRange(values, chart);
}

function derivedChartValueRange(chart, points) {
    const baseLap = laps()[0];
    const values = points.map(point => {
        const pointIndex = nearestIndexByKey(baseLap, point.distance, "distance");
        return chart.key === "timeDelta" ? getTimeDelta(point) : getLineDelta(pointIndex, point);
    });
    return computeValueRange(values, chart);
}

function computeValueRange(values, chart) {
    const filtered = values.filter(value => Number.isFinite(value));
    if (!filtered.length) {
        return { min: chart.min, max: chart.max };
    }
    let min = Math.min(...filtered);
    let max = Math.max(...filtered);
    if (min === max) {
        const pad = Math.max(Math.abs(min) * 0.08, 1);
        min -= pad;
        max += pad;
    } else {
        const pad = (max - min) * 0.08;
        min -= pad;
        max += pad;
    }
    if (Number.isFinite(chart.min)) min = Math.max(min, Math.min(...filtered, chart.min));
    if (Number.isFinite(chart.max)) max = Math.min(max, Math.max(...filtered, chart.max));
    if (min === max) max = min + 1;
    return { min, max };
}

function updateChartBounds(entry, chart, valueRange) {
    if (!valueRange) {
        entry.maxLabel.textContent = "";
        entry.minLabel.textContent = "";
        return;
    }
    entry.maxLabel.textContent = formatChartBound(chart, valueRange.max);
    entry.minLabel.textContent = formatChartBound(chart, valueRange.min);
}

function drawChartFrame(ctx, width, height) {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, width, height);
    drawGrid(ctx, width, height);
}

function drawEmptyMessage(ctx, width, height, message) {
    ctx.fillStyle = "rgba(148,163,184,0.72)";
    ctx.font = "500 14px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(message, width / 2, height / 2);
}

function drawGrid(ctx, width, height) {
    ctx.strokeStyle = "rgba(148,163,184,0.22)";
    ctx.setLineDash([5, 5]);
    for (let index = 1; index < 4; index += 1) {
        ctx.beginPath();
        ctx.moveTo(0, height * index / 4);
        ctx.lineTo(width, height * index / 4);
        ctx.stroke();
    }
    ctx.setLineDash([]);
}

function drawReferenceLine(ctx, width, height, min, max, value, color) {
    const y = scaleValue(value, min, max, height);
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
}

function drawCursor(ctx, width, height, range) {
    if (state.hoverValue === null || state.hoverValue < range[0] || state.hoverValue > range[1]) return;
    const x = scaleAxis(state.hoverValue, range, width);
    ctx.strokeStyle = "rgba(255,255,255,0.72)";
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
}

function onChartHover(event) {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const range = plotRange();
    state.hoverValue = range[0] + x * (range[1] - range[0]);
    drawAll();
}

function updateReadout() {
    const idleText = `Hover telemetry · ${state.axisMode === "distance" ? "distance axis" : "time axis"}`;
    if (state.hoverValue === null) {
        readout.textContent = idleText;
        return;
    }
    const [baseLap, compareLap] = comparisonLaps();
    const base = baseLap ? getHoverPoint(baseLap) : null;
    const compare = compareLap ? getHoverPoint(compareLap) : null;
    if (!base) {
        readout.textContent = idleText;
        return;
    }
    const lead = state.axisMode === "distance" ? `${(state.hoverValue / 1000).toFixed(2)} km` : `${state.hoverValue.toFixed(2)} s`;
    if (!compare || !compareLap) {
        readout.textContent = `${lead} · V ${base.speed.toFixed(0)} km/h · T ${base.throttle.toFixed(0)} · B ${base.brake.toFixed(0)} · G ${base.gear}`;
        return;
    }
    const deltaSeconds = distanceToTime(compareLap, base.distance) - base.time;
    readout.textContent = `${lead} · V ${base.speed.toFixed(0)} / ${compare.speed.toFixed(0)} km/h · Δt ${formatDelta(deltaSeconds)}`;
}

function stepSector(direction) {
    const count = sectorCount();
    if (!count) return;
    if (!state.range) {
        selectSector(direction > 0 ? 0 : count - 1);
        return;
    }
    selectSector((state.sector + direction + count) % count);
}

function selectSector(index) {
    state.sector = index;
    state.range = [laps()[0].sectors[index], laps()[0].sectors[index + 1]];
    clearHover();
    renderSectors();
    drawAll();
}

function resetSectorSelection() {
    state.range = null;
    clearHover();
    renderSectors();
    drawAll();
}

function setView(view) {
    const steering = channels.find(channel => channel.key === "steering");
    steering.label = view === "cornering" ? "Yaw / Steering" : "Steering wheel angle";
    const labels = chartsEl.querySelectorAll(".chart-label");
    const steeringIndex = channels.findIndex(channel => channel.key === "steering");
    if (labels[steeringIndex]) labels[steeringIndex].textContent = steering.label;
    drawAll();
}

function setAxisMode(mode) {
    state.axisMode = mode;
    state.zoomAnchor = null;
    clearHover();
    drawAll();
}

function zoomHorizontal(direction) {
    const nextZoom = direction > 0 ? Math.min(state.xZoom * 1.6, 24) : Math.max(state.xZoom / 1.6, 1);
    const baseRange = basePlotRange();
    state.zoomAnchor = clamp(state.hoverValue ?? state.zoomAnchor ?? midpoint(baseRange), baseRange[0], baseRange[1]);
    state.xZoom = nextZoom < 1.01 ? 1 : nextZoom;
    drawAll();
}

function resetHorizontalZoom() {
    state.xZoom = 1;
    state.zoomAnchor = null;
    clearHover();
    drawAll();
}

function startResize(event) {
    if (event.button !== 0) return;
    event.preventDefault();
    activeResize = {
        mode: event.currentTarget.dataset.splitter,
        rect: workspace.getBoundingClientRect(),
        splitter: event.currentTarget,
    };
    activeResize.splitter.classList.add("dragging");
    document.body.classList.add("resizing");
}

function resizeMove(event) {
    if (!activeResize) return;
    if (activeResize.mode === "left") {
        const width = clamp(event.clientX - activeResize.rect.left, 170, 420);
        workspace.style.setProperty("--left-panel", `${width}px`);
    } else {
        const width = clamp(activeResize.rect.right - event.clientX, 260, 620);
        workspace.style.setProperty("--right-panel", `${width}px`);
    }
    scheduleResizeAll();
}

function stopResize() {
    if (!activeResize) return;
    activeResize.splitter.classList.remove("dragging");
    document.body.classList.remove("resizing");
    activeResize = null;
    if (resizeFrame) {
        window.cancelAnimationFrame(resizeFrame);
        resizeFrame = 0;
    }
    resizeAll();
}

function scheduleResizeAll() {
    if (resizeFrame) return;
    resizeFrame = window.requestAnimationFrame(() => {
        resizeFrame = 0;
        resizeAll();
    });
}

function resizeAll() {
    fitCanvas(map);
    chartCanvases.forEach(({ canvas }) => fitCanvas(canvas));
    centerChartCanvases.forEach(({ canvas }) => fitCanvas(canvas));
    drawAll();
}

function fitCanvas(canvas) {
    const box = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(box.width * ratio));
    canvas.height = Math.max(1, Math.floor(box.height * ratio));
}

function laps() {
    return analysis?.laps || [];
}

function comparisonLaps() {
    return laps().slice(0, 2);
}

function overlayLaps() {
    return [...laps()].reverse();
}

function trackPoints() {
    if (analysis?.trackPoints?.length) return analysis.trackPoints;
    return laps()[0]?.points || [];
}

function sectorCount() {
    return Math.max(0, (laps()[0]?.sectors?.length || 0) - 1);
}

function axisKey() {
    return state.axisMode === "time" ? "time" : "distance";
}

function plotRange() {
    const baseRange = basePlotRange();
    const span = baseRange[1] - baseRange[0];
    if (state.xZoom <= 1 || span <= 0) return baseRange;
    const visibleSpan = span / state.xZoom;
    const half = visibleSpan / 2;
    const center = clamp(state.zoomAnchor ?? midpoint(baseRange), baseRange[0] + half, baseRange[1] - half);
    return [center - half, center + half];
}

function basePlotRange() {
    if (!laps().length) return [0, 1];
    if (state.axisMode === "distance") {
        return state.range || [0, laps()[0].total];
    }
    if (!state.range) {
        return [0, Math.max(...laps().map(lap => lap.lapSeconds))];
    }
    const ranges = laps().map(lap => lapVisibleRange(lap));
    return [Math.min(...ranges.map(range => range[0])), Math.max(...ranges.map(range => range[1]))];
}

function lapVisibleRange(lap) {
    if (state.axisMode === "distance") {
        return state.range || [0, lap.total];
    }
    if (!state.range) {
        return [0, lap.lapSeconds];
    }
    return [distanceToTime(lap, state.range[0]), distanceToTime(lap, state.range[1])];
}

function isPointVisible(point, range) {
    const value = getAxisValue(point);
    return value >= range[0] && value <= range[1];
}

function getAxisValue(point) {
    return point[axisKey()];
}

function getHoverPoint(lap) {
    if (state.hoverValue === null) return null;
    const range = lapVisibleRange(lap);
    if (state.hoverValue < range[0] || state.hoverValue > range[1]) return null;
    const index = nearestIndexByKey(lap, state.hoverValue, axisKey());
    return lap.points[index];
}

function distanceToTime(lap, distance) {
    return interpolateLapValue(lap, distance, "distance", "time");
}

function getSectorTime(lap, index) {
    return distanceToTime(lap, lap.sectors[index + 1]) - distanceToTime(lap, lap.sectors[index]);
}

function getTimeDelta(basePoint) {
    const [baseLap, compareLap] = comparisonLaps();
    if (!baseLap || !compareLap) return 0;
    return interpolateLapPoint(compareLap, basePoint.distance).time - basePoint.time;
}

function getLineDelta(index, basePoint) {
    const [baseLap, compareLap] = comparisonLaps();
    if (!baseLap || !compareLap) return 0;
    const compare = interpolateLapPoint(compareLap, basePoint.distance);
    const neighborIndex = index === baseLap.points.length - 1 ? index - 1 : index + 1;
    const neighbor = baseLap.points[neighborIndex];
    const tangentX = neighbor.x - basePoint.x;
    const tangentY = neighbor.y - basePoint.y;
    const deltaX = compare.x - basePoint.x;
    const deltaY = compare.y - basePoint.y;
    const sign = Math.sign(tangentX * deltaY - tangentY * deltaX) || 1;
    return sign * Math.hypot(deltaX, deltaY);
}

function interpolateLapPoint(lap, distance) {
    return {
        distance,
        time: interpolateLapValue(lap, distance, "distance", "time"),
        x: interpolateLapValue(lap, distance, "distance", "x"),
        y: interpolateLapValue(lap, distance, "distance", "y"),
    };
}

function interpolateLapValue(lap, value, fromKey, toKey) {
    const points = lap.points;
    if (value <= points[0][fromKey]) return points[0][toKey];
    if (value >= points[points.length - 1][fromKey]) return points[points.length - 1][toKey];
    let lo = 0;
    let hi = points.length - 1;
    while (hi - lo > 1) {
        const mid = Math.floor((lo + hi) / 2);
        if (points[mid][fromKey] <= value) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const start = points[lo];
    const end = points[hi];
    const denom = end[fromKey] - start[fromKey] || 1;
    const t = (value - start[fromKey]) / denom;
    return start[toKey] + (end[toKey] - start[toKey]) * t;
}

function nearestIndexByKey(lap, value, key) {
    const points = lap.points;
    let lo = 0;
    let hi = points.length - 1;
    while (lo < hi) {
        const mid = Math.floor((lo + hi) / 2);
        if (points[mid][key] < value) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo === 0) return 0;
    return Math.abs(points[lo][key] - value) < Math.abs(points[lo - 1][key] - value) ? lo : lo - 1;
}

function setLoadStatus(message, isError) {
    loadStatus.textContent = message;
    loadStatus.classList.toggle("error", Boolean(isError));
}

function sessionRows() {
    return Array.from(sessionRowsEl.querySelectorAll(".session-row"));
}

function sessionPathInput(row) {
    return row.querySelector(".session-path");
}

function sessionLapSelect(row) {
    return row.querySelector(".session-lap");
}

function sessionDriverMeta(row) {
    return row.querySelector(".session-driver");
}

function restorePathInputs() {
    try {
        trackPathInput.value = localStorage.getItem(STORAGE_KEYS.track) || trackPathInput.value;
    } catch {
        // Ignore unavailable storage.
    }
    const storedEntries = readStoredSessionEntries();
    storedEntries.slice(0, MAX_SESSION_ROWS).forEach(entry => addSessionRow(entry));
    while (sessionRows().length < 2) {
        addSessionRow();
    }
}


function readStoredSessionEntries() {
    try {
        const stored = localStorage.getItem(STORAGE_KEYS.sessions);
        if (stored) {
            const parsed = JSON.parse(stored);
            if (Array.isArray(parsed)) {
                return parsed
                    .map(entry => ({
                        path: String(entry?.path || "").trim(),
                        lapId: entry?.lapId == null || entry.lapId === "" ? "best" : String(entry.lapId),
                    }))
                    .filter(entry => entry.path);
            }
        }
        const legacyPaths = [
            localStorage.getItem(LEGACY_STORAGE_KEYS.sessionA) || "",
            localStorage.getItem(LEGACY_STORAGE_KEYS.sessionB) || "",
        ].filter(Boolean);
        return legacyPaths.map(path => ({ path, lapId: "best" }));
    } catch {
        return [];
    }
}

function persistPathInputs() {
    try {
        localStorage.setItem(STORAGE_KEYS.track, trackPathInput.value.trim());
        localStorage.setItem(STORAGE_KEYS.sessions, JSON.stringify(sessionRows().map(row => ({
            path: sessionPathInput(row).value.trim(),
            lapId: sessionLapSelect(row).value || row.dataset.preferredLapId || "best",
        })).filter(entry => entry.path)));
        localStorage.removeItem(LEGACY_STORAGE_KEYS.sessionA);
        localStorage.removeItem(LEGACY_STORAGE_KEYS.sessionB);
    } catch {
        // Ignore unavailable storage.
    }
}

function scaleAxis(value, range, width) {
    return ((value - range[0]) / (range[1] - range[0] || 1)) * width;
}

function scaleValue(value, min, max, height) {
    return height - ((value - min) / (max - min || 1)) * height;
}

function midpoint(range) {
    return range[0] + (range[1] - range[0]) / 2;
}

function formatChartBound(chart, value) {
    const digits = chart.key === "timeDelta" ? 3 : chart.key === "lineDelta" ? 2 : chart.key === "steering" ? 1 : 0;
    return Number(value).toFixed(digits);
}

function displayLapLabel(lap) {
    return lap.label || `Lap ${lap.id}`;
}

function displayLapName(lap) {
    const label = displayLapLabel(lap);
    return label.includes("·") ? label.split("·")[0].trim() : label;
}

function displayLapSummary(lap) {
    const label = displayLapLabel(lap);
    return label.includes(lap.time) ? label : `${lap.time} · ${label}`;
}

function basenamePath(path) {
    const parts = String(path || "").split(/[\\/]/);
    return parts[parts.length - 1] || "";
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function formatLapTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainder = seconds - minutes * 60;
    return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(3).padStart(6, "0")}`;
}

function formatSeconds(seconds) {
    return seconds.toFixed(3);
}

function formatDelta(seconds) {
    const prefix = seconds > 0 ? "+" : "";
    return `${prefix}${seconds.toFixed(3)}s`;
}

function clearHover() {
    state.hoverValue = null;
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}
