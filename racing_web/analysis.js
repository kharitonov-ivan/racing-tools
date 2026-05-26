const lap = makeDemoLap();
const state = { index: 0, sector: 0, zoom: 1, range: null };
const channels = [
    { key: "speed", label: "Speed", unit: "km/h", min: 60, max: 190 },
    { key: "throttle", label: "Throttle", unit: "%", min: 0, max: 100 },
    { key: "brake", label: "Brake", unit: "%", min: 0, max: 100 },
    { key: "gear", label: "Gear", unit: "", min: 1, max: 6 },
    { key: "rpm", label: "RPM", unit: "", min: 4000, max: 11000 },
    { key: "steering", label: "Steering", unit: "°", min: -80, max: 80 },
];

const map = document.getElementById("track-map");
const mapCtx = map.getContext("2d");
const chartsEl = document.getElementById("charts");
const readout = document.getElementById("readout");
const sectorTable = document.getElementById("sector-table");
const sectorStrip = document.getElementById("sector-strip");
const chartCanvases = [];

init();

function init() {
    renderLaps();
    renderSectors();
    renderCharts();
    bindControls();
    resizeAll();
}

function makeDemoLap() {
    const total = 2430;
    const points = [];
    for (let i = 0; i <= 720; i += 1) {
        const u = i / 720;
        const a = u * Math.PI * 2;
        const r = 1 + 0.23 * Math.sin(3 * a) + 0.13 * Math.cos(5 * a);
        const x = r * Math.cos(a) + 0.22 * Math.sin(2 * a);
        const y = 0.66 * r * Math.sin(a) + 0.16 * Math.cos(4 * a);
        const corner = Math.max(0, Math.sin(7 * a - 0.8));
        const braking = Math.pow(Math.max(0, Math.sin(7 * a + 0.6)), 8) * 100;
        const speed = 178 - 72 * corner + 8 * Math.sin(13 * a);
        const throttle = clamp(105 - braking * 1.4 - corner * 35, 0, 100);
        const gear = clamp(Math.round((speed - 50) / 24), 1, 6);
        const rpm = 4700 + gear * 820 + ((speed * 57) % 2500);
        const steering = 55 * Math.sin(7 * a - 0.25) + 18 * Math.sin(14 * a);
        points.push({ distance: u * total, x, y, speed, throttle, brake: braking, gear, rpm, steering });
    }
    const sectors = [0, 0.124, 0.252, 0.422, 0.548, 0.744, 0.872, 1].map(v => v * total);
    return { color: "#ff4058", time: "01:37.029", driver: "Ivan Kharitonov", total, points, sectors };
}

function renderLaps() {
    document.getElementById("lap-list").innerHTML = `
        <div class="lap-row"><span class="color-dot"></span><span><span class="lap-time">${lap.time}</span><br><span class="driver">${lap.driver}</span></span><strong>1</strong></div>`;
    document.getElementById("map-legend").innerHTML = `<span class="color-dot"></span> ${lap.time} · ${lap.driver}`;
}

function renderSectors() {
    const rows = [];
    const chips = [];
    for (let i = 0; i < 7; i += 1) {
        const start = lap.sectors[i];
        const end = lap.sectors[i + 1];
        const secTime = ((end - start) / lap.total * 97.029).toFixed(3);
        rows.push(`<div class="sector-row ${i === state.sector ? "active" : ""}" data-sector="${i}"><span class="color-dot"></span><span>S${i + 1}</span><span>${secTime}</span></div>`);
        chips.push(`<button class="sector-chip ${i === state.sector ? "active" : ""}" data-sector="${i}">S${i + 1}<small>${secTime}</small></button>`);
    }
    sectorTable.innerHTML = rows.join("");
    sectorStrip.innerHTML = chips.join("");
    document.querySelectorAll("[data-sector]").forEach(el => el.onclick = () => selectSector(Number(el.dataset.sector)));
}

function renderCharts() {
    chartsEl.innerHTML = "";
    channels.forEach(ch => {
        const card = document.createElement("div");
        card.className = "chart-card";
        card.innerHTML = `<span class="chart-label">${ch.label}</span><canvas></canvas>`;
        chartsEl.appendChild(card);
        const canvas = card.querySelector("canvas");
        canvas.addEventListener("mousemove", onChartHover);
        canvas.addEventListener("mouseleave", () => { state.index = -1; drawAll(); });
        chartCanvases.push({ canvas, ch });
    });
}

function bindControls() {
    window.addEventListener("resize", resizeAll);
    document.getElementById("zoom-in").onclick = () => { state.zoom = Math.min(2.2, state.zoom + 0.15); drawAll(); };
    document.getElementById("zoom-out").onclick = () => { state.zoom = Math.max(0.75, state.zoom - 0.15); drawAll(); };
    document.getElementById("prev-btn").onclick = () => selectSector((state.sector + 6) % 7);
    document.getElementById("next-btn").onclick = () => selectSector((state.sector + 1) % 7);
    document.getElementById("navigate-btn").onclick = () => selectSector(0);
    document.getElementById("view-select").onchange = event => setView(event.target.value);
    bindSplitters();
}

function bindSplitters() {
    document.querySelectorAll(".splitter").forEach(splitter => {
        splitter.addEventListener("pointerdown", event => startResize(event, splitter));
    });
}

function startResize(event, splitter) {
    const workspace = document.querySelector(".workspace");
    const rect = workspace.getBoundingClientRect();
    const mode = splitter.dataset.splitter;
    splitter.setPointerCapture(event.pointerId);
    splitter.classList.add("dragging");
    document.body.classList.add("resizing");

    splitter.onpointermove = moveEvent => {
        if (mode === "left") {
            const width = clamp(moveEvent.clientX - rect.left, 170, 420);
            workspace.style.setProperty("--left-panel", `${width}px`);
        } else {
            const width = clamp(rect.right - moveEvent.clientX, 260, 620);
            workspace.style.setProperty("--right-panel", `${width}px`);
        }
        resizeAll();
    };
    splitter.onpointerup = () => stopResize(splitter);
    splitter.onpointercancel = () => stopResize(splitter);
}

function stopResize(splitter) {
    splitter.onpointermove = null;
    splitter.classList.remove("dragging");
    document.body.classList.remove("resizing");
    resizeAll();
}

function setView(view) {
    const steering = channels.find(ch => ch.key === "steering");
    steering.label = view === "cornering" ? "Yaw / Steering" : "Steering";
    drawAll();
}

function resizeAll() {
    fitCanvas(map);
    chartCanvases.forEach(({ canvas }) => fitCanvas(canvas));
    drawAll();
}

function fitCanvas(canvas) {
    const box = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(box.width * ratio));
    canvas.height = Math.max(1, Math.floor(box.height * ratio));
}

function selectSector(idx) {
    state.sector = idx;
    state.range = [lap.sectors[idx], lap.sectors[idx + 1]];
    renderSectors();
    drawAll();
}

function onChartHover(event) {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const range = state.range || [0, lap.total];
    const distance = range[0] + x * (range[1] - range[0]);
    state.index = nearestIndex(distance);
    drawAll();
}

function nearestIndex(distance) {
    return Math.max(0, Math.min(lap.points.length - 1, Math.round(distance / lap.total * (lap.points.length - 1))));
}

function drawAll() {
    drawMap();
    chartCanvases.forEach(drawChart);
    updateReadout();
}

function drawMap() {
    const w = map.width, h = map.height;
    mapCtx.clearRect(0, 0, w, h);
    mapCtx.fillStyle = "#080a0d";
    mapCtx.fillRect(0, 0, w, h);
    const toScreen = mapScale(w, h);
    drawTrackPath(mapCtx, toScreen, "#526070", 7);
    drawTrackPath(mapCtx, toScreen, lap.color, 3);
    if (state.range) drawSegment(mapCtx, toScreen, state.range, "#ffd166", 5);
    if (state.index >= 0) drawMapMarker(mapCtx, toScreen(lap.points[state.index]));
}

function mapScale(w, h) {
    const xs = lap.points.map(p => p.x), ys = lap.points.map(p => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const scale = Math.min(w / (maxX - minX), h / (maxY - minY)) * 0.38 * state.zoom;
    return p => ({ x: w / 2 + (p.x - (minX + maxX) / 2) * scale, y: h / 2 - (p.y - (minY + maxY) / 2) * scale });
}

function drawTrackPath(ctx, toScreen, color, width) {
    ctx.beginPath();
    lap.points.forEach((p, i) => { const q = toScreen(p); i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y); });
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = "round";
    ctx.stroke();
}

function drawSegment(ctx, toScreen, range, color, width) {
    const pts = lap.points.filter(p => p.distance >= range[0] && p.distance <= range[1]);
    ctx.beginPath();
    pts.forEach((p, i) => { const q = toScreen(p); i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y); });
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();
}

function drawMapMarker(ctx, p) {
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = lap.color;
    ctx.lineWidth = 3;
    ctx.stroke();
}

function drawChart({ canvas, ch }) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    const range = state.range || [0, lap.total];
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, w, h);
    drawGrid(ctx, w, h);
    const pts = lap.points.filter(p => p.distance >= range[0] && p.distance <= range[1]);
    ctx.beginPath();
    pts.forEach((p, i) => {
        const x = ((p.distance - range[0]) / (range[1] - range[0])) * w;
        const y = h - ((p[ch.key] - ch.min) / (ch.max - ch.min)) * h;
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.strokeStyle = lap.color;
    ctx.lineWidth = 2;
    ctx.stroke();
    if (state.index >= 0) drawCursor(ctx, w, h, range);
}

function drawGrid(ctx, w, h) {
    ctx.strokeStyle = "rgba(148,163,184,0.25)";
    ctx.setLineDash([5, 5]);
    for (let i = 1; i < 4; i += 1) { ctx.beginPath(); ctx.moveTo(0, h * i / 4); ctx.lineTo(w, h * i / 4); ctx.stroke(); }
    ctx.setLineDash([]);
}

function drawCursor(ctx, w, h, range) {
    const distance = lap.points[state.index].distance;
    if (distance < range[0] || distance > range[1]) return;
    const x = ((distance - range[0]) / (range[1] - range[0])) * w;
    ctx.strokeStyle = "rgba(255,255,255,0.75)";
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
}

function updateReadout() {
    if (state.index < 0) return;
    const p = lap.points[state.index];
    readout.textContent = `${(p.distance / 1000).toFixed(2)} km · ${p.speed.toFixed(0)} km/h · T ${p.throttle.toFixed(0)} · B ${p.brake.toFixed(0)} · G ${p.gear}`;
}

function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
