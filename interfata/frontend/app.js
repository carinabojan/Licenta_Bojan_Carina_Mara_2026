const API = "";

const PALETTE = [
  "rgb(0,200,255)", "rgb(255,130,0)", "rgb(100,255,100)", "rgb(255,50,180)",
  "rgb(200,255,50)", "rgb(150,80,255)", "rgb(255,210,0)",  "rgb(50,235,200)",
];

const state = {
  origW: 0, origH: 0,
  cars: [],
  processed: 0,
  running: false,
  maskData: [],
  allImages: [],
  currentStem: null,
};

document.addEventListener("DOMContentLoaded", function () {

  const searchInput      = document.getElementById("searchInput");
  const imageList        = document.getElementById("imageList");
  const btnProcess       = document.getElementById("btnProcess");
  const btnReset         = document.getElementById("btnReset");
  const btnSave          = document.getElementById("btnSave");
  const progressSection  = document.getElementById("progressSection");
  const progressText     = document.getElementById("progressText");
  const progressFill     = document.getElementById("progressFill");
  const carList          = document.getElementById("carList");
  const statusText       = document.getElementById("statusText");
  const scenePlaceholder = document.getElementById("scenePlaceholder");
  const sceneWrapper     = document.getElementById("sceneWrapper");
  const bgCanvas         = document.getElementById("bgCanvas");
  const overlayLayer     = document.getElementById("overlayLayer");
  const bgCtx            = bgCanvas.getContext("2d");
  const chkDebug         = document.getElementById("chkDebug");
  const ctxSlider        = document.getElementById("ctxSlider");
  const ctxValue         = document.getElementById("ctxValue");
  const debugPanel       = document.getElementById("debugPanel");
  const dbgCarName       = document.getElementById("dbgCarName");
  const dbgCrop          = document.getElementById("dbgCrop");
  const dbgMask512       = document.getElementById("dbgMask512");
  const dbgBboxMask      = document.getElementById("dbgBboxMask");
  const dbgClose         = document.getElementById("dbgClose");
  const dbgReopen        = document.getElementById("dbgReopen");

  // Correction refs
  const btnCorrectMode   = document.getElementById("btnCorrectMode");
  const correctPanel     = document.getElementById("correctPanel");
  const correctCarName   = document.getElementById("correctCarName");
  const correctPtsList   = document.getElementById("correctPtsList");
  const btnApplyCorrect  = document.getElementById("btnApplyCorrect");
  const btnClearCorrect  = document.getElementById("btnClearCorrect");

  const corr = { active: false, carIdx: null, points: [] };

  const CTX_MAX = 0.7;
  ctxSlider.addEventListener("input", function () {
    ctxValue.textContent = parseFloat(ctxSlider.value).toFixed(1);
  });
  function getContextFactor() { return parseFloat(ctxSlider.value) * CTX_MAX; }


  function setStatus(msg, type) {
    statusText.textContent = msg;
    statusText.className   = "status " + (type || "");
  }

  function getScale() {
    const r = bgCanvas.getBoundingClientRect();
    return { sx: r.width / state.origW, sy: r.height / state.origH };
  }

  function updateProgress() {
    const total = state.cars.length, done = state.processed;
    progressText.textContent = done + " / " + total;
    progressFill.style.width = total > 0 ? (done / total * 100) + "%" : "0%";
  }

  function updateCarStatus(idx, status) {
    const item = carList.querySelector(".car-item[data-idx='" + idx + "']");
    if (!item) return;
    item.className = "car-item " + status;
    const badge = item.querySelector(".car-badge");
    const labels = { pending: "Asteapta", active: "Procesare...", done: "Gata", error: "Eroare", manual: "Manual" };
    if (badge) badge.textContent = labels[status] || status;
    const bbox = overlayLayer.querySelector(".car-bbox[data-car='" + idx + "']");
    if (bbox && status !== "manual") bbox.className = "car-bbox bbox-" + status;
    // Auto-scroll the active car into view within the list
    if (status === "active") item.scrollIntoView({ block: "nearest" });
  }

  function renderImageList(stems) {
    imageList.innerHTML = "";
    if (!stems || stems.length === 0) {
      imageList.innerHTML = '<p class="list-empty">Niciun rezultat.</p>';
      return;
    }
    stems.forEach(function (stem) {
      const el = document.createElement("div");
      el.className   = "img-item" + (stem === state.currentStem ? " active" : "");
      el.textContent = stem;
      el.title       = stem;
      el.addEventListener("click", function () { selectImage(stem); });
      imageList.appendChild(el);
    });
  }

  function renderCarList() {
    carList.innerHTML = "";
    if (state.cars.length === 0) {
      carList.innerHTML = '<p class="list-empty">Nicio masina</p>';
      return;
    }
    state.cars.forEach(function (car) {
      const color = PALETTE[car.idx % PALETTE.length];
      const el = document.createElement("div");
      el.className   = "car-item";
      el.dataset.idx = car.idx;
      el.innerHTML =
        '<div class="car-dot" style="background:' + color + '"></div>' +
        '<span class="car-name">Masina ' + (car.idx + 1) + '</span>' +
        '<span class="car-size">' + car.w + "\xD7" + car.h + "</span>" +
        '<span class="car-badge">Asteapta</span>';
      carList.appendChild(el);
    });
  }

  function renderBboxes() {
    overlayLayer.querySelectorAll(".car-bbox").forEach(function (el) { el.remove(); });
    const s = getScale();
    state.cars.forEach(function (car) {
      const div = document.createElement("div");
      div.className   = "car-bbox bbox-pending";
      div.dataset.car = car.idx;
      div.style.cssText =
        "left:"   + (car.x * s.sx) + "px;" +
        "top:"    + (car.y * s.sy) + "px;" +
        "width:"  + (car.w * s.sx) + "px;" +
        "height:" + (car.h * s.sy) + "px;";
      const lbl = document.createElement("span");
      lbl.className   = "bbox-label";
      lbl.textContent = "M" + (car.idx + 1);
      div.appendChild(lbl);
      overlayLayer.appendChild(div);
    });
  }

  function addMaskOverlay(data) {
    const s = getScale();

    
    overlayLayer.querySelectorAll(
      ".mask-overlay[data-car='" + data.car_idx + "'], .pt-dot[data-car='" + data.car_idx + "']"
    ).forEach(function (el) { el.remove(); });

    const div = document.createElement("div");
    div.className   = "mask-overlay";
    div.dataset.car = data.car_idx;
    div.style.cssText =
      "left:"   + (data.car_x * s.sx) + "px;" +
      "top:"    + (data.car_y * s.sy) + "px;" +
      "width:"  + (data.car_w * s.sx) + "px;" +
      "height:" + (data.car_h * s.sy) + "px;";
    const img = new Image();
    img.src = "data:image/png;base64," + data.mask_b64;
    div.appendChild(img);
    overlayLayer.appendChild(div);

    if (data.point_orig) {
      const pt = document.createElement("div");
      pt.className   = "pt-dot";
      pt.dataset.car = data.car_idx;
      pt.style.left  = (data.point_orig.x * s.sx) + "px";
      pt.style.top   = (data.point_orig.y * s.sy) + "px";
      overlayLayer.appendChild(pt);
    }

    state.maskData = state.maskData.filter(function (m) { return m.car_idx !== data.car_idx; });
    state.maskData.push({
      car_idx: data.car_idx,
      x1: data.car_x, y1: data.car_y, w: data.car_w, h: data.car_h,
      src: "data:image/png;base64," + data.mask_b64,
    });

    if (chkDebug.checked && data.debug) {
      dbgCarName.textContent = "Masina " + (data.car_idx + 1);
      dbgCrop.src     = "data:image/jpeg;base64," + data.debug.crop_b64;
      dbgMask512.src  = "data:image/png;base64,"  + data.debug.mask512_b64;
      dbgBboxMask.src = "data:image/png;base64,"  + data.debug.bbox_mask_b64;
      if (!debugPanel.classList.contains("user-hidden")) debugPanel.style.display = "block";
    }
  }

  function drawBackground(b64, mime, origW, origH) {
    return new Promise(function (resolve) {
      const img = new Image();
      img.onload  = function () {
        bgCanvas.width  = origW;
        bgCanvas.height = origH;
        bgCtx.drawImage(img, 0, 0, origW, origH);
        resolve();
      };
      img.onerror = function () { resolve(); };
      img.src = "data:" + mime + ";base64," + b64;
    });
  }


  dbgClose.addEventListener("click", function () {
    debugPanel.style.display = "none";
    debugPanel.classList.add("user-hidden");
    if (chkDebug.checked) dbgReopen.style.display = "block";
  });
  dbgReopen.addEventListener("click", function () {
    debugPanel.classList.remove("user-hidden");
    debugPanel.style.display = "block";
    dbgReopen.style.display = "none";
  });
  chkDebug.addEventListener("change", function () {
    if (!chkDebug.checked) {
      debugPanel.style.display = "none";
      dbgReopen.style.display = "none";
    } else {
      // Re-activarea debug-ului anuleaza inchiderea manuala (×)
      debugPanel.classList.remove("user-hidden");
      dbgReopen.style.display = "none";
      if (dbgCrop.getAttribute("src")) debugPanel.style.display = "block";
    }
  });


  function screenToImage(clientX, clientY) {
    const r = bgCanvas.getBoundingClientRect();
    return {
      x: (clientX - r.left) / (r.width  / state.origW),
      y: (clientY - r.top)  / (r.height / state.origH),
    };
  }

  function findCarAt(ix, iy) {
    let best = null, bestArea = Infinity;
    state.cars.forEach(function (c) {
      if (ix >= c.x && ix <= c.x + c.w && iy >= c.y && iy <= c.y + c.h) {
        const area = c.w * c.h;
        if (area < bestArea) { bestArea = area; best = c; }
      }
    });
    return best;
  }

  function renderCorrPoints() {
    overlayLayer.querySelectorAll(".corr-dot").forEach(function (el) { el.remove(); });
    const s = getScale();
    corr.points.forEach(function (p) {
      const d = document.createElement("div");
      d.className  = "corr-dot " + (p.label === 1 ? "pos" : "neg");
      d.style.left = (p.x * s.sx) + "px";
      d.style.top  = (p.y * s.sy) + "px";
      overlayLayer.appendChild(d);
    });
    correctPtsList.innerHTML = "";
    corr.points.forEach(function (p) {
      const c = document.createElement("span");
      c.className   = "corr-chip " + (p.label === 1 ? "pos" : "neg");
      c.textContent = (p.label === 1 ? "+" : "−");
      correctPtsList.appendChild(c);
    });
    const has = corr.points.length > 0;
    btnApplyCorrect.disabled = !has;
    btnClearCorrect.disabled = !has;
  }

  function clearCorrection() {
    corr.carIdx = null;
    corr.points = [];
    correctCarName.textContent = "—";
    renderCorrPoints();
  }

  function setCorrectionMode(on) {
    corr.active = on;
    btnCorrectMode.classList.toggle("active", on);
    correctPanel.style.display = on ? "flex" : "none";
    sceneWrapper.classList.toggle("correcting", on);
    if (!on) clearCorrection();
    else setStatus("Mod corectie: click pe o masina.", "");
  }

  function addCorrPoint(clientX, clientY, label) {
    if (!corr.active || state.origW === 0) return;
    const p = screenToImage(clientX, clientY);

    if (corr.carIdx === null) {
      const car = findCarAt(p.x, p.y);
      if (!car) { setStatus("Primul click trebuie pe o masina.", "error"); return; }
      corr.carIdx = car.idx;
      correctCarName.textContent = "Masina " + (car.idx + 1);
    }
    corr.points.push({ x: p.x, y: p.y, label: label });
    renderCorrPoints();
  }

  btnCorrectMode.addEventListener("click", function () { setCorrectionMode(!corr.active); });
  btnClearCorrect.addEventListener("click", clearCorrection);

  overlayLayer.addEventListener("click", function (e) {
    if (!corr.active) return;
    addCorrPoint(e.clientX, e.clientY, 1);
  });
  overlayLayer.addEventListener("contextmenu", function (e) {
    if (!corr.active) return;
    e.preventDefault();
    addCorrPoint(e.clientX, e.clientY, 0);
  });

  btnApplyCorrect.addEventListener("click", async function () {
    if (corr.carIdx === null || corr.points.length === 0) return;
    const carIdx = corr.carIdx;
    btnApplyCorrect.disabled = true;
    setStatus("Corectie masina " + (carIdx + 1) + "...", "loading");
    try {
      const res = await fetch(API + "/correct", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          car_idx: carIdx,
          points: corr.points.map(function (p) { return { x: p.x, y: p.y, label: p.label }; }),
          debug: chkDebug.checked,
          context_factor: getContextFactor(),
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Eroare server");
      addMaskOverlay(data);
      updateCarStatus(carIdx, "manual");
      btnSave.disabled = state.maskData.length === 0;
      clearCorrection();
      setStatus("Masca masinii " + (carIdx + 1) + " corectata manual.", "ok");
    } catch (err) {
      console.error("correct:", err);
      setStatus("Eroare corectie: " + err.message, "error");
      btnApplyCorrect.disabled = false;
    }
  });


  async function loadImageList() {
    imageList.innerHTML = '<p class="list-empty">Se conecteaza la server...</p>';
    try {
      const res = await fetch(API + "/list_images");
      if (!res.ok) {
        imageList.innerHTML =
          '<p class="list-empty">Eroare server ' + res.status + '. Verifica ca app.py este pornit.</p>';
        return;
      }
      const data = await res.json();
      if (data.error) {
        imageList.innerHTML = '<p class="list-empty">' + data.error + "</p>";
        return;
      }
      state.allImages = data.images || [];
      if (state.allImages.length === 0) {
        imageList.innerHTML = '<p class="list-empty">Folderul PozeInterfata este gol sau nu exista.</p>';
      } else {
        renderImageList(state.allImages);
        setStatus(state.allImages.length + " imagini disponibile.", "ok");
      }
    } catch (e) {
      console.error("loadImageList:", e);
      imageList.innerHTML = '<p class="list-empty">Serverul nu raspunde.<br>Porneste app.py si reincarca.</p>';
      setStatus("Serverul nu raspunde.", "error");
    }
  }


  async function selectImage(stem) {
    if (state.running || stem === state.currentStem) return;
    state.currentStem = stem;

    document.querySelectorAll(".img-item").forEach(function (el) {
      el.classList.toggle("active", el.textContent === stem);
    });

    setCorrectionMode(false);
    overlayLayer.innerHTML = "";
    state.processed = 0; state.maskData = []; state.cars = [];
    btnProcess.disabled = true; btnReset.disabled = true; btnSave.disabled = true;
    btnCorrectMode.disabled = true;
    progressSection.style.display = "none";
    carList.innerHTML = '<p class="list-empty">Se incarca...</p>';
    setStatus("Se incarca imaginea...", "loading");

    try {
      const res  = await fetch(API + "/load_image/" + encodeURIComponent(stem));
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Eroare " + res.status);

      state.origW = data.orig_w;
      state.origH = data.orig_h;
      state.cars  = data.cars || [];

      await drawBackground(data.image_b64, data.image_mime || "image/jpeg", data.orig_w, data.orig_h);
      scenePlaceholder.style.display = "none";
      sceneWrapper.style.display     = "inline-block";

      renderCarList();
      renderBboxes();

      if (state.cars.length === 0) {
        setStatus("Imagine incarcata — nicio adnotare .txt gasita.", "error");
      } else {
        setStatus(state.cars.length + " masini detectate. Apasa Segmenteaza toate.", "ok");
        btnProcess.disabled     = false;
        btnReset.disabled       = false;
        btnCorrectMode.disabled = false;
      }
    } catch (err) {
      console.error("selectImage:", err);
      setStatus("Eroare: " + err.message, "error");
    }
  }


  btnProcess.addEventListener("click", async function () {
    if (state.running || state.cars.length === 0) return;
    state.running = true; state.processed = 0; state.maskData = [];
    btnProcess.disabled = true; btnSave.disabled = true;

    setCorrectionMode(false);
    overlayLayer.querySelectorAll(".mask-overlay,.pt-dot,.corr-dot").forEach(function (el) { el.remove(); });
    renderCarList(); renderBboxes();
    progressSection.style.display = "block";
    updateProgress();

    for (let i = 0; i < state.cars.length; i++) {
      const car = state.cars[i];
      updateCarStatus(car.idx, "active");
      setStatus("Procesare masina " + (car.idx + 1) + " din " + state.cars.length + "...", "loading");
      try {
        const res  = await fetch(API + "/process_crop", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body:   JSON.stringify({ car_idx: car.idx, debug: chkDebug.checked, context_factor: getContextFactor() }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Eroare server");
        addMaskOverlay(data);
        updateCarStatus(car.idx, "done");
        state.processed++;
        updateProgress();
        await new Promise(function (r) { setTimeout(r, 120); });
      } catch (err) {
        console.error("process_crop:", err);
        updateCarStatus(car.idx, "error");
      }
    }

    state.running = false;
    btnProcess.disabled = false;
    btnSave.disabled    = state.maskData.length === 0;
    const ok = state.processed === state.cars.length;
    setStatus("Finalizat — " + state.processed + "/" + state.cars.length + " procesate.", ok ? "ok" : "error");
  });


  btnReset.addEventListener("click", function () {
    if (state.running) return;
    setCorrectionMode(false);
    overlayLayer.querySelectorAll(".mask-overlay,.pt-dot,.corr-dot").forEach(function (el) { el.remove(); });
    state.processed = 0; state.maskData = [];
    debugPanel.style.display = "none";
    dbgReopen.style.display = "none";
    renderCarList(); renderBboxes();
    btnProcess.disabled = false; btnSave.disabled = true;
    updateProgress();
    setStatus("Reset. Apasa Segmenteaza toate.", "");
  });


  btnSave.addEventListener("click", async function () {
    if (!state.maskData.length) return;
    const tmp = document.createElement("canvas");
    tmp.width = state.origW; tmp.height = state.origH;
    const ctx = tmp.getContext("2d");
    ctx.drawImage(bgCanvas, 0, 0, state.origW, state.origH);
    for (let i = 0; i < state.maskData.length; i++) {
      const m = state.maskData[i];
      await new Promise(function (resolve) {
        const img = new Image();
        img.onload = function () { ctx.drawImage(img, m.x1, m.y1, m.w, m.h); resolve(); };
        img.src = m.src;
      });
    }
    tmp.toBlob(function (blob) {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "segmentare_" + (state.currentStem || "scene") + ".png";
      a.click(); URL.revokeObjectURL(a.href);
    }, "image/png");
  });


  searchInput.addEventListener("input", function () {
    const q = searchInput.value.trim().toLowerCase();
    renderImageList(q
      ? state.allImages.filter(function (s) { return s.toLowerCase().indexOf(q) !== -1; })
      : state.allImages);
  });

  loadImageList();
});
