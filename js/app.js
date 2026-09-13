/**
 * Main Application Orchestrator
 * Connects Camera, Contour Detectors, Worker Inference, Vector Matching, and UI.
 */

import { CameraManager } from './camera.js';
import { FastContourDetector } from './contour-fast.js';
import { extractSpatialColorDescriptor, findBestMatch } from './matcher.js';
import * as db from './db.js';

class VisionIDApp {
  constructor() {
    this.activeTab = 'scan';
    this.contourMode = 'fast'; // 'fast' | 'neural'
    this.isModelReady = false;
    this.isSegmenterReady = false;
    this.isExtracting = false;
    this.isSegmenting = false;
    this.database = [];
    this.currentShots = [];
    this.downloadProgress = 0;

    // Pending worker requests
    this.reqCounter = 0;
    this.pendingRequests = new Map();

    // Elements
    this.videoEl = document.getElementById('camera-video');
    this.overlayCanvas = document.getElementById('overlay-canvas');
    this.overlayCtx = this.overlayCanvas.getContext('2d');
    this.reticleEl = document.getElementById('reticle');
    this.reticleCornersEl = document.querySelector('.reticle-corners');

    // UI Feedback Elements
    this.resultCard = document.getElementById('result-card');
    this.resultName = document.getElementById('result-name');
    this.resultThumb = document.getElementById('result-thumb');
    this.confidenceBadge = document.getElementById('confidence-badge');
    this.hudStatusText = document.getElementById('hud-status-text');
    this.brandStatus = document.getElementById('brand-status');
    this.statusDot = document.getElementById('status-dot');

    // Audio Cue
    this.audioCtx = null;
    this.lastMatchSoundTime = 0;

    // Controllers
    this.camera = new CameraManager(this.videoEl, this.overlayCanvas);
    this.fastContour = new FastContourDetector();
    this.neuralPolygon = null;

    this.lastInferenceTime = 0;
    this.inferenceInterval = 140; // ~7 FPS inference for stable mobile performance
  }

  async init() {
    this.bindEvents();
    await this.initDatabase();
    await this.initWorker();
    await this.startCamera();
    this.startLiveLoop();
  }

  async initDatabase() {
    try {
      this.database = await db.getAllObjects();
      this.updateLibraryCount();
      this.renderLibrary();
    } catch (e) {
      console.error('Fehler beim Laden der Datenbank:', e);
    }
  }

  async initWorker() {
    const loadingBanner = document.getElementById('loading-banner');
    const loadingText = document.getElementById('loading-text');
    const loadingProgress = document.getElementById('loading-progress');

    // Show initial loading banner
    if (loadingBanner) loadingBanner.classList.remove('hidden');

    this.worker = new Worker(new URL('./worker.js?v=2.3', import.meta.url));

    this.worker.onerror = (err) => {
      console.error('Worker-Fehler:', err);
      if (loadingText) loadingText.textContent = 'Worker-Fehler: ' + (err.message || 'Script-Fehler');
    };

    this.worker.onmessage = (e) => {
      const data = e.data;

      if (data.type === 'progress') {
        this.downloadProgress = data.percent;
        if (loadingBanner) loadingBanner.classList.remove('hidden');
        if (loadingText) loadingText.textContent = `${data.model}: ${data.percent}%`;
        if (loadingProgress) loadingProgress.style.width = `${data.percent}%`;
      } else if (data.type === 'status') {
        if (loadingText) loadingText.textContent = data.msg;
      } else if (data.type === 'ready') {
        this.isModelReady = true;
        if (loadingBanner) loadingBanner.classList.add('hidden');
        this.statusDot.className = 'status-indicator ready';
        this.brandStatus.textContent = 'Bereit';
        this.hudStatusText.textContent = 'Bereit für Scan';
      } else if (data.type === 'segmenter_ready') {
        this.isSegmenterReady = true;
        if (loadingBanner) loadingBanner.classList.add('hidden');
        this.hudStatusText.textContent = 'KI-Maske aktiv';
      } else if (data.type === 'extract_result') {
        const resolver = this.pendingRequests.get(data.reqId);
        if (resolver) {
          resolver(data.vector);
          this.pendingRequests.delete(data.reqId);
        }
      } else if (data.type === 'segment_result') {
        const resolver = this.pendingRequests.get(data.reqId);
        if (resolver) {
          resolver(data.polygon);
          this.pendingRequests.delete(data.reqId);
        }
      } else if (data.type === 'error') {
        if (loadingText) loadingText.textContent = `Fehler: ${data.error}`;
        this.statusDot.className = 'status-indicator';
        this.statusDot.style.background = '#f87171';
      }
    };

    this.worker.postMessage({ type: 'init' });
  }

  async startCamera() {
    try {
      await this.camera.start();
      this.hudStatusText.textContent = 'Bereit für Scan';
    } catch (err) {
      this.hudStatusText.textContent = 'Kamerazugriff erforderlich';
    }
  }

  // --- Real-time Loop (60 FPS Canvas + Throttled AI Inference) ---
  startLiveLoop() {
    const loop = (timestamp) => {
      // Auto-resume video stream if paused by browser autoplay policy
      if (this.videoEl && this.videoEl.paused && this.videoEl.srcObject) {
        this.videoEl.play().catch(() => {});
      }

      if (this.activeTab === 'scan' && this.videoEl.readyState >= 1) {
        this.renderScanFrame(timestamp);
      } else if (this.activeTab === 'train' && this.videoEl.readyState >= 1) {
        this.renderTrainFrame();
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  async renderScanFrame(timestamp) {
    const coords = this.camera.getReticleVideoCoords(this.reticleEl);
    this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);

    // 1. Draw Object Outline according to selected mode
    let contourColor = '#38bdf8'; // Default cyan outline
    if (this.currentBestMatch && this.currentBestMatch.confidence >= 70) {
      contourColor = '#34d399'; // Emerald match outline
    }

    if (this.contourMode === 'fast') {
      // High-precision Moore-Neighbor boundary tracing (no solid fill)
      this.fastContour.detectAndDraw(this.videoEl, this.overlayCtx, coords, contourColor, 2);
    } else if (this.contourMode === 'neural') {
      this.renderNeuralMask(coords, contourColor);
    }

    // 2. Trigger async inference if ready and interval elapsed
    if (this.isModelReady && !this.isExtracting && (timestamp - this.lastInferenceTime > this.inferenceInterval)) {
      this.lastInferenceTime = timestamp;
      this.runInference(coords);
    }
  }

  async runInference(coords) {
    if (this.database.length === 0) {
      this.hudStatusText.textContent = 'Keine Objekte angelernt';
      this.hideResultCard();
      return;
    }

    this.isExtracting = true;
    try {
      const crop = this.camera.grabCrop(coords, 224);

      // Async Deep Feature Extraction in Worker using Transferable ArrayBuffer
      const reqId = ++this.reqCounter;
      const extractPromise = new Promise((resolve) => this.pendingRequests.set(reqId, resolve));
      this.worker.postMessage(
        { type: 'extract', reqId, buffer: crop.buffer, width: crop.width, height: crop.height },
        [crop.buffer]
      );

      // Spatial Color Descriptor in Main Thread
      const colorVector = extractSpatialColorDescriptor(crop.imageData);

      const deepVector = await extractPromise;
      if (deepVector) {
        const match = findBestMatch({ deepVector, colorVector }, this.database, { threshold: 0.68 });
        this.handleMatchResult(match);
      }

      // If neural segmentation mode is active, fetch segmentation polygon
      if (this.contourMode === 'neural' && this.isSegmenterReady && !this.isSegmenting) {
        this.fetchNeuralSegments(coords);
      }
    } catch (err) {
      console.warn('Inferenz-Aussetzer:', err);
    } finally {
      this.isExtracting = false;
    }
  }

  async fetchNeuralSegments(coords) {
    this.isSegmenting = true;
    try {
      const crop = this.camera.grabCrop(coords, 224);
      const reqId = ++this.reqCounter;
      const segmentPromise = new Promise((resolve) => this.pendingRequests.set(reqId, resolve));
      this.worker.postMessage(
        { type: 'segment', reqId, buffer: crop.buffer, width: crop.width, height: crop.height },
        [crop.buffer]
      );
      this.neuralPolygon = await segmentPromise;
    } catch (e) {
      // ignore
    } finally {
      this.isSegmenting = false;
    }
  }

  renderNeuralMask(coords, color) {
    if (!this.neuralPolygon || this.neuralPolygon.length < 3) return;

    this.overlayCtx.save();
    this.overlayCtx.strokeStyle = color;
    this.overlayCtx.lineWidth = 2.2;
    this.overlayCtx.lineJoin = 'round';
    this.overlayCtx.lineCap = 'round';
    this.overlayCtx.shadowColor = color;
    this.overlayCtx.shadowBlur = 6;

    this.overlayCtx.beginPath();
    const pts = this.neuralPolygon;
    const len = pts.length;

    const startX = coords.x + pts[0].x * coords.width;
    const startY = coords.y + pts[0].y * coords.height;
    this.overlayCtx.moveTo(startX, startY);

    for (let i = 1; i < len; i++) {
      const px = coords.x + pts[i].x * coords.width;
      const py = coords.y + pts[i].y * coords.height;
      this.overlayCtx.lineTo(px, py);
    }

    this.overlayCtx.closePath();
    this.overlayCtx.fillStyle = color.includes('#') ? `${color}38` : 'rgba(16, 185, 129, 0.22)';
    this.overlayCtx.fill();
    this.overlayCtx.stroke();
    this.overlayCtx.restore();
  }

  handleMatchResult(match) {
    this.currentBestMatch = match;

    if (match) {
      this.reticleCornersEl.classList.add('match-found');
      this.hudStatusText.textContent = `Erkannt: ${match.item.name}`;

      this.resultName.textContent = match.item.name;
      this.resultThumb.src = match.item.thumbnail || match.item.shots[0]?.thumb || '';
      this.confidenceBadge.textContent = `${match.confidence}% Match`;

      if (match.confidence >= 80) {
        this.confidenceBadge.className = 'confidence-badge confidence-high';
      } else {
        this.confidenceBadge.className = 'confidence-badge confidence-med';
      }

      this.resultCard.classList.add('visible', 'match');

      const now = Date.now();
      if (now - this.lastMatchSoundTime > 2500) {
        this.lastMatchSoundTime = now;
        if (navigator.vibrate) navigator.vibrate(25);
        this.playMatchChime();
      }
    } else {
      this.reticleCornersEl.classList.remove('match-found');
      this.hudStatusText.textContent = 'Suche nach Objekt...';
      this.hideResultCard();
    }
  }

  hideResultCard() {
    this.resultCard.classList.remove('visible', 'match');
    this.currentBestMatch = null;
  }

  playMatchChime() {
    try {
      if (!this.audioCtx) {
        this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (this.audioCtx.state === 'suspended') {
        this.audioCtx.resume();
      }
      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(880, this.audioCtx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(1320, this.audioCtx.currentTime + 0.08);
      gain.gain.setValueAtTime(0.04, this.audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.audioCtx.currentTime + 0.12);
      osc.connect(gain);
      gain.connect(this.audioCtx.destination);
      osc.start();
      osc.stop(this.audioCtx.currentTime + 0.12);
    } catch (e) {
      // ignore
    }
  }

  // --- Training Flow ---
  renderTrainFrame() {
    const trainCanvas = document.getElementById('train-overlay-canvas');
    if (!trainCanvas || this.videoEl.readyState < 1) return;
    const ctx = trainCanvas.getContext('2d');
    const w = this.videoEl.videoWidth || 640;
    const h = this.videoEl.videoHeight || 480;
    if (trainCanvas.width !== w || trainCanvas.height !== h) {
      trainCanvas.width = w;
      trainCanvas.height = h;
    }

    // 1. Live camera feed
    ctx.drawImage(this.videoEl, 0, 0, w, h);

    // 2. Target reticle
    const rw = w * 0.7;
    const rh = h * 0.7;
    const rx = (w - rw) / 2;
    const ry = (h - rh) / 2;
    const reticleRect = { x: rx, y: ry, width: rw, height: rh };

    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.restore();

    // 3. Crisp outline if object present
    this.fastContour.detectAndDraw(this.videoEl, ctx, reticleRect, '#38bdf8', 2);
  }

  async captureTrainingShot() {
    if (!this.isModelReady) {
      alert(`Vision-Modell lädt noch (${this.downloadProgress}%)... Bitte kurz warten.`);
      return;
    }

    if (this.currentShots.length >= 5) {
      alert('Maximal 5 Blickwinkel pro Objekt!');
      return;
    }

    const btn = document.getElementById('btn-capture-shot');
    btn.disabled = true;
    btn.textContent = 'Verarbeite Bild...';

    try {
      const w = this.videoEl.videoWidth || 640;
      const h = this.videoEl.videoHeight || 480;
      const rw = w * 0.7;
      const rh = h * 0.7;
      const rx = (w - rw) / 2;
      const ry = (h - rh) / 2;

      const coords = { x: rx, y: ry, width: rw, height: rh };
      const crop = this.camera.grabCrop(coords, 224);

      // Extract deep vector in worker using transferable ArrayBuffer
      const reqId = ++this.reqCounter;
      const extractPromise = new Promise((resolve) => this.pendingRequests.set(reqId, resolve));
      this.worker.postMessage(
        { type: 'extract', reqId, buffer: crop.buffer, width: crop.width, height: crop.height },
        [crop.buffer]
      );

      const colorVector = extractSpatialColorDescriptor(crop.imageData);
      const deepVector = await extractPromise;

      if (deepVector) {
        this.currentShots.push({
          deepVector,
          colorVector,
          thumb: crop.thumbDataUrl
        });

        this.renderSnapshotSlots();
        if (navigator.vibrate) navigator.vibrate([15, 30, 15]);
      }
    } catch (err) {
      console.error('Fehler bei Aufnahme:', err);
      alert('Fehler bei der Aufnahme: ' + err.message);
    } finally {
      btn.disabled = false;
      this.updateCaptureButtonText();
    }
  }

  renderSnapshotSlots() {
    for (let i = 0; i < 5; i++) {
      const slot = document.getElementById(`slot-${i}`);
      if (!slot) continue;
      const shot = this.currentShots[i];
      if (shot) {
        slot.className = 'snapshot-slot filled';
        slot.innerHTML = `<img src="${shot.thumb}" alt="Winkel ${i + 1}">`;
      } else {
        slot.className = 'snapshot-slot';
        slot.innerHTML = '';
      }
    }

    const saveBtn = document.getElementById('btn-save-object');
    const nameInput = document.getElementById('input-obj-name');
    if (saveBtn) {
      saveBtn.disabled = this.currentShots.length === 0 || !nameInput.value.trim();
    }
  }

  updateCaptureButtonText() {
    const btn = document.getElementById('btn-capture-shot');
    if (btn) {
      btn.textContent = `Winkel aufnehmen (${this.currentShots.length}/5)`;
    }
  }

  async saveCurrentObject() {
    const nameInput = document.getElementById('input-obj-name');
    const name = nameInput.value.trim();

    if (!name || this.currentShots.length === 0) return;

    const saveBtn = document.getElementById('btn-save-object');
    saveBtn.disabled = true;
    saveBtn.textContent = 'Speichere...';

    try {
      await db.saveObject({
        name,
        thumbnail: this.currentShots[0].thumb,
        shots: this.currentShots
      });

      this.database = await db.getAllObjects();
      this.updateLibraryCount();
      this.renderLibrary();

      this.currentShots = [];
      nameInput.value = '';
      this.renderSnapshotSlots();
      this.updateCaptureButtonText();

      this.switchTab('scan');
      this.hudStatusText.textContent = `„${name}“ gespeichert`;
      setTimeout(() => {
        this.hudStatusText.textContent = 'Bereit für Scan';
      }, 2500);
    } catch (e) {
      console.error('Fehler beim Speichern:', e);
      alert('Fehler: ' + e.message);
    } finally {
      saveBtn.disabled = false;
      saveBtn.textContent = 'Objekt speichern';
    }
  }

  // --- Objects Library View ---
  renderLibrary() {
    const listEl = document.getElementById('objects-list');
    if (!listEl) return;

    if (this.database.length === 0) {
      listEl.innerHTML = `
        <div class="empty-state">
          Noch keine Objekte registriert.<br>
          Wechsle zu <strong>„Anlernen“</strong>, um Gegenstände zu erfassen.
        </div>
      `;
      return;
    }

    listEl.innerHTML = this.database.map((item) => `
      <div class="object-card">
        <div class="object-card-left">
          <img src="${item.thumbnail || ''}" class="object-card-thumb" alt="${item.name}">
          <div>
            <div class="object-card-title">${item.name}</div>
            <div class="object-card-meta">${item.shots.length} Blickwinkel erfasst</div>
          </div>
        </div>
        <button onclick="window.app.deleteObject(${item.id})" class="btn-delete-obj" title="Löschen">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="3 6 5 6 21 6"></polyline>
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
          </svg>
        </button>
      </div>
    `).join('');
  }

  async deleteObject(id) {
    if (!confirm('Objekt wirklich löschen?')) return;
    await db.deleteObject(id);
    this.database = await db.getAllObjects();
    this.updateLibraryCount();
    this.renderLibrary();
  }

  updateLibraryCount() {
    const countEl = document.getElementById('lib-count');
    if (countEl) countEl.textContent = `${this.database.length} Objekte`;
  }

  // --- Navigation & Controls ---
  switchTab(tab) {
    this.activeTab = tab;

    document.querySelectorAll('.view-section').forEach(sec => sec.classList.remove('active'));
    document.querySelectorAll('.nav-tab-btn').forEach(btn => btn.classList.remove('active'));

    const activeSec = document.getElementById(`view-${tab}`);
    const activeBtn = document.getElementById(`nav-${tab}`);
    if (activeSec) activeSec.classList.add('active');
    if (activeBtn) activeBtn.classList.add('active');

    if (tab === 'scan') {
      this.fastContour.reset();
      this.hideResultCard();
    } else if (tab === 'train') {
      this.updateCaptureButtonText();
    } else if (tab === 'objects') {
      this.renderLibrary();
    }
  }

  setContourMode(mode) {
    this.contourMode = mode;
    document.getElementById('btn-contour-fast').classList.toggle('active', mode === 'fast');
    document.getElementById('btn-contour-neural').classList.toggle('active', mode === 'neural');
    this.fastContour.reset();
    this.neuralPolygon = null;

    if (mode === 'neural' && !this.isSegmenterReady) {
      const banner = document.getElementById('loading-banner');
      const text = document.getElementById('loading-text');
      if (banner) banner.classList.remove('hidden');
      if (text) text.textContent = 'Lade KI-Segmentierer...';
      this.worker.postMessage({ type: 'load_segmenter' });
    }
  }

  bindEvents() {
    // Navigation Tabs
    document.getElementById('nav-scan')?.addEventListener('click', () => this.switchTab('scan'));
    document.getElementById('nav-train')?.addEventListener('click', () => this.switchTab('train'));
    document.getElementById('nav-objects')?.addEventListener('click', () => this.switchTab('objects'));

    // Contour Toggle
    document.getElementById('btn-contour-fast')?.addEventListener('click', () => this.setContourMode('fast'));
    document.getElementById('btn-contour-neural')?.addEventListener('click', () => this.setContourMode('neural'));

    // User gesture video unpause listener (desktop & mobile autoplay policy)
    const unlockCamera = () => {
      if (this.videoEl && this.videoEl.paused && this.videoEl.srcObject) {
        this.videoEl.play().catch(console.error);
      }
    };
    document.addEventListener('click', unlockCamera);
    document.addEventListener('touchstart', unlockCamera);

    // Tap on camera viewport to retry camera if not running or resume if paused
    document.querySelector('.camera-wrapper')?.addEventListener('click', () => {
      if (!this.camera.stream) {
        this.startCamera();
      } else if (this.videoEl && this.videoEl.paused) {
        this.videoEl.play().catch(console.error);
      }
    });

    // Single Tap Manual Trigger
    document.getElementById('btn-trigger-scan')?.addEventListener('click', () => {
      const coords = this.camera.getReticleVideoCoords(this.reticleEl);
      this.runInference(coords);
    });

    // Training Actions
    document.getElementById('btn-capture-shot')?.addEventListener('click', () => this.captureTrainingShot());
    document.getElementById('btn-save-object')?.addEventListener('click', () => this.saveCurrentObject());
    document.getElementById('input-obj-name')?.addEventListener('input', () => this.renderSnapshotSlots());

    // Data Export / Import
    document.getElementById('btn-export-vault')?.addEventListener('click', async () => {
      const json = await db.exportVaultJSON();
      const blob = new Blob([json], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `visionid-vault-${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    });

    document.getElementById('btn-import-vault')?.addEventListener('click', () => {
      document.getElementById('file-import').click();
    });

    document.getElementById('file-import')?.addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const text = await file.text();
      try {
        await db.importVaultJSON(text);
        this.database = await db.getAllObjects();
        this.updateLibraryCount();
        this.renderLibrary();
        alert('Datenbank erfolgreich importiert!');
      } catch (err) {
        alert('Import fehlgeschlagen: ' + err.message);
      }
    });

    // Tap on Result Card to dismiss
    this.resultCard?.addEventListener('click', () => this.hideResultCard());
  }
}

// Bootstrap
window.addEventListener('DOMContentLoaded', () => {
  window.app = new VisionIDApp();
  window.app.init();
});
