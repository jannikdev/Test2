/**
 * Fast Real-time Contour & Edge Tracker (Mode 1: 0 MB, ~5ms, 60fps)
 * Computes gradient salience and traces the prominent object contour inside the viewfinder.
 */
export class FastContourDetector {
  constructor() {
    this.offscreen = document.createElement('canvas');
    this.offCtx = this.offscreen.getContext('2d', { willReadFrequently: true });
    this.smoothedContour = null;
    this.smoothingAlpha = 0.35; // Exponential smoothing to prevent edge jitter
  }

  /**
   * Process a frame and draw the object contour directly to targetCtx
   * @param {HTMLVideoElement|HTMLCanvasElement} source - Video source
   * @param {CanvasRenderingContext2D} targetCtx - Overlay canvas context
   * @param {Object} reticleRect - { x, y, width, height } in canvas coords
   * @param {string} strokeColor - CSS color for the contour outline
   * @param {number} lineWidth - Width of contour line
   * @returns {Object|null} Bounding box { x, y, width, height, points } or null
   */
  detectAndDraw(source, targetCtx, reticleRect, strokeColor = '#38bdf8', lineWidth = 2.5) {
    const rx = Math.max(0, Math.floor(reticleRect.x));
    const ry = Math.max(0, Math.floor(reticleRect.y));
    const rw = Math.min(source.videoWidth || source.width, Math.floor(reticleRect.width));
    const rh = Math.min(source.videoHeight || source.height, Math.floor(reticleRect.height));

    if (rw <= 20 || rh <= 20) return null;

    // Downscale for fast ~5ms processing: analyze at fixed 140x140
    const sampleW = 140;
    const sampleH = 140;
    this.offscreen.width = sampleW;
    this.offscreen.height = sampleH;

    this.offCtx.drawImage(source, rx, ry, rw, rh, 0, 0, sampleW, sampleH);
    const imgData = this.offCtx.getImageData(0, 0, sampleW, sampleH);
    const data = imgData.data;

    // 1. Grayscale & Edge Gradient (Sobel approximation)
    const gray = new Uint8Array(sampleW * sampleH);
    for (let i = 0; i < data.length; i += 4) {
      // Fast luminance: 0.299R + 0.587G + 0.114B
      gray[i >> 2] = (data[i] * 77 + data[i + 1] * 150 + data[i + 2] * 29) >> 8;
    }

    const edges = new Uint8Array(sampleW * sampleH);
    let edgeSum = 0;
    let edgeCount = 0;

    for (let y = 1; y < sampleH - 1; y++) {
      const row = y * sampleW;
      for (let x = 1; x < sampleW - 1; x++) {
        const idx = row + x;
        // Fast Sobel gradient
        const gx =
          -gray[idx - sampleW - 1] + gray[idx - sampleW + 1]
          - (gray[idx - 1] << 1) + (gray[idx + 1] << 1)
          - gray[idx + sampleW - 1] + gray[idx + sampleW + 1];

        const gy =
          -gray[idx - sampleW - 1] - (gray[idx - sampleW] << 1) - gray[idx - sampleW + 1]
          + gray[idx + sampleW - 1] + (gray[idx + sampleW] << 1) + gray[idx + sampleW + 1];

        const mag = Math.abs(gx) + Math.abs(gy);
        edges[idx] = mag > 255 ? 255 : mag;
        if (mag > 40) {
          edgeSum += mag;
          edgeCount++;
        }
      }
    }

    const avgEdge = edgeCount > 0 ? (edgeSum / edgeCount) * 0.75 : 55;
    const threshold = Math.max(45, Math.min(120, avgEdge));

    // 2. Find Radial Extreme Boundary Points from Center (Convex / Radial Contour)
    const centerX = sampleW / 2;
    const centerY = sampleH / 2;
    const numRays = 36; // 36 radial rays (every 10 degrees) for smooth contour
    const rawPoints = [];

    for (let i = 0; i < numRays; i++) {
      const angle = (i * 2 * Math.PI) / numRays;
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);
      const maxRadius = Math.min(sampleW, sampleH) * 0.46;

      let foundR = 0;
      // Scan outward from center to find first solid edge transition
      for (let r = 8; r < maxRadius; r += 2) {
        const px = Math.round(centerX + cosA * r);
        const py = Math.round(centerY + sinA * r);
        if (px < 1 || px >= sampleW - 1 || py < 1 || py >= sampleH - 1) break;

        const idx = py * sampleW + px;
        if (edges[idx] > threshold) {
          foundR = r;
          // Look slightly ahead to see if edge continues
          const nextPx = Math.round(centerX + cosA * (r + 4));
          const nextPy = Math.round(centerY + sinA * (r + 4));
          if (nextPx >= 1 && nextPx < sampleW - 1 && nextPy >= 1 && nextPy < sampleH - 1) {
            if (edges[nextPy * sampleW + nextPx] > threshold * 0.8) {
              foundR = r + 2;
            }
          }
        }
      }

      // If no strong edge found on this ray, default to fallback boundary
      const radius = foundR > 10 ? foundR : maxRadius * 0.65;

      // Map back to canvas coordinates
      const normX = (centerX + cosA * radius) / sampleW;
      const normY = (centerY + sinA * radius) / sampleH;
      rawPoints.push({
        x: rx + normX * rw,
        y: ry + normY * rh
      });
    }

    // 3. Temporal Smoothing (EMA) to eliminate jumpy contour lines
    if (!this.smoothedContour || this.smoothedContour.length !== rawPoints.length) {
      this.smoothedContour = rawPoints.map(p => ({ x: p.x, y: p.y }));
    } else {
      for (let i = 0; i < rawPoints.length; i++) {
        this.smoothedContour[i].x += (rawPoints[i].x - this.smoothedContour[i].x) * this.smoothingAlpha;
        this.smoothedContour[i].y += (rawPoints[i].y - this.smoothedContour[i].y) * this.smoothingAlpha;
      }
    }

    // 4. Render smooth spline curve on targetCtx
    targetCtx.save();
    targetCtx.strokeStyle = strokeColor;
    targetCtx.lineWidth = lineWidth;
    targetCtx.lineJoin = 'round';
    targetCtx.lineCap = 'round';
    targetCtx.shadowColor = strokeColor;
    targetCtx.shadowBlur = 8;

    targetCtx.beginPath();
    const pts = this.smoothedContour;
    const len = pts.length;

    targetCtx.moveTo((pts[0].x + pts[len - 1].x) / 2, (pts[0].y + pts[len - 1].y) / 2);
    for (let i = 0; i < len; i++) {
      const next = (i + 1) % len;
      const midX = (pts[i].x + pts[next].x) / 2;
      const midY = (pts[i].y + pts[next].y) / 2;
      targetCtx.quadraticCurveTo(pts[i].x, pts[i].y, midX, midY);
    }
    targetCtx.closePath();
    targetCtx.stroke();

    // Subtle inner glow
    targetCtx.fillStyle = strokeColor.replace(')', ', 0.05)').replace('rgb', 'rgba');
    targetCtx.fill();

    targetCtx.restore();

    // Calculate bounding box of contour
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of pts) {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }

    return {
      x: minX,
      y: minY,
      width: maxX - minX,
      height: maxY - minY,
      points: pts
    };
  }

  reset() {
    this.smoothedContour = null;
  }
}
