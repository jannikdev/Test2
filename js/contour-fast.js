/**
 * High-Precision Real-time Object Contour Tracker (Mode 1: 0 MB, ~5ms, 60fps)
 * Uses Corner Background Modeling, Sobel Edge Magnitude + Color Distance,
 * and 48-Ray Perimeter Contour Tracing to trace the exact physical silhouette.
 */
export class FastContourDetector {
  constructor() {
    this.offscreen = document.createElement('canvas');
    this.offCtx = this.offscreen.getContext('2d', { willReadFrequently: true });
    this.smoothedPoints = null;
    this.smoothingFactor = 0.35;
    this.hasObject = false;
  }

  /**
   * Process a frame and draw a high-precision silhouette contour
   * @param {HTMLVideoElement|HTMLCanvasElement} source - Video source
   * @param {CanvasRenderingContext2D} targetCtx - Overlay canvas context
   * @param {Object} reticleRect - { x, y, width, height } in canvas coords
   * @param {string} strokeColor - CSS color for the contour outline
   * @param {number} lineWidth - Width of contour line
   */
  detectAndDraw(source, targetCtx, reticleRect, strokeColor = '#38bdf8', lineWidth = 2.5) {
    const rx = Math.max(0, Math.floor(reticleRect.x));
    const ry = Math.max(0, Math.floor(reticleRect.y));
    const rw = Math.min(source.videoWidth || source.width, Math.floor(reticleRect.width));
    const rh = Math.min(source.videoHeight || source.height, Math.floor(reticleRect.height));

    if (rw <= 40 || rh <= 40) {
      this.smoothedPoints = null;
      return null;
    }

    // High enough resolution for fine features, fast enough for ~4ms execution
    const W = 130;
    const H = 130;
    this.offscreen.width = W;
    this.offscreen.height = H;

    this.offCtx.drawImage(source, rx, ry, rw, rh, 0, 0, W, H);
    const imgData = this.offCtx.getImageData(0, 0, W, H);
    const data = imgData.data;

    const centerX = W / 2, centerY = H / 2;

    // 1. Background color reference from perimeter corners
    let bgR = 0, bgG = 0, bgB = 0, bgCount = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        if (Math.hypot(x - centerX, y - centerY) > W * 0.44) {
          const idx = (y * W + x) << 2;
          bgR += data[idx];
          bgG += data[idx + 1];
          bgB += data[idx + 2];
          bgCount++;
        }
      }
    }
    if (bgCount > 0) {
      bgR /= bgCount; bgG /= bgCount; bgB /= bgCount;
    }

    // 2. Grayscale & Sobel
    const gray = new Uint8Array(W * H);
    for (let i = 0; i < data.length; i += 4) {
      gray[i >> 2] = (data[i] * 77 + data[i + 1] * 150 + data[i + 2] * 29) >> 8;
    }

    // 3. Foreground determination (Color Distance + Edge Magnitude)
    const fg = new Uint8Array(W * H);
    let fgCount = 0, sumX = 0, sumY = 0;

    for (let y = 1; y < H - 1; y++) {
      const row = y * W;
      for (let x = 1; x < W - 1; x++) {
        const idx = row + x;
        const pIdx = idx << 2;

        const gx =
          -gray[idx - W - 1] + gray[idx - W + 1]
          - (gray[idx - 1] << 1) + (gray[idx + 1] << 1)
          - gray[idx + W - 1] + gray[idx + W + 1];

        const gy =
          -gray[idx - W - 1] - (gray[idx - W] << 1) - gray[idx - W + 1]
          + gray[idx + W - 1] + (gray[idx + W] << 1) + gray[idx + W + 1];

        const edgeMag = Math.abs(gx) + Math.abs(gy);

        const dR = data[pIdx] - bgR;
        const dG = data[pIdx + 1] - bgG;
        const dB = data[pIdx + 2] - bgB;
        const colorDist = Math.sqrt(dR * dR + dG * dG + dB * dB);

        if (colorDist > 26 || edgeMag > 36) {
          fg[idx] = 1;
          fgCount++;
          sumX += x;
          sumY += y;
        }
      }
    }

    // Significance check: If fewer than 200 pixels, it's an empty table/wall -> DRAW NOTHING!
    if (fgCount < 200) {
      this.smoothedPoints = null;
      this.hasObject = false;
      return null;
    }

    const comX = sumX / fgCount;
    const comY = sumY / fgCount;

    // 4. Ray-Casting for exact outer silhouette (48 radial angles)
    const numRays = 48;
    const rawPts = [];
    const maxR = Math.min(W, H) * 0.46;

    for (let i = 0; i < numRays; i++) {
      const angle = (i * 2 * Math.PI) / numRays;
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);

      let hitR = 0;
      for (let r = 2; r < maxR; r += 1) {
        const px = Math.round(comX + cosA * r);
        const py = Math.round(comY + sinA * r);
        if (px < 1 || px >= W - 1 || py < 1 || py >= H - 1) break;
        if (fg[py * W + px] === 1) hitR = r;
      }

      if (hitR > 6) {
        rawPts.push({
          x: rx + (Math.round(comX + cosA * hitR) / W) * rw,
          y: ry + (Math.round(comY + sinA * hitR) / H) * rh
        });
      }
    }

    if (rawPts.length < 12) {
      this.smoothedPoints = null;
      this.hasObject = false;
      return null;
    }

    // 5. Exponential Smoothing across frames to prevent jitter
    if (!this.smoothedPoints || this.smoothedPoints.length !== rawPts.length) {
      this.smoothedPoints = rawPts.map(p => ({ x: p.x, y: p.y }));
    } else {
      for (let i = 0; i < rawPts.length; i++) {
        this.smoothedPoints[i].x += (rawPts[i].x - this.smoothedPoints[i].x) * this.smoothingFactor;
        this.smoothedPoints[i].y += (rawPts[i].y - this.smoothedPoints[i].y) * this.smoothingFactor;
      }
    }

    // 6. Draw Crisp Smooth Outline (Stroke ONLY, NO solid fill!)
    targetCtx.save();
    targetCtx.strokeStyle = strokeColor;
    targetCtx.lineWidth = lineWidth;
    targetCtx.lineJoin = 'round';
    targetCtx.lineCap = 'round';
    targetCtx.shadowColor = strokeColor;
    targetCtx.shadowBlur = 5;

    targetCtx.beginPath();
    const pts = this.smoothedPoints;
    const len = pts.length;

    targetCtx.moveTo((pts[0].x + pts[len - 1].x) / 2, (pts[0].y + pts[len - 1].y) / 2);
    for (let i = 0; i < len; i++) {
      const next = (i + 1) % len;
      const midX = (pts[i].x + pts[next].x) / 2;
      const midY = (pts[i].y + pts[next].y) / 2;
      targetCtx.quadraticCurveTo(pts[i].x, pts[i].y, midX, midY);
    }
    targetCtx.closePath();
    targetCtx.stroke(); // STROKE ONLY!

    targetCtx.restore();
    this.hasObject = true;

    return { points: pts };
  }

  reset() {
    this.smoothedPoints = null;
    this.hasObject = false;
  }
}
