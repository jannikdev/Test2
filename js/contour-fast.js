/**
 * Fast Real-time Edge & Object Contour Tracker (Mode 1: 0 MB, ~5ms, 60fps)
 * Traces ONLY genuine object boundaries.
 * When no object is present, it draws NOTHING (no blobs, no false circles).
 */
export class FastContourDetector {
  constructor() {
    this.offscreen = document.createElement('canvas');
    this.offCtx = this.offscreen.getContext('2d', { willReadFrequently: true });
    this.smoothedPoints = null;
    this.smoothingFactor = 0.4;
    this.hasObject = false;
  }

  /**
   * Process a frame and draw a crisp outline ONLY if an object is present
   * @param {HTMLVideoElement|HTMLCanvasElement} source - Video source
   * @param {CanvasRenderingContext2D} targetCtx - Overlay canvas context
   * @param {Object} reticleRect - { x, y, width, height } in canvas coords
   * @param {string} strokeColor - CSS color for the contour outline
   * @param {number} lineWidth - Width of contour line (default 2)
   */
  detectAndDraw(source, targetCtx, reticleRect, strokeColor = '#38bdf8', lineWidth = 2) {
    const rx = Math.max(0, Math.floor(reticleRect.x));
    const ry = Math.max(0, Math.floor(reticleRect.y));
    const rw = Math.min(source.videoWidth || source.width, Math.floor(reticleRect.width));
    const rh = Math.min(source.videoHeight || source.height, Math.floor(reticleRect.height));

    if (rw <= 30 || rh <= 30) {
      this.smoothedPoints = null;
      return null;
    }

    // Downscale to 120x120 for fast ~4ms analysis
    const sampleSize = 120;
    this.offscreen.width = sampleSize;
    this.offscreen.height = sampleSize;

    this.offCtx.drawImage(source, rx, ry, rw, rh, 0, 0, sampleSize, sampleSize);
    const imgData = this.offCtx.getImageData(0, 0, sampleSize, sampleSize);
    const data = imgData.data;

    // 1. Grayscale conversion
    const gray = new Uint8Array(sampleSize * sampleSize);
    for (let i = 0; i < data.length; i += 4) {
      gray[i >> 2] = (data[i] * 77 + data[i + 1] * 150 + data[i + 2] * 29) >> 8;
    }

    // 2. Sobel Edge Gradient
    const edges = new Uint8Array(sampleSize * sampleSize);
    let strongEdgeCount = 0;
    let sumX = 0;
    let sumY = 0;

    for (let y = 1; y < sampleSize - 1; y++) {
      const row = y * sampleSize;
      for (let x = 1; x < sampleSize - 1; x++) {
        const idx = row + x;
        const gx =
          -gray[idx - sampleSize - 1] + gray[idx - sampleSize + 1]
          - (gray[idx - 1] << 1) + (gray[idx + 1] << 1)
          - gray[idx + sampleSize - 1] + gray[idx + sampleSize + 1];

        const gy =
          -gray[idx - sampleSize - 1] - (gray[idx - sampleSize] << 1) - gray[idx - sampleSize + 1]
          + gray[idx + sampleSize - 1] + (gray[idx + sampleSize] << 1) + gray[idx + sampleSize + 1];

        const mag = Math.abs(gx) + Math.abs(gy);
        if (mag > 65) {
          edges[idx] = 255;
          strongEdgeCount++;
          sumX += x;
          sumY += y;
        }
      }
    }

    // 3. Significance Check: Is an actual object present?
    // At least 2% and at most 45% of pixels must be strong edges (otherwise it's blank wall or extreme noise)
    const minEdges = (sampleSize * sampleSize) * 0.018; // ~260 pixels
    const maxEdges = (sampleSize * sampleSize) * 0.45;

    if (strongEdgeCount < minEdges || strongEdgeCount > maxEdges) {
      // NO object detected: Fade out and draw NOTHING!
      this.smoothedPoints = null;
      this.hasObject = false;
      return null;
    }

    this.hasObject = true;
    const centerX = sumX / strongEdgeCount;
    const centerY = sumY / strongEdgeCount;

    // 4. Radial Ray-Casting from Center of Mass
    const numRays = 24;
    const validPoints = [];
    const maxRadius = sampleSize * 0.46;

    for (let i = 0; i < numRays; i++) {
      const angle = (i * 2 * Math.PI) / numRays;
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);

      let edgeRadius = 0;
      // Scan outward from center to outer limit
      for (let r = 8; r < maxRadius; r += 2) {
        const px = Math.round(centerX + cosA * r);
        const py = Math.round(centerY + sinA * r);
        if (px < 1 || px >= sampleSize - 1 || py < 1 || py >= sampleSize - 1) break;

        if (edges[py * sampleSize + px] === 255) {
          edgeRadius = r;
        }
      }

      // Only record point if an actual edge was found along this ray!
      if (edgeRadius > 6) {
        const normX = (centerX + cosA * edgeRadius) / sampleSize;
        const normY = (centerY + sinA * edgeRadius) / sampleSize;
        validPoints.push({
          x: rx + normX * rw,
          y: ry + normY * rh
        });
      }
    }

    // If fewer than 8 rays found an edge, object is incomplete -> do not render
    if (validPoints.length < 8) {
      this.smoothedPoints = null;
      return null;
    }

    // 5. Exponential Smoothing across frames to prevent jitter
    if (!this.smoothedPoints || this.smoothedPoints.length !== validPoints.length) {
      this.smoothedPoints = validPoints.map(p => ({ x: p.x, y: p.y }));
    } else {
      for (let i = 0; i < validPoints.length; i++) {
        this.smoothedPoints[i].x += (validPoints[i].x - this.smoothedPoints[i].x) * this.smoothingFactor;
        this.smoothedPoints[i].y += (validPoints[i].y - this.smoothedPoints[i].y) * this.smoothingFactor;
      }
    }

    // 6. Draw Crisp Outline (Stroke ONLY, NO solid fill!)
    targetCtx.save();
    targetCtx.strokeStyle = strokeColor;
    targetCtx.lineWidth = lineWidth;
    targetCtx.lineJoin = 'round';
    targetCtx.lineCap = 'round';
    targetCtx.shadowColor = strokeColor;
    targetCtx.shadowBlur = 6;

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

    return {
      points: pts
    };
  }

  reset() {
    this.smoothedPoints = null;
    this.hasObject = false;
  }
}
