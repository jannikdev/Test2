/**
 * Background AI Inference Worker
 * Runs DINOv2 (~23MB quantized) for visual feature representation
 * Runs SegFormer for true neural object mask segmentation
 */

/* global importScripts */

importScripts('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js');

const { pipeline, env, RawImage } = self.transformers;
env.allowLocalModels = false;
env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/';
env.backends.onnx.wasm.numThreads = 1; // Single-thread WASM for 100% mobile compatibility

let featureExtractor = null;
let neuralSegmenter = null;
let isSegmenterLoading = false;

// Boundary tracing for neural mask
function traceMaskBoundary(maskData, w, h) {
  let startX = -1, startY = -1;
  // Look for top-most mask pixel near center
  for (let y = 8; y < h - 8 && startX === -1; y++) {
    for (let x = 8; x < w - 8; x++) {
      if (maskData[y * w + x] > 128) {
        startX = x;
        startY = y;
        break;
      }
    }
  }
  if (startX === -1) return null;

  const dirs = [
    [-1, 0], [-1, -1], [0, -1], [1, -1],
    [1, 0], [1, 1], [0, 1], [-1, 1]
  ];

  const points = [];
  let currX = startX;
  let currY = startY;
  let dirIdx = 7;
  let steps = 0;

  do {
    // Return normalized coordinates [0.0, 1.0]
    points.push({ x: currX / w, y: currY / h });

    let foundNext = false;
    for (let i = 0; i < 8; i++) {
      const checkDir = (dirIdx + i) % 8;
      const nx = currX + dirs[checkDir][0];
      const ny = currY + dirs[checkDir][1];
      if (nx >= 0 && nx < w && ny >= 0 && ny < h && maskData[ny * w + nx] > 128) {
        currX = nx;
        currY = ny;
        dirIdx = (checkDir + 5) % 8;
        foundNext = true;
        break;
      }
    }

    if (!foundNext) break;
    steps++;
  } while ((currX !== startX || currY !== startY) && steps < 500);

  // Subsample to ~35-45 clean points
  if (points.length < 10) return null;
  const step = Math.max(1, Math.floor(points.length / 40));
  const subsampled = [];
  for (let i = 0; i < points.length; i += step) {
    subsampled.push(points[i]);
  }
  return subsampled;
}

self.onmessage = async (e) => {
  const { type, reqId, buffer, width, height } = e.data;

  if (type === 'init') {
    try {
      self.postMessage({ type: 'status', msg: 'Lade Vision-Modell (DINOv2)...' });

      // 1. DINOv2 Feature Extractor (~23MB quantized)
      featureExtractor = await pipeline('image-feature-extraction', 'Xenova/dinov2-small', {
        quantized: true,
        progress_callback: (progress) => {
          if (progress.status === 'progress' || progress.status === 'downloading') {
            const percent = Math.round(progress.progress || 0);
            self.postMessage({
              type: 'progress',
              model: 'DINOv2',
              percent
            });
          }
        }
      });

      self.postMessage({ type: 'ready' });
    } catch (err) {
      console.error('Worker Init Fehler:', err);
      self.postMessage({ type: 'error', error: err.message || 'DINOv2 Ladefehler' });
    }
  }

  else if (type === 'load_segmenter') {
    if (neuralSegmenter || isSegmenterLoading) return;
    isSegmenterLoading = true;
    try {
      self.postMessage({ type: 'status', msg: 'Lade KI-Segmentierer (SegFormer)...' });
      neuralSegmenter = await pipeline('image-segmentation', 'Xenova/segformer-b0-finetuned-ade-512-512', {
        quantized: true,
        progress_callback: (progress) => {
          if (progress.status === 'progress' || progress.status === 'downloading') {
            const percent = Math.round(progress.progress || 0);
            self.postMessage({
              type: 'progress',
              model: 'KI-Maske',
              percent
            });
          }
        }
      });
      self.postMessage({ type: 'segmenter_ready' });
    } catch (segErr) {
      console.warn('KI-Segmentierer Fehler:', segErr);
      self.postMessage({ type: 'segmenter_error', error: segErr.message });
    } finally {
      isSegmenterLoading = false;
    }
  }

  else if (type === 'extract') {
    if (!featureExtractor || !buffer) {
      self.postMessage({ type: 'extract_result', reqId, vector: null });
      return;
    }

    try {
      const raw = new RawImage(new Uint8Array(buffer), width, height, 3);
      const output = await featureExtractor(raw);

      // Extract 384-dim CLS token
      const cls = output.slice(0, 0).data;

      // L2 Normalization
      let sumSq = 0;
      for (let i = 0; i < cls.length; i++) {
        sumSq += cls[i] * cls[i];
      }
      const norm = Math.sqrt(sumSq) || 1;
      const vector = new Array(cls.length);
      for (let i = 0; i < cls.length; i++) {
        vector[i] = cls[i] / norm;
      }

      self.postMessage({ type: 'extract_result', reqId, vector });
    } catch (err) {
      console.error('Extraction Error:', err);
      self.postMessage({ type: 'extract_result', reqId, vector: null, error: err.message });
    }
  }

  else if (type === 'segment') {
    if (!neuralSegmenter || !buffer) {
      self.postMessage({ type: 'segment_result', reqId, polygon: null });
      return;
    }

    try {
      const raw = new RawImage(new Uint8Array(buffer), width, height, 3);
      const output = await neuralSegmenter(raw);

      // Find foreground object segment
      const bgLabels = ['wall', 'floor', 'ceiling', 'sky', 'ground', 'earth', 'mountain'];
      let bestSegment = null;
      let maxScore = -1;

      for (const seg of output) {
        if (!bgLabels.includes(seg.label.toLowerCase()) && seg.score > maxScore) {
          maxScore = seg.score;
          bestSegment = seg;
        }
      }

      if (!bestSegment && output.length > 0) {
        bestSegment = output[0];
      }

      if (bestSegment && bestSegment.mask) {
        const polygon = traceMaskBoundary(bestSegment.mask.data, bestSegment.mask.width, bestSegment.mask.height);
        self.postMessage({ type: 'segment_result', reqId, polygon });
      } else {
        self.postMessage({ type: 'segment_result', reqId, polygon: null });
      }
    } catch (err) {
      console.error('Segmentation Error:', err);
      self.postMessage({ type: 'segment_result', reqId, polygon: null });
    }
  }
};
