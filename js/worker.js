/**
 * Background AI Inference Worker
 * Models:
 * 1. DINOv2 (~23MB quantized) for visual feature representation & matching
 * 2. SlimSAM (~13MB quantized) for pixel-perfect Segment-Anything object contours
 */

/* global importScripts */

// Load transformers library: local bundle first, then CDNs
const workerBase = (typeof self !== 'undefined' && self.location)
  ? self.location.href.substring(0, self.location.href.lastIndexOf('/') + 1)
  : './';

const scriptCandidates = [
  workerBase + 'transformers.min.js?v=2.17.3',
  './transformers.min.js?v=2.17.3',
  './transformers.min.js'
];

for (const src of scriptCandidates) {
  if (self.transformers) break;
  try {
    importScripts(src);
    if (self.transformers) {
      console.log('Transformers.js erfolgreich geladen von:', src);
      break;
    }
  } catch (err) {
    console.warn('Konnte Transformers nicht laden von ' + src + ':', err);
  }
}

if (!self.transformers) {
  throw new Error('Konnte transformers.min.js lokal nicht laden.');
}

const { pipeline, env, RawImage, SamModel, AutoProcessor } = self.transformers;
env.allowLocalModels = false;
env.backends.onnx.wasm.wasmPaths = workerBase;
env.backends.onnx.wasm.numThreads = 1;

let featureExtractor = null;
let samModel = null;
let samProcessor = null;
let isSamLoading = false;

// Flood-fill connected component and Moore-Neighbor perimeter tracer on SAM binary mask
function extractSamPolygon(maskData, W, H) {
  const cx = Math.floor(W / 2);
  const cy = Math.floor(H / 2);

  // Find seed pixel near center prompt point
  let seedX = cx, seedY = cy;
  if (maskData[cy * W + cx] !== 1) {
    let found = false;
    for (let r = 1; r < 60 && !found; r++) {
      for (let dy = -r; dy <= r && !found; dy++) {
        for (let dx = -r; dx <= r && !found; dx++) {
          const px = cx + dx, py = cy + dy;
          if (px >= 4 && px < W - 4 && py >= 4 && py < H - 4 && maskData[py * W + px] === 1) {
            seedX = px;
            seedY = py;
            found = true;
          }
        }
      }
    }
    if (!found) return null;
  }

  // Flood fill to isolate ONLY the connected component containing the target object.
  // This cleanly discards all isolated noise islands, background artifacts, or stray shadow pixels.
  const cleanMask = new Uint8Array(W * H);
  const queue = [seedY * W + seedX];
  cleanMask[seedY * W + seedX] = 1;
  let head = 0;
  let startX = seedX, startY = seedY;

  while (head < queue.length) {
    const idx = queue[head++];
    const x = idx % W;
    const y = Math.floor(idx / W);

    // Track top-most pixel of the object for Moore-Neighbor start
    if (y < startY || (y === startY && x < startX)) {
      startX = x;
      startY = y;
    }

    const nbs = [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]];
    for (let i = 0; i < 4; i++) {
      const nx = nbs[i][0], ny = nbs[i][1];
      if (nx >= 2 && nx < W - 2 && ny >= 2 && ny < H - 2) {
        const nidx = ny * W + nx;
        if (cleanMask[nidx] === 0 && maskData[nidx] === 1) {
          cleanMask[nidx] = 1;
          queue.push(nidx);
        }
      }
    }
  }

  // Reject tiny noise (< 40 pixels)
  if (queue.length < 40) return null;

  const dirs = [
    [0, -1], [1, -1], [1, 0], [1, 1],
    [0, 1], [-1, 1], [-1, 0], [-1, -1]
  ];

  const rawBoundary = [];
  let currX = startX, currY = startY;
  let dir = 0, steps = 0;

  while (steps < 4000) {
    rawBoundary.push({ x: currX / W, y: currY / H });
    let found = false;
    const startScan = (dir + 5) % 8;
    for (let i = 0; i < 8; i++) {
      const checkDir = (startScan + i) % 8;
      const nx = currX + dirs[checkDir][0];
      const ny = currY + dirs[checkDir][1];
      if (nx >= 0 && nx < W && ny >= 0 && ny < H && cleanMask[ny * W + nx] === 1) {
        currX = nx;
        currY = ny;
        dir = checkDir;
        found = true;
        break;
      }
    }
    if (!found) break;
    steps++;
    if (currX === startX && currY === startY && steps > 5) break;
  }

  if (rawBoundary.length < 15) return null;

  // Subsample to ~60-80 clean vector vertices
  const step = Math.max(1, Math.floor(rawBoundary.length / 75));
  const subsampled = [];
  for (let i = 0; i < rawBoundary.length; i += step) {
    subsampled.push(rawBoundary[i]);
  }
  return subsampled;
}

self.onmessage = async (e) => {
  const { type, reqId, buffer, width, height } = e.data;

  if (type === 'init') {
    try {
      self.postMessage({ type: 'status', msg: 'Lade DINOv2 (23 MB)...' });

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
    if (samModel || isSamLoading) return;
    isSamLoading = true;
    try {
      self.postMessage({ type: 'status', msg: 'Lade SlimSAM (13 MB)...' });
      samModel = await SamModel.from_pretrained('Xenova/slimsam-77-uniform', {
        quantized: true,
        progress_callback: (progress) => {
          if (progress.status === 'progress' || progress.status === 'downloading') {
            const percent = Math.round(progress.progress || 0);
            self.postMessage({
              type: 'progress',
              model: 'SlimSAM',
              percent
            });
          }
        }
      });
      samProcessor = await AutoProcessor.from_pretrained('Xenova/slimsam-77-uniform');
      self.postMessage({ type: 'segmenter_ready' });
    } catch (segErr) {
      console.warn('SlimSAM konnte nicht geladen werden:', segErr);
      self.postMessage({ type: 'segmenter_error', error: segErr.message });
    } finally {
      isSamLoading = false;
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
      for (let i = 0; i < cls.length; i++) sumSq += cls[i] * cls[i];
      const norm = Math.sqrt(sumSq) || 1;
      const vector = new Array(cls.length);
      for (let i = 0; i < cls.length; i++) vector[i] = cls[i] / norm;

      self.postMessage({ type: 'extract_result', reqId, vector });
    } catch (err) {
      console.error('Extraction Error:', err);
      self.postMessage({ type: 'extract_result', reqId, vector: null, error: err.message });
    }
  }

  else if (type === 'segment') {
    if (!samModel || !samProcessor || !buffer) {
      self.postMessage({ type: 'segment_result', reqId, polygon: null });
      return;
    }

    try {
      const raw = new RawImage(new Uint8Array(buffer), width, height, 3);
      const input_points = [[[[width / 2, height / 2]]]];
      const input_labels = [[[1]]];

      const inputs = await samProcessor(raw, input_points, input_labels);
      const outputs = await samModel(inputs);
      const masks = await samProcessor.post_process_masks(outputs.pred_masks, inputs.original_sizes, inputs.reshaped_input_sizes);

      // Select mask: SAM outputs [0: whole scene, 1: object body, 2: subpart]
      // Mask 1 represents the focused object without surrounding context/background noise
      const scores = outputs.iou_scores.data;
      let chosenIdx = 1;
      if (scores[1] < 0.65) {
        chosenIdx = scores[0] > scores[2] ? 0 : 2;
      }

      const rawMask = masks[0].data.subarray(chosenIdx * width * height, (chosenIdx + 1) * width * height);
      const polygon = extractSamPolygon(rawMask, width, height);

      self.postMessage({ type: 'segment_result', reqId, polygon, iou: scores[chosenIdx] });
    } catch (err) {
      console.error('SlimSAM Error:', err);
      self.postMessage({ type: 'segment_result', reqId, polygon: null });
    }
  }
};
