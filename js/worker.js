/**
 * Background AI Inference Worker
 * Runs DINOv2 (~23MB quantized) for visual feature representation
 * Uses RawImage RGB buffers to ensure 100% mobile browser compatibility.
 */

/* global importScripts */

importScripts('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js');

const { pipeline, env, RawImage } = self.transformers;
env.allowLocalModels = false;
env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/';
// CRITICAL: numThreads = 1 prevents SharedArrayBuffer crashes on GitHub Pages / Mobile
env.backends.onnx.wasm.numThreads = 1;

let featureExtractor = null;
let neuralSegmenter = null;
let isSegmenterLoading = false;

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
      console.warn('KI-Segmentierer konnte nicht geladen werden:', segErr);
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

      // Slice out the 384-dimensional CLS token
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
      self.postMessage({ type: 'segment_result', reqId, mask: null });
      return;
    }

    try {
      const raw = new RawImage(new Uint8Array(buffer), width, height, 3);
      const output = await neuralSegmenter(raw);

      // Extract detected foreground segments
      self.postMessage({ type: 'segment_result', reqId, segments: output });
    } catch (err) {
      console.error('Segmentation Error:', err);
      self.postMessage({ type: 'segment_result', reqId, segments: null });
    }
  }
};
