/**
 * Background AI Inference Worker
 * Loads lightweight models (DINOv2 for feature extraction, DeepLabV3-MobileViT for segmentation)
 * Receives frames via zero-copy Transferable ImageBitmap.
 */

/* global importScripts, OffscreenCanvas */

// Load Transformers.js
importScripts('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js');

const { pipeline, env } = self.transformers;
env.allowLocalModels = false;
env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/';
env.backends.onnx.wasm.numThreads = 2;

let featureExtractor = null;
let neuralSegmenter = null;

let isReady = false;

self.onmessage = async (e) => {
  const { type, reqId, imageBitmap } = e.data;

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
              percent,
              loaded: progress.loaded,
              total: progress.total
            });
          }
        }
      });

      self.postMessage({ type: 'status', msg: 'Lade KI-Segmentierer (MobileViT)...' });

      // 2. Ultra-lightweight Neural Segmentation (~2.4MB quantized)
      try {
        neuralSegmenter = await pipeline('image-segmentation', 'Xenova/deeplabv3-mobilevit-xx-small', {
          quantized: true,
          progress_callback: (progress) => {
            if (progress.status === 'progress') {
              self.postMessage({
                type: 'progress',
                model: 'MobileViT',
                percent: Math.round(progress.progress || 0)
              });
            }
          }
        });
      } catch (segErr) {
        console.warn('Optionale neuronale Segmentierung nicht geladen, Fallback aktiv:', segErr);
      }

      isReady = true;
      self.postMessage({ type: 'ready' });
    } catch (err) {
      console.error('Worker Init Fehler:', err);
      self.postMessage({ type: 'error', error: err.message || 'Initialisierung fehlgeschlagen' });
    }
  }

  else if (type === 'extract') {
    if (!featureExtractor || !imageBitmap) {
      if (imageBitmap) imageBitmap.close();
      self.postMessage({ type: 'extract_result', reqId, vector: null });
      return;
    }

    try {
      // Draw ImageBitmap to OffscreenCanvas
      const offscreen = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
      const ctx = offscreen.getContext('2d');
      ctx.drawImage(imageBitmap, 0, 0);
      imageBitmap.close(); // Immediate memory release

      // Extract 384-dimensional normalized feature embedding
      const output = await featureExtractor(offscreen, { pooling: 'mean', normalize: true });
      const vector = Array.from(output.data);

      self.postMessage({ type: 'extract_result', reqId, vector });
    } catch (err) {
      console.error('Extraction Error:', err);
      self.postMessage({ type: 'extract_result', reqId, vector: null, error: err.message });
    }
  }

  else if (type === 'segment') {
    if (!neuralSegmenter || !imageBitmap) {
      if (imageBitmap) imageBitmap.close();
      self.postMessage({ type: 'segment_result', reqId, mask: null });
      return;
    }

    try {
      const offscreen = new OffscreenCanvas(imageBitmap.width, imageBitmap.height);
      const ctx = offscreen.getContext('2d');
      ctx.drawImage(imageBitmap, 0, 0);
      imageBitmap.close();

      const output = await neuralSegmenter(offscreen);

      // Extract foreground object silhouette/mask
      let bestMask = null;
      if (Array.isArray(output) && output.length > 0) {
        // Pick primary salient foreground mask
        bestMask = output[0].mask;
      }

      self.postMessage({ type: 'segment_result', reqId, mask: bestMask });
    } catch (err) {
      console.error('Segmentation Error:', err);
      self.postMessage({ type: 'segment_result', reqId, mask: null });
    }
  }
};
