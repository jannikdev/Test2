/**
 * Generate Comparative Benchmarks:
 * Compares the Classical Heuristic Contour vs Meta's SlimSAM Neural Segmentation.
 */

import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import { SamModel, AutoProcessor, RawImage } from '@xenova/transformers';

async function run() {
  console.log('Loading SlimSAM Model (quantized ~13MB)...');
  const model = await SamModel.from_pretrained('Xenova/slimsam-77-uniform', { quantized: true });
  const processor = await AutoProcessor.from_pretrained('Xenova/slimsam-77-uniform');
  console.log('SlimSAM ready!');

  const dir = path.join(process.cwd(), 'test-data');
  const files = fs.readdirSync(dir).filter(f => f.startsWith('raw_') && f.endsWith('.jpg'));

  for (const file of files) {
    const rawPath = path.join(dir, file);
    const id = file.replace('raw_', '').replace('.jpg', '');
    console.log(`Processing Object ${id} with SlimSAM...`);

    const img = await RawImage.read(rawPath);
    const W = img.width;
    const H = img.height;

    // Prompt point at center
    const input_points = [[[[W / 2, H / 2]]]];
    const input_labels = [[[1]]];

    const inputs = await processor(img, input_points, input_labels);
    const outputs = await model(inputs);
    const masks = await processor.post_process_masks(outputs.pred_masks, inputs.original_sizes, inputs.reshaped_input_sizes);

    // Pick mask with best IoU score
    const scores = outputs.iou_scores.data;
    let bestIdx = 0;
    for (let i = 1; i < scores.length; i++) {
      if (scores[i] > scores[bestIdx]) bestIdx = i;
    }

    const rawMask = masks[0].data.subarray(bestIdx * W * H, (bestIdx + 1) * W * H);

    // Extract exact perimeter pixels (boundary where mask=1 touches mask=0)
    const rgba = new Uint8Array(W * H * 4);
    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const idx = y * W + x;
        if (rawMask[idx] === 1) {
          if (
            rawMask[idx - 1] === 0 || rawMask[idx + 1] === 0 ||
            rawMask[idx - W] === 0 || rawMask[idx + W] === 0 ||
            rawMask[idx - W - 1] === 0 || rawMask[idx - W + 1] === 0 ||
            rawMask[idx + W - 1] === 0 || rawMask[idx + W + 1] === 0
          ) {
            const p = idx * 4;
            rgba[p] = 52;     // Emerald #34d399 R
            rgba[p + 1] = 211; // G
            rgba[p + 2] = 153; // B
            rgba[p + 3] = 255; // A
          }
        }
      }
    }

    // Dilate by 1px for clean crisp 2px stroke
    const dilated = new Uint8Array(rgba);
    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const idx = y * W + x;
        if (rgba[idx * 4 + 3] === 255) {
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              const n = (y + dy) * W + (x + dx);
              const p = n * 4;
              if (dilated[p + 3] === 0) {
                dilated[p] = 52;
                dilated[p + 1] = 211;
                dilated[p + 2] = 153;
                dilated[p + 3] = 200;
              }
            }
          }
        }
      }
    }

    // Overlay Badge
    const badgeSvg = `
      <svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg">
        <rect x="18" y="18" width="${W - 36}" height="${H - 36}" rx="18" fill="none" stroke="rgba(255,255,255,0.2)" stroke-width="1.5" />
        <g transform="translate(24, 28)">
          <rect x="0" y="0" width="190" height="30" rx="6" fill="rgba(10, 14, 20, 0.88)" stroke="rgba(52, 211, 153, 0.6)" stroke-width="1"/>
          <circle cx="14" cy="15" r="4" fill="#34d399"/>
          <text x="26" y="19" fill="#ffffff" font-family="-apple-system, sans-serif" font-size="11" font-weight="600">SlimSAM Neural Mask</text>
          <text x="150" y="19" fill="#34d399" font-family="monospace" font-size="11" font-weight="bold">${(scores[bestIdx] * 100).toFixed(0)}%</text>
        </g>
      </svg>
    `;

    const rawBuf = fs.readFileSync(rawPath);
    const result = await sharp(rawBuf)
      .composite([
        { input: Buffer.from(dilated), raw: { width: W, height: H, channels: 4 }, blend: 'over' },
        { input: Buffer.from(badgeSvg), blend: 'over' }
      ])
      .jpeg({ quality: 92 })
      .toBuffer();

    const outFile = path.join(dir, `sam_${id}.jpg`);
    fs.writeFileSync(outFile, result);
    console.log(`Saved sam_${id}.jpg (IoU: ${(scores[bestIdx] * 100).toFixed(1)}%)`);
  }

  console.log('All SlimSAM masks generated successfully!');
}

run();
