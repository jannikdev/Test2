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

    // Pick mask: SAM outputs [0: whole scene, 1: object body, 2: subpart]
    // Mask 1 represents the focused object without surrounding context/background noise
    const scores = outputs.iou_scores.data;
    let chosenIdx = 1;
    if (scores[1] < 0.65) {
      chosenIdx = scores[0] > scores[2] ? 0 : 2;
    }

    const rawMask = masks[0].data.subarray(chosenIdx * W * H, (chosenIdx + 1) * W * H);

    // Connected component flood-fill from query point (removes all isolated background / shadow noise)
    const cx = Math.floor(W / 2);
    const cy = Math.floor(H / 2);
    let seedX = cx, seedY = cy;
    if (rawMask[cy * W + cx] !== 1) {
      let found = false;
      for (let r = 1; r < 60 && !found; r++) {
        for (let dy = -r; dy <= r && !found; dy++) {
          for (let dx = -r; dx <= r && !found; dx++) {
            const px = cx + dx, py = cy + dy;
            if (px >= 0 && px < W && py >= 0 && py < H && rawMask[py * W + px] === 1) {
              seedX = px; seedY = py; found = true;
            }
          }
        }
      }
    }

    const cleanMask = new Uint8Array(W * H);
    if (rawMask[seedY * W + seedX] === 1) {
      const queue = [seedY * W + seedX];
      cleanMask[seedY * W + seedX] = 1;
      let head = 0;
      while (head < queue.length) {
        const idx = queue[head++];
        const x = idx % W;
        const y = Math.floor(idx / W);
        const nbs = [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]];
        for (const [nx, ny] of nbs) {
          if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
            const nidx = ny * W + nx;
            if (cleanMask[nidx] === 0 && rawMask[nidx] === 1) {
              cleanMask[nidx] = 1;
              queue.push(nidx);
            }
          }
        }
      }
    }

    // Extract exact perimeter pixels of the connected object
    const rgba = new Uint8Array(W * H * 4);
    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const idx = y * W + x;
        if (cleanMask[idx] === 1) {
          if (
            cleanMask[idx - 1] === 0 || cleanMask[idx + 1] === 0 ||
            cleanMask[idx - W] === 0 || cleanMask[idx + W] === 0 ||
            cleanMask[idx - W - 1] === 0 || cleanMask[idx - W + 1] === 0 ||
            cleanMask[idx + W - 1] === 0 || cleanMask[idx + W + 1] === 0
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
          <text x="150" y="19" fill="#34d399" font-family="monospace" font-size="11" font-weight="bold">${(scores[chosenIdx] * 100).toFixed(0)}%</text>
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
    console.log(`Saved sam_${id}.jpg (IoU: ${(scores[chosenIdx] * 100).toFixed(1)}%)`);
  }

  console.log('All SlimSAM masks generated successfully!');
}

run();
