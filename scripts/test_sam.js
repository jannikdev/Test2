import fs from 'fs';
import sharp from 'sharp';
import { SamModel, AutoProcessor, RawImage } from '@xenova/transformers';

async function run() {
  console.log('Loading SlimSAM (quantized ~13MB)...');
  const model = await SamModel.from_pretrained('Xenova/slimsam-77-uniform', { quantized: true });
  const processor = await AutoProcessor.from_pretrained('Xenova/slimsam-77-uniform');
  console.log('SlimSAM loaded!');

  const testIds = [1, 2, 3, 4, 13, 14, 15, 20]; // Mug, Wrench, Scissors, Phone, Hammer, Keys, Bottle, Screwdriver

  for (const id of testIds) {
    const rawPath = `test-data/raw_${String(id).padStart(2, '0')}.jpg`;
    if (!fs.existsSync(rawPath)) continue;

    console.log(`Verarbeite Testbild ${id}...`);
    const img = await RawImage.read(rawPath);
    const W = img.width;
    const H = img.height;

    // Prompt center of image
    const input_points = [[[[W / 2, H / 2]]]];
    const input_labels = [[[1]]];

    const inputs = await processor(img, input_points, input_labels);
    const outputs = await model(inputs);
    const masks = await processor.post_process_masks(outputs.pred_masks, inputs.original_sizes, inputs.reshaped_input_sizes);

    // Pick best IoU mask
    const scores = outputs.iou_scores.data;
    let bestIdx = 0;
    for (let i = 1; i < scores.length; i++) {
      if (scores[i] > scores[bestIdx]) bestIdx = i;
    }

    const maskData = masks[0].data.subarray(bestIdx * W * H, (bestIdx + 1) * W * H);

    // Binary mask threshold
    const binary = new Uint8Array(W * H);
    let count = 0;
    for (let i = 0; i < W * H; i++) {
      if (maskData[i] > 0) {
        binary[i] = 1;
        count++;
      }
    }

    // Find start pixel
    let startX = -1, startY = -1;
    for (let y = 1; y < H - 1 && startX === -1; y++) {
      for (let x = 1; x < W - 1; x++) {
        if (binary[y * W + x] === 1) {
          startX = x;
          startY = y;
          break;
        }
      }
    }

    if (startX === -1) {
      console.log(`Kein Objekt für ID ${id} gefunden`);
      continue;
    }

    // Trace boundary with Moore-Neighbor
    const dirs = [
      [0, -1], [1, -1], [1, 0], [1, 1],
      [0, 1], [-1, 1], [-1, 0], [-1, -1]
    ];

    const rawBoundary = [];
    let currX = startX, currY = startY;
    let dir = 0, steps = 0;

    while (steps < 8000) {
      rawBoundary.push({ x: currX, y: currY });
      let found = false;
      const startScan = (dir + 5) % 8;
      for (let i = 0; i < 8; i++) {
        const checkDir = (startScan + i) % 8;
        const nx = currX + dirs[checkDir][0];
        const ny = currY + dirs[checkDir][1];
        if (nx >= 0 && nx < W && ny >= 0 && ny < H && binary[ny * W + nx] === 1) {
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

    console.log(`Objekt ${id}: ${rawBoundary.length} Perimeter-Punkte, IoU: ${scores[bestIdx].toFixed(2)}`);

    // Subsample points
    const step = Math.max(1, Math.floor(rawBoundary.length / 120));
    const sub = [];
    for (let i = 0; i < rawBoundary.length; i += step) {
      sub.push(rawBoundary[i]);
    }

    let pathD = `M ${sub[0].x} ${sub[0].y} ` + sub.slice(1).map(p => `L ${p.x} ${p.y}`).join(' ') + ' Z';

    const svg = `
      <svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <filter id="glow-sam" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="0" stdDeviation="3" flood-color="#10b981" flood-opacity="0.9"/>
          </filter>
        </defs>
        <path d="${pathD}" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" filter="url(#glow-sam)"/>
        <g transform="translate(24, 28)">
          <rect x="0" y="0" width="180" height="30" rx="6" fill="rgba(10, 14, 20, 0.85)" stroke="rgba(16, 185, 129, 0.6)" stroke-width="1"/>
          <circle cx="14" cy="15" r="4" fill="#10b981"/>
          <text x="26" y="19" fill="#ffffff" font-family="-apple-system, sans-serif" font-size="11" font-weight="600">SlimSAM Neural Mask</text>
          <text x="140" y="19" fill="#10b981" font-family="monospace" font-size="11" font-weight="bold">${(scores[bestIdx]*100).toFixed(0)}%</text>
        </g>
      </svg>
    `;

    const rawBuf = fs.readFileSync(rawPath);
    const outBuf = await sharp(rawBuf)
      .composite([{ input: Buffer.from(svg), blend: 'over' }])
      .jpeg({ quality: 95 })
      .toBuffer();

    fs.writeFileSync(`test-data/sam_${String(id).padStart(2, '0')}.jpg`, outBuf);
  }

  console.log('SlimSAM Batch abgeschlossen!');
}

run();
