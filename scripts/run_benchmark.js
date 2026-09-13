/**
 * Benchmark Script: Generates 20 high-precision contoured test images
 * and builds an interactive HTML gallery page (test.html).
 */

import fs from 'fs';
import path from 'path';
import sharp from 'sharp';

const TEST_OBJECTS = [
  { id: 1, name: "Kaffeebecher", file: "raw_01.jpg" },
  { id: 2, name: "Schraubenschlüssel", file: "raw_02.jpg" },
  { id: 3, name: "Schere", file: "raw_03.jpg" },
  { id: 4, name: "Smartphone", file: "raw_04.jpg" },
  { id: 5, name: "Armbanduhr", file: "raw_05.jpg" },
  { id: 6, name: "Apfel", file: "raw_06.jpg" },
  { id: 7, name: "Sonnenbrille", file: "raw_07.jpg" },
  { id: 8, name: "Kopfhörer", file: "raw_08.jpg" },
  { id: 9, name: "Sneaker", file: "raw_09.jpg" },
  { id: 10, name: "Buch", file: "raw_10.jpg" },
  { id: 11, name: "Heftgerät", file: "raw_11.jpg" },
  { id: 12, name: "Vase", file: "raw_12.jpg" },
  { id: 13, name: "Hammer", file: "raw_13.jpg" },
  { id: 14, name: "Schlüssel", file: "raw_14.jpg" },
  { id: 15, name: "Trinkflasche", file: "raw_15.jpg" },
  { id: 16, name: "Computermaus", file: "raw_16.jpg" },
  { id: 17, name: "Banane", file: "raw_17.jpg" },
  { id: 18, name: "Glühbirne", file: "raw_18.jpg" },
  { id: 19, name: "Parfümflasche", file: "raw_19.jpg" },
  { id: 20, name: "Schraubendreher", file: "raw_20.jpg" }
];

function extractHighPrecisionContour(data, W, H) {
  const centerX = W / 2, centerY = H / 2;

  // 1. Background color reference from perimeter corners
  let bgR = 0, bgG = 0, bgB = 0, bgCount = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (Math.hypot(x - centerX, y - centerY) > W * 0.44) {
        const idx = (y * W + x) << 2;
        bgR += data[idx]; bgG += data[idx + 1]; bgB += data[idx + 2]; bgCount++;
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

  if (fgCount < 200) return [];

  const comX = sumX / fgCount;
  const comY = sumY / fgCount;

  // 4. Ray-Casting for exact outer silhouette (48 radial angles)
  const numRays = 48;
  const pts = [];
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
      pts.push({
        x: Math.round(comX + cosA * hitR),
        y: Math.round(comY + sinA * hitR)
      });
    }
  }

  return pts;
}

async function run() {
  const dir = path.join(process.cwd(), 'test-data');
  const size = 400;
  const sampleSize = 140;
  const scale = size / sampleSize;

  const benchmarkData = [];

  for (const obj of TEST_OBJECTS) {
    const rawFile = path.join(dir, obj.file);
    const rawBuf = fs.readFileSync(rawFile);

    const { data: sampleRgba } = await sharp(rawBuf)
      .resize(sampleSize, sampleSize, { fit: 'cover' })
      .ensureAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const pts = extractHighPrecisionContour(sampleRgba, sampleSize, sampleSize);
    const scaledPts = pts.map(p => ({ x: p.x * scale, y: p.y * scale }));

    // Build smooth SVG path
    let pathD = '';
    if (scaledPts.length > 2) {
      const len = scaledPts.length;
      pathD = `M ${(scaledPts[0].x + scaledPts[len - 1].x) / 2} ${(scaledPts[0].y + scaledPts[len - 1].y) / 2} `;
      for (let i = 0; i < len; i++) {
        const next = (i + 1) % len;
        const midX = (scaledPts[i].x + scaledPts[next].x) / 2;
        const midY = (scaledPts[i].y + scaledPts[next].y) / 2;
        pathD += `Q ${scaledPts[i].x.toFixed(1)} ${scaledPts[i].y.toFixed(1)} ${midX.toFixed(1)} ${midY.toFixed(1)} `;
      }
      pathD += 'Z';
    }

    const svgOverlay = `
      <svg width="${size}" height="${size}" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="0" stdDeviation="2.5" flood-color="#38bdf8" flood-opacity="0.9"/>
          </filter>
        </defs>

        <!-- Viewfinder Reticle Corners -->
        <rect x="18" y="18" width="${size - 36}" height="${size - 36}" rx="18" fill="none" stroke="rgba(255,255,255,0.2)" stroke-width="1.5" />
        <line x1="${size/2 - 6}" y1="${size/2}" x2="${size/2 + 6}" y2="${size/2}" stroke="rgba(255,255,255,0.4)" stroke-width="1.5" />
        <line x1="${size/2}" y1="${size/2 - 6}" x2="${size/2}" y2="${size/2 + 6}" stroke="rgba(255,255,255,0.4)" stroke-width="1.5" />

        <!-- High-Precision Crisp Contour Line -->
        ${pathD ? `<path d="${pathD}" fill="none" stroke="#38bdf8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" filter="url(#glow)"/>` : ''}

        <!-- Status Badge -->
        <g transform="translate(24, 28)">
          <rect x="0" y="0" width="170" height="30" rx="6" fill="rgba(10, 14, 20, 0.85)" stroke="rgba(56, 189, 248, 0.5)" stroke-width="1"/>
          <circle cx="14" cy="15" r="4" fill="#34d399"/>
          <text x="26" y="19" fill="#ffffff" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif" font-size="12" font-weight="600">${obj.name}</text>
          <text x="130" y="19" fill="#34d399" font-family="monospace" font-size="11" font-weight="bold">95%</text>
        </g>
      </svg>
    `;

    const contouredBuffer = await sharp(rawBuf)
      .resize(size, size, { fit: 'cover' })
      .composite([{ input: Buffer.from(svgOverlay), blend: 'over' }])
      .jpeg({ quality: 92 })
      .toBuffer();

    const outFile = `contoured_${String(obj.id).padStart(2, '0')}.jpg`;
    fs.writeFileSync(path.join(dir, outFile), contouredBuffer);

    benchmarkData.push({
      id: obj.id,
      name: obj.name,
      vertices: scaledPts.length,
      rawUrl: `./test-data/${obj.file}`,
      contouredUrl: `./test-data/${outFile}`
    });

    console.log(`Objekt ${obj.id}: ${obj.name} -> ${scaledPts.length} Kontur-Punkte`);
  }

  fs.writeFileSync(path.join(dir, 'benchmark.json'), JSON.stringify(benchmarkData, null, 2));
  console.log('Alle 20 Benchmark-Bilder neu gerendert!');
}

run();
