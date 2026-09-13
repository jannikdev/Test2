/**
 * Precision Vector Matcher & Spatial Color Descriptor
 * Features Hand & Finger Color Filtering to prevent skin tones from polluting object signatures.
 */

// Cosine similarity between two unit-normalized Float32/Number arrays
export function cosineSimilarity(vecA, vecB) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  const len = vecA.length;
  for (let i = 0; i < len; i++) {
    const a = vecA[i];
    const b = vecB[i];
    dot += a * b;
    normA += a * a;
    normB += b * b;
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Extract a 30-dimensional Spatial Color & Texture Descriptor
 * Filters out human skin/finger pixels so holding hands do not pollute object color.
 * @param {ImageData} imageData 
 * @returns {Float32Array}
 */
export function extractSpatialColorDescriptor(imageData) {
  const w = imageData.width;
  const h = imageData.height;
  const data = imageData.data;

  // 5 Regions: 4 quadrants + center overlapping crop
  const regions = [
    { x0: 0, y0: 0, x1: w * 0.55, y1: h * 0.55 },             // Top-Left
    { x0: w * 0.45, y0: 0, x1: w, y1: h * 0.55 },             // Top-Right
    { x0: w * 0.25, y0: h * 0.25, x1: w * 0.75, y1: h * 0.75 },// Center (focal)
    { x0: 0, y0: h * 0.45, x1: w * 0.55, y1: h },             // Bottom-Left
    { x0: w * 0.45, y0: h * 0.45, x1: w, y1: h }              // Bottom-Right
  ];

  const descriptor = new Float32Array(regions.length * 6);

  regions.forEach((r, rIdx) => {
    let sumR = 0, sumG = 0, sumB = 0;
    let count = 0;

    const startX = Math.floor(r.x0);
    const startY = Math.floor(r.y0);
    const endX = Math.floor(r.x1);
    const endY = Math.floor(r.y1);

    for (let y = startY; y < endY; y += 2) {
      const row = y * w;
      for (let x = startX; x < endX; x += 2) {
        const idx = (row + x) << 2;
        const red = data[idx];
        const green = data[idx + 1];
        const blue = data[idx + 2];

        // Filter out skin/finger pixels
        const cb = -0.168736 * red - 0.331264 * green + 0.5 * blue + 128;
        const cr =  0.5 * red - 0.418688 * green - 0.081312 * blue + 128;
        if (cb >= 77 && cb <= 127 && cr >= 133 && cr <= 173) {
          continue; // Skip holding fingers!
        }

        sumR += red;
        sumG += green;
        sumB += blue;
        count++;
      }
    }

    if (count === 0) count = 1;
    const meanR = sumR / count;
    const meanG = sumG / count;
    const meanB = sumB / count;

    let varR = 0, varG = 0, varB = 0;
    for (let y = startY; y < endY; y += 4) {
      const row = y * w;
      for (let x = startX; x < endX; x += 4) {
        const idx = (row + x) << 2;
        const dr = data[idx] - meanR;
        const dg = data[idx + 1] - meanG;
        const db = data[idx + 2] - meanB;
        varR += dr * dr;
        varG += dg * dg;
        varB += db * db;
      }
    }

    const stdR = Math.sqrt(varR / (count * 0.5 + 1));
    const stdG = Math.sqrt(varG / (count * 0.5 + 1));
    const stdB = Math.sqrt(varB / (count * 0.5 + 1));

    const offset = rIdx * 6;
    descriptor[offset + 0] = meanR / 255;
    descriptor[offset + 1] = meanG / 255;
    descriptor[offset + 2] = meanB / 255;
    descriptor[offset + 3] = stdR / 128;
    descriptor[offset + 4] = stdG / 128;
    descriptor[offset + 5] = stdB / 128;
  });

  return descriptor;
}

/**
 * Match a query vector against a database of registered objects using multi-shot k-NN
 * and calibrated thresholding.
 */
export function findBestMatch(querySample, database, options = {}) {
  const threshold = options.threshold ?? 0.70;
  const deepWeight = options.deepWeight ?? 0.75;
  const colorWeight = options.colorWeight ?? 0.25;

  if (!database || database.length === 0) {
    return null;
  }

  const matches = [];

  for (const item of database) {
    let maxItemScore = -1;
    let bestDeepSim = -1;
    let bestColorSim = -1;

    for (const shot of item.shots) {
      const deepSim = cosineSimilarity(querySample.deepVector, shot.deepVector);

      let colorSim = 0.8;
      if (querySample.colorVector && shot.colorVector) {
        colorSim = cosineSimilarity(querySample.colorVector, shot.colorVector);
        colorSim = Math.max(0, (colorSim + 1) * 0.5);
      }

      const hybridScore = (deepSim * deepWeight) + (colorSim * colorWeight);

      if (hybridScore > maxItemScore) {
        maxItemScore = hybridScore;
        bestDeepSim = deepSim;
        bestColorSim = colorSim;
      }
    }

    matches.push({
      item,
      score: maxItemScore,
      deepSim: bestDeepSim,
      colorSim: bestColorSim
    });
  }

  matches.sort((a, b) => b.score - a.score);
  const best = matches[0];

  if (!best || best.score < threshold) {
    return null;
  }

  const normPercent = Math.round(
    Math.min(99, Math.max(60, 60 + ((best.score - threshold) / (1.0 - threshold)) * 39))
  );

  return {
    item: best.item,
    confidence: normPercent,
    rawScore: best.score,
    deepSim: best.deepSim,
    colorSim: best.colorSim,
    runnerUp: matches[1] ? { item: matches[1].item, score: matches[1].score } : null
  };
}
