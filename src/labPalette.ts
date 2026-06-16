import type { ColorClusterResult, ProcessingProgress, ProcessingSettings } from "./types";

type LabColor = {
  l: number;
  a: number;
  b: number;
};

type ColorFamily = 0 | 1 | 2 | 3 | 4;

const CHANNEL_COUNT = 4;
const MAX_SAMPLES = 50_000;
const CHUNK_SIZE = 35_000;
const KMEANS_ITERATIONS = 18;
const BASE_DIRTY_ISLAND_AREA = 260;
const BASE_AREA_PIXELS = 512 * 512;

export async function stabilizeLabPaletteAsync(
  imageData: ImageData,
  settings: ProcessingSettings,
  onProgress?: (progress: ProcessingProgress) => void,
  signal?: AbortSignal,
): Promise<ColorClusterResult> {
  const source = imageData.data;
  const totalPixelCount = source.length / CHANNEL_COUNT;
  const labPixels = new Array<LabColor>(totalPixelCount);

  for (let pixelIndex = 0; pixelIndex < totalPixelCount; pixelIndex += 1) {
    throwIfAborted(signal);
    const offset = pixelIndex * CHANNEL_COUNT;
    labPixels[pixelIndex] = rgbToLab(source[offset], source[offset + 1], source[offset + 2]);

    if (pixelIndex % CHUNK_SIZE === 0) {
      onProgress?.({
        phase: "clustering",
        percent: Math.min(20, Math.round((pixelIndex / totalPixelCount) * 20)),
      });
      await yieldToBrowser();
    }
  }

  const activeMask = buildActiveMask(source, labPixels, imageData.width, imageData.height);
  const palette = fitPalette(labPixels, settings.paletteSize, activeMask);
  onProgress?.({ phase: "clustering", percent: 55 });
  await yieldToBrowser();

  const labels = assignPaletteLabels(labPixels, palette);
  const labelFamilies = palette.map(classifyColorFamily);
  const familyCleanLabels = cleanDirtyFamilyIslands(
    labels,
    labelFamilies,
    activeMask,
    imageData.width,
    imageData.height,
  );
  const cleanLabels = smoothLabelMap(
    cleanDirtyLabelIslands(familyCleanLabels, activeMask, imageData.width, imageData.height),
    activeMask,
    imageData.width,
    imageData.height,
  );
  const edges = computeEdgeStrength(labPixels, imageData.width, imageData.height);
  const output = new Uint8ClampedArray(source);
  let changedPixelCount = 0;
  const strength = clamp01(settings.strength / 100);
  const lumaStrength = clamp01(settings.lumaStrength / 100);
  const chromaStrength = clamp01(settings.chromaStrength / 100);
  const edgeProtect = clamp01(settings.edgeProtect / 100);

  for (let pixelIndex = 0; pixelIndex < totalPixelCount; pixelIndex += 1) {
    throwIfAborted(signal);
    const offset = pixelIndex * CHANNEL_COUNT;
    if (activeMask[pixelIndex]) {
      const lab = labPixels[pixelIndex];
      const target = palette[cleanLabels[pixelIndex]];
      const isDirtyIslandPixel = labels[pixelIndex] !== cleanLabels[pixelIndex];
      const edgeFactor = isDirtyIslandPixel ? 1 : 1 - edgeProtect * edges[pixelIndex];
      const lumaAlpha = isDirtyIslandPixel
        ? Math.max(strength * lumaStrength * edgeFactor, strength * 0.65)
        : strength * lumaStrength * edgeFactor;
      const chromaAlpha = isDirtyIslandPixel
        ? Math.max(strength * chromaStrength * edgeFactor, strength)
        : strength * chromaStrength * edgeFactor;
      const nextLab = {
        l: lerp(lab.l, target.l, lumaAlpha),
        a: lerp(lab.a, target.a, chromaAlpha),
        b: lerp(lab.b, target.b, chromaAlpha),
      };
      const nextRgb = labToRgb(nextLab);

      if (
        nextRgb.r !== source[offset] ||
        nextRgb.g !== source[offset + 1] ||
        nextRgb.b !== source[offset + 2]
      ) {
        changedPixelCount += 1;
      }

      output[offset] = nextRgb.r;
      output[offset + 1] = nextRgb.g;
      output[offset + 2] = nextRgb.b;
      output[offset + 3] = source[offset + 3];
    }

    if (pixelIndex % CHUNK_SIZE === 0) {
      onProgress?.({
        phase: "rendering",
        percent: 55 + Math.min(44, Math.round((pixelIndex / totalPixelCount) * 44)),
      });
      await yieldToBrowser();
    }
  }

  onProgress?.({ phase: "rendering", percent: 100 });
  return {
    imageData: new ImageData(output, imageData.width, imageData.height),
    stats: {
      changedPixelCount,
      clusterCount: palette.length,
      totalPixelCount,
    },
  };
}

export function stabilizeLabPalette(
  imageData: ImageData,
  settings: ProcessingSettings,
): ColorClusterResult {
  const source = imageData.data;
  const labPixels = new Array<LabColor>(source.length / CHANNEL_COUNT);

  for (let pixelIndex = 0; pixelIndex < labPixels.length; pixelIndex += 1) {
    const offset = pixelIndex * CHANNEL_COUNT;
    labPixels[pixelIndex] = rgbToLab(source[offset], source[offset + 1], source[offset + 2]);
  }

  const activeMask = buildActiveMask(source, labPixels, imageData.width, imageData.height);
  const palette = fitPalette(labPixels, settings.paletteSize, activeMask);
  const labels = assignPaletteLabels(labPixels, palette);
  const labelFamilies = palette.map(classifyColorFamily);
  const familyCleanLabels = cleanDirtyFamilyIslands(
    labels,
    labelFamilies,
    activeMask,
    imageData.width,
    imageData.height,
  );
  const cleanLabels = smoothLabelMap(
    cleanDirtyLabelIslands(familyCleanLabels, activeMask, imageData.width, imageData.height),
    activeMask,
    imageData.width,
    imageData.height,
  );
  const edges = computeEdgeStrength(labPixels, imageData.width, imageData.height);
  const output = new Uint8ClampedArray(source);
  const strength = clamp01(settings.strength / 100);
  const lumaStrength = clamp01(settings.lumaStrength / 100);
  const chromaStrength = clamp01(settings.chromaStrength / 100);
  const edgeProtect = clamp01(settings.edgeProtect / 100);
  let changedPixelCount = 0;

  for (let pixelIndex = 0; pixelIndex < labPixels.length; pixelIndex += 1) {
    const offset = pixelIndex * CHANNEL_COUNT;
    if (!activeMask[pixelIndex]) {
      continue;
    }
    const lab = labPixels[pixelIndex];
    const target = palette[cleanLabels[pixelIndex]];
    const isDirtyIslandPixel = labels[pixelIndex] !== cleanLabels[pixelIndex];
    const edgeFactor = isDirtyIslandPixel ? 1 : 1 - edgeProtect * edges[pixelIndex];
    const lumaAlpha = isDirtyIslandPixel
      ? Math.max(strength * lumaStrength * edgeFactor, strength * 0.65)
      : strength * lumaStrength * edgeFactor;
    const chromaAlpha = isDirtyIslandPixel
      ? Math.max(strength * chromaStrength * edgeFactor, strength)
      : strength * chromaStrength * edgeFactor;
    const nextLab = {
      l: lerp(lab.l, target.l, lumaAlpha),
      a: lerp(lab.a, target.a, chromaAlpha),
      b: lerp(lab.b, target.b, chromaAlpha),
    };
    const nextRgb = labToRgb(nextLab);

    if (
      nextRgb.r !== source[offset] ||
      nextRgb.g !== source[offset + 1] ||
      nextRgb.b !== source[offset + 2]
    ) {
      changedPixelCount += 1;
    }

    output[offset] = nextRgb.r;
    output[offset + 1] = nextRgb.g;
    output[offset + 2] = nextRgb.b;
  }

  return {
    imageData: new ImageData(output, imageData.width, imageData.height),
    stats: {
      changedPixelCount,
      clusterCount: palette.length,
      totalPixelCount: labPixels.length,
    },
  };
}

export function rgbToLab(r: number, g: number, b: number): LabColor {
  const [linearR, linearG, linearB] = [r, g, b].map((value) => srgbToLinear(value / 255));
  const x = (0.4124564 * linearR + 0.3575761 * linearG + 0.1804375 * linearB) / 0.95047;
  const y = 0.2126729 * linearR + 0.7151522 * linearG + 0.072175 * linearB;
  const z = (0.0193339 * linearR + 0.119192 * linearG + 0.9503041 * linearB) / 1.08883;
  const fx = labPivot(x);
  const fy = labPivot(y);
  const fz = labPivot(z);

  return {
    l: 116 * fy - 16,
    a: 500 * (fx - fy),
    b: 200 * (fy - fz),
  };
}

export function labToRgb(lab: LabColor): { r: number; g: number; b: number } {
  const fy = (lab.l + 16) / 116;
  const fx = fy + lab.a / 500;
  const fz = fy - lab.b / 200;
  const x = labPivotInverse(fx) * 0.95047;
  const y = labPivotInverse(fy);
  const z = labPivotInverse(fz) * 1.08883;
  const linearR = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
  const linearG = -0.969266 * x + 1.8760108 * y + 0.041556 * z;
  const linearB = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;

  return {
    r: clampByte(linearToSrgb(linearR) * 255),
    g: clampByte(linearToSrgb(linearG) * 255),
    b: clampByte(linearToSrgb(linearB) * 255),
  };
}

function fitPalette(labPixels: LabColor[], paletteSize: number, activeMask?: Uint8Array): LabColor[] {
  const samples = samplePixels(labPixels, MAX_SAMPLES, activeMask);
  const k = Math.max(1, Math.min(Math.round(paletteSize), samples.length));
  let centers = initializeCenters(samples, k);

  for (let iteration = 0; iteration < KMEANS_ITERATIONS; iteration += 1) {
    const sums = Array.from({ length: k }, () => ({ l: 0, a: 0, b: 0, count: 0 }));
    for (const pixel of samples) {
      const index = nearestPaletteIndex(pixel, centers);
      sums[index].l += pixel.l;
      sums[index].a += pixel.a;
      sums[index].b += pixel.b;
      sums[index].count += 1;
    }

    centers = centers.map((center, index) => {
      const sum = sums[index];
      if (!sum.count) {
        return center;
      }
      return {
        l: sum.l / sum.count,
        a: sum.a / sum.count,
        b: sum.b / sum.count,
      };
    });
  }

  return centers;
}

function initializeCenters(samples: LabColor[], k: number): LabColor[] {
  const centers = [samples[0]];
  while (centers.length < k) {
    let farthest = samples[0];
    let farthestDistance = -1;
    for (const sample of samples) {
      const distance = nearestPaletteDistance(sample, centers);
      if (distance > farthestDistance) {
        farthestDistance = distance;
        farthest = sample;
      }
    }
    centers.push(farthest);
  }
  return centers;
}

function samplePixels(labPixels: LabColor[], maxSamples: number, activeMask?: Uint8Array): LabColor[] {
  const source = activeMask ? labPixels.filter((_, index) => activeMask[index] === 1) : labPixels;
  if (!source.length) {
    return labPixels;
  }
  if (source.length <= maxSamples) {
    return source;
  }

  const step = source.length / maxSamples;
  const samples: LabColor[] = [];
  for (let index = 0; index < maxSamples; index += 1) {
    samples.push(source[Math.floor(index * step)]);
  }
  return samples;
}

function nearestPaletteColor(color: LabColor, palette: LabColor[]): LabColor {
  return palette[nearestPaletteIndex(color, palette)];
}

function nearestPaletteIndex(color: LabColor, palette: LabColor[]): number {
  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < palette.length; index += 1) {
    const distance = labDistanceSquared(color, palette[index]);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  }
  return bestIndex;
}

function nearestPaletteDistance(color: LabColor, palette: LabColor[]): number {
  return labDistanceSquared(color, nearestPaletteColor(color, palette));
}

function assignPaletteLabels(labPixels: LabColor[], palette: LabColor[]): Uint16Array {
  const labels = new Uint16Array(labPixels.length);
  for (let index = 0; index < labPixels.length; index += 1) {
    labels[index] = nearestPaletteIndex(labPixels[index], palette);
  }
  return labels;
}

function buildActiveMask(
  source: Uint8ClampedArray,
  labPixels: LabColor[],
  width: number,
  height: number,
): Uint8Array {
  const active = new Uint8Array(labPixels.length);
  const background = detectEdgeConnectedBackground(source, labPixels, width, height);
  for (let index = 0; index < labPixels.length; index += 1) {
    active[index] = source[index * CHANNEL_COUNT + 3] > 0 && !background[index] ? 1 : 0;
  }
  return active;
}

function detectEdgeConnectedBackground(
  source: Uint8ClampedArray,
  labPixels: LabColor[],
  width: number,
  height: number,
): Uint8Array {
  const background = new Uint8Array(labPixels.length);
  const queue: number[] = [];

  const enqueue = (index: number) => {
    if (background[index] || !isBackgroundLike(source, labPixels[index], index)) {
      return;
    }
    background[index] = 1;
    queue.push(index);
  };

  for (let x = 0; x < width; x += 1) {
    enqueue(x);
    enqueue((height - 1) * width + x);
  }
  for (let y = 1; y < height - 1; y += 1) {
    enqueue(y * width);
    enqueue(y * width + width - 1);
  }

  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const index = queue[cursor];
    const x = index % width;
    const y = Math.floor(index / width);
    if (x > 0) enqueue(index - 1);
    if (x + 1 < width) enqueue(index + 1);
    if (y > 0) enqueue(index - width);
    if (y + 1 < height) enqueue(index + width);
  }

  return background;
}

function isBackgroundLike(source: Uint8ClampedArray, lab: LabColor, index: number): boolean {
  const offset = index * CHANNEL_COUNT;
  if (source[offset + 3] === 0) {
    return true;
  }
  return lab.l > 92 && Math.abs(lab.a) < 5.5 && Math.abs(lab.b) < 7;
}

function cleanDirtyLabelIslands(
  labels: Uint16Array,
  activeMask: Uint8Array,
  width: number,
  height: number,
): Uint16Array {
  const clean = new Uint16Array(labels);
  const visited = new Uint8Array(labels.length);
  const scale = (width * height) / BASE_AREA_PIXELS;
  const minArea = Math.max(2, Math.round(BASE_DIRTY_ISLAND_AREA * scale));

  for (let index = 0; index < clean.length; index += 1) {
    if (!activeMask[index] || visited[index]) {
      continue;
    }

    const label = clean[index];
    const component = collectComponent(clean, activeMask, visited, width, height, index, label);
    if (component.length >= minArea) {
      continue;
    }

    const replacement = majorityNeighborLabel(clean, activeMask, component, width, height, label);
    if (replacement === null) {
      continue;
    }

    for (const pixelIndex of component) {
      clean[pixelIndex] = replacement;
    }
  }

  return clean;
}

function classifyColorFamily(color: LabColor): ColorFamily {
  const chroma = Math.sqrt(color.a ** 2 + color.b ** 2);
  if (color.l > 78 && chroma < 18) {
    return 3;
  }
  if (color.a < -7 && color.b > 0) {
    return 2;
  }
  if (color.a > 10 && color.b > -5) {
    return 1;
  }
  if (chroma < 14) {
    return 4;
  }
  return 0;
}

function cleanDirtyFamilyIslands(
  labels: Uint16Array,
  labelFamilies: ColorFamily[],
  activeMask: Uint8Array,
  width: number,
  height: number,
): Uint16Array {
  const clean = new Uint16Array(labels);
  const visited = new Uint8Array(labels.length);
  const scale = (width * height) / BASE_AREA_PIXELS;
  const minArea = Math.max(2, Math.round(BASE_DIRTY_ISLAND_AREA * scale));

  for (let index = 0; index < clean.length; index += 1) {
    if (!activeMask[index] || visited[index]) {
      continue;
    }

    const family = labelFamilies[clean[index]];
    const component = collectFamilyComponent(
      clean,
      labelFamilies,
      activeMask,
      visited,
      width,
      height,
      index,
      family,
    );
    if (component.length >= minArea) {
      continue;
    }

    const replacement = majorityNeighborLabelByFamily(
      clean,
      labelFamilies,
      activeMask,
      component,
      width,
      height,
      family,
    );
    if (replacement === null) {
      continue;
    }

    for (const pixelIndex of component) {
      clean[pixelIndex] = replacement;
    }
  }

  return clean;
}

function collectFamilyComponent(
  labels: Uint16Array,
  labelFamilies: ColorFamily[],
  activeMask: Uint8Array,
  visited: Uint8Array,
  width: number,
  height: number,
  startIndex: number,
  family: ColorFamily,
): number[] {
  const component: number[] = [];
  const queue = [startIndex];
  visited[startIndex] = 1;

  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const index = queue[cursor];
    component.push(index);
    const x = index % width;
    const y = Math.floor(index / width);

    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        if (dx === 0 && dy === 0) {
          continue;
        }
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
          continue;
        }
        const nextIndex = ny * width + nx;
        if (
          visited[nextIndex] ||
          !activeMask[nextIndex] ||
          labelFamilies[labels[nextIndex]] !== family
        ) {
          continue;
        }
        visited[nextIndex] = 1;
        queue.push(nextIndex);
      }
    }
  }

  return component;
}

function majorityNeighborLabelByFamily(
  labels: Uint16Array,
  labelFamilies: ColorFamily[],
  activeMask: Uint8Array,
  component: number[],
  width: number,
  height: number,
  ownFamily: ColorFamily,
): number | null {
  const componentMask = new Uint8Array(labels.length);
  const counts = new Map<number, number>();
  for (const index of component) {
    componentMask[index] = 1;
  }

  for (const index of component) {
    const x = index % width;
    const y = Math.floor(index / width);
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        if (dx === 0 && dy === 0) {
          continue;
        }
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
          continue;
        }
        const nextIndex = ny * width + nx;
        const neighborLabel = labels[nextIndex];
        if (
          !activeMask[nextIndex] ||
          componentMask[nextIndex] ||
          labelFamilies[neighborLabel] === ownFamily
        ) {
          continue;
        }
        counts.set(neighborLabel, (counts.get(neighborLabel) ?? 0) + 1);
      }
    }
  }

  let bestLabel: number | null = null;
  let bestCount = 0;
  for (const [label, count] of counts) {
    if (count > bestCount) {
      bestLabel = label;
      bestCount = count;
    }
  }
  return bestLabel;
}

function smoothLabelMap(
  labels: Uint16Array,
  activeMask: Uint8Array,
  width: number,
  height: number,
): Uint16Array {
  const smooth = new Uint16Array(labels);
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const index = y * width + x;
      if (!activeMask[index]) {
        continue;
      }
      const counts = new Map<number, number>();
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          const nextIndex = (y + dy) * width + x + dx;
          if (!activeMask[nextIndex]) {
            continue;
          }
          const label = labels[nextIndex];
          counts.set(label, (counts.get(label) ?? 0) + 1);
        }
      }

      let bestLabel = labels[index];
      let bestCount = 0;
      for (const [label, count] of counts) {
        if (count > bestCount) {
          bestLabel = label;
          bestCount = count;
        }
      }
      if (bestLabel !== labels[index] && bestCount >= 5) {
        smooth[index] = bestLabel;
      }
    }
  }
  return smooth;
}

function collectComponent(
  labels: Uint16Array,
  activeMask: Uint8Array,
  visited: Uint8Array,
  width: number,
  height: number,
  startIndex: number,
  label: number,
): number[] {
  const component: number[] = [];
  const queue = [startIndex];
  visited[startIndex] = 1;

  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const index = queue[cursor];
    component.push(index);
    const x = index % width;
    const y = Math.floor(index / width);

    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        if (dx === 0 && dy === 0) {
          continue;
        }
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
          continue;
        }
        const nextIndex = ny * width + nx;
        if (visited[nextIndex] || !activeMask[nextIndex] || labels[nextIndex] !== label) {
          continue;
        }
        visited[nextIndex] = 1;
        queue.push(nextIndex);
      }
    }
  }

  return component;
}

function majorityNeighborLabel(
  labels: Uint16Array,
  activeMask: Uint8Array,
  component: number[],
  width: number,
  height: number,
  ownLabel: number,
): number | null {
  const componentMask = new Uint8Array(labels.length);
  const counts = new Map<number, number>();
  for (const index of component) {
    componentMask[index] = 1;
  }

  for (const index of component) {
    const x = index % width;
    const y = Math.floor(index / width);
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        if (dx === 0 && dy === 0) {
          continue;
        }
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
          continue;
        }
        const nextIndex = ny * width + nx;
        const neighborLabel = labels[nextIndex];
        if (!activeMask[nextIndex] || componentMask[nextIndex] || neighborLabel === ownLabel) {
          continue;
        }
        counts.set(neighborLabel, (counts.get(neighborLabel) ?? 0) + 1);
      }
    }
  }

  let bestLabel: number | null = null;
  let bestCount = 0;
  for (const [label, count] of counts) {
    if (count > bestCount) {
      bestLabel = label;
      bestCount = count;
    }
  }
  return bestLabel;
}

function computeEdgeStrength(labPixels: LabColor[], width: number, height: number): Float32Array {
  const edges = new Float32Array(labPixels.length);
  let maxEdge = 0;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const left = labPixels[y * width + Math.max(0, x - 1)].l;
      const right = labPixels[y * width + Math.min(width - 1, x + 1)].l;
      const top = labPixels[Math.max(0, y - 1) * width + x].l;
      const bottom = labPixels[Math.min(height - 1, y + 1) * width + x].l;
      const edge = Math.sqrt((right - left) ** 2 + (bottom - top) ** 2);
      edges[y * width + x] = edge;
      maxEdge = Math.max(maxEdge, edge);
    }
  }

  if (maxEdge === 0) {
    return edges;
  }

  for (let index = 0; index < edges.length; index += 1) {
    edges[index] = Math.min(1, edges[index] / maxEdge);
  }
  return edges;
}

function labDistanceSquared(a: LabColor, b: LabColor): number {
  return (a.l - b.l) ** 2 + (a.a - b.a) ** 2 + (a.b - b.b) ** 2;
}

function srgbToLinear(value: number): number {
  return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
}

function linearToSrgb(value: number): number {
  const clipped = Math.max(0, Math.min(1, value));
  return clipped <= 0.0031308 ? clipped * 12.92 : 1.055 * clipped ** (1 / 2.4) - 0.055;
}

function labPivot(value: number): number {
  const epsilon = 216 / 24389;
  const kappa = 24389 / 27;
  return value > epsilon ? Math.cbrt(value) : (kappa * value + 16) / 116;
}

function labPivotInverse(value: number): number {
  const epsilon = 216 / 24389;
  const kappa = 24389 / 27;
  const cubed = value ** 3;
  return cubed > epsilon ? cubed : (116 * value - 16) / kappa;
}

function lerp(source: number, target: number, alpha: number): number {
  return source * (1 - alpha) + target * alpha;
}

function clampByte(value: number): number {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function yieldToBrowser(): Promise<void> {
  return new Promise((resolve) => globalThis.setTimeout(resolve, 0));
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new DOMException("Processing aborted.", "AbortError");
  }
}
