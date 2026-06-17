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
const BASE_AREA_PIXELS = 512 * 512;
const MIN_COMPONENT_AREA = 8;

type DirtyCleanResult = {
  labPixels: LabColor[];
  changedMask: Uint8Array;
  changedPixelCount: number;
  clusterCount: number;
};

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
  const dirtyResult =
    settings.processingMode === "palette"
      ? null
      : cleanDirtyBlocks(labPixels, activeMask, imageData.width, imageData.height, settings);
  const workingLabPixels = dirtyResult?.labPixels ?? labPixels;
  if (dirtyResult && settings.processingMode === "dirtyOnly") {
    const dirtyOutput = renderLabPixelsToImageData(
      workingLabPixels,
      source,
      activeMask,
      imageData.width,
      imageData.height,
    );
    onProgress?.({ phase: "rendering", percent: 100 });
    return {
      imageData: dirtyOutput,
      stats: {
        changedPixelCount: dirtyResult.changedPixelCount,
        clusterCount: dirtyResult.clusterCount,
        totalPixelCount,
      },
    };
  }

  const palette = fitPalette(workingLabPixels, settings.paletteSize, activeMask);
  onProgress?.({ phase: "clustering", percent: 55 });
  await yieldToBrowser();

  const labels = assignPaletteLabels(workingLabPixels, palette);
  const cleanLabels = cleanPaletteLabels(
    labels,
    palette,
    activeMask,
    imageData.width,
    imageData.height,
    settings,
  );
  const edges = computeEdgeStrength(workingLabPixels, imageData.width, imageData.height);
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
      const lab = workingLabPixels[pixelIndex];
      const target = palette[cleanLabels[pixelIndex]];
      const isDirtyIslandPixel = labels[pixelIndex] !== cleanLabels[pixelIndex];
      const nextLab = renderLabPixel(
        lab,
        target,
        edges[pixelIndex],
        isDirtyIslandPixel,
        settings.cleanupMode,
        strength,
        lumaStrength,
        chromaStrength,
        edgeProtect,
      );
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
  const dirtyResult =
    settings.processingMode === "palette"
      ? null
      : cleanDirtyBlocks(labPixels, activeMask, imageData.width, imageData.height, settings);
  const workingLabPixels = dirtyResult?.labPixels ?? labPixels;
  if (dirtyResult && settings.processingMode === "dirtyOnly") {
    return {
      imageData: renderLabPixelsToImageData(
        workingLabPixels,
        source,
        activeMask,
        imageData.width,
        imageData.height,
      ),
      stats: {
        changedPixelCount: dirtyResult.changedPixelCount,
        clusterCount: dirtyResult.clusterCount,
        totalPixelCount: labPixels.length,
      },
    };
  }

  const palette = fitPalette(workingLabPixels, settings.paletteSize, activeMask);
  const labels = assignPaletteLabels(workingLabPixels, palette);
  const cleanLabels = cleanPaletteLabels(
    labels,
    palette,
    activeMask,
    imageData.width,
    imageData.height,
    settings,
  );
  const edges = computeEdgeStrength(workingLabPixels, imageData.width, imageData.height);
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
    const lab = workingLabPixels[pixelIndex];
    const target = palette[cleanLabels[pixelIndex]];
    const isDirtyIslandPixel = labels[pixelIndex] !== cleanLabels[pixelIndex];
    const nextLab = renderLabPixel(
      lab,
      target,
      edges[pixelIndex],
      isDirtyIslandPixel,
      settings.cleanupMode,
      strength,
      lumaStrength,
      chromaStrength,
      edgeProtect,
    );
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

function cleanDirtyBlocks(
  labPixels: LabColor[],
  activeMask: Uint8Array,
  width: number,
  height: number,
  settings: ProcessingSettings,
): DirtyCleanResult {
  if (!settings.dirtyBlock.enabled) {
    return {
      labPixels,
      changedMask: new Uint8Array(labPixels.length),
      changedPixelCount: 0,
      clusterCount: 0,
    };
  }

  const dirtySettings = settings.dirtyBlock;
  const cleaned = labPixels.map((pixel) => ({ ...pixel }));
  const changedMask = new Uint8Array(labPixels.length);
  const analysisPalette = fitPalette(
    labPixels,
    dirtySettings.analysisPaletteSize,
    activeMask,
  );
  const labels = assignPaletteLabels(labPixels, analysisPalette);
  const paletteFamilies = analysisPalette.map(classifyLchFamily);
  const familyMap = new Int16Array(labels.length);
  for (let index = 0; index < labels.length; index += 1) {
    familyMap[index] = paletteFamilies[labels[index]];
  }

  const edges = computeEdgeStrength(labPixels, width, height);
  const maxDirtyArea = scaledArea(dirtySettings.maxDirtyArea, width, height, MIN_COMPONENT_AREA);
  const minSpeckArea = scaledArea(dirtySettings.minSpeckArea, width, height, 2);
  const connectivity = dirtySettings.connectivity;
  let changedPixelCount = 0;

  const markChanged = (component: number[], ring: number[], strengthScale = 1) => {
    if (!ring.length) {
      return;
    }
    const median = medianLab(ring.map((index) => cleaned[index]));
    const repair = clamp01((dirtySettings.repairStrength / 100) * strengthScale);
    for (const index of component) {
      const current = cleaned[index];
      cleaned[index] = {
        l: lerp(current.l, median.l, repair * 0.35),
        a: lerp(current.a, median.a, repair * 0.9),
        b: lerp(current.b, median.b, repair * 0.9),
      };
      if (!changedMask[index]) {
        changedMask[index] = 1;
        changedPixelCount += 1;
      }
    }
  };

  const familyVisited = new Uint8Array(familyMap.length);
  for (let index = 0; index < familyMap.length; index += 1) {
    if (!activeMask[index] || familyVisited[index]) {
      continue;
    }
    const family = familyMap[index];
    const component = collectValueComponent(
      familyMap,
      activeMask,
      familyVisited,
      width,
      height,
      index,
      family,
      connectivity,
    );
    if (component.length > maxDirtyArea && component.length > minSpeckArea) {
      continue;
    }
    if (isProtectedComponent(component, edges, dirtySettings.edgeProtect / 100)) {
      continue;
    }

    const ring = collectRing(component, activeMask, width, height, dirtySettings.surroundRadius);
    const vote = majorityRingValue(familyMap, ring);
    const isTinySpeck = component.length <= minSpeckArea;
    if (
      vote.value !== null &&
      vote.value !== family &&
      (isTinySpeck || vote.dominance >= dirtySettings.surroundDominance)
    ) {
      markChanged(component, ring, isTinySpeck ? 1 : 0.95);
    }
  }

  const labelVisited = new Uint8Array(labels.length);
  for (let index = 0; index < labels.length; index += 1) {
    if (!activeMask[index] || labelVisited[index]) {
      continue;
    }
    const label = labels[index];
    const component = collectValueComponent(
      labels,
      activeMask,
      labelVisited,
      width,
      height,
      index,
      label,
      connectivity,
    );
    if (component.length > maxDirtyArea && component.length > minSpeckArea) {
      continue;
    }
    if (isProtectedComponent(component, edges, dirtySettings.detailProtect / 100)) {
      continue;
    }

    const ring = collectRing(component, activeMask, width, height, dirtySettings.surroundRadius);
    if (!ring.length) {
      continue;
    }
    const ownFamily = familyMap[index];
    const familyVote = majorityRingValue(familyMap, ring);
    const componentMedian = medianLab(component.map((componentIndex) => cleaned[componentIndex]));
    const ringMedian = medianLab(ring.map((ringIndex) => cleaned[ringIndex]));
    const delta = Math.sqrt(labDistanceSquared(componentMedian, ringMedian));
    const isTinySpeck = component.length <= minSpeckArea;
    const isSameFamilyBlotch =
      familyVote.value === ownFamily &&
      delta >= dirtySettings.sameFamilyDeltaE &&
      familyVote.dominance >= Math.max(0, dirtySettings.surroundDominance - 0.08);
    const isCrossFamilyIsland =
      familyVote.value !== null &&
      familyVote.value !== ownFamily &&
      familyVote.dominance >= dirtySettings.surroundDominance;

    if (isTinySpeck || isSameFamilyBlotch || isCrossFamilyIsland) {
      markChanged(component, ring, isSameFamilyBlotch ? 0.85 : 1);
    }
  }

  return {
    labPixels: cleaned,
    changedMask,
    changedPixelCount,
    clusterCount: analysisPalette.length,
  };
}

function renderLabPixelsToImageData(
  labPixels: LabColor[],
  source: Uint8ClampedArray,
  activeMask: Uint8Array,
  width: number,
  height: number,
): ImageData {
  const output = new Uint8ClampedArray(source);
  for (let index = 0; index < labPixels.length; index += 1) {
    if (!activeMask[index]) {
      continue;
    }
    const offset = index * CHANNEL_COUNT;
    const rgb = labToRgb(labPixels[index]);
    output[offset] = rgb.r;
    output[offset + 1] = rgb.g;
    output[offset + 2] = rgb.b;
  }
  return new ImageData(output, width, height);
}

function classifyLchFamily(color: LabColor): number {
  const chroma = Math.sqrt(color.a ** 2 + color.b ** 2);
  if (color.l > 78 && chroma < 18) {
    return 1000;
  }
  if (chroma < 12) {
    return 1001;
  }
  const hue = (Math.atan2(color.b, color.a) * 180) / Math.PI;
  const normalizedHue = (hue + 360) % 360;
  return Math.floor(normalizedHue / 30);
}

function collectValueComponent(
  values: Uint16Array | Int16Array,
  activeMask: Uint8Array,
  visited: Uint8Array,
  width: number,
  height: number,
  startIndex: number,
  value: number,
  connectivity: 4 | 8,
): number[] {
  const component: number[] = [];
  const queue = [startIndex];
  visited[startIndex] = 1;
  const directions =
    connectivity === 4
      ? [
          [-1, 0],
          [1, 0],
          [0, -1],
          [0, 1],
        ]
      : [
          [-1, -1],
          [0, -1],
          [1, -1],
          [-1, 0],
          [1, 0],
          [-1, 1],
          [0, 1],
          [1, 1],
        ];

  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const index = queue[cursor];
    component.push(index);
    const x = index % width;
    const y = Math.floor(index / width);

    for (const [dx, dy] of directions) {
      const nx = x + dx;
      const ny = y + dy;
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
        continue;
      }
      const nextIndex = ny * width + nx;
      if (visited[nextIndex] || !activeMask[nextIndex] || values[nextIndex] !== value) {
        continue;
      }
      visited[nextIndex] = 1;
      queue.push(nextIndex);
    }
  }

  return component;
}

function collectRing(
  component: number[],
  activeMask: Uint8Array,
  width: number,
  height: number,
  radius: number,
): number[] {
  const componentMask = new Uint8Array(activeMask.length);
  const ringMask = new Uint8Array(activeMask.length);
  const ring: number[] = [];
  const r = Math.max(1, Math.round(radius));
  for (const index of component) {
    componentMask[index] = 1;
  }

  for (const index of component) {
    const x = index % width;
    const y = Math.floor(index / width);
    for (let dy = -r; dy <= r; dy += 1) {
      for (let dx = -r; dx <= r; dx += 1) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) > r || (dx === 0 && dy === 0)) {
          continue;
        }
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
          continue;
        }
        const nextIndex = ny * width + nx;
        if (!activeMask[nextIndex] || componentMask[nextIndex] || ringMask[nextIndex]) {
          continue;
        }
        ringMask[nextIndex] = 1;
        ring.push(nextIndex);
      }
    }
  }

  return ring;
}

function majorityRingValue(
  values: Uint16Array | Int16Array,
  ring: number[],
): { value: number | null; dominance: number } {
  const counts = new Map<number, number>();
  for (const index of ring) {
    const value = values[index];
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }
  let bestValue: number | null = null;
  let bestCount = 0;
  for (const [value, count] of counts) {
    if (count > bestCount) {
      bestValue = value;
      bestCount = count;
    }
  }
  return {
    value: bestValue,
    dominance: ring.length ? bestCount / ring.length : 0,
  };
}

function medianLab(values: LabColor[]): LabColor {
  if (!values.length) {
    return { l: 0, a: 0, b: 0 };
  }
  return {
    l: median(values.map((value) => value.l)),
    a: median(values.map((value) => value.a)),
    b: median(values.map((value) => value.b)),
  };
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2) {
    return sorted[middle];
  }
  return (sorted[middle - 1] + sorted[middle]) / 2;
}

function isProtectedComponent(component: number[], edges: Float32Array, threshold: number): boolean {
  if (!component.length) {
    return true;
  }
  let total = 0;
  for (const index of component) {
    total += edges[index];
  }
  return total / component.length > threshold;
}

function cleanPaletteLabels(
  labels: Uint16Array,
  palette: LabColor[],
  activeMask: Uint8Array,
  width: number,
  height: number,
  settings: ProcessingSettings,
): Uint16Array {
  if (!settings.dirtyClean) {
    return labels;
  }

  const speckArea = scaledArea(settings.speckArea, width, height, 2);
  const labelArea = scaledArea(settings.labelArea, width, height, MIN_COMPONENT_AREA);
  const familyArea = scaledArea(settings.familyArea, width, height, MIN_COMPONENT_AREA);
  const labelFamilies = palette.map(classifyColorFamily);
  const speckCleanLabels = cleanDirtyLabelIslands(
    labels,
    palette,
    activeMask,
    width,
    height,
    speckArea,
  );
  const familyCleanLabels = cleanDirtyFamilyIslands(
    speckCleanLabels,
    palette,
    labelFamilies,
    activeMask,
    width,
    height,
    familyArea,
  );
  const labelCleanLabels = cleanDirtyLabelIslands(
    familyCleanLabels,
    palette,
    activeMask,
    width,
    height,
    labelArea,
  );

  return smoothLabelMap(
    labelCleanLabels,
    activeMask,
    width,
    height,
    settings.labelFilterSize,
  );
}

function renderLabPixel(
  lab: LabColor,
  target: LabColor,
  edgeStrength: number,
  isDirtyIslandPixel: boolean,
  mode: ProcessingSettings["cleanupMode"],
  strength: number,
  lumaStrength: number,
  chromaStrength: number,
  edgeProtect: number,
): LabColor {
  if (mode === "flat") {
    return target;
  }

  const edgeFactor = isDirtyIslandPixel ? 1 : 1 - edgeProtect * edgeStrength;
  const lumaAlpha = isDirtyIslandPixel
    ? Math.max(strength * lumaStrength * edgeFactor, strength * 0.65)
    : strength * lumaStrength * edgeFactor;
  const chromaAlpha = isDirtyIslandPixel
    ? Math.max(strength * chromaStrength * edgeFactor, strength)
    : strength * chromaStrength * edgeFactor;

  return {
    l: lerp(lab.l, target.l, lumaAlpha),
    a: lerp(lab.a, target.a, chromaAlpha),
    b: lerp(lab.b, target.b, chromaAlpha),
  };
}

function scaledArea(baseArea: number, width: number, height: number, minimum: number): number {
  const scale = (width * height) / BASE_AREA_PIXELS;
  return Math.max(minimum, Math.round(Math.max(1, baseArea) * scale));
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
  palette: LabColor[],
  activeMask: Uint8Array,
  width: number,
  height: number,
  minArea: number,
): Uint16Array {
  const clean = new Uint16Array(labels);
  const visited = new Uint8Array(labels.length);

  for (let index = 0; index < clean.length; index += 1) {
    if (!activeMask[index] || visited[index]) {
      continue;
    }

    const label = clean[index];
    const component = collectComponent(clean, activeMask, visited, width, height, index, label);
    if (component.length >= minArea) {
      continue;
    }

    const replacement = majorityNeighborLabel(
      clean,
      palette,
      activeMask,
      component,
      width,
      height,
      label,
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
  palette: LabColor[],
  labelFamilies: ColorFamily[],
  activeMask: Uint8Array,
  width: number,
  height: number,
  minArea: number,
): Uint16Array {
  const clean = new Uint16Array(labels);
  const visited = new Uint8Array(labels.length);

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
      palette,
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
  palette: LabColor[],
  labelFamilies: ColorFamily[],
  activeMask: Uint8Array,
  component: number[],
  width: number,
  height: number,
  ownFamily: ColorFamily,
): number | null {
  const componentMask = new Uint8Array(labels.length);
  const counts = new Map<number, number>();
  const sourceColor = averageComponentColor(labels, palette, component);
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
  let bestScore = -1;
  for (const [label, count] of counts) {
    const score = neighborScore(count, sourceColor, palette[label]);
    if (score > bestScore) {
      bestLabel = label;
      bestScore = score;
    }
  }
  return bestLabel;
}

function averageComponentColor(
  labels: Uint16Array,
  palette: LabColor[],
  component: number[],
): LabColor {
  const sum = { l: 0, a: 0, b: 0 };
  for (const index of component) {
    const color = palette[labels[index]];
    sum.l += color.l;
    sum.a += color.a;
    sum.b += color.b;
  }
  const count = Math.max(1, component.length);
  return {
    l: sum.l / count,
    a: sum.a / count,
    b: sum.b / count,
  };
}

function neighborScore(contactCount: number, source: LabColor, candidate: LabColor): number {
  const distance = Math.sqrt(labDistanceSquared(source, candidate));
  return contactCount / (1 + distance / 20);
}

function smoothLabelMap(
  labels: Uint16Array,
  activeMask: Uint8Array,
  width: number,
  height: number,
  filterSize: 3 | 5,
): Uint16Array {
  const smooth = new Uint16Array(labels);
  const radius = filterSize === 5 ? 2 : 1;
  const minVotes = filterSize === 5 ? 13 : 5;

  for (let y = radius; y < height - radius; y += 1) {
    for (let x = radius; x < width - radius; x += 1) {
      const index = y * width + x;
      if (!activeMask[index]) {
        continue;
      }
      const counts = new Map<number, number>();
      for (let dy = -radius; dy <= radius; dy += 1) {
        for (let dx = -radius; dx <= radius; dx += 1) {
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
      if (bestLabel !== labels[index] && bestCount >= minVotes) {
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
  palette: LabColor[],
  activeMask: Uint8Array,
  component: number[],
  width: number,
  height: number,
  ownLabel: number,
): number | null {
  const componentMask = new Uint8Array(labels.length);
  const counts = new Map<number, number>();
  const sourceColor = palette[ownLabel];
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
  let bestScore = -1;
  for (const [label, count] of counts) {
    const score = neighborScore(count, sourceColor, palette[label]);
    if (score > bestScore) {
      bestLabel = label;
      bestScore = score;
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
