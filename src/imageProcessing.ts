import type {
  ColorClusterResult,
  ImageDocument,
  ProcessingProgress,
  ProcessingSettings,
  RgbColor,
} from "./types";

type Cluster = {
  r: number;
  g: number;
  b: number;
  count: number;
};

const CHANNEL_COUNT = 4;
const DEFAULT_CHUNK_SIZE = 35_000;
const LOCAL_RADIUS = 2;

export function rgbDistance(a: RgbColor, b: RgbColor): number {
  const dr = a.r - b.r;
  const dg = a.g - b.g;
  const db = a.b - b.b;
  return Math.sqrt(dr * dr + dg * dg + db * db);
}

export async function loadImageToDocument(file: File): Promise<ImageDocument> {
  const bitmap = await createImageBitmap(file);
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;

  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    throw new Error("Could not create canvas context.");
  }

  context.drawImage(bitmap, 0, 0);
  const original = context.getImageData(0, 0, bitmap.width, bitmap.height);
  bitmap.close();

  return {
    name: file.name,
    width: original.width,
    height: original.height,
    original,
  };
}

export function stabilizeImage(
  document: ImageDocument,
  settings: ProcessingSettings,
): ColorClusterResult {
  return stabilizeLocalColors(document.original, settings.tolerance, settings.strength);
}

export async function stabilizeImageAsync(
  document: ImageDocument,
  settings: ProcessingSettings,
  onProgress?: (progress: ProcessingProgress) => void,
  signal?: AbortSignal,
): Promise<ColorClusterResult> {
  return stabilizeLocalColorsAsync(
    document.original,
    settings.tolerance,
    settings.strength,
    onProgress,
    signal,
  );
}

export function stabilizeLocalColors(
  imageData: ImageData,
  tolerance: number,
  strength = 100,
): ColorClusterResult {
  const normalizedTolerance = Math.max(0, tolerance);
  const normalizedStrength = Math.max(0, Math.min(100, strength)) / 100;
  const source = imageData.data;
  const output = new Uint8ClampedArray(source);
  let changedPixelCount = 0;

  for (let pixelIndex = 0; pixelIndex < source.length / CHANNEL_COUNT; pixelIndex += 1) {
    const offset = pixelIndex * CHANNEL_COUNT;
    const next = stabilizePixel(source, imageData.width, imageData.height, pixelIndex, normalizedTolerance);

    const nextR = blend(source[offset], next.r, normalizedStrength);
    const nextG = blend(source[offset + 1], next.g, normalizedStrength);
    const nextB = blend(source[offset + 2], next.b, normalizedStrength);

    if (nextR !== source[offset] || nextG !== source[offset + 1] || nextB !== source[offset + 2]) {
      changedPixelCount += 1;
    }

    output[offset] = nextR;
    output[offset + 1] = nextG;
    output[offset + 2] = nextB;
    output[offset + 3] = source[offset + 3];
  }

  return {
    imageData: new ImageData(output, imageData.width, imageData.height),
    stats: {
      changedPixelCount,
      clusterCount: countApproximateColors(output),
      totalPixelCount: source.length / CHANNEL_COUNT,
    },
  };
}

export async function stabilizeLocalColorsAsync(
  imageData: ImageData,
  tolerance: number,
  strength = 100,
  onProgress?: (progress: ProcessingProgress) => void,
  signal?: AbortSignal,
): Promise<ColorClusterResult> {
  const normalizedTolerance = Math.max(0, tolerance);
  const normalizedStrength = Math.max(0, Math.min(100, strength)) / 100;
  const source = imageData.data;
  const output = new Uint8ClampedArray(source);
  const totalPixelCount = source.length / CHANNEL_COUNT;
  let changedPixelCount = 0;

  for (let pixelIndex = 0; pixelIndex < totalPixelCount; pixelIndex += 1) {
    throwIfAborted(signal);
    const offset = pixelIndex * CHANNEL_COUNT;
    const next = stabilizePixel(source, imageData.width, imageData.height, pixelIndex, normalizedTolerance);

    const nextR = blend(source[offset], next.r, normalizedStrength);
    const nextG = blend(source[offset + 1], next.g, normalizedStrength);
    const nextB = blend(source[offset + 2], next.b, normalizedStrength);

    if (nextR !== source[offset] || nextG !== source[offset + 1] || nextB !== source[offset + 2]) {
      changedPixelCount += 1;
    }

    output[offset] = nextR;
    output[offset + 1] = nextG;
    output[offset + 2] = nextB;
    output[offset + 3] = source[offset + 3];

    if (pixelIndex % DEFAULT_CHUNK_SIZE === 0) {
      onProgress?.({
        phase: "clustering",
        percent: Math.min(92, Math.round((pixelIndex / totalPixelCount) * 92)),
      });
      await yieldToBrowser();
    }
  }

  onProgress?.({ phase: "rendering", percent: 96 });
  await yieldToBrowser();

  const result = {
    imageData: new ImageData(output, imageData.width, imageData.height),
    stats: {
      changedPixelCount,
      clusterCount: countApproximateColors(output),
      totalPixelCount,
    },
  };

  onProgress?.({ phase: "rendering", percent: 100 });
  return result;
}

export function clusterSimilarColors(
  imageData: ImageData,
  tolerance: number,
  strength = 100,
): ColorClusterResult {
  const normalizedTolerance = Math.max(0, tolerance);
  const normalizedStrength = Math.max(0, Math.min(100, strength)) / 100;
  const source = imageData.data;
  const output = new Uint8ClampedArray(source);
  const assignments = new Int32Array(source.length / CHANNEL_COUNT);
  const clusters: Cluster[] = [];
  const buckets = new Map<string, number[]>();
  const bucketSize = Math.max(1, normalizedTolerance || 1);

  for (let pixelIndex = 0; pixelIndex < assignments.length; pixelIndex += 1) {
    const offset = pixelIndex * CHANNEL_COUNT;
    const alpha = source[offset + 3];

    if (alpha === 0) {
      assignments[pixelIndex] = -1;
      continue;
    }

    const color = {
      r: source[offset],
      g: source[offset + 1],
      b: source[offset + 2],
    };

    const clusterIndex = findOrCreateCluster(
      color,
      clusters,
      buckets,
      bucketSize,
      normalizedTolerance,
    );
    assignments[pixelIndex] = clusterIndex;
  }

  let changedPixelCount = 0;

  for (let pixelIndex = 0; pixelIndex < assignments.length; pixelIndex += 1) {
    const clusterIndex = assignments[pixelIndex];
    if (clusterIndex < 0) {
      continue;
    }

    const offset = pixelIndex * CHANNEL_COUNT;
    const cluster = clusters[clusterIndex];
    const representative = {
      r: Math.round(cluster.r / cluster.count),
      g: Math.round(cluster.g / cluster.count),
      b: Math.round(cluster.b / cluster.count),
    };

    const nextR = blend(source[offset], representative.r, normalizedStrength);
    const nextG = blend(source[offset + 1], representative.g, normalizedStrength);
    const nextB = blend(source[offset + 2], representative.b, normalizedStrength);

    if (nextR !== source[offset] || nextG !== source[offset + 1] || nextB !== source[offset + 2]) {
      changedPixelCount += 1;
    }

    output[offset] = nextR;
    output[offset + 1] = nextG;
    output[offset + 2] = nextB;
    output[offset + 3] = source[offset + 3];
  }

  return {
    imageData: new ImageData(output, imageData.width, imageData.height),
    stats: {
      changedPixelCount,
      clusterCount: clusters.length,
      totalPixelCount: assignments.length,
    },
  };
}

export async function clusterSimilarColorsAsync(
  imageData: ImageData,
  tolerance: number,
  strength = 100,
  onProgress?: (progress: ProcessingProgress) => void,
  signal?: AbortSignal,
): Promise<ColorClusterResult> {
  const normalizedTolerance = Math.max(0, tolerance);
  const normalizedStrength = Math.max(0, Math.min(100, strength)) / 100;
  const source = imageData.data;
  const output = new Uint8ClampedArray(source);
  const assignments = new Int32Array(source.length / CHANNEL_COUNT);
  const clusters: Cluster[] = [];
  const buckets = new Map<string, number[]>();
  const bucketSize = Math.max(1, normalizedTolerance || 1);

  for (let pixelIndex = 0; pixelIndex < assignments.length; pixelIndex += 1) {
    throwIfAborted(signal);
    const offset = pixelIndex * CHANNEL_COUNT;
    const alpha = source[offset + 3];

    if (alpha === 0) {
      assignments[pixelIndex] = -1;
    } else {
      const color = {
        r: source[offset],
        g: source[offset + 1],
        b: source[offset + 2],
      };

      assignments[pixelIndex] = findOrCreateCluster(
        color,
        clusters,
        buckets,
        bucketSize,
        normalizedTolerance,
      );
    }

    if (pixelIndex % DEFAULT_CHUNK_SIZE === 0) {
      onProgress?.({
        phase: "clustering",
        percent: Math.min(50, Math.round((pixelIndex / assignments.length) * 50)),
      });
      await yieldToBrowser();
    }
  }

  let changedPixelCount = 0;

  for (let pixelIndex = 0; pixelIndex < assignments.length; pixelIndex += 1) {
    throwIfAborted(signal);
    const clusterIndex = assignments[pixelIndex];
    if (clusterIndex >= 0) {
      const offset = pixelIndex * CHANNEL_COUNT;
      const cluster = clusters[clusterIndex];
      const representative = {
        r: Math.round(cluster.r / cluster.count),
        g: Math.round(cluster.g / cluster.count),
        b: Math.round(cluster.b / cluster.count),
      };

      const nextR = blend(source[offset], representative.r, normalizedStrength);
      const nextG = blend(source[offset + 1], representative.g, normalizedStrength);
      const nextB = blend(source[offset + 2], representative.b, normalizedStrength);

      if (
        nextR !== source[offset] ||
        nextG !== source[offset + 1] ||
        nextB !== source[offset + 2]
      ) {
        changedPixelCount += 1;
      }

      output[offset] = nextR;
      output[offset + 1] = nextG;
      output[offset + 2] = nextB;
      output[offset + 3] = source[offset + 3];
    }

    if (pixelIndex % DEFAULT_CHUNK_SIZE === 0) {
      onProgress?.({
        phase: "rendering",
        percent: 50 + Math.min(50, Math.round((pixelIndex / assignments.length) * 50)),
      });
      await yieldToBrowser();
    }
  }

  onProgress?.({ phase: "rendering", percent: 100 });

  return {
    imageData: new ImageData(output, imageData.width, imageData.height),
    stats: {
      changedPixelCount,
      clusterCount: clusters.length,
      totalPixelCount: assignments.length,
    },
  };
}

export async function exportPng(imageData: ImageData): Promise<Blob> {
  const canvas = document.createElement("canvas");
  canvas.width = imageData.width;
  canvas.height = imageData.height;

  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Could not create canvas context.");
  }

  context.putImageData(imageData, 0, 0);

  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!blob) {
    throw new Error("Could not export PNG.");
  }

  return blob;
}

function findOrCreateCluster(
  color: RgbColor,
  clusters: Cluster[],
  buckets: Map<string, number[]>,
  bucketSize: number,
  tolerance: number,
): number {
  const bucket = bucketFor(color, bucketSize);
  let bestIndex = -1;
  let bestDistance = Number.POSITIVE_INFINITY;

  for (const candidateIndex of nearbyClusterIndexes(bucket, buckets)) {
    const candidate = clusters[candidateIndex];
    const candidateColor = {
      r: candidate.r / candidate.count,
      g: candidate.g / candidate.count,
      b: candidate.b / candidate.count,
    };
    const distance = rgbDistance(color, candidateColor);

    if (distance <= tolerance && distance < bestDistance) {
      bestDistance = distance;
      bestIndex = candidateIndex;
    }
  }

  if (bestIndex >= 0) {
    const cluster = clusters[bestIndex];
    cluster.r += color.r;
    cluster.g += color.g;
    cluster.b += color.b;
    cluster.count += 1;
    return bestIndex;
  }

  const newIndex = clusters.length;
  clusters.push({ ...color, count: 1 });
  const key = bucketKey(bucket);
  const bucketIndexes = buckets.get(key) ?? [];
  bucketIndexes.push(newIndex);
  buckets.set(key, bucketIndexes);
  return newIndex;
}

function* nearbyClusterIndexes(
  bucket: [number, number, number],
  buckets: Map<string, number[]>,
): Generator<number> {
  for (let r = bucket[0] - 1; r <= bucket[0] + 1; r += 1) {
    for (let g = bucket[1] - 1; g <= bucket[1] + 1; g += 1) {
      for (let b = bucket[2] - 1; b <= bucket[2] + 1; b += 1) {
        const indexes = buckets.get(bucketKey([r, g, b]));
        if (indexes) {
          yield* indexes;
        }
      }
    }
  }
}

function bucketFor(color: RgbColor, bucketSize: number): [number, number, number] {
  return [
    Math.floor(color.r / bucketSize),
    Math.floor(color.g / bucketSize),
    Math.floor(color.b / bucketSize),
  ];
}

function bucketKey(bucket: [number, number, number]): string {
  return `${bucket[0]}:${bucket[1]}:${bucket[2]}`;
}

function blend(source: number, target: number, strength: number): number {
  return Math.round(source * (1 - strength) + target * strength);
}

function stabilizePixel(
  source: Uint8ClampedArray,
  width: number,
  height: number,
  pixelIndex: number,
  tolerance: number,
): RgbColor {
  const offset = pixelIndex * CHANNEL_COUNT;
  const alpha = source[offset + 3];
  if (alpha === 0 || tolerance === 0) {
    return {
      r: source[offset],
      g: source[offset + 1],
      b: source[offset + 2],
    };
  }

  const x = pixelIndex % width;
  const y = Math.floor(pixelIndex / width);
  const base = {
    r: source[offset],
    g: source[offset + 1],
    b: source[offset + 2],
  };
  let totalR = 0;
  let totalG = 0;
  let totalB = 0;
  let totalWeight = 0;
  const groups = new Map<
    string,
    {
      r: number;
      g: number;
      b: number;
      count: number;
      distance: number;
    }
  >();
  const quantizeStep = Math.max(4, Math.round(tolerance / 2));

  for (
    let nextY = Math.max(0, y - LOCAL_RADIUS);
    nextY <= Math.min(height - 1, y + LOCAL_RADIUS);
    nextY += 1
  ) {
    for (
      let nextX = Math.max(0, x - LOCAL_RADIUS);
      nextX <= Math.min(width - 1, x + LOCAL_RADIUS);
      nextX += 1
    ) {
      const nextOffset = (nextY * width + nextX) * CHANNEL_COUNT;
      const nextAlpha = source[nextOffset + 3];
      if (nextAlpha === 0 || Math.abs(nextAlpha - alpha) > 24) {
        continue;
      }

      const neighbor = {
        r: source[nextOffset],
        g: source[nextOffset + 1],
        b: source[nextOffset + 2],
      };
      const distance = rgbDistance(base, neighbor);
      if (distance > tolerance * 1.6) {
        continue;
      }

      const isCenter = nextX === x && nextY === y;
      const weight = isCenter ? 2 : 1 - distance / Math.max(1, tolerance + 1);
      totalR += neighbor.r * weight;
      totalG += neighbor.g * weight;
      totalB += neighbor.b * weight;
      totalWeight += weight;

      const key = quantizedColorKey(neighbor, quantizeStep);
      const group = groups.get(key) ?? {
        r: 0,
        g: 0,
        b: 0,
        count: 0,
        distance: 0,
      };
      group.r += neighbor.r;
      group.g += neighbor.g;
      group.b += neighbor.b;
      group.count += 1;
      group.distance += distance;
      groups.set(key, group);
    }
  }

  if (totalWeight === 0 || groups.size === 0) {
    return base;
  }

  const localAverage = {
    r: Math.round(totalR / totalWeight),
    g: Math.round(totalG / totalWeight),
    b: Math.round(totalB / totalWeight),
  };
  const dominant = dominantLocalGroup(groups);
  const localSampleCount = Array.from(groups.values()).reduce((total, group) => total + group.count, 0);

  if (!dominant || dominant.count < 3) {
    return localAverage;
  }

  const dominantColor = {
    r: Math.round(dominant.r / dominant.count),
    g: Math.round(dominant.g / dominant.count),
    b: Math.round(dominant.b / dominant.count),
  };
  const dominance = dominant.count / Math.max(1, localSampleCount);
  const snapStrength = dominance >= 0.45 ? 0.9 : 0.72;

  return {
    r: blend(localAverage.r, dominantColor.r, snapStrength),
    g: blend(localAverage.g, dominantColor.g, snapStrength),
    b: blend(localAverage.b, dominantColor.b, snapStrength),
  };
}

function countApproximateColors(data: Uint8ClampedArray): number {
  const colors = new Set<string>();

  for (let offset = 0; offset < data.length; offset += CHANNEL_COUNT) {
    if (data[offset + 3] === 0) {
      continue;
    }

    colors.add(
      `${Math.round(data[offset] / 8)}:${Math.round(data[offset + 1] / 8)}:${Math.round(
        data[offset + 2] / 8,
      )}`,
    );
  }

  return colors.size;
}

function quantizedColorKey(color: RgbColor, step: number): string {
  return `${Math.round(color.r / step)}:${Math.round(color.g / step)}:${Math.round(
    color.b / step,
  )}`;
}

function dominantLocalGroup(
  groups: Map<
    string,
    {
      r: number;
      g: number;
      b: number;
      count: number;
      distance: number;
    }
  >,
) {
  let best:
    | {
        r: number;
        g: number;
        b: number;
        count: number;
        distance: number;
      }
    | null = null;

  for (const group of groups.values()) {
    if (
      !best ||
      group.count > best.count ||
      (group.count === best.count && group.distance < best.distance)
    ) {
      best = group;
    }
  }

  return best;
}

function yieldToBrowser(): Promise<void> {
  return new Promise((resolve) => globalThis.setTimeout(resolve, 0));
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new DOMException("Processing aborted.", "AbortError");
  }
}
