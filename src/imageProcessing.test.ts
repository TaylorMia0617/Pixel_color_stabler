import { describe, expect, it } from "vitest";
import { DEFAULT_SETTINGS } from "./appConfig";
import { clusterSimilarColors, rgbDistance, stabilizeLocalColors } from "./imageProcessing";
import { labToRgb, rgbToLab, stabilizeLabPalette } from "./labPalette";

class TestImageData {
  data: Uint8ClampedArray;
  width: number;
  height: number;

  constructor(data: Uint8ClampedArray, width: number, height: number) {
    this.data = data;
    this.width = width;
    this.height = height;
  }
}

globalThis.ImageData = TestImageData as unknown as typeof ImageData;

function imageData(pixels: number[][]): ImageData {
  return new ImageData(new Uint8ClampedArray(pixels.flat()), pixels.length, 1);
}

describe("image processing", () => {
  it("measures RGB distance", () => {
    expect(rgbDistance({ r: 0, g: 0, b: 0 }, { r: 3, g: 4, b: 0 })).toBe(5);
  });

  it("merges near-identical colors into their representative color", () => {
    const result = clusterSimilarColors(
      imageData([
        [100, 100, 100, 255],
        [104, 100, 100, 255],
        [250, 0, 0, 255],
      ]),
      8,
      100,
    );

    expect(Array.from(result.imageData.data)).toEqual([
      102, 100, 100, 255,
      102, 100, 100, 255,
      250, 0, 0, 255,
    ]);
    expect(result.stats.clusterCount).toBe(2);
    expect(result.stats.changedPixelCount).toBe(2);
  });

  it("keeps pixels unchanged when tolerance is zero", () => {
    const source = imageData([
      [10, 20, 30, 255],
      [11, 20, 30, 255],
    ]);
    const result = clusterSimilarColors(source, 0, 100);

    expect(Array.from(result.imageData.data)).toEqual(Array.from(source.data));
    expect(result.stats.clusterCount).toBe(2);
    expect(result.stats.changedPixelCount).toBe(0);
  });

  it("preserves alpha values", () => {
    const result = clusterSimilarColors(
      imageData([
        [50, 50, 50, 20],
        [54, 50, 50, 200],
        [200, 200, 200, 0],
      ]),
      8,
      100,
    );

    expect(result.imageData.data[3]).toBe(20);
    expect(result.imageData.data[7]).toBe(200);
    expect(result.imageData.data[11]).toBe(0);
  });

  it("uses strength to blend toward the representative color", () => {
    const result = clusterSimilarColors(
      imageData([
        [100, 100, 100, 255],
        [104, 100, 100, 255],
      ]),
      8,
      50,
    );

    expect(Array.from(result.imageData.data)).toEqual([
      101, 100, 100, 255,
      103, 100, 100, 255,
    ]);
  });

  it("locally softens isolated color noise without crossing unrelated colors", () => {
    const result = stabilizeLocalColors(
      new ImageData(
        new Uint8ClampedArray([
          100, 100, 100, 255,
          101, 100, 100, 255,
          100, 101, 100, 255,
          102, 100, 100, 255,
          110, 100, 100, 255,
          250, 0, 0, 255,
          100, 100, 101, 255,
          101, 101, 100, 255,
          100, 100, 102, 255,
        ]),
        3,
        3,
      ),
      16,
      100,
    );

    expect(result.imageData.data[16]).toBeLessThan(110);
    expect(result.imageData.data[20]).toBe(250);
  });

  it("round-trips RGB through Lab with small channel drift", () => {
    const source = { r: 40, g: 120, b: 200 };
    const next = labToRgb(rgbToLab(source.r, source.g, source.b));

    expect(Math.abs(next.r - source.r)).toBeLessThanOrEqual(1);
    expect(Math.abs(next.g - source.g)).toBeLessThanOrEqual(1);
    expect(Math.abs(next.b - source.b)).toBeLessThanOrEqual(1);
  });

  it("uses Lab palette compression as the main denoise path", () => {
    const result = stabilizeLabPalette(
      new ImageData(
        new Uint8ClampedArray([
          40, 120, 200, 255,
          42, 122, 202, 255,
          220, 80, 30, 255,
          221, 82, 29, 255,
        ]),
        2,
        2,
      ),
      {
        ...DEFAULT_SETTINGS,
        strength: 100,
        paletteSize: 2,
        lumaStrength: 30,
        chromaStrength: 90,
        edgeProtect: 0,
      },
    );

    expect(result.stats.clusterCount).toBe(2);
    expect(result.stats.changedPixelCount).toBeGreaterThan(0);
    expect(result.imageData.data[3]).toBe(255);
  });

  it("repaints small dirty palette islands with the surrounding label", () => {
    const pixels: number[] = [];
    for (let index = 0; index < 25; index += 1) {
      const isCenter = index === 12;
      pixels.push(...(isCenter ? [30, 180, 30, 255] : [220, 40, 35, 255]));
    }

    const result = stabilizeLabPalette(
      new ImageData(new Uint8ClampedArray(pixels), 5, 5),
      {
        ...DEFAULT_SETTINGS,
        strength: 100,
        paletteSize: 2,
        lumaStrength: 30,
        chromaStrength: 90,
        edgeProtect: 100,
      },
    );

    const centerOffset = 12 * 4;
    expect(result.imageData.data[centerOffset]).toBeGreaterThan(150);
    expect(result.imageData.data[centerOffset + 1]).toBeLessThan(100);
  });

  it("keeps dirty cleanup aggressive when palette size is high", () => {
    const pixels: number[] = [];
    for (let index = 0; index < 49; index += 1) {
      const isDirtyPatch = index === 24 || index === 25;
      const shade = 35 + (index % 4) * 4;
      pixels.push(...(isDirtyPatch ? [40, 175, 35, 255] : [215 + (index % 3) * 5, shade, 35, 255]));
    }

    const result = stabilizeLabPalette(
      new ImageData(new Uint8ClampedArray(pixels), 7, 7),
      {
        ...DEFAULT_SETTINGS,
        strength: 100,
        paletteSize: 12,
        lumaStrength: 70,
        chromaStrength: 100,
        edgeProtect: 80,
      },
    );

    const centerOffset = 24 * 4;
    expect(result.imageData.data[centerOffset]).toBeGreaterThan(150);
    expect(result.imageData.data[centerOffset + 1]).toBeLessThan(110);
  });

  it("repaints red islands inside a green subject", () => {
    const pixels: number[] = [];
    for (let index = 0; index < 25; index += 1) {
      const isCenter = index === 12;
      pixels.push(...(isCenter ? [220, 45, 35, 255] : [35, 155, 45, 255]));
    }

    const result = stabilizeLabPalette(
      new ImageData(new Uint8ClampedArray(pixels), 5, 5),
      {
        ...DEFAULT_SETTINGS,
        strength: 100,
        paletteSize: 2,
        chromaStrength: 100,
        edgeProtect: 100,
      },
    );

    const centerOffset = 12 * 4;
    expect(result.imageData.data[centerOffset]).toBeLessThan(100);
    expect(result.imageData.data[centerOffset + 1]).toBeGreaterThan(100);
  });

  it("dirty-only mode repairs a green island without palette-compressing the whole image", () => {
    const pixels: number[] = [];
    for (let index = 0; index < 25; index += 1) {
      const isCenter = index === 12;
      const redShade = 210 + (index % 3) * 4;
      pixels.push(...(isCenter ? [35, 180, 35, 255] : [redShade, 45, 38, 255]));
    }

    const result = stabilizeLabPalette(
      new ImageData(new Uint8ClampedArray(pixels), 5, 5),
      {
        ...DEFAULT_SETTINGS,
        processingMode: "dirtyOnly",
        strength: 0,
        paletteSize: 12,
        dirtyBlock: {
          ...DEFAULT_SETTINGS.dirtyBlock,
          maxDirtyArea: 80,
          repairStrength: 100,
        },
      },
    );

    const centerOffset = 12 * 4;
    expect(result.imageData.data[centerOffset]).toBeGreaterThan(150);
    expect(result.imageData.data[centerOffset + 1]).toBeLessThan(100);
    expect(Math.abs(result.imageData.data[0] - pixels[0])).toBeLessThanOrEqual(6);
  });

  it("dirty-only mode repairs a red island inside a green subject", () => {
    const pixels: number[] = [];
    for (let index = 0; index < 25; index += 1) {
      const isCenter = index === 12;
      pixels.push(...(isCenter ? [225, 45, 35, 255] : [38, 155 + (index % 3) * 3, 45, 255]));
    }

    const result = stabilizeLabPalette(
      new ImageData(new Uint8ClampedArray(pixels), 5, 5),
      {
        ...DEFAULT_SETTINGS,
        processingMode: "dirtyOnly",
        strength: 0,
        paletteSize: 12,
        dirtyBlock: {
          ...DEFAULT_SETTINGS.dirtyBlock,
          maxDirtyArea: 80,
          repairStrength: 100,
        },
      },
    );

    const centerOffset = 12 * 4;
    expect(result.imageData.data[centerOffset]).toBeLessThan(100);
    expect(result.imageData.data[centerOffset + 1]).toBeGreaterThan(110);
  });

  it("uses flat mode to repaint active pixels directly to palette colors", () => {
    const result = stabilizeLabPalette(
      new ImageData(
        new Uint8ClampedArray([
          220, 50, 40, 255,
          222, 52, 42, 255,
          40, 150, 45, 255,
          42, 152, 47, 255,
        ]),
        2,
        2,
      ),
      {
        ...DEFAULT_SETTINGS,
        cleanupMode: "flat",
        strength: 10,
        paletteSize: 2,
        lumaStrength: 0,
        chromaStrength: 10,
      },
    );

    expect(result.stats.clusterCount).toBe(2);
    expect(result.imageData.data[0]).not.toBe(220);
  });
});
