import { describe, expect, it } from "vitest";
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
});
