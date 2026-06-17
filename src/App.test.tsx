import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { App } from "./App";

describe("App settings controls", () => {
  it("allows editing processing defaults before an image is imported", () => {
    render(<App />);

    const controls = [
      screen.getByText("色板细节").closest("label")?.querySelector("input"),
      screen.getByText("处理强度").closest("label")?.querySelector("input"),
      screen.getByText("亮度强度").closest("label")?.querySelector("input"),
      screen.getByText("色度强度").closest("label")?.querySelector("input"),
      screen.getByText("边缘保护").closest("label")?.querySelector("input"),
    ];

    for (const control of controls) {
      expect(control).toBeInstanceOf(HTMLInputElement);
      expect(control).not.toBeDisabled();
    }
  });
});
