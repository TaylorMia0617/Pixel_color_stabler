import { useEffect, useRef } from "react";

type CanvasPreviewProps = {
  imageData?: ImageData;
  label: string;
  zoom: number;
};

export function CanvasPreview({ imageData, label, zoom }: CanvasPreviewProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageData) {
      return;
    }

    canvas.width = imageData.width;
    canvas.height = imageData.height;
    const context = canvas.getContext("2d");
    context?.putImageData(imageData, 0, 0);
  }, [imageData]);

  return (
    <section className="preview-panel" aria-label={label}>
      <div className="preview-label">{label}</div>
      <div className="canvas-frame">
        {imageData ? (
          <canvas
            ref={canvasRef}
            style={{
              width: `${imageData.width * zoom}px`,
              height: `${imageData.height * zoom}px`,
            }}
          />
        ) : (
          <div className="empty-preview" />
        )}
      </div>
    </section>
  );
}
