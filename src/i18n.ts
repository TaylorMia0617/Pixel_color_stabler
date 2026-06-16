const zhCN = {
  appTitle: "像素颜色稳定器",
  appSubtitle: "导入图片后，自动合并全图中的近似色像素。",
  importImage: "导入图片",
  controls: "控制",
  tolerance: "色差容差",
  strength: "处理强度",
  zoom: "预览缩放",
  resetZoom: "恢复 100% 缩放",
  resetControls: "重置参数",
  reprocess: "重新处理",
  exportPng: "导出 PNG",
  changed: "变更像素",
  clusters: "颜色簇",
  size: "尺寸",
  original: "原图",
  stabilized: "降噪后",
  loadingImage: "正在载入图片...",
  dropImage: "拖入图片，或点击导入",
  processing: "正在处理",
  clustering: "正在进行局部降噪",
  rendering: "正在生成结果",
  chooseImageFile: "请选择图片文件。",
  loadImageFailed: "无法载入图片。",
  processingFailed: "无法完成图片处理。",
  exportFailed: "无法导出 PNG。",
  stabilizedFileName: "降噪图片.png",
} as const;

export type TranslationKey = keyof typeof zhCN;

export function t(key: TranslationKey): string {
  return zhCN[key];
}
