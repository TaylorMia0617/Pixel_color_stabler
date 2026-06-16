这张图里的“脏块”主要不是普通噪声，而是**跨色系小孤岛**：比如草莓红色区域里出现了绿色块、白色高光边缘里夹杂灰红碎块、叶子里夹了错误红块。单纯做 KMeans / 调色板压缩会把这些脏点变成更明显的“统一脏色块”，所以要在**颜色标签图 label map** 上清理，而不是只在 RGB 图上模糊。

我用你这张图快速试了一版方向，可以看这个对比：
[原图 / 强清理平涂 / 保留层次清理 对比图](sandbox:/mnt/data/strawberry_dirty_cleaner_compare.png)

单图版本：
[强清理平涂版](sandbox:/mnt/data/strawberry_dirty_cleaner_flat.png)
[保留层次清理版](sandbox:/mnt/data/strawberry_dirty_cleaner_shaded.png)

## 推荐方案：色系级“脏块清理器”

核心流程：

```text
输入图
  ↓
背景/主体分离
  ↓
LAB 色彩空间聚类，得到颜色标签图
  ↓
把颜色标签归并成色系：红 / 绿 / 白高光 / 阴影
  ↓
连通域分析，删除小色系孤岛
  ↓
用周围主色重填脏块
  ↓
重新上色：平涂版或保留明暗版
  ↓
输出
```

关键点是这一步：

```text
红色草莓主体里的小绿色块  →  周围是红色，所以改成红色
绿色叶子里的小红色块      →  周围是绿色，所以改成绿色
白色高光里的碎红/灰块      →  小块删除，大高光保留
红色主体里的细碎暗斑       →  合并到最近的红色明暗层
```

## 具体算法设计

### 1. 先做背景保护

不要直接全图聚类，否则白背景、白高光、浅色边缘容易混在一起。

可以先用边缘 flood fill 找背景：

```text
从图像四边开始，找接近白色的连通区域 → 背景
没有连到边缘的白色区域 → 可能是草莓高光，保留为主体
```

这样白色背景不会污染草莓内部高光。

### 2. 用 LAB 做颜色聚类

不要直接 RGB 聚类。RGB 下亮度变化会让同一种红色被切成很多脏块。建议转 LAB：

```text
L = 明度
A/B = 色相和色度
```

聚类时可以用：

```text
KMeans / MiniBatchKMeans
k = 5 ~ 8
```

例如这张图大概会得到：

```text
红亮
红中
红暗
绿亮
绿暗
白高光
背景白
```

但注意：**聚类只是第一步，不是最终结果**。真正清理脏块要看空间连通性。

### 3. 把 cluster 合并成“色系 family”

聚类标签太细，需要先归并成大色系：

```text
red_family      = 所有红色 cluster
green_family    = 所有绿色 cluster
highlight_family = 白色 / 粉白高光 cluster
background      = 边缘连通白色
```

这一步很重要。你不是只想减少颜色数量，而是想让每个视觉区域内部干净。

### 4. 连通域删除小孤岛

这是清理脏块的核心。

对每个色系 mask 做 connected components：

```text
green_family:
  保留叶子/茎的大连通域
  删除草莓内部的小绿色孤岛

red_family:
  保留草莓主体大连通域
  删除叶子内部的小红色孤岛

highlight_family:
  保留大高光区域
  删除零散白点或灰白脏点
```

被删除的小块不要简单涂黑或涂白，而是：

```text
膨胀这个小块一圈
统计周围邻居最多的色系
把小块改成周围最多的色系
```

也就是“邻域多数投票重填”。

### 5. 对 label map 做中值滤波，不要对 RGB 做大模糊

很多人会直接 `GaussianBlur` 或 `medianBlur` 原图，这会导致边缘糊掉。更好的方式是：

```text
对颜色标签图做 medianBlur
再按标签重新上色
```

例如：

```python
label_map = cv2.medianBlur(label_map, 5)
```

但要注意，只在主体内部做，边缘区域少动，否则叶子尖、草莓轮廓会被吃掉。

### 6. 两种输出模式

建议你的工具提供两个模式。

**A. 强清理平涂模式**

适合图标、贴纸、UI、小尺寸素材：

```text
每个色系只保留 1 个主色
红色区域统一红
绿色区域统一绿
高光统一淡粉白
```

优点：最干净。
缺点：会变得更平面、更像矢量图。

**B. 保留层次清理模式**

适合 SD 图片观感优化：

```text
保留 L 明度层次
统一 A/B 色相
把红色区域压成 3~4 个明暗层
绿色区域压成 2~3 个明暗层
高光压成 1~2 个明暗层
```

也就是：

```text
颜色干净，但不完全丢失体积感
```

这个模式更适合你说的“优化 SD 模型生成图片观感”。

## 推荐默认参数

可以先给工具设计这些参数：

```python
palette_colors = 6          # 初始聚类颜色数
family_clean = True         # 是否启用色系级清理
speck_area = 40             # 删除极小噪点
island_area = 160           # 删除跨色系小孤岛
green_keep_area = 600       # 绿色主连通域保留阈值
highlight_min_area = 120    # 小高光删除阈值
label_median_kernel = 5     # 标签图中值滤波核
mode = "shaded"             # flat / shaded
```

面积参数建议按图像尺寸缩放：

```python
scale = (width * height) / (512 * 512)

speck_area = int(40 * scale)
island_area = int(160 * scale)
green_keep_area = int(600 * scale)
highlight_min_area = int(120 * scale)
```

## 代码核心结构示意

你的项目里可以做成这样：

```text
pixel_color_stabler/
  main.py
  stabilizer/
    io.py
    background.py
    palette.py
    family.py
    island_cleaner.py
    repaint.py
    presets.py
```

核心函数可以这样设计：

```python
def stabilize_image(
    image,
    palette_colors=6,
    mode="shaded",
    speck_area=40,
    island_area=160,
    preserve_highlight=True,
    preserve_edges=True,
):
    bg_mask = detect_edge_connected_background(image)

    lab = rgb_to_lab(image)
    lab_smooth = bilateral_filter(lab)

    label_map, palette = cluster_lab_colors(
        lab_smooth,
        mask=~bg_mask,
        k=palette_colors,
    )

    family_map = classify_color_families(
        palette,
        label_map,
        bg_mask=bg_mask,
    )

    family_map = remove_dirty_islands(
        family_map,
        bg_mask=bg_mask,
        island_area=island_area,
    )

    label_map = smooth_label_map(
        label_map,
        family_map,
        kernel_size=5,
    )

    output = repaint_from_families(
        image,
        lab,
        family_map,
        mode=mode,
    )

    return output
```

## 最重要的清理函数

重点是这个逻辑：

```python
def remove_small_components(family_map, family_id, min_area):
    mask = (family_map == family_id).astype("uint8")

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    for component_id in range(1, num):
        area = stats[component_id, cv2.CC_STAT_AREA]

        if area < min_area:
            component = labels == component_id

            # 找这个小块周围一圈的主色系
            border = dilate(component) & ~component
            neighbor_families = family_map[border]

            # 去掉背景和自己
            target_family = majority_vote(neighbor_families)

            # 用周围主色替换
            family_map[component] = target_family

    return family_map
```

这比直接模糊有效很多，因为它会明确识别：

```text
“这是红色区域里的绿色脏块”
“这是高光里的灰色脏块”
“这是叶子里的红色脏块”
```

## 针对 SD 图片的建议

SD 图常见问题是局部颜色抖动、材质脏斑、边缘脏色。建议处理顺序是：

```text
SD 原图
  ↓
色系清理
  ↓
轻微锐化 / upscale
  ↓
最终导出
```

不要先 upscale 再清理。放大后脏块面积也会变大，阈值更难调，而且会被超分模型“合理化”为纹理。

比较适合的默认 preset：

```text
clean_icon:
  mode = flat
  palette_colors = 4~6
  island_area = high
  highlight_min_area = high

clean_sd_soft:
  mode = shaded
  palette_colors = 6~10
  island_area = medium
  label_median_kernel = 3~5

clean_sd_strong:
  mode = shaded
  palette_colors = 5~7
  island_area = high
  label_median_kernel = 5
```

你的项目方向建议不要只叫“color compression”，更准确应该是：

```text
Palette Stabilization + Dirty Island Removal
```

也就是：**调色板稳定 + 跨色系脏块清理**。这才是解决这类 SD 脏色块的关键。