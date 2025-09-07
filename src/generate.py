# ==========================================
# generate.py – 立绘合成器
# 读取 PSD 导出的 txt 坐标 + PNG 切片，合成最终立绘
# ==========================================

import csv
import cv2
import numpy as np

def generate_fgimage(target, embeddings_layers):
    """
    target: "ムラサメa" 或 "ムラサメb"，对应两套资源
    embeddings_layers: 如 [1717, 1475, 1261]
    返回: BGRA 的 numpy 画布，可直接被 Qt 显示
    """
    assert target in ["ムラサメa", "ムラサメb"]
    # 读取 PSD 导出信息（utf-16 le 是 PhotoShop 默认）
    with open(f"../fgimages/{target}.txt", encoding='utf-16 le') as cf:
        infos = list(csv.reader(cf, delimiter='\t'))    # 每行：图层名/坐标/尺寸/PNG 路径等

    # 根据 target 选基础人物图层区间（用于坐标对齐）
    if target == "ムラサメa":
        all_base = infos[57:65]
    else:
        all_base = infos[47:51]

    # 收集要叠加的图层坐标
    all_positions = [(int(x[2]), int(x[3]), int(x[4]), int(x[5]))
                     for name in embeddings_layers for x in infos if x[9] == str(name)]

    # 基础人物坐标（用于整体偏移归一化）
    all_base = [(int(x[2]), int(x[3]), int(x[4]), int(x[5]))
                for x in all_base]

    # 左上角对齐：所有坐标减去最小 x,y
    all_positions = [(pos[0] - min(pos[0] for pos in all_base), pos[1] - min(pos[1] for pos in all_base), pos[2], pos[3])
                     for pos in all_positions]

    # 画布大小 = 最大右下坐标
    canvas_scale = (max([(x[0] + x[2]) for x in all_positions]),
                    max([(x[1] + x[3]) for x in all_positions]))

    # 4 通道全透明
    canvas = np.zeros((canvas_scale[1], canvas_scale[0], 4), dtype=np.uint8)

    # 逐层叠加 PNG（带透明通道）
    for idx, pos in enumerate(all_positions):
        path = f"../fgimages/{target}_{embeddings_layers[idx]}.png"
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)     # -1 保留 alpha
        if image is not None:
            x_offset = pos[0]
            y_offset = pos[1]
            h, w = image.shape[:2]
            alpha_img = image[..., 3:] / 255.0      # 0~1
            alpha_canvas = 1.0 - alpha_img
            for c in range(3):      # BGR
                canvas[y_offset:y_offset + h, x_offset:x_offset + w, c] = (
                    alpha_img[..., 0] * image[..., c] +
                    alpha_canvas[..., 0] * canvas[y_offset:y_offset +
                                                  h, x_offset:x_offset + w, c]
                )

            # alpha 通道取最大值，避免半透明叠加出错
            canvas[y_offset:y_offset + h, x_offset:x_offset + w, 3] = (
                np.maximum(
                    image[..., 3], canvas[y_offset:y_offset + h, x_offset:x_offset + w, 3])
            )

    return canvas