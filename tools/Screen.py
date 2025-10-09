import os
import shutil
import numpy as np
import tifffile

# ———— 配置区 ————
tiles_dir        = "/home/user/Downloads/孙嘉鸿(1)/tiles_tif"                    # 原始 TIFF 子图文件夹
uninformative_dir = "/home/user/Downloads/孙嘉鸿(1)/tiles_tif/tiles_uninformative"         # 存放无信息图
informative_dir   = "/home/user/Downloads/孙嘉鸿(1)/tiles_tif/tiles_informative"           # 存放有用图
low_thresh       = 10                             # 黑色阈值 (<10 视作黑)
high_thresh      = 245                            # 白色阈值 (>245 视作白)
mid_ratio_thresh = 0.01                           # 中间像素比例阈值 (<1% 视为无信息)
# ————————————

os.makedirs(uninformative_dir, exist_ok=True)
os.makedirs(informative_dir, exist_ok=True)

for fname in os.listdir(tiles_dir):
    if not fname.lower().endswith(".tif"):
        continue
    path = os.path.join(tiles_dir, fname)
    arr  = tifffile.imread(path)           # shape = (H, W, C) 或 (H, W)

    # 如果是单通道直接用 arr，否则对多通道取平均
    if arr.ndim == 3:
        gray = arr.mean(axis=2).astype(np.uint8)
    else:
        gray = arr

    total = gray.size
    # 计算“中间”像素：low_thresh ≤ gray ≤ high_thresh
    mid_mask = (gray >= low_thresh) & (gray <= high_thresh)
    mid_ratio = mid_mask.sum() / total

    if mid_ratio < mid_ratio_thresh:
        # 几乎没有中间灰度，基本只黑白两种块
        shutil.move(path, os.path.join(uninformative_dir, fname))
        print(f"[UNINFO] {fname}  中间灰度比={mid_ratio:.3%}")
    else:
        shutil.move(path, os.path.join(informative_dir, fname))
        print(f"[KEEP]    {fname}  中间灰度比={mid_ratio:.3%}")
