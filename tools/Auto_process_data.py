# -*- coding: utf-8 -*-
import os
import glob
import tifffile
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import shutil
import traceback
import math
from tqdm import tqdm # 用于显示进度条

# --- 尝试导入 scipy ---
try:
    from scipy.ndimage import laplace
    SCIPY_AVAILABLE = True
    print("SciPy library found, advanced tile filtering enabled.")
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: SciPy library not found. Advanced tile filtering ('laplacian', 'combined_...') will not work.")
    print("         Install using: pip install scipy")

# --------------------------------------------------
# 需要修改的参数 (Parameters to modify)
# --------------------------------------------------
input_dir = "/home/user/Documents/UNSB-main/datasets/organoids_0826/AF"       # 输入目录: 包含 JPG, PNG, TIF 等
output_base_dir_all = "//home/user/Documents/UNSB-main/datasets/organoids_0826/AF" # 输出目录: 存放所有切割出的图块
output_base_dir_filtered = "/home/user/Documents/UNSB-main/datasets/organoids_0826/AF" # 输出目录: 只存放通过过滤的图块

# --- 切片参数 ---
tile_width = 256
tile_height = 256
overlap_x = 32
overlap_y = 32
save_format = 'tif' # 'tif' or 'png'

# --- 原始大图过滤参数 (可选) ---
filter_low_info_original = True
info_threshold_original = 0.0
rich_images_dir = "/home/user/Downloads/output_rich_originals"

# --- 图块过滤方法选择 ---
# 'none': 不过滤
# 'std_dev': 基于标准差过滤
# 'combined_dark_bright_laplacian_uniform': 推荐，综合过滤暗/亮背景、低信息量和单一颜色 (需要 scipy)
default_filter = 'combined_dark_bright_laplacian_uniform' if SCIPY_AVAILABLE else 'std_dev'
tile_filter_method = default_filter

# --- 'combined_...' 过滤阈值 ---
dark_intensity_cutoff = 2          # 低于此值的像素视为"暗"
dark_percent_threshold = 0.90       # 暗像素占比超过此值则过滤
bright_intensity_cutoff = 235       # 高于此值的像素视为"亮" (对于8位图像)
bright_percent_threshold = 0.80     # 亮像素占比超过此值则过滤
laplacian_threshold = 15            # 拉普拉斯方差低于此值则过滤 (低信息量)
uniform_color_percent_threshold = 0.90 # 单一颜色占比超过此值则过滤 (可选，作为补充)

# --- 'std_dev' 过滤阈值 ---
tile_std_dev_threshold = 8.0        # 低于此标准差的图块将被过滤

# --- 图块处理选项 (可选) ---
perform_invert = False
convert_to_rgb = False

# --- 支持的输入文件格式 ---
supported_extensions = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]

# Error checking for overlap
if overlap_x >= tile_width or overlap_y >= tile_height:
    print(f"错误: 重叠 ({overlap_x}x{overlap_y}) 不能大于或等于图块尺寸 ({tile_width}x{tile_height})。")
    exit(1)

# --------------------------------------------------
# 图块过滤函数 (更新版)
# --------------------------------------------------
def keep_tile(tile_np, method, **kwargs):
    """
    根据选择的方法和阈值判断是否保留图块。
    Handles grayscale and RGB images.
    """
    if not isinstance(tile_np, np.ndarray) or tile_np.ndim < 2: return False
    if tile_np.shape[0] == 0 or tile_np.shape[1] == 0: return False

    # --- Get basic image properties ---
    is_rgb = tile_np.ndim == 3 and tile_np.shape[2] == 3
    is_gray = tile_np.ndim == 2
    tile_h, tile_w = tile_np.shape[:2]
    total_pixels = tile_h * tile_w
    if total_pixels == 0: return False

    # --- Ensure we have grayscale version for intensity/laplacian checks ---
    gray_tile = None
    if is_gray:
        gray_tile = tile_np
    elif is_rgb:
        try:
            # Convert RGB to Grayscale using Pillow's method (Luminosity)
            gray_tile = np.array(Image.fromarray(tile_np).convert('L'))
        except Exception as e:
            # print(f"Debug: Failed to convert RGB tile to grayscale: {e}")
            return True # Fail safe: keep tile if conversion fails
    else: # Handle other formats (e.g., RGBA) - attempt conversion or keep
        try:
            temp_pil = Image.fromarray(tile_np)
            # Try getting RGB first, then grayscale
            rgb_pil = temp_pil.convert('RGB')
            gray_tile = np.array(rgb_pil.convert('L'))
            tile_np = np.array(rgb_pil) # Update tile_np to the RGB version for uniformity check later
            is_rgb = True
            is_gray = False # Now effectively processing as RGB/Gray
            # print("Debug: Converted multi-channel image to RGB/Grayscale for filtering.")
        except Exception as e:
            # print(f"Debug: Could not convert tile with shape {tile_np.shape} to RGB/Gray: {e}")
            return True # Fail safe: keep tile

    if gray_tile is None: # Should not happen if logic above is correct, but check
         # print("Debug: Grayscale tile is None unexpectedly.")
         return True

    # --- Filter Method Logic ---
    if method == 'none':
        return True

    elif method == 'std_dev':
        threshold = kwargs.get('std_dev_threshold', 8.0)
        # Use grayscale standard deviation as a simple measure of information
        if gray_tile.max() == gray_tile.min(): std_dev = 0.0
        else: std_dev = np.std(gray_tile)
        return std_dev >= threshold

    elif method == 'combined_dark_bright_laplacian_uniform':
        if not SCIPY_AVAILABLE:
            print("警告: SciPy不可用，无法执行'combined_...'过滤，默认保留图块。")
            return True

        dark_cutoff = kwargs.get('dark_intensity_cutoff', 60)
        dark_perc = kwargs.get('dark_percent_threshold', 0.80)
        bright_cutoff = kwargs.get('bright_intensity_cutoff', 235) # Assumes 8-bit range for default
        bright_perc = kwargs.get('bright_percent_threshold', 0.80)
        lap_thresh = kwargs.get('laplacian_threshold', 15)
        uniform_perc = kwargs.get('uniform_color_percent_threshold', 0.90) # Threshold for the *optional* uniform color check

        try:
            # 1. Dark Background Check (using grayscale)
            dark_pixels = np.sum(gray_tile < dark_cutoff)
            if (dark_pixels / total_pixels) >= dark_perc:
                # print("Debug: Filtered by dark background")
                return False

            # 2. Bright Background Check (using grayscale)
            bright_pixels = np.sum(gray_tile > bright_cutoff)
            if (bright_pixels / total_pixels) >= bright_perc:
                # print("Debug: Filtered by bright background")
                return False

            # 3. Low Information Check (Laplacian variance on grayscale)
            # Avoid calculation if already determined to be background
            lap_var = np.var(laplace(gray_tile))
            if lap_var <= lap_thresh:
                # print(f"Debug: Filtered by laplacian variance ({lap_var:.2f})")
                return False

            # 4. (Optional) Uniform Color Check (using RGB if available, else pseudo-RGB)
            # This catches cases like a solid blue tile that might pass other checks.
            # Create RGB representation if needed for uniformity check
            tile_rgb_for_uniformity = None
            if is_rgb:
                tile_rgb_for_uniformity = tile_np
            elif is_gray: # If original was gray, create pseudo-RGB
                tile_rgb_for_uniformity = np.stack((gray_tile,)*3, axis=-1)

            if tile_rgb_for_uniformity is not None:
                 colors, counts = np.unique(tile_rgb_for_uniformity.reshape(-1, 3), axis=0, return_counts=True)
                 if len(counts) > 0:
                     most_frequent_count = np.max(counts)
                     if (most_frequent_count / total_pixels) > uniform_perc:
                         # print(f"Debug: Filtered by uniform color ({most_frequent_count / total_pixels * 100:.1f}%)")
                         return False
                 # else: pass # No colors counted? Strange, likely empty tile, handled earlier.

            # If none of the above conditions triggered filtering, keep the tile
            return True

        except Exception as e:
            # print(f"组合过滤计算时出错: {e}") # Can be noisy
            return False # Error during filter -> reject

    else:
        print(f"警告: 未知的过滤方法 '{method}'。将保留图块。")
        return True


# --------------------------------------------------
# 函数：处理单个图片 (No changes needed here, logic relies on keep_tile)
# --------------------------------------------------
# --------------------------------------------------
# 函数：处理单个图片 (修正版)
# --------------------------------------------------
def process_single_image(
    image_path,
    output_dir_all_img,
    output_dir_filtered_img,
    tile_w, tile_h,
    ovlp_x, ovlp_y,
    save_fmt,
    # 原始图过滤参数
    enable_orig_filter=False,
    std_dev_thresh_orig=10.0,
    rich_img_output_dir=None,
    # 图块过滤参数
    tile_flt_method='none',
    tile_filter_params={},
    # 图块处理参数
    inv_op=False,
    rgb_op=False
    ):
    """
    处理单个标准图像文件。(修正了 is_rich 定义)
    """
    original_filename = os.path.basename(image_path)
    print(f"---\n处理图像 Processing image: {original_filename}")

    img_pil = None
    try:
        # ---------------------------------------
        # 1. 打开图像 & (可选) 原始图像信息评估和复制
        # ---------------------------------------
        is_rich = True # Default to True, will be updated if filtering is enabled

        try:
            img_pil = Image.open(image_path)
            pic_width, pic_height = img_pil.size
            print(f"  类型: 标准图像 (模式: {img_pil.mode}, 尺寸: {pic_width}x{pic_height})")
            if pic_width < tile_w or pic_height < tile_h:
                 print(f"  警告: 图像尺寸 ({pic_width}x{pic_height}) 小于目标图块尺寸 ({tile_w}x{tile_h})。")
        # Keep concise error handling for file opening
        except UnidentifiedImageError: print(f"  错误：Pillow 无法识别 '{original_filename}'"); return
        except MemoryError: print(f"  错误：打开 '{original_filename}' 时内存不足。"); return
        except Exception as e: print(f"  错误：打开标准图像时出错: {e}"); traceback.print_exc(); return


        # --- 原始图像过滤逻辑 ---
        if enable_orig_filter:
            print("  评估原始标准图像信息量...")
            std_dev_orig = -1.0 # Initialize std_dev_orig
            try:
                # 使用灰度图计算标准差
                img_gray_pil = img_pil.convert('L')
                img_gray_data = np.array(img_gray_pil)
                if img_gray_data.size == 0:
                    std_dev_orig = 0.0
                    print("    警告: 图像为空，无法计算标准差。")
                    is_rich = False # Treat empty image as not rich
                else:
                    std_dev_orig = np.std(img_gray_data)
                    # *** 这就是 is_rich 被赋值的地方 ***
                    is_rich = std_dev_orig >= std_dev_thresh_orig
                    print(f"    原始图标准差: {std_dev_orig:.4f} -> {'丰富 Rich' if is_rich else '低 Low'} (阈值 {std_dev_thresh_orig})")
            except MemoryError:
                 print(f"    内存不足评估标准差，默认视为丰富。")
                 is_rich = True # Assign True on MemoryError, proceed with caution
            except Exception as e_eval:
                print(f"    评估出错: {e_eval}，默认视为丰富。")
                is_rich = True # Assign True on other errors

            # --- 如果过滤开启且判定为低信息量，则跳过切片 ---
            if not is_rich:
                print(f"  图像 '{original_filename}' 信息量低，跳过切片。")
                if img_pil: img_pil.close() # Ensure file is closed
                return # Exit this function for this image

            # --- 复制信息量丰富的原始文件 (如果需要) ---
            if is_rich and rich_img_output_dir: # Check is_rich again (it's True here if reached)
                # 创建目录（如果不存在）
                os.makedirs(rich_img_output_dir, exist_ok=True)
                dest_path = os.path.join(rich_img_output_dir, original_filename)
                print(f"  复制信息量丰富的原图到: {dest_path}")
                try:
                    shutil.copy2(image_path, dest_path)
                except Exception as copy_err:
                    print(f"    复制文件时出错: {copy_err}")
                    # Decide: continue tiling or stop? Let's continue.


        # ---------------------------------------
        # 2. 计算切片坐标 (带重叠，边缘对齐)
        # ---------------------------------------
        print(f"  开始切片 (Tile: {tile_w}x{tile_h}, Overlap: {ovlp_x}x{ovlp_y}, Filter: '{tile_flt_method}')...")
        # Create output directories for this specific image
        os.makedirs(output_dir_all_img, exist_ok=True)
        os.makedirs(output_dir_filtered_img, exist_ok=True)
        # print(f"    输出目录 (所有): {output_dir_all_img}") # Reduce verbosity
        # print(f"    输出目录 (过滤后): {output_dir_filtered_img}")

        if pic_width == 0 or pic_height == 0: # Check again after potential early exit
             print("  错误: 图像尺寸为0，无法切片。")
             if img_pil: img_pil.close()
             return

        # --- Coordinate Calculation ---
        step_x = tile_w - ovlp_x
        step_y = tile_h - ovlp_y
        x_coords = list(range(0, pic_width - tile_w, step_x))
        if pic_width >= tile_w: x_coords.append(pic_width - tile_w)
        elif pic_width > 0: x_coords = [0]
        else: x_coords = []
        x_coords = sorted(list(set(x_coords)))

        y_coords = list(range(0, pic_height - tile_h, step_y))
        if pic_height >= tile_h: y_coords.append(pic_height - tile_h)
        elif pic_height > 0: y_coords = [0]
        else: y_coords = []
        y_coords = sorted(list(set(y_coords)))

        num_tiles = len(y_coords) * len(x_coords)
        if num_tiles == 0 and (pic_width > 0 and pic_height > 0):
             print(f"    警告: 计算得到的图块数量为0，请检查图像/图块/重叠尺寸。")
        else:
             print(f"    计算得到网格坐标数: {len(y_coords)} rows x {len(x_coords)} cols = {num_tiles} tiles")

        # ---------------------------------------
        # 3. 遍历坐标进行切片、处理、保存
        # ---------------------------------------
        saved_all_count = 0
        saved_filtered_count = 0
        processed_tile_count = 0
        pbar_desc = f"Tiles ({os.path.basename(image_path)[:15]}..)"

        # --- Tiling Loop ---
        with tqdm(total=num_tiles, desc=pbar_desc, leave=False) as pbar:
             for y in y_coords:
                 for x in x_coords:
                     box_x1, box_y1 = x, y
                     box_x2, box_y2 = x + tile_w, y + tile_h

                     try:
                         # 1. Crop
                         tile_pil = img_pil.crop((box_x1, box_y1, box_x2, box_y2))
                         processed_tile_count += 1

                         # 2. Optional Processing (RGB, Invert)
                         tile_to_process_pil = tile_pil
                         if rgb_op:
                             if tile_to_process_pil.mode != 'RGB':
                                 try: tile_to_process_pil = tile_to_process_pil.convert('RGB')
                                 except ValueError: pass # Log or handle conversion error if needed

                         tile_np = np.array(tile_to_process_pil)

                         if inv_op:
                             # Inversion logic...
                              try:
                                   if np.issubdtype(tile_np.dtype, np.integer):
                                        max_val = np.iinfo(tile_np.dtype).max
                                        tile_np_processed = max_val - tile_np
                                   elif np.issubdtype(tile_np.dtype, np.floating):
                                        max_val = 1.0 if np.nanmax(tile_np) <= 1.1 else 255.0 # Use nanmax
                                        tile_np_processed = max_val - tile_np
                                   else: tile_np_processed = tile_np
                              except Exception: tile_np_processed = tile_np # Fallback on error
                         else:
                             tile_np_processed = tile_np # Use the (potentially RGB converted) numpy array


                         # 3. Save to "all" directory
                         tile_base_filename = f"{original_filename}_y{y:05d}_x{x:05d}"
                         save_path_all = None
                         if save_fmt == 'tif':
                             save_path_all = os.path.join(output_dir_all_img, tile_base_filename + ".tif")
                             try:
                                 tifffile.imwrite(save_path_all, tile_np_processed, imagej=True); saved_all_count += 1
                             except Exception as e_save:
                                 print(f"\n      保存TIF到'all'失败 ({y},{x}): {e_save}"); save_path_all = None
                         elif save_fmt == 'png':
                             save_path_all = os.path.join(output_dir_all_img, tile_base_filename + ".png")
                             try:
                                 Image.fromarray(tile_np_processed).save(save_path_all); saved_all_count += 1
                             except Exception as e_save:
                                 print(f"\n      保存PNG到'all'失败 ({y},{x}): {e_save}"); save_path_all = None

                         # 4. Apply Tile Filter using the processed numpy array
                         is_kept = keep_tile(tile_np_processed, tile_flt_method, **tile_filter_params)

                         # 5. Save to "filtered" directory if kept
                         if is_kept:
                             save_path_filtered = None
                             if save_fmt == 'tif':
                                 save_path_filtered = os.path.join(output_dir_filtered_img, tile_base_filename + ".tif")
                                 # Optimization: Copy if already saved, otherwise save anew
                                 if save_path_all and os.path.exists(save_path_all):
                                     try: shutil.copy2(save_path_all, save_path_filtered); saved_filtered_count += 1
                                     except Exception as e_copy: print(f"\n    复制TIF到'filtered'失败: {e_copy}")
                                 else: # Re-save if initial save failed or wasn't attempted
                                     try: tifffile.imwrite(save_path_filtered, tile_np_processed, imagej=True); saved_filtered_count += 1
                                     except Exception as e_save: print(f"\n    保存TIF到'filtered'失败: {e_save}")
                             elif save_fmt == 'png':
                                 save_path_filtered = os.path.join(output_dir_filtered_img, tile_base_filename + ".png")
                                 if save_path_all and os.path.exists(save_path_all):
                                     try: shutil.copy2(save_path_all, save_path_filtered); saved_filtered_count += 1
                                     except Exception as e_copy: print(f"\n    复制PNG到'filtered'失败: {e_copy}")
                                 else:
                                     try: Image.fromarray(tile_np_processed).save(save_path_filtered); saved_filtered_count += 1
                                     except Exception as e_save: print(f"\n    保存PNG到'filtered'失败: {e_save}")

                     # Handle errors during tile processing
                     except MemoryError: print(f"\n      处理图块时内存不足 (y={y}, x={x})，跳过。")
                     except Exception as e_proc: print(f"\n      处理图块失败 (y={y}, x={x}): {e_proc}"); traceback.print_exc()

                     pbar.update(1)
                     pbar.set_postfix({"All": saved_all_count, "Kept": saved_filtered_count})


        print(f"  完成 '{original_filename}'. 处理: {processed_tile_count}, 保存(all): {saved_all_count}, 保存(filtered): {saved_filtered_count}")

    # --- Main exception handler for the whole image processing ---
    except Exception as e_main:
        print(f"  处理文件 '{original_filename}' 时发生主流程错误: {e_main}"); traceback.print_exc()
    # --- Finally block to ensure resources are released ---
    finally:
        if img_pil: # Check if img_pil was successfully assigned
            try: img_pil.close()
            except Exception: pass # Ignore potential errors during close

# --------------------------------------------------
# 主程序 - 批量处理
# --------------------------------------------------
if __name__ == "__main__":
    print("="*50)
    print("开始批量图像切片脚本 (带重叠和过滤)")
    print(f"SciPy 可用: {SCIPY_AVAILABLE}")
    print("="*50)
    # --- Print Parameters ---
    print(f"输入目录: {input_dir}")
    print(f"输出目录 (所有图块): {output_base_dir_all}")
    print(f"输出目录 (过滤后图块): {output_base_dir_filtered}")
    print(f"--- 切片设置 ---")
    print(f"  图块尺寸 (WxH): {tile_width}x{tile_height}")
    print(f"  重叠 (X, Y): {overlap_x}, {overlap_y}")
    print(f"  保存格式: {save_format}")
    print(f"--- 原始图过滤 ---")
    print(f"  启用: {filter_low_info_original}")
    if filter_low_info_original:
        print(f"  标准差阈值: {info_threshold_original}")
        print(f"  丰富图片输出到: {rich_images_dir if rich_images_dir else '未指定'}")
    print(f"--- 图块过滤 ---")
    print(f"  过滤方法: '{tile_filter_method}'")
    # Print thresholds based on selected method
    if tile_filter_method == 'combined_dark_bright_laplacian_uniform':
        print(f"    Combined阈值: Dark Area(<{dark_intensity_cutoff}) > {dark_percent_threshold*100:.0f}% OR "
              f"Bright Area(>{bright_intensity_cutoff}) > {bright_percent_threshold*100:.0f}% OR "
              f"Laplacian Var <= {laplacian_threshold} OR "
              f"Uniform Color > {uniform_color_percent_threshold*100:.0f}%")
    elif tile_filter_method == 'std_dev':
        print(f"    标准差阈值: < {tile_std_dev_threshold}")
    print(f"--- 图块处理 ---")
    print(f"  反转颜色: {perform_invert}")
    print(f"  强制RGB: {convert_to_rgb}")
    print("="*50)

    # --- Setup Directories ---
    if not os.path.isdir(input_dir): print(f"\n错误：输入目录 '{input_dir}' 不存在。"); exit(1)
    try:
        os.makedirs(output_base_dir_all, exist_ok=True)
        os.makedirs(output_base_dir_filtered, exist_ok=True)
        if filter_low_info_original and rich_images_dir:
            os.makedirs(rich_images_dir, exist_ok=True)
    except OSError as e: print(f"\n错误：无法创建输出目录: {e}"); exit(1)

    # --- Find Files ---
    print("正在扫描输入目录...")
    all_image_files = []
    # ... (File scanning logic - same as before) ...
    for ext in supported_extensions:
        search_pattern = os.path.join(input_dir, ext)
        found_files = glob.glob(search_pattern, recursive=False)
        if found_files:
            print(f"  找到 {len(found_files)} 个 '{ext}' 文件")
            all_image_files.extend(found_files)
    unique_files = sorted(list(set(all_image_files)))
    if not unique_files: print("\n未找到支持的图片文件。"); exit(0)
    print(f"\n总共找到 {len(unique_files)} 个唯一的图片文件进行处理。")

    # --- Prepare Filter Kwargs ---
    filter_kwargs = {
        'std_dev_threshold': tile_std_dev_threshold, # For std_dev method
        'dark_intensity_cutoff': dark_intensity_cutoff, # For combined method
        'dark_percent_threshold': dark_percent_threshold,
        'bright_intensity_cutoff': bright_intensity_cutoff,
        'bright_percent_threshold': bright_percent_threshold,
        'laplacian_threshold': laplacian_threshold,
        'uniform_color_percent_threshold': uniform_color_percent_threshold,
    }

    # --- Main Loop ---
    total_processed_files = 0
    total_error_files = 0
    for image_file_path in unique_files:
        try:
            base_filename = os.path.basename(image_file_path)
            filename_without_ext = os.path.splitext(base_filename)[0]
            output_subdir_all = os.path.join(output_base_dir_all, filename_without_ext)
            output_subdir_filtered = os.path.join(output_base_dir_filtered, filename_without_ext)

            # Call the processing function
            process_single_image(
                image_path=image_file_path,
                output_dir_all_img=output_subdir_all,
                output_dir_filtered_img=output_subdir_filtered,
                tile_w=tile_width, tile_h=tile_height,
                ovlp_x=overlap_x, ovlp_y=overlap_y,
                save_fmt=save_format,
                enable_orig_filter=filter_low_info_original,
                std_dev_thresh_orig=info_threshold_original,
                rich_img_output_dir=rich_images_dir,
                tile_flt_method=tile_filter_method,
                tile_filter_params=filter_kwargs,
                inv_op=perform_invert,
                rgb_op=convert_to_rgb
            )
            total_processed_files += 1
        except KeyboardInterrupt: print("\n !!! 用户中断操作 !!!"); break
        except Exception as e:
            print(f"!!! 处理文件 {os.path.basename(image_file_path)} 时发生顶层错误，跳过 !!!")
            print(f"!!! Error: {e} !!!"); traceback.print_exc()
            total_error_files += 1

    # --- Final Summary ---
    print("\n" + "="*50)
    print("--- 批量处理完成 ---")
    # ... (Summary printing - same as before) ...
    print(f"总尝试处理文件数: {len(unique_files)}")
    print(f"成功完成处理流程的文件数: {total_processed_files}")
    print(f"发生顶层错误的文件数: {total_error_files}")
    copied_rich_count = 0
    if filter_low_info_original and rich_images_dir and os.path.isdir(rich_images_dir):
         try: copied_rich_count = len([f for f in os.listdir(rich_images_dir) if os.path.isfile(os.path.join(rich_images_dir, f))])
         except Exception: pass
         print(f"复制到 '{os.path.basename(rich_images_dir)}' 的原始图片数: {copied_rich_count}")
    print(f"所有图块保存在 '{os.path.basename(output_base_dir_all)}' 下对应子目录中。")
    print(f"过滤后图块保存在 '{os.path.basename(output_base_dir_filtered)}' 下对应子目录中。")
    print("="*50)