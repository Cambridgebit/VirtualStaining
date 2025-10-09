# -*- coding: utf-8 -*-
import os
import glob
import re
import tifffile
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import math
from collections import defaultdict # 用于方便地分组

# --------------------------------------------------
# Parameters to Modify
# --------------------------------------------------
# --- Input ---
# 现在指向包含 *所有* 来源图片的小图块的目录
# (可以是平铺的，也可以是下面有多层子目录，脚本会递归查找)
# 例如: "/path/to/output_patches_filtered/"
# 或者: "/path/to/flat_directory_with_all_tiles/"
tile_input_dir = "/home/user/Documents/UNSB-main/results/AF2HE_SB_base_100/test_latest0/images/fake_5" # <<< 指向包含所有图块的父目录或平铺目录

# --- Original Tiling Parameters (MUST match the slicing script) ---
tile_width = 256
tile_height = 256
overlap_x = 32
overlap_y = 32
tile_format = 'png' # 'tif' or 'png' - must match how tiles were saved

# --- Output ---
output_dir = "/home/user/Documents/UNSB-main/VirtualStain/UTOM_fake5" # 存储所有拼接后大图的目录
# output_filename 不再需要，会自动根据前缀生成

# --- Stitching Parameters ---
weight_mode = 'linear'
save_format = 'tif' # 'tif' or 'png' for the *output* stitched image

# --- Optional: Force Output Data Type ---
force_output_dtype = None # e.g., np.uint8

# --------------------------------------------------
# Helper Function: Generate Weight Map (No changes needed)
# --------------------------------------------------
def generate_tile_weights(tile_h, tile_w, ovlp_y, ovlp_x, mode='linear'):
    """Generates a 2D weight map for a single tile for blending."""
    if mode == 'linear':
        if ovlp_x > 0:
            weight_x = np.ones(tile_w, dtype=np.float32)
            ramp_x = np.linspace(0.0, 1.0, ovlp_x + 1, dtype=np.float32)[1:]
            weight_x[:ovlp_x] = ramp_x
            weight_x[tile_w-ovlp_x:] = ramp_x[::-1]
        else:
            weight_x = np.ones(tile_w, dtype=np.float32)
        if ovlp_y > 0:
            weight_y = np.ones(tile_h, dtype=np.float32)
            ramp_y = np.linspace(0.0, 1.0, ovlp_y + 1, dtype=np.float32)[1:]
            weight_y[:ovlp_y] = ramp_y
            weight_y[tile_h-ovlp_y:] = ramp_y[::-1]
        else:
            weight_y = np.ones(tile_h, dtype=np.float32)
        weights_2d = np.outer(weight_y, weight_x)
        return weights_2d
    else:
        print(f"Warning: Unsupported weight mode '{mode}'. Using constant weights.")
        return np.ones((tile_h, tile_w), dtype=np.float32)

# --------------------------------------------------
# Main Stitching Logic (Batch Processing)
# --------------------------------------------------
if __name__ == "__main__":
    print("="*50)
    print("Starting Batch Tile Stitching Script")
    print(f"Input Tile Directory (scan root): {tile_input_dir}")
    print(f"Tile Format: {tile_format}")
    print(f"Tile Dimensions (WxH): {tile_width}x{tile_height}")
    print(f"Overlap (X, Y): {overlap_x}, {overlap_y}")
    print(f"Output Directory: {output_dir}")
    # print(f"Output Filename: Automatically generated based on prefix")
    print(f"Blending Mode: {weight_mode}")
    print(f"Output Save Format: {save_format}")
    print("="*50)

    # --- Validate Input ---
    if not os.path.isdir(tile_input_dir):
        print(f"Error: Input directory '{tile_input_dir}' not found.")
        exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # --- Find ALL Tiles Recursively and Group by Prefix ---
    print(f"Scanning for '{tile_format}' tiles recursively...")
    # Use recursive=True to search subdirectories as well
    search_pattern = os.path.join(tile_input_dir, '**', f"*.{tile_format}")
    all_tile_files = glob.glob(search_pattern, recursive=True)

    if not all_tile_files:
        print(f"Error: No '.{tile_format}' files found recursively in '{tile_input_dir}'.")
        exit(1)

    print(f"Found {len(all_tile_files)} potential tile files.")

    # Regex to extract prefix AND coordinates
    # Captures: group(1)=prefix, group(2)=y, group(3)=x
    prefix_coord_pattern = re.compile(r"^(.*?)_y(\d+)_x(\d+)\." + re.escape(tile_format) + "$")

    # Use defaultdict to easily group files by prefix
    grouped_tiles = defaultdict(list)
    skipped_files = []

    print("Grouping tiles by prefix...")
    for tile_path in tqdm(all_tile_files, desc="Parsing filenames"):
        filename = os.path.basename(tile_path)
        match = prefix_coord_pattern.match(filename) # Use match (start of string)
        if match:
            prefix = match.group(1)
            y_coord = int(match.group(2))
            x_coord = int(match.group(3))
            # Store relevant info for each tile in its group
            grouped_tiles[prefix].append({
                'path': tile_path,
                'x': x_coord,
                'y': y_coord
            })
        else:
            skipped_files.append(filename)

    if not grouped_tiles:
        print("Error: No tile files matched the expected naming pattern (prefix_y#####_x#####.ext).")
        if skipped_files:
            print(f"  ({len(skipped_files)} files were skipped, e.g., '{skipped_files[0]}')")
        exit(1)

    if skipped_files:
         print(f"Warning: Skipped {len(skipped_files)} files that didn't match naming pattern (e.g., '{skipped_files[0]}')")

    print(f"Found {len(grouped_tiles)} unique prefixes (images) to process.")

    # --- Process Each Group (Each Original Image) ---
    total_stitched_images = 0
    for prefix, tiles_info in grouped_tiles.items():
        print(f"\n--- Processing Prefix: '{prefix}' ({len(tiles_info)} tiles) ---")

        # --- Determine properties for THIS group ---
        max_x, max_y = 0, 0
        first_tile_shape = None
        first_tile_dtype = None
        valid_tiles_in_group = [] # Keep track of tiles we can actually read

        for tile_info in tiles_info:
            tile_path = tile_info['path']
            x_coord = tile_info['x']
            y_coord = tile_info['y']
            max_x = max(max_x, x_coord)
            max_y = max(max_y, y_coord)

            # Read the first valid tile in this group to get shape/dtype
            if first_tile_shape is None:
                try:
                    temp_data = None
                    if tile_format == 'tif':
                        with tifffile.TiffFile(tile_path) as tif:
                            # Check if file contains valid pages before reading
                            if tif.pages and tif.pages[0].shape:
                                temp_data = tif.asarray(key=0) # Read first page/series
                            else:
                                print(f"  Warning: TIF file seems empty or invalid: {os.path.basename(tile_path)}")
                                continue # Skip to next tile in group
                    elif tile_format == 'png':
                        with Image.open(tile_path) as img:
                            temp_data = np.array(img)

                    if temp_data is not None:
                        # Basic validation of shape (at least 2D)
                        if temp_data.ndim >= 2 and temp_data.shape[0] > 0 and temp_data.shape[1] > 0 :
                            first_tile_shape = temp_data.shape
                            first_tile_dtype = temp_data.dtype
                            print(f"  Detected Tile Shape: {first_tile_shape}, Data Type: {first_tile_dtype}")
                            # Verify against parameters (optional but recommended)
                            if not (first_tile_shape[0] == tile_height and first_tile_shape[1] == tile_width):
                                print(f"  Warning: Tile dims {first_tile_shape[:2]} differ from params ({tile_height}x{tile_width}). Using parameters.")
                                # Adjust expected shape based on parameters if needed
                                if len(first_tile_shape) == 3: # RGB
                                     first_tile_shape = (tile_height, tile_width, first_tile_shape[2])
                                else: # Grayscale
                                     first_tile_shape = (tile_height, tile_width)
                            valid_tiles_in_group.append(tile_info) # Add this valid tile
                        else:
                            print(f"  Warning: Tile has invalid dimensions {temp_data.shape}: {os.path.basename(tile_path)}")
                    else:
                        # Reading failed or format issue handled inside try block
                        pass

                except UnidentifiedImageError:
                     print(f"  Warning: Cannot identify image file (may be corrupt): {os.path.basename(tile_path)}")
                except MemoryError:
                     print(f"  Error: Memory Error reading tile {os.path.basename(tile_path)} for shape check.")
                     # Maybe exit or just skip group? Let's skip group.
                     print(f"  Skipping prefix '{prefix}' due to memory error on first tile read.")
                     break # Break inner loop, go to next prefix
                except Exception as e:
                    print(f"  Warning: Error reading tile '{os.path.basename(tile_path)}' for shape check: {e}")
            else:
                 valid_tiles_in_group.append(tile_info) # Add subsequent tiles

        # Check if we have valid tiles and properties for this group
        if first_tile_shape is None or not valid_tiles_in_group:
            print(f"  Error: Could not determine tile properties or find valid tiles for prefix '{prefix}'. Skipping this group.")
            continue # Skip to the next prefix

        # --- Determine Output Image Dimensions for THIS group ---
        output_height = max_y + tile_height
        output_width = max_x + tile_width
        num_channels = first_tile_shape[2] if len(first_tile_shape) == 3 else 1
        is_rgb = num_channels > 1

        print(f"  Inferred Output Dimensions (WxH): {output_width}x{output_height}, Channels: {num_channels}")

        # --- Initialize Canvas and Weight Map for THIS group ---
        canvas_dtype = np.float64
        output_shape = (output_height, output_width, num_channels) if is_rgb else (output_height, output_width)
        try:
            canvas = np.zeros(output_shape, dtype=canvas_dtype)
            weight_map = np.zeros((output_height, output_width), dtype=np.float32)
        except MemoryError:
            print(f"  Error: Not enough memory to create canvas ({output_shape}) for prefix '{prefix}'. Skipping.")
            continue
        except Exception as e:
            print(f"  Error creating canvas/weight map for '{prefix}': {e}. Skipping.")
            continue

        print(f"  Initialized canvas ({canvas.shape}, {canvas.dtype})")

        # --- Generate the blending weights ONCE per group (same pattern) ---
        tile_weight_pattern = generate_tile_weights(tile_height, tile_width, overlap_y, overlap_x, mode=weight_mode)
        tile_weight_pattern_broadcast = tile_weight_pattern[:, :, np.newaxis] if is_rgb else tile_weight_pattern

        # --- Place Tiles onto Canvas with Weighted Averaging ---
        error_in_placement = False
        for tile_info in tqdm(valid_tiles_in_group, desc=f"  Placing '{prefix}' tiles"):
            x, y = tile_info['x'], tile_info['y']
            tile_path = tile_info['path']

            try:
                # Read tile data
                tile_data = None
                if tile_format == 'tif':
                    # Consider reading only if shape matches? Maybe too slow. Read then check.
                    tile_data = tifffile.imread(tile_path)
                elif tile_format == 'png':
                     with Image.open(tile_path) as img:
                          tile_data = np.array(img)

                if tile_data is None:
                    print(f"\n    Warning: Failed to read tile data (previously checked OK?): {os.path.basename(tile_path)}. Skipping placement.")
                    continue

                # Check dimensions again (paranoid check)
                current_tile_shape = tile_data.shape
                expected_h, expected_w = tile_height, tile_width
                if current_tile_shape[0] != expected_h or current_tile_shape[1] != expected_w:
                     print(f"\n    Warning: Tile {os.path.basename(tile_path)} has unexpected dims {current_tile_shape[:2]} during placement. Skipping.")
                     continue

                # Define slices for canvas and weight map
                y_slice = slice(y, y + tile_height)
                x_slice = slice(x, x + tile_width)

                # Add weighted tile data to canvas
                canvas[y_slice, x_slice] += tile_data.astype(canvas_dtype) * tile_weight_pattern_broadcast
                # Add weights to the weight map
                weight_map[y_slice, x_slice] += tile_weight_pattern

            except MemoryError:
                 print(f"\n    Error: Memory Error while placing tile {os.path.basename(tile_path)}. Stopping group '{prefix}'.")
                 error_in_placement = True; break
            except UnidentifiedImageError:
                 print(f"\n    Warning: Pillow could not identify tile image {os.path.basename(tile_path)} during placement. Skipping.")
            except Exception as e:
                 print(f"\n    Error placing tile {os.path.basename(tile_path)}: {e}")
                 # Decide whether to continue or stop group? Let's continue placement if possible.

        if error_in_placement:
             print(f"  Skipping normalization and saving for '{prefix}' due to placement error.")
             del canvas, weight_map # Try to free memory
             continue

        # --- Normalize Canvas ---
        print(f"  Normalizing canvas for '{prefix}'...")
        weight_map_nonzero = weight_map > 1e-6
        final_image_float = np.zeros_like(canvas, dtype=canvas_dtype)

        try:
            if is_rgb:
                weight_map_broadcast = weight_map[:, :, np.newaxis]
                weight_map_nonzero_broadcast = weight_map_nonzero[:, :, np.newaxis]
                np.divide(canvas, weight_map_broadcast, where=weight_map_nonzero_broadcast, out=final_image_float)
            else:
                np.divide(canvas, weight_map, where=weight_map_nonzero, out=final_image_float)
            del canvas, weight_map # Free memory sooner
        except MemoryError:
             print(f"  Error: Memory Error during normalization for '{prefix}'. Skipping save.")
             continue
        except Exception as e:
             print(f"  Error during normalization for '{prefix}': {e}. Skipping save.")
             continue


        # --- Convert to Target Data Type and Save ---
        target_dtype = force_output_dtype if force_output_dtype is not None else first_tile_dtype
        output_filename = f"{prefix}_reconstructed.{save_format}" # Generate output name
        output_path = os.path.join(output_dir, output_filename)
        print(f"  Converting and saving '{prefix}' to: {output_path} (dtype: {target_dtype})")

        try:
            final_image = None
            if np.issubdtype(target_dtype, np.integer):
                max_val = np.iinfo(target_dtype).max
                final_image = np.clip(final_image_float, 0, max_val).astype(target_dtype)
            elif np.issubdtype(target_dtype, np.floating):
                final_image = final_image_float.astype(target_dtype)
            else:
                 print(f"    Warning: Unknown target dtype '{target_dtype}'. Saving as float64.")
                 final_image = final_image_float

            if save_format == 'tif':
                tifffile.imwrite(output_path, final_image, imagej=True)
            elif save_format == 'png':
                # Prepare for PIL save (handle single channel, ensure uint8/16)
                if final_image.ndim == 3 and final_image.shape[2] == 1:
                     final_image_pil = final_image[:,:,0] # Reshape to 2D
                else:
                     final_image_pil = final_image

                if final_image_pil.dtype not in [np.uint8, np.uint16]:
                    print(f"    Warning: PNG save needs uint8/uint16. Current: {final_image_pil.dtype}. Converting to uint8.")
                    if np.issubdtype(final_image_pil.dtype, np.floating):
                        max_img_val = np.nanmax(final_image_pil) # Use nanmax
                        scale = 255.0 if max_img_val <= 1.1 else 1.0 # Adjust scale based on range
                        final_image_pil = np.clip(final_image_pil * scale, 0, 255)
                    elif np.issubdtype(final_image_pil.dtype, np.integer) and np.iinfo(final_image_pil.dtype).max > 255:
                         # Scale larger integers (e.g., uint16) down to uint8 if needed
                         final_image_pil = (final_image_pil / np.iinfo(final_image_pil.dtype).max * 255).astype(np.uint8)

                    final_image_pil = final_image_pil.astype(np.uint8)

                img_out = Image.fromarray(final_image_pil)
                img_out.save(output_path)
            else:
                 print(f"    Error: Unsupported save format '{save_format}'.")
                 continue # Skip saving this image

            total_stitched_images += 1
            print(f"  Successfully saved: {output_path}")

        except MemoryError:
             print(f"\n    Error: Memory Error during final conversion/saving for {prefix}.")
        except Exception as e:
            print(f"\n    Error during final conversion or saving for {prefix}: {e}")

        finally:
             # Explicitly delete large arrays if they might still exist
             del final_image_float
             if 'final_image' in locals(): del final_image
             if 'final_image_pil' in locals(): del final_image_pil


    # --- Final Summary ---
    print("\n" + "="*50)
    print("--- Batch Stitching Complete ---")
    print(f"Processed {len(grouped_tiles)} unique prefixes.")
    print(f"Successfully stitched and saved {total_stitched_images} images.")
    print(f"Output saved to directory: {output_dir}")
    print("="*50)
