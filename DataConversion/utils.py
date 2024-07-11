import os
import numpy as np
import tifffile as tiff
from skimage.transform import resize
from log import info

def calculate_global_min_max(input_dir):
    file_list = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tiff')])
    if not file_list:
        raise ValueError(f"No TIFF files found in the directory: {input_dir}")

    global_min, global_max = np.inf, -np.inf

    for file in file_list:
        image = tiff.imread(file).astype(np.float32)
        global_min = min(global_min, image.min())
        global_max = max(global_max, image.max())

    info(f"Global min: {global_min}, Global max: {global_max}")
    return global_min, global_max

def load_tiff_chunked(input_dir, dtype, chunk_size, start_index=0, global_min=None, global_max=None):
    file_list = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tiff')])
    if not file_list:
        raise ValueError(f"No TIFF files found in the directory: {input_dir}")

    end_index = min(start_index + chunk_size, len(file_list))
    chunk_files = file_list[start_index:end_index]
    if not chunk_files:
        return None, end_index

    sample_image = tiff.imread(chunk_files[0])
    chunk_shape = (len(chunk_files),) + sample_image.shape

    zarr_chunk = np.zeros(chunk_shape, dtype=dtype)
    
    for i, file in enumerate(chunk_files):
        image = tiff.imread(file).astype(np.float32)
        if global_min is not None and global_max is not None:
            image = (image - global_min) / (global_max - global_min)  # Normalize to [0, 1]
            image = image * (2**15 - 1)  # Scale to int16 range
        zarr_chunk[i] = image.astype(dtype)
    
    info(f"Loaded TIFF chunk with shape: {zarr_chunk.shape}, dtype: {zarr_chunk.dtype}")
    return zarr_chunk, end_index

def downsample(data, scale_factor=2, max_levels=6):
    current_level = data.astype(np.float32)
    levels = [current_level]
    for _ in range(max_levels):
        new_shape = tuple(max(1, dim // scale_factor) for dim in current_level.shape)
        if min(new_shape) <= 1:
            break
        current_level = resize(current_level, new_shape, preserve_range=True, anti_aliasing=True)
        levels.append(current_level)
        info(f"Downsampled to shape: {current_level.shape}, dtype: {current_level.dtype}")
    return levels

