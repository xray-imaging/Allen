import os
import numpy as np
import tifffile as tiff
from skimage.transform import resize
from log import info

def calculate_global_min_max(input_dir, bin_factor=4):
    """
    Calculate the global minimum and maximum pixel values of all TIFF images in a directory after downsampling (binning).

    Parameters:
    - input_dir (str): Path to the directory containing TIFF images.
    - bin_factor (int, optional): Factor by which to downsample the images before calculating min and max values. Default is 4.

    Returns:
    - global_min (float): The minimum pixel value across all images.
    - global_max (float): The maximum pixel value across all images.

    Raises:
    - ValueError: If no TIFF files are found in the directory.
    """
    file_list = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.tiff','.tif'))])
    if not file_list:
        raise ValueError(f"No TIFF files found in the directory: {input_dir}")

    global_min = np.inf
    global_max = -np.inf
    
    for file in file_list:
        image = tiff.imread(file)
        binned_image = resize(image, 
                              (image.shape[0] // bin_factor, image.shape[1] // bin_factor), 
                              order=1, preserve_range=True, anti_aliasing=False).astype(image.dtype)
        global_min = min(global_min, binned_image.min())
        global_max = max(global_max, binned_image.max())
    
    info(f"Global min and max found: {global_min}, dtype: {global_max}")
    return global_min, global_max

def load_tiff_chunked(input_dir, dtype, chunk_size, start_index=0, global_min=None, global_max=None):
    """
    Load TIFF images from a directory in chunks and convert them to a specified data type, optionally normalizing the values.

    Parameters:
    - input_dir (str): Path to the directory containing TIFF images.
    - dtype (numpy dtype): Target data type for the output array.
    - chunk_size (int): Number of images to load in each chunk.
    - start_index (int, optional): Starting index for loading images. Default is 0.
    - global_min (float, optional): Minimum pixel value for normalization. If None, no normalization is performed. Default is None.
    - global_max (float, optional): Maximum pixel value for normalization. If None, no normalization is performed. Default is None.

    Returns:
    - zarr_chunk (numpy array): A chunk of loaded images converted to the specified data type.
    - end_index (int): The end index for the current chunk of images.

    Raises:
    - ValueError: If no TIFF files are found in the directory.
    """
    file_list = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.tiff','.tif'))])
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
        image = tiff.imread(file)
        if global_min is not None and global_max is not None:
            image = (image - global_min) / (global_max - global_min)  # Normalize to [0, 1]
            if dtype == np.uint16 or dtype == np.int16:
                image = image * (2**15 - 1)  # Scale to int16 range
            elif dtype == np.uint8 or dtype == np.int8:
                image = image * (2**8 - 1)  # Scale to int8 range

        zarr_chunk[i] = image.astype(dtype)
        
    info(f"Loaded TIFF chunk with shape: {zarr_chunk.shape}, dtype: {zarr_chunk.dtype}")
    return zarr_chunk, end_index

def downsample(data, scale_factor=2, max_levels=6):
    """
    Create a multi-level downsampled version of the input data.

    Parameters:
    - data (numpy array): The input image data to be downsampled.
    - scale_factor (int, optional): Factor by which to downsample the data at each level. Default is 2.
    - max_levels (int, optional): Maximum number of downsampled levels to generate. Default is 6.

    Returns:
    - levels (list of numpy arrays): A list containing the original data and each downsampled level.

    Logs:
    - Information about the shape and data type of each downsampled level.
    """
    current_level = data
    levels = [current_level]
    for _ in range(max_levels):
        new_shape = tuple(max(1, dim // scale_factor) for dim in current_level.shape)
        if min(new_shape) <= 1:
            break
        current_level = resize(current_level, new_shape, order=0, preserve_range=True, anti_aliasing=True)
        levels.append(current_level)
        info(f"Downsampled to shape: {current_level.shape}, dtype: {current_level.dtype}")
    return levels

