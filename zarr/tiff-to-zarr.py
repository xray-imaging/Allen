import os
import shutil
import numpy as np
import zarr
import click
import tifffile as tiff
import json
from skimage.transform import resize
from numcodecs import Blosc

def load_tiff_chunked(input_dir, dtype, chunk_size, start_index=0):
    """
    Load a chunk of TIFF images from the specified directory.

    Parameters:
    - input_dir (str): Directory containing TIFF files.
    - dtype (numpy.dtype): Data type for the images.
    - chunk_size (int): Number of images to load per chunk.
    - start_index (int): Starting index for loading the chunk.

    Returns:
    - tuple: (zarr.core.Array, int) Loaded TIFF chunk and next starting index.
    """
    file_list = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tiff')])
    end_index = min(start_index + chunk_size, len(file_list))
    chunk_files = file_list[start_index:end_index]
    sample_image = tiff.imread(chunk_files[0])
    chunk_shape = (len(chunk_files),) + sample_image.shape

    zarr_chunk = zarr.array(np.zeros(chunk_shape, dtype=dtype), chunks=(1,) + sample_image.shape)
    
    for i, file in enumerate(chunk_files):
        zarr_chunk[i] = tiff.imread(file).astype(dtype)
    
    print(f"Loaded TIFF chunk with shape: {zarr_chunk.shape}, dtype: {zarr_chunk.dtype}")
    return zarr_chunk, end_index

def downsample(data, scale_factor=2, max_levels=6):
    """
    Generate a pyramid of downsampled versions of the image stack.

    Parameters:
    - data (zarr.core.Array): Original image stack.
    - scale_factor (int, optional): Factor by which to downsample. Default is 2.
    - max_levels (int, optional): Maximum number of downsampled levels. Default is 6.

    Returns:
    - list: List of downsampled image stacks.
    """
    current_level = data[:].astype(np.float32)  # Use float32 for processing to avoid clipping
    levels = [current_level]
    for _ in range(max_levels):
        new_shape = tuple(max(1, dim // scale_factor) for dim in current_level.shape)
        if min(new_shape) <= 1:
            break
        current_level = resize(current_level, new_shape, preserve_range=True, anti_aliasing=True)
        levels.append(current_level)
        print(f"Downsampled to shape: {current_level.shape}, dtype: {current_level.dtype}")
    return levels

def clip_data(data, dtype):
    """
    Clip data values based on the specified data type.

    Parameters:
    - data (numpy.ndarray): Data to be clipped.
    - dtype (numpy.dtype): Data type of the images.

    Returns:
    - numpy.ndarray: Clipped data.
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(data, info.min, info.max).astype(dtype)
    elif np.issubdtype(dtype, np.floating):
        return np.clip(data, 0.0, 1.0).astype(dtype)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def save_zarr(stack, output_path, chunks, compression, pixel_size, mode='w'):
    """
    Save the image stack and its downsampled versions to a Zarr file.

    Parameters:
    - stack (zarr.core.Array): Original image stack.
    - output_path (str): Path to save the Zarr file.
    - chunks (tuple): Chunk size for the Zarr array.
    - compression (str): Compression algorithm to use.
    - pixel_size (float): Pixel size in micrometers.
    - mode (str): Mode to open the Zarr file ('w' for write, 'a' for append).
    """
    store = zarr.DirectoryStore(output_path)
    compressor = Blosc(cname=compression, clevel=5, shuffle=2)

    if mode == 'w':
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        root_group = zarr.group(store=store)
    else:
        root_group = zarr.open(store=store, mode='a')

    pyramid_levels = downsample(stack)
    datasets = []

    for level, data in enumerate(pyramid_levels):
        data = clip_data(data, stack.dtype)  # Clip values based on dtype and convert back to original dtype
        dataset_name = f"{level}"
        if dataset_name in root_group:
            z = root_group[dataset_name]
            z.append(data, axis=0)
        else:
            z = root_group.create_dataset(name=dataset_name, shape=data.shape, chunks=chunks, dtype=data.dtype, compressor=compressor)
            z[:] = data
        scale_factor = 2 ** level
        datasets.append({
            "path": dataset_name,
            "coordinateTransformations": [{"type": "scale", "scale": [pixel_size * scale_factor, pixel_size * scale_factor, pixel_size * scale_factor]}]
        })

    if mode == 'w':
        multiscales = [{
            "version": "0.4",
            "name": "example",
            "axes": [
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"}
            ],
            "datasets": datasets,
            "type": "gaussian",
            "metadata": {
                "method": "skimage.transform.resize",
                "version": "0.16.1",
                "args": "[true]",
                "kwargs": {"anti_aliasing": True, "preserve_range": True}
            }
        }]

        root_group.attrs.update({"multiscales": multiscales})
        with open(os.path.join(output_path, 'multiscales.json'), 'w') as f:
            json.dump({"multiscales": multiscales}, f, indent=2)
        print(f"Metadata saved to {output_path}")

@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--dtype', type=click.Choice(['int8', 'int16', 'int32', 'uint8', 'uint16']), default='uint8', help='Data type of the images.')
@click.option('--chunks', type=(int, int, int), default=(64, 64, 64), help='Chunk size for the Zarr array.')
@click.option('--compression', type=click.Choice(['blosclz', 'lz4', 'lz4hc', 'zlib', 'zstd']), default='blosclz', help='Compression algorithm to use.')
@click.option('--pixel_size', type=float, default=1.0, help='Pixel size in micrometers.')
@click.option('--chunk_size', type=int, default=10, help='Chunk size for loading TIFF images.')
def main(input_dir, output_path, dtype, chunks, compression, pixel_size, chunk_size):
    """
    Main function to load TIFF images, downsample them, and save to a Zarr file.

    Parameters:
    - input_dir (str): Directory containing TIFF files.
    - output_path (str): Path to save the Zarr file.
    - dtype (str): Data type of the images.
    - chunks (tuple): Chunk size for the Zarr array.
    - compression (str): Compression algorithm to use.
    - pixel_size (float): Pixel size in micrometers.
    - chunk_size (int): Chunk size for loading TIFF images.
    """
    dtype_map = {'int8': np.int8, 'int16': np.int16, 'int32': np.int32, 'uint8': np.uint8, 'uint16': np.uint16}
    start_index = 0
    mode = 'w'
    
    while True:
        stack, start_index = load_tiff_chunked(input_dir, dtype_map[dtype], chunk_size, start_index)
        save_zarr(stack, output_path, chunks, compression, pixel_size, mode)
        mode = 'a'
        if start_index >= len(os.listdir(input_dir)):
            break

if __name__ == '__main__':
    main()

