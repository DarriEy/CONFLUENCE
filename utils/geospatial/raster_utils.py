# In utils/geospatial_utils/raster_utils.py

import numpy as np
import rasterio
from scipy import stats
from pathlib import Path
from typing import Union


def calculate_landcover_mode(input_dir: Path, output_file: Path, start_year: int, end_year: int, domain_name: str):
    """
    Calculate the mode of landcover data across multiple years.
    
    Args:
        input_dir: Directory containing input geotiff files
        output_file: Path for output file
        start_year: Starting year for calculation
        end_year: Ending year for calculation
        domain_name: Name of the domain
    """
    # List all the geotiff files for the years we're interested in
    geotiff_files = [input_dir / f"domain_{domain_name}_{year}.tif" for year in range(start_year, end_year + 1)]
    
    # Read the first file to get metadata
    with rasterio.open(geotiff_files[0]) as src:
        meta = src.meta
        shape = src.shape
    
    # Initialize an array to store all the data
    all_data = np.zeros((len(geotiff_files), *shape), dtype=np.int16)
    
    # Read all the geotiffs into the array
    for i, file in enumerate(geotiff_files):
        with rasterio.open(file) as src:
            all_data[i] = src.read(1)
    
    # Calculate the mode along the time axis
    mode_data, _ = stats.mode(all_data, axis=0)
    mode_data = mode_data.astype(np.int16).squeeze()
    
    # Update metadata for output
    meta.update(count=1, dtype='int16')
    
    # Write the result
    with rasterio.open(output_file, 'w', **meta) as dst:
        dst.write(mode_data, 1)
    
    print(f"Mode calculation complete. Result saved to {output_file}")