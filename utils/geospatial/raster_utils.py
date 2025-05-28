# In utils/geospatial_utils/raster_utils.py

import numpy as np
import rasterio # type: ignore
from scipy import stats
import glob
import numpy as np
import rasterio # type: ignore
from scipy import stats


def calculate_landcover_mode(input_dir, output_file, start_year, end_year, domain_name):
    """
    Calculate the mode of land cover data across multiple years.
    
    Args:
        input_dir (Path): Directory containing the yearly land cover files
        output_file (Path): Path to save the output mode raster
        start_year (int): Start year for mode calculation
        end_year (int): End year for mode calculation
        domain_name (str): Name of the domain
    """
    
    # Create a list to store the data from each year
    yearly_data = []
    meta = None
    
    # Get a list of files matching the pattern for the specified years
    file_pattern = f"{input_dir}/domain_{domain_name}_*_{start_year}*.tif"
    files = glob.glob(str(file_pattern))
    
    if not files:
        # If no files match the start year, try to find any files in the directory
        file_pattern = f"{input_dir}/domain_{domain_name}_*.tif"
        files = glob.glob(str(file_pattern))
    
    if not files:
        raise FileNotFoundError(f"No land cover files found matching pattern: {file_pattern}")
    
    # Read metadata from the first file
    with rasterio.open(files[0]) as src:
        meta = src.meta.copy()
        shape = (src.height, src.width)
    
    # Read data for each year
    for year in range(start_year, end_year + 1):
        pattern = f"{input_dir}/domain_{domain_name}_*_{year}*.tif"
        year_files = glob.glob(str(pattern))
        
        if year_files:
            with rasterio.open(year_files[0]) as src:
                # Read the data and append to our list
                data = src.read(1)
                yearly_data.append(data)
    
    if not yearly_data:
        # If no yearly data was found, use the first file we found
        with rasterio.open(files[0]) as src:
            data = src.read(1)
            yearly_data.append(data)
    
    # Check if we have only one year of data
    if len(yearly_data) == 1:
        # Just use that single year's data
        mode_data = yearly_data[0]
    else:
        # Stack the arrays
        stacked_data = np.stack(yearly_data, axis=0)
        
        # Calculate the mode along the year axis (axis=0)
        # Using scipy.stats.mode with keepdims=False for newer scipy versions
        try:
            mode_data, _ = stats.mode(stacked_data, axis=0, keepdims=False)
        except TypeError:
            # For older scipy versions that don't have keepdims parameter
            mode_result = stats.mode(stacked_data, axis=0)
            mode_data = mode_result[0][0]  # Extract the mode values
    
    # Update the metadata for the output file
    meta.update({
        'count': 1,
        'nodata': 0
    })
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the result
    with rasterio.open(output_file, 'w', **meta) as dst:
        # Make sure mode_data has the right shape
        if mode_data.ndim == 1 or mode_data.shape != shape:
            # If the shape doesn't match, reshape it to the expected dimensions
            if mode_data.size == shape[0] * shape[1]:
                mode_data = mode_data.reshape(shape)
            else:
                # Create a new array with the correct shape
                new_mode_data = np.zeros(shape, dtype=meta['dtype'])
                
                # If mode_data is 1D but should be 2D
                if mode_data.ndim == 1:
                    # Take as many values as we can from mode_data
                    size = min(mode_data.size, shape[0] * shape[1])
                    new_mode_data.flat[:size] = mode_data[:size]
                    mode_data = new_mode_data
                else:
                    # If dimensions don't match but we can copy partial data
                    min_h = min(mode_data.shape[0], shape[0])
                    min_w = min(mode_data.shape[1], shape[1])
                    new_mode_data[:min_h, :min_w] = mode_data[:min_h, :min_w]
                    mode_data = new_mode_data
        
        # Now write the data
        dst.write(mode_data, 1)