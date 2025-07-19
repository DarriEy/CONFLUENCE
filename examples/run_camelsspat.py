import geopandas as gpd # type: ignore
import pandas as pd # type: ignore
import os
import subprocess
from pathlib import Path
import time
import re
import glob
import shutil
from shapely.geometry import Point # type: ignore

def setup_confluence_directory(watershed_id, basin_source_path, basin_filename, scale_name, river_source_path=None, river_filename=None):
    """
    Set up the CONFLUENCE directory structure for a watershed and copy relevant shapefiles.
    For CAMELS-SPAT implementation, uses lumped basins and distributed river networks.
    Also ensures that shapefiles have the required fields for CONFLUENCE.
    Now also processes CAMELS-SPAT observational streamflow data.
    
    Args:
        watershed_id: ID of the watershed
        basin_source_path: Path to the source basin shapefile directory
        basin_filename: Name of the basin shapefile
        river_source_path: Path to the source river shapefile directory (optional)
        river_filename: Name of the river shapefile (optional)
        
    Returns:
        Tuple of (basin_target_path, river_target_path)
    """        
    # Base CONFLUENCE data directory
    confluence_data_dir = Path("/anvil/scratch/x-deythorsson/CONFLUENCE_data")
    
    # Create the domain directory structure
    domain_dir = confluence_data_dir / "camels_spat" / f"domain_{watershed_id}"
    basin_target_dir = domain_dir / "shapefiles" / "river_basins"
    river_target_dir = domain_dir / "shapefiles" / "river_network"
    
    # Create directories if they don't exist
    for directory in [domain_dir, basin_target_dir, river_target_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Get the basin shapefile and associated files
    basin_source_base = str(Path(basin_source_path) / basin_filename).rsplit('.', 1)[0]
    basin_target_base = str(basin_target_dir / basin_filename).rsplit('.', 1)[0]
    

    # Copy all shapefile components (shp, shx, dbf, prj, cpg)
    for extension in ['shp', 'shx', 'dbf', 'prj', 'cpg']:
        # Copy basin files
        basin_source_file = f"{basin_source_base}.{extension}"
        basin_target_file = f"{basin_target_base}.{extension}"
        if os.path.exists(basin_source_file):
            shutil.copy2(basin_source_file, basin_target_file)
            print(f"Copied {basin_source_file} to {basin_target_file}")
        
        # Copy river files if they exist
        if river_source_path and river_filename and isinstance(river_filename, str) and river_filename.strip():
            #river_source_base = str(Path(river_source_path) / river_filename).rsplit('.', 1)[0]
            #river_target_base = str(river_target_dir / river_filename).rsplit('.', 1)[0]
            #river_source_file = f"{river_source_base}.{extension}"
            #river_target_file = f"{river_target_base}.{extension}"
            
            #if os.path.exists(river_source_file):
                #shutil.copy2(river_source_file, river_target_file)
                #print(f"Copied {river_source_file} to {river_target_file}")
            pass  # River file copying is currently disabled
    
    # Now modify the shapefiles to add required attributes
    try:
        # Read the basin shapefile
        basin_shp = gpd.read_file(f"{basin_target_base}.shp")
        
        print(f"Original basin shapefile columns: {basin_shp.columns.tolist()}")
        
        # For CAMELS-SPAT lumped basins, we need to create a simple structure
        # Add GRU_ID - for lumped basins, this should be 1
        basin_shp['GRU_ID'] = 1
        basin_shp['gru_to_seg'] = 1  # Points to a single outlet segment
        print(f"Set GRU_ID and gru_to_seg to 1 for lumped basin")
        
        # Calculate GRU_area in square meters
        if basin_shp.crs is None:
            print(f"Warning: CRS not defined for {basin_filename}. Trying to set to EPSG:4326 (WGS84).")
            basin_shp.set_crs(epsg=4326, inplace=True)
        
        # Convert to equal area projection for accurate area calculation
        basin_shp_ea = basin_shp.to_crs('+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96')
        basin_shp['GRU_area'] = basin_shp_ea.geometry.area
        print(f"Calculated GRU_area: {basin_shp['GRU_area'].iloc[0]:.2f} m²")
        
        # Save the modified basin shapefile
        basin_shp.to_file(f"{basin_target_base}.shp")
        print(f"Added GRU_ID, GRU_area, and gru_to_seg columns to {basin_filename}")


        # Process river shapefile if it exists
        if river_source_path and river_filename and isinstance(river_filename, str) and river_filename.strip():
            river_target_base = str(river_target_dir / river_filename).rsplit('.', 1)[0]
            river_target_file = f"{river_target_base}.shp"
            
            if os.path.exists(river_target_file):
                print("Processing river shapefile...")
                
                # Read the river shapefile
                river_shp = gpd.read_file(river_target_file)
                
                print(f"Original river shapefile columns: {river_shp.columns.tolist()}")
                
                # Check if CRS is defined
                if river_shp.crs is None:
                    print(f"Warning: CRS not defined for {river_filename}. Trying to set to EPSG:4326 (WGS84).")
                    river_shp.set_crs(epsg=4326, inplace=True)
                
                # Get column mapping for source vs required columns
                column_mappings = {
                    'Length': None,  # Will calculate if not found
                    'LINKNO': None,  # Will look for alternatives
                    'DSLINKNO': None,  # Will look for alternatives
                    'Slope': None  # Will set default if not found
                }
                
                # Try to map existing columns to required ones
                existing_cols = river_shp.columns.tolist()
                
                # Look for LINKNO equivalents
                linkno_candidates = ['LINKNO', 'COMID', 'segmentID', 'FID', 'OBJECTID', 'ID']
                for candidate in linkno_candidates:
                    if candidate in existing_cols:
                        column_mappings['LINKNO'] = candidate
                        break
                
                # Look for DSLINKNO equivalents
                dslinkno_candidates = ['DSLINKNO', 'NextDownID', 'ToNode', 'DOWNSTREAM', 'TO_LINK']
                for candidate in dslinkno_candidates:
                    if candidate in existing_cols:
                        column_mappings['DSLINKNO'] = candidate
                        break
                
                # Look for Length equivalents
                length_candidates = ['Length', 'LENGTH', 'SHAPE_Leng', 'Shape_Length']
                for candidate in length_candidates:
                    if candidate in existing_cols:
                        column_mappings['Length'] = candidate
                        break
                
                # Look for Slope equivalents
                slope_candidates = ['Slope', 'SLOPE', 'slope', 'gradient']
                for candidate in slope_candidates:
                    if candidate in existing_cols:
                        column_mappings['Slope'] = candidate
                        break
                
                print("Column mappings found:", column_mappings)
                
                # Add missing fields based on mappings
                for target_field, source_field in column_mappings.items():
                    if target_field in river_shp.columns:
                        print(f"Field {target_field} already exists in river shapefile")
                        continue
                        
                    if source_field and source_field in river_shp.columns:
                        # Copy from source field
                        river_shp[target_field] = river_shp[source_field]
                        print(f"Copied values from {source_field} to {target_field}")
                    else:
                        # Initialize with default values
                        if target_field == 'Length':
                            # Calculate length for rivers
                            river_shp_proj = river_shp.to_crs('+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96')
                            river_shp[target_field] = river_shp_proj.geometry.length
                            print(f"Calculated {target_field} in meters from geometry")
                        elif target_field == 'Slope':
                            # Default to 0.001 slope (1 m/km) if not available
                            river_shp[target_field] = 0.001
                            print(f"Set default value 0.001 for {target_field}")
                        elif target_field == 'LINKNO':
                            # Create sequential ID if not available
                            river_shp[target_field] = range(1, len(river_shp) + 1)
                            print(f"Created sequential IDs for {target_field}")
                        elif target_field == 'DSLINKNO':
                            # Set to -1 (outlet) for all by default
                            # A more sophisticated approach would use flow direction/network analysis
                            river_shp[target_field] = -1
                            print(f"Set default value -1 for {target_field} (indicating outlets)")
                            
                            # For the last segment (presumably outlet), keep it as -1
                            # For other segments, try to create a simple downstream connectivity
                            if len(river_shp) > 1:
                                # Simple approach: each segment drains to the next one, except the last
                                for i in range(len(river_shp) - 1):
                                    river_shp.loc[i, 'DSLINKNO'] = river_shp.loc[i+1, 'LINKNO']
                                print(f"Created simple downstream connectivity")
                
                # Make sure the required columns have the right data types
                river_shp['Length'] = river_shp['Length'].astype(float)
                river_shp['LINKNO'] = river_shp['LINKNO'].astype(int)
                river_shp['DSLINKNO'] = river_shp['DSLINKNO'].astype(int)
                river_shp['Slope'] = river_shp['Slope'].astype(float)
                
                # Validate and fix data as needed
                # Ensure Length is non-zero (replace zeros with 1 meter to avoid errors)
                river_shp.loc[river_shp['Length'] == 0, 'Length'] = 1.0
                
                # Ensure Slope is non-zero (replace zeros with small value to avoid errors)
                river_shp.loc[river_shp['Slope'] == 0, 'Slope'] = 0.0001
                
                # Save the modified river shapefile
                river_shp.to_file(river_target_file)
                print(f"Modified river shapefile saved with all required fields")

        
    except Exception as e:
        print(f"Error modifying shapefile attributes: {e}")
        import traceback
        print(traceback.format_exc())
    
    # ========== NEW: Process CAMELS-SPAT Observational Data ==========
    print(f"\nProcessing CAMELS-SPAT observational data for {watershed_id}...")
    
    try:
        # Extract station ID from watershed_id (remove country prefix and scale suffix if present)
        station_id = watershed_id
        
        # Remove country prefix (e.g., "USA_" or "CAN_")
        if '_' in station_id:
            parts = station_id.split('_')
            if len(parts) >= 2 and parts[0] in ['USA', 'CAN']:
                station_id = '_'.join(parts[1:])  # Remove country prefix
        
        # Remove scale suffix if present (e.g., "_meso", "_macro", "_headwater")
        scale_suffixes = ['_meso', '_macro', '_headwater']
        for suffix in scale_suffixes:
            if station_id.endswith(suffix):
                station_id = station_id[:-len(suffix)]
                break
        
        print(f"Extracted station ID: {station_id}")
        
        # Process the observational data
        output_file = process_camels_spat_observations(
            station_id=station_id,
            domain_name=watershed_id,
            output_dir=domain_dir,
            scale_name=scale_name,
            camels_base_path=None,  # Use default path
            logger=None  # Use default logging
        )
        
        if output_file:
            print(f"✓ Successfully processed observational data: {output_file}")
        else:
            print(f"⚠ Warning: Could not process observational data for station {station_id}")
            
    except Exception as e:
        print(f"Error processing observational data: {e}")
        import traceback
        print(traceback.format_exc())
    
    # ================================================================
    
    return basin_target_dir, river_target_dir

def process_camels_spat_observations(station_id, domain_name, output_dir, scale_name, camels_base_path=None, logger=None):
    """
    Process CAMELS-spat observational streamflow data and convert to CONFLUENCE format.
    
    Args:
        station_id (str): Station ID (e.g., '05BB001' or '02140991')
        domain_name (str): Domain name for output file naming
        output_dir (str or Path): Directory to save processed data
        camels_base_path (str or Path, optional): Base path to CAMELS data. 
            Defaults to "/work/comphyd_lab/data/_to-be-moved/camels-spat-upload/observations/meso-scale/obs-daily"
        logger (logging.Logger, optional): Logger instance for messages
    
    Returns:
        str: Path to the saved processed file, or None if processing failed
    """
    import xarray as xr
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    # Set up logging
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    # Set default CAMELS base path if not provided
    if camels_base_path is None:
        camels_base_path = Path(f"/home/x-deythorsson/data/camels-spat-upload/observations/{scale_name}/obs-daily")
    else:
        camels_base_path = Path(camels_base_path)
    
    # Convert output_dir to Path
    output_dir = Path(output_dir)
    
    if not station_id:
        logger.warning("No station_id provided - skipping observation processing")
        return None
    
    # Determine country prefix based on station ID format
    if station_id.isdigit():
        country_prefix = 'USA'
    else:
        country_prefix = 'CAN'
    
    # Construct the expected filename
    obs_filename = f"{country_prefix}_{station_id}_daily_flow_observations.nc"
    obs_file_path = camels_base_path / obs_filename
    
    if not obs_file_path.exists():
        logger.warning(f"Observation file not found: {obs_file_path}")
        return None
    
    try:
        logger.info(f"Processing observation data from: {obs_file_path}")
        
        # Load the NetCDF file
        ds = xr.open_dataset(obs_file_path)
        
        # Extract streamflow data
        q_obs = ds.q_obs.values  # Already in m³/s (same as cms)
        time_daily = pd.to_datetime(ds.time.values)
        
        # Create daily DataFrame
        df_daily = pd.DataFrame({
            'datetime': time_daily,
            'discharge_cms': q_obs
        })
        
        # Remove any NaN values
        df_daily = df_daily.dropna()
        
        if len(df_daily) == 0:
            logger.warning("No valid data found after removing NaN values")
            ds.close()
            return None
        
        # Convert daily data to hourly through interpolation
        # Create hourly time index
        start_time = df_daily['datetime'].min()
        end_time = df_daily['datetime'].max() + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
        hourly_index = pd.date_range(start=start_time, end=end_time, freq='h')
        
        # Interpolate to hourly values
        df_daily_indexed = df_daily.set_index('datetime')
        df_hourly = df_daily_indexed.reindex(
            df_daily_indexed.index.union(hourly_index)
        ).interpolate(method='time').reindex(hourly_index)
        
        # Reset index to get datetime as column
        df_hourly = df_hourly.reset_index()
        df_hourly.columns = ['datetime', 'discharge_cms']
        
        # Ensure proper datetime formatting
        df_hourly['datetime'] = df_hourly['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Create output directory structure
        streamflow_dir = output_dir / "observations" / "streamflow" / "preprocessed"
        streamflow_dir.mkdir(parents=True, exist_ok=True)
        
        # Save processed data
        output_file = streamflow_dir / f"{domain_name}_streamflow_processed.csv"
        df_hourly.to_csv(output_file, index=False)
        
        logger.info(f"Processed observation data saved to: {output_file}")
        logger.info(f"Data range: {df_hourly['datetime'].iloc[0]} to {df_hourly['datetime'].iloc[-1]}")
        logger.info(f"Total records: {len(df_hourly)}")
        
        # Close the dataset
        ds.close()
        
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Error processing observation data: {str(e)}")
        if 'ds' in locals():
            ds.close()
        return None


# Example usage function you can call from your main script
def setup_camels_observations_for_confluence(config, project_dir, logger=None):
    """
    Wrapper function to process CAMELS observations using CONFLUENCE config.
    
    Args:
        config (dict): CONFLUENCE configuration dictionary
        project_dir (str or Path): Project directory path
        logger (logging.Logger, optional): Logger instance
    
    Returns:
        str: Path to processed file or None if failed
    """
    station_id = config.get('STATION_ID')
    domain_name = config.get('DOMAIN_NAME')
    
    if not station_id or not domain_name:
        if logger:
            logger.warning("Missing STATION_ID or DOMAIN_NAME in config")
        return None
    
    return process_camels_spat_observations(
        station_id=station_id,
        domain_name=domain_name,
        output_dir=project_dir,
        logger=logger
    )

def process_camels_spat_metadata(metadata_df):
    """
    Process CAMELS-SPAT metadata to create the expected columns for CONFLUENCE
    
    Args:
        metadata_df: Raw CAMELS-SPAT metadata DataFrame
        
    Returns:
        Processed metadata DataFrame with standardized columns
    """
    # Create a copy to avoid modifying the original
    processed_metadata = metadata_df.copy()
    
    # Create ID column from Station_id
    if 'Station_id' in processed_metadata.columns:
        processed_metadata['ID'] = processed_metadata['Station_id'].astype(str)
        print("Created ID column from Station_id")
    else:
        print("Warning: Station_id column not found in metadata")
        return None
    
    # Create POUR_POINT_COORDS from Station_lat and Station_lon
    if ('Station_lat' in processed_metadata.columns and 
        'Station_lon' in processed_metadata.columns):
        
        # Create coordinate string in lat/lon format for all rows
        pour_point_coords = []
        valid_coords = 0
        
        for _, row in processed_metadata.iterrows():
            if (not pd.isna(row['Station_lat']) and 
                not pd.isna(row['Station_lon']) and
                row['Station_lat'] != 0 and row['Station_lon'] != 0):
                
                coords = f"{row['Station_lat']}/{row['Station_lon']}"
                pour_point_coords.append(coords)
                valid_coords += 1
            else:
                pour_point_coords.append(None)
        
        processed_metadata['POUR_POINT_COORDS'] = pour_point_coords
        print(f"Created POUR_POINT_COORDS from Station coordinates for {valid_coords}/{len(processed_metadata)} watersheds")
        
    else:
        print("Warning: Station_lat and/or Station_lon columns not found in metadata")
        processed_metadata['POUR_POINT_COORDS'] = None
    
    return processed_metadata

def calculate_fresh_bounding_box(shapefile_path):
    """
    Calculate a bounding box directly from the shapefile
    
    Args:
        shapefile_path: Path to shapefile
        
    Returns:
        String with bounding box in lat_max/lon_min/lat_min/lon_max format
    """
    try:        
        # Read the shapefile
        gdf = gpd.read_file(shapefile_path)
        
        # Ensure it's in lat/lon coordinate system
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        
        # Get bounds (minx, miny, maxx, maxy)
        bounds = gdf.total_bounds
        
        # Convert to CONFLUENCE format (lat_max/lon_min/lat_min/lon_max)
        bounding_box = f"{bounds[3]}/{bounds[0]}/{bounds[1]}/{bounds[2]}"
        return bounding_box
    except Exception as e:
        print(f"Error calculating bounding box: {e}")
        return None

def validate_coords(coords):
    """
    Validate coordinate string
    
    Args:
        coords: String with coordinates
        
    Returns:
        Boolean indicating if coordinates are valid
    """
    if coords is None:
        return False
    
    # Convert to string to handle any type issues
    coords_str = str(coords).strip()
    
    # Check if it's NaN or empty
    if coords_str.lower() == 'nan' or coords_str == '' or coords_str == '-999/-999':
        return False
    
    # Check if format is likely valid (contains at least one / and numbers)
    if '/' not in coords_str:
        return False
    
    # Split and check both parts are numeric
    try:
        lat, lon = coords_str.split('/')
        float(lat)
        float(lon)
        return True
    except (ValueError, TypeError):
        return False

def generate_distributed_config_file(template_path, output_path, domain_name, basin_path, basin_name, 
                                    river_network_path, river_network_name, bounding_box=None, pour_point=None,
                                    station_id=None, sim_reach_id=None):
    """
    Generate a new config file based on the template with updated parameters for distributed basins
    
    Args:
        template_path: Path to the template config file
        output_path: Path to save the new config file
        domain_name: Name of the domain to set
        basin_path: Path to the river basin shapefile
        basin_name: Name of the river basin shapefile
        river_network_path: Path to the river network shapefile
        river_network_name: Name of the river network shapefile
        bounding_box: Optional bounding box coordinates
        pour_point: Optional pour point coordinates
        station_id: Optional station ID for streamflow data
        sim_reach_id: Optional simulation reach ID for evaluation
    """
    # Read the template config file
    with open(template_path, 'r') as f:
        config_content = f.read()
    
    # Update the domain name using regex
    config_content = re.sub(r'DOMAIN_NAME:.*', f'DOMAIN_NAME: "{domain_name}"', config_content)

    # Update the data directory 
    config_content = re.sub(r'CONFLUENCE_DATA_DIR:.*', f'CONFLUENCE_DATA_DIR: /anvil/scratch/x-deythorsson/CONFLUENCE_data/camels_spat', config_content)

    # Update the river basin name
    config_content = re.sub(r'RIVER_BASINS_NAME:.*', f'RIVER_BASINS_NAME: "{basin_name}"', config_content)
    
    # Update the river network name
    config_content = re.sub(r'RIVER_NETWORK_SHP_NAME:.*', f'RIVER_NETWORK_SHP_NAME: "{river_network_name}"', config_content)
    
    # Update domain definition method to use delineate
    config_content = re.sub(r'DOMAIN_DEFINITION_METHOD:.*', f'DOMAIN_DEFINITION_METHOD: lumped', config_content)
    
    # Update pour point coordinates if provided and valid
    if pour_point and str(pour_point).lower() != 'nan' and '/' in str(pour_point):
        config_content = re.sub(r'POUR_POINT_COORDS:.*', f'POUR_POINT_COORDS: {pour_point}', config_content)
    
    # Update bounding box coordinates if provided and valid
    if bounding_box and str(bounding_box).lower() != 'nan' and '/' in str(bounding_box):
        config_content = re.sub(r'BOUNDING_BOX_COORDS:.*', f'BOUNDING_BOX_COORDS: {bounding_box}', config_content)
    
    # Determine data provider based on domain_name prefix
    if domain_name.startswith('CAN_'):
        data_provider = 'WSC'
        download_wsc = 'True'
        download_usgs = 'False'
    elif domain_name.startswith('USA_'):
        data_provider = 'USGS'
        download_wsc = 'False'
        download_usgs = 'True'
    
    # Update the streamflow data provider
    config_content = re.sub(r'STREAMFLOW_DATA_PROVIDER:.*', f'STREAMFLOW_DATA_PROVIDER: {data_provider}', config_content)
    
    # Update download flags
    if re.search(r'DOWNLOAD_USGS_DATA:', config_content):
        config_content = re.sub(r'DOWNLOAD_USGS_DATA:.*', f'DOWNLOAD_USGS_DATA: {download_usgs}', config_content)
    else:
        # Add it after the STREAMFLOW_DATA_PROVIDER line
        config_content = re.sub(
            r'(STREAMFLOW_DATA_PROVIDER:.*)',
            f'\\1\nDOWNLOAD_USGS_DATA: {download_usgs}',
            config_content
        )
        
    if re.search(r'DOWNLOAD_WSC_DATA:', config_content):
        config_content = re.sub(r'DOWNLOAD_WSC_DATA:.*', f'DOWNLOAD_WSC_DATA: {download_wsc}', config_content)
    else:
        # Add it after the DOWNLOAD_USGS_DATA line
        config_content = re.sub(
            r'(DOWNLOAD_USGS_DATA:.*)',
            f'\\1\nDOWNLOAD_WSC_DATA: {download_wsc}',
            config_content
        )
    
    # Update station ID if provided
    if station_id and str(station_id).lower() != 'nan':
        # Check if STATION_ID line exists
        if re.search(r'STATION_ID:', config_content):
            config_content = re.sub(r'STATION_ID:.*', f'STATION_ID: {station_id}', config_content)
        else:
            # Add it after the DOWNLOAD_WSC_DATA line
            config_content = re.sub(
                r'(DOWNLOAD_WSC_DATA:.*)',
                f'\\1\nSTATION_ID: {station_id}',
                config_content
            )
    
    # Update simulation reach ID if provided
    if sim_reach_id and str(sim_reach_id).lower() != 'nan':
        # Check if SIM_REACH_ID line exists
        if re.search(r'SIM_REACH_ID:', config_content):
            config_content = re.sub(r'SIM_REACH_ID:.*', f'SIM_REACH_ID: {sim_reach_id}', config_content)
        else:
           # Find the Evaluation settings section to add it
            if re.search(r'#+ *5\. Evaluation settings', config_content):
                config_content = re.sub(
                    r'(#+ *5\. Evaluation settings.*?\n)',
                    f'\\1\nSIM_REACH_ID: {sim_reach_id}\n',
                    config_content
                )
            else:
                # If section heading not found, add after station ID as fallback
                config_content = re.sub(
                    r'(STATION_ID:.*)',
                    f'\\1\nSIM_REACH_ID: {sim_reach_id}',
                    config_content
                )
    
    # Save the new config file
    with open(output_path, 'w') as f:
        f.write(config_content)
    
    # Verify the changes were made
    print(f"Config file created at {output_path}")
    print(f"Checking for proper updates...")
    
    with open(output_path, 'r') as f:
        new_content = f.read()
    
    # Verify key settings
    patterns = {
        'Domain name': r'DOMAIN_NAME:.*',
        'River basin path': r'RIVER_BASINS_PATH:.*',
        'River basin name': r'RIVER_BASINS_NAME:.*',
        'River network path': r'RIVER_NETWORK_SHP_PATH:.*',
        'River network name': r'RIVER_NETWORK_SHP_NAME:.*',
        'Domain definition method': r'DOMAIN_DEFINITION_METHOD:.*',
        'Pour point': r'POUR_POINT_COORDS:.*',
        'Bounding box': r'BOUNDING_BOX_COORDS:.*',
        'Streamflow data provider': r'STREAMFLOW_DATA_PROVIDER:.*',
        'Download USGS data': r'DOWNLOAD_USGS_DATA:.*',
        'Download WSC data': r'DOWNLOAD_WSC_DATA:.*',
        'Station ID': r'STATION_ID:.*',
        'Simulation reach ID': r'SIM_REACH_ID:.*'
    }
    
    for label, pattern in patterns.items():
        match = re.search(pattern, new_content)
        if match:
            print(f"{label} setting: {match.group().strip()}")
        else:
            print(f"Warning: {label} not found in config!")
    
    return output_path

def find_closest_river_segment(river_shapefile_path, pour_point_coords):
    """
    Find the river segment ID closest to the pour point coordinates
    
    Args:
        river_shapefile_path: Path to the river network shapefile
        pour_point_coords: String with pour point coordinates in format "lat/lon"
        
    Returns:
        Integer segment ID of the closest river segment, or None if not found
    """
    try:        
        # Validate pour point coordinates
        if not pour_point_coords or not validate_coords(pour_point_coords):
            print(f"Invalid pour point coordinates: {pour_point_coords}")
            return None
            
        # Parse coordinates
        lat, lon = map(float, pour_point_coords.split('/'))
        pour_point = Point(lon, lat)  # Note: Point takes (x, y) which is (lon, lat)
        
        # Read the river shapefile
        river_gdf = gpd.read_file(river_shapefile_path)
        
        # Ensure it's in lat/lon coordinate system for consistent distance calculation
        if river_gdf.crs is None or river_gdf.crs.to_epsg() != 4326:
            # Try to set to EPSG:4326 (WGS84)
            river_gdf.set_crs(epsg=4326, inplace=True)
        
        # Check if the segment ID column exists
        segment_id_column = None
        for col_name in ['LINKNO', 'COMID', 'segmentID', 'RIVER_NETWORK_SHP_SEGID']:
            if col_name in river_gdf.columns:
                segment_id_column = col_name
                break
                
        if segment_id_column is None:
            print(f"Warning: Could not find segment ID column in river shapefile")
            return None
            
        # Convert pour point to GeoDataFrame with same CRS
        pour_point_gdf = gpd.GeoDataFrame(geometry=[pour_point], crs="EPSG:4326")
        
        # Calculate distances from pour point to each river segment
        distances = []
        for idx, row in river_gdf.iterrows():
            distance = row.geometry.distance(pour_point)
            distances.append((idx, distance))
            
        # Find the closest river segment
        closest_idx, min_distance = min(distances, key=lambda x: x[1])
        closest_segment = river_gdf.iloc[closest_idx]
        
        # Get the segment ID
        segment_id = closest_segment[segment_id_column]
        
        print(f"Found closest river segment: {segment_id} (distance: {min_distance:.6f} degrees)")
        return segment_id
        
    except Exception as e:
        print(f"Error finding closest river segment: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def run_confluence(config_path, watershed_name, dry_run=False):
    """
    Run CONFLUENCE with the specified config file
    
    Args:
        config_path: Path to the config file
        watershed_name: Name of the watershed for job naming
        dry_run: If True, generate the script but don't submit the job
        
    Returns:
        job_id or None if dry_run is True
    """
    # Create a temporary batch script for this specific run
    batch_script = f"run_{watershed_name}.sh"
    
    with open(batch_script, 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name={watershed_name}
#SBATCH --output=CONFLUENCE_{watershed_name}_%j.log
#SBATCH --error=CONFLUENCE_{watershed_name}_%j.err
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --mem=5G

# Load necessary modules
module restore confluence_modules

# Activate Python environment
conda activate confluence

# Run CONFLUENCE with the specified config
python ../CONFLUENCE/CONFLUENCE.py --config {config_path}

echo "CONFLUENCE job for {watershed_name} complete"
""")
    
    # Make the script executable
    os.chmod(batch_script, 0o755)
    
    # If dry run, just return the path to the script
    if dry_run:
        print(f"Dry run - job script created at {batch_script}")
        return None
    
    # Submit the job
    result = subprocess.run(['sbatch', batch_script], capture_output=True, text=True)
    
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted job for {watershed_name}, job ID: {job_id}")
        return job_id
    else:
        print(f"Failed to submit job for {watershed_name}: {result.stderr}")
        return None

def extract_shapefile_info(shapefile_dir, scale_name):
    """
    Extract information about watershed shapefiles in the directory
    and calculate bounding box coordinates from shapefiles
    
    Args:
        shapefile_dir: Directory containing watershed shapefiles
        scale_name: Name of the scale (e.g., 'meso', 'macro', 'headwater')
        
    Returns:
        DataFrame with information about each watershed
    """
    watershed_data = []
    
    # Try to import geopandas for shapefile reading
    try:
        import geopandas as gpd # type: ignore
        has_geopandas = True
        print(f"Using geopandas to extract shapefile information for {scale_name} scale")
    except ImportError:
        has_geopandas = False
        print(f"Geopandas not available. Will not calculate bounding boxes for {scale_name} scale.")
    
    # Get all watershed folders
    watershed_folders = [f for f in os.listdir(shapefile_dir) if os.path.isdir(os.path.join(shapefile_dir, f))]
    
    print(f"Found {len(watershed_folders)} watershed folders in {shapefile_dir}")
    
    for folder in watershed_folders:
        folder_path = os.path.join(shapefile_dir, folder)
        
        # For CAMELS-SPAT, look for lumped shapefiles (these are the basin shapefiles)
        basin_files = glob.glob(os.path.join(folder_path, "*_lumped.shp"))
        
        # For now, we'll assume no separate river shapefiles for CAMELS-SPAT lumped
        # We'll need to create or find river network data separately
        river_files = []
        
        # Check if we have a basin file
        if basin_files:
            basin_file = os.path.basename(basin_files[0])
            
            # Extract gauge ID from folder name
            gauge_id = folder
            
            watershed_info = {
                'ID': gauge_id,
                'Scale': scale_name,  # Add scale information
                'Basin_File': basin_file,
                'River_File': None,  # No separate river file for lumped CAMELS-SPAT
                'Basin_Path': os.path.dirname(basin_files[0]),
                'River_Path': None  # No separate river path for lumped CAMELS-SPAT
            }
            
            print(f"Processing watershed {gauge_id} with basin file {basin_file}")
            
            # Calculate bounding box and extract pour point if geopandas is available
            if has_geopandas:
                try:
                    # Read basin shapefile
                    basin_gdf = gpd.read_file(basin_files[0])
                    
                    # Ensure it's in lat/lon (EPSG:4326)
                    if basin_gdf.crs is None:
                        print(f"Warning: No CRS defined for {gauge_id}. Assuming EPSG:4326 (WGS84).")
                        basin_gdf.set_crs(epsg=4326, inplace=True)
                    elif basin_gdf.crs.to_epsg() != 4326:
                        basin_gdf = basin_gdf.to_crs(epsg=4326)
                    
                    # Calculate basin bounding box
                    bounds = basin_gdf.total_bounds  # returns (minx, miny, maxx, maxy)
                    
                    # Convert to CONFLUENCE bounding box format (lat_max/lon_min/lat_min/lon_max)
                    bounding_box = f"{bounds[3]}/{bounds[0]}/{bounds[1]}/{bounds[2]}"
                    watershed_info['BOUNDING_BOX_COORDS'] = bounding_box
                    
                    # Calculate centroid for visualization
                    centroid = basin_gdf.dissolve().centroid.iloc[0]
                    watershed_info['Lat'] = centroid.y
                    watershed_info['Lon'] = centroid.x
                    
                    # Calculate area in km²
                    # Convert to equal-area projection for accurate area calculation
                    basin_gdf_ea = basin_gdf.to_crs('+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96')
                    area_m2 = basin_gdf_ea.area.sum()
                    watershed_info['Area_km2'] = area_m2 / 1e6
                    
                    # Check available columns in the shapefile
                    available_columns = basin_gdf.columns.tolist()
                    watershed_info['Available_Columns'] = str(available_columns)
                    
                    print(f"  Bounding box: {bounding_box}")
                    print(f"  Area: {watershed_info['Area_km2']:.2f} km²")
                    print(f"  Available columns: {available_columns}")
                    
                except Exception as e:
                    print(f"Error extracting shapefile info for {gauge_id} ({scale_name}): {e}")
                    import traceback
                    print(traceback.format_exc())
            
            watershed_data.append(watershed_info)
        else:
            print(f"Warning: No lumped shapefile found in {folder_path}")
    
    print(f"Successfully processed {len(watershed_data)} watersheds in {scale_name} scale")
    return pd.DataFrame(watershed_data)

def main():
    # Path to the template config file
    template_config_path = "/home/x-deythorsson/code/CONFLUENCE/0_config_files/config_Bow_lumped.yaml"
    
    # Directory to store generated config files
    config_dir = "/home/x-deythorsson/code/CONFLUENCE/0_config_files/camels_spat"
    
    # Create the config directory if it doesn't exist
    os.makedirs(config_dir, exist_ok=True)
    
    # Directory paths for CAMELS-SPAT distributed shapefiles (all scales)
    spatial_scheme = 'lumped'
    
    meso_spat_dir = f"/home/x-deythorsson/data/camels-spat-upload/shapefiles/meso-scale/shapes-{spatial_scheme}/"
    macro_spat_dir = f"/home/x-deythorsson/data/camels-spat-upload/shapefiles/macro-scale/shapes-{spatial_scheme}/"
    head_spat_dir = f"/home/x-deythorsson/data/camels-spat-upload/shapefiles/headwater/shapes-{spatial_scheme}/"
    
    camels_spat_dirs = [
        (meso_spat_dir, "meso"),
        (macro_spat_dir, "macro"), 
        (head_spat_dir, "headwater")
    ]

    # Path to the CAMELS-SPAT metadata CSV file
    metadata_csv_path = "camels-spat-metadata.csv"

    # Check if the metadata file exists
    metadata_df = None
    if os.path.exists(metadata_csv_path):
        print(f"Loading CAMELS-SPAT metadata from {metadata_csv_path}")
        try:
            # Try reading with default parameters first
            raw_metadata_df = pd.read_csv(metadata_csv_path)
            
            # Clean column names and convert to string to prevent NaN issues
            raw_metadata_df.columns = [col.strip() for col in raw_metadata_df.columns]
            
            print(f"Found metadata with {len(raw_metadata_df)} rows and columns: {raw_metadata_df.columns.tolist()}")
            
            # Process the metadata to create expected columns
            metadata_df = process_camels_spat_metadata(raw_metadata_df)
            
            if metadata_df is not None:
                use_metadata = True
                print(f"Successfully processed metadata with {len(metadata_df)} rows")
                
                # Print first few rows for verification
                print("\nSample processed metadata:")
                print(metadata_df[['ID', 'POUR_POINT_COORDS', 'POUR_POINT_SOURCE']].head())
            else:
                print("Failed to process metadata - will extract information from shapefiles only.")
                use_metadata = False
                
        except Exception as e:
            print(f"Error reading metadata file: {e}")
            metadata_df = None
            use_metadata = False
    else:
        print(f"Metadata file not found at {metadata_csv_path}")
        print("Will extract information from shapefiles only.")
        use_metadata = False
    
    # Check if we already have the watershed data in a CSV file
    watersheds_csv_path = "camels_spat_watersheds_all_scales.csv"
    if os.path.exists(watersheds_csv_path):
        print(f"Found existing watershed information at {watersheds_csv_path}")
        reload_data = input("Do you want to reload the shapefile information? (y/n): ").lower().strip()
        
        if reload_data == 'y':
            print(f"Extracting shapefile information from all scales...")
            
            # Extract information from all directories
            all_watersheds = []
            for spat_dir, scale_name in camels_spat_dirs:
                if os.path.exists(spat_dir):
                    print(f"Processing {scale_name} scale from {spat_dir}...")
                    scale_watersheds = extract_shapefile_info(spat_dir, scale_name)
                    print(f"Found {len(scale_watersheds)} watersheds in {scale_name} scale")
                    all_watersheds.append(scale_watersheds)
                else:
                    print(f"Warning: Directory not found: {spat_dir}")
            
            # Combine all watersheds
            if all_watersheds:
                watersheds = pd.concat(all_watersheds, ignore_index=True)
                print(f"Total watersheds found across all scales: {len(watersheds)}")
                
                # Save watershed information to CSV for reference
                watersheds.to_csv(watersheds_csv_path, index=False)
                print(f"Saved updated watershed information to {watersheds_csv_path}")
            else:
                print("No watershed data found in any directory!")
                return
        else:
            print(f"Loading watershed information from {watersheds_csv_path}...")
            watersheds = pd.read_csv(watersheds_csv_path)
    else:
        # Extract shapefile information from all directories
        print(f"Extracting shapefile information from all scales...")
        
        all_watersheds = []
        for spat_dir, scale_name in camels_spat_dirs:
            if os.path.exists(spat_dir):
                print(f"Processing {scale_name} scale from {spat_dir}...")
                scale_watersheds = extract_shapefile_info(spat_dir, scale_name)
                print(f"Found {len(scale_watersheds)} watersheds in {scale_name} scale")
                all_watersheds.append(scale_watersheds)
            else:
                print(f"Warning: Directory not found: {spat_dir}")
        
        # Combine all watersheds
        if all_watersheds:
            watersheds = pd.concat(all_watersheds, ignore_index=True)
            print(f"Total watersheds found across all scales: {len(watersheds)}")
            
            # Save watershed information to CSV for reference
            watersheds.to_csv(watersheds_csv_path, index=False)
            print(f"Saved watershed information to {watersheds_csv_path}")
        else:
            print("No watershed data found in any directory!")
            return
    
    # Print summary by scale
    print("\nWatershed count by scale:")
    if 'Scale' in watersheds.columns:
        scale_counts = watersheds['Scale'].value_counts()
        for scale, count in scale_counts.items():
            print(f"  {scale}: {count} watersheds")
    
    # Merge metadata with watershed information if available
    if use_metadata and metadata_df is not None:
        print("Merging shapefile information with metadata...")
        
        # Create a standardized ID column for joining
        # First, extract IDs without the country prefix from the watershed data
        watersheds['Metadata_ID'] = watersheds['ID'].str.replace(r'^[A-Z]+_', '', regex=True)
        
        # Print sample of IDs for verification
        print("\nSample of watershed IDs:")
        print(watersheds[['ID', 'Metadata_ID', 'Scale']].head())
        
        # Print sample of metadata IDs for verification
        print("\nSample of metadata IDs:")
        print(metadata_df['ID'].head())
        
        # Merge on the standardized ID
        watersheds = pd.merge(
            watersheds, 
            metadata_df, 
            left_on='Metadata_ID',  # ID without country prefix
            right_on='ID',          # ID in metadata
            how='left',
            suffixes=('', '_metadata')
        )
        
        # Print sample of merged data to verify
        print("\nSample of merged data:")
        print(watersheds[['ID', 'Scale', 'ID_metadata', 'POUR_POINT_COORDS']].head())
        
        # Update columns with metadata where available, but keep shapefile data as backup
        # For columns like Lat, Lon, BOUNDING_BOX_COORDS, etc., prioritize shapefile-derived values
        for col in ['Lat', 'Lon', 'BOUNDING_BOX_COORDS']:
            if col in watersheds.columns:
                watersheds[col + '_shapefile'] = watersheds[col]
        
        # Add metadata-specific columns
        if 'Station_name' in metadata_df.columns:
            watersheds['Watershed_Name'] = watersheds['Station_name']
        
        # Save the merged information
        watersheds.to_csv("camels_spat_watersheds_merged_all_scales.csv", index=False)
        print("Saved merged watershed information to camels_spat_watersheds_merged_all_scales.csv")
    
    # Process each watershed for CONFLUENCE runs
    submitted_jobs = []
    
    # Ask if user wants to submit CONFLUENCE jobs
    submit_jobs = input("\nDo you want to submit CONFLUENCE jobs for these watersheds? (y/n/dry): ").lower().strip()
    
    dry_run = (submit_jobs == 'dry')
    if submit_jobs == 'y' or dry_run:
        # Ask which scales to process
        available_scales = watersheds['Scale'].unique() if 'Scale' in watersheds.columns else ['all']
        print(f"\nAvailable scales: {', '.join(available_scales)}")
        selected_scales = input("Which scales to process? (comma-separated or 'all'): ").strip()
        
        if selected_scales.lower() == 'all':
            watersheds_to_process = watersheds
        else:
            selected_scale_list = [s.strip() for s in selected_scales.split(',')]
            watersheds_to_process = watersheds[watersheds['Scale'].isin(selected_scale_list)]
            print(f"Selected {len(watersheds_to_process)} watersheds from scales: {selected_scale_list}")
        
        # Ask how many watersheds to process (in case user wants to limit)
        max_watersheds = input("How many watersheds to process? (Enter a number or 'all'): ").strip()
        
        if max_watersheds.lower() != 'all':
            try:
                num_watersheds = int(max_watersheds)
                watersheds_to_process = watersheds_to_process.head(num_watersheds)
            except ValueError:
                print("Invalid input. Processing all selected watersheds.")
        
        print(f"\nProcessing {len(watersheds_to_process)} watersheds...")
        
        for _, watershed in watersheds_to_process.iterrows():
            # Get watershed parameters
            watershed_id = watershed['ID']
            basin_file = watershed['Basin_File']
            river_file = watershed['River_File']
            basin_path = watershed['Basin_Path']
            river_path = watershed['River_Path']
            scale = watershed.get('Scale', 'unknown')
            
            print(f"\nProcessing {watershed_id} ({scale} scale)")
            if scale == 'headwater':
                scale_name = f"{scale}"
            elif scale == 'meso':
                scale_name = f"{scale}-scale"
            elif scale == 'macro':
                scale_name = f"{scale}-scale"

            # Get watershed name from metadata if available
            if 'Watershed_Name' in watershed and not pd.isna(watershed['Watershed_Name']):
                watershed_name = watershed['Watershed_Name']
            else:
                watershed_name = watershed_id
            
            # Create a unique domain name including scale
            domain_name = f"{watershed_id}_{scale}"
            
            # Check if the simulations directory already exists
            simulation_dir = Path(f"/anvil/projects/x-ees240082/data/CONFLUENCE_data/camels_spat/domain_{domain_name}.tar.gz")
            
            if simulation_dir.exists():
                print(f"Skipping {domain_name} - simulation directory already exists: {simulation_dir}")
                continue
            
            # Generate the config file path
            config_path = os.path.join(config_dir, f"config_{domain_name}.yaml")
            
            # Set up CONFLUENCE directory and copy shapefiles
            print(f"Setting up CONFLUENCE directory for {domain_name}...")
            basin_target_dir, river_target_dir = setup_confluence_directory(
                domain_name,  # Use domain_name that includes scale
                basin_path, 
                basin_file, 
                scale_name,
                river_path, 
                river_file
            )
            
            # Calculate fresh bounding box from copied shapefile
            basin_shapefile_path = os.path.join(basin_target_dir, basin_file)
            print(f"Calculating fresh bounding box from {basin_shapefile_path}")
            bounding_box = calculate_fresh_bounding_box(basin_shapefile_path)
            
            if bounding_box:
                print(f"Using freshly calculated bounding box: {bounding_box}")
            else:
                # Fall back to stored bounding box if available
                if 'BOUNDING_BOX_COORDS' in watershed and not pd.isna(watershed['BOUNDING_BOX_COORDS']):
                    bounding_box = watershed['BOUNDING_BOX_COORDS']
                    print(f"Using stored bounding box: {bounding_box}")
                else:
                    print(f"Warning: No bounding box available for {watershed_id}")
            
            # Get pour point from metadata if available
            pour_point = None
            
            # First try metadata's POUR_POINT_COORDS
            if 'POUR_POINT_COORDS' in watershed and validate_coords(watershed['POUR_POINT_COORDS']):
                pour_point = watershed['POUR_POINT_COORDS']
                print(f"Using pour point from metadata: {pour_point}")
            # Then try station coordinates
            elif all(col in watershed and not pd.isna(watershed[col]) for col in ['Station_lat', 'Station_lon']):
                pour_point = f"{watershed['Station_lat']}/{watershed['Station_lon']}"
                print(f"Using station coordinates as pour point: {pour_point}")
            # Finally try shapefile-derived coordinates 
            elif 'POUR_POINT_COORDS_shapefile' in watershed and not pd.isna(watershed['POUR_POINT_COORDS_shapefile']):
                pour_point = watershed['POUR_POINT_COORDS_shapefile']
                print(f"Using pour point from shapefile analysis: {pour_point}")
            else:
                print(f"Warning: No pour point coordinates found for {watershed_id}")
            
            # Determine station ID for streamflow data
            station_id = None
            # First try to use the Station_id from metadata
            if 'Station_id' in watershed and not pd.isna(watershed['Station_id']):
                station_id = watershed['Station_id']
            # If that doesn't exist, extract it from the ID (removing country prefix)
            elif 'Metadata_ID' in watershed and not pd.isna(watershed['Metadata_ID']):
                station_id = watershed['Metadata_ID']
            # As a last resort, extract from the ID itself (removing country prefix)
            else:
                # Extract the ID without country prefix
                match = re.search(r'^[A-Z]+_(.+)$', watershed_id)
                if match:
                    station_id = match.group(1)
            
            print(f"Using station ID for streamflow: {station_id}")
            
            # Find the closest river segment to the pour point for SIM_REACH_ID
            sim_reach_id = None
            if pour_point:
                river_shapefile_path = os.path.join(river_target_dir, river_file)
                sim_reach_id = find_closest_river_segment(river_shapefile_path, pour_point)
                print(f"Using simulation reach ID: {sim_reach_id}")
            
            # Generate the config file using the new target directories
            print(f"Generating config file for {domain_name}...")
            generate_distributed_config_file(
                template_config_path, 
                config_path, 
                domain_name, 
                str(basin_target_dir), 
                basin_file, 
                str(river_target_dir),
                river_file,
                bounding_box,
                pour_point,
                station_id,
                sim_reach_id  # Add the simulation reach ID
            )
            
            # Run CONFLUENCE with the generated config
            if dry_run:
                print(f"Preparing dry run for {domain_name}...")
                job_id = run_confluence(config_path, domain_name, dry_run=True)
                submitted_jobs.append((domain_name, "DRY_RUN"))
            else:
                print(f"Submitting CONFLUENCE job for {domain_name}...")
                job_id = run_confluence(config_path, domain_name)
                
                if job_id:
                    submitted_jobs.append((domain_name, job_id))
            
            # Add a small delay between job submissions to avoid overwhelming the scheduler
            time.sleep(20)
        
        # Print summary of submitted jobs
        if dry_run:
            print("\nDry run summary:")
            print(f"Created config files and job scripts for {len(submitted_jobs)} watersheds")
            print(f"To submit the jobs, run the scripts in the current directory")
        else:
            print("\nSubmitted jobs summary:")
            for domain_name, job_id in submitted_jobs:
                print(f"Domain: {domain_name}, Job ID: {job_id}")
            
            print(f"\nTotal jobs submitted: {len(submitted_jobs)}")
            
            # Print summary by scale
            if submitted_jobs:
                scale_summary = {}
                for domain_name, job_id in submitted_jobs:
                    # Extract scale from domain name
                    scale = domain_name.split('_')[-1]
                    scale_summary[scale] = scale_summary.get(scale, 0) + 1
                
                print("\nJobs submitted by scale:")
                for scale, count in scale_summary.items():
                    print(f"  {scale}: {count} jobs")
    else:
        print("\nNo CONFLUENCE jobs submitted.")
        print(f"To run CONFLUENCE jobs later, use the config files generated in {config_dir}")

if __name__ == "__main__":
    main()



