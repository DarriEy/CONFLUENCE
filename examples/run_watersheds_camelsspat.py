import pandas as pd # type: ignore
import os
import yaml # type: ignore
import subprocess
from pathlib import Path
import time
import sys
import re
import glob
import shutil

def setup_confluence_directory(watershed_id, basin_source_path, basin_filename, river_source_path, river_filename):
    """
    Set up the CONFLUENCE directory structure for a watershed and copy relevant shapefiles.
    For CAMELS-SPAT implementation, adds GRU_area and sets gru_to_seg and GRU_ID to equal COMID.
    Also ensures that river network shapefiles have the required fields for MizuRoute.
    
    Args:
        watershed_id: ID of the watershed
        basin_source_path: Path to the source basin shapefile directory
        basin_filename: Name of the basin shapefile
        river_source_path: Path to the source river shapefile directory
        river_filename: Name of the river shapefile
        
    Returns:
        Tuple of (basin_target_path, river_target_path)
    """
    try:
        import geopandas as gpd # type: ignore
        from shapely.geometry import LineString # type: ignore
        import numpy as np # type: ignore
    except ImportError:
        print("Warning: geopandas or other required packages not installed. Cannot modify shapefile attributes.")
        return None, None
        
    # Base CONFLUENCE data directory
    confluence_data_dir = Path("/work/comphyd_lab/data/CONFLUENCE_data")
    
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
        
        # For distributed case, use the river file
        river_source_base = str(Path(river_source_path) / river_filename).rsplit('.', 1)[0]
        river_target_base = str(river_target_dir / river_filename).rsplit('.', 1)[0]
        river_source_file = f"{river_source_base}.{extension}"
        
        river_target_file = f"{river_target_base}.{extension}"
        if os.path.exists(river_source_file):
            shutil.copy2(river_source_file, river_target_file)
            print(f"Copied {river_source_file} to {river_target_file}")
    
    # Now modify the shapefiles to add required attributes
    try:
        # Read the basin shapefile
        basin_shp = gpd.read_file(f"{basin_target_base}.shp")
        
        # Add GRU_ID and gru_to_seg fields based on COMID
        if 'COMID' in basin_shp.columns:
            basin_shp['GRU_ID'] = basin_shp['COMID']
            basin_shp['gru_to_seg'] = basin_shp['COMID']
            print(f"Set GRU_ID and gru_to_seg to match COMID values")
        else:
            # Fallback if COMID doesn't exist
            basin_shp['GRU_ID'] = 1
            basin_shp['gru_to_seg'] = 1
            print(f"Warning: COMID column not found. Set GRU_ID and gru_to_seg to 1")
        
        # Calculate GRU_area in square meters
        if basin_shp.crs is None:
            print(f"Warning: CRS not defined for {basin_filename}. Trying to set to EPSG:4326 (WGS84).")
            basin_shp.set_crs(epsg=4326, inplace=True)
        
        # Convert to equal area projection for accurate area calculation
        basin_shp_ea = basin_shp.to_crs('+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96')
        basin_shp['GRU_area'] = basin_shp_ea.geometry.area
        print(f"Calculated GRU_area based on geometry")
        
        # Save the modified basin shapefile
        basin_shp.to_file(f"{basin_target_base}.shp")
        print(f"Added GRU_ID, GRU_area, and gru_to_seg columns to {basin_filename}")
        
        # Now read and modify the river shapefile
        river_shp = gpd.read_file(f"{river_target_base}.shp")
        
        # Check if CRS is defined
        if river_shp.crs is None:
            print(f"Warning: CRS not defined for {river_filename}. Trying to set to EPSG:4326 (WGS84).")
            river_shp.set_crs(epsg=4326, inplace=True)
        
        # Get column mapping for source vs required columns
        column_mappings = {
            'Length': None,  # Will calculate if not found
            'LINKNO': 'COMID' if 'COMID' in river_shp.columns else None,
            'DSLINKNO': 'NextDownID' if 'NextDownID' in river_shp.columns else None,
            'Slope': 'slope' if 'slope' in river_shp.columns else None
        }
        
        print("River shapefile columns:", river_shp.columns.tolist())
        
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
                if target_field == 'LINKNO' and 'FEATUREID' in river_shp.columns:
                    river_shp[target_field] = river_shp['FEATUREID']
                    print(f"Copied values from FEATUREID to {target_field}")
                elif target_field == 'DSLINKNO' and 'TO_NODE' in river_shp.columns:
                    river_shp[target_field] = river_shp['TO_NODE']
                    print(f"Copied values from TO_NODE to {target_field}")
                else:
                    # Initialize with a default value based on field type
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
                        
                        # Attempt to infer downstream connectivity if possible
                        # In many cases, this would require network analysis which is complex
                        # For this example we'll just set it to -1 for outlets
        
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
        river_shp.to_file(f"{river_target_base}.shp")
        print(f"Modified river shapefile saved with all required fields at: {river_target_base}")
        
    except Exception as e:
        print(f"Error modifying shapefile attributes: {e}")
        import traceback
        print(traceback.format_exc())
    
    return basin_target_dir, river_target_dir

def calculate_fresh_bounding_box(shapefile_path):
    """
    Calculate a bounding box directly from the shapefile
    
    Args:
        shapefile_path: Path to shapefile
        
    Returns:
        String with bounding box in lat_max/lon_min/lat_min/lon_max format
    """
    try:
        import geopandas as gpd # type: ignore
        
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
    
    # Update the river basin name
    config_content = re.sub(r'RIVER_BASINS_NAME:.*', f'RIVER_BASINS_NAME: "{basin_name}"', config_content)
    
    # Update the river network name
    config_content = re.sub(r'RIVER_NETWORK_SHP_NAME:.*', f'RIVER_NETWORK_SHP_NAME: "{river_network_name}"', config_content)
    
    # Update domain definition method to use delineate
    config_content = re.sub(r'DOMAIN_DEFINITION_METHOD:.*', f'DOMAIN_DEFINITION_METHOD: delineate', config_content)
    
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
    else:
        # Default case
        data_provider = 'WSC'  # Default to WSC
        download_wsc = 'True'
        download_usgs = 'False'
    
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
        import geopandas as gpd
        from shapely.geometry import Point
        import numpy as np
        
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
#SBATCH --time=40:00:00
#SBATCH --ntasks=1
#SBATCH --mem=10G

# Load necessary modules
. /work/comphyd_lab/local/modules/spack/2024v5/lmod-init-bash
module unuse $MODULEPATH
module use /work/comphyd_lab/local/modules/spack/2024v5/modules/linux-rocky8-x86_64/Core/

module load netcdf-fortran/4.6.1
module load openblas/0.3.27
module load hdf/4.3.0
module load hdf5/1.14.3
module load gdal/3.9.2
module load netlib-lapack/3.11.0
module load openmpi/4.1.6
module load python/3.11.7
module load r/4.4.1

# Activate Python environment
source /work/comphyd_lab/users/darri/data/CONFLUENCE_data/installs/conf-env/bin/activate

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

def extract_shapefile_info(shapefile_dir):
    """
    Extract information about watershed shapefiles in the directory
    and calculate bounding box coordinates from shapefiles
    
    Args:
        shapefile_dir: Directory containing watershed shapefiles
        
    Returns:
        DataFrame with information about each watershed
    """
    watershed_data = []
    
    # Try to import geopandas for shapefile reading
    try:
        import geopandas as gpd # type: ignore
        has_geopandas = True
        print("Using geopandas to extract shapefile information")
    except ImportError:
        has_geopandas = False
        print("Geopandas not available. Will not calculate bounding boxes.")
    
    # Get all watershed folders
    watershed_folders = [f for f in os.listdir(shapefile_dir) if os.path.isdir(os.path.join(shapefile_dir, f))]
    
    for folder in watershed_folders:
        folder_path = os.path.join(shapefile_dir, folder)
        
        # Get basin shapefile
        basin_files = glob.glob(os.path.join(folder_path, "*_basin.shp"))
        # Get river shapefile
        river_files = glob.glob(os.path.join(folder_path, "*_river.shp"))
        
        if basin_files and river_files:
            basin_file = os.path.basename(basin_files[0])
            river_file = os.path.basename(river_files[0])
            
            # Extract gauge ID from folder name
            gauge_id = folder
            
            watershed_info = {
                'ID': gauge_id,
                'Basin_File': basin_file,
                'River_File': river_file,
                'Basin_Path': os.path.dirname(basin_files[0]),
                'River_Path': os.path.dirname(river_files[0])
            }
            
            # Calculate bounding box and extract pour point if geopandas is available
            if has_geopandas:
                try:
                    # Read basin shapefile
                    basin_gdf = gpd.read_file(basin_files[0])
                    # Read river shapefile
                    river_gdf = gpd.read_file(river_files[0])
                    
                    # Ensure both are in the same CRS and use lat/lon (EPSG:4326)
                    if basin_gdf.crs is None or basin_gdf.crs.to_epsg() != 4326:
                        # Try to get the CRS from the river file if basin CRS is missing
                        if basin_gdf.crs is None and river_gdf.crs is not None:
                            basin_gdf.set_crs(river_gdf.crs, inplace=True)
                        # Convert to EPSG:4326 (lat/lon)
                        basin_gdf = basin_gdf.to_crs(epsg=4326)
                    
                    if river_gdf.crs is None or river_gdf.crs.to_epsg() != 4326:
                        # Try to get the CRS from the basin file if river CRS is missing
                        if river_gdf.crs is None and basin_gdf.crs is not None:
                            river_gdf.set_crs(basin_gdf.crs, inplace=True)
                        # Convert to EPSG:4326 (lat/lon)
                        river_gdf = river_gdf.to_crs(epsg=4326)
                    
                    # Calculate basin bounding box
                    bounds = basin_gdf.total_bounds  # returns (minx, miny, maxx, maxy)
                    
                    # Convert to CONFLUENCE bounding box format (lat_max/lon_min/lat_min/lon_max)
                    bounding_box = f"{bounds[3]}/{bounds[0]}/{bounds[1]}/{bounds[2]}"
                    watershed_info['BOUNDING_BOX_COORDS'] = bounding_box
                    
                    # Try to extract pour point (if the river shapefile has a pour point feature)
                    # This is a simplified approach - might need custom logic depending on your data
                    if 'LINKNO' in river_gdf.columns and 'DSLINKNO' in river_gdf.columns:
                        try:
                            # Find outlets (segments that don't drain to another segment within the basin)
                            outlets = river_gdf[~river_gdf['DSLINKNO'].isin(river_gdf['LINKNO'])]
                            
                            if not outlets.empty:
                                # Use the first outlet as the pour point (might need more sophisticated logic)
                                pour_point_geom = outlets.iloc[0].geometry
                                
                                # Try to get the point at the end of the line
                                if hasattr(pour_point_geom, 'coords') and len(list(pour_point_geom.coords)) > 0:
                                    # Take the last point in the line
                                    coords = list(pour_point_geom.coords)
                                    last_point = coords[-1]
                                    pour_point = f"{last_point[1]}/{last_point[0]}"  # lat/lon format
                                    watershed_info['POUR_POINT_COORDS_shapefile'] = pour_point
                        except Exception as e:
                            print(f"Could not extract pour point for {gauge_id}: {e}")
                    
                    # Calculate centroid for visualization
                    centroid = basin_gdf.dissolve().centroid.iloc[0]
                    watershed_info['Lat'] = centroid.y
                    watershed_info['Lon'] = centroid.x
                    
                    # Calculate area in km²
                    # Convert to equal-area projection for accurate area calculation
                    basin_gdf_ea = basin_gdf.to_crs('+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96')
                    area_m2 = basin_gdf_ea.area.sum()
                    watershed_info['Area_km2'] = area_m2 / 1e6
                    
                except Exception as e:
                    print(f"Error extracting shapefile info for {gauge_id}: {e}")
            
            watershed_data.append(watershed_info)
    
    return pd.DataFrame(watershed_data)

def main():
    # Path to the template config file
    template_config_path = "/home/darri.eythorsson/code/CONFLUENCE/0_config_files/config_distributed_basin_template.yaml"
    
    # Directory to store generated config files
    config_dir = "/home/darri.eythorsson/code/CONFLUENCE/0_config_files/camels_spat"
    
    # Create the config directory if it doesn't exist
    os.makedirs(config_dir, exist_ok=True)
    
    # Directory containing CAMELS-SPAT distributed shapefiles
    camels_spat_dir = "/work/comphyd_lab/data/_to-be-moved/camels-spat-upload/shapefiles/meso-scale/shapes-distributed"
    
    # Path to the CAMELS-SPAT metadata CSV file
    metadata_csv_path = "camels-spat-metadata.csv"
    
    # Check if the metadata file exists
    metadata_df = None
    if os.path.exists(metadata_csv_path):
        print(f"Loading CAMELS-SPAT metadata from {metadata_csv_path}")
        try:
            # Try reading with default parameters first
            metadata_df = pd.read_csv(metadata_csv_path)
            
            # Clean column names and convert to string to prevent NaN issues
            metadata_df.columns = [col.strip() for col in metadata_df.columns]
            
            # Convert POUR_POINT_COORDS to string to ensure it's always accessible
            if 'POUR_POINT_COORDS' in metadata_df.columns:
                metadata_df['POUR_POINT_COORDS'] = metadata_df['POUR_POINT_COORDS'].astype(str)
            
            print(f"Found metadata with {len(metadata_df)} rows and columns: {metadata_df.columns.tolist()}")
            
            # Check for required columns
            required_columns = ['ID', 'POUR_POINT_COORDS']
            missing_columns = [col for col in required_columns if col not in metadata_df.columns]
            
            if missing_columns:
                print(f"Warning: Missing required columns in metadata: {missing_columns}")
                print("Will attempt to extract information from shapefiles only.")
                use_metadata = False
            else:
                use_metadata = True
                # Print first few rows of POUR_POINT_COORDS for verification
                print("Sample POUR_POINT_COORDS values:")
                print(metadata_df[['ID', 'POUR_POINT_COORDS']].head())
        except Exception as e:
            print(f"Error reading metadata file: {e}")
            metadata_df = None
            use_metadata = False
    else:
        print(f"Metadata file not found at {metadata_csv_path}")
        print("Will extract information from shapefiles only.")
        use_metadata = False
    
    # Check if we already have the watershed data in a CSV file
    watersheds_csv_path = "camels_spat_watersheds.csv"
    if os.path.exists(watersheds_csv_path):
        print(f"Found existing watershed information at {watersheds_csv_path}")
        reload_data = input("Do you want to reload the shapefile information? (y/n): ").lower().strip()
        
        if reload_data == 'y':
            print(f"Extracting shapefile information from {camels_spat_dir}...")
            watersheds = extract_shapefile_info(camels_spat_dir)
            # Save watershed information to CSV for reference
            watersheds.to_csv(watersheds_csv_path, index=False)
            print(f"Saved updated watershed information to {watersheds_csv_path}")
        else:
            print(f"Loading watershed information from {watersheds_csv_path}...")
            watersheds = pd.read_csv(watersheds_csv_path)
    else:
        # Extract shapefile information
        print(f"Extracting shapefile information from {camels_spat_dir}...")
        watersheds = extract_shapefile_info(camels_spat_dir)
        
        # Save watershed information to CSV for reference
        watersheds.to_csv(watersheds_csv_path, index=False)
        print(f"Saved watershed information to {watersheds_csv_path}")
        
    # Merge metadata with watershed information if available
    if use_metadata and metadata_df is not None:
        print("Merging shapefile information with metadata...")
        
        # Create a standardized ID column for joining
        # First, extract IDs without the country prefix from the watershed data
        watersheds['Metadata_ID'] = watersheds['ID'].str.replace(r'^[A-Z]+_', '', regex=True)
        
        # Print sample of IDs for verification
        print("\nSample of watershed IDs:")
        print(watersheds[['ID', 'Metadata_ID']].head())
        
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
        print(watersheds[['ID', 'ID_metadata', 'POUR_POINT_COORDS']].head())
        
        # Update columns with metadata where available, but keep shapefile data as backup
        # For columns like Lat, Lon, BOUNDING_BOX_COORDS, etc., prioritize shapefile-derived values
        for col in ['Lat', 'Lon', 'BOUNDING_BOX_COORDS']:
            if col in watersheds.columns:
                watersheds[col + '_shapefile'] = watersheds[col]
        
        # Add metadata-specific columns
        if 'Station_name' in metadata_df.columns:
            watersheds['Watershed_Name'] = watersheds['Station_name']
        
        # Save the merged information
        watersheds.to_csv("camels_spat_watersheds_merged.csv", index=False)
        print("Saved merged watershed information to camels_spat_watersheds_merged.csv")
    
    # Process each watershed for CONFLUENCE runs
    submitted_jobs = []
    
    # Ask if user wants to submit CONFLUENCE jobs
    submit_jobs = input("\nDo you want to submit CONFLUENCE jobs for these watersheds? (y/n/dry): ").lower().strip()
    
    dry_run = (submit_jobs == 'dry')
    if submit_jobs == 'y' or dry_run:
        # Ask how many watersheds to process (in case user wants to limit)
        max_watersheds = input("How many watersheds to process? (Enter a number or 'all'): ").strip()
        
        if max_watersheds.lower() == 'all':
            watersheds_to_process = watersheds
        else:
            try:
                num_watersheds = int(max_watersheds)
                watersheds_to_process = watersheds.head(num_watersheds)
            except ValueError:
                print("Invalid input. Processing all watersheds.")
                watersheds_to_process = watersheds
        
        for _, watershed in watersheds_to_process.iterrows():
            # Get watershed parameters
            watershed_id = watershed['ID']
            basin_file = watershed['Basin_File']
            river_file = watershed['River_File']
            basin_path = watershed['Basin_Path']
            river_path = watershed['River_Path']
            print(watershed)
            
            # Get watershed name from metadata if available
            if 'Watershed_Name' in watershed and not pd.isna(watershed['Watershed_Name']):
                watershed_name = watershed['Watershed_Name']
            else:
                watershed_name = watershed_id
            
            # Create a unique domain name
            domain_name = f"{watershed_id}"
            
            # Check if the simulations directory already exists
            simulation_dir = Path(f"/work/comphyd_lab/data/CONFLUENCE_data/camels_spat/domain_{domain_name}/")
            
            if simulation_dir.exists():
                print(f"Skipping {domain_name} - simulation directory already exists: {simulation_dir}")
                continue
            
            # Generate the config file path
            config_path = os.path.join(config_dir, f"config_{domain_name}.yaml")
            
            # Set up CONFLUENCE directory and copy shapefiles
            print(f"Setting up CONFLUENCE directory for {domain_name}...")
            basin_target_dir, river_target_dir = setup_confluence_directory(
                watershed_id, 
                basin_path, 
                basin_file, 
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
    else:
        print("\nNo CONFLUENCE jobs submitted.")
        print(f"To run CONFLUENCE jobs later, use the config files generated in {config_dir}")

if __name__ == "__main__":
    main()