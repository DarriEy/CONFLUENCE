import geopandas as gpd #type:ignore
import pandas as pd #type:ignore
import os
import subprocess
from pathlib import Path
import time
import re
import glob
import shutil
import argparse

def setup_confluence_directory(dataset, watershed_id, basin_shapefile_path, basin_filename, basin_id_column=None, is_multibasin=False, geometry=None):
    """
    Set up the CONFLUENCE directory structure for a watershed and copy relevant shapefiles
    
    Args:
        dataset: The CARAVAN dataset name (e.g., 'camels', 'camelsaus', etc.)
        watershed_id: ID of the watershed
        basin_shapefile_path: Path to the source basin shapefile
        basin_filename: Name of the basin shapefile
        basin_id_column: Column name containing basin IDs (for multi-basin shapefiles)
        is_multibasin: Whether this is a multi-basin shapefile
        geometry: Optional geometry object for this specific basin (for multi-basin shapefiles)
        
    Returns:
        Tuple of (basin_target_path, success_flag)
    """        
    # Base CONFLUENCE data directory
    confluence_data_dir = Path("/work/comphyd_lab/data/CONFLUENCE_data")
    
    # Create the domain directory structure
    domain_dir = confluence_data_dir / "caravan" / f"domain_{dataset}_{watershed_id}"
    basin_target_dir = domain_dir / "shapefiles" / "river_basins"
    river_target_dir = domain_dir / "shapefiles" / "river_network"
    
    # Create directories for observations
    obs_dir = domain_dir / "observations" / "streamflow" / "raw_data"
    obs_processed_dir = domain_dir / "observations" / "streamflow" / "preprocessed"
    
    # Create directories if they don't exist
    for directory in [domain_dir, basin_target_dir, river_target_dir, obs_dir, obs_processed_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Define target filenames for this specific watershed
    target_basin_filename = f"{dataset}_{watershed_id}_basin.shp"
    target_river_filename = f"{dataset}_{watershed_id}_river.shp"
    
    basin_source_path = Path(basin_shapefile_path)
    basin_source_file = basin_source_path / basin_filename
    
    # Handle differently based on whether this is a multi-basin shapefile
    if is_multibasin and basin_id_column and geometry is not None:
        # For multi-basin shapefiles, extract just this watershed's geometry
        print(f"Extracting basin {watershed_id} from multi-basin shapefile")
        
        # Read the source shapefile
        gdf_all = gpd.read_file(basin_source_file)
        
        # Create a new GeoDataFrame with just this watershed's data
        if basin_id_column in gdf_all.columns:
            # Extract the specific basin by ID
            basin_gdf = gdf_all[gdf_all[basin_id_column].astype(str) == watershed_id].copy()
            
            if len(basin_gdf) == 0:
                print(f"Warning: Basin ID {watershed_id} not found in {basin_id_column} column")
                return domain_dir, False
        else:
            # Create a new GeoDataFrame with the provided geometry
            basin_gdf = gpd.GeoDataFrame(geometry=[geometry])
            if gdf_all.crs:
                basin_gdf.crs = gdf_all.crs
            
        # Save to the target location
        basin_target_file = basin_target_dir / target_basin_filename
        river_target_file = river_target_dir / target_river_filename
        
        # Save to both locations (basin and river use same shape for lumped watershed)
        basin_gdf.to_file(basin_target_file)
        basin_gdf.to_file(river_target_file)
        
        print(f"Saved basin {watershed_id} to {basin_target_file}")
        print(f"Saved river {watershed_id} to {river_target_file}")
        
    else:
        # For single watersheds, copy the entire shapefile
        basin_source_base = str(basin_source_file).rsplit('.', 1)[0]
        basin_target_base = str(basin_target_dir / target_basin_filename).rsplit('.', 1)[0]
        river_target_base = str(river_target_dir / target_river_filename).rsplit('.', 1)[0]
        
        # Copy all shapefile components (shp, shx, dbf, prj, cpg, etc.)
        for extension in ['shp', 'shx', 'dbf', 'prj', 'cpg']:
            source_file = f"{basin_source_base}.{extension}"
            basin_target_file = f"{basin_target_base}.{extension}"
            river_target_file = f"{river_target_base}.{extension}"
            
            if os.path.exists(source_file):
                # Copy to basin directory
                shutil.copy2(source_file, basin_target_file)
                print(f"Copied {source_file} to {basin_target_file}")
                
                # Copy to river directory (same file for lumped case)
                shutil.copy2(source_file, river_target_file)
                print(f"Copied {source_file} to {river_target_file}")
    
    # Now modify the shapefiles to add required attributes
    try:
        # Read the basin shapefile
        basin_shp = gpd.read_file(basin_target_dir / target_basin_filename)
        
        # Add GRU_ID column (set to 1 for all features)
        basin_shp['GRU_ID'] = 1
        basin_shp['gru_to_seg'] = 1
        
        # Calculate area in square meters if not present
        if 'GRU_area' not in basin_shp.columns:
            if basin_shp.crs is None:
                print(f"Warning: CRS not defined for {target_basin_filename}. Trying to set to EPSG:4326 (WGS84).")
                basin_shp.set_crs(epsg=4326, inplace=True)
            
            # Convert to equal area projection for accurate area calculation
            basin_shp_ea = basin_shp.to_crs('+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96')
            basin_shp['GRU_area'] = basin_shp_ea.geometry.area
        
        # Add centroid coordinates if not present
        if 'center_lat' not in basin_shp.columns or 'center_lon' not in basin_shp.columns:
            centroids = basin_shp.geometry.centroid
            basin_shp['center_lat'] = centroids.y
            basin_shp['center_lon'] = centroids.x
        
        # Add HRU columns if not present
        if 'HRU_ID' not in basin_shp.columns:
            basin_shp['HRU_ID'] = 1
        if 'HRU_area' not in basin_shp.columns:
            basin_shp['HRU_area'] = basin_shp['GRU_area']  # Same as GRU_area for lumped watershed
        
        # Save the modified basin shapefile
        basin_shp.to_file(basin_target_dir / target_basin_filename)
        print(f"Added required attributes to basin shapefile {target_basin_filename}")
        
        # Also modify the river shapefile (same file for lumped watersheds)
        river_shp = gpd.read_file(river_target_dir / target_river_filename)
        
        # Add required properties for the river shapefile
        if 'LINKNO' not in river_shp.columns:
            river_shp['LINKNO'] = 1
        if 'DSLINKNO' not in river_shp.columns:
            river_shp['DSLINKNO'] = 0  # Outlet
        if 'Length' not in river_shp.columns:
            # For polygons, derive length as perimeter or sqrt of area
            river_shp_ea = river_shp.to_crs('+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96')
            if river_shp.geometry.type.iloc[0] == 'Polygon' or river_shp.geometry.type.iloc[0] == 'MultiPolygon':
                # Use square root of area as approximate river length
                river_shp['Length'] = river_shp_ea.geometry.area.apply(lambda x: x ** 0.5)
            else:
                # Use actual length for LineString geometries
                river_shp['Length'] = river_shp_ea.geometry.length
        if 'Slope' not in river_shp.columns:
            river_shp['Slope'] = 0.001  # Default gentle slope
        
        # Save the modified river shapefile
        river_shp.to_file(river_target_dir / target_river_filename)
        print(f"Added required attributes to river shapefile {target_river_filename}")
        
        return domain_dir, True
        
    except Exception as e:
        print(f"Error modifying shapefile attributes: {e}")
        import traceback
        print(traceback.format_exc())
        return domain_dir, False

def copy_attributes(dataset, watershed_id, domain_dir):
    """
    Copy relevant attribute files from CARAVAN to the CONFLUENCE directory
    
    Args:
        dataset: The CARAVAN dataset name (e.g., 'camels', 'camelsaus', etc.)
        watershed_id: ID of the watershed
        domain_dir: Path to the domain directory
    
    Returns:
        Success flag
    """
    # Source path for attributes
    caravan_base = Path("/work/comphyd_lab/data/misc-data/Caravan/usr/local/google/home/kratzert/Data/Caravan-Jan25-csv")
    attributes_dir = caravan_base / "attributes" / dataset
    
    # Target directory for attributes
    target_attributes_dir = domain_dir / "attributes"
    target_attributes_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all available attribute files for this dataset
    attribute_files = glob.glob(str(attributes_dir / "*.csv"))
    if not attribute_files:
        print(f"Warning: No attribute files found for dataset {dataset}")
        return False
    
    # Copy all attribute files to the domain directory with watershed_id in the filename
    for attr_file in attribute_files:
        attr_filename = os.path.basename(attr_file)
        target_filename = f"{watershed_id}_{attr_filename}"
        target_path = target_attributes_dir / target_filename
        
        # Copy the file
        try:
            # Read the original CSV to extract just this watershed's data
            attr_df = pd.read_csv(attr_file)
            
            # Look for gauge_id column that might contain the watershed_id
            id_col = None
            for col in attr_df.columns:
                if col.lower() in ['gauge_id', 'id', 'basin_id', 'station_id']:
                    id_col = col
                    break
            
            if id_col is None:
                print(f"Warning: Could not find ID column in {attr_filename}. Skipping.")
                continue
            
            # Filter for just this watershed's data
            watershed_attr = attr_df[attr_df[id_col].astype(str).str.contains(watershed_id)]
            
            if len(watershed_attr) == 0:
                print(f"Warning: No data found for watershed {watershed_id} in {attr_filename}. Skipping.")
                continue
            
            # Save the filtered data
            watershed_attr.to_csv(target_path, index=False)
            print(f"Copied and filtered attributes for {watershed_id} to {target_path}")
            
        except Exception as e:
            print(f"Error copying attribute file {attr_file}: {e}")
            return False
    
    return True

def copy_streamflow_data(dataset, watershed_id, domain_dir):
    """
    Copy streamflow data from CARAVAN timeseries to CONFLUENCE observation directory
    
    Args:
        dataset: The CARAVAN dataset name (e.g., 'camels', 'camelsaus', etc.)
        watershed_id: ID of the watershed
        domain_dir: Path to the domain directory
        
    Returns:
        Path to copied streamflow file or None if failed
    """
    # Source path for streamflow data
    caravan_base = Path("/work/comphyd_lab/data/misc-data/Caravan/usr/local/google/home/kratzert/Data/Caravan-Jan25-csv")
    streamflow_dir = caravan_base / "timeseries" / "csv" / dataset
    
    # Target directory for streamflow data
    raw_data_dir = domain_dir / "observations" / "streamflow" / "raw_data"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Define possible patterns for the watershed ID in filenames
    patterns = [
        f"{watershed_id}.csv",              # Just ID
        f"{dataset}_{watershed_id}.csv",    # dataset_ID
        f"{watershed_id}_*.csv",            # ID with suffix
        f"*_{watershed_id}_*.csv",          # ID embedded
        f"*_{watershed_id}.csv",            # ID at end
        f"*{watershed_id}*.csv"             # ID anywhere
    ]
    
    # Look for streamflow file for this watershed
    streamflow_file = None
    for pattern in patterns:
        matches = list(streamflow_dir.glob(pattern))
        if matches:
            streamflow_file = matches[0]
            print(f"Found streamflow file: {streamflow_file}")
            break
    
    if not streamflow_file:
        print(f"Warning: No streamflow data found for watershed {watershed_id} in dataset {dataset}")
        print(f"Searched in: {streamflow_dir}")
        print(f"Using patterns: {patterns}")
        
        # Alternative approach - search all CSV files for ID in first column
        print("Searching all CSV files for watershed ID in data...")
        
        try:
            csv_files = list(streamflow_dir.glob("*.csv"))
            
            for csv_file in csv_files[:10]:  # Limit to first 10 files to avoid excessive processing
                try:
                    # Read just the first few rows to check headers
                    df = pd.read_csv(csv_file, nrows=5)
                    
                    # Check if any column contains the ID
                    for col in df.columns:
                        if any(str(watershed_id) in str(val) for val in df[col]):
                            print(f"Found watershed ID in file: {csv_file}, column: {col}")
                            streamflow_file = csv_file
                            break
                        
                    if streamflow_file:
                        break
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
                
        except Exception as e:
            print(f"Error searching CSV files: {e}")
        
        if not streamflow_file:
            return None
    
    # Target path for the streamflow file
    target_filename = f"{dataset}_{watershed_id}_Discharge.csv"
    target_path = raw_data_dir / target_filename
    
    try:
        # Read the original CSV to ensure it has the right format
        flow_df = pd.read_csv(streamflow_file)
        
        # Check if it has date and streamflow columns
        date_col = None
        flow_col = None
        
        for col in flow_df.columns:
            col_lower = col.lower()
            if col_lower in ['date', 'time', 'datetime', 'timestamp']:
                date_col = col
            elif col_lower in ['streamflow', 'discharge', 'flow', 'q', 'qobs']:
                flow_col = col
        
        if date_col is None or flow_col is None:
            # If columns aren't clearly labeled, assume first column is date and second is flow
            if len(flow_df.columns) >= 2:
                date_col = flow_df.columns[0]
                flow_col = flow_df.columns[1]
            else:
                print(f"Warning: Could not identify date and flow columns in {streamflow_file}")
                return None
        
        # Create a new dataframe with standardized column names
        new_df = pd.DataFrame()
        new_df['date'] = flow_df[date_col]
        new_df['discharge_m3s'] = flow_df[flow_col]
        
        # Save the standardized dataframe
        new_df.to_csv(target_path, index=False)
        print(f"Copied and standardized streamflow data for {watershed_id} to {target_path}")
        
        return target_path
    
    except Exception as e:
        print(f"Error copying streamflow file {streamflow_file}: {e}")
        return None

def calculate_bounding_box(shapefile_path):
    """
    Calculate a bounding box from the shapefile
    
    Args:
        shapefile_path: Path to shapefile
        
    Returns:
        String with bounding box in lat_max/lon_min/lat_min/lon_max format
    """
    try:
        # Read the shapefile
        gdf = gpd.read_file(shapefile_path)
        
        # Ensure it's in lat/lon coordinate system
        if gdf.crs is None:
            print(f"Warning: CRS not defined for {shapefile_path}. Assuming EPSG:4326 (WGS84).")
            gdf.set_crs(epsg=4326, inplace=True)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        
        # Get bounds (minx, miny, maxx, maxy)
        bounds = gdf.total_bounds
        
        # Add a buffer of 0.1 degrees to ensure we include all data
        buffer = 0.1
        bounds = [
            bounds[0] - buffer,  # minx
            bounds[1] - buffer,  # miny
            bounds[2] + buffer,  # maxx
            bounds[3] + buffer   # maxy
        ]
        
        # Convert to CONFLUENCE format (lat_max/lon_min/lat_min/lon_max)
        bounding_box = f"{bounds[3]}/{bounds[0]}/{bounds[1]}/{bounds[2]}"
        return bounding_box
    except Exception as e:
        print(f"Error calculating bounding box: {e}")
        return None

def extract_pour_point(shapefile_path):
    """
    Extract pour point coordinates from the shapefile centroid
    
    Args:
        shapefile_path: Path to shapefile
        
    Returns:
        String with pour point coordinates in lat/lon format
    """
    try:        
        # Read the shapefile
        gdf = gpd.read_file(shapefile_path)
        
        # Ensure it's in lat/lon coordinate system
        if gdf.crs is None:
            print(f"Warning: CRS not defined for {shapefile_path}. Assuming EPSG:4326 (WGS84).")
            gdf.set_crs(epsg=4326, inplace=True)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        
        # Calculate centroid of dissolved shape
        centroid = gdf.dissolve().centroid.iloc[0]
        
        # Format as lat/lon
        pour_point = f"{centroid.y}/{centroid.x}"
        return pour_point
    except Exception as e:
        print(f"Error extracting pour point: {e}")
        return None

def generate_config_file(template_path, output_path, domain_name, basin_name, 
                        river_network_name, bounding_box=None, pour_point=None,
                        streamflow_file=None):
    """
    Generate a new config file based on the template with updated parameters
    
    Args:
        template_path: Path to the template config file
        output_path: Path to save the new config file
        domain_name: Name of the domain to set
        basin_name: Name of the river basin shapefile
        river_network_name: Name of the river network shapefile
        bounding_box: Optional bounding box coordinates
        pour_point: Optional pour point coordinates
        streamflow_file: Optional name of streamflow file
    """
    # Read the template config file
    with open(template_path, 'r') as f:
        config_content = f.read()
    
    # Update the domain name using regex
    config_content = re.sub(r'DOMAIN_NAME:.*', f'DOMAIN_NAME: "{domain_name}"', config_content)

    # Update the data directory to point to the large sample dir
    config_content = re.sub(r'CONFLUENCE_DATA_DIR:.*', f'CONFLUENCE_DATA_DIR: /work/comphyd_lab/data/CONFLUENCE_data/caravan', config_content)

    # Update the river basin and river network names
    config_content = re.sub(r'RIVER_BASINS_NAME:.*', f'RIVER_BASINS_NAME: "{basin_name}"', config_content)
    config_content = re.sub(r'RIVER_NETWORK_SHP_NAME:.*', f'RIVER_NETWORK_SHP_NAME: "{river_network_name}"', config_content)
    
    # Update domain definition method to use lumped
    config_content = re.sub(r'DOMAIN_DEFINITION_METHOD:.*', f'DOMAIN_DEFINITION_METHOD: lumped', config_content)
    
    # Update pour point coordinates if provided
    if pour_point and '/' in pour_point:
        config_content = re.sub(r'POUR_POINT_COORDS:.*', f'POUR_POINT_COORDS: {pour_point}', config_content)
    
    # Update bounding box coordinates if provided
    if bounding_box and '/' in bounding_box:
        config_content = re.sub(r'BOUNDING_BOX_COORDS:.*', f'BOUNDING_BOX_COORDS: {bounding_box}', config_content)
    
    # Update streamflow file name if provided
    if streamflow_file:
        streamflow_name = os.path.basename(streamflow_file)
        config_content = re.sub(r'STREAMFLOW_RAW_NAME:.*', f'STREAMFLOW_RAW_NAME: {streamflow_name}', config_content)
        config_content = re.sub(r'PROCESS_CARAVANS:.*', f'PROCESS_CARAVANS: True', config_content)
        
    # Save the new config file
    with open(output_path, 'w') as f:
        f.write(config_content)
    
    # Verify key settings in the updated config
    with open(output_path, 'r') as f:
        new_content = f.read()
    
    print(f"Config file created at {output_path}")
    
    # Verify key settings
    for label, pattern in {
        'Domain name': r'DOMAIN_NAME:.*',
        'River basin name': r'RIVER_BASINS_NAME:.*',
        'River network name': r'RIVER_NETWORK_SHP_NAME:.*',
        'Domain definition method': r'DOMAIN_DEFINITION_METHOD:.*',
        'Pour point': r'POUR_POINT_COORDS:.*',
        'Bounding box': r'BOUNDING_BOX_COORDS:.*',
        'Streamflow raw name': r'STREAMFLOW_RAW_NAME:.*'
    }.items():
        match = re.search(pattern, new_content)
        if match:
            print(f"  {label}: {match.group().strip()}")
    
    return output_path

def run_confluence(config_path, job_name, dry_run=False):
    """
    Run CONFLUENCE with the specified config file
    
    Args:
        config_path: Path to the config file
        job_name: Name for the job
        dry_run: If True, generate the script but don't submit the job
        
    Returns:
        job_id or None if dry_run is True
    """
    # Create a temporary batch script for this specific run
    batch_script = f"run_{job_name}.sh"
    
    with open(batch_script, 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=CONFLUENCE_{job_name}_%j.log
#SBATCH --error=CONFLUENCE_{job_name}_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8G

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
python /home/darri.eythorsson/code/CONFLUENCE/CONFLUENCE.py --config {config_path}

echo "CONFLUENCE job for {job_name} complete"
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
        print(f"Submitted job for {job_name}, job ID: {job_id}")
        return job_id
    else:
        print(f"Failed to submit job for {job_name}: {result.stderr}")
        return None

def find_watershed_shapefiles(caravan_dir, dataset):
    """
    Find all watershed shapefiles for a specific CARAVAN dataset
    
    Args:
        caravan_dir: Base directory of CARAVAN data
        dataset: Dataset name (e.g., 'camels', 'camelsaus', etc.)
        
    Returns:
        Dictionary mapping watershed IDs to shapefile information
    """    
    # Path to dataset shapefiles
    shapefile_dir = caravan_dir / "shapefiles" / dataset
    
    # Dictionary to store watershed information
    watersheds = {}
    
    # Find all shapefiles in the directory
    shapefile_paths = list(shapefile_dir.glob("**/*.shp"))
    
    if not shapefile_paths:
        print(f"No shapefiles found for dataset {dataset} in {shapefile_dir}")
        return watersheds
    
    print(f"Found {len(shapefile_paths)} shapefiles for dataset {dataset}")
    
    # Check for the special case of a single shapefile with multiple watersheds
    if len(shapefile_paths) == 1:
        print(f"Checking if shapefile contains multiple watersheds with gauge_id column...")
        
        try:
            # Try to read the shapefile
            shapefile_path = shapefile_paths[0]
            gdf = gpd.read_file(shapefile_path)
            
            # Check if there's a gauge_id column
            if 'gauge_id' in gdf.columns:
                print(f"Found gauge_id column with {len(gdf)} watersheds")
                
                # Extract watershed IDs from gauge_id column
                for _, row in gdf.iterrows():
                    watershed_id = str(row['gauge_id'])
                    watersheds[watershed_id] = {
                        "id": watershed_id,
                        "shapefile_path": str(shapefile_path.parent),
                        "shapefile_name": shapefile_path.name,
                        "is_multibasin": True,
                        "basin_id_column": "gauge_id",
                        "geometry": row.geometry
                    }
                
                # Success - return early
                print(f"Successfully identified {len(watersheds)} watersheds from gauge_id column")
                return watersheds
            
            # Look for any other potential ID columns
            potential_id_cols = []
            for col in gdf.columns:
                if any(id_term in col.lower() for id_term in ['id', 'code', 'gauge', 'station', 'basin']):
                    potential_id_cols.append(col)
            
            if potential_id_cols:
                print(f"Found potential ID columns: {potential_id_cols}")
                # Try the first potential ID column
                id_col = potential_id_cols[0]
                print(f"Using {id_col} as watershed ID column")
                
                for _, row in gdf.iterrows():
                    watershed_id = str(row[id_col])
                    watersheds[watershed_id] = {
                        "id": watershed_id,
                        "shapefile_path": str(shapefile_path.parent),
                        "shapefile_name": shapefile_path.name,
                        "is_multibasin": True,
                        "basin_id_column": id_col,
                        "geometry": row.geometry
                    }
                
                print(f"Successfully identified {len(watersheds)} watersheds from {id_col} column")
                return watersheds
                
        except Exception as e:
            print(f"Error reading shapefile as multi-basin: {e}")
            print("Falling back to regular shapefile processing")
    
    # Process each shapefile to extract ID (standard approach for multiple files)
    for shp_path in shapefile_paths:
        try:
            # Try to extract the watershed ID from the filename or parent directory
            filename = shp_path.name
            parent_dir = shp_path.parent.name
            
            # Try different patterns to extract the ID
            watershed_id = None
            
            # Try from filename (remove dataset prefix if present)
            if dataset.lower() in filename.lower():
                # Handle patterns like "camels_01013500.shp" or similar
                match = re.search(fr"{dataset.lower()}_?(\d+)", filename.lower())
                if match:
                    watershed_id = match.group(1)
            else:
                # If dataset name not in filename, just use the filename without extension
                watershed_id = filename.split('.')[0]
            
            # If still no ID, try from parent directory name
            if not watershed_id and parent_dir.lower() != dataset.lower():
                # Use parent directory name if it's not the dataset name
                watershed_id = parent_dir
            
            if watershed_id:
                watersheds[watershed_id] = {
                    "id": watershed_id,
                    "shapefile_path": str(shp_path.parent),
                    "shapefile_name": filename,
                    "is_multibasin": False
                }
            else:
                print(f"Could not extract watershed ID from {shp_path}")
        
        except Exception as e:
            print(f"Error processing shapefile {shp_path}: {e}")
    
    print(f"Successfully identified {len(watersheds)} watersheds for dataset {dataset}")
    return watersheds

def main():
    parser = argparse.ArgumentParser(description='Set up CONFLUENCE runs for CARAVAN dataset watersheds')
    parser.add_argument('--dataset', type=str, default='camels', 
                        help='CARAVAN dataset to process (e.g., camels, camelsaus, camelsbr, etc.)')
    parser.add_argument('--template', type=str, 
                        default='/home/darri.eythorsson/code/CONFLUENCE/0_config_files/config_template.yaml',
                        help='Path to the template config file')
    parser.add_argument('--config-dir', type=str, 
                        default='/home/darri.eythorsson/code/CONFLUENCE/0_config_files/caravan',
                        help='Directory to store generated config files')
    parser.add_argument('--max-watersheds', type=int, default=0, 
                        help='Maximum number of watersheds to process (0 = all)')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Generate configs but do not submit jobs')
    parser.add_argument('--list-datasets', action='store_true',
                        help='List available datasets and exit')
    parser.add_argument('--watersheds-csv', type=str, default=None,
                        help='Path to CSV file with watershed information (will be created if it does not exist)')
    args = parser.parse_args()
    
    # Base paths
    caravan_base = Path("/work/comphyd_lab/data/misc-data/Caravan/usr/local/google/home/kratzert/Data/Caravan-Jan25-csv")
    
    # Check if CARAVAN directory exists
    if not caravan_base.exists():
        print(f"Error: CARAVAN base directory not found at {caravan_base}")
        print("Please check the path and try again")
        return
    
    # List datasets if requested
    if args.list_datasets:
        # Find all dataset directories in shapefile and attributes directories
        shapefile_datasets = [d.name for d in (caravan_base / "shapefiles").iterdir() if d.is_dir()]
        attribute_datasets = [d.name for d in (caravan_base / "attributes").iterdir() if d.is_dir()]
        timeseries_datasets = [d.name for d in (caravan_base / "timeseries" / "csv").iterdir() if d.is_dir()]
        
        # Find datasets that exist in all three directories
        common_datasets = set(shapefile_datasets) & set(attribute_datasets) & set(timeseries_datasets)
        
        print("\nAvailable CARAVAN datasets:")
        print("---------------------------")
        for dataset in sorted(common_datasets):
            print(f" - {dataset}")
        
        print("\nDatasets with shapefiles only:")
        for dataset in sorted(set(shapefile_datasets) - common_datasets):
            print(f" - {dataset}")
        
        return
    
    # Make sure the dataset exists
    shapefile_dir = caravan_base / "shapefiles" / args.dataset
    attributes_dir = caravan_base / "attributes" / args.dataset
    timeseries_dir = caravan_base / "timeseries" / "csv" / args.dataset
    
    for directory, name in [(shapefile_dir, "shapefiles"), (attributes_dir, "attributes"), (timeseries_dir, "timeseries")]:
        if not directory.exists():
            print(f"Error: {name} directory for dataset '{args.dataset}' not found at {directory}")
            print("Use --list-datasets to see available datasets")
            return
    
    # Create config directory if it doesn't exist
    config_dir = Path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Get or create watersheds CSV
    watersheds_csv_path = args.watersheds_csv or f"caravan_{args.dataset}_watersheds.csv"
    
    # Check if we already have watershed info cached
    reload_data = True
    if os.path.exists(watersheds_csv_path):
        print(f"Found existing watershed information at {watersheds_csv_path}")
        choice = input("Reload watershed information? (y/n) [n]: ").lower()
        reload_data = choice == 'y'
    
    # Get watershed information
    if reload_data:
        print(f"Finding watersheds for dataset {args.dataset}...")
        watersheds = find_watershed_shapefiles(caravan_base, args.dataset)
        
        # Convert to DataFrame and save
        watersheds_df = pd.DataFrame.from_dict(watersheds, orient='index')
        watersheds_df.to_csv(watersheds_csv_path, index=False)
        print(f"Saved watershed information to {watersheds_csv_path}")
    else:
        # Load existing watershed information
        print(f"Loading watershed information from {watersheds_csv_path}")
        watersheds_df = pd.read_csv(watersheds_csv_path)
        
        # Convert DataFrame back to dictionary
        watersheds = {}
        for _, row in watersheds_df.iterrows():
            try:
                watersheds[row['id']] = row.to_dict()
            except KeyError:
                # Try to find the ID column
                id_col = None
                for col in row.index:
                    if col.lower() in ['id', 'watershed_id', 'gauge_id']:
                        id_col = col
                        break
                        
                if id_col:
                    watersheds[row[id_col]] = row.to_dict()
                else:
                    print(f"Warning: Could not identify ID column in row: {row}")
    
    # Limit number of watersheds if specified
    watershed_ids = list(watersheds.keys())
    if args.max_watersheds > 0 and args.max_watersheds < len(watershed_ids):
        print(f"Limiting to {args.max_watersheds} watersheds out of {len(watershed_ids)}")
        watershed_ids = watershed_ids[:args.max_watersheds]
    
    # Prepare for running CONFLUENCE
    template_config_path = args.template
    submitted_jobs = []
    
    # Process each watershed
    for watershed_id in watershed_ids:
        watershed_info = watersheds[watershed_id]
        
        print(f"\nProcessing watershed {watershed_id}...")

        # Create domain name combining dataset and watershed ID
        domain_name = f"{args.dataset}_{watershed_id}"        

        # Check if the simulations directory already exists
        simulation_dir = Path(f"/work/comphyd_lab/data/CONFLUENCE_data/caravan/domain_{domain_name}/simulations")
        
        if simulation_dir.exists():
            print(f"Skipping {domain_name} - simulation directory already exists: {simulation_dir}")
            continue

        # Get shapefile information
        basin_path = watershed_info['shapefile_path']
        basin_file = watershed_info['shapefile_name']
        is_multibasin = watershed_info.get('is_multibasin', False)
        basin_id_column = watershed_info.get('basin_id_column', None)
        geometry = watershed_info.get('geometry', None)
        
        # Full paths for basin shapefile
        basin_shapefile_path = os.path.join(basin_path, basin_file)
        
        # Set up CONFLUENCE directory structure and copy files
        print(f"Setting up CONFLUENCE directory for {domain_name}...")
        domain_dir, setup_success = setup_confluence_directory(
            args.dataset, 
            watershed_id, 
            basin_path,
            basin_file,
            basin_id_column=basin_id_column,
            is_multibasin=is_multibasin,
            geometry=geometry
        )
        
        if not setup_success:
            print(f"Failed to set up CONFLUENCE directory for {domain_name}. Skipping.")
            continue
        
        # Use the new filenames for basin and river shapefiles
        target_basin_filename = f"{args.dataset}_{watershed_id}_basin.shp"
        target_river_filename = f"{args.dataset}_{watershed_id}_river.shp"
        
        # Calculate bounding box from the shapefile
        basin_shapefile_path = os.path.join(domain_dir, "shapefiles", "river_basins", target_basin_filename)
        bounding_box = calculate_bounding_box(basin_shapefile_path)
        if bounding_box:
            print(f"Calculated bounding box: {bounding_box}")
        else:
            print(f"Warning: Could not calculate bounding box for {domain_name}")
        
        # Extract pour point from the shapefile
        pour_point = extract_pour_point(basin_shapefile_path)
        if pour_point:
            print(f"Extracted pour point: {pour_point}")
        else:
            print(f"Warning: Could not extract pour point for {domain_name}")
        
        # Copy attribute data
        print(f"Copying attribute data for {domain_name}...")
        copy_attributes(args.dataset, watershed_id, domain_dir)
        
        # Copy streamflow data
        print(f"Copying streamflow data for {domain_name}...")
        streamflow_file = copy_streamflow_data(args.dataset, watershed_id, domain_dir)
        
        # Generate the config file
        config_filename = f"config_{domain_name}.yaml"
        config_path = config_dir / config_filename
        
        print(f"Generating config file for {domain_name}...")
        generate_config_file(
            template_config_path, 
            config_path, 
            domain_name, 
            target_basin_filename,  
            target_river_filename,  
            bounding_box,
            pour_point,
            streamflow_file
        )
        
        # Run CONFLUENCE with the generated config
        job_name = f"{args.dataset}_{watershed_id}"
        
        if args.dry_run:
            print(f"Dry run - preparing job for {domain_name}")
            job_id = run_confluence(config_path, job_name, dry_run=True)
            submitted_jobs.append((domain_name, "DRY_RUN"))
        else:
            print(f"Submitting CONFLUENCE job for {domain_name}")
            job_id = run_confluence(config_path, job_name)
            
            if job_id:
                submitted_jobs.append((domain_name, job_id))
            
            # Add a small delay between job submissions
            time.sleep(10)
    
    # Print summary of submitted jobs
    print("\nSummary of processed watersheds:")
    print(f"Total watersheds processed: {len(submitted_jobs)}")
    
    if args.dry_run:
        print("\nDry run summary:")
        print(f"Created config files and job scripts for {len(submitted_jobs)} watersheds")
        print("To submit these jobs, run the generated shell scripts")
    else:
        print("\nSubmitted jobs summary:")
        for domain_name, job_id in submitted_jobs:
            print(f"  {domain_name}: Job ID {job_id}")

if __name__ == "__main__":
    main()
