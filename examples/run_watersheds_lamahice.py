#!/usr/bin/env python
import pandas as pd
import os
import yaml
import subprocess
from pathlib import Path
import time
import sys
import re
import glob
import shutil
import geopandas as gpd
import argparse

def setup_confluence_directory(watershed_id, basin_source_path, basin_filename, river_source_path, river_filename):
    """
    Set up the CONFLUENCE directory structure for a watershed and copy relevant shapefiles.
    For LamaH-Ice implementation, adds GRU_area and sets gru_to_seg and GRU_ID to equal id.
    
    Args:
        watershed_id: ID of the watershed (e.g., lamahice_12)
        basin_source_path: Path to the source basin shapefile directory
        basin_filename: Name of the basin shapefile
        river_source_path: Path to the source river shapefile directory
        river_filename: Name of the river shapefile
        
    Returns:
        Tuple of (basin_target_path, river_target_path)
    """
    # Base CONFLUENCE data directory
    confluence_data_dir = Path("/anvil/projects/x-ees240082/data/CONFLUENCE_data")
    
    # Create the domain directory structure
    domain_dir = confluence_data_dir / "lamahice" / f"domain_{watershed_id}"
    basin_target_dir = domain_dir / "shapefiles" / "river_basins"
    river_target_dir = domain_dir / "shapefiles" / "river_network"
    
    # Create directories if they don't exist
    for directory in [domain_dir, basin_target_dir, river_target_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Get the basin shapefile and associated files
    basin_source_base = str(Path(basin_source_path) / basin_filename).rsplit('.', 1)[0]
    basin_target_base = str(basin_target_dir / basin_filename).rsplit('.', 1)[0]
    
    # For the lumped case - convert the lumped shapefile into basin and river formats
    is_lumped = True  # LamaH-Ice is using lumped watersheds
    
    # Copy all shapefile components (shp, shx, dbf, prj, etc.)
    for extension in ['shp', 'shx', 'dbf', 'prj', 'cpg']:
        # Copy basin files
        basin_source_file = f"{basin_source_base}.{extension}"
        basin_target_file = f"{basin_target_base}.{extension}"
        if os.path.exists(basin_source_file):
            shutil.copy2(basin_source_file, basin_target_file)
            print(f"Copied {basin_source_file} to {basin_target_file}")
        
        # Copy river files (or reuse the same file if it's a lumped watershed)
        if is_lumped:
            # For lumped case, use the basin file for the river network too
            river_target_base = str(river_target_dir / basin_filename).rsplit('.', 1)[0]
            river_source_file = basin_source_file
        else:
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
        
        # Add GRU_ID and gru_to_seg fields based on id
        if 'id' in basin_shp.columns:
            # Extract numeric ID from id (e.g., lamahice_12 -> 12)
            # But keep full id for GRU_ID and gru_to_seg
            basin_shp['GRU_ID'] = basin_shp['id']
            basin_shp['gru_to_seg'] = basin_shp['id']
            print(f"Set GRU_ID and gru_to_seg to match id values")
        else:
            # Fallback if id doesn't exist
            basin_shp['GRU_ID'] = watershed_id
            basin_shp['gru_to_seg'] = watershed_id
            print(f"Warning: id column not found. Set GRU_ID and gru_to_seg to watershed_id")
        
        # Calculate GRU_area in square meters
        if basin_shp.crs is None:
            print(f"Warning: CRS not defined for {basin_filename}. Trying to set to EPSG:4326 (WGS84).")
            basin_shp.set_crs(epsg=4326, inplace=True)
        
        # Convert to equal area projection for accurate area calculation
        basin_shp_ea = basin_shp.to_crs('+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96')
        basin_shp['GRU_area'] = basin_shp_ea.geometry.area
        print(f"Calculated GRU_area based on geometry")
        
        # Add other required properties for the basin shapefile
        basin_shp['center_lat'] = basin_shp.geometry.centroid.y
        basin_shp['center_lon'] = basin_shp.geometry.centroid.x
        basin_shp['HRU_ID'] = basin_shp['GRU_ID']
        basin_shp['HRU_area'] = basin_shp['GRU_area']
        
        # Save the modified shapefile
        basin_shp.to_file(f"{basin_target_base}.shp")
        print(f"Added GRU_ID, GRU_area, gru_to_seg, center_lat, center_lon, HRU_ID, and HRU_area columns to {basin_filename}")
        
        # For lumped case, need to add river properties to the basin file used as river
        river_shp = gpd.read_file(f"{river_target_base}.shp")
        
        # Add required properties for the river shapefile
        if 'LINKNO' not in river_shp.columns:
            river_shp['LINKNO'] = 1
        
        if 'DSLINKNO' not in river_shp.columns:
            river_shp['DSLINKNO'] = 0  # Outlet
        
        if 'Length' not in river_shp.columns:
            # For polygons, use the perimeter as an approximation of river length
            river_shp_ea = river_shp.to_crs('+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96')
            # Simplified river length calculation - use square root of area as an approximation
            river_shp['Length'] = river_shp_ea.geometry.area.apply(lambda x: x ** 0.5)
        
        if 'Slope' not in river_shp.columns:
            river_shp['Slope'] = 0.001  # Default gentle slope
            
        # Save the modified river shapefile
        river_shp.to_file(f"{river_target_base}.shp")
        print(f"Added LINKNO, DSLINKNO, Length, and Slope columns to river file")
        
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
        # Read the shapefile
        gdf = gpd.read_file(shapefile_path)
        
        # Ensure it's in lat/lon coordinate system
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        
        # Get bounds (minx, miny, maxx, maxy)
        bounds = gdf.total_bounds
        
        # Add a small buffer to the bounding box (0.1 degrees) to ensure coverage
        bounds_buffered = [
            bounds[0] - 0.1,  # minx - buffer
            bounds[1] - 0.1,  # miny - buffer
            bounds[2] + 0.1,  # maxx + buffer
            bounds[3] + 0.1   # maxy + buffer
        ]
        
        # Convert to CONFLUENCE format (lat_max/lon_min/lat_min/lon_max)
        bounding_box = f"{bounds_buffered[3]}/{bounds_buffered[0]}/{bounds_buffered[1]}/{bounds_buffered[2]}"
        return bounding_box
    except Exception as e:
        print(f"Error calculating bounding box: {e}")
        return None

def generate_config_file(template_path, output_path, domain_name, basin_path, basin_name, 
                       river_network_path, river_network_name, bounding_box=None, pour_point=None,
                       station_id=None):
    """
    Generate a new config file based on the template with updated parameters for LamaH-Ice basins
    
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
    """
    # Read the template config file
    with open(template_path, 'r') as f:
        config_content = f.read()
    
    # Update the domain name using regex
    config_content = re.sub(r'DOMAIN_NAME:.*', f'DOMAIN_NAME: "{domain_name}"', config_content)

    # Update the domain name using regex
    config_content = re.sub(r'CONFLUENCE_DATA_DIR:.*', f'CONFLUENCE_DATA_DIR: /anvil/projects/x-ees240082/data/CONFLUENCE_data/lamahice', config_content)

    # Update domain definition method to use lumped
    config_content = re.sub(r'DOMAIN_DEFINITION_METHOD:.*', f'DOMAIN_DEFINITION_METHOD: lumped', config_content)
    
    # Update domain discretization to use GRUs
    config_content = re.sub(r'DOMAIN_DISCRETIZATION:.*', f'DOMAIN_DISCRETIZATION: GRUs', config_content)

    # Update optimisation method to use GRUs
    config_content = re.sub(r'OPTIMISATION_METHODS:.*', f'OPTIMISATION_METHODS: [iteration]', config_content)

    # Update domain shapefile 
    config_content = re.sub(r'RIVER_BASINS_NAME:.*', f'RIVER_BASINS_NAME: temp_basin.shp', config_content)

    # Update pour point coordinates if provided and valid
    if pour_point and str(pour_point).lower() != 'nan' and '/' in str(pour_point):
        config_content = re.sub(r'POUR_POINT_COORDS:.*', f'POUR_POINT_COORDS: {pour_point}', config_content)
    
    # Update bounding box coordinates if provided and valid
    if bounding_box and str(bounding_box).lower() != 'nan' and '/' in str(bounding_box):
        config_content = re.sub(r'BOUNDING_BOX_COORDS:.*', f'BOUNDING_BOX_COORDS: {bounding_box}', config_content)
    
    # Set SIM_REACH_ID to 1 for lumped catchments
    config_content = re.sub(r'SIM_REACH_ID:.*', f'SIM_REACH_ID: 1', config_content)
    
    # Update streamflow data provider to VI for LamaH-Ice
    config_content = re.sub(r'STREAMFLOW_DATA_PROVIDER:.*', f'STREAMFLOW_DATA_PROVIDER: VI', config_content)
    
    # Add or update download flags
    if re.search(r'DOWNLOAD_USGS_DATA:', config_content):
        config_content = re.sub(r'DOWNLOAD_USGS_DATA:.*', f'DOWNLOAD_USGS_DATA: False', config_content)
    else:
        # Add it after the STREAMFLOW_DATA_PROVIDER line
        config_content = re.sub(
            r'(STREAMFLOW_DATA_PROVIDER:.*)',
            f'\\1\nDOWNLOAD_USGS_DATA: False',
            config_content
        )
        
    if re.search(r'DOWNLOAD_WSC_DATA:', config_content):
        config_content = re.sub(r'DOWNLOAD_WSC_DATA:.*', f'DOWNLOAD_WSC_DATA: False', config_content)
    else:
        # Add it after the DOWNLOAD_USGS_DATA line or after STREAMFLOW_DATA_PROVIDER if not found
        if re.search(r'DOWNLOAD_USGS_DATA:', config_content):
            config_content = re.sub(
                r'(DOWNLOAD_USGS_DATA:.*)',
                f'\\1\nDOWNLOAD_WSC_DATA: False',
                config_content
            )
        else:
            config_content = re.sub(
                r'(STREAMFLOW_DATA_PROVIDER:.*)',
                f'\\1\nDOWNLOAD_WSC_DATA: False',
                config_content
            )
    
    # Update station ID - extract from id if not provided
    if not station_id and domain_name:
        # Extract ID number from domain_name (e.g., lamahice_12 -> 12)
        match = re.search(r'(\d+)$', domain_name)
        if match:
            station_id = match.group(1)
    
    # Update STATION_ID if provided or extracted
    if station_id:
        if re.search(r'STATION_ID:', config_content):
            config_content = re.sub(r'STATION_ID:.*', f'STATION_ID: {station_id}', config_content)
        else:
            # Add it after the DOWNLOAD_WSC_DATA line
            if re.search(r'DOWNLOAD_WSC_DATA:', config_content):
                config_content = re.sub(
                    r'(DOWNLOAD_WSC_DATA:.*)',
                    f'\\1\nSTATION_ID: {station_id}',
                    config_content
                )
            else:
                config_content = re.sub(
                    r'(STREAMFLOW_DATA_PROVIDER:.*)',
                    f'\\1\nSTATION_ID: {station_id}',
                    config_content
                )
        
    # Save the new config file
    with open(output_path, 'w') as f:
        f.write(config_content)
    
    print(f"Config file created at {output_path}")
    
    # Verify the changes were made
    print(f"Checking for proper updates...")
    
    with open(output_path, 'r') as f:
        new_content = f.read()
    
    return output_path

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
#SBATCH --mem=1G

# Load necessary modules
module restore confluence_modules


# Activate Python environment
conda activate confluence 

# Run CONFLUENCE with the specified config
python /home/x-deythorsson/code/CONFLUENCE/CONFLUENCE.py --config {config_path}

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

def copy_streamflow_data(id, domain_dir):
    """
    Copy relevant streamflow data files to the domain directory
    
    Args:
        id: ID of the gauge/watershed (e.g., lamahice_12)
        domain_dir: Path to the domain directory
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Extract numeric ID from id (e.g., lamahice_12 -> 12)
        id_str = str(id)
        numeric_id = re.search(r'(\d+)$', id_str)
        if not numeric_id:
            print(f"Warning: Could not extract numeric ID from {id}")
            return False
            
        numeric_id = numeric_id.group(1)
        
        # Source directory with LamaH-Ice streamflow data
        source_dir = Path("/anvil/projects/x-ees240082/data/geospatial-data/lamah_ice/D_gauges/2_timeseries/daily_filtered/")
        
        # Find the file matching the ID (e.g., ID_12.csv)
        source_file = source_dir / f"ID_{numeric_id}.csv"
        
        if not source_file.exists():
            print(f"Warning: Streamflow data file not found at {source_file}")
            return False
            
        # Create the target directory structure
        target_dir = Path(domain_dir) / "observations" / "streamflow" / "raw_data"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the file
        target_file = target_dir / f"{id}_streamflow.csv"
        shutil.copy2(source_file, target_file)
        print(f"Copied streamflow data from {source_file} to {target_file}")
        
        return True
        
    except Exception as e:
        print(f"Error copying streamflow data: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def process_lamahice_watersheds(gauges_file, basins_file, output_dir, template_config, 
                               max_watersheds=None, submit_jobs=False, dry_run=False):
    """
    Process LamaH-Ice watersheds and optionally submit CONFLUENCE jobs
    
    Args:
        gauges_file: Path to the gauges shapefile
        basins_file: Path to the basins shapefile
        output_dir: Directory to save config files
        template_config: Path to the template config file
        max_watersheds: Maximum number of watersheds to process (None for all)
        submit_jobs: Whether to submit CONFLUENCE jobs
        dry_run: If True, generate scripts without submitting jobs
        
    Returns:
        DataFrame with information about processed watersheds
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load gauge and basin shapefiles
    gauges = gpd.read_file(gauges_file)
    basins = gpd.read_file(basins_file)
    
    print(f"Loaded {len(gauges)} gauges and {len(basins)} basins")
    print(f"Gauge columns: {gauges.columns.tolist()}")
    print(f"Basin columns: {basins.columns.tolist()}")
    
    # Verify they share the same id field
    if 'id' not in gauges.columns or 'id' not in basins.columns:
        print("Error: id field missing from one or both shapefiles")
        return pd.DataFrame()
    
    # Create a data frame to track processing
    watershed_data = []
    
    # Limit the number of watersheds if specified
    if max_watersheds is not None:
        gauges = gauges.head(int(max_watersheds))
    
    # Track submitted jobs
    submitted_jobs = []
    
    # Create temporary filenames for extracting individual basin shapes
    temp_basin_file = "temp_basin.shp"
    
    # Process each gauge
    for idx, gauge in gauges.iterrows():
        id = gauge['id']
        print(f"\nProcessing gauge {id} ({idx+1}/{len(gauges)})")
        
        # Extract the corresponding basin
        basin = basins[basins['id'] == id]
        
        if len(basin) == 0:
            print(f"Warning: No basin found for gauge {id}")
            continue
        
        # Extract pour point coordinates from gauge
        if 'gauge_lat' in gauge and 'gauge_lon' in gauge:
            pour_point = f"{gauge['gauge_lon']}/{gauge['gauge_lat']}"
        else:
            pour_point = None
        
        # Save the basin as a temporary shapefile
        temp_dir = Path(output_dir) / "temp"
        temp_dir.mkdir(exist_ok=True)
        temp_basin_path = temp_dir / temp_basin_file
        basin.to_file(temp_basin_path)
        
        # Set up CONFLUENCE directory structure and copy/modify shapefiles
        basin_target_dir, river_target_dir = setup_confluence_directory(
            id,
            str(temp_dir),
            temp_basin_file,
            str(temp_dir),  # Same for lumped case
            temp_basin_file  # Same for lumped case
        )
        
        if not basin_target_dir or not river_target_dir:
            print(f"Error setting up CONFLUENCE directory for {id}")
            continue
        
        # Calculate bounding box from the basin shapefile
        basin_shapefile_path = os.path.join(basin_target_dir, temp_basin_file)
        bounding_box = calculate_fresh_bounding_box(basin_shapefile_path)
        
        # Extract numeric ID for station ID
        numeric_id = None
        # Convert id to string to handle non-string types
        id_str = str(id)
        match = re.search(r'(\d+)$', id_str)
        if match:
            numeric_id = match.group(1)
            
        # Generate config file
        config_path = os.path.join(output_dir, f"config_{id}.yaml")
        generate_config_file(
            template_config,
            config_path,
            id,
            str(basin_target_dir),
            temp_basin_file,
            str(river_target_dir),
            temp_basin_file,
            bounding_box,
            pour_point,
            numeric_id  # Pass the numeric ID as station_id
        )
        
        # Copy streamflow data to domain directory
        domain_dir = Path("/anvil/projects/x-ees240082//data/CONFLUENCE_data/lamahice") / f"domain_{id}"
        copy_success = copy_streamflow_data(id, domain_dir)

        # Add to tracking data
        watershed_info = {
            'id': id,
            'basin_file': temp_basin_file,
            'config_file': config_path,
            'pour_point': pour_point,
            'bounding_box': bounding_box
        }
        
        if 'gauge_name' in gauge:
            watershed_info['gauge_name'] = gauge['gauge_name']
        
        if 'river' in gauge:
            watershed_info['river'] = gauge['river']
        
        watershed_data.append(watershed_info)
        
        # Submit job if requested
        if submit_jobs:
            if dry_run:
                job_id = run_confluence(config_path, id, dry_run=True)
                submitted_jobs.append((id, "DRY_RUN"))
            else:
                job_id = run_confluence(config_path, id)
                if job_id:
                    submitted_jobs.append((id, job_id))
            
            # Add a small delay between job submissions
            time.sleep(2)
    
    # Create summary DataFrame
    watersheds_df = pd.DataFrame(watershed_data)
    
    # Save to CSV for reference
    csv_path = os.path.join(output_dir, "lamahice_watersheds.csv")
    watersheds_df.to_csv(csv_path, index=False)
    print(f"Saved watershed information to {csv_path}")
    
    # Print summary of submitted jobs
    if submit_jobs:
        if dry_run:
            print("\nDry run summary:")
            print(f"Created config files and job scripts for {len(submitted_jobs)} watersheds")
            print(f"To submit the jobs, run the scripts in the current directory")
        else:
            print("\nSubmitted jobs summary:")
            for domain_name, job_id in submitted_jobs:
                print(f"Domain: {domain_name}, Job ID: {job_id}")
            
            print(f"\nTotal jobs submitted: {len(submitted_jobs)}")
    
    # Remove temporary directory
    shutil.rmtree(temp_dir)
    
    return watersheds_df

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process LamaH-Ice watersheds for CONFLUENCE')
    parser.add_argument('--gauges', type=str, 
                       default='/anvil/projects/x-ees240082/data/geospatial-data/lamah_ice/D_gauges/3_shapefiles/gauges.shp',
                       help='Path to the gauges shapefile')
    parser.add_argument('--basins', type=str, 
                       default='/anvil/projects/x-ees240082/data/geospatial-data/lamah_ice/A_basins_total_upstrm/3_shapefiles/Basins_A.shp',
                       help='Path to the basin shapes shapefile')
    parser.add_argument('--template', type=str, 
                       default='/home/x-deythorsson/code/CONFLUENCE/0_config_files/config_Bow_lumped.yaml',
                       help='Path to the template config file')
    parser.add_argument('--output', type=str, 
                       default='/home/x-deythorsson/code/CONFLUENCE/0_config_files/lamahice',
                       help='Directory to save config files')
    parser.add_argument('--max', type=int, default=None,
                       help='Maximum number of watersheds to process')
    parser.add_argument('--submit', action='store_true',
                       help='Submit CONFLUENCE jobs')
    parser.add_argument('--dry-run', action='store_true',
                       help='Generate scripts without submitting jobs')
    
    args = parser.parse_args()
    
    # Verify files exist
    for file_path in [args.gauges, args.basins, args.template]:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return
    
    # Process watersheds
    watersheds = process_lamahice_watersheds(
        args.gauges,
        args.basins,
        args.output,
        args.template,
        args.max,
        args.submit,
        args.dry_run
    )
    
    print(f"Processed {len(watersheds)} watersheds")

if __name__ == "__main__":
    main()

