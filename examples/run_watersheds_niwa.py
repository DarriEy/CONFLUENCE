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
import netCDF4 as nc
import numpy as np
from datetime import datetime

def setup_confluence_directory(station_id, station_name, latitude, longitude, nz_shapefile=None):
    """
    Set up the CONFLUENCE directory structure for a NZ watershed
    
    Args:
        station_id: ID of the station
        station_name: Name of the station
        latitude: Station latitude
        longitude: Station longitude
        nz_shapefile: Optional path to shapefile for this catchment
        
    Returns:
        Tuple of (basin_target_path, river_target_path)
    """
    # Base CONFLUENCE data directory
    confluence_data_dir = Path("/anvil/projects/x-ees240082/data/CONFLUENCE_data")
    
    # Create watershed ID with NZ prefix
    watershed_id = f"NZ_{station_id}"
    
    # Create the domain directory structure
    domain_dir = confluence_data_dir / "niwa" / f"domain_{watershed_id}"
    basin_target_dir = domain_dir / "shapefiles" / "river_basins"
    river_target_dir = domain_dir / "shapefiles" / "river_network"
    
    # Create directories if they don't exist
    for directory in [domain_dir, basin_target_dir, river_target_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # If we have a shapefile, use it, otherwise create a simple point shapefile
    if nz_shapefile and os.path.exists(nz_shapefile):
        # Copy existing shapefile
        basin_filename = f"{watershed_id}_basin.shp"
        basin_target_base = str(basin_target_dir / basin_filename).rsplit('.', 1)[0]
        
        # Copy all shapefile components (shp, shx, dbf, prj, etc.)
        for extension in ['shp', 'shx', 'dbf', 'prj', 'cpg']:
            source_file = f"{nz_shapefile.rsplit('.', 1)[0]}.{extension}"
            target_file = f"{basin_target_base}.{extension}"
            if os.path.exists(source_file):
                shutil.copy2(source_file, target_file)
                print(f"Copied {source_file} to {target_file}")
    else:
        # Create a simple point buffer shapefile for the catchment
        try:
            # Create a point geometry
            point = gpd.GeoDataFrame({
                'gauge_id': [watershed_id],
                'name': [station_name],
                'geometry': [gpd.points_from_xy([longitude], [latitude])[0]]
            }, crs="EPSG:4326")
            
            # Buffer the point to create a simple polygon (5km buffer)
            # First convert to projected CRS for accurate buffering
            point_proj = point.to_crs("+proj=aea +lat_1=-30 +lat_2=-45 +lat_0=-40 +lon_0=175")
            buffered = point_proj.buffer(5000)  # 5km buffer
            polygon = gpd.GeoDataFrame({
                'gauge_id': point['gauge_id'],
                'name': point['name'],
                'geometry': buffered
            }, crs=point_proj.crs)
            
            # Convert back to WGS84
            polygon = polygon.to_crs("EPSG:4326")
            
            # Save as basin shapefile
            basin_filename = f"{watershed_id}_basin.shp"
            basin_target = basin_target_dir / basin_filename
            polygon.to_file(basin_target)
            print(f"Created simple buffer shapefile at {basin_target}")
        except Exception as e:
            print(f"Error creating buffer shapefile: {e}")
            basin_filename = None
    
    # For lumped approach, use the same shapefile for river network
    if basin_filename:
        river_filename = basin_filename
        river_target = river_target_dir / river_filename
        
        # Copy from basin to river if needed
        if not os.path.exists(river_target):
            # Copy all shapefile components
            for extension in ['shp', 'shx', 'dbf', 'prj', 'cpg']:
                source_file = f"{str(basin_target_dir / basin_filename).rsplit('.', 1)[0]}.{extension}"
                target_file = f"{str(river_target_dir / river_filename).rsplit('.', 1)[0]}.{extension}"
                if os.path.exists(source_file):
                    shutil.copy2(source_file, target_file)
    else:
        river_filename = None
        
    # Now modify the shapefiles to add required attributes
    try:
        if basin_filename:
            # Read the basin shapefile
            basin_shp = gpd.read_file(basin_target_dir / basin_filename)
            
            # Add GRU_ID and gru_to_seg fields
            basin_shp['GRU_ID'] = watershed_id
            basin_shp['gru_to_seg'] = 1
            basin_shp['HRU_ID'] = watershed_id
            
            # Calculate GRU_area in square meters
            if basin_shp.crs is None:
                print(f"Warning: CRS not defined for {basin_filename}. Setting to EPSG:4326 (WGS84).")
                basin_shp.set_crs(epsg=4326, inplace=True)
            
            # Convert to equal area projection for accurate area calculation
            basin_shp_ea = basin_shp.to_crs('+proj=aea +lat_1=-30 +lat_2=-45 +lat_0=-40 +lon_0=175')
            basin_shp['GRU_area'] = basin_shp_ea.geometry.area
            basin_shp['HRU_area'] = basin_shp['GRU_area']
            print(f"Calculated GRU_area based on geometry")
            
            # Add centroid coordinates
            basin_shp['center_lat'] = basin_shp.geometry.centroid.y
            basin_shp['center_lon'] = basin_shp.geometry.centroid.x
            
            # Save the modified basin shapefile
            basin_shp.to_file(basin_target_dir / basin_filename)
            print(f"Added GRU_ID, GRU_area, and other attributes to {basin_filename}")
            
            # Modify river shapefile (for lumped, this is the same as basin)
            river_shp = gpd.read_file(river_target_dir / river_filename)
            
            # Add required properties for the river shapefile
            if 'LINKNO' not in river_shp.columns:
                river_shp['LINKNO'] = 1
            
            if 'DSLINKNO' not in river_shp.columns:
                river_shp['DSLINKNO'] = 0  # Outlet
            
            if 'Length' not in river_shp.columns:
                # For polygons, use the perimeter as an approximation of river length
                river_shp_ea = river_shp.to_crs('+proj=aea +lat_1=-30 +lat_2=-45 +lat_0=-40 +lon_0=175')
                river_shp['Length'] = river_shp_ea.geometry.area ** 0.5  # Simplified approximation
            
            if 'Slope' not in river_shp.columns:
                river_shp['Slope'] = 0.001  # Default gentle slope
                
            # Save the modified river shapefile
            river_shp.to_file(river_target_dir / river_filename)
            print(f"Added LINKNO, DSLINKNO, Length, and Slope columns to river file")
            
            return basin_target_dir, river_target_dir, basin_filename, river_filename
        else:
            return None, None, None, None
            
    except Exception as e:
        print(f"Error modifying shapefile attributes: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None, None, None

def calculate_bounding_box(lat, lon, buffer_degrees=0.1):
    """
    Calculate a bounding box around a point with a buffer
    
    Args:
        lat: Latitude of point
        lon: Longitude of point
        buffer_degrees: Buffer in degrees to add around point
        
    Returns:
        String with bounding box in lat_max/lon_min/lat_min/lon_max format
    """
    try:
        # Add buffer to create bounding box
        lat_min = lat - buffer_degrees
        lat_max = lat + buffer_degrees
        lon_min = lon - buffer_degrees
        lon_max = lon + buffer_degrees
        
        # Format in CONFLUENCE format (lat_max/lon_min/lat_min/lon_max)
        bounding_box = f"{lat_max}/{lon_min}/{lat_min}/{lon_max}"
        return bounding_box
    except Exception as e:
        print(f"Error calculating bounding box: {e}")
        return None

def extract_station_timeseries(netcdf_file, station_id, output_file):
    """
    Extract timeseries data for a specific station from a NetCDF file
    
    Args:
        netcdf_file: Path to NetCDF file
        station_id: ID of the station to extract
        output_file: Path to save extracted data
        
    Returns:
        Boolean indicating success
    """
    try:
        # Open NetCDF file
        ds = nc.Dataset(netcdf_file, 'r')
        
        # Get station index
        station_ids = ds.variables['station_id'][:]
        
        # Find the index of the requested station
        station_index = None
        for i, sid in enumerate(station_ids):
            if sid == station_id:
                station_index = i
                break
        
        if station_index is None:
            print(f"Station ID {station_id} not found in NetCDF file")
            return False
        
        # Extract timestamp information
        time_var = ds.variables['time']
        time_units = time_var.units
        calendar = getattr(time_var, 'calendar', 'standard')
        
        # Extract time values as datetime objects
        times = nc.num2date(time_var[:], units=time_var.units, calendar=calendar)
        
        # Extract flow data for this station
        flow_data = ds.variables['river_flow_rate'][:, station_index]
        
        # Create DataFrame
        data = {
            'datetime': times,
            'discharge_cms': flow_data
        }
        df = pd.DataFrame(data)
        
        # Replace NaN values with -9999
        df['discharge_cms'] = df['discharge_cms'].replace(-9999, np.nan)
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        
        print(f"Extracted timeseries for station {station_id} to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error extracting station data: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def generate_config_file(template_path, output_path, domain_name, basin_path, basin_name, 
                       river_network_path, river_network_name, bounding_box=None, pour_point=None,
                       station_id=None):
    """
    Generate a new config file based on the template with updated parameters for NZ basins
    
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
        station_id: Station ID for streamflow data
    """
    # Read the template config file
    with open(template_path, 'r') as f:
        config_content = f.read()
    
    # Update the domain name using regex
    config_content = re.sub(r'DOMAIN_NAME:.*', f'DOMAIN_NAME: "{domain_name}"', config_content)
    
    # Update domain definition method to use lumped
    config_content = re.sub(r'DOMAIN_DEFINITION_METHOD:.*', f'DOMAIN_DEFINITION_METHOD: lumped', config_content)

    # Update the domain name using regex
    config_content = re.sub(r'CONFLUENCE_DATA_DIR:.*', f'CONFLUENCE_DATA_DIR: /anvil/projects/x-ees240082/data/CONFLUENCE_data/niwa', config_content)

    # Update domain discretization to use GRUs
    config_content = re.sub(r'DOMAIN_DISCRETIZATION:.*', f'DOMAIN_DISCRETIZATION: GRUs', config_content)
    
    config_content = re.sub(r'EM_EARTH_PRCP_DIR:.*', f'EM_EARTH_PRCP_DIR: /anvil/datasets/meteorological/EM-Earth/EM_Earth_v1/deterministic_hourly/prcp/Oceania', config_content)
    config_content = re.sub(r'EM_EARTH_TMEAN_DIR:.*', f'EM_EARTH_TMEAN_DIR: /anvil/datasets/meteorological/EM-Earth/EM_Earth_v1/deterministic_hourly/tmean/Oceania', config_content)
    config_content = re.sub(r'EM_EARTH_REGION:.*', f'EM_EARTH_REGION: Oceania', config_content)

    # Update pour point coordinates if provided and valid
    if pour_point and str(pour_point).lower() != 'nan' and '/' in str(pour_point):
        config_content = re.sub(r'POUR_POINT_COORDS:.*', f'POUR_POINT_COORDS: {pour_point}', config_content)
    
    # Update bounding box coordinates if provided and valid
    if bounding_box and str(bounding_box).lower() != 'nan' and '/' in str(bounding_box):
        config_content = re.sub(r'BOUNDING_BOX_COORDS:.*', f'BOUNDING_BOX_COORDS: {bounding_box}', config_content)
    
    # Set SIM_REACH_ID to 1 for lumped catchments
    config_content = re.sub(r'SIM_REACH_ID:.*', f'SIM_REACH_ID: 1', config_content)
    
    # Update streamflow data provider to NIWA
    config_content = re.sub(r'STREAMFLOW_DATA_PROVIDER:.*', f'STREAMFLOW_DATA_PROVIDER: NIWA', config_content)
    
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
    
    # Update STATION_ID if provided
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
    
    # Verify key settings
    patterns = {
        'Domain name': r'DOMAIN_NAME:.*',
        'Domain definition method': r'DOMAIN_DEFINITION_METHOD:.*',
        'Domain discretization': r'DOMAIN_DISCRETIZATION:.*',
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
#SBATCH --time=20:00:00
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

def process_nz_watersheds(flowdata_nc, metainfo_excel, attrs_excel, output_dir, template_config, 
                         max_watersheds=None, submit_jobs=False, dry_run=False, filter_good_sites=True):
    """
    Process NZ watersheds and optionally submit CONFLUENCE jobs
    
    Args:
        flowdata_nc: Path to NetCDF flow data
        metainfo_excel: Path to metadata Excel file
        attrs_excel: Path to attributes Excel file
        output_dir: Directory to save config files
        template_config: Path to the template config file
        max_watersheds: Maximum number of watersheds to process (None for all)
        submit_jobs: Whether to submit CONFLUENCE jobs
        dry_run: If True, generate scripts without submitting jobs
        filter_good_sites: If True, only use sites with IsGoodSite=1
        
    Returns:
        DataFrame with information about processed watersheds
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata and attributes
    try:
        meta_df = pd.read_excel(metainfo_excel)
        attrs_df = pd.read_excel(attrs_excel)
        
        print(f"Loaded metadata for {len(meta_df)} stations and attributes for {len(attrs_df)} stations")
        
        # Filter for good sites if requested
        if filter_good_sites:
            meta_df = meta_df[meta_df['IsGoodSite'] == 1]
            attrs_df = attrs_df[attrs_df['IsGoodSite'] == True]
            print(f"Filtered to {len(meta_df)} good sites from metadata and {len(attrs_df)} from attributes")
        
        # Merge metadata and attributes on Station_ID
        stations_df = pd.merge(meta_df, attrs_df, on='Station_ID', how='inner')
        print(f"Merged data has {len(stations_df)} stations")
        
        # Limit the number of watersheds if specified
        if max_watersheds is not None:
            max_watersheds = int(max_watersheds)
            stations_df = stations_df.head(max_watersheds)
            print(f"Limited to {len(stations_df)} watersheds as requested")
    except Exception as e:
        print(f"Error loading station information: {e}")
        import traceback
        print(traceback.format_exc())
        return pd.DataFrame()
    
    # Create a data frame to track processing
    watershed_data = []
    
    # Track submitted jobs
    submitted_jobs = []
    
    # Process each station
    for idx, station in stations_df.iterrows():
        station_id = station['Station_ID']
        station_name = station['Station_name'] if 'Station_name' in station else station['StationName']
        
        watershed_id = f"NZ_{station_id}"
        print(f"\nProcessing station {station_id} - {station_name} ({idx+1}/{len(stations_df)})")
        
        # Prepare coordinates
        latitude = station['latitude']
        longitude = station['longitude']
        
        # Extract pour point coordinates
        pour_point = f"{latitude}/{longitude}"  # Format as lat/lon
        
        # Check if the domain directory already exists
        domain_dir = Path("/anvil/projects/x-ees240082/data/CONFLUENCE_data/niwa") / f"domain_{watershed_id}"
        
        if domain_dir.exists():
            print(f"Skipping {watershed_id} - directory already exists: {domain_dir}")
            continue
        
        # Set up CONFLUENCE directory and create/modify shapefiles
        basin_target_dir, river_target_dir, basin_filename, river_filename = setup_confluence_directory(
            station_id, 
            station_name,
            latitude,
            longitude
        )
        
        if not basin_target_dir or not river_target_dir:
            print(f"Error setting up CONFLUENCE directory for {station_id}")
            continue
        
        # Calculate bounding box from the station coordinates
        bounding_box = calculate_bounding_box(latitude, longitude, buffer_degrees=0.1)
        
        # Extract streamflow data for this station
        obs_dir = domain_dir / "observations" / "streamflow" / "preprocessed"
        obs_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = obs_dir / f"{watershed_id}_streamflow_processed.csv"
        extract_success = extract_station_timeseries(flowdata_nc, station_id, output_file)
        
        if not extract_success:
            print(f"Warning: Failed to extract streamflow data for station {station_id}")
        
        # Generate config file
        config_path = os.path.join(output_dir, f"config_{watershed_id}.yaml")
        generate_config_file(
            template_config,
            config_path,
            watershed_id,
            str(basin_target_dir),
            basin_filename,
            str(river_target_dir),
            river_filename,
            bounding_box,
            pour_point,
            station_id
        )
        
        # Add to tracking data
        watershed_info = {
            'station_id': station_id,
            'station_name': station_name,
            'watershed_id': watershed_id,
            'basin_file': basin_filename,
            'river_file': river_filename,
            'config_file': config_path,
            'pour_point': pour_point,
            'bounding_box': bounding_box,
            'latitude': latitude,
            'longitude': longitude
        }
        
        watershed_data.append(watershed_info)
        
        # Submit job if requested
        if submit_jobs:
            if dry_run:
                job_id = run_confluence(config_path, watershed_id, dry_run=True)
                submitted_jobs.append((watershed_id, "DRY_RUN"))
            else:
                job_id = run_confluence(config_path, watershed_id)
                if job_id:
                    submitted_jobs.append((watershed_id, job_id))
            
            # Add a small delay between job submissions
            time.sleep(2)
    
    # Create summary DataFrame
    watersheds_df = pd.DataFrame(watershed_data)
    
    # Save to CSV for reference
    csv_path = os.path.join(output_dir, "nz_watersheds.csv")
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
    
    return watersheds_df

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process NZ watersheds for CONFLUENCE')
    parser.add_argument('--flowdata', type=str, 
                       default='/anvil/projects/x-ees240082/data/misc-data/NewZealand_Information/Flow_data/Observed_Flow_DN2_3_01Jan1960-31Mar2024_12Aug2024 (1).nc',
                       help='Path to the flow data NetCDF file')
    parser.add_argument('--metainfo', type=str, 
                       default='/anvil/projects/x-ees240082/data/misc-data/NewZealand_Information/Flow_data/Metadata_flow_stations_Flood_forecasting_questionnaire.xlsx',
                       help='Path to the metadata Excel file')
    parser.add_argument('--attributes', type=str, 
                       default='/anvil/projects/x-ees240082/data/misc-data/NewZealand_Information/Flow_data/Station_data_catchment_attributes.xls',
                       help='Path to the attributes Excel file')
    parser.add_argument('--template', type=str, 
                       default='/home/x-deythorsson/code/CONFLUENCE/0_config_files/config_Bow_lumped.yaml',
                       help='Path to the template config file')
    parser.add_argument('--output', type=str, 
                       default='/home/x-deythorsson/code/CONFLUENCE/0_config_files/nz',
                       help='Directory to save config files')
    parser.add_argument('--max', type=int, default=None,
                       help='Maximum number of watersheds to process')
    parser.add_argument('--all-sites', action='store_true',
                       help='Process all sites, not just good sites')
    parser.add_argument('--submit', action='store_true',
                       help='Submit CONFLUENCE jobs')
    parser.add_argument('--dry-run', action='store_true',
                       help='Generate scripts without submitting jobs')
    
    args = parser.parse_args()
    
    # Verify files exist
    for file_path in [args.flowdata, args.metainfo, args.attributes, args.template]:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return
    
    # Process watersheds
    watersheds = process_nz_watersheds(
        args.flowdata,
        args.metainfo,
        args.attributes,
        args.output,
        args.template,
        args.max,
        args.submit,
        args.dry_run,
        not args.all_sites  # Filter good sites unless --all-sites is specified
    )
    
    print(f"Processed {len(watersheds)} watersheds")

if __name__ == "__main__":
    main()
