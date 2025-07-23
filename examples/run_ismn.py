#!/usr/bin/env python3
"""
ISMN Station Processing for CONFLUENCE Large Sample Simulations

This script processes ISMN (International Soil Moisture Network) station data,
filters for North American stations, and sets up CONFLUENCE point simulations.

Usage:
    python run_ismn.py --ismn_path /path/to/ismn/data --template_config config_template.yaml

Author: CONFLUENCE Team
Date: January 2025
"""

import pandas as pd
import numpy as np
import os
import yaml
import subprocess
from pathlib import Path
import time
import sys
import re
import xarray as xr
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

def find_ismn_data_files(ismn_dir, target_vars=None, verbose=False):
    """
    Recursively find all data files in the ISMN directory structure.
    
    Args:
        ismn_dir: Root directory of ISMN data
        target_vars: List of target variable names to filter for
        verbose: Whether to print detailed processing info
        
    Returns:
        dict: Dictionary of {organization: {station: [file_paths]}}
    """
    print(f"Scanning ISMN data structure in {ismn_dir}")
    
    ismn_files = {}
    
    # Default target variables if not specified
    if target_vars is None:
        target_vars = ['sm', 'soil_moisture', 'swv', 'wfv']
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(ismn_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        # Filter for STM files - they contain the data and coordinates
        stm_files = [f for f in files if f.endswith('.stm')]
        
        # Filter for target variables if possible
        if target_vars:
            stm_files = [f for f in stm_files if any(var in f.lower() for var in target_vars)]
        
        if stm_files:
            # Determine organization and station from path
            rel_path = os.path.relpath(root, ismn_dir)
            path_parts = rel_path.split(os.sep)
            
            # Handle different directory structures
            if len(path_parts) >= 2 and path_parts[0] != '.':
                organization = path_parts[0]
                station = path_parts[1]
                
                # Record the files
                if organization not in ismn_files:
                    ismn_files[organization] = {}
                
                if station not in ismn_files[organization]:
                    ismn_files[organization][station] = []
                
                # Add full paths to files
                for file in stm_files:
                    ismn_files[organization][station].append(os.path.join(root, file))
    
    # Print summary
    total_orgs = len(ismn_files)
    total_stations = sum(len(stations) for stations in ismn_files.values())
    total_files = sum(sum(len(files) for files in stations.values()) for stations in ismn_files.values())
    
    print(f"Found {total_files} data files from {total_stations} stations across {total_orgs} organizations")
    
    if verbose:
        # Print detailed breakdown
        print("\nOrganization breakdown:")
        for org, stations in ismn_files.items():
            station_count = len(stations)
            file_count = sum(len(files) for files in stations.values())
            print(f"  {org}: {station_count} stations, {file_count} files")
    
    return ismn_files

def extract_coordinates_from_stm(file_path, verbose=False):
    """
    Extract coordinates and other metadata from STM file header.
    
    Args:
        file_path: Path to STM file
        verbose: Whether to print detailed processing info
        
    Returns:
        dict: Dictionary with extracted metadata
    """
    metadata = {
        'latitude': None,
        'longitude': None,
        'elevation': None,
        'depth_from': None,
        'depth_to': None,
        'sensor': None,
        'station_name': os.path.basename(os.path.dirname(file_path)),
        'organization': os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    }
    
    try:
        # Read the first line of the file which contains the header
        with open(file_path, 'r') as f:
            header_line = f.readline().strip()
        
        # Parse the header line
        # Format is typically:
        # NETWORK SUBNETWORK STATION LAT LON ELEV DEPTH_FROM DEPTH_TO SENSOR
        parts = header_line.split()
        
        if len(parts) >= 5:
            # Extract coordinates
            try:
                # The positions may vary slightly based on format, but usually:
                # Latitude is usually the 4th element
                # Longitude is usually the 5th element
                metadata['latitude'] = float(parts[3])
                metadata['longitude'] = float(parts[4])
                
                if verbose:
                    print(f"Extracted coordinates: {metadata['latitude']}, {metadata['longitude']}")
                
                # Extract elevation if available
                if len(parts) >= 6:
                    try:
                        metadata['elevation'] = float(parts[5])
                    except ValueError:
                        pass
                
                # Extract depth information if available
                if len(parts) >= 8:
                    try:
                        metadata['depth_from'] = float(parts[6])
                        metadata['depth_to'] = float(parts[7])
                    except ValueError:
                        pass
                
                # Extract sensor information if available
                if len(parts) >= 9:
                    metadata['sensor'] = ' '.join(parts[8:])
            except (ValueError, IndexError):
                if verbose:
                    print(f"Warning: Could not parse coordinates from header: {header_line}")
        
        # Get variable from filename
        filename = os.path.basename(file_path)
        for var_name in ['sm', 'soil_moisture', 'swv', 'wfv', 'swd', 'swp']:
            if var_name in filename.lower():
                metadata['variable'] = var_name
                break
        
        return metadata
        
    except Exception as e:
        print(f"Error extracting metadata from {file_path}: {e}")
        if verbose:
            import traceback
            print(traceback.format_exc())
        return metadata

def load_stm_data(file_path, verbose=False):
    """
    Load data from STM file.
    
    Args:
        file_path: Path to STM file
        verbose: Whether to print detailed processing info
        
    Returns:
        pandas.DataFrame: DataFrame with time series data
    """
    try:
        # Read file, skipping the header line
        data_lines = []
        with open(file_path, 'r') as f:
            next(f)  # Skip header
            for line in f:
                data_lines.append(line.strip())
        
        # Parse data lines
        data_records = []
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    # Format is typically: DATE TIME VALUE [FLAGS]
                    date_str = parts[0]
                    time_str = parts[1]
                    value_str = parts[2]
                    
                    # Combine date and time
                    timestamp = f"{date_str} {time_str}"
                    
                    # Convert value to float
                    value = float(value_str)
                    
                    # Extract quality flags if available
                    flags = ' '.join(parts[3:]) if len(parts) > 3 else None
                    
                    data_records.append({
                        'timestamp': timestamp,
                        'value': value,
                        'flags': flags
                    })
                except (ValueError, IndexError):
                    if verbose:
                        print(f"Warning: Could not parse data line: {line}")
                    continue
        
        # Create DataFrame
        if data_records:
            df = pd.DataFrame(data_records)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Drop rows with invalid timestamps
            df = df.dropna(subset=['timestamp'])
            
            # Add source file info
            df['source_file'] = os.path.basename(file_path)
            
            return df
            
        else:
            return pd.DataFrame()
        
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        if verbose:
            import traceback
            print(traceback.format_exc())
        return pd.DataFrame()

def process_station_data(station_files, verbose=False):
    """
    Process all data files for a station.
    
    Args:
        station_files: List of file paths for a station
        verbose: Whether to print detailed processing info
        
    Returns:
        tuple: (DataFrame with time series data, dict with station metadata)
    """
    all_data_dfs = []
    metadata = None
    
    # First, extract metadata from the first file
    if station_files:
        metadata = extract_coordinates_from_stm(station_files[0], verbose)
    
    # Process each data file
    for file_path in station_files:
        # Load data
        df = load_stm_data(file_path, verbose)
        
        if not df.empty:
            # Add depth and variable info to column names
            file_metadata = extract_coordinates_from_stm(file_path, verbose)
            depth_from = file_metadata.get('depth_from', 0)
            depth_to = file_metadata.get('depth_to', 0)
            variable = file_metadata.get('variable', 'soil_moisture')
            
            # Rename value column with depth info
            if depth_from is not None and depth_to is not None:
                new_col_name = f"{variable}_{depth_from}_{depth_to}"
            else:
                new_col_name = variable
            
            df = df.rename(columns={'value': new_col_name})
            all_data_dfs.append(df)
    
    # Combine all data
    if all_data_dfs:
        # Merge data frames on timestamp
        combined_df = all_data_dfs[0]
        
        for df in all_data_dfs[1:]:
            # Merge on timestamp, keeping all columns
            combined_df = pd.merge(combined_df, df, on='timestamp', how='outer', suffixes=('', '_dup'))
            
            # Drop duplicate columns that aren't measurement data
            cols_to_drop = [col for col in combined_df.columns if col.endswith('_dup') and not any(var in col for var in ['sm', 'soil_moisture', 'swv', 'wfv'])]
            combined_df = combined_df.drop(columns=cols_to_drop)
        
        # Sort by timestamp
        combined_df.sort_values('timestamp', inplace=True)
        
        return combined_df, metadata
    else:
        return pd.DataFrame(), metadata

def filter_north_america_stations(stations_data, verbose=False):
    """
    Filter stations to only include those in North America.
    
    Args:
        stations_data: Dictionary of station data
        verbose: Whether to print detailed processing info
        
    Returns:
        dict: Filtered stations_data containing only North American stations
    """
    # North America bounding box (approximate)
    # Latitude: 7° to 84° N
    # Longitude: -180° to -30° W
    north_america_bounds = {
        'lat_min': 7.0,
        'lat_max': 84.0,
        'lon_min': -180.0,
        'lon_max': -30.0
    }
    
    filtered_stations = {}
    original_count = 0
    filtered_count = 0
    
    for org_name, org_stations in stations_data.items():
        filtered_org = {}
        
        for station_name, station_info in org_stations.items():
            original_count += 1
            metadata = station_info['metadata']
            
            # Check if station has coordinates
            if metadata['latitude'] is not None and metadata['longitude'] is not None:
                lat = metadata['latitude']
                lon = metadata['longitude']
                
                # Check if coordinates are within North America bounds
                if (north_america_bounds['lat_min'] <= lat <= north_america_bounds['lat_max'] and
                    north_america_bounds['lon_min'] <= lon <= north_america_bounds['lon_max']):
                    
                    filtered_org[station_name] = station_info
                    filtered_count += 1
                    
                    if verbose:
                        print(f"Included station: {org_name}/{station_name} at {lat:.3f}, {lon:.3f}")
                elif verbose:
                    print(f"Excluded station: {org_name}/{station_name} at {lat:.3f}, {lon:.3f} (outside North America)")
            elif verbose:
                print(f"Excluded station: {org_name}/{station_name} (no coordinates)")
        
        if filtered_org:
            filtered_stations[org_name] = filtered_org
    
    print(f"Filtered {original_count} stations to {filtered_count} North American stations")
    
    return filtered_stations

def process_ismn_data(ismn_path, output_csv_path, start_year=None, end_year=None, use_existing_csv=True, verbose=False):
    """
    Process ISMN dataset and extract station metadata to a CSV file
    
    Args:
        ismn_path: Path to the ISMN data directory
        output_csv_path: Path to save the processed station data as CSV
        start_year: Optional start year for filtering data
        end_year: Optional end year for filtering data
        use_existing_csv: Whether to use existing CSV file if it exists (default: True)
        verbose: Whether to print detailed processing info
    
    Returns:
        DataFrame containing processed station data
    """
    # Check if CSV already exists and we're allowed to use it
    if use_existing_csv and os.path.exists(output_csv_path):
        print(f"Using existing processed station data from {output_csv_path}")
        return pd.read_csv(output_csv_path)
    
    print(f"Reading ISMN data from {ismn_path}...")
    
    # Find all ISMN data files
    ismn_files = find_ismn_data_files(ismn_path, verbose=verbose)
    
    # Process stations to extract metadata and data availability
    stations_data = {}
    
    for organization, stations in ismn_files.items():
        print(f"Processing organization: {organization}")
        
        if organization not in stations_data:
            stations_data[organization] = {}
        
        for station, file_paths in stations.items():
            if verbose:
                print(f"  Processing station: {station} ({len(file_paths)} files)")
            
            # Process station data
            data, metadata = process_station_data(file_paths, verbose=verbose)
            
            # Store the results
            stations_data[organization][station] = {
                'data': data,
                'metadata': metadata
            }
    
    # Filter for North America
    print("Filtering for North American stations...")
    stations_data = filter_north_america_stations(stations_data, verbose=verbose)
    
    # Convert to DataFrame format
    station_records = []
    
    for org_name, org_stations in stations_data.items():
        for station_name, station_info in org_stations.items():
            metadata = station_info['metadata']
            data = station_info['data']
            
            # Skip stations without coordinates
            if metadata['latitude'] is None or metadata['longitude'] is None:
                continue
            
            # Calculate data availability
            if not data.empty and 'timestamp' in data.columns:
                # Filter by year range if specified
                if start_year is not None and end_year is not None:
                    data_filtered = data[
                        (data['timestamp'].dt.year >= start_year) & 
                        (data['timestamp'].dt.year <= end_year)
                    ]
                else:
                    data_filtered = data
                
                # Count valid measurements
                numeric_cols = data_filtered.select_dtypes(include=[np.number]).columns
                valid_count = 0
                completeness = 0
                
                if len(numeric_cols) > 0:
                    # Use the first numeric column for completeness calculation
                    main_col = numeric_cols[0]
                    valid_count = data_filtered[main_col].notna().sum()
                    total_timesteps = len(data_filtered)
                    completeness = (valid_count / total_timesteps * 100) if total_timesteps > 0 else 0
                
                start_date = data_filtered['timestamp'].min() if len(data_filtered) > 0 else None
                end_date = data_filtered['timestamp'].max() if len(data_filtered) > 0 else None
            else:
                valid_count = 0
                completeness = 0
                start_date = None
                end_date = None
            
            # Create station record
            station_record = {
                'station_id': f"{org_name}_{station_name}",
                'organization': org_name,
                'station_name': station_name,
                'lat': metadata['latitude'],
                'lon': metadata['longitude'],
                'elevation': metadata.get('elevation'),
                'depth_from': metadata.get('depth_from'),
                'depth_to': metadata.get('depth_to'),
                'sensor': metadata.get('sensor'),
                'valid_count': valid_count,
                'completeness': completeness,
                'start_date': start_date,
                'end_date': end_date,
                'variables': ', '.join(data.select_dtypes(include=[np.number]).columns.tolist()) if not data.empty else ''
            }
            
            # Calculate bounding box coordinates for each station
            # For point simulations, we create a small bounding box around each station
            # Default buffer is 0.1 degrees in each direction
            buffer = 0.1
            station_record['BOUNDING_BOX_COORDS'] = (
                f"{metadata['latitude'] + buffer}/{metadata['longitude'] - buffer}/"
                f"{metadata['latitude'] - buffer}/{metadata['longitude'] + buffer}"
            )
            
            # Format pour point coordinates as lat/lon
            station_record['POUR_POINT_COORDS'] = f"{metadata['latitude']}/{metadata['longitude']}"
            
            # Create Watershed_Name column based on station_id (cleaned up for file naming)
            # Replace any characters that might cause issues in file paths
            station_record['Watershed_Name'] = re.sub(r'[^a-zA-Z0-9_]', '_', station_record['station_id'])
            
            station_records.append(station_record)
    
    # Create DataFrame
    stations_df = pd.DataFrame(station_records)
    
    # Save to CSV
    print(f"Saving processed station data to {output_csv_path}...")
    stations_df.to_csv(output_csv_path, index=False)
    
    return stations_df

def extract_soil_moisture_data(ismn_path, station_id, organization, station_name, output_dir, start_year=None, end_year=None, verbose=False):
    """
    Extract soil moisture data for a specific station and save to CSV files
    
    Args:
        ismn_path: Path to the ISMN data directory
        station_id: Full station ID (org_station format)
        organization: Organization name
        station_name: Station name
        output_dir: Directory to save the extracted data
        start_year: Optional start year for filtering data
        end_year: Optional end year for filtering data
        verbose: Whether to print detailed processing info
        
    Returns:
        Path to saved soil moisture file
    """
    if verbose:
        print(f"Extracting soil moisture data for station {station_id}...")
    
    # Find the station files
    station_path = os.path.join(ismn_path, organization, station_name)
    
    if not os.path.exists(station_path):
        print(f"Warning: Station path not found: {station_path}")
        return None
    
    # Find STM files
    stm_files = []
    for root, dirs, files in os.walk(station_path):
        for file in files:
            if file.endswith('.stm') and any(var in file.lower() for var in ['sm', 'soil_moisture', 'swv', 'wfv']):
                stm_files.append(os.path.join(root, file))
    
    if not stm_files:
        print(f"Warning: No soil moisture files found for station {station_id}")
        return None
    
    # Process station data
    combined_df, metadata = process_station_data(stm_files, verbose=verbose)
    
    if combined_df.empty:
        print(f"Warning: No data extracted for station {station_id}")
        return None
    
    # Filter by year range if specified
    if start_year is not None and end_year is not None:
        time_mask = (combined_df['timestamp'].dt.year >= start_year) & (combined_df['timestamp'].dt.year <= end_year)
        combined_df = combined_df[time_mask]
    
    # Create output directory
    sm_dir = Path(output_dir) / 'observations' / 'soil_moisture' / 'raw_data'
    sm_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV file
    sm_file = sm_dir / f"{station_id}_soil_moisture.csv"
    combined_df.to_csv(sm_file, index=False)
    
    if verbose:
        print(f"Saved soil moisture data to {sm_file} ({len(combined_df)} records)")
    
    return str(sm_file)

def generate_config_file(template_path, output_path, domain_name, pour_point, bounding_box):
    """
    Generate a new config file based on the template with updated parameters
    
    Args:
        template_path: Path to the template config file
        output_path: Path to save the new config file
        domain_name: Name of the domain to set
        pour_point: Pour point coordinates to set
        bounding_box: Bounding box coordinates to set
    """
    # Read the template config file
    with open(template_path, 'r') as f:
        config_content = f.read()
    
    # Update the domain name using regex
    config_content = re.sub(r'DOMAIN_NAME:.*', f'DOMAIN_NAME: "{domain_name}"', config_content)
    
    # Update the pour point coordinates using regex
    config_content = re.sub(r'POUR_POINT_COORDS:.*', f'POUR_POINT_COORDS: {pour_point}', config_content)
    
    # Update the bounding box coordinates using regex
    config_content = re.sub(r'BOUNDING_BOX_COORDS:.*', f'BOUNDING_BOX_COORDS: {bounding_box}', config_content)
    
    # Add or modify soil moisture specific settings
    # Enable soil moisture observations
    if 'DOWNLOAD_ISMN' not in config_content:
        # Add ISMN settings if not present
        ismn_settings = """
# ISMN soil moisture data settings
DOWNLOAD_ISMN: 'true'                                          # Download ISMN data, options: 'true' or 'false'
ISMN_PATH: '/path/to/ismn/data'                                # Path to directory containing ISMN data files
"""
        # Insert after other observation settings
        if 'FLUXNET_PATH:' in config_content:
            config_content = config_content.replace(
                'FLUXNET_PATH: \'/path/to/fluxnet\'',
                'FLUXNET_PATH: \'/path/to/fluxnet\'\n' + ismn_settings
            )
        else:
            # Add at the end of evaluation settings
            config_content = config_content.replace(
                '### ============================================= 6. Optimisation settings:',
                ismn_settings + '\n### ============================================= 6. Optimisation settings:'
            )
    
    # Save the new config file
    with open(output_path, 'w') as f:
        f.write(config_content)
    
    # Verify the changes were made
    print(f"Config file created at {output_path}")
    
    return output_path

def run_confluence(config_path, watershed_name):
    """
    Run CONFLUENCE with the specified config file
    
    Args:
        config_path: Path to the config file
        watershed_name: Name of the watershed for job naming
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

# Load necessary modules (adjust as needed for your HPC environment)
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
    
    # Submit the job
    result = subprocess.run(['sbatch', batch_script], capture_output=True, text=True)
    
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted job for {watershed_name}, job ID: {job_id}")
        return job_id
    else:
        print(f"Failed to submit job for {watershed_name}: {result.stderr}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process ISMN data and set up CONFLUENCE point simulations for North America')
    parser.add_argument('--ismn_path', type=str, required=True,
                        help='Path to the ISMN data directory')
    parser.add_argument('--template_config', type=str, required=True,
                        help='Path to the template config file for CONFLUENCE')
    parser.add_argument('--output_dir', type=str, default='ismn_output',
                        help='Directory to store processed data and outputs')
    parser.add_argument('--config_dir', type=str, default='ismn_configs',
                        help='Directory to store generated config files')
    parser.add_argument('--min_completeness', type=float, default=50.0,
                        help='Minimum data completeness percentage to include a station')
    parser.add_argument('--max_stations', type=int, default=None,
                        help='Maximum number of stations to process')
    parser.add_argument('--start_year', type=int, default=None,
                        help='Start year for filtering data')
    parser.add_argument('--end_year', type=int, default=None,
                        help='End year for filtering data')
    parser.add_argument('--no_submit', action='store_true',
                        help='Generate configs but don\'t submit jobs')
    parser.add_argument('--base_path', type=str, default='/work/comphyd_lab/data/CONFLUENCE_data/ismn',
                        help='Base path for CONFLUENCE data directory')
    parser.add_argument('--force_reprocess', action='store_true',
                        help='Force reprocessing of station data even if CSV exists')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.config_dir, exist_ok=True)
    
    # Process ISMN data
    print(f"Processing ISMN data from {args.ismn_path}...")
    stations_csv = os.path.join(args.output_dir, 'ismn_stations_north_america.csv')
    
    stations_df = process_ismn_data(
        args.ismn_path, 
        stations_csv,
        start_year=args.start_year,
        end_year=args.end_year,
        use_existing_csv=not args.force_reprocess,
        verbose=args.verbose
    )
    
    # Filter stations based on completeness
    complete_stations = stations_df[
        stations_df['completeness'] >= args.min_completeness
    ]
    
    print(f"Found {len(complete_stations)} stations with at least {args.min_completeness}% data completeness")
    
    # Limit to max_stations if specified
    if args.max_stations is not None and len(complete_stations) > args.max_stations:
        print(f"Limiting to {args.max_stations} stations")
        # Prioritize stations with higher completeness
        complete_stations = complete_stations.sort_values(
            by='completeness', 
            ascending=False
        ).head(args.max_stations)
    
    # Process each station for CONFLUENCE runs
    submitted_jobs = []
    skipped_jobs = []
    
    # Ask if user wants to submit CONFLUENCE jobs (unless --no_submit is specified)
    submit_jobs = 'n' if args.no_submit else input("\nDo you want to submit CONFLUENCE jobs for these stations? (y/n): ").lower().strip()
    
    for _, station in complete_stations.iterrows():
        # Get station parameters
        station_id = station['station_id']
        organization = station['organization']
        station_name = station['station_name']
        watershed_name = station['Watershed_Name']
        pour_point = station['POUR_POINT_COORDS']
        bounding_box = station['BOUNDING_BOX_COORDS']
        
        # Create a unique domain name
        domain_name = f"{watershed_name}"
        
        # Check if the simulations directory already exists
        simulation_dir = Path(f"{args.base_path}/domain_{domain_name}")
        simulation_archive = Path(f"{args.base_path}/domain_{domain_name}.tar.gz")
        
        if simulation_dir.exists() or simulation_archive.exists():
            if args.verbose:
                print(f"Skipping {domain_name} - simulation result already exists")
            skipped_jobs.append(domain_name)
            continue
        
        # Extract soil moisture data for this station
        station_dir = f"{args.base_path}/domain_{domain_name}"
        sm_file = extract_soil_moisture_data(
            args.ismn_path,
            station_id,
            organization,
            station_name,
            station_dir,
            start_year=args.start_year,
            end_year=args.end_year,
            verbose=args.verbose
        )
        
        if sm_file and args.verbose:
            print(f"Extracted soil moisture data for {station_id}: {sm_file}")
        
        # Generate the config file path
        config_path = os.path.join(args.config_dir, f"config_{domain_name}.yaml")
        
        # Generate the config file
        if args.verbose:
            print(f"Generating config file for {domain_name}...")
        generate_config_file(args.template_config, config_path, domain_name, pour_point, bounding_box)
        
        # Run CONFLUENCE with the generated config if requested
        if submit_jobs == 'y':
            print(f"Submitting CONFLUENCE job for {domain_name}...")
            job_id = run_confluence(config_path, domain_name)
            
            if job_id:
                submitted_jobs.append((domain_name, job_id))
            
            # Add a small delay between job submissions to avoid overwhelming the scheduler
            time.sleep(2)
    
    if submit_jobs == 'y':
        # Print summary of submitted jobs
        print("\nSubmitted jobs summary:")
        for domain_name, job_id in submitted_jobs:
            print(f"Domain: {domain_name}, Job ID: {job_id}")
        
        print(f"\nTotal jobs submitted: {len(submitted_jobs)}")
        print(f"Total jobs skipped: {len(skipped_jobs)}")
        
        if skipped_jobs:
            print("\nSkipped domains (simulations already exist):")
            for domain_name in skipped_jobs:
                print(f"- {domain_name}")
    else:
        print("\nNo CONFLUENCE jobs submitted per request.")
        print(f"Soil moisture data has been extracted to the appropriate directories.")
        print(f"Config files have been generated in {args.config_dir}")
        print(f"\nTotal configs generated: {len(complete_stations)}")
        print(f"Total jobs skipped: {len(skipped_jobs)}")

if __name__ == "__main__":
    main()