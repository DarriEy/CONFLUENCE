#!/usr/bin/env python3
"""
Process GGMN (Global Groundwater Monitoring Network) data and set up CONFLUENCE point simulations.

This script processes GGMN groundwater station data, filters stations based on data availability,
and generates CONFLUENCE configuration files for point simulations at each station location.

Usage:
    python run_ggmn.py --ggmn_stations <stations_csv> --ggmn_data_dir <data_directory> 
                       --template_config <template_config> [options]

Author: Claude
Date: July 22, 2025
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
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def process_ggmn_stations(stations_csv, ggmn_data_dir, output_csv_path, 
                         start_year=None, end_year=None, use_existing_csv=True):
    """
    Process GGMN station metadata and assess data availability
    
    Args:
        stations_csv: Path to GGMN stations metadata CSV file
        ggmn_data_dir: Path to directory containing GGMN monitoring data
        output_csv_path: Path to save the processed station data as CSV
        start_year: Optional start year for filtering data
        end_year: Optional end year for filtering data
        use_existing_csv: Whether to use existing CSV file if it exists (default: True)
    
    Returns:
        DataFrame containing processed station data with data availability metrics
    """
    # Check if CSV already exists and we're allowed to use it
    if use_existing_csv and os.path.exists(output_csv_path):
        print(f"Using existing processed station data from {output_csv_path}")
        return pd.read_csv(output_csv_path)
    
    print(f"Reading GGMN station metadata from {stations_csv}...")
    
    # Load station metadata
    if not os.path.exists(stations_csv):
        raise FileNotFoundError(f"GGMN stations file not found: {stations_csv}")
    
    stations_df = pd.read_csv(stations_csv)
    print(f"Loaded {len(stations_df)} GGMN stations")
    
    # Standardize column names (handle different possible column names)
    column_mapping = {
        'station_id': ['station_id', 'id', 'well_id', 'site_id'],
        'station_name': ['station_name', 'name', 'well_name', 'site_name'],
        'latitude': ['latitude', 'lat', 'y', 'Latitude'],
        'longitude': ['longitude', 'lon', 'lng', 'x', 'Longitude'],
        'elevation': ['elevation', 'elev', 'altitude', 'z'],
        'country': ['country', 'Country']
    }
    
    # Map columns to standard names
    for standard_name, possible_names in column_mapping.items():
        for possible_name in possible_names:
            if possible_name in stations_df.columns:
                if standard_name not in stations_df.columns:
                    stations_df[standard_name] = stations_df[possible_name]
                break
    
    # Check for required columns
    required_cols = ['latitude', 'longitude']
    missing_cols = [col for col in required_cols if col not in stations_df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing from stations file: {missing_cols}")
    
    # Filter for North American stations (rough bounding box)
    print("Filtering for North American stations...")
    na_mask = (
        (stations_df['latitude'] >= 10) & (stations_df['latitude'] <= 85) &
        (stations_df['longitude'] >= -180) & (stations_df['longitude'] <= -50)
    )
    
    stations_df = stations_df[na_mask].copy()
    print(f"Found {len(stations_df)} stations in North America")
    
    # Add country information if not present
    if 'country' not in stations_df.columns:
        stations_df['country'] = 'Unknown'
        # Simple country assignment based on coordinates
        us_mask = (
            (stations_df['latitude'] >= 25) & (stations_df['latitude'] <= 49) &
            (stations_df['longitude'] >= -125) & (stations_df['longitude'] <= -66)
        )
        canada_mask = (
            (stations_df['latitude'] >= 42) & (stations_df['latitude'] <= 85) &
            (stations_df['longitude'] >= -141) & (stations_df['longitude'] <= -52)
        )
        mexico_mask = (
            (stations_df['latitude'] >= 14) & (stations_df['latitude'] <= 32) &
            (stations_df['longitude'] >= -118) & (stations_df['longitude'] <= -86)
        )
        
        stations_df.loc[us_mask, 'country'] = 'USA'
        stations_df.loc[canada_mask, 'country'] = 'Canada'
        stations_df.loc[mexico_mask, 'country'] = 'Mexico'
    
    # Assess data availability by scanning monitoring files
    print("Assessing data availability...")
    stations_df = assess_data_availability(stations_df, ggmn_data_dir, start_year, end_year)
    
    # Calculate bounding box coordinates for each station
    # For point simulations, we create a small bounding box around each station
    buffer = 0.1  # degrees
    stations_df['BOUNDING_BOX_COORDS'] = (
        (stations_df['latitude'] + buffer).astype(str) + '/' +
        (stations_df['longitude'] - buffer).astype(str) + '/' +
        (stations_df['latitude'] - buffer).astype(str) + '/' +
        (stations_df['longitude'] + buffer).astype(str)
    )
    
    # Format pour point coordinates as lat/lon
    stations_df['POUR_POINT_COORDS'] = (
        stations_df['latitude'].astype(str) + '/' + 
        stations_df['longitude'].astype(str)
    )
    
    # Create watershed name column based on station_id (cleaned for file naming)
    if 'station_id' not in stations_df.columns and 'station_name' in stations_df.columns:
        stations_df['station_id'] = stations_df['station_name']
    
    stations_df['Watershed_Name'] = (
        stations_df['station_id'].astype(str)
        .str.replace('[^a-zA-Z0-9_]', '_', regex=True)
        .str.replace('__+', '_', regex=True)  # Replace multiple underscores with single
        .str.strip('_')  # Remove leading/trailing underscores
    )
    
    # Ensure unique watershed names
    duplicated_names = stations_df['Watershed_Name'].duplicated()
    if duplicated_names.any():
        print(f"Found {duplicated_names.sum()} duplicate watershed names, making unique...")
        for i, is_dup in enumerate(duplicated_names):
            if is_dup:
                base_name = stations_df.iloc[i]['Watershed_Name']
                counter = 1
                new_name = f"{base_name}_{counter}"
                while new_name in stations_df['Watershed_Name'].values:
                    counter += 1
                    new_name = f"{base_name}_{counter}"
                stations_df.iloc[i, stations_df.columns.get_loc('Watershed_Name')] = new_name
    
    # Save processed data
    print(f"Saving processed station data to {output_csv_path}...")
    stations_df.to_csv(output_csv_path, index=False)
    
    return stations_df

def assess_data_availability(stations_df, ggmn_data_dir, start_year=None, end_year=None):
    """
    Assess data availability for each station by scanning monitoring files
    
    Args:
        stations_df: DataFrame with station metadata
        ggmn_data_dir: Directory containing GGMN monitoring data
        start_year: Optional start year for filtering
        end_year: Optional end year for filtering
    
    Returns:
        DataFrame with added data availability columns
    """
    print(f"Scanning monitoring data directory: {ggmn_data_dir}")
    
    # Initialize data availability columns
    stations_df['data_files_found'] = 0
    stations_df['record_count'] = 0
    stations_df['start_date'] = pd.NaT
    stations_df['end_date'] = pd.NaT
    stations_df['data_completeness'] = 0.0
    stations_df['data_years'] = 0
    
    # Build a mapping of station IDs to files
    station_files = find_station_data_files(ggmn_data_dir, stations_df)
    
    for idx, station in stations_df.iterrows():
        station_id = str(station.get('station_id', ''))
        
        if station_id in station_files:
            files = station_files[station_id]
            stations_df.at[idx, 'data_files_found'] = len(files)
            
            # Quick assessment of data availability
            total_records = 0
            start_dates = []
            end_dates = []
            
            for file_path in files[:3]:  # Limit to first 3 files for performance
                try:
                    records, start_date, end_date = quick_file_assessment(file_path)
                    total_records += records
                    if start_date:
                        start_dates.append(start_date)
                    if end_date:
                        end_dates.append(end_date)
                except Exception as e:
                    print(f"Warning: Could not assess {file_path}: {e}")
                    continue
            
            if total_records > 0:
                stations_df.at[idx, 'record_count'] = total_records
                
                if start_dates:
                    stations_df.at[idx, 'start_date'] = min(start_dates)
                if end_dates:
                    stations_df.at[idx, 'end_date'] = max(end_dates)
                
                # Calculate data years and completeness
                if start_dates and end_dates:
                    data_span_years = (max(end_dates) - min(start_dates)).days / 365.25
                    stations_df.at[idx, 'data_years'] = data_span_years
                    
                    # Rough completeness estimate (records per year)
                    if data_span_years > 0:
                        expected_records = data_span_years * 12  # Assume monthly data
                        completeness = min(100, (total_records / expected_records) * 100)
                        stations_df.at[idx, 'data_completeness'] = completeness
    
    return stations_df

def find_station_data_files(ggmn_data_dir, stations_df):
    """
    Find monitoring data files for each station
    
    Args:
        ggmn_data_dir: Directory containing monitoring data
        stations_df: DataFrame with station information
    
    Returns:
        Dictionary mapping station IDs to lists of file paths
    """
    station_files = {}
    
    # Get list of station IDs to search for
    station_ids = []
    for col in ['station_id', 'station_name']:
        if col in stations_df.columns:
            ids = stations_df[col].dropna().astype(str).unique().tolist()
            station_ids.extend(ids)
    
    print(f"Scanning for data files for {len(station_ids)} station identifiers...")
    
    # Scan monitoring directory
    file_count = 0
    for root, dirs, files in os.walk(ggmn_data_dir):
        for filename in files:
            if filename.lower().endswith(('.csv', '.txt', '.ods', '.xlsx')):
                file_count += 1
                file_path = os.path.join(root, filename)
                file_stem = os.path.splitext(filename)[0].lower()
                
                # Check if filename contains any station ID
                for station_id in station_ids:
                    clean_station_id = str(station_id).lower().replace(' ', '').replace('-', '').replace('_', '')
                    clean_filename = file_stem.replace('-', '').replace('_', '').replace(' ', '')
                    
                    if clean_station_id in clean_filename:
                        if station_id not in station_files:
                            station_files[station_id] = []
                        station_files[station_id].append(file_path)
                        break
    
    print(f"Scanned {file_count} files, found data for {len(station_files)} stations")
    return station_files

def quick_file_assessment(file_path):
    """
    Quickly assess a data file to estimate record count and date range
    
    Args:
        file_path: Path to data file
    
    Returns:
        Tuple of (record_count, start_date, end_date)
    """
    try:
        if file_path.lower().endswith('.csv'):
            # For CSV files, quickly read and assess
            df = pd.read_csv(file_path, nrows=1000)  # Read first 1000 rows for assessment
            
            # Look for date columns
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            
            if date_cols:
                date_col = date_cols[0]
                try:
                    dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
                    if len(dates) > 0:
                        return len(df), dates.min(), dates.max()
                except:
                    pass
            
            return len(df), None, None
        
        elif file_path.lower().endswith('.txt'):
            # For text files, count lines
            with open(file_path, 'r') as f:
                line_count = sum(1 for line in f)
            return max(0, line_count - 1), None, None  # Subtract header
        
        else:
            # For other files, return minimal info
            return 1, None, None
    
    except Exception as e:
        return 0, None, None

def extract_groundwater_data(ggmn_data_dir, station_id, output_dir, start_year=None, end_year=None):
    """
    Extract groundwater data for a specific station and save to CSV
    
    Args:
        ggmn_data_dir: Directory containing GGMN monitoring data
        station_id: Station ID to extract data for
        output_dir: Directory to save extracted data
        start_year: Optional start year for filtering
        end_year: Optional end year for filtering
    
    Returns:
        Path to saved groundwater data file
    """
    print(f"Extracting groundwater data for station {station_id}...")
    
    # Create output directory
    gw_dir = Path(output_dir) / 'observations' / 'groundwater' / 'raw_data'
    gw_dir.mkdir(parents=True, exist_ok=True)
    
    # Find data files for this station
    station_files = find_station_data_files(ggmn_data_dir, 
                                          pd.DataFrame({'station_id': [station_id]}))
    
    if station_id not in station_files:
        print(f"No data files found for station {station_id}")
        return None
    
    files = station_files[station_id]
    print(f"Found {len(files)} data files for station {station_id}")
    
    all_data = []
    
    for file_path in files:
        try:
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
                
                # Look for date and value columns
                date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                value_cols = [col for col in df.columns if any(term in col.lower() 
                            for term in ['level', 'depth', 'value', 'measurement'])]
                
                if date_cols and value_cols:
                    date_col = date_cols[0]
                    value_col = value_cols[0]
                    
                    # Create standardized DataFrame
                    gw_df = df[[date_col, value_col]].copy()
                    gw_df.columns = ['date', 'groundwater_level']
                    gw_df['date'] = pd.to_datetime(gw_df['date'], errors='coerce')
                    gw_df = gw_df.dropna()
                    
                    # Apply year filter if specified
                    if start_year is not None and end_year is not None:
                        year_mask = (gw_df['date'].dt.year >= start_year) & (gw_df['date'].dt.year <= end_year)
                        gw_df = gw_df[year_mask]
                    
                    gw_df['source_file'] = os.path.basename(file_path)
                    all_data.append(gw_df)
        
        except Exception as e:
            print(f"Warning: Could not process {file_path}: {e}")
            continue
    
    if all_data:
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['date', 'groundwater_level'])
        combined_df = combined_df.sort_values('date')
        
        # Save to file
        safe_station_id = str(station_id).replace(' ', '_').replace('/', '_').replace('\\', '_')
        gw_file = gw_dir / f"{safe_station_id}_groundwater.csv"
        combined_df.to_csv(gw_file, index=False)
        
        print(f"Saved {len(combined_df)} groundwater records for station {station_id}")
        return str(gw_file)
    
    return None

def generate_config_file(template_path, output_path, domain_name, pour_point, bounding_box, station_id):
    """
    Generate a CONFLUENCE config file for a groundwater station
    
    Args:
        template_path: Path to template config file
        output_path: Path to save new config file
        domain_name: Domain name for the simulation
        pour_point: Pour point coordinates
        bounding_box: Bounding box coordinates
        station_id: Station ID for observations
    """
    # Read template config
    with open(template_path, 'r') as f:
        config_content = f.read()
    
    # Update domain-specific parameters
    config_content = re.sub(r'DOMAIN_NAME:.*', f'DOMAIN_NAME: "{domain_name}"', config_content)
    config_content = re.sub(r'POUR_POINT_COORDS:.*', f'POUR_POINT_COORDS: {pour_point}', config_content)
    config_content = re.sub(r'BOUNDING_BOX_COORDS:.*', f'BOUNDING_BOX_COORDS: {bounding_box}', config_content)
    
    # Update observation settings for groundwater
    config_content = re.sub(r'DOWNLOAD_USGS_GW:.*', 'DOWNLOAD_USGS_GW: false', config_content)
    config_content = re.sub(r'USGS_STATION:.*', f'USGS_STATION: "{station_id}"', config_content)
    
    # Enable groundwater evaluation
    if 'ANALYSES:' in config_content:
        config_content = re.sub(r'ANALYSES:.*', 'ANALYSES: [benchmarking, groundwater]', config_content)
    
    # Save new config
    with open(output_path, 'w') as f:
        f.write(config_content)
    
    print(f"Generated config file: {output_path}")
    return output_path

def run_confluence(config_path, watershed_name, job_time="24:00:00", memory="8G"):
    """
    Submit a CONFLUENCE job with the specified config file
    
    Args:
        config_path: Path to config file
        watershed_name: Name for job identification
        job_time: SLURM time limit
        memory: Memory requirement
    
    Returns:
        Job ID if successful, None otherwise
    """
    # Create batch script
    batch_script = f"run_ggmn_{watershed_name}.sh"
    
    batch_content = f"""#!/bin/bash
#SBATCH --job-name=ggmn_{watershed_name}
#SBATCH --output=CONFLUENCE_ggmn_{watershed_name}_%j.log
#SBATCH --error=CONFLUENCE_ggmn_{watershed_name}_%j.err
#SBATCH --time={job_time}
#SBATCH --ntasks=1
#SBATCH --mem={memory}

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

# Run CONFLUENCE
python CONFLUENCE.py --config {config_path}

echo "CONFLUENCE job for {watershed_name} complete"
"""
    
    with open(batch_script, 'w') as f:
        f.write(batch_content)
    
    os.chmod(batch_script, 0o755)
    
    # Submit job
    result = subprocess.run(['sbatch', batch_script], capture_output=True, text=True)
    
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted job for {watershed_name}, job ID: {job_id}")
        return job_id
    else:
        print(f"Failed to submit job for {watershed_name}: {result.stderr}")
        return None

def main():
    """Main function to process GGMN data and set up CONFLUENCE simulations"""
    parser = argparse.ArgumentParser(description='Process GGMN data and set up CONFLUENCE point simulations')
    parser.add_argument('--ggmn_stations', type=str, required=True,
                        help='Path to GGMN stations metadata CSV file')
    parser.add_argument('--ggmn_data_dir', type=str, required=True,
                        help='Path to directory containing GGMN monitoring data')
    parser.add_argument('--template_config', type=str, required=True,
                        help='Path to template CONFLUENCE config file')
    parser.add_argument('--output_dir', type=str, default='ggmn_output',
                        help='Directory to store processed data and outputs')
    parser.add_argument('--config_dir', type=str, default='ggmn_configs',
                        help='Directory to store generated config files')
    parser.add_argument('--min_completeness', type=float, default=30.0,
                        help='Minimum data completeness percentage')
    parser.add_argument('--min_records', type=int, default=50,
                        help='Minimum number of records required')
    parser.add_argument('--max_stations', type=int, default=None,
                        help='Maximum number of stations to process')
    parser.add_argument('--start_year', type=int, default=None,
                        help='Start year for filtering data')
    parser.add_argument('--end_year', type=int, default=None,
                        help='End year for filtering data')
    parser.add_argument('--no_submit', action='store_true',
                        help='Generate configs but don\'t submit jobs')
    parser.add_argument('--base_path', type=str, default='/work/comphyd_lab/data/CONFLUENCE_data/ggmn',
                        help='Base path for CONFLUENCE data directory')
    parser.add_argument('--force_reprocess', action='store_true',
                        help='Force reprocessing even if CSV exists')
    parser.add_argument('--job_time', type=str, default='24:00:00',
                        help='SLURM time limit for jobs')
    parser.add_argument('--memory', type=str, default='8G',
                        help='Memory requirement for jobs')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.config_dir, exist_ok=True)
    
    # Process GGMN station data
    print(f"Processing GGMN stations from {args.ggmn_stations}...")
    stations_csv = os.path.join(args.output_dir, 'ggmn_stations_processed.csv')
    
    stations_df = process_ggmn_stations(
        args.ggmn_stations,
        args.ggmn_data_dir,
        stations_csv,
        start_year=args.start_year,
        end_year=args.end_year,
        use_existing_csv=not args.force_reprocess
    )
    
    # Filter stations based on data quality
    print(f"Filtering stations with data completeness >= {args.min_completeness}% and >= {args.min_records} records...")
    
    quality_stations = stations_df[
        (stations_df['data_completeness'] >= args.min_completeness) &
        (stations_df['record_count'] >= args.min_records) &
        (stations_df['data_files_found'] > 0)
    ].copy()
    
    print(f"Found {len(quality_stations)} stations meeting quality criteria")
    
    if len(quality_stations) == 0:
        print("No stations meet the quality criteria. Consider lowering thresholds.")
        return
    
    # Limit stations if specified
    if args.max_stations is not None and len(quality_stations) > args.max_stations:
        print(f"Limiting to {args.max_stations} stations with highest data completeness")
        quality_stations = quality_stations.sort_values(
            ['data_completeness', 'record_count'], 
            ascending=False
        ).head(args.max_stations)
    
    # Print summary by country
    if 'country' in quality_stations.columns:
        country_summary = quality_stations['country'].value_counts()
        print("\nStations by country:")
        for country, count in country_summary.items():
            print(f"  {country}: {count}")
    
    # Process each station
    submitted_jobs = []
    skipped_jobs = []
    
    # Ask about job submission
    submit_jobs = 'n' if args.no_submit else input(f"\nDo you want to submit CONFLUENCE jobs for {len(quality_stations)} stations? (y/n): ").lower().strip()
    
    for idx, station in quality_stations.iterrows():
        station_id = station.get('station_id', f'station_{idx}')
        watershed_name = station['Watershed_Name']
        pour_point = station['POUR_POINT_COORDS']
        bounding_box = station['BOUNDING_BOX_COORDS']
        
        domain_name = f"{watershed_name}"
        
        # Check if simulation already exists
        simulation_check = Path(f"{args.base_path}/domain_{domain_name}")
        if simulation_check.exists():
            print(f"Skipping {domain_name} - simulation directory already exists")
            skipped_jobs.append(domain_name)
            continue
        
        # Extract groundwater data for this station
        #station_dir = f"{args.base_path}/domain_{domain_name}"
        #gw_file = extract_groundwater_data(
        #    args.ggmn_data_dir,
        #    station_id,
        #    station_dir,
        #    start_year=args.start_year,
        #    end_year=args.end_year
        #)
        
        #if gw_file:
        #    print(f"Extracted groundwater data: {gw_file}")
        
        # Generate config file
        config_path = os.path.join(args.config_dir, f"config_{domain_name}.yaml")
        generate_config_file(
            args.template_config, 
            config_path, 
            domain_name, 
            pour_point, 
            bounding_box,
            station_id
        )
        
        # Submit job if requested
        if submit_jobs == 'y':
            print(f"Submitting CONFLUENCE job for {domain_name}...")
            job_id = run_confluence(config_path, domain_name, args.job_time, args.memory)
            
            if job_id:
                submitted_jobs.append((domain_name, job_id))
            
            time.sleep(2)  # Delay between submissions
    
    # Print summary
    if submit_jobs == 'y':
        print(f"\nSubmitted {len(submitted_jobs)} CONFLUENCE jobs")
        print(f"Skipped {len(skipped_jobs)} existing simulations")
        
        if submitted_jobs:
            print("\nSubmitted jobs:")
            for domain_name, job_id in submitted_jobs:
                print(f"  {domain_name}: {job_id}")
    else:
        print(f"\nGenerated {len(quality_stations) - len(skipped_jobs)} config files in {args.config_dir}")
        print("Use --no_submit flag to skip this prompt in future runs")

if __name__ == "__main__":
    main()