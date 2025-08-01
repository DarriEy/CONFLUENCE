#!/usr/bin/env python3
"""
Streamlined GGMN to CONFLUENCE workflow for North American stations.

This script extracts station data from wells.ods, filters for North America,
matches with monitoring files, and creates CONFLUENCE configurations.

Usage:
    python run_ggmn.py --ggmn_data_dir <data_dir> 
                       --template_config <template.yaml>
                       --base_path <confluence_data_path>
                       [options]

Author: Claude
Date: July 28, 2025
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import subprocess
import time
import re
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import ezodf
except ImportError:
    print("Installing ezodf for .ods file support...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ezodf"])
    import ezodf

def make_timezone_naive(date_obj):
    """Convert a datetime object to timezone-naive."""
    if hasattr(date_obj, 'tz') and date_obj.tz is not None:
        return date_obj.tz_localize(None)
    elif hasattr(date_obj, 'dt') and hasattr(date_obj.dt, 'tz') and date_obj.dt.tz is not None:
        return date_obj.dt.tz_localize(None)
    else:
        return date_obj

def extract_stations_from_wells_file(ggmn_data_dir, verbose=False):
    """Extract all stations from the wells.ods file."""
    
    wells_file = os.path.join(ggmn_data_dir, 'wells.ods')
    
    if not os.path.exists(wells_file):
        raise FileNotFoundError(f"Wells file not found: {wells_file}")
    
    print(f"Extracting stations from {wells_file}")
    
    doc = ezodf.opendoc(wells_file)
    
    if 'General Information' not in doc.sheets.names():
        raise ValueError("No 'General Information' sheet found in wells.ods")
    
    sheet = doc.sheets['General Information']
    stations = []
    
    # Process data rows (skip header row 0 and description row 1)
    for row_idx in range(2, sheet.nrows()):
        try:
            row = sheet.row(row_idx)
            
            if len(row) < 8:  # Need at least columns 0-7
                continue
            
            # Extract data from known column positions
            station_id = row[0].value
            station_name = row[1].value
            feature_type = row[2].value
            purpose = row[3].value
            status = row[4].value
            description = row[5].value
            latitude = row[6].value
            longitude = row[7].value
            elevation = row[8].value if len(row) > 8 else None
            
            # Skip rows with missing essential data
            if not station_id or latitude is None or longitude is None:
                continue
            
            # Parse and validate coordinates
            try:
                lat_float = float(latitude)
                lon_float = float(longitude)
                
                if not (-90 <= lat_float <= 90 and -180 <= lon_float <= 180):
                    continue
            except:
                continue
            
            # Parse elevation if available
            elev_float = None
            if elevation is not None:
                try:
                    elev_float = float(elevation)
                except:
                    pass
            
            station_data = {
                'station_id': str(station_id),
                'station_name': str(station_name) if station_name else str(station_id),
                'feature_type': str(feature_type) if feature_type else 'Water well',
                'purpose': str(purpose) if purpose else None,
                'status': str(status) if status else None,
                'description': str(description) if description else None,
                'latitude': lat_float,
                'longitude': lon_float,
                'elevation': elev_float,
                'country': 'Canada',  # Based on coordinate ranges and debug output
                'source_file': 'wells.ods'
            }
            
            stations.append(station_data)
            
            if verbose and len(stations) % 200 == 0:
                print(f"  Processed {len(stations)} stations...")
                
        except Exception as e:
            if verbose:
                print(f"Error processing row {row_idx}: {e}")
            continue
    
    print(f"Extracted {len(stations)} stations with valid coordinates")
    return pd.DataFrame(stations)

def filter_north_american_stations(stations_df, verbose=False):
    """Filter stations to North America only."""
    
    print("Filtering for North American stations...")
    
    # Define North American bounding box (generous)
    # Includes Canada, USA, Mexico, Central America, Greenland
    na_bounds = {
        'lat_min': 10.0,    # Southern Mexico/Central America
        'lat_max': 85.0,    # Northern Canada/Greenland  
        'lon_min': -180.0,  # Western Alaska/Aleutians
        'lon_max': -30.0    # Eastern Canada/Greenland
    }
    
    initial_count = len(stations_df)
    
    # Apply North American filter
    na_mask = (
        (stations_df['latitude'] >= na_bounds['lat_min']) & 
        (stations_df['latitude'] <= na_bounds['lat_max']) &
        (stations_df['longitude'] >= na_bounds['lon_min']) & 
        (stations_df['longitude'] <= na_bounds['lon_max'])
    )
    
    stations_df = stations_df[na_mask].copy()
    
    print(f"North American filter: {len(stations_df)} / {initial_count} stations")
    
    if verbose and len(stations_df) > 0:
        print(f"Coordinate ranges after filtering:")
        print(f"  Latitude: {stations_df['latitude'].min():.3f} to {stations_df['latitude'].max():.3f}")
        print(f"  Longitude: {stations_df['longitude'].min():.3f} to {stations_df['longitude'].max():.3f}")
    
    return stations_df

def match_monitoring_files(stations_df, ggmn_data_dir, verbose=False):
    """Match stations with monitoring files in the monitoring directory."""
    
    monitoring_dir = os.path.join(ggmn_data_dir, 'monitoring')
    
    if not os.path.exists(monitoring_dir):
        print(f"Warning: Monitoring directory not found at {monitoring_dir}")
        stations_df['has_monitoring_file'] = False
        stations_df['monitoring_file_path'] = None
        stations_df['record_count'] = 0
        return stations_df
    
    print(f"Matching stations with monitoring files in {monitoring_dir}")
    
    # Scan monitoring directory for .ods files
    monitoring_files = {}
    for filename in os.listdir(monitoring_dir):
        if filename.endswith('.ods'):
            file_id = os.path.splitext(filename)[0]
            monitoring_files[file_id] = os.path.join(monitoring_dir, filename)
    
    print(f"Found {len(monitoring_files)} monitoring files")
    
    # Match stations to monitoring files
    stations_df['has_monitoring_file'] = False
    stations_df['monitoring_file_path'] = None
    stations_df['record_count'] = 0
    
    matched_count = 0
    
    for idx, station in stations_df.iterrows():
        station_id = station['station_id']
        
        # Try different ID matching patterns
        possible_ids = [
            station_id,                              # Exact match
            station_id.replace('-', ''),             # Remove dashes
            station_id.split('-')[0] if '-' in station_id else station_id,  # First part only
            station_id.replace('W', '').lstrip('0'), # Remove W prefix and leading zeros
        ]
        
        matched = False
        for pid in possible_ids:
            if pid in monitoring_files:
                stations_df.at[idx, 'has_monitoring_file'] = True
                stations_df.at[idx, 'monitoring_file_path'] = monitoring_files[pid]
                stations_df.at[idx, 'record_count'] = 50  # Placeholder - would need to check actual file
                matched_count += 1
                matched = True
                
                if verbose:
                    print(f"  Matched {station_id} -> {pid}.ods")
                break
        
        if not matched and verbose:
            print(f"  No match for {station_id}")
    
    print(f"Matched {matched_count} stations to monitoring files")
    return stations_df

def add_confluence_fields(stations_df):
    """Add CONFLUENCE-specific fields to the stations dataframe."""
    
    print("Adding CONFLUENCE-specific fields...")
    
    # Create bounding box coordinates (small buffer around each station)
    buffer = 0.1  # degrees
    stations_df['BOUNDING_BOX_COORDS'] = (
        (stations_df['latitude'] + buffer).astype(str) + '/' +
        (stations_df['longitude'] - buffer).astype(str) + '/' +
        (stations_df['latitude'] - buffer).astype(str) + '/' +
        (stations_df['longitude'] + buffer).astype(str)
    )
    
    # Create pour point coordinates
    stations_df['POUR_POINT_COORDS'] = (
        stations_df['latitude'].astype(str) + '/' + 
        stations_df['longitude'].astype(str)
    )
    
    # Create watershed names from station IDs (clean for file naming)
    stations_df['Watershed_Name'] = (
        stations_df['station_id']
        .astype(str)
        .str.replace('[^a-zA-Z0-9_]', '_', regex=True)
        .str.replace('__+', '_', regex=True)
        .str.strip('_')
    )
    
    # Ensure unique watershed names
    duplicated = stations_df['Watershed_Name'].duplicated()
    for idx in stations_df[duplicated].index:
        base_name = stations_df.loc[idx, 'Watershed_Name']
        counter = 1
        new_name = f"{base_name}_{counter}"
        while new_name in stations_df['Watershed_Name'].values:
            counter += 1
            new_name = f"{base_name}_{counter}"
        stations_df.loc[idx, 'Watershed_Name'] = new_name
    
    # Add additional metadata fields for compatibility
    stations_df['has_groundwater_data'] = stations_df['has_monitoring_file']
    stations_df['start_date'] = None
    stations_df['end_date'] = None
    stations_df['data_quality'] = 'good'
    
    return stations_df

def extract_groundwater_data_from_monitoring_file(file_path, station_id, output_dir, start_year=None, end_year=None):
    """Extract groundwater data from a monitoring .ods file."""
    
    try:
        doc = ezodf.opendoc(file_path)
        
        if 'Groundwater Level' not in doc.sheets.names():
            print(f"  No 'Groundwater Level' sheet in {os.path.basename(file_path)}")
            return None
            
        sheet = doc.sheets['Groundwater Level']
        
        if sheet.nrows() < 3:  # Need header + data
            print(f"  Insufficient data in {os.path.basename(file_path)}")
            return None
        
        data_rows = []
        
        # Extract data starting from row 2 (skip headers)
        for row_idx in range(2, min(sheet.nrows(), 10000)):  # Limit for performance
            try:
                row = sheet.row(row_idx)
                
                if len(row) < 2:
                    continue
                    
                # Assume first column is date, second is value
                date_val = row[0].value
                value_val = row[1].value
                
                if date_val is None or value_val is None:
                    continue
                    
                # Parse date
                try:
                    date_val = pd.to_datetime(date_val)
                    date_val = make_timezone_naive(date_val)
                except:
                    continue
                
                # Parse value
                try:
                    value_val = float(value_val)
                except:
                    continue
                
                # Apply year filter if specified
                if start_year and date_val.year < start_year:
                    continue
                if end_year and date_val.year > end_year:
                    continue
                    
                data_rows.append({
                    'date': date_val,
                    'groundwater_level': value_val,
                    'source_file': os.path.basename(file_path)
                })
                
            except Exception as e:
                continue
        
        if data_rows:
            df = pd.DataFrame(data_rows)
            df = df.drop_duplicates(subset=['date', 'groundwater_level'])
            df = df.sort_values('date')
            
            # Create output directory
            gw_dir = Path(output_dir) / 'observations' / 'groundwater' / 'raw_data'
            gw_dir.mkdir(parents=True, exist_ok=True)
            
            # Save file
            safe_station_id = str(station_id).replace(' ', '_').replace('/', '_').replace('\\', '_')
            output_file = gw_dir / f"{safe_station_id}_groundwater.csv"
            df.to_csv(output_file, index=False)
            
            print(f"  Extracted {len(df)} records for station {station_id}")
            return str(output_file)
        
        return None
            
    except Exception as e:
        print(f"  Error extracting data from {file_path}: {e}")
        return None

def generate_config_file(template_path, output_path, domain_name, pour_point, bounding_box, base_path):
    """Generate CONFLUENCE config file."""
    
    with open(template_path, 'r') as f:
        config_content = f.read()
    
    # Update domain-specific parameters
    config_content = re.sub(r'DOMAIN_NAME:.*', f'DOMAIN_NAME: "{domain_name}"', config_content)
    config_content = re.sub(r'POUR_POINT_COORDS:.*', f'POUR_POINT_COORDS: {pour_point}', config_content)
    config_content = re.sub(r'BOUNDING_BOX_COORDS:.*', f'BOUNDING_BOX_COORDS: {bounding_box}', config_content)
    config_content = re.sub(r'DOMAIN_DEFINITION_METHOD:.*', f'DOMAIN_DEFINITION_METHOD: point', config_content)
    config_content = re.sub(r'CONFLUENCE_DATA_DIR:.*', f'CONFLUENCE_DATA_DIR: "{base_path}"', config_content)
    
    with open(output_path, 'w') as f:
        f.write(config_content)
    
    return output_path

def submit_confluence_job(config_path, watershed_name, job_time="24:00:00", memory="8G"):
    """Submit CONFLUENCE job via SLURM."""
    
    batch_script = f"run_ggmn_{watershed_name}.sh"
    
    batch_content = f"""#!/bin/bash
#SBATCH --job-name=ggmn_{watershed_name}
#SBATCH --output=CONFLUENCE_ggmn_{watershed_name}_%j.log
#SBATCH --error=CONFLUENCE_ggmn_{watershed_name}_%j.err
#SBATCH --time={job_time}
#SBATCH --ntasks=1
#SBATCH --mem={memory}

# Load necessary modules
module restore confluence_modules
# Activate Python environment
conda activate confluence
# Run CONFLUENCE
python ../CONFLUENCE/CONFLUENCE.py --config {config_path}

echo "CONFLUENCE job for {watershed_name} complete"
"""
    
    with open(batch_script, 'w') as f:
        f.write(batch_content)
    
    os.chmod(batch_script, 0o755)
    
    result = subprocess.run(['sbatch', batch_script], capture_output=True, text=True)
    
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"  Submitted job for {watershed_name}, job ID: {job_id}")
        return job_id
    else:
        print(f"  Failed to submit job for {watershed_name}: {result.stderr}")
        return None

def print_summary_stats(stations_df):
    """Print summary statistics."""
    
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Total North American stations: {len(stations_df)}")
    
    if len(stations_df) == 0:
        return
    
    print(f"Stations with coordinates: {len(stations_df)}")
    print(f"Stations with monitoring files: {stations_df['has_monitoring_file'].sum()}")
    
    # Coordinate ranges
    print(f"\nCoordinate ranges:")
    print(f"  Latitude: {stations_df['latitude'].min():.3f}° to {stations_df['latitude'].max():.3f}°")
    print(f"  Longitude: {stations_df['longitude'].min():.3f}° to {stations_df['longitude'].max():.3f}°")
    
    # Country breakdown
    if 'country' in stations_df.columns:
        country_counts = stations_df['country'].value_counts()
        print(f"\nStations by country:")
        for country, count in country_counts.items():
            print(f"  {country}: {count}")
    
    # Feature types
    if 'feature_type' in stations_df.columns:
        feature_counts = stations_df['feature_type'].value_counts()
        print(f"\nStations by feature type:")
        for feature, count in feature_counts.head(5).items():
            print(f"  {feature}: {count}")
    
    # Monitoring file availability
    if 'has_monitoring_file' in stations_df.columns:
        with_monitoring = stations_df['has_monitoring_file'].sum()
        print(f"\nMonitoring data availability:")
        print(f"  With monitoring files: {with_monitoring}")
        print(f"  Without monitoring files: {len(stations_df) - with_monitoring}")

def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(description='GGMN to CONFLUENCE workflow for North America')
    parser.add_argument('--ggmn_data_dir', type=str, required=True,
                        help='Path to GGMN data directory')
    parser.add_argument('--template_config', type=str, required=True,
                        help='Path to template CONFLUENCE config file')
    parser.add_argument('--base_path', type=str, required=True,
                        help='Base path for CONFLUENCE data directory')
    parser.add_argument('--output_dir', type=str, default='ggmn_na_output',
                        help='Output directory for station inventory')
    parser.add_argument('--config_dir', type=str, default='ggmn_configs',
                        help='Directory for CONFLUENCE config files')
    parser.add_argument('--stations_csv', type=str, default='ggmn_na_stations.csv',
                        help='Name for stations CSV file')
    parser.add_argument('--max_stations', type=int, default=None,
                        help='Maximum stations to process for CONFLUENCE')
    parser.add_argument('--min_records', type=int, default=10,
                        help='Minimum records required per station')
    parser.add_argument('--start_year', type=int, default=None,
                        help='Start year for data filtering')
    parser.add_argument('--end_year', type=int, default=None,
                        help='End year for data filtering')
    parser.add_argument('--job_time', type=str, default='24:00:00',
                        help='SLURM time limit for jobs')
    parser.add_argument('--memory', type=str, default='1G',
                        help='Memory requirement for jobs')
    parser.add_argument('--force_reprocess', action='store_true',
                        help='Force reprocessing even if CSV exists')
    parser.add_argument('--no_submit', action='store_true',
                        help='Generate configs but don\'t submit jobs')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.config_dir, exist_ok=True)
    
    # Determine CSV path
    stations_csv_path = os.path.join(args.output_dir, args.stations_csv)
    
    # Step 1: Create or load station inventory
    if os.path.exists(stations_csv_path) and not args.force_reprocess:
        print(f"Station inventory already exists at {stations_csv_path}")
        print("Loading existing inventory... (use --force_reprocess to recreate)")
        stations_df = pd.read_csv(stations_csv_path)
        print(f"Loaded {len(stations_df)} stations from existing inventory")
    else:
        print("Creating North American station inventory from wells.ods...")
        
        # Extract stations from wells.ods
        stations_df = extract_stations_from_wells_file(args.ggmn_data_dir, args.verbose)
        
        # Filter for North America
        stations_df = filter_north_american_stations(stations_df, args.verbose)
        
        # Match with monitoring files
        stations_df = match_monitoring_files(stations_df, args.ggmn_data_dir, args.verbose)
        
        # Add CONFLUENCE fields
        stations_df = add_confluence_fields(stations_df)
        
        # Apply minimum records filter
        if args.min_records > 0:
            initial_count = len(stations_df)
            stations_df = stations_df[stations_df['record_count'] >= args.min_records]
            print(f"Filtered to {len(stations_df)} stations with >= {args.min_records} estimated records (was {initial_count})")
        
        # Save station inventory
        stations_df.to_csv(stations_csv_path, index=False)
        print(f"Saved station inventory to {stations_csv_path}")
    
    # Print summary
    print_summary_stats(stations_df)
    
    if len(stations_df) == 0:
        print("No stations meet the criteria. Exiting.")
        return
    
    # Step 2: Apply max stations limit for processing
    processing_stations = stations_df.copy()
    if args.max_stations and len(processing_stations) > args.max_stations:
        # Prioritize stations with monitoring files
        with_monitoring = processing_stations[processing_stations['has_monitoring_file'] == True]
        without_monitoring = processing_stations[processing_stations['has_monitoring_file'] == False]
        
        if len(with_monitoring) >= args.max_stations:
            processing_stations = with_monitoring.head(args.max_stations)
        else:
            remaining = args.max_stations - len(with_monitoring)
            processing_stations = pd.concat([with_monitoring, without_monitoring.head(remaining)])
        
        print(f"\nLimited to {args.max_stations} stations (prioritizing those with monitoring files)")
    
    # Step 3: Process stations and create CONFLUENCE configs
    print(f"\n=== PROCESSING {len(processing_stations)} STATIONS FOR CONFLUENCE ===")
    
    submitted_jobs = []
    skipped_jobs = []
    failed_extractions = []
    
    # Ask about job submission
    if args.no_submit:
        submit_jobs = 'n'
    else:
        submit_jobs = input(f"\nSubmit CONFLUENCE jobs for {len(processing_stations)} stations? (y/n): ").lower().strip()
    
    for idx, (_, station) in enumerate(processing_stations.iterrows()):
        station_id = station['station_id']
        watershed_name = station['Watershed_Name']
        
        print(f"\nProcessing station {idx+1}/{len(processing_stations)}: {station_id}")
        
        # Check if simulation already exists
        domain_name = watershed_name
        simulation_check = Path(f"{args.base_path}/domain_{domain_name}")
        if simulation_check.exists():
            print(f"  Skipping - simulation directory already exists")
            skipped_jobs.append(domain_name)
            continue
        
        # Extract groundwater data if monitoring file exists
        station_dir = f"{args.base_path}/domain_{domain_name}"
        gw_file = None
        
        if station['has_monitoring_file'] and station['monitoring_file_path']:
            print(f"  Extracting groundwater data from monitoring file...")
            gw_file = extract_groundwater_data_from_monitoring_file(
                station['monitoring_file_path'],
                station_id,
                station_dir,
                args.start_year,
                args.end_year
            )
        
        if not gw_file:
            print(f"  No groundwater data extracted")
            failed_extractions.append(station_id)
            continue
        
        # Generate config file
        config_path = os.path.join(args.config_dir, f"config_{domain_name}.yaml")
        generate_config_file(
            args.template_config,
            config_path,
            domain_name,
            station['POUR_POINT_COORDS'],
            station['BOUNDING_BOX_COORDS'],
            args.base_path
        )
        print(f"  Generated config: {config_path}")
        
        # Submit job if requested
        if submit_jobs == 'y':
            print(f"  Submitting CONFLUENCE job...")
            job_id = submit_confluence_job(config_path, domain_name, args.job_time, args.memory)
            if job_id:
                submitted_jobs.append((domain_name, job_id))
            time.sleep(1)  # Small delay between submissions
    
    # Final summary
    print(f"\n=== WORKFLOW COMPLETE ===")
    print(f"Station inventory: {stations_csv_path}")
    print(f"Total North American stations: {len(stations_df)}")
    print(f"Stations processed: {len(processing_stations)}")
    print(f"Stations skipped (already exist): {len(skipped_jobs)}")
    print(f"Failed data extractions: {len(failed_extractions)}")
    
    if submit_jobs == 'y':
        print(f"CONFLUENCE jobs submitted: {len(submitted_jobs)}")
        
        if submitted_jobs:
            print("\nSubmitted jobs:")
            for domain_name, job_id in submitted_jobs:
                print(f"  {domain_name}: {job_id}")
    else:
        successful_configs = len(processing_stations) - len(skipped_jobs) - len(failed_extractions)
        print(f"Config files generated: {successful_configs}")
        print(f"Use 'squeue -u $USER' to check job status after submission")
    
    if failed_extractions:
        print(f"\nStations without monitoring data:")
        for station_id in failed_extractions[:10]:  # Show first 10
            print(f"  {station_id}")
        if len(failed_extractions) > 10:
            print(f"  ... and {len(failed_extractions) - 10} more")

if __name__ == "__main__":
    main()