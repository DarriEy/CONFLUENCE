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

def generate_config_file(template_path, output_path, domain_name, pour_point, bounding_box, station_id):
    """
    Generate a new config file based on the template with updated parameters
    
    Args:
        template_path: Path to the template config file
        output_path: Path to save the new config file
        domain_name: Name of the domain to set
        pour_point: Pour point coordinates to set
        bounding_box: Bounding box coordinates to set
        station_id: Station ID for data retrieval
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

    # Set up data provider and station information
    config_content = re.sub(r'STREAMFLOW_DATA_PROVIDER:.*', f'STREAMFLOW_DATA_PROVIDER: WSC', config_content)
    config_content = re.sub(r'DOWNLOAD_WSC_DATA:.*', f'DOWNLOAD_WSC_DATA: True', config_content)
    config_content = re.sub(r'STATION_ID:.*', f'STATION_ID: {station_id}', config_content)

    # For Yukon stations, use North America EM-Earth settings
    # Yukon Territory is part of the NorthAmerica region in EM-Earth
    config_content = re.sub(r'EM_EARTH_PRCP_DIR:.*', f'EM_EARTH_PRCP_DIR: /anvil/datasets/meteorological/EM-Earth/EM_Earth_v1/deterministic_hourly/prcp/NorthAmerica', config_content)
    config_content = re.sub(r'EM_EARTH_TMEAN_DIR:.*', f'EM_EARTH_TMEAN_DIR: /anvil/datasets/meteorological/EM-Earth/EM_Earth_v1/deterministic_hourly/tmean/NorthAmerica', config_content)
    config_content = re.sub(r'EM_EARTH_REGION:.*', f'EM_EARTH_REGION: NorthAmerica', config_content)

    # Save the new config file
    with open(output_path, 'w') as f:
        f.write(config_content)
    
    # Verify the changes were made
    print(f"Config file created at {output_path}")
    
    return output_path

def run_confluence(config_path, station_name):
    """
    Run CONFLUENCE with the specified config file
    
    Args:
        config_path: Path to the config file
        station_name: Name of the Yukon station for job naming
    """
    # Create a temporary batch script for this specific run
    batch_script = f"run_{station_name}.sh"
    
    with open(batch_script, 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=YK_{station_name}
#SBATCH --output=CONFLUENCE_YK_{station_name}_%j.log
#SBATCH --error=CONFLUENCE_YK_{station_name}_%j.err
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --mem=1G

# Load necessary modules
module restore confluence_modules

# Activate Python environment
conda activate confluence

# Run CONFLUENCE with the specified config
python ../CONFLUENCE/CONFLUENCE.py --config {config_path}

echo "CONFLUENCE job for Yukon station {station_name} complete"
""")
    
    # Make the script executable
    os.chmod(batch_script, 0o755)
    
    # Submit the job
    result = subprocess.run(['sbatch', batch_script], capture_output=True, text=True)
    
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted job for Yukon station {station_name}, job ID: {job_id}")
        return job_id
    else:
        print(f"Failed to submit job for Yukon station {station_name}: {result.stderr}")
        return None

def load_yukon_stations(yukon_csv_path):
    """
    Load the Yukon stations data from CSV file
    
    Args:
        yukon_csv_path: Path to the Yukon stations CSV file
        
    Returns:
        DataFrame containing Yukon station information
    """
    
    if not os.path.exists(yukon_csv_path):
        raise FileNotFoundError(f"Yukon stations CSV file not found: {yukon_csv_path}")
    
    print(f"Loading Yukon stations from file: {yukon_csv_path}")
    yukon_df = pd.read_csv(yukon_csv_path)
    
    # Clean up column names - remove leading/trailing spaces
    yukon_df.columns = yukon_df.columns.str.strip()
    
    # Clean up data in object columns
    for col in yukon_df.columns:
        if yukon_df[col].dtype == 'object':
            yukon_df[col] = yukon_df[col].astype(str).str.strip()
    
    # Map column names to standardized format based on the CSV structure provided
    column_mapping = {
        'Latitude (deg)': 'LAT',
        'Longitude (deg)': 'LON', 
        'Station Name': 'STATION',
        'Station ID': 'Site_No'
    }
    
    # Rename columns to match expected format
    yukon_df = yukon_df.rename(columns=column_mapping)
    
    # Ensure we have the required columns
    required_cols = ['LAT', 'LON', 'STATION', 'Site_No']
    missing_cols = [col for col in required_cols if col not in yukon_df.columns]
    if missing_cols:
        print("Available columns:", yukon_df.columns.tolist())
        raise ValueError(f"Missing required columns in Yukon CSV: {missing_cols}")
    
    # Filter out any stations with missing coordinates
    initial_count = len(yukon_df)
    yukon_df = yukon_df.dropna(subset=['LAT', 'LON'])
    if len(yukon_df) < initial_count:
        print(f"Filtered out {initial_count - len(yukon_df)} stations with missing coordinates")
    
    # Calculate bounding box coordinates for each station
    # Using a larger buffer for Yukon stations (0.1 degrees ~ 10km)
    # Yukon watersheds are generally larger than Hawaiian ones
    buffer = 0.1
    yukon_df['BOUNDING_BOX_COORDS'] = (
        (yukon_df['LAT'] + buffer).astype(str) + '/' + 
        (yukon_df['LON'] - buffer).astype(str) + '/' + 
        (yukon_df['LAT'] - buffer).astype(str) + '/' + 
        (yukon_df['LON'] + buffer).astype(str)
    )
    
    # Format pour point coordinates as lat/lon
    yukon_df['POUR_POINT_COORDS'] = yukon_df['LAT'].astype(str) + '/' + yukon_df['LON'].astype(str)
    
    # Extract watershed region and create clean station ID
    def extract_region_and_clean_name(station_name, latitude):
        """Extract region and create a clean station identifier"""
        # Determine region based on location in Yukon Territory
        if latitude >= 67.0:
            region = "Arctic"
        elif latitude >= 64.0:
            region = "Central_Yukon"
        else:
            region = "Southern_Yukon"
        
        # Create a clean station identifier
        # Remove common suffixes and clean up for file naming
        clean_name = station_name.replace(' ', '_')
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', clean_name)
        clean_name = re.sub(r'_+', '_', clean_name)  # Remove multiple underscores
        clean_name = clean_name.strip('_')  # Remove leading/trailing underscores
        
        # Limit length for file system compatibility
        if len(clean_name) > 50:
            clean_name = clean_name[:50]
        
        return region, clean_name
    
    # Apply the extraction function
    region_and_name = yukon_df.apply(lambda row: extract_region_and_clean_name(row['STATION'], row['LAT']), axis=1)
    yukon_df['Region'] = [x[0] for x in region_and_name]
    yukon_df['Clean_Station_Name'] = [x[1] for x in region_and_name]
    
    # Create domain name using Site_No and clean station name
    yukon_df['Domain_Name'] = 'YUKON_' + yukon_df['Site_No'].astype(str) + '_' + yukon_df['Clean_Station_Name']
    
    # Create a station ID for job naming (use Site_No)
    yukon_df['Station_ID'] = yukon_df['Site_No'].astype(str)
    
    # Create a descriptive name
    yukon_df['Description'] = yukon_df['Clean_Station_Name'] + '_' + yukon_df['Region']
    
    return yukon_df

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Set up CONFLUENCE point simulations for Yukon Territory water monitoring stations')
    parser.add_argument('--template_config', type=str, default='/home/x-deythorsson/code/CONFLUENCE/0_config_files/config_Bow_lumped.yaml',
                        help='Path to the template config file for CONFLUENCE')
    parser.add_argument('--yukon_csv', type=str, default='/home/x-deythorsson/code/apps_confluence/yukon_stations.csv',
                        help='Path to Yukon stations CSV file')
    parser.add_argument('--output_dir', type=str, default='/anvil/scratch/x-deythorsson/CONFLUENCE_data/yukon',
                        help='Directory to store processed data and outputs')
    parser.add_argument('--config_dir', type=str, default='/home/x-deythorsson/code/CONFLUENCE/0_config_files/',
                        help='Directory to store generated config files')
    parser.add_argument('--no_submit', action='store_true',
                        help='Generate configs but don\'t submit jobs')
    parser.add_argument('--base_path', type=str, default='/anvil/scratch/x-deythorsson/CONFLUENCE_data/yukon',
                        help='Base path for CONFLUENCE data directory')
    parser.add_argument('--subset_stations', type=str, nargs='+', default=None,
                        help='Run only specific stations (provide Station ID values)')
    parser.add_argument('--region_filter', type=str, choices=['Arctic', 'Central_Yukon', 'Southern_Yukon'], 
                        default=None, help='Run stations only from specific region')
    parser.add_argument('--lat_range', type=float, nargs=2, default=None,
                        help='Filter stations by latitude range (min_lat max_lat)')
    parser.add_argument('--lon_range', type=float, nargs=2, default=None,
                        help='Filter stations by longitude range (min_lon max_lon)')
    parser.add_argument('--drainage_area_range', type=float, nargs=2, default=None,
                        help='Filter stations by drainage area range in sq km (min_area max_area)')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.config_dir, exist_ok=True)
    
    # Load Yukon station data
    print("Loading Yukon Territory stations data...")
    yukon_df = load_yukon_stations(args.yukon_csv)
    
    # Apply filters if specified
    original_count = len(yukon_df)
    
    # Filter by specific stations (using Station ID)
    if args.subset_stations:
        print(f"Filtering to subset stations: {args.subset_stations}")
        yukon_df = yukon_df[yukon_df['Site_No'].isin(args.subset_stations)]
        if len(yukon_df) == 0:
            print("No matching stations found! Available Station ID values:")
            original_df = load_yukon_stations(args.yukon_csv)
            print(original_df['Site_No'].tolist())
            return
    
    # Filter by region
    if args.region_filter:
        print(f"Filtering to {args.region_filter} stations only")
        yukon_df = yukon_df[yukon_df['Region'] == args.region_filter]
        if len(yukon_df) == 0:
            print(f"No stations found in {args.region_filter}!")
            return
    
    # Filter by coordinate ranges
    if args.lat_range:
        min_lat, max_lat = args.lat_range
        print(f"Filtering to stations between {min_lat}° and {max_lat}° latitude")
        yukon_df = yukon_df[(yukon_df['LAT'] >= min_lat) & (yukon_df['LAT'] <= max_lat)]
    
    if args.lon_range:
        min_lon, max_lon = args.lon_range
        print(f"Filtering to stations between {min_lon}° and {max_lon}° longitude")
        yukon_df = yukon_df[(yukon_df['LON'] >= min_lon) & (yukon_df['LON'] <= max_lon)]
    
    # Filter by drainage area if the column exists and filter is specified
    if args.drainage_area_range and 'Drainage Area (sq km)' in yukon_df.columns:
        min_area, max_area = args.drainage_area_range
        print(f"Filtering to stations with drainage area between {min_area} and {max_area} sq km")
        # Convert drainage area to numeric, handling any non-numeric values
        yukon_df['Drainage_Area_Numeric'] = pd.to_numeric(yukon_df['Drainage Area (sq km)'], errors='coerce')
        yukon_df = yukon_df[(yukon_df['Drainage_Area_Numeric'] >= min_area) & (yukon_df['Drainage_Area_Numeric'] <= max_area)]
    
    if len(yukon_df) != original_count:
        print(f"Filtered from {original_count} to {len(yukon_df)} stations")
    
    print(f"Found {len(yukon_df)} Yukon Territory stations to process:")
    
    # Group by region for better organization
    region_groups = yukon_df.groupby('Region')
    for region, group in region_groups:
        print(f"\n  {region} ({len(group)} stations):")
        for _, station in group.iterrows():
            print(f"    - Station {station['Site_No']}: {station['Clean_Station_Name']} ({station['LAT']:.4f}, {station['LON']:.4f})")
    
    # Save processed Yukon stations data to CSV for reference
    yukon_csv = os.path.join(args.output_dir, 'processed_yukon_stations.csv')
    yukon_df.to_csv(yukon_csv, index=False)
    print(f"\nSaved processed station data to {yukon_csv}")
    
    # Process each Yukon station for CONFLUENCE runs
    submitted_jobs = []
    skipped_jobs = []
    
    # Ask if user wants to submit CONFLUENCE jobs (unless --no_submit is specified)
    submit_jobs = 'n' if args.no_submit else input(f"\nDo you want to submit CONFLUENCE jobs for these {len(yukon_df)} Yukon Territory stations? (y/n): ").lower().strip()
    
    for _, station in yukon_df.iterrows():
        # Get station parameters
        station_id = station['Station_ID']
        domain_name = station['Domain_Name']
        pour_point = station['POUR_POINT_COORDS']
        bounding_box = station['BOUNDING_BOX_COORDS']
        
        # Check if the simulations directory already exists
        print(f"Checking if simulation exists for {domain_name}...")
        simulation_dir = Path(f"{args.base_path}/domain_{domain_name}.tar.gz")
        sim_dir2 = Path(f"{args.base_path}/domain_{domain_name}/")
        
        if simulation_dir.exists():
            print(f"Skipping {domain_name} - simulation result already exists: {simulation_dir}")
            skipped_jobs.append(domain_name)
            continue

        if sim_dir2.exists():
            print(f"Skipping {domain_name} - simulation result already exists: {sim_dir2}")
            skipped_jobs.append(domain_name)
            continue
        
        # Generate the config file path
        config_path = os.path.join(args.config_dir, f"config_{domain_name}.yaml")
        
        # Generate the config file
        print(f"Generating config file for {domain_name}...")
        generate_config_file(args.template_config, config_path, domain_name, pour_point, bounding_box, station_id)
        
        # Run CONFLUENCE with the generated config if requested
        if submit_jobs == 'y':
            print(f"Submitting CONFLUENCE job for {domain_name}...")
            job_id = run_confluence(config_path, station_id)
            
            if job_id:
                submitted_jobs.append((domain_name, job_id))
            
            # Add a small delay between job submissions to avoid overwhelming the scheduler
            time.sleep(5)
    
    if submit_jobs == 'y':
        # Print summary of submitted jobs
        print("\n" + "="*60)
        print("SUBMITTED JOBS SUMMARY")
        print("="*60)
        
        # Group submitted jobs by region for better organization
        submitted_df = pd.DataFrame(submitted_jobs, columns=['Domain_Name', 'Job_ID'])
        submitted_df['Site_No'] = submitted_df['Domain_Name'].str.extract(r'YUKON_([^_]+)_')[0]
        submitted_with_regions = submitted_df.merge(yukon_df[['Site_No', 'Region', 'Clean_Station_Name']], on='Site_No', how='left')
        
        region_job_groups = submitted_with_regions.groupby('Region')
        for region, group in region_job_groups:
            print(f"\n{region} ({len(group)} jobs):")
            for _, job in group.iterrows():
                print(f"  Station {job['Site_No']} ({job['Clean_Station_Name']}): Job ID {job['Job_ID']}")
        
        print(f"\nTotal jobs submitted: {len(submitted_jobs)}")
        print(f"Total jobs skipped: {len(skipped_jobs)}")
        
        if skipped_jobs:
            print("\nSkipped domains (simulations already exist):")
            for domain_name in skipped_jobs:
                print(f"- {domain_name}")
                
        print(f"\nTo monitor jobs, use: squeue -u $USER | grep YK_")
        print(f"To check job status: sacct -j <job_id>")
        
    else:
        print("\n" + "="*60)
        print("SETUP COMPLETE - NO JOBS SUBMITTED")  
        print("="*60)
        print(f"Config files have been generated in {args.config_dir}")
        print(f"Station data saved to {yukon_csv}")
        print(f"Total jobs skipped (already exist): {len(skipped_jobs)}")
        
        if len(yukon_df) - len(skipped_jobs) > 0:
            print(f"\nTo submit jobs later, run:")
            print(f"python run_yukon_confluence.py --template_config {args.template_config} --yukon_csv {args.yukon_csv} --config_dir {args.config_dir} --base_path {args.base_path}")
            print(f"\nOr submit individual jobs using:")
            print(f"sbatch run_<station_id>.sh")

if __name__ == "__main__":
    main()