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

    config_content = re.sub(r'STREAMFLOW_DATA_PROVIDER:.*', f'STREAMFLOW_DATA_PROVIDER: USGS', config_content)
    config_content = re.sub(r'DOWNLOAD_USGS_DATA:.*', f'DOWNLOAD_USGS_DATA: True', config_content)
    config_content = re.sub(r'STATION_ID:.*', f'STATION_ID: {station_id}', config_content)


    # For Hawaii stations, we'll use North America EM-Earth settings
    # Note: You may need to verify if Hawaii data is included in the NorthAmerica region
    # or if there's a separate Pacific/Hawaii dataset available
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
        station_name: Name of the Hawaii station for job naming
    """
    # Create a temporary batch script for this specific run
    batch_script = f"run_{station_name}.sh"
    
    with open(batch_script, 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=HI_{station_name}
#SBATCH --output=CONFLUENCE_HI_{station_name}_%j.log
#SBATCH --error=CONFLUENCE_HI_{station_name}_%j.err
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --mem=1G

# Load necessary modules
module restore confluence_modules

# Activate Python environment
conda activate confluence

# Run CONFLUENCE with the specified config
python ../CONFLUENCE/CONFLUENCE.py --config {config_path}

echo "CONFLUENCE job for Hawaii station {station_name} complete"
""")
    
    # Make the script executable
    os.chmod(batch_script, 0o755)
    
    # Submit the job
    result = subprocess.run(['sbatch', batch_script], capture_output=True, text=True)
    
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted job for Hawaii station {station_name}, job ID: {job_id}")
        return job_id
    else:
        print(f"Failed to submit job for Hawaii station {station_name}: {result.stderr}")
        return None

def load_hawaii_stations(hawaii_csv_path):
    """
    Load the Hawaii stations data from CSV file
    
    Args:
        hawaii_csv_path: Path to the Hawaii stations CSV file
        
    Returns:
        DataFrame containing Hawaii station information
    """
    
    if not os.path.exists(hawaii_csv_path):
        raise FileNotFoundError(f"Hawaii stations CSV file not found: {hawaii_csv_path}")
    
    print(f"Loading Hawaii stations from file: {hawaii_csv_path}")
    hawaii_df = pd.read_csv(hawaii_csv_path)
    
    # Clean up column names - remove leading/trailing spaces
    hawaii_df.columns = hawaii_df.columns.str.strip()
    
    # Clean up data in object columns
    for col in hawaii_df.columns:
        if hawaii_df[col].dtype == 'object':
            hawaii_df[col] = hawaii_df[col].astype(str).str.strip()
    
    # Ensure we have the required columns (now without leading spaces)
    required_cols = ['LAT', 'LON', 'STATION', 'Site_No']
    missing_cols = [col for col in required_cols if col not in hawaii_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in Hawaii CSV: {missing_cols}")
    
    # Filter out any stations with missing coordinates
    initial_count = len(hawaii_df)
    hawaii_df = hawaii_df.dropna(subset=['LAT', 'LON'])
    if len(hawaii_df) < initial_count:
        print(f"Filtered out {initial_count - len(hawaii_df)} stations with missing coordinates")
    
    # Calculate bounding box coordinates for each station
    # Using a small buffer for Hawaii stations (0.02 degrees ~ 2km)
    # Hawaii islands are relatively small, so smaller buffer is appropriate
    buffer = 0.02
    hawaii_df['BOUNDING_BOX_COORDS'] = (
        (hawaii_df['LAT'] + buffer).astype(str) + '/' + 
        (hawaii_df['LON'] - buffer).astype(str) + '/' + 
        (hawaii_df['LAT'] - buffer).astype(str) + '/' + 
        (hawaii_df['LON'] + buffer).astype(str)
    )
    
    # Format pour point coordinates as lat/lon
    hawaii_df['POUR_POINT_COORDS'] = hawaii_df['LAT'].astype(str) + '/' + hawaii_df['LON'].astype(str)
    
    # Extract island name from station name and create clean station ID
    def extract_island_and_clean_name(station_name):
        """Extract island name and create a clean station identifier"""
        # Extract island name (look for common Hawaiian island names in the station string)
        island = "Unknown"
        if "Oahu" in station_name:
            island = "Oahu"
        elif "Maui" in station_name:
            island = "Maui"
        elif "Hawaii" in station_name or "Big_Island" in station_name:
            island = "Hawaii_Big_Island"
        elif "Kauai" in station_name:
            island = "Kauai"
        elif "Molokai" in station_name:
            island = "Molokai"
        elif "Lanai" in station_name:
            island = "Lanai"
        
        # Create a clean station identifier by taking the first part before location info
        # Remove common suffixes and clean up for file naming
        clean_name = station_name.split('__')[0]  # Take part before double underscore
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', clean_name)
        clean_name = re.sub(r'_+', '_', clean_name)  # Remove multiple underscores
        clean_name = clean_name.strip('_')  # Remove leading/trailing underscores
        
        return island, clean_name
    
    # Apply the extraction function
    island_and_name = hawaii_df['STATION'].apply(extract_island_and_clean_name)
    hawaii_df['Island'] = [x[0] for x in island_and_name]
    hawaii_df['Clean_Station_Name'] = [x[1] for x in island_and_name]
    
    # Create domain name using Site_No and clean station name
    hawaii_df['Domain_Name'] = 'HAWAII_' + hawaii_df['Site_No'].astype(str) + '_' + hawaii_df['Clean_Station_Name']
    
    # Create a shorter station ID for job naming (just use Site_No)
    hawaii_df['Station_ID'] = hawaii_df['Site_No'].astype(str)
    
    # Create a descriptive name
    hawaii_df['Description'] = hawaii_df['Clean_Station_Name'] + '_' + hawaii_df['Island']
    
    return hawaii_df

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Set up CONFLUENCE point simulations for Hawaiian Islands water monitoring stations')
    parser.add_argument('--template_config', type=str, default='/home/x-deythorsson/code/CONFLUENCE/0_config_files/config_Bow_lumped.yaml',
                        help='Path to the template config file for CONFLUENCE')
    parser.add_argument('--hawaii_csv', type=str, default='/home/x-deythorsson/code/apps_confluence/hawaiian_islands_stations.csv' ,
                        help='Path to Hawaiian islands stations CSV file')
    parser.add_argument('--output_dir', type=str, default='/anvil/scratch/x-deythorsson/CONFLUENCE_data/hawaii',
                        help='Directory to store processed data and outputs')
    parser.add_argument('--config_dir', type=str, default='/home/x-deythorsson/code/CONFLUENCE/0_config_files/',
                        help='Directory to store generated config files')
    parser.add_argument('--no_submit', action='store_true',
                        help='Generate configs but don\'t submit jobs')
    parser.add_argument('--base_path', type=str, default='/anvil/scratch/x-deythorsson/CONFLUENCE_data/hawaii',
                        help='Base path for CONFLUENCE data directory')
    parser.add_argument('--subset_stations', type=str, nargs='+', default=None,
                        help='Run only specific stations (provide Site_No values like 16211600 16330000)')
    parser.add_argument('--island_filter', type=str, choices=['Oahu', 'Maui', 'Hawaii_Big_Island', 'Kauai', 'Molokai', 'Lanai'], 
                        default=None, help='Run stations only from specific island')
    parser.add_argument('--lat_range', type=float, nargs=2, default=None,
                        help='Filter stations by latitude range (min_lat max_lat)')
    parser.add_argument('--lon_range', type=float, nargs=2, default=None,
                        help='Filter stations by longitude range (min_lon max_lon)')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.config_dir, exist_ok=True)
    
    # Load Hawaii station data
    print("Loading Hawaiian Islands stations data...")
    hawaii_df = load_hawaii_stations(args.hawaii_csv)
    
    # Apply filters if specified
    original_count = len(hawaii_df)
    
    # Filter by specific stations (using Site_No)
    if args.subset_stations:
        print(f"Filtering to subset stations: {args.subset_stations}")
        # Convert to integers for comparison
        subset_site_nos = [int(x) for x in args.subset_stations]
        hawaii_df = hawaii_df[hawaii_df['Site_No'].isin(subset_site_nos)]
        if len(hawaii_df) == 0:
            print("No matching stations found! Available Site_No values:")
            original_df = load_hawaii_stations(args.hawaii_csv)
            print(original_df['Site_No'].tolist())
            return
    
    # Filter by island
    if args.island_filter:
        print(f"Filtering to {args.island_filter} stations only")
        hawaii_df = hawaii_df[hawaii_df['Island'] == args.island_filter]
        if len(hawaii_df) == 0:
            print(f"No stations found on {args.island_filter}!")
            return
    
    # Filter by coordinate ranges
    if args.lat_range:
        min_lat, max_lat = args.lat_range
        print(f"Filtering to stations between {min_lat}° and {max_lat}° latitude")
        hawaii_df = hawaii_df[(hawaii_df['LAT'] >= min_lat) & (hawaii_df['LAT'] <= max_lat)]
    
    if args.lon_range:
        min_lon, max_lon = args.lon_range
        print(f"Filtering to stations between {min_lon}° and {max_lon}° longitude")
        hawaii_df = hawaii_df[(hawaii_df['LON'] >= min_lon) & (hawaii_df['LON'] <= max_lon)]
    
    if len(hawaii_df) != original_count:
        print(f"Filtered from {original_count} to {len(hawaii_df)} stations")
    
    print(f"Found {len(hawaii_df)} Hawaiian Islands stations to process:")
    
    # Group by island for better organization
    island_groups = hawaii_df.groupby('Island')
    for island, group in island_groups:
        print(f"\n  {island} ({len(group)} stations):")
        for _, station in group.iterrows():
            print(f"    - Site {station['Site_No']}: {station['Clean_Station_Name']} ({station['LAT']:.4f}, {station['LON']:.4f})")
    
    # Save processed Hawaii stations data to CSV for reference
    hawaii_csv = os.path.join(args.output_dir, 'processed_hawaii_stations.csv')
    hawaii_df.to_csv(hawaii_csv, index=False)
    print(f"\nSaved processed station data to {hawaii_csv}")
    
    # Process each Hawaii station for CONFLUENCE runs
    submitted_jobs = []
    skipped_jobs = []
    
    # Ask if user wants to submit CONFLUENCE jobs (unless --no_submit is specified)
    submit_jobs = 'n' if args.no_submit else input(f"\nDo you want to submit CONFLUENCE jobs for these {len(hawaii_df)} Hawaiian Islands stations? (y/n): ").lower().strip()
    
    for _, station in hawaii_df.iterrows():
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
        
        # Group submitted jobs by island for better organization
        submitted_df = pd.DataFrame(submitted_jobs, columns=['Domain_Name', 'Job_ID'])
        submitted_df['Site_No'] = submitted_df['Domain_Name'].str.extract(r'HAWAII_(\d+)_')[0].astype(int)
        submitted_with_islands = submitted_df.merge(hawaii_df[['Site_No', 'Island', 'Clean_Station_Name']], on='Site_No', how='left')
        
        island_job_groups = submitted_with_islands.groupby('Island')
        for island, group in island_job_groups:
            print(f"\n{island} ({len(group)} jobs):")
            for _, job in group.iterrows():
                print(f"  Site {job['Site_No']} ({job['Clean_Station_Name']}): Job ID {job['Job_ID']}")
        
        print(f"\nTotal jobs submitted: {len(submitted_jobs)}")
        print(f"Total jobs skipped: {len(skipped_jobs)}")
        
        if skipped_jobs:
            print("\nSkipped domains (simulations already exist):")
            for domain_name in skipped_jobs:
                print(f"- {domain_name}")
                
        print(f"\nTo monitor jobs, use: squeue -u $USER | grep HI_")
        print(f"To check job status: sacct -j <job_id>")
        
    else:
        print("\n" + "="*60)
        print("SETUP COMPLETE - NO JOBS SUBMITTED")  
        print("="*60)
        print(f"Config files have been generated in {args.config_dir}")
        print(f"Station data saved to {hawaii_csv}")
        print(f"Total jobs skipped (already exist): {len(skipped_jobs)}")
        
        if len(hawaii_df) - len(skipped_jobs) > 0:
            print(f"\nTo submit jobs later, run:")
            print(f"python run_hawaii_confluence.py --template_config {args.template_config} --hawaii_csv {args.hawaii_csv} --config_dir {args.config_dir} --base_path {args.base_path}")
            print(f"\nOr submit individual jobs using:")
            print(f"sbatch run_<site_no>.sh")

if __name__ == "__main__":
    main()