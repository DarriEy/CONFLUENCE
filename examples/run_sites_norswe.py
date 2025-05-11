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

def process_norswe_data(norswe_path, output_csv_path, start_year=None, end_year=None, use_existing_csv=True):
    """
    Process NorSWE dataset and extract station metadata to a CSV file
    
    Args:
        norswe_path: Path to the NorSWE NetCDF file
        output_csv_path: Path to save the processed station data as CSV
        start_year: Optional start year for filtering data
        end_year: Optional end year for filtering data
        use_existing_csv: Whether to use existing CSV file if it exists (default: True)
    
    Returns:
        DataFrame containing processed station data
    """
    # Check if CSV already exists and we're allowed to use it
    if use_existing_csv and os.path.exists(output_csv_path):
        print(f"Using existing processed station data from {output_csv_path}")
        return pd.read_csv(output_csv_path)
    
    print(f"Reading NorSWE data from {norswe_path}...")
    
    # Open dataset using dask for efficient processing
    ds = xr.open_dataset(norswe_path, chunks={'time': 'auto'})
    
    # Extract station metadata
    print("Extracting station metadata...")
    
    # Convert station metadata to dataframe
    stations_df = pd.DataFrame({
        'station_id': ds.station_id.values,
        'station_name': ds.station_name.values,
        'lat': ds.lat.values,
        'lon': ds.lon.values,
        'elevation': ds.elevation.values,
        'source': ds.source.values,
        'type_mes': ds.type_mes.values,
        'mmask': ds.mmask.values  # Mountain mask
    })
    
    # Add column to indicate if the station is SNOTEL
    stations_df['is_snotel'] = stations_df['station_id'].str.startswith('SNOTEL')
    
    # Check for data availability by computing mean SWE per station
    print("Checking data availability...")
    
    # Create a time mask if year range is specified
    if start_year is not None and end_year is not None:
        time_mask = (ds.time.dt.year >= start_year) & (ds.time.dt.year <= end_year)
        ds_subset = ds.sel(time=time_mask)
    else:
        ds_subset = ds
    
    # Compute number of valid measurements per station
    swq_valid_count = (~np.isnan(ds_subset.snw)).sum(dim='time').compute()
    snd_valid_count = (~np.isnan(ds_subset.snd)).sum(dim='time').compute()
    
    # Add data availability information to dataframe
    stations_df['swq_valid_count'] = swq_valid_count.values
    stations_df['snd_valid_count'] = snd_valid_count.values
    
    # Calculate data completeness percentage
    total_timesteps = len(ds_subset.time)
    stations_df['swq_completeness'] = (stations_df['swq_valid_count'] / total_timesteps) * 100
    stations_df['snd_completeness'] = (stations_df['snd_valid_count'] / total_timesteps) * 100
    
    # Calculate bounding box coordinates for each station
    # For point simulations, we create a small bounding box around each station
    # Default buffer is 0.1 degrees in each direction
    buffer = 0.1
    stations_df['BOUNDING_BOX_COORDS'] = (
        stations_df['lat'] + buffer
    ).astype(str) + '/' + (
        stations_df['lon'] - buffer
    ).astype(str) + '/' + (
        stations_df['lat'] - buffer
    ).astype(str) + '/' + (
        stations_df['lon'] + buffer
    ).astype(str)
    
    # Format pour point coordinates as lat/lon
    stations_df['POUR_POINT_COORDS'] = stations_df['lat'].astype(str) + '/' + stations_df['lon'].astype(str)
    
    # Create Watershed_Name column based on station_id (cleaned up for file naming)
    # Replace any characters that might cause issues in file paths
    stations_df['Watershed_Name'] = stations_df['station_id'].str.replace('[^a-zA-Z0-9_]', '_', regex=True)
    
    # Save to CSV
    print(f"Saving processed station data to {output_csv_path}...")
    stations_df.to_csv(output_csv_path, index=False)
    
    # Close dataset to free up memory
    ds.close()
    
    return stations_df

def extract_snow_data(norswe_path, station_id, output_dir, start_year=None, end_year=None):
    """
    Extract snow depth and SWE data for a specific station and save to CSV files
    
    Args:
        norswe_path: Path to the NorSWE NetCDF file
        station_id: Station ID to extract data for
        output_dir: Directory to save the extracted data
        start_year: Optional start year for filtering data
        end_year: Optional end year for filtering data
        
    Returns:
        Tuple of paths to saved SWE and snow depth files
    """
    print(f"Extracting snow data for station {station_id}...")
    
    # Open dataset using dask for efficient processing
    ds = xr.open_dataset(norswe_path, chunks={'time': 'auto'})
    
    # Select data for the specified station
    ds_station = ds.sel(station_id=station_id)
    
    # Create a time mask if year range is specified
    if start_year is not None and end_year is not None:
        time_mask = (ds_station.time.dt.year >= start_year) & (ds_station.time.dt.year <= end_year)
        ds_station = ds_station.sel(time=time_mask)
    
    # Convert to pandas DataFrame
    df_snow = ds_station.to_dataframe()
    
    # Reset index to get time as a column
    df_snow = df_snow.reset_index()
    
    # Create output directories
    swe_dir = Path(output_dir) / 'observations' / 'snow' / 'raw_data' / 'swe'
    snd_dir = Path(output_dir) / 'observations' / 'snow' / 'raw_data' / 'depth'
    
    swe_dir.mkdir(parents=True, exist_ok=True)
    snd_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrames for SWE and snow depth
    df_swe = df_snow[['time', 'snw']].copy()
    df_swe.rename(columns={'snw': 'SWE_kg_m2'}, inplace=True)
    
    df_snd = df_snow[['time', 'snd']].copy()
    df_snd.rename(columns={'snd': 'Depth_m'}, inplace=True)
    
    # Save to CSV files
    swe_file = swe_dir / f"{station_id}_swe.csv"
    snd_file = snd_dir / f"{station_id}_depth.csv"
    
    df_swe.to_csv(swe_file, index=False)
    df_snd.to_csv(snd_file, index=False)
    
    # Close dataset to free up memory
    ds.close()
    
    return str(swe_file), str(snd_file)

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
    parser = argparse.ArgumentParser(description='Process NorSWE data and set up CONFLUENCE point simulations')
    parser.add_argument('--norswe_path', type=str, required=True,
                        help='Path to the NorSWE NetCDF file')
    parser.add_argument('--template_config', type=str, required=True,
                        help='Path to the template config file for CONFLUENCE')
    parser.add_argument('--output_dir', type=str, default='norswe_output',
                        help='Directory to store processed data and outputs')
    parser.add_argument('--config_dir', type=str, default='norswe_configs',
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
    parser.add_argument('--base_path', type=str, default='/work/comphyd_lab/data/CONFLUENCE_data/norswe',
                        help='Base path for CONFLUENCE data directory')
    parser.add_argument('--force_reprocess', action='store_true',
                        help='Force reprocessing of station data even if CSV exists')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.config_dir, exist_ok=True)
    
    # Process NorSWE data
    print(f"Processing NorSWE data from {args.norswe_path}...")
    stations_csv = os.path.join(args.output_dir, 'norswe_stations.csv')
    
    stations_df = process_norswe_data(
        args.norswe_path, 
        stations_csv,
        start_year=args.start_year,
        end_year=args.end_year,
        use_existing_csv=not args.force_reprocess  # Use existing CSV unless force_reprocess is specified
    )
    
    # Filter stations based on completeness
    complete_stations = stations_df[
        (stations_df['swq_completeness'] >= args.min_completeness) &
        (stations_df['snd_completeness'] >= args.min_completeness)
    ]
    
    print(f"Found {len(complete_stations)} stations with at least {args.min_completeness}% data completeness")
    
    # Limit to max_stations if specified
    if args.max_stations is not None and len(complete_stations) > args.max_stations:
        print(f"Limiting to {args.max_stations} stations")
        # Prioritize stations with higher completeness
        complete_stations = complete_stations.sort_values(
            by=['swq_completeness', 'snd_completeness'], 
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
        station_name = station['Watershed_Name']
        pour_point = station['POUR_POINT_COORDS']
        bounding_box = station['BOUNDING_BOX_COORDS']
        
        # Create a unique domain name
        domain_name = f"{station_name}"
        
        # Check if the simulations directory already exists
        simulation_dir = Path(f"{args.base_path}/domain_{domain_name}/simulations/run_1/SUMMA/run_1_timestep.nc")
        
        if simulation_dir.exists():
            print(f"Skipping {domain_name} - simulation result already exists: {simulation_dir}")
            skipped_jobs.append(domain_name)
            continue
        
        # Extract snow data for this station
        #station_dir = f"{args.base_path}/domain_{domain_name}"
        #swe_file, snd_file = extract_snow_data(
        #    args.norswe_path,
        #    station_id,
        #    station_dir,
        #    start_year=args.start_year,
        #    end_year=args.end_year
        #)
        
        #print(f"Extracted snow data for {station_id}:")
        #print(f"  SWE data: {swe_file}")
        #print(f"  Snow depth data: {snd_file}")
        
        # Generate the config file path
        config_path = os.path.join(args.config_dir, f"config_{domain_name}.yaml")
        
        # Generate the config file
        print(f"Generating config file for {domain_name}...")
        generate_config_file(args.template_config, config_path, domain_name, pour_point, bounding_box)
        
        # Run CONFLUENCE with the generated config if requested
        if submit_jobs == 'y':
            print(f"Submitting CONFLUENCE job for {domain_name}...")
            job_id = run_confluence(config_path, domain_name)
            
            if job_id:
                submitted_jobs.append((domain_name, job_id))
            
            # Add a small delay between job submissions to avoid overwhelming the scheduler
            time.sleep(5)
    
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
        print(f"Snow data has been extracted to the appropriate directories.")
        print(f"Config files have been generated in {args.config_dir}")
        print(f"\nTo run CONFLUENCE jobs later, you can use the --no_submit flag and then manually submit using:\n")
        print(f"  for config in {args.config_dir}/config_*.yaml; do")
        print(f"    python norswe_confluence_setup.py --norswe_path {args.norswe_path} --template_config {args.template_config} --output_dir {args.output_dir} --config_dir {args.config_dir} --min_completeness {args.min_completeness}")
        if args.max_stations:
            print(f"    --max_stations {args.max_stations}", end="")
        if args.start_year:
            print(f" --start_year {args.start_year}", end="")
        if args.end_year:
            print(f" --end_year {args.end_year}", end="")
        print(f"\n  done")

if __name__ == "__main__":
    main()
