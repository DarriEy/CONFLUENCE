import pandas as pd
import os
import yaml
import subprocess
from pathlib import Path
import time
import sys

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
    
    # Update the domain name
    config_content = config_content.replace('DOMAIN_NAME: "North_America"', f'DOMAIN_NAME: "{domain_name}"')
    
    # Update the pour point coordinates
    config_content = config_content.replace('POUR_POINT_COORDS: 62.09444/136.27223', f'POUR_POINT_COORDS: {pour_point}')
    
    # Update the bounding box coordinates
    config_content = config_content.replace('BOUNDING_BOX_COORDS: 83.0/-169.0/7.0/-52.0', f'BOUNDING_BOX_COORDS: {bounding_box}')
    
    # Save the new config file
    with open(output_path, 'w') as f:
        f.write(config_content)
    
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
#SBATCH --time=120:00:00
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=50G

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
python CONFLUENCE.py --config {config_path}

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
    # Path to the CSV file
    csv_path = "global_watershed.csv"
    
    # Path to the template config file
    template_config_path = "0_config_files/config_North_America.yaml"
    
    # Directory to store generated config files
    config_dir = "0_config_files/watersheds"
    
    # Create the config directory if it doesn't exist
    os.makedirs(config_dir, exist_ok=True)
    
    # Read the watershed data
    watersheds = pd.read_csv(csv_path)
    
    # Process each watershed
    submitted_jobs = []
    
    for _, watershed in watersheds.iterrows():
        # Get watershed parameters
        watershed_id = watershed['ID']
        watershed_name = watershed['Watershed_Name']
        pour_point = watershed['POUR_POINT_COORDS']
        bounding_box = watershed['BOUNDING_BOX_COORDS']
        
        # Create a unique domain name (using ID and name)
        domain_name = f"{watershed_id}_{watershed_name.replace(' ', '_')}"
        
        # Generate the config file path
        config_path = os.path.join(config_dir, f"config_{domain_name}.yaml")
        
        # Generate the config file
        print(f"Generating config file for {domain_name}...")
        generate_config_file(template_config_path, config_path, domain_name, pour_point, bounding_box)
        
        # Run CONFLUENCE with the generated config
        print(f"Submitting CONFLUENCE job for {domain_name}...")
        job_id = run_confluence(config_path, domain_name)
        
        if job_id:
            submitted_jobs.append((domain_name, job_id))
        
        # Add a small delay between job submissions to avoid overwhelming the scheduler
        time.sleep(2)
    
    # Print summary of submitted jobs
    print("\nSubmitted jobs summary:")
    for domain_name, job_id in submitted_jobs:
        print(f"Domain: {domain_name}, Job ID: {job_id}")
    
    print(f"\nTotal jobs submitted: {len(submitted_jobs)}")

if __name__ == "__main__":
    main()