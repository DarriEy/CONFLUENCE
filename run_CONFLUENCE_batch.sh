#!/bin/bash
#SBATCH --job-name=North_America
#SBATCH --output=CONFLUENCE_single_%j.log
#SBATCH --error=CONFLUENCE_single_%j.err
#SBATCH --time=120:00:00
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=10G

# Load necessary modules (adjust as needed for your HPC environment)
. /work/comphyd_lab/local/modules/spack/2024v5/lmod-init-bash
module unuse $MODULEPATH
module use /work/comphyd_lab/local/modules/spack/2024v5/modules/linux-rocky8-x86_64/Core/

#module restore confluence_modules
module load netcdf-fortran/4.6.1
module load openblas/0.3.27
module load hdf/4.3.0
module load hdf5/1.14.3
module load gdal/3.9.2
module load netlib-lapack/3.11.0
module load openmpi/4.1.6
module load python/3.11.7

# Activate your Python environment if necessary
source /work/comphyd_lab/users/darri/data/CONFLUENCE_data/installs/conf-env/bin/activate

# Run the Python script
python CONFLUENCE.py --config 0_config_files/config_North_America.yaml 

echo "CONFLUENCE job complete"
