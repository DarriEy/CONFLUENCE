#!/bin/bash
#SBATCH --job-name=CONFLUNCE
#SBATCH --output=CONFLUENCE_single_%j.log
#SBATCH --error=CONFLUENCE_single_%j.err
#SBATCH --time=120:00:00
<<<<<<< HEAD
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=5G

# Load necessary modules (adjust as needed for your HPC environment)
. /work/comphyd_lab/local/modules/spack/2024v4/lmod-init-bash
module unuse $MODULEPATH
module use /work/comphyd_lab/local/modules/spack/2024v4/modules/linux-rocky8-x86_64/Core/

#module restore confluence_modules
module load netcdf-fortran/4.6.1
module load openblas/0.3.27
module load hdf5/1.14.3
module load gdal/3.9.2
module load netlib-lapack/3.11.0
module load openmpi/4.1.6
module load python/3.11.7

# Activate your Python environment if necessary
source /home/darri.eythorsson/code/scienv/bin/activate
=======
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G

# Load the required modules
module restore confluence_modules

# Activate your Python environment if necessary
source /home/darri/code/confluence_env/bin/activate
>>>>>>> 40f653c (working on notebooks and refactoring)

# Run the Python script
python CONFLUENCE.py

echo "CONFLUENCE job complete"