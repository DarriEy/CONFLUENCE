#!/bin/bash
#SBATCH --output=CONFLUENCE_single_%j.log
#SBATCH --error=CONFLUENCE_single_%j.err
#SBATCH --time=120:00:00
#SBATCH --ntasks=20
#SBATCH --mem-per-cpu=5G

# Load necessary modules (adjust as needed for your HPC environment)
module restore confluence_modules

# Activate your Python environment if necessary
source /home/darri/code/confluence_env/bin/activate

# Your commands here
python CONFLUENCE.py