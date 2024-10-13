#!/bin/bash

set -e

# Set the Python executable path
PYTHON_EXECUTABLE="/Users/darrieythorsson/compHydro/code/CWARHM/scienv/bin/python"

# Set the path to run_trial.py
RUN_TRIAL_PY="/Users/darrieythorsson/compHydro/data/CONFLUENCE_data/installs/ostrich/run_trial.py"

# Extract rank from the current directory name
CURRENT_DIR=$(basename "$(pwd)")
RANK=${CURRENT_DIR#model_run}

echo "Running trial for rank $RANK"

# Set PYTHONPATH to include the directory containing utils
export PYTHONPATH="/Users/darrieythorsson/compHydro/code/CWARHM:$PYTHONPATH"

# Execute Python script
$PYTHON_EXECUTABLE $RUN_TRIAL_PY $RANK 2>&1 | tee run_trial_output.log

# Check if the objective file was created
if [ ! -f "ostrich_objective.txt" ]; then
    echo "Error: ostrich_objective.txt file not created"
    exit 1
fi

echo "Trial completed for rank $RANK"
