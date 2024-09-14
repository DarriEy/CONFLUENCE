import os
import subprocess
from typing import List, Tuple, Dict, Any
import numpy as np # type: ignore
import netCDF4 as nc # type: ignore
import xarray as xr # type: ignore
import sys
from pathlib import Path
import shutil
import datetime
import time
import pandas as pd # type: ignore
from dateutil.relativedelta import relativedelta # type: ignore
import traceback

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.calibration_utils import read_param_bounds # type: ignore

class OstrichOptimizer:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.rank = int(os.environ.get('PMI_RANK', 0))  # Get rank from environment variable
        self.logger = logger
        self.num_ranks = self.config.get('MPI_PROCESSES')  # Get total number of ranks
        self.root_path = Path(self.config.get('CONFLUENCE_DATA_DIR')) 
        self.project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"
        self.logger.info(f"Initialized OstrichOptimizer for rank {self.rank} out of {self.num_ranks} ranks")

    def run_optimization(self):
        self.create_rank_specific_directories()
        self.prepare_initial_parameters()
        self.create_ostrich_config()
        self.run_ostrich()
        
        results = self.parse_ostrich_results()
        if results:
            if len(self.config.optimization_metrics) == 1:
                self.logger.info("Optimization completed. Best solution:")
                self.logger.info(f"Objective value: {results['objective']}")
                self.logger.info("Parameters:")
                all_params = self.config.params_to_calibrate + self.config.basin_params_to_calibrate
                for param, value in zip(all_params, results['parameters']):
                    self.logger.info(f"  {param}: {value}")
            else:
                self.logger.info("Optimization completed. Pareto front:")
                for i, solution in enumerate(results):
                    self.logger.info(f"Solution {i+1}:")
                    for j, metric in enumerate(self.config.optimization_metrics):
                        self.logger.info(f"  {metric}: {solution['objectives'][j]}")
                    self.logger.info("  Parameters:")
                    all_params = self.config.params_to_calibrate + self.config.basin_params_to_calibrate
                    for param, value in zip(all_params, solution['parameters']):
                        self.logger.info(f"    {param}: {value}")
        else:
            self.logger.error("Failed to parse Ostrich results.")
        
    def create_rank_specific_directories(self):
        for rank in range(1, self.num_ranks):
            for model in ["SUMMA", "mizuRoute"]:
                rank_specific_path = self.project_dir / f'simulations/{self.config.get('EXPERIMENT_ID')}_rank{rank}/{model}'
                rank_specific_path.mkdir(parents=True, exist_ok=True)
                run_settings_path = rank_specific_path / "run_settings"
                run_settings_path.mkdir(exist_ok=True)
    
                # Ensure the directory has the correct permissions
                os.chmod(run_settings_path, 0o755)
    
                settings_summa_path = self.project_dir / 'settings/SUMMA'
                settings_mizu_path = self.project_dir / 'settings/mizuroute'

                if model == "SUMMA":
                    self.copy_summa_settings(settings_summa_path, run_settings_path, rank)
                elif model == "mizuRoute":
                    self.copy_mizuroute_settings(settings_mizu_path, run_settings_path, rank)

        self.logger.info(f"Created and populated rank-specific directories for all {self.num_ranks} ranks")

    def create_initial_multipliers(self):
        for rank in range(1, self.config.num_ranks):  # Start from 1 to exclude rank 0
            initial_file = self.config.ostrich_path / f'multipliers_rank{rank}.txt'
            with open(initial_file, 'w') as f:
                for _ in self.config.params_to_calibrate:
                    f.write("1.0\n")  # Start with neutral multipliers
            self.logger.info(f"Initial multipliers file created for rank {rank}: {initial_file}")

    def prepare_initial_parameters(self):
        self.logger.info("Preparing initial parameters for all ranks")

        for rank in range(1, self.num_ranks):
            self.logger.info(f"Preparing initial parameters for rank {rank}")
            self.generate_multiplier_bounds(rank)
            self.create_multiplier_template()
        
        #self.create_initial_multipliers()

    def create_multiplier_template(self):
        template_file = Path(self.config.get("OSTRICH_PATH")) / 'multipliers.tpl'
        all_params = self.config.get('PARAMS_TO_CALIBRATE').split(',') + self.config.get('BASIN_PARAMS_TO_CALIBRATE').split(',')
        with open(template_file, 'w') as f:
            for param in all_params:
                f.write(f"{param}_multp\n")
        self.logger.info(f"Multiplier template file created: {template_file}")

    def generate_run_trial_script(self):
        run_trial_content = """#!/bin/bash

set -e

# Set the Python executable path
PYTHON_EXECUTABLE="{}"

# Set the path to run_trial.py
RUN_TRIAL_PY="{}"

# Extract rank from the current directory name
CURRENT_DIR=$(basename "$(pwd)")
RANK=${{CURRENT_DIR#model_run}}

echo "Running trial for rank $RANK"

# Set PYTHONPATH to include the directory containing utils
export PYTHONPATH="{}:$PYTHONPATH"

# Execute Python script
$PYTHON_EXECUTABLE $RUN_TRIAL_PY $RANK 2>&1 | tee run_trial_output.log

# Check if the objective file was created
if [ ! -f "ostrich_objective.txt" ]; then
    echo "Error: ostrich_objective.txt file not created"
    exit 1
fi

echo "Trial completed for rank $RANK"
""".format(
    sys.executable,
    os.path.join(self.config.ostrich_path, 'run_trial.py'),
    "/Users/darrieythorsson/compHydro/code/CWARHM"
)

        run_trial_path = self.config.ostrich_path / 'run_trial.sh'
        with open(run_trial_path, 'w') as f:
            f.write(run_trial_content)
        
        # Make the script executable
        os.chmod(run_trial_path, 0o755)

        self.logger.info(f"Generated run_trial.sh script: {run_trial_path}")

    def create_ostrich_config(self):
        ostrich_config = Path(self.config.get('OSTRICH_PATH')) / 'ostIn.txt'

        self.logger.info(f"Creating Ostrich configuration file: {ostrich_config}")
        template_content = f"""ProgramType {self.config.get('OSTRICH_ALGORITHM')}

ModelSubdir model_run

BeginFilePairs
multipliers.tpl ; multipliers.txt
EndFilePairs

BeginExtraFiles
run_trial.sh
run_trial.py
EndExtraFiles

BeginParams
{self.generate_param_definitions()}
EndParams

BeginObservations
{self.generate_observations()}
EndObservations

BeginResponseVars
{self.generate_response_vars()}
EndResponseVars

BeginGCOP
CostFunction GCOP
{self.generate_obj_function()}
EndGCOP

BeginConstraints
EndConstraints

ModelExecutable ./run_trial.sh

BeginParallelStuff
ParallelType   MPI
NumNodes       {self.num_ranks}
EndParallelStuff

{self.generate_algorithm_config()}
"""
        with open(ostrich_config, 'w') as f:
            f.write(template_content)
        
        self.logger.info(f"Ostrich configuration file created: {ostrich_config}")

    def generate_observations(self):
        if len(self.config.get('MOO_OPTIMIZATION_METRICS')) == 1:
            return f"Obj 1 1 ostrich_objective.txt; OST_NULL 0 1"
        else:
            return "\n".join([f"Obj{i+1} 1 1 ostrich_objective.txt; OST_NULL 0 1" for i in range(len(self.config.get('MOO_OPTIMIZATION_METRICS')))])

    def generate_response_vars(self):
        if len(self.config.get('MOO_OPTIMIZATION_METRICS')) == 1:
            return f"Obj ostrich_objective.txt; OST_NULL 0 1"
        else:
            return "\n".join([f"Obj{i+1} ostrich_objective.txt; OST_NULL 0 1" for i in range(len(self.config.get('MOO_OPTIMIZATION_METRICS')))])
        
    def generate_obj_function(self):
        return "\n".join([f"ObjFunc Obj{i+1}" for i in range(len(self.config.get('MOO_OPTIMIZATION_METRICS')))])

    def generate_extra_dirs(self):
        extra_dirs = ""
        for rank in range(1, self.config.num_ranks):
            rank_dir = f"{self.config.root_path}/domain_{self.config.domain_name}/simulations/{self.config.experiment_id}_rank{rank}"
            extra_dirs += f"{rank_dir}\n"
        return extra_dirs

    def generate_param_definitions(self):
        definitions = []
        all_params = self.config.get('PARAMS_TO_CALIBRATE').split(',') + self.config.get('BASIN_PARAMS_TO_CALIBRATE').split(',')
        local_parameters_file = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f'domain_{self.config.get('DOMAIN_NAME')}' / 'settings/summa/localParamInfo.txt'
        basin_parameters_file = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f'domain_{self.config.get('DOMAIN_NAME')}' / 'settings/summa/basinParamInfo.txt'
        params_to_calibrate = self.config.get('PARAMS_TO_CALIBRATE').split(',')
        basin_params_to_calibrate = self.config.get('BASIN_PARAMS_TO_CALIBRATE').split(',')


        local_bounds_dict = read_param_bounds(local_parameters_file, params_to_calibrate)
        basin_bounds_dict = read_param_bounds(basin_parameters_file, basin_params_to_calibrate)
        local_bounds = [local_bounds_dict[param] for param in params_to_calibrate]
        basin_bounds = [basin_bounds_dict[param] for param in basin_params_to_calibrate]
        all_bounds = local_bounds + basin_bounds

        for param, (lower, upper) in zip(all_params, all_bounds):
            definitions.append(f"{param}_multp\t1.0\t{lower}\t{upper}\tnone\tnone\tnone\tfree")
        return "\n".join(definitions)

    def generate_algorithm_config(self):
        algorithm = self.config.get('OSTRICH_ALGORITHM')
        config = ""

        if algorithm == "DDS":
            config = f"""
BeginDDSAlg
PerturbationValue   {self.config.get('DDS_R')}
MaxIterations   {self.config.get('NUMBER_OF_ITERATIONS')}
UseRandomParamValues
UseOpt  Standard
EnableDebugging
EndDDSAlg
"""         
        elif algorithm == "ParallelDDS":
            config = f"""
BeginParallelDDSAlg
PerturbationValue {self.config.get('DDS_R')}
MaxIterations {self.config.num_iter}
UseRandomParamValues

EndParallelDDSAlg
"""         
        elif algorithm == "APPSO":
            config = f"""
BeginAPPSO
SwarmSize   {self.config.pso_swarm_size}
NumGenerations  {self.config.pso_num_generations} 
ConstrictionFactor  {self.config.pso_constriction_factor}
CognitiveParam  {self.config.pso_cognitive_param} 
SocialParam {self.config.pso_social_param}
InertiaWeight   {self.config.pso_inertia_weight}
InertiaReductionRate    {self.config.pso_inertia_reduction_rate}
ConvergenceVal  {self.config.pso_convergence_val}
EndAPPSO
"""        
        elif algorithm == "SCE":
            config = f"""
BeginSCEUA
Budget {self.config.sce_budget}
LoopStagnationCriteria {self.config.sce_loop_stagnation}
PctChangeCriteria {self.config.sce_pct_change}
PopConvCriteria {self.config.sce_pop_conv}
NumComplexes {self.config.sce_num_complexes}
NumPointsPerComplex {self.config.sce_points_per_complex}
NumPointsPerSubComplex {self.config.sce_points_per_subcomplex}
NumEvolutionSteps {self.config.sce_num_evolution_steps}
MinNumberOfComplexes {self.config.sce_min_complexes}
UseInitialPoint {self.config.sce_use_initial_point}
EndSCEUA
"""
        elif algorithm == "PSO":
            config = f"""
BeginParticleSwarm
SwarmSize {self.config.pso_swarm_size}
NumGenerations {self.config.pso_num_generations}
ConstrictionFactor {self.config.pso_constriction_factor}
CognitiveParam {self.config.pso_cognitive_param}
SocialParam {self.config.pso_social_param}
InertiaWeight {self.config.pso_inertia_weight}
InertiaReductionRate {self.config.pso_inertia_reduction_rate}
InitPopulationMethod {self.config.pso_init_population_method}
ConvergenceVal {self.config.pso_convergence_val}
EndParticleSwarm
"""
        elif algorithm == "ParaPADDS":
            config = f"""
BeginParaPADDS
PerturbationValue  {self.config.get('DDS_R')}
PopulationSize     {self.config.get('POPULATION_SIZE')}
NumGenerations     {self.config.get('PSO_NUM_GENERATIONS')}
ArchiveSize        100
UpdateInterval     10
MaxIterations      1000
EndParaPADDS
"""
        # Add more algorithms as needed

        return config

    def run_ostrich(self):
        ostrich_path = Path(self.config.get('OSTRICH_PATH'))
        ostrich_exe = ostrich_path / self.config.get('OSTRICH_EXE')

        ostrich_command = f"mpiexec -n {self.num_ranks} --verbose {ostrich_exe}"
        self.logger.info(f"Running Ostrich: {ostrich_command}")
        
        result = subprocess.run(ostrich_command, shell=True, check=True, capture_output=True, text=True, cwd=ostrich_path)
        
        self.logger.info(f"Ostrich run completed with return code: {result.returncode}")
        if result.stdout:
            self.logger.debug(f"Ostrich stdout: {result.stdout}")
        if result.stderr:
            self.logger.warning(f"Ostrich stderr: {result.stderr}")

        return result.returncode == 0

    def parse_ostrich_results(self):
        ostrich_output = Path(self.config.ostrich_path) / 'OstOutput0.txt'
        
        with open(ostrich_output, 'r') as f:
            lines = f.readlines()
        
        if len(self.config.optimization_metrics) == 1:
            # Single-objective case
            best_line = None
            for line in reversed(lines):
                if line.strip().startswith('0'):
                    best_line = line
                    break
            
            if best_line is None:
                self.logger.error("Could not find best parameter set in Ostrich output")
                return None

            parts = best_line.split()
            best_value = float(parts[1])
            best_params = [float(p) for p in parts[2:]]
            
            self.logger.info(f"Best objective value: {best_value}")
            self.logger.info(f"Best parameters: {best_params}")
            
            return {'objective': best_value, 'parameters': best_params}
        else:
            # Multi-objective case
            pareto_front = []
            pareto_start = False
            for line in lines:
                if line.strip().startswith('Pareto Front'):
                    pareto_start = True
                    continue
                if pareto_start and line.strip():
                    parts = line.split()
                    if len(parts) == len(self.config.optimization_metrics) + len(self.config.params_to_calibrate) + len(self.config.basin_params_to_calibrate) + 1:
                        pareto_front.append({
                            'objectives': [float(x) for x in parts[1:len(self.config.optimization_metrics)+1]],
                            'parameters': [float(x) for x in parts[len(self.config.optimization_metrics)+1:]]
                        })
                    else:
                        break
            
            self.logger.info(f"Parsed Pareto front with {len(pareto_front)} solutions")
            return pareto_front

    def get_rank_specific_path(self, base_path: Path, experiment_id: str, model: str, rank: int) -> Path:
        # Remove any existing rank suffix from the experiment_id
        experiment_id = experiment_id.split('_rank')[0]
        return base_path / "simulations" / f"{experiment_id}_rank{rank}" / model

    def get_summa_settings_path(self, rank):
        # Update this method to return rank-specific path
        rank_specific_path = self.get_rank_specific_path(self.project_dir, self.config.get('EXPERIMENT_ID'), "SUMMA", rank)
        return rank_specific_path / "run_settings"

    def get_mizuroute_settings_path(self) -> Path:
        rank_specific_path = self.get_rank_specific_path(Path(self.config.root_path) / f"domain_{self.config.domain_name}", 
                                                        self.config.experiment_id, "mizuRoute")
        return rank_specific_path / "run_settings"

    def copy_summa_settings(self, source_path, destination_path, rank):
        files_to_copy = [
            'outputControl.txt', self.config.get('SETTINGS_SUMMA_FILEMANAGER'), 'attributes.nc', 
            'trialParams.nc', 'forcingFileList.txt', 'modelDecisions.txt', 
            'localParamInfo.txt', 'basinParamInfo.txt', 'coldState.nc', 
            'TBL_GENPARM.TBL', 'TBL_MPTABLE.TBL', 'TBL_SOILPARM.TBL', 'TBL_VEGPARM.TBL'
        ]
        for file in files_to_copy:
            shutil.copy2(source_path / file, destination_path / file)
        
        # Update file manager for the specific rank
        self.update_summa_file_manager(destination_path / self.config.get('SETTINGS_SUMMA_FILEMANAGER'), rank)

    def copy_mizuroute_settings(self, source_path, destination_path, rank):
        files_to_copy = [self.config.get('SETTINGS_MIZU_CONTROL_FILE'), 'param.nml.default', 'topology.nc']
        for file in files_to_copy:
            shutil.copy2(source_path / file, destination_path / file)
        
        # Update control file for the specific rank
        self.update_mizu_control_file(destination_path / self.config.get('SETTINGS_MIZU_CONTROL_FILE'), rank)

    def update_summa_file_manager(self, file_path, rank):
        with open(file_path, 'r') as f:
            content = f.read()
        
        content = content.replace(self.config.get('EXPERIMENT_ID'), f'{self.config.get('EXPERIMENT_ID')}_rank{rank}')
        content = content.replace('/settings/SUMMA/', f'/simulations/{self.config.get('EXPERIMENT_ID')}_rank{rank}/SUMMA/run_settings/')
        
        with open(file_path, 'w') as f:
            f.write(content)

    def update_mizu_control_file(self, file_path, rank):
        with open(file_path, 'r') as f:
            content = f.read()
        
        content = content.replace(self.config.get('EXPERIMENT_ID'), f'{self.config.get('EXPERIMENT_ID')}_rank{rank}')
        content = content.replace('/settings/mizuRoute/', f'/simulations/{self.config.get('EXPERIMENT_ID')}_rank{rank}/mizuRoute/run_settings/')
        
        with open(file_path, 'w') as f:
            f.write(content)

    def generate_priori_trial_params(self, rank):
        self.logger.info(f"Generating a priori trial parameters for rank {rank}")
        
        summa_settings_path = self.get_summa_settings_path(rank)

        # Update outputControl.txt
        self.update_output_control(summa_settings_path)
        
        # Update fileManager.txt
        self.update_file_manager(summa_settings_path, rank)
        
        # Run SUMMA to generate a priori parameter values
        self.run_summa_for_priori(rank)
        
        # Extract a priori parameter values and create trialParam.priori.nc
        self.create_priori_trial_param_file(summa_settings_path, rank)
        
        self.reset_file_manager(summa_settings_path, rank)
        
        self.logger.info(f"A priori trial parameters generated successfully for rank {rank}")

    def update_output_control(self, settings_path):
        output_control_file = settings_path / self.config.get('SETTINGS_SUMMA_OUTPUT')
        output_control_temp = output_control_file.with_suffix('.temp')
        
        output_params = self.config.get('PARAMS_TO_CALIBRATE').split(',').copy()
        
        # Add additional parameters if needed (e.g., soil and canopy height parameters)
        soil_params = ['theta_res', 'critSoilWilting', 'critSoilTranspire', 'fieldCapacity', 'theta_sat', 'vGn_alpha', 'vGn_n', 'k_soil', 'k_macropore']
        height_params = ['heightCanopyBottom', 'heightCanopyTop']
        basin_params = ['basin__aquiferHydCond','routingGammaScale','routingGammaShape','basin__aquiferBaseflowExp','basin__aquiferScaleFactor']
        
        output_params.extend([param for param in soil_params + height_params + basin_params if param not in output_params])
        
        with open(output_control_file, 'r') as src, open(output_control_temp, 'w') as dst:
            for param in output_params:
                dst.write(f"{param}\n")
            dst.write(src.read())
        
        shutil.move(output_control_temp, output_control_file)

    def update_file_manager(self, settings_path, rank):
        file_manager = settings_path / self.config.get('SETTINGS_SUMMA_FILEMANAGER')
        file_manager_temp = file_manager.with_suffix('.temp')
        start, end = [datetime.datetime.strptime(date.strip(), '%Y-%m-%d') for date in self.config.get('CALIBRATION_PERIOD').split(',')]
        sim_start_time = start
        sim_end_time = (sim_start_time + datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M')
        
        with open(file_manager, 'r') as src, open(file_manager_temp, 'w') as dst:
            for line in src:
                if line.startswith('simStartTime'):
                    line = f"simStartTime '{sim_start_time}'\n"
                elif line.startswith('simEndTime'):
                    line = f"simEndTime '{sim_end_time}'\n"
                dst.write(line)
        
        shutil.move(file_manager_temp, file_manager)

    def reset_file_manager(self, settings_path, rank):
        file_manager = settings_path / self.config.get('SETTINGS_SUMMA_FILEMANAGER')
        file_manager_temp = file_manager.with_suffix('.temp')

        start, end = [datetime.datetime.strptime(date.strip(), '%Y-%m-%d') for date in self.config.get('CALIBRATION_PERIOD').split(',')]
        sim_start_time = (start - relativedelta(years=1))
        sim_end_time = end + datetime.timedelta(hours=23)
        
        with open(file_manager, 'r') as src, open(file_manager_temp, 'w') as dst:
            for line in src:
                if line.startswith('simStartTime'):
                    line = f"simStartTime '{sim_start_time}'\n"
                elif line.startswith('simEndTime'):
                    line = f"simEndTime '{sim_end_time}'\n"
                dst.write(line)
        
        shutil.move(file_manager_temp, file_manager)

    def run_summa_for_priori(self, rank):
        summa_exe = self.root_path / 'installs' / 'summa' / 'bin' / self.config.get('SUMMA_EXE')
        file_manager = self.get_summa_settings_path(rank) / self.config.get('SETTINGS_SUMMA_FILEMANAGER')
        
        # Create SUMMA_logs directory in the rank-specific path
        log_path = self.get_rank_specific_path(
            self.root_path / f"domain_{self.config.get('DOMAIN_NAME')}",
            self.config.get('EXPERIMENT_ID'),
            "SUMMA",
            rank
        ) / "SUMMA_logs"
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Create a unique log file name for this run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"summa_priori_rank{rank}_{timestamp}.log"
        
        # Construct SUMMA command
        summa_command = [str(summa_exe), "-m", str(file_manager)]
        
        self.logger.info(f"Running SUMMA for priori with command: {' '.join(summa_command)}")
        self.logger.info(f"SUMMA log will be written to: {log_path / log_name}")
        
        try:
            # Run SUMMA and capture output in the log file
            with open(log_path / log_name, 'w') as log_file:
                result = subprocess.run(summa_command, check=True, stdout=log_file, stderr=subprocess.STDOUT, text=True)
            
            self.logger.info(f"SUMMA run completed successfully for rank {rank}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"SUMMA run failed for rank {rank} with return code: {e.returncode}")
            self.logger.error(f"Please check the log file for details: {log_path / log_name}")
            return False
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while running SUMMA for rank {rank}: {str(e)}")
            return False

    def create_priori_trial_param_file(self, settings_path, rank):
        self.logger.info(f"Creating a priori trial parameter file for rank {rank}")
        
        rank_experiment_id = f"{self.config.get('EXPERIMENT_ID')}_rank{rank}"
        output_path = self.get_rank_specific_path(
            Path(self.root_path) / f"domain_{self.config.get('DOMAIN_NAME')}",
            self.config.get('EXPERIMENT_ID'),
            "SUMMA",
            rank
        )
        output_prefix = rank_experiment_id
        output_file = output_path / f"{output_prefix}_timestep.nc"
        
        attribute_file = settings_path / self.config.get('SETTINGS_SUMMA_ATTRIBUTES')
        trial_param_file = settings_path / self.config.get('SETTINGS_SUMMA_TRIALPARAMS')
        trial_param_file_priori = trial_param_file.with_name(f"{trial_param_file.stem}.priori.nc")

        if not output_file.exists():
            self.logger.warning(f"Output file not found: {output_file}. Skipping priori parameter extraction.")
            return

        try:
            # Create an empty priori file if it doesn't exist
            if not trial_param_file_priori.exists():
                self.logger.info(f"Creating empty priori file: {trial_param_file_priori}")
                with nc.Dataset(attribute_file, 'r') as src, nc.Dataset(trial_param_file_priori, 'w') as dst:
                    for name, dimension in src.dimensions.items():
                        dst.createDimension(name, len(dimension) if not dimension.isunlimited() else None)
                    for name in ['gruId', 'hruId']:
                        x = dst.createVariable(name, src[name].datatype, src[name].dimensions)
                        dst[name].setncatts(src[name].__dict__)
                        dst[name][:] = src[name][:]

            # Process data
            all_params = self.config.get('PARAMS_TO_CALIBRATE').split(',') + self.config.get('BASIN_PARAMS_TO_CALIBRATE').split(',')
            with nc.Dataset(output_file, 'r') as ff, nc.Dataset(attribute_file, 'r') as src, nc.Dataset(trial_param_file_priori, 'r+') as dst:
                for param_name in all_params:
                    if param_name in ff.variables:
                        param_dims = ff[param_name].dimensions
                        param_dim = 'hru' if 'hru' in param_dims else 'gru' if 'gru' in param_dims else None
                        if param_dim is None:
                            self.logger.warning(f"Unexpected dimensions for parameter {param_name}: {param_dims}")
                            continue
                        
                        if param_name not in dst.variables:
                            dst.createVariable(param_name, 'f8', param_dim, fill_value=np.nan)
                        
                        if param_dims == ('depth', 'hru'):
                            dst[param_name][:] = ff[param_name][0, :]  # Use first depth value
                        else:
                            dst[param_name][:] = ff[param_name][:]
                    else:
                        self.logger.warning(f"Parameter {param_name} not found in SUMMA output")

            # Copy priori file to trial param file
            shutil.copy2(trial_param_file_priori, trial_param_file)
            self.logger.info(f"Copied priori file to trial param file: {trial_param_file}")

            # Verify the copied file
            with nc.Dataset(trial_param_file, 'r') as verify:
                self.logger.info(f"Successfully verified trial param file: {trial_param_file}")

        except Exception as e:
            self.logger.error(f"Error creating priori trial parameter file: {e}")
            raise
        finally:
            # Ensure all files are closed
            for file in [output_file, attribute_file, trial_param_file_priori, trial_param_file]:
                try:
                    nc.Dataset(file).close()
                except:
                    pass

        self.logger.info(f"A priori trial parameter file created successfully for rank {rank}")

    def generate_multiplier_bounds(self, rank):
        self.generate_priori_trial_params(rank)
        self.logger.info("Generating multiplier bounds")
        
        summa_settings_path = self.get_summa_settings_path(rank)
        
        # Combine local and basin parameters
        all_params = self.config.get('PARAMS_TO_CALIBRATE').split(',') + self.config.get('BASIN_PARAMS_TO_CALIBRATE').split(',')
        self.logger.info(f"Parameters to calibrate: {all_params}")
        
        # Read parameter files
        basin_param_file = summa_settings_path / self.config.get('SETTINGS_SUMMA_BASIN_PARAMS_FILE')
        local_param_file = summa_settings_path / self.config.get('SETTINGS_SUMMA_LOCAL_PARAMS_FILE')
        
        basin_params = self.read_param_file(basin_param_file)
        local_params = self.read_param_file(local_param_file)
        
        # Read a priori parameter values
        trial_param_file = summa_settings_path / self.config.get('SETTINGS_SUMMA_TRIALPARAMS') 
        self.logger.info(f"Reading trial parameter file: {trial_param_file}")
        with xr.open_dataset(trial_param_file) as ds:
            self.logger.info(f"Variables in trial parameter file: {list(ds.variables)}")
            a_priori_params = {}
            for param in all_params:
                if param in ds.variables:
                    a_priori_params[param] = ds[param].values
                else:
                    self.logger.warning(f"Parameter {param} not found in trial parameter file")
        
        # Calculate multiplier bounds
        multp_bounds_list = []
        for param in all_params:
            if param not in a_priori_params:
                self.logger.error(f"No a priori value found for parameter {param}")
                continue
            
            if param in local_params:
                bounds = self.calculate_local_param_bounds(param, local_params[param], a_priori_params[param])
            elif param in basin_params:
                bounds = self.calculate_basin_param_bounds(param, basin_params[param], a_priori_params[param])
            else:
                self.logger.error(f"Parameter {param} not found in local or basin parameters")
                continue
            
            multp_bounds_list.append([f"{param}_multp"] + bounds)
        
        # Save multiplier bounds
        self.logger.info(f'Saving multiplier bounds for {multp_bounds_list}')
        self.save_multiplier_bounds(multp_bounds_list)
        
        self.logger.info("Multiplier bounds generated successfully")
        self.logger.info(f"Generated bounds for {len(multp_bounds_list)} parameters")

    def read_param_file(self, file_path):
        params = {}
        with open(file_path, 'r') as f:
            for line in f:
                if not line.startswith(('!', "'")):
                    parts = line.strip().split('|')
                    if len(parts) >= 4:
                        name = parts[0].strip()
                        min_val = float(parts[2].strip().replace('d', 'e'))
                        max_val = float(parts[3].strip().replace('d', 'e'))
                        params[name] = (min_val, max_val)
        return params

    def calculate_local_param_bounds(self, param, bounds, a_priori):
        min_val, max_val = bounds
        multp_min = np.max(min_val / a_priori)
        multp_max = np.min(max_val / a_priori)
        multp_initial = 1.0 if multp_min < 1 < multp_max else np.mean([multp_min, multp_max])
        return [multp_initial, multp_min, multp_max]

    def calculate_basin_param_bounds(self, param, bounds, a_priori):
        return self.calculate_local_param_bounds(param, bounds, a_priori)

    def save_multiplier_bounds(self, multp_bounds_list):
        output_file = Path(self.config.get('OSTRICH_PATH')) / 'multiplier_bounds.txt'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write('# MultiplierName,InitialValue,LowerLimit,UpperLimit\n')
            for bounds in multp_bounds_list:
                f.write(f"{bounds[0]},{bounds[1]:.6f},{bounds[2]:.6f},{bounds[3]:.6f}\n")

def main():
    #control_file_path = Path('/Users/darrieythorsson/compHydro/code/CWARHM/0_control_files/control_active.txt')
    #config = Config.from_control_file(control_file_path)
    #optimizer = OstrichOptimizer(config)
    
   #optimizer.run_optimization()
    pass

if __name__ == "__main__":
    main()
