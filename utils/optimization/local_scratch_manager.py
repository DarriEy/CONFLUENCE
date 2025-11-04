#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CONFLUENCE Local Scratch Manager

This module provides functionality to use local scratch storage on HPC systems
to reduce IOPS on shared filesystems during parallel optimization.

The manager:
1. Detects HPC environment (SLURM)
2. Creates scratch directory structure mirroring CONFLUENCE_DATA_DIR
3. Copies necessary data to local scratch
4. Updates paths for optimization run
5. Stages results back to permanent storage

Usage:
    scratch_mgr = LocalScratchManager(config, logger, project_dir, algorithm_name)
    
    if scratch_mgr.use_scratch:
        # Setup scratch space
        scratch_mgr.setup_scratch_space()
        
        # Get scratch paths for optimization
        scratch_paths = scratch_mgr.get_scratch_paths()
        
        # After optimization completes
        scratch_mgr.stage_results_back()
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging


class LocalScratchManager:
    """
    Manages local scratch space for optimization on HPC systems.
    
    This class handles the complete lifecycle of using local scratch:
    - Detection of HPC/SLURM environment
    - Directory structure creation
    - Data copying to scratch
    - Path management
    - Results staging back to permanent storage
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger,
                 project_dir: Path, algorithm_name: str):
        """
        Initialize the local scratch manager.
        
        Args:
            config: CONFLUENCE configuration dictionary
            logger: Logger instance
            project_dir: Main project directory path
            algorithm_name: Name of optimization algorithm (for directory naming)
        """
        self.config = config
        self.logger = logger
        self.project_dir = project_dir
        self.algorithm_name = algorithm_name
        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')
        
        # Check if local scratch should be used
        self.use_scratch = self._should_use_scratch()
        
        # Initialize paths (will be set during setup)
        self.scratch_root = None
        self.scratch_data_dir = None
        self.scratch_project_dir = None
        self.original_data_dir = None
        
        if self.use_scratch:
            self.logger.info("Local scratch mode ENABLED for optimization")
            self._initialize_scratch_paths()
        else:
            self.logger.info("Local scratch mode DISABLED - using standard filesystem")
    
    def _should_use_scratch(self) -> bool:
        """
        Determine if local scratch should be used.
        
        Returns:
            True if scratch should be used, False otherwise
        """
        # Check config flag
        use_scratch_config = self.config.get('USE_LOCAL_SCRATCH', False)
        
        if not use_scratch_config:
            return False
        
        # Check if we're in a SLURM environment
        slurm_tmpdir = os.environ.get('SLURM_TMPDIR')
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        
        if not slurm_tmpdir:
            self.logger.warning(
                "USE_LOCAL_SCRATCH is True but SLURM_TMPDIR not found. "
                "Falling back to standard filesystem."
            )
            return False
        
        if not Path(slurm_tmpdir).exists():
            self.logger.warning(
                f"USE_LOCAL_SCRATCH is True but SLURM_TMPDIR ({slurm_tmpdir}) "
                "does not exist. Falling back to standard filesystem."
            )
            return False
        
        self.logger.info(f"SLURM environment detected: TMPDIR={slurm_tmpdir}, JOB_ID={slurm_job_id}")
        return True
    
    def _initialize_scratch_paths(self) -> None:
        """Initialize scratch directory paths."""
        self.scratch_root = Path(os.environ.get('SLURM_TMPDIR'))
        
        # Create scratch data directory with same structure as CONFLUENCE_DATA_DIR
        self.scratch_data_dir = self.scratch_root / "conf_data"
        self.scratch_project_dir = self.scratch_data_dir / f"domain_{self.domain_name}"
        
        # Store original for reference
        self.original_data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        
        self.logger.info(f"Scratch root: {self.scratch_root}")
        self.logger.info(f"Scratch data dir: {self.scratch_data_dir}")
        self.logger.info(f"Scratch project dir: {self.scratch_project_dir}")
    
    def setup_scratch_space(self) -> bool:
        """
        Setup complete scratch space with all necessary data.
        
        This method:
        1. Creates directory structure
        2. Copies settings files
        3. Copies forcing data
        4. Copies observation data
        5. Updates file paths in configuration files
        
        Returns:
            True if setup successful, False otherwise
        """
        if not self.use_scratch:
            return True
        
        try:
            self.logger.info("="*60)
            self.logger.info("Setting up local scratch space for optimization")
            self.logger.info("="*60)
            
            # Create directory structure
            self._create_scratch_directories()
            
            # Copy data to scratch
            self._copy_settings_to_scratch()
            self._copy_forcing_to_scratch()
            self._copy_observations_to_scratch()
            self._copy_mizuroute_settings_to_scratch()
            
            # Update configuration files with scratch paths
            self._update_file_paths_for_scratch()
            
            self.logger.info("Local scratch setup completed successfully")
            self.logger.info("="*60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup scratch space: {str(e)}")
            self.logger.error("Falling back to standard filesystem")
            self.use_scratch = False
            return False
    
    def _create_scratch_directories(self) -> None:
        """Create all necessary directories in scratch space."""
        self.logger.info("Creating scratch directory structure...")
        
        # Main directories
        dirs_to_create = [
            self.scratch_data_dir,
            self.scratch_project_dir,
            self.scratch_project_dir / "settings" / "SUMMA",
            self.scratch_project_dir / "settings" / "mizuRoute",
            self.scratch_project_dir / "forcing" / "SUMMA_input",
            self.scratch_project_dir / "observations" / "streamflow" / "preprocessed",
            self.scratch_project_dir / "simulations" / f"run_{self.algorithm_name}",
        ]
        
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"  Created: {directory}")
    
    def _copy_settings_to_scratch(self) -> None:
        """Copy SUMMA settings files to scratch."""
        source_dir = self.project_dir / "settings" / "SUMMA"
        dest_dir = self.scratch_project_dir / "settings" / "SUMMA"
        
        if not source_dir.exists():
            self.logger.warning(f"Settings directory not found: {source_dir}")
            return
        
        self.logger.info(f"Copying SUMMA settings: {source_dir} -> {dest_dir}")
        self._rsync_directory(source_dir, dest_dir)
    
    def _copy_forcing_to_scratch(self) -> None:
        """Copy forcing data to scratch."""
        source_dir = self.project_dir / "forcing" / "SUMMA_input"
        dest_dir = self.scratch_project_dir / "forcing" / "SUMMA_input"
        
        if not source_dir.exists():
            self.logger.warning(f"Forcing directory not found: {source_dir}")
            return
        
        self.logger.info(f"Copying forcing data: {source_dir} -> {dest_dir}")
        self._rsync_directory(source_dir, dest_dir)
    
    def _copy_observations_to_scratch(self) -> None:
        """Copy observation data to scratch."""
        source_dir = self.project_dir / "observations" / "streamflow" / "preprocessed"
        dest_dir = self.scratch_project_dir / "observations" / "streamflow" / "preprocessed"
        
        if not source_dir.exists():
            self.logger.info(f"Observations directory not found (ok if not needed): {source_dir}")
            return
        
        self.logger.info(f"Copying observation data: {source_dir} -> {dest_dir}")
        self._rsync_directory(source_dir, dest_dir)
    
    def _copy_mizuroute_settings_to_scratch(self) -> None:
        """Copy mizuRoute settings to scratch."""
        if self.config['ROUTING_DELINEATION'] != 'lumped': 

            source_dir = self.project_dir / "settings" / "mizuRoute"
            dest_dir = self.scratch_project_dir / "settings" / "mizuRoute"
            
            if not source_dir.exists():
                self.logger.warning(f"mizuRoute settings directory not found: {source_dir}")
                return
            
            self.logger.info(f"Copying mizuRoute settings: {source_dir} -> {dest_dir}")
            self._rsync_directory(source_dir, dest_dir)
            
            # Verify critical file
            topology_file = dest_dir / "topology.nc"
            if not topology_file.exists():
                self.logger.error(f"Critical file missing after copy: {topology_file}")
                raise FileNotFoundError(f"topology.nc not found in {dest_dir}")
        else:
            self.logger.info('lumped model no need to copy routing files')
            return
    
    def _rsync_directory(self, source: Path, dest: Path) -> None:
        """
        Copy directory using rsync for efficient transfer.
        
        Args:
            source: Source directory
            dest: Destination directory
        """
        try:
            # Use rsync for efficient copying
            # -a: archive mode (preserves permissions, timestamps, etc.)
            # -q: quiet mode
            subprocess.run(
                ['rsync', '-aq', f"{source}/", f"{dest}/"],
                check=True,
                capture_output=True
            )
            self.logger.debug(f"  Copied: {source} -> {dest}")
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"rsync failed, falling back to shutil.copytree: {e}")
            # Fallback to shutil if rsync not available
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(source, dest)
    
    def _update_file_paths_for_scratch(self) -> None:
        """Update file paths in configuration files to point to scratch."""
        self.logger.info("Updating file paths for scratch space...")
        
        # Update fileManager.txt if it exists
        file_manager = self.scratch_project_dir / "settings" / "SUMMA" / "fileManager.txt"
        if file_manager.exists():
            self._update_file_manager(file_manager)
        
        # Update mizuRoute control file if it exists  
        mizu_control = self.scratch_project_dir / "settings" / "mizuRoute" / "mizuroute.control"
        if mizu_control.exists():
            self._update_mizuroute_control(mizu_control)
    
    def _update_file_manager(self, file_manager_path: Path) -> None:
        """
        Update paths in SUMMA fileManager.txt.
        
        Args:
            file_manager_path: Path to fileManager.txt
        """
        self.logger.debug(f"Updating fileManager: {file_manager_path}")
        
        # Read file
        with open(file_manager_path, 'r') as f:
            lines = f.readlines()
        
        # Prepare paths
        forcing_path = str(self.scratch_project_dir / "forcing" / "SUMMA_input") + "/"
        
        # Update lines
        updated_lines = []
        for line in lines:
            if line.strip().startswith('forcingPath'):
                # Replace forcing path
                updated_lines.append(f"forcingPath          '{forcing_path}'\n")
            else:
                updated_lines.append(line)
        
        # Write back
        with open(file_manager_path, 'w') as f:
            f.writelines(updated_lines)
        
        self.logger.debug("  fileManager.txt updated")
    
    def _update_mizuroute_control(self, control_path: Path) -> None:
        """
        Update paths in mizuRoute control file.
        
        Args:
            control_path: Path to mizuroute.control
        """
        self.logger.debug(f"Updating mizuRoute control: {control_path}")
        
        # This will be updated by the optimizer when it creates process-specific
        # directories, so we just ensure the file exists
        self.logger.debug("  mizuroute.control will be updated by optimizer")
    
    def get_scratch_paths(self) -> Dict[str, Path]:
        """
        Get scratch paths for use in optimization.
        
        Returns:
            Dictionary with scratch paths if scratch is enabled,
            otherwise returns standard paths
        """
        if self.use_scratch:
            return {
                'data_dir': self.scratch_data_dir,
                'project_dir': self.scratch_project_dir,
                'settings_dir': self.scratch_project_dir / "settings" / "SUMMA",
                'mizuroute_settings_dir': self.scratch_project_dir / "settings" / "mizuRoute",
                'forcing_dir': self.scratch_project_dir / "forcing" / "SUMMA_input",
                'observations_dir': self.scratch_project_dir / "observations" / "streamflow" / "preprocessed",
            }
        else:
            return {
                'data_dir': self.original_data_dir,
                'project_dir': self.project_dir,
                'settings_dir': self.project_dir / "settings" / "SUMMA",
                'mizuroute_settings_dir': self.project_dir / "settings" / "mizuRoute",
                'forcing_dir': self.project_dir / "forcing" / "SUMMA_input",
                'observations_dir': self.project_dir / "observations" / "streamflow" / "preprocessed",
            }
    
    def stage_results_back(self) -> None:
        """
        Stage optimization results back to permanent storage.
        
        This copies:
        - Optimization results
        - Best parameter files
        - Log files
        - Any other important outputs
        """
        if not self.use_scratch:
            return
        
        self.logger.info("="*60)
        self.logger.info("Staging results back to permanent storage")
        self.logger.info("="*60)
        
        try:
            # Determine output location
            node_name = os.environ.get('SLURMD_NODENAME', 'unknown_node')
            output_dir = (
                self.project_dir / "simulations" / 
                f"run_{self.algorithm_name}" / 
                f"scratch_results_{node_name}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy simulation results
            self._stage_simulation_results(output_dir)
            
            # Copy optimization results
            self._stage_optimization_results(output_dir)
            
            # Copy work logs if they exist
            self._stage_work_logs(output_dir)
            
            self.logger.info(f"Results staged to: {output_dir}")
            self.logger.info("="*60)
            
        except Exception as e:
            self.logger.error(f"Error staging results back: {str(e)}")
            # Don't raise - we want to keep the results in scratch for manual recovery
            self.logger.error(
                f"Results remain in scratch at: "
                f"{self.scratch_project_dir}"
            )
    
    def _stage_simulation_results(self, output_dir: Path) -> None:
        """Stage simulation outputs back to permanent storage."""
        # SUMMA outputs
        scratch_summa = self.scratch_project_dir / "simulations" / f"run_{self.algorithm_name}" / "SUMMA"
        if scratch_summa.exists():
            dest_summa = output_dir / "SUMMA"
            self.logger.info(f"Staging SUMMA results: {scratch_summa} -> {dest_summa}")
            self._rsync_directory(scratch_summa, dest_summa)
        
        # mizuRoute outputs
        scratch_mizu = self.scratch_project_dir / "simulations" / f"run_{self.algorithm_name}" / "mizuRoute"
        if scratch_mizu.exists():
            dest_mizu = output_dir / "mizuRoute"
            self.logger.info(f"Staging mizuRoute results: {scratch_mizu} -> {dest_mizu}")
            self._rsync_directory(scratch_mizu, dest_mizu)
    
    def _stage_optimization_results(self, output_dir: Path) -> None:
        """Stage optimization-specific results."""
        # Optimization outputs directory
        scratch_opt = self.scratch_project_dir / "optimisation"
        if scratch_opt.exists():
            dest_opt = output_dir / "optimisation"
            self.logger.info(f"Staging optimization results: {scratch_opt} -> {dest_opt}")
            self._rsync_directory(scratch_opt, dest_opt)
    
    def _stage_work_logs(self, output_dir: Path) -> None:
        """Stage work logs if they exist."""
        work_log_dir = self.scratch_project_dir / f"_workLog_domain_{self.domain_name}"
        if work_log_dir.exists():
            dest_log = output_dir / f"_workLog_domain_{self.domain_name}"
            self.logger.info(f"Staging work logs: {work_log_dir} -> {dest_log}")
            self._rsync_directory(work_log_dir, dest_log)
    
    def get_effective_data_dir(self) -> Path:
        """
        Get the effective data directory (scratch or original).
        
        Returns:
            Path to use for CONFLUENCE_DATA_DIR
        """
        if self.use_scratch:
            return self.scratch_data_dir
        else:
            return self.original_data_dir
    
    def get_effective_project_dir(self) -> Path:
        """
        Get the effective project directory (scratch or original).
        
        Returns:
            Path to use for project operations
        """
        if self.use_scratch:
            return self.scratch_project_dir
        else:
            return self.project_dir
    
    def cleanup_scratch(self) -> None:
        """
        Clean up scratch space (optional - SLURM usually handles this).
        
        This is mainly for manual cleanup if needed.
        """
        if not self.use_scratch:
            return
        
        self.logger.info("Scratch space will be automatically cleaned by SLURM")
        # SLURM_TMPDIR is automatically cleaned up by SLURM after job completion
        # So we don't need to do anything here