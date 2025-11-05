#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CONFLUENCE Node-Agnostic Local Scratch Manager

This module provides functionality to use local scratch storage on HPC systems
to reduce IOPS on shared filesystems during parallel optimization, with support
for multi-node jobs.

Key improvements:
- Works across multiple nodes
- Each MPI rank copies data to its local node's scratch
- Node-aware result staging
- Automatic detection of node topology

The manager:
1. Detects HPC environment (SLURM) and node configuration
2. Each worker creates scratch directory on its local node
3. Each worker copies necessary data to its local scratch
4. Updates paths for optimization run on each node
5. Stages results back from each node to permanent storage

Usage:
    scratch_mgr = LocalScratchManager(config, logger, project_dir, algorithm_name)
    
    if scratch_mgr.use_scratch:
        # Setup scratch space (each worker does this)
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
import socket
import hashlib


class LocalScratchManager:
    """
    Manages local scratch space for optimization on HPC systems with multi-node support.
    
    This class handles the complete lifecycle of using local scratch:
    - Detection of HPC/SLURM environment
    - Node-aware directory structure creation
    - Data copying to each node's local scratch
    - Path management per node
    - Results staging back to permanent storage from each node
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger,
                 project_dir: Path, algorithm_name: str, mpi_rank: Optional[int] = None):
        """
        Initialize the local scratch manager.
        
        Args:
            config: CONFLUENCE configuration dictionary
            logger: Logger instance
            project_dir: Main project directory path
            algorithm_name: Name of optimization algorithm (for directory naming)
            mpi_rank: MPI rank of this process (None for serial execution)
        """
        self.config = config
        self.logger = logger
        self.project_dir = project_dir
        self.algorithm_name = algorithm_name
        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')
        self.mpi_rank = mpi_rank if mpi_rank is not None else 0
        
        # Get node information
        self.node_name = socket.gethostname()
        self.node_id = self._get_node_id()
        
        # Check if local scratch should be used
        self.use_scratch = self._should_use_scratch()
        
        # Initialize paths (will be set during setup)
        self.scratch_root = None
        self.scratch_data_dir = None
        self.scratch_project_dir = None
        self.original_data_dir = None
        
        if self.use_scratch:
            self.logger.info(f"Local scratch mode ENABLED for rank {self.mpi_rank} on node {self.node_name}")
            self._initialize_scratch_paths()
        else:
            self.logger.info(f"Local scratch mode DISABLED for rank {self.mpi_rank} - using standard filesystem")
    
    def _get_node_id(self) -> str:
        """
        Get a unique identifier for the current node.
        
        Returns:
            String identifier for the node
        """
        # Try to get SLURM node ID first
        slurm_nodeid = os.environ.get('SLURM_NODEID')
        if slurm_nodeid:
            return f"node{slurm_nodeid}"
        
        # Otherwise use hostname hash for uniqueness
        node_hash = hashlib.md5(self.node_name.encode()).hexdigest()[:8]
        return f"node_{node_hash}"
    
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
                f"Rank {self.mpi_rank}: USE_LOCAL_SCRATCH is True but SLURM_TMPDIR not found. "
                "Falling back to standard filesystem."
            )
            return False
        
        if not Path(slurm_tmpdir).exists():
            self.logger.warning(
                f"Rank {self.mpi_rank}: USE_LOCAL_SCRATCH is True but SLURM_TMPDIR ({slurm_tmpdir}) "
                "does not exist. Falling back to standard filesystem."
            )
            return False
        
        # Multi-node support: Log node configuration but don't disable
        num_nodes = os.environ.get('SLURM_JOB_NUM_NODES', '1')
        try:
            num_nodes = int(num_nodes)
            if num_nodes > 1:
                self.logger.info(
                    f"Rank {self.mpi_rank}: Job spans {num_nodes} nodes. "
                    f"Using local scratch on node {self.node_name}"
                )
        except ValueError:
            pass
        
        self.logger.info(
            f"Rank {self.mpi_rank}: SLURM environment detected: "
            f"TMPDIR={slurm_tmpdir}, JOB_ID={slurm_job_id}, NODE={self.node_name}"
        )
        return True
    
    def _initialize_scratch_paths(self) -> None:
        """Initialize scratch directory paths with node awareness."""
        self.scratch_root = Path(os.environ.get('SLURM_TMPDIR'))
        
        # Create scratch data directory with rank-specific subdirectory to avoid conflicts
        # when multiple ranks are on the same node
        self.scratch_data_dir = self.scratch_root / "conf_data" / f"rank_{self.mpi_rank}"
        self.scratch_project_dir = self.scratch_data_dir / f"domain_{self.domain_name}"
        
        # Store original for reference
        self.original_data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        
        self.logger.info(f"Rank {self.mpi_rank} on {self.node_name}:")
        self.logger.info(f"  Scratch root: {self.scratch_root}")
        self.logger.info(f"  Scratch data dir: {self.scratch_data_dir}")
        self.logger.info(f"  Scratch project dir: {self.scratch_project_dir}")
    
    def _needs_routing(self) -> bool:
        """
        Determine if mizuRoute routing is needed based on configuration.
        
        Returns:
            True if routing is needed, False otherwise
        """
        # Check routing delineation setting
        routing_delineation = self.config.get('ROUTING_DELINEATION', 'lumped').lower()
        
        # Skip routing for lumped or none
        if routing_delineation in ['lumped', 'none', 'off', 'false']:
            return False
        
        # Check domain definition method
        domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', 'lumped').lower()
        
        # If domain is point or lumped, typically no routing needed
        if domain_method in ['point', 'lumped']:
            # Unless specifically using river_network routing
            if routing_delineation == 'river_network':
                return True
            return False
        
        # For distributed domains, routing is typically needed
        return True
    
    def setup_scratch_space(self) -> bool:
        """
        Setup complete scratch space with all necessary data.
        Each MPI rank sets up its own scratch space on its local node.
        
        This method:
        1. Creates directory structure on local node
        2. Copies settings files to local scratch
        3. Copies forcing data to local scratch
        4. Copies observation data to local scratch
        5. Updates file paths in configuration files
        
        Returns:
            True if setup successful, False otherwise
        """
        if not self.use_scratch:
            return True
        
        try:
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Rank {self.mpi_rank}: Setting up local scratch on {self.node_name}")
            self.logger.info(f"{'='*60}")
            
            # Check if we've already set up scratch for this rank
            # (useful when multiple workers share the same rank on a node)
            setup_marker = self.scratch_project_dir / ".scratch_setup_complete"
            if setup_marker.exists():
                self.logger.info(
                    f"Rank {self.mpi_rank}: Scratch already set up on {self.node_name}, skipping"
                )
                return True
            
            # Create directory structure
            self._create_scratch_directories()
            
            # Copy data to scratch
            self._copy_settings_to_scratch()
            self._copy_forcing_to_scratch()
            self._copy_observations_to_scratch()
            
            # Only copy mizuRoute settings if routing is needed
            if self._needs_routing():
                self._copy_mizuroute_settings_to_scratch()
            else:
                self.logger.info(f"Rank {self.mpi_rank}: Routing not needed - skipping mizuRoute settings copy")
            
            # Update configuration files with scratch paths
            self._update_file_paths_for_scratch()
            
            # Mark setup as complete
            setup_marker.touch()
            
            self.logger.info(f"Rank {self.mpi_rank}: Scratch setup completed successfully on {self.node_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Rank {self.mpi_rank}: Error setting up scratch space on {self.node_name}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Fall back to standard filesystem
            self.use_scratch = False
            return False
    
    def _create_scratch_directories(self) -> None:
        """Create necessary directory structure in scratch space."""
        self.logger.info(f"Rank {self.mpi_rank}: Creating scratch directories on {self.node_name}...")
        
        # Create main directories
        directories = [
            self.scratch_data_dir,
            self.scratch_project_dir,
            self.scratch_project_dir / "settings" / "SUMMA",
            self.scratch_project_dir / "settings" / "mizuRoute",
            self.scratch_project_dir / "forcing" / "SUMMA_input",
            self.scratch_project_dir / "observations" / "streamflow" / "preprocessed",
            self.scratch_project_dir / "simulations" / f"run_{self.algorithm_name}" / "SUMMA",
            self.scratch_project_dir / "simulations" / f"run_{self.algorithm_name}" / "mizuRoute",
            self.scratch_project_dir / "optimisation",
            self.scratch_project_dir / f"_workLog_domain_{self.domain_name}",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"  Created: {directory}")
    
    def _copy_settings_to_scratch(self) -> None:
        """Copy SUMMA settings to scratch."""
        self.logger.info(f"Rank {self.mpi_rank}: Copying SUMMA settings to scratch on {self.node_name}...")
        
        source_settings = self.project_dir / "settings" / "SUMMA"
        dest_settings = self.scratch_project_dir / "settings" / "SUMMA"
        
        # Copy all settings files
        self._rsync_directory(source_settings, dest_settings)
    
    def _copy_forcing_to_scratch(self) -> None:
        """Copy forcing data to scratch."""
        self.logger.info(f"Rank {self.mpi_rank}: Copying forcing data to scratch on {self.node_name}...")
        
        source_forcing = self.project_dir / "forcing" / "SUMMA_input"
        dest_forcing = self.scratch_project_dir / "forcing" / "SUMMA_input"
        
        # Copy forcing data
        self._rsync_directory(source_forcing, dest_forcing)
    
    def _copy_observations_to_scratch(self) -> None:
        """Copy observation data to scratch."""
        self.logger.info(f"Rank {self.mpi_rank}: Copying observation data to scratch on {self.node_name}...")
        
        source_obs = self.project_dir / "observations" / "streamflow" / "preprocessed"
        dest_obs = self.scratch_project_dir / "observations" / "streamflow" / "preprocessed"
        
        # Only copy if source exists
        if source_obs.exists():
            self._rsync_directory(source_obs, dest_obs)
        else:
            self.logger.warning(f"Rank {self.mpi_rank}: No observation data found to copy")
    
    def _copy_mizuroute_settings_to_scratch(self) -> None:
        """Copy mizuRoute settings to scratch."""
        self.logger.info(f"Rank {self.mpi_rank}: Copying mizuRoute settings to scratch on {self.node_name}...")
        
        source_mizu = self.project_dir / "settings" / "mizuRoute"
        dest_mizu = self.scratch_project_dir / "settings" / "mizuRoute"
        
        # Only copy if source exists
        if source_mizu.exists():
            self._rsync_directory(source_mizu, dest_mizu)
        else:
            self.logger.warning(f"Rank {self.mpi_rank}: No mizuRoute settings found to copy")
    
    def _rsync_directory(self, source: Path, dest: Path) -> None:
        """
        Use rsync to efficiently copy directory contents.
        Falls back to shutil if rsync is not available.
        
        Args:
            source: Source directory
            dest: Destination directory
        """
        try:
            # Use rsync for efficient copying (handles existing files better)
            subprocess.run(
                ['rsync', '-a', '--delete', f'{source}/', f'{dest}/'],
                check=True,
                capture_output=True,
                text=True
            )
            self.logger.debug(f"  Copied: {source} -> {dest}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            self.logger.warning(f"Rank {self.mpi_rank}: rsync failed, falling back to shutil.copytree: {e}")
            # Fallback to shutil if rsync not available
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(source, dest)
    
    def _update_file_paths_for_scratch(self) -> None:
        """Update file paths in configuration files to point to scratch."""
        self.logger.info(f"Rank {self.mpi_rank}: Updating file paths for scratch space...")
        
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
        self.logger.debug(f"Rank {self.mpi_rank}: Updating fileManager: {file_manager_path}")
        
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
        
        self.logger.debug(f"  Rank {self.mpi_rank}: fileManager.txt updated")
    
    def _update_mizuroute_control(self, control_path: Path) -> None:
        """
        Update paths in mizuRoute control file.
        
        Args:
            control_path: Path to mizuroute.control
        """
        self.logger.debug(f"Rank {self.mpi_rank}: Updating mizuRoute control: {control_path}")
        
        # This will be updated by the optimizer when it creates process-specific
        # directories, so we just ensure the file exists
        self.logger.debug(f"  Rank {self.mpi_rank}: mizuroute.control will be updated by optimizer")
    
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
                'data_dir': self.original_data_dir if self.original_data_dir else Path(self.config.get('CONFLUENCE_DATA_DIR')),
                'project_dir': self.project_dir,
                'settings_dir': self.project_dir / "settings" / "SUMMA",
                'mizuroute_settings_dir': self.project_dir / "settings" / "mizuRoute",
                'forcing_dir': self.project_dir / "forcing" / "SUMMA_input",
                'observations_dir': self.project_dir / "observations" / "streamflow" / "preprocessed",
            }
    
    def stage_results_back(self) -> None:
        """
        Stage optimization results back to permanent storage.
        Each rank stages its own results with node and rank identification.
        
        This copies:
        - Optimization results
        - Best parameter files
        - Log files
        - Any other important outputs
        """
        if not self.use_scratch:
            return
        
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Rank {self.mpi_rank}: Staging results back from {self.node_name}")
        self.logger.info(f"{'='*60}")
        
        try:
            # Determine output location with node and rank info
            output_dir = (
                self.project_dir / "simulations" / 
                f"run_{self.algorithm_name}" / 
                f"scratch_results_{self.node_id}_rank{self.mpi_rank}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy simulation results
            self._stage_simulation_results(output_dir)
            
            # Copy optimization results
            self._stage_optimization_results(output_dir)
            
            # Copy work logs if they exist
            self._stage_work_logs(output_dir)
            
            self.logger.info(f"Rank {self.mpi_rank}: Results staged to: {output_dir}")
            self.logger.info(f"{'='*60}")
            
        except Exception as e:
            self.logger.error(f"Rank {self.mpi_rank}: Error staging results back: {str(e)}")
            # Don't raise - we want to keep the results in scratch for manual recovery
            self.logger.error(
                f"Rank {self.mpi_rank}: Results remain in scratch at: "
                f"{self.scratch_project_dir}"
            )
    
    def _stage_simulation_results(self, output_dir: Path) -> None:
        """Stage simulation outputs back to permanent storage."""
        # SUMMA outputs
        scratch_summa = self.scratch_project_dir / "simulations" / f"run_{self.algorithm_name}" / "SUMMA"
        if scratch_summa.exists():
            dest_summa = output_dir / "SUMMA"
            self.logger.info(f"Rank {self.mpi_rank}: Staging SUMMA results: {scratch_summa} -> {dest_summa}")
            self._rsync_directory(scratch_summa, dest_summa)
        
        # mizuRoute outputs
        scratch_mizu = self.scratch_project_dir / "simulations" / f"run_{self.algorithm_name}" / "mizuRoute"
        if scratch_mizu.exists():
            dest_mizu = output_dir / "mizuRoute"
            self.logger.info(f"Rank {self.mpi_rank}: Staging mizuRoute results: {scratch_mizu} -> {dest_mizu}")
            self._rsync_directory(scratch_mizu, dest_mizu)
    
    def _stage_optimization_results(self, output_dir: Path) -> None:
        """Stage optimization-specific results."""
        # Optimization outputs directory
        scratch_opt = self.scratch_project_dir / "optimisation"
        if scratch_opt.exists():
            dest_opt = output_dir / "optimisation"
            self.logger.info(f"Rank {self.mpi_rank}: Staging optimization results: {scratch_opt} -> {dest_opt}")
            self._rsync_directory(scratch_opt, dest_opt)
    
    def _stage_work_logs(self, output_dir: Path) -> None:
        """Stage work logs if they exist."""
        work_log_dir = self.scratch_project_dir / f"_workLog_domain_{self.domain_name}"
        if work_log_dir.exists():
            dest_log = output_dir / f"_workLog_domain_{self.domain_name}"
            self.logger.info(f"Rank {self.mpi_rank}: Staging work logs: {work_log_dir} -> {dest_log}")
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
            return self.original_data_dir if self.original_data_dir else Path(self.config.get('CONFLUENCE_DATA_DIR'))
    
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
        
        self.logger.info(f"Rank {self.mpi_rank}: Scratch space will be automatically cleaned by SLURM")
        # SLURM_TMPDIR is automatically cleaned up by SLURM after job completion
        # So we don't need to do anything here
    
    def sync_scratch_between_ranks(self, comm=None) -> None:
        """
        Optional: Synchronize scratch setup between MPI ranks on the same node.
        This can be used to avoid duplicate copying when multiple ranks are on the same node.
        
        Args:
            comm: MPI communicator (if using mpi4py)
        """
        if not self.use_scratch or comm is None:
            return
        
        # This is an advanced feature that requires mpi4py
        try:
            from mpi4py import MPI
            
            # Create a communicator for ranks on the same node
            node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
            node_rank = node_comm.Get_rank()
            
            # Only rank 0 on each node does the setup
            if node_rank == 0:
                self.logger.info(f"Rank {self.mpi_rank}: Primary rank on {self.node_name}, setting up scratch")
                success = self.setup_scratch_space()
            else:
                self.logger.info(f"Rank {self.mpi_rank}: Secondary rank on {self.node_name}, waiting for scratch setup")
                success = None
            
            # Broadcast success status to all ranks on the node
            success = node_comm.bcast(success, root=0)
            
            # All ranks wait for setup to complete
            node_comm.Barrier()
            
            if not success:
                self.use_scratch = False
                self.logger.warning(f"Rank {self.mpi_rank}: Scratch setup failed, falling back to standard filesystem")
            
            node_comm.Free()
            
        except ImportError:
            self.logger.info(f"Rank {self.mpi_rank}: mpi4py not available, each rank will set up its own scratch")
            self.setup_scratch_space()