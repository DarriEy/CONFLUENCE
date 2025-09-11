#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FUSE Parameter Manager - FIXED for NetCDF Index Issues

This version fixes the "Index exceeds dimension bound" error by ensuring
proper parameter file structure and indexing.
"""

import numpy as np
import xarray as xr
import netCDF4 as nc
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging


class FUSEParameterManager:
    """Handles FUSE parameter bounds, normalization, and file updates - FIXED VERSION"""
    
    def __init__(self, config: Dict, logger: logging.Logger, fuse_settings_dir: Path):
        self.config = config
        self.logger = logger
        self.fuse_settings_dir = fuse_settings_dir
        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')
        
        # Parse FUSE parameters to calibrate
        fuse_params_str = config.get('SETTINGS_FUSE_PARAMS_TO_CALIBRATE', '')
        self.fuse_params = [p.strip() for p in fuse_params_str.split(',') if p.strip()]
        
        # Define parameter bounds for common FUSE parameters
        self.param_bounds = self._get_default_fuse_bounds()
        
        # Path to FUSE parameter files
        self.data_dir = Path(config.get('CONFLUENCE_DATA_DIR'))
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.fuse_sim_dir = self.project_dir / 'simulations' / self.experiment_id / 'FUSE'
        
        # Parameter file paths
        self.para_def_path = self.fuse_sim_dir / f"{self.domain_name}_{self.experiment_id}_para_def.nc"
        self.para_sce_path = self.fuse_sim_dir / f"{self.domain_name}_{self.experiment_id}_para_sce.nc"  
        self.para_best_path = self.fuse_sim_dir / f"{self.domain_name}_{self.experiment_id}_para_best.nc"
        
        # CRITICAL: Use para_sce.nc for calibration iterations, but ensure it's properly structured
        self.param_file_path = self.para_sce_path
    
    def _get_default_fuse_bounds(self) -> Dict[str, Dict[str, float]]:
        """Define reasonable bounds for FUSE parameters"""
        return {
            # Snow parameters (most important for snow-dominated basins)
            'MBASE': {'min': -5.0, 'max': 5.0},        # Base melt temperature (°C)
            'MFMAX': {'min': 1.0, 'max': 10.0},        # Maximum melt factor (mm/(°C·day))
            'MFMIN': {'min': 0.5, 'max': 5.0},         # Minimum melt factor (mm/(°C·day))
            'PXTEMP': {'min': -2.0, 'max': 2.0},       # Rain-snow partition temperature (°C)
            'LAPSE': {'min': 3.0, 'max': 10.0},        # Temperature lapse rate (°C/km)
            
            # Soil parameters
            'MAXWATR_1': {'min': 50.0, 'max': 1000.0},   # Maximum storage upper layer (mm)
            'MAXWATR_2': {'min': 100.0, 'max': 2000.0},  # Maximum storage lower layer (mm)
            'FRACTEN': {'min': 0.1, 'max': 0.9},         # Fraction tension storage
            'PERCRTE': {'min': 0.01, 'max': 100.0},      # Percolation rate (mm/day)
            'PERCEXP': {'min': 1.0, 'max': 20.0},        # Percolation exponent
            
            # Baseflow parameters
            'BASERTE': {'min': 0.001, 'max': 1.0},       # Baseflow rate (mm/day)
            'QB_POWR': {'min': 1.0, 'max': 10.0},        # Baseflow exponent
            'QBRATE_2A': {'min': 0.001, 'max': 0.1},     # Primary baseflow depletion (day⁻¹)
            'QBRATE_2B': {'min': 0.0001, 'max': 0.01},   # Secondary baseflow depletion (day⁻¹)
            
            # Routing parameters
            'TIMEDELAY': {'min': 0.0, 'max': 10.0},      # Time delay in routing (days)
            
            # Evapotranspiration parameters
            'RTFRAC1': {'min': 0.1, 'max': 0.9},         # Fraction roots upper layer
            'RTFRAC2': {'min': 0.1, 'max': 0.9},         # Fraction roots lower layer
        }
    
    def verify_and_fix_parameter_files(self) -> bool:
        """Verify parameter file structure and fix indexing issues"""
        try:
            self.logger.debug("Verifying FUSE parameter file structure...")
            
            # Check each parameter file
            for file_path in [self.para_def_path, self.para_sce_path, self.para_best_path]:
                if file_path.exists():
                    self.logger.debug(f"Checking {file_path.name}")
                    
                    with xr.open_dataset(file_path) as ds:
                        #self.logger.debug(f"  Dimensions: {dict(ds.dims)}")
                        #self.logger.debug(f"  Parameters available: {list(ds.data_vars.keys())}")
                        
                        # Check if 'par' dimension exists and has correct size
                        if 'par' in ds.dims:
                            par_size = ds.sizes['par']
                            self.logger.debug(f"  Parameter dimension size: {par_size}")
                            
                            if par_size == 0:
                                self.logger.error(f"  ERROR: {file_path.name} has empty parameter dimension!")
                                return False
                            
                            # Verify that we can access parameter set 0
                            try:
                                for param in self.fuse_params:
                                    if param in ds.variables:
                                        test_value = ds[param].isel(par=0).values
                                        self.logger.debug(f"  {param}[0] = {test_value}")
                                    else:
                                        self.logger.warning(f"  Parameter {param} not found in {file_path.name}")
                                        
                                self.logger.debug(f"  ✓ {file_path.name} parameter indexing OK")
                                
                            except Exception as e:
                                self.logger.error(f"  ERROR: Cannot access parameter set 0 in {file_path.name}: {str(e)}")
                                # Try to fix the file
                                if self._fix_parameter_file_indexing(file_path):
                                    self.logger.info(f"  ✓ Fixed parameter indexing in {file_path.name}")
                                else:
                                    return False
                        else:
                            self.logger.error(f"  ERROR: {file_path.name} missing 'par' dimension!")
                            return False
                else:
                    self.logger.warning(f"Parameter file {file_path.name} does not exist")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying parameter files: {str(e)}")
            return False
    
    def _fix_parameter_file_indexing(self, file_path: Path) -> bool:
        """Fix parameter file indexing issues"""
        try:
            self.logger.info(f"Rebuilding parameter file: {file_path.name}")
            
            # Create backup
            backup_path = file_path.with_suffix('.nc.backup')
            if file_path.exists() and not backup_path.exists():
                import shutil
                shutil.copy2(file_path, backup_path)
            
            # Create new file with proper structure
            import netCDF4 as nc
            with nc.Dataset(file_path, 'w', format='NETCDF4') as ds:
                # Create dimensions - CRITICAL: Use size 1, not unlimited
                ds.createDimension('par', 1)
                
                # Create coordinate variable
                par_var = ds.createVariable('par', 'i4', ('par',))
                par_var[:] = [0]  # Set to 0-based indexing
                
                # Create parameter variables with default values
                for param_name in self.fuse_params:
                    param_var = ds.createVariable(param_name, 'f8', ('par',))
                    default_val = self._get_default_parameter_value(param_name)
                    param_var[:] = [default_val]
                    
                # Ensure the file is properly closed and synced
                ds.sync()
            
            self.logger.info(f"Successfully rebuilt {file_path.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to fix {file_path.name}: {str(e)}")
            return False
    
    def _get_default_parameter_value(self, param_name: str) -> float:
        """Get default value for a parameter"""
        if param_name in self.param_bounds:
            bounds = self.param_bounds[param_name]
            return (bounds['min'] + bounds['max']) / 2.0
        else:
            # Generic default for unknown parameters
            return 1.0
    
    @property
    def all_param_names(self) -> List[str]:
        """Get list of all FUSE parameter names to calibrate"""
        return self.fuse_params
    
    def get_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """Get parameter bounds for calibration"""
        bounds = {}
        for param in self.fuse_params:
            if param in self.param_bounds:
                bounds[param] = self.param_bounds[param]
            else:
                self.logger.warning(f"No bounds defined for parameter {param}, using default [0.1, 10.0]")
                bounds[param] = {'min': 0.1, 'max': 10.0}
        return bounds
    
    def update_parameter_file(self, params: Dict[str, float], use_best_file: bool = False) -> bool:
        """Update FUSE parameter NetCDF file with new parameter values - FIXED VERSION"""
        try:
            # FIRST: Verify parameter files are properly structured
            if not self.verify_and_fix_parameter_files():
                self.logger.error("Cannot proceed - parameter files have structural issues")
                return False
            
            # Choose which file to update
            target_file = self.param_file_path
            
            if not target_file.exists():
                self.logger.error(f"Parameter file does not exist: {target_file}")
                return False
            
            self.logger.debug(f"Updating parameter file: {target_file}")
            
            # SAFE NetCDF writing with proper error handling
            try:
                with nc.Dataset(target_file, 'r+') as ds:
                    # Verify the file structure first
                    if 'par' not in ds.dimensions:
                        self.logger.error(f"Missing 'par' dimension in {target_file}")
                        return False
                    
                    par_size = ds.dimensions['par'].size
                    if par_size == 0:
                        self.logger.error(f"Empty 'par' dimension in {target_file}")
                        return False
                    
                    changed = 0
                    for p, v in params.items():
                        if p in ds.variables:
                            try:
                                # Always use index 0 for single parameter set
                                before = float(ds.variables[p][0])
                                ds.variables[p][0] = float(v)
                                after = float(ds.variables[p][0])
                                self.logger.debug(f"[param write] {p}: {before} -> {after}")
                                changed += (abs(after - before) > 1e-10)
                            except Exception as e:
                                self.logger.error(f"Error updating parameter {p}: {str(e)}")
                                return False
                        else:
                            self.logger.error(f"[param write] {p} NOT FOUND in {target_file}")
                            return False
                    
                    # Force sync to disk
                    ds.sync()
                
                if changed == 0:
                    self.logger.warning("[param write] No values changed - check parameter bounds or file structure")
                else:
                    self.logger.debug(f"Successfully updated {changed} FUSE parameters in {target_file}")

                
                return True
                
            except Exception as e:
                self.logger.error(f"NetCDF error updating {target_file}: {str(e)}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error updating parameter file: {str(e)}")
            return False
    
    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from existing FUSE parameter file"""
        try:
            # First ensure files are properly structured
            if not self.verify_and_fix_parameter_files():
                self.logger.warning("Parameter files need fixing - using default values")
                return self._get_default_initial_values()
            
            if not self.param_file_path.exists():
                self.logger.warning(f"FUSE parameter file not found: {self.param_file_path}")
                return self._get_default_initial_values()
            
            with xr.open_dataset(self.param_file_path) as ds:
                params = {}
                for param_name in self.fuse_params:
                    if param_name in ds.variables:
                        # Get the parameter value (assuming parameter set 0)
                        params[param_name] = float(ds[param_name].isel(par=0).values)
                    else:
                        self.logger.warning(f"Parameter {param_name} not found in file")
                        # Use default value from bounds
                        bounds = self.param_bounds.get(param_name, {'min': 0.1, 'max': 10.0})
                        params[param_name] = (bounds['min'] + bounds['max']) / 2
                
                return params
        
        except Exception as e:
            self.logger.error(f"Error reading initial parameters: {str(e)}")
            return self._get_default_initial_values()
    
    def _get_default_initial_values(self) -> Dict[str, float]:
        """Get default initial parameter values"""
        params = {}
        bounds = self.get_parameter_bounds()
        
        for param_name in self.fuse_params:
            param_bounds = bounds[param_name]
            # Use middle of bounds as default
            params[param_name] = (param_bounds['min'] + param_bounds['max']) / 2
        
        return params
    
    def normalize_parameters(self, params: Dict[str, float]) -> np.ndarray:
        """Normalize parameters to [0, 1] range for optimization"""
        bounds = self.get_parameter_bounds()
        normalized = []
        
        for param_name in self.fuse_params:
            value = params[param_name]
            param_bounds = bounds[param_name]
            norm_value = (value - param_bounds['min']) / (param_bounds['max'] - param_bounds['min'])
            normalized.append(np.clip(norm_value, 0.0, 1.0))
        
        return np.array(normalized)
    
    def denormalize_parameters(self, normalized_params: np.ndarray) -> Dict[str, float]:
        """Denormalize parameters from [0, 1] range to actual values"""
        bounds = self.get_parameter_bounds()
        params = {}
        
        for i, param_name in enumerate(self.fuse_params):
            param_bounds = bounds[param_name]
            value = (normalized_params[i] * (param_bounds['max'] - param_bounds['min']) + 
                    param_bounds['min'])
            params[param_name] = float(value)
        
        return params
    
    def validate_parameters(self, params: Dict[str, float]) -> bool:
        """Validate that parameters are within bounds"""
        bounds = self.get_parameter_bounds()
        
        for param_name, value in params.items():
            if param_name in bounds:
                param_bounds = bounds[param_name]
                if not (param_bounds['min'] <= value <= param_bounds['max']):
                    self.logger.warning(
                        f"Parameter {param_name}={value} outside bounds "
                        f"[{param_bounds['min']}, {param_bounds['max']}]"
                    )
                    return False
        
        return True