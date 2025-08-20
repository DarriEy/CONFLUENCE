#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CONFLUENCE Optimizer

This module provides parameter optimization for CONFLUENCE hydrological models using with support for multiple calibration targets (streamflow, snow, etc.),
parallel processing, and parameter management.

Features:
- Multi-target calibration (streamflow, snow SWE)
- Parallel processing with process isolation
- Soil depth calibration using shape method
- mizuRoute routing parameter optimization
- Parameter persistence and restart capabilities
- Comprehensive results tracking and visualization

Usage:
    from iterative_optimizer import Optimizer
    
    optimizer = Optimizer(config, logger)
    results = optimizer.run_optimization()
"""

import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
import subprocess
from pathlib import Path
import logging
from datetime import datetime
import shutil
from typing import Dict, Any, List, Tuple, Optional
from abc import ABC, abstractmethod
import re
import logging
import gc
import signal
import sys
import time
import random
import traceback
import yaml
import pickle 
import tempfile

# ============= ABSTRACT BASE CLASSES =============

class CalibrationTarget(ABC):
    """Abstract base class for different calibration variables (streamflow, snow, etc.)"""
    
    def __init__(self, config: Dict, project_dir: Path, logger: logging.Logger):
        self.config = config
        self.project_dir = project_dir
        self.logger = logger
        self.domain_name = config.get('DOMAIN_NAME')
        
        # Parse time periods
        self.calibration_period = self._parse_date_range(config.get('CALIBRATION_PERIOD', ''))
        self.evaluation_period = self._parse_date_range(config.get('EVALUATION_PERIOD', ''))
    
    def calculate_metrics(self, sim_dir: Path, mizuroute_dir: Optional[Path] = None, 
                         calibration_only: bool = True) -> Optional[Dict[str, float]]:
        """
        Calculate performance metrics for this calibration target
        
        Args:
            sim_dir: SUMMA simulation directory
            mizuroute_dir: mizuRoute simulation directory (if needed)
            calibration_only: If True, only calculate calibration period metrics
                            If False, calculate both calibration and evaluation metrics
        """
        try:
            # Determine which simulation directory to use
            if self.needs_routing() and mizuroute_dir:
                output_dir = mizuroute_dir
            else:
                output_dir = sim_dir
            
            # Get simulation files
            sim_files = self.get_simulation_files(output_dir)
            if not sim_files:
                self.logger.error(f"No simulation files found in {output_dir}")
                return None
            
            # Extract simulated data
            sim_data = self.extract_simulated_data(sim_files)
            if sim_data is None or len(sim_data) == 0:
                self.logger.error("Failed to extract simulated data")
                return None
            
            # Load observed data
            obs_data = self._load_observed_data()
            if obs_data is None or len(obs_data) == 0:
                self.logger.error("Failed to load observed data")
                return None
            
            # Align time series and calculate metrics
            metrics = {}
            
            # Always calculate metrics for calibration period if available
            if self.calibration_period[0] and self.calibration_period[1]:
                calib_metrics = self._calculate_period_metrics(
                    obs_data, sim_data, self.calibration_period, "Calib"
                )
                metrics.update(calib_metrics)
            
            # Only calculate evaluation period metrics if requested (final evaluation)
            if not calibration_only and self.evaluation_period[0] and self.evaluation_period[1]:
                eval_metrics = self._calculate_period_metrics(
                    obs_data, sim_data, self.evaluation_period, "Eval"
                )
                metrics.update(eval_metrics)
            
            # If no specific periods, calculate for full overlap (fallback)
            if not metrics:
                full_metrics = self._calculate_period_metrics(obs_data, sim_data, (None, None), "")
                metrics.update(full_metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics for {self.__class__.__name__}: {str(e)}")
            return None
    
    @abstractmethod
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Get relevant simulation output files for this calibration target"""
        pass
    
    @abstractmethod
    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract simulated data from output files"""
        pass
    
    @abstractmethod
    def get_observed_data_path(self) -> Path:
        """Get path to observed data file"""
        pass
    
    @abstractmethod
    def needs_routing(self) -> bool:
        """Whether this calibration target requires mizuRoute routing"""
        pass
    
    def _load_observed_data(self) -> Optional[pd.Series]:
        """Load observed data from file"""
        try:
            obs_path = self.get_observed_data_path()
            if not obs_path.exists():
                self.logger.error(f"Observed data file not found: {obs_path}")
                return None
            
            obs_df = pd.read_csv(obs_path)
            
            # Find date and data columns
            date_col = next((col for col in obs_df.columns 
                           if any(term in col.lower() for term in ['date', 'time', 'datetime'])), None)
            
            data_col = self._get_observed_data_column(obs_df.columns)
            
            if not date_col or not data_col:
                self.logger.error(f"Could not identify date/data columns in {obs_path}")
                return None
            
            # Process data
            obs_df['DateTime'] = pd.to_datetime(obs_df[date_col])
            obs_df.set_index('DateTime', inplace=True)
            
            return obs_df[data_col]
            
        except Exception as e:
            self.logger.error(f"Error loading observed data: {str(e)}")
            return None
    
    @abstractmethod
    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        """Identify the data column in observed data file"""
        pass
        
    def _calculate_period_metrics(self, obs_data: pd.Series, sim_data: pd.Series, 
                                period: Tuple, prefix: str) -> Dict[str, float]:
        """Calculate metrics for a specific time period with explicit filtering"""
        try:
            # EXPLICIT filtering for both datasets (consistent with parallel worker)
            if period[0] and period[1]:
                # Filter observed data to period
                period_mask = (obs_data.index >= period[0]) & (obs_data.index <= period[1])
                obs_period = obs_data[period_mask]
                
                # Explicitly filter simulated data to same period (like parallel worker)
                sim_data.index = sim_data.index.round('h')  # Round first for consistency
                sim_period_mask = (sim_data.index >= period[0]) & (sim_data.index <= period[1])
                sim_period = sim_data[sim_period_mask]
                
                # Log filtering results for debugging
                self.logger.debug(f"{prefix} period filtering: {period[0]} to {period[1]}")
                self.logger.debug(f"{prefix} observed points: {len(obs_period)}")
                self.logger.debug(f"{prefix} simulated points: {len(sim_period)}")
            else:
                obs_period = obs_data
                sim_period = sim_data
                sim_period.index = sim_period.index.round('h')
            
            # Find common time indices
            common_idx = obs_period.index.intersection(sim_period.index)
            
            if len(common_idx) == 0:
                self.logger.warning(f"No common time indices for {prefix} period")
                return {}
            
            obs_common = obs_period.loc[common_idx]
            sim_common = sim_period.loc[common_idx]
            
            # Log final aligned data for debugging
            self.logger.debug(f"{prefix} aligned data points: {len(common_idx)}")
            self.logger.debug(f"{prefix} obs range: {obs_common.min():.3f} to {obs_common.max():.3f}")
            self.logger.debug(f"{prefix} sim range: {sim_common.min():.3f} to {sim_common.max():.3f}")
            
            # Calculate metrics
            base_metrics = self._calculate_performance_metrics(obs_common, sim_common)
            
            # Add prefix if specified
            if prefix:
                return {f"{prefix}_{k}": v for k, v in base_metrics.items()}
            else:
                return base_metrics
                
        except Exception as e:
            self.logger.error(f"Error calculating period metrics: {str(e)}")
            return {}
    
    def _calculate_performance_metrics(self, observed: pd.Series, simulated: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics between observed and simulated data"""
        try:
            # Clean data
            observed = pd.to_numeric(observed, errors='coerce')
            simulated = pd.to_numeric(simulated, errors='coerce')
            
            valid = ~(observed.isna() | simulated.isna())
            observed = observed[valid]
            simulated = simulated[valid]
            
            if len(observed) == 0:
                return {'KGE': np.nan, 'NSE': np.nan, 'RMSE': np.nan, 'PBIAS': np.nan, 'MAE': np.nan}
            
            # Nash-Sutcliffe Efficiency
            mean_obs = observed.mean()
            nse_num = ((observed - simulated) ** 2).sum()
            nse_den = ((observed - mean_obs) ** 2).sum()
            nse = 1 - (nse_num / nse_den) if nse_den > 0 else np.nan
            
            # Root Mean Square Error
            rmse = np.sqrt(((observed - simulated) ** 2).mean())
            
            # Percent Bias
            pbias = 100 * (simulated.sum() - observed.sum()) / observed.sum() if observed.sum() != 0 else np.nan
            
            # Kling-Gupta Efficiency
            r = observed.corr(simulated)
            alpha = simulated.std() / observed.std() if observed.std() != 0 else np.nan
            beta = simulated.mean() / mean_obs if mean_obs != 0 else np.nan
            kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2) if not np.isnan(r + alpha + beta) else np.nan
            
            # Mean Absolute Error
            mae = (observed - simulated).abs().mean()
            
            return {
                'KGE': kge,
                'NSE': nse,
                'RMSE': rmse,
                'PBIAS': pbias,
                'MAE': mae,
                'r': r,
                'alpha': alpha,
                'beta': beta
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {'KGE': np.nan, 'NSE': np.nan, 'RMSE': np.nan, 'PBIAS': np.nan, 'MAE': np.nan}
    
    def _parse_date_range(self, date_range_str: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Parse date range string from config"""
        if not date_range_str:
            return None, None
        
        try:
            dates = [d.strip() for d in date_range_str.split(',')]
            if len(dates) >= 2:
                return pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
        except Exception as e:
            self.logger.warning(f"Could not parse date range '{date_range_str}': {str(e)}")
        
        return None, None


class StreamflowTarget(CalibrationTarget):
    """Streamflow calibration target"""
    
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Get SUMMA timestep files or mizuRoute output files"""
        # First try mizuRoute files if in mizuRoute directory
        if 'mizuRoute' in str(sim_dir):
            mizu_files = list(sim_dir.glob("*.nc"))
            if mizu_files:
                return mizu_files
        
        # Otherwise look for SUMMA timestep files
        return list(sim_dir.glob("*timestep.nc"))
    
    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract streamflow data from simulation files"""
        sim_file = sim_files[0]  # Use first file
        
        try:
            # Determine if this is mizuRoute or SUMMA output
            if self._is_mizuroute_output(sim_file):
                return self._extract_mizuroute_streamflow(sim_file)
            else:
                return self._extract_summa_streamflow(sim_file)
        except Exception as e:
            self.logger.error(f"Error extracting streamflow data from {sim_file}: {str(e)}")
            raise
    
    def _is_mizuroute_output(self, sim_file: Path) -> bool:
        """Check if file is mizuRoute output based on variables"""
        try:
            with xr.open_dataset(sim_file) as ds:
                mizuroute_vars = ['IRFroutedRunoff', 'KWTroutedRunoff', 'reachID']
                return any(var in ds.variables for var in mizuroute_vars)
        except:
            return False
    
    def _extract_mizuroute_streamflow(self, sim_file: Path) -> pd.Series:
        """Extract streamflow from mizuRoute output"""
        with xr.open_dataset(sim_file) as ds:
            reach_id = int(self.config.get('SIM_REACH_ID'))
            
            # Find reach index
            if 'reachID' not in ds.variables:
                raise ValueError("reachID variable not found in mizuRoute output")
            
            reach_ids = ds['reachID'].values
            reach_indices = np.where(reach_ids == reach_id)[0]
            
            if len(reach_indices) == 0:
                raise ValueError(f"Reach ID {reach_id} not found in mizuRoute output")
            
            reach_index = reach_indices[0]
            
            # Find streamflow variable
            streamflow_vars = ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff']
            for var_name in streamflow_vars:
                if var_name in ds.variables:
                    var = ds[var_name]
                    if 'seg' in var.dims:
                        return var.isel(seg=reach_index).to_pandas()
                    elif 'reachID' in var.dims:
                        return var.isel(reachID=reach_index).to_pandas()
            
            raise ValueError("No suitable streamflow variable found in mizuRoute output")
    
    def _extract_summa_streamflow(self, sim_file: Path) -> pd.Series:
        """Extract streamflow from SUMMA output"""
        with xr.open_dataset(sim_file) as ds:
            # Find streamflow variable
            streamflow_vars = ['averageRoutedRunoff', 'basin__TotalRunoff', 'scalarTotalRunoff']
            
            for var_name in streamflow_vars:
                if var_name in ds.variables:
                    var = ds[var_name]
                    
                    # Extract data based on dimensions
                    if len(var.shape) > 1:
                        if 'hru' in var.dims:
                            sim_data = var.isel(hru=0).to_pandas()
                        elif 'gru' in var.dims:
                            sim_data = var.isel(gru=0).to_pandas()
                        else:
                            # Take first spatial index
                            non_time_dims = [dim for dim in var.dims if dim != 'time']
                            if non_time_dims:
                                sim_data = var.isel({non_time_dims[0]: 0}).to_pandas()
                            else:
                                sim_data = var.to_pandas()
                    else:
                        sim_data = var.to_pandas()
                    
                    # Convert units (m/s to m³/s) for SUMMA output
                    catchment_area = self._get_catchment_area()
                    return sim_data * catchment_area
            
            raise ValueError("No suitable streamflow variable found in SUMMA output")
    
    def _get_catchment_area(self) -> float:
        """Get catchment area for unit conversion"""
        try:
            import geopandas as gpd
            
            # Try basin shapefile first
            basin_path = self.project_dir / "shapefiles" / "river_basins"
            basin_files = list(basin_path.glob("*.shp"))
            
            if basin_files:
                gdf = gpd.read_file(basin_files[0])
                area_col = self.config.get('RIVER_BASIN_SHP_AREA', 'GRU_area')
                
                if area_col in gdf.columns:
                    total_area = gdf[area_col].sum()
                    if 0 < total_area < 1e12:  # Reasonable area
                        return total_area
            
            # Fallback: calculate from geometry
            if basin_files:
                gdf = gpd.read_file(basin_files[0])
                if gdf.crs and gdf.crs.is_geographic:
                    # Reproject to UTM for area calculation
                    centroid = gdf.dissolve().centroid.iloc[0]
                    utm_zone = int(((centroid.x + 180) / 6) % 60) + 1
                    utm_crs = f"+proj=utm +zone={utm_zone} +north +datum=WGS84 +units=m +no_defs"
                    gdf = gdf.to_crs(utm_crs)
                
                return gdf.geometry.area.sum()
            
        except Exception as e:
            self.logger.warning(f"Could not calculate catchment area: {str(e)}")
        
        # Default fallback
        return 1e6  # 1 km²
    
    def get_observed_data_path(self) -> Path:
        """Get path to observed streamflow data"""
        obs_path = self.config.get('OBSERVATIONS_PATH')
        if obs_path == 'default' or not obs_path:
            return self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.domain_name}_streamflow_processed.csv"
        return Path(obs_path)
    
    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        """Find streamflow data column"""
        for col in columns:
            if any(term in col.lower() for term in ['flow', 'discharge', 'q_', 'streamflow']):
                return col
        return None
    
    def needs_routing(self) -> bool:
        """Check if streamflow calibration needs mizuRoute routing"""
        domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
        routing_delineation = self.config.get('ROUTING_DELINEATION', 'lumped')
        
        # Distributed domain always needs routing
        if domain_method not in ['point', 'lumped']:
            return True
        
        # Lumped domain with river network routing
        if domain_method == 'lumped' and routing_delineation == 'river_network':
            return True
        
        return False


class SnowTarget(CalibrationTarget):
    """Snow Water Equivalent (SWE) calibration target"""
    
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Get SUMMA daily output files for SWE"""
        return list(sim_dir.glob("*day.nc"))
    
    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract SWE data from SUMMA daily output"""
        sim_file = sim_files[0]
        
        try:
            with xr.open_dataset(sim_file) as ds:
                if 'scalarSWE' not in ds.variables:
                    raise ValueError("scalarSWE variable not found in daily output")
                
                var = ds['scalarSWE']
                
                # Extract data based on dimensions
                if len(var.shape) > 1:
                    if 'hru' in var.dims:
                        sim_data = var.isel(hru=0).to_pandas()
                    elif 'gru' in var.dims:
                        sim_data = var.isel(gru=0).to_pandas()
                    else:
                        # Take first spatial index
                        non_time_dims = [dim for dim in var.dims if dim != 'time']
                        if non_time_dims:
                            sim_data = var.isel({non_time_dims[0]: 0}).to_pandas()
                        else:
                            sim_data = var.to_pandas()
                else:
                    sim_data = var.to_pandas()
                
                # No unit conversion needed - SWE is already in mm
                return sim_data
                
        except Exception as e:
            self.logger.error(f"Error extracting SWE data from {sim_file}: {str(e)}")
            raise
    
    def get_observed_data_path(self) -> Path:
        """Get path to observed SWE data"""
        return self.project_dir / "observations" / "snow" / "swe" / "processed" / f"{self.domain_name}_swe_processed.csv"
    
    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        """Find SWE data column"""
        for col in columns:
            if any(term in col.lower() for term in ['swe', 'snow', 'water_equivalent']):
                return col
        return None
    
    def needs_routing(self) -> bool:
        """Snow calibration never needs routing"""
        return False


# ============= PARAMETER MANAGEMENT =============

class ParameterManager:
    """Handles parameter bounds, normalization, file generation, and soil depth calculations"""
    
    def __init__(self, config: Dict, logger: logging.Logger, optimization_settings_dir: Path):
        self.config = config
        self.logger = logger
        self.optimization_settings_dir = optimization_settings_dir
        
        # Parse parameter lists
        self.local_params = [p.strip() for p in config.get('PARAMS_TO_CALIBRATE', '').split(',') if p.strip()]
        self.basin_params = [p.strip() for p in config.get('BASIN_PARAMS_TO_CALIBRATE', '').split(',') if p.strip()]
        self.depth_params = ['total_mult', 'shape_factor'] if config.get('CALIBRATE_DEPTH', False) else []
        self.mizuroute_params = []
        
        if config.get('CALIBRATE_MIZUROUTE', False):
            mizuroute_params_str = config.get('MIZUROUTE_PARAMS_TO_CALIBRATE', 'velo,diff')
            self.mizuroute_params = [p.strip() for p in mizuroute_params_str.split(',') if p.strip()]
        
        # Load parameter bounds
        self.param_bounds = self._parse_all_bounds()
        
        # Load original soil depths if depth calibration enabled
        self.original_depths = None
        if self.depth_params:
            self.original_depths = self._load_original_depths()
        
        # Get attribute file path
        self.attr_file_path = self.optimization_settings_dir / config.get('SETTINGS_SUMMA_ATTRIBUTES', 'attributes.nc')
    
    @property
    def all_param_names(self) -> List[str]:
        """Get list of all parameter names"""
        return self.local_params + self.basin_params + self.depth_params + self.mizuroute_params
    
    def get_initial_parameters(self) -> Optional[Dict[str, np.ndarray]]:
        """Get initial parameter values from existing files or defaults"""
        # Try to load existing optimized parameters
        existing_params = self._load_existing_optimized_parameters()
        if existing_params:
            self.logger.info("Loaded existing optimized parameters")
            return existing_params
        
        # Extract parameters from model files
        self.logger.info("Extracting initial parameters from default values")
        return self._extract_default_parameters()
    
    def normalize_parameters(self, params: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert parameter dictionary to normalized array [0,1]"""
        normalized = np.zeros(len(self.all_param_names))
        
        for i, param_name in enumerate(self.all_param_names):
            if param_name in params and param_name in self.param_bounds:
                bounds = self.param_bounds[param_name]
                
                # Get parameter value
                param_values = params[param_name]
                if isinstance(param_values, np.ndarray) and len(param_values) > 1:
                    value = np.mean(param_values)  # Use mean for multi-value parameters
                else:
                    value = param_values[0] if isinstance(param_values, np.ndarray) else param_values
                
                # Normalize to [0,1]
                normalized[i] = (value - bounds['min']) / (bounds['max'] - bounds['min'])
        
        return np.clip(normalized, 0, 1)
    
    def denormalize_parameters(self, normalized_array: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert normalized array back to parameter dictionary"""
        params = {}
        
        for i, param_name in enumerate(self.all_param_names):
            if param_name in self.param_bounds:
                bounds = self.param_bounds[param_name]
                denorm_value = bounds['min'] + normalized_array[i] * (bounds['max'] - bounds['min'])
                
                # Validate bounds
                denorm_value = np.clip(denorm_value, bounds['min'], bounds['max'])
                
                # Format based on parameter type
                if param_name in self.depth_params:
                    params[param_name] = np.array([denorm_value])
                elif param_name in self.mizuroute_params:
                    params[param_name] = denorm_value
                elif param_name in self.basin_params:
                    params[param_name] = np.array([denorm_value])
                else:
                    # Local parameters - expand to HRU count
                    params[param_name] = self._expand_to_hru_count(denorm_value)
        
        return params
    
    def apply_parameters(self, params: Dict[str, np.ndarray]) -> bool:
        """Apply parameters to model files"""
        try:
            # Update soil depths if depth calibration enabled
            if self.depth_params and 'total_mult' in params and 'shape_factor' in params:
                if not self._update_soil_depths(params):
                    return False
            
            # Update mizuRoute parameters if enabled
            if self.mizuroute_params:
                if not self._update_mizuroute_parameters(params):
                    return False
            
            # Generate trial parameters file (excluding depth and mizuRoute parameters)
            hydraulic_params = {k: v for k, v in params.items() 
                              if k not in self.depth_params + self.mizuroute_params}
            
            if hydraulic_params:
                if not self._generate_trial_params_file(hydraulic_params):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying parameters: {str(e)}")
            return False
    
    def _parse_all_bounds(self) -> Dict[str, Dict[str, float]]:
        """Parse parameter bounds from all parameter info files"""
        bounds = {}
        
        # Parse local parameter bounds
        if self.local_params:
            local_param_file = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}" / 'settings' / 'SUMMA' / 'localParamInfo.txt'
            local_bounds = self._parse_param_info_file(local_param_file, self.local_params)
            bounds.update(local_bounds)
        
        # Parse basin parameter bounds
        if self.basin_params:
            basin_param_file = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}" / 'settings' / 'SUMMA' / 'basinParamInfo.txt'
            basin_bounds = self._parse_param_info_file(basin_param_file, self.basin_params)
            bounds.update(basin_bounds)
        
        # Add depth parameter bounds
        if self.depth_params:
            bounds['total_mult'] = {'min': 0.1, 'max': 5.0}
            bounds['shape_factor'] = {'min': 0.1, 'max': 3.0}
        
        # Add mizuRoute parameter bounds
        if self.mizuroute_params:
            mizuroute_bounds = self._get_mizuroute_bounds()
            bounds.update(mizuroute_bounds)
        
        return bounds
    
    def _parse_param_info_file(self, file_path: Path, param_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Parse parameter bounds from a SUMMA parameter info file"""
        bounds = {}
        
        if not file_path.exists():
            self.logger.error(f"Parameter file not found: {file_path}")
            return bounds
        
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('!') or line.startswith("'"):
                        continue
                    
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) < 4:
                        continue
                    
                    param_name = parts[0]
                    if param_name in param_names:
                        try:
                            min_val = float(parts[2].replace('d','e').replace('D','e'))
                            max_val = float(parts[3].replace('d','e').replace('D','e'))
                            
                            if min_val > max_val:
                                min_val, max_val = max_val, min_val
                            
                            if min_val == max_val:
                                range_val = abs(min_val) * 0.1 if min_val != 0 else 0.1
                                min_val -= range_val
                                max_val += range_val
                            
                            bounds[param_name] = {'min': min_val, 'max': max_val}
                            
                        except ValueError as e:
                            self.logger.error(f"Could not parse bounds for {param_name}: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error reading parameter file {file_path}: {str(e)}")
        
        return bounds
    
    def _get_mizuroute_bounds(self) -> Dict[str, Dict[str, float]]:
        """Get parameter bounds for mizuRoute parameters"""
        default_bounds = {
            'velo': {'min': 0.1, 'max': 5.0},
            'diff': {'min': 100.0, 'max': 5000.0},
            'mann_n': {'min': 0.01, 'max': 0.1},
            'wscale': {'min': 0.0001, 'max': 0.01},
            'fshape': {'min': 1.0, 'max': 5.0},
            'tscale': {'min': 3600, 'max': 172800}
        }
        
        bounds = {}
        for param in self.mizuroute_params:
            if param in default_bounds:
                bounds[param] = default_bounds[param]
            else:
                self.logger.warning(f"Unknown mizuRoute parameter: {param}")
        
        return bounds
    
    def _load_existing_optimized_parameters(self) -> Optional[Dict[str, np.ndarray]]:
        """Load existing optimized parameters from default settings"""
        trial_params_path = self.config.get('CONFLUENCE_DATA_DIR')
        if trial_params_path == 'default':
            return None
        
        # Implementation would check for existing trialParams.nc file
        # For brevity, returning None - full implementation would load existing params
        return None
    
    def _extract_default_parameters(self) -> Dict[str, np.ndarray]:
        """Extract default parameter values from parameter info files"""
        defaults = {}
        
        # Parse local parameters
        if self.local_params:
            local_defaults = self._parse_defaults_from_file(
                self.optimization_settings_dir / 'localParamInfo.txt', 
                self.local_params
            )
            defaults.update(local_defaults)
        
        # Parse basin parameters
        if self.basin_params:
            basin_defaults = self._parse_defaults_from_file(
                self.optimization_settings_dir / 'basinParamInfo.txt',
                self.basin_params
            )
            defaults.update(basin_defaults)
        
        # Add depth parameters
        if self.depth_params:
            defaults['total_mult'] = np.array([1.0])
            defaults['shape_factor'] = np.array([1.0])
        
        # Add mizuRoute parameters
        if self.mizuroute_params:
            for param in self.mizuroute_params:
                defaults[param] = self._get_default_mizuroute_value(param)
        
        # Expand to HRU count
        return self._expand_defaults_to_hru_count(defaults)
    
    def _parse_defaults_from_file(self, file_path: Path, param_names: List[str]) -> Dict[str, np.ndarray]:
        """Parse default values from parameter info file"""
        defaults = {}
        
        if not file_path.exists():
            return defaults
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('!') or line.startswith("'"):
                        continue
                    
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 4:
                        param_name = parts[0]
                        if param_name in param_names:
                            try:
                                default_val = float(parts[1].replace('d','e').replace('D','e'))
                                defaults[param_name] = np.array([default_val])
                            except ValueError:
                                continue
        except Exception as e:
            self.logger.error(f"Error parsing defaults from {file_path}: {str(e)}")
        
        return defaults
    
    def _get_default_mizuroute_value(self, param_name: str) -> float:
        """Get default value for mizuRoute parameter"""
        defaults = {
            'velo': 1.0,
            'diff': 1000.0,
            'mann_n': 0.025,
            'wscale': 0.001,
            'fshape': 2.5,
            'tscale': 86400
        }
        return defaults.get(param_name, 1.0)
    
    def _expand_defaults_to_hru_count(self, defaults: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Expand parameter defaults to match HRU count"""
        try:
            # Get HRU count from attributes file
            with xr.open_dataset(self.attr_file_path) as ds:
                num_hrus = ds.sizes.get('hru', 1)
            
            expanded_defaults = {}
            routing_params = ['routingGammaShape', 'routingGammaScale']
            
            for param_name, values in defaults.items():
                if param_name in self.basin_params or param_name in routing_params:
                    expanded_defaults[param_name] = values
                elif param_name in self.depth_params or param_name in self.mizuroute_params:
                    expanded_defaults[param_name] = values
                else:
                    expanded_defaults[param_name] = np.full(num_hrus, values[0])
            
            return expanded_defaults
            
        except Exception as e:
            self.logger.error(f"Error expanding defaults: {str(e)}")
            return defaults
    
    def _expand_to_hru_count(self, value: float) -> np.ndarray:
        """Expand single value to HRU count"""
        try:
            with xr.open_dataset(self.attr_file_path) as ds:
                num_hrus = ds.sizes.get('hru', 1)
            return np.full(num_hrus, value)
        except:
            return np.array([value])
    
    def _load_original_depths(self) -> Optional[np.ndarray]:
        """Load original soil depths from coldState.nc"""
        try:
            coldstate_path = self.optimization_settings_dir / self.config.get('SETTINGS_SUMMA_COLDSTATE', 'coldState.nc')
            
            with nc.Dataset(coldstate_path, 'r') as ds:
                if 'mLayerDepth' in ds.variables:
                    return ds.variables['mLayerDepth'][:, 0].copy()
            
        except Exception as e:
            self.logger.error(f"Error loading original depths: {str(e)}")
        
        return None
    
    def _update_soil_depths(self, params: Dict[str, np.ndarray]) -> bool:
        """Update soil depths in coldState.nc"""
        if self.original_depths is None:
            return False
        
        try:
            total_mult = params['total_mult'][0] if isinstance(params['total_mult'], np.ndarray) else params['total_mult']
            shape_factor = params['shape_factor'][0] if isinstance(params['shape_factor'], np.ndarray) else params['shape_factor']
            
            # Calculate new depths using shape method
            new_depths = self._calculate_new_depths(total_mult, shape_factor)
            if new_depths is None:
                return False
            
            # Calculate layer heights
            heights = np.zeros(len(new_depths) + 1)
            for i in range(len(new_depths)):
                heights[i + 1] = heights[i] + new_depths[i]
            
            # Update coldState.nc
            coldstate_path = self.optimization_settings_dir / self.config.get('SETTINGS_SUMMA_COLDSTATE', 'coldState.nc')
            
            with nc.Dataset(coldstate_path, 'r+') as ds:
                if 'mLayerDepth' not in ds.variables or 'iLayerHeight' not in ds.variables:
                    return False
                
                num_hrus = ds.dimensions['hru'].size
                for h in range(num_hrus):
                    ds.variables['mLayerDepth'][:, h] = new_depths
                    ds.variables['iLayerHeight'][:, h] = heights
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating soil depths: {str(e)}")
            return False
    
    def _calculate_new_depths(self, total_mult: float, shape_factor: float) -> Optional[np.ndarray]:
        """Calculate new soil depths using shape method"""
        if self.original_depths is None:
            return None
        
        arr = self.original_depths.copy()
        n = len(arr)
        idx = np.arange(n)
        
        # Calculate shape weights
        if shape_factor > 1:
            w = np.exp(idx / (n - 1) * np.log(shape_factor))
        elif shape_factor < 1:
            w = np.exp((n - 1 - idx) / (n - 1) * np.log(1 / shape_factor))
        else:
            w = np.ones(n)
        
        # Normalize weights
        w /= w.mean()
        
        # Apply multipliers
        new_depths = arr * w * total_mult
        
        return new_depths
    
    def _update_mizuroute_parameters(self, params: Dict) -> bool:
        """Update mizuRoute parameters in param.nml.default"""
        try:
            mizuroute_settings_dir = self.optimization_settings_dir.parent / "mizuRoute"
            param_file = mizuroute_settings_dir / "param.nml.default"
            
            if not param_file.exists():
                return True  # Skip if file doesn't exist
            
            # Read file
            with open(param_file, 'r') as f:
                content = f.read()
            
            # Update parameters
            updated_content = content
            for param_name in self.mizuroute_params:
                if param_name in params:
                    param_value = params[param_name]
                    pattern = rf'(\s+{param_name}\s*=\s*)[0-9.-]+'
                    
                    if param_name in ['tscale']:
                        replacement = rf'\g<1>{int(param_value)}'
                    else:
                        replacement = rf'\g<1>{param_value:.6f}'
                    
                    updated_content = re.sub(pattern, replacement, updated_content)
            
            # Write updated file
            with open(param_file, 'w') as f:
                f.write(updated_content)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating mizuRoute parameters: {str(e)}")
            return False
    
    def _generate_trial_params_file(self, params: Dict[str, np.ndarray]) -> bool:
        """Generate trialParams.nc file with proper dimensions"""
        try:
            trial_params_path = self.optimization_settings_dir / self.config.get('SETTINGS_SUMMA_TRIALPARAMS', 'trialParams.nc')
            
            # Get HRU and GRU counts from attributes
            with xr.open_dataset(self.attr_file_path) as ds:
                num_hrus = ds.sizes.get('hru', 1)
                num_grus = ds.sizes.get('gru', 1)
            
            # Define parameter levels
            routing_params = ['routingGammaShape', 'routingGammaScale']
            
            with nc.Dataset(trial_params_path, 'w', format='NETCDF4') as output_ds:
                # Create dimensions
                output_ds.createDimension('hru', num_hrus)
                output_ds.createDimension('gru', num_grus)
                
                # Create coordinate variables
                hru_var = output_ds.createVariable('hruId', 'i4', ('hru',), fill_value=-9999)
                hru_var[:] = range(1, num_hrus + 1)
                
                gru_var = output_ds.createVariable('gruId', 'i4', ('gru',), fill_value=-9999)
                gru_var[:] = range(1, num_grus + 1)
                
                # Add parameters
                for param_name, param_values in params.items():
                    param_values_array = np.asarray(param_values)
                    
                    if param_name in routing_params or param_name in self.basin_params:
                        # GRU-level parameters
                        param_var = output_ds.createVariable(param_name, 'f8', ('gru',), fill_value=np.nan)
                        param_var.long_name = f"Trial value for {param_name}"
                        
                        if len(param_values_array) == 1:
                            param_var[:] = param_values_array[0]
                        else:
                            param_var[:] = param_values_array[:num_grus]
                    else:
                        # HRU-level parameters
                        param_var = output_ds.createVariable(param_name, 'f8', ('hru',), fill_value=np.nan)
                        param_var.long_name = f"Trial value for {param_name}"
                        
                        if len(param_values_array) == num_hrus:
                            param_var[:] = param_values_array
                        elif len(param_values_array) == 1:
                            param_var[:] = param_values_array[0]
                        else:
                            param_var[:] = param_values_array[:num_hrus]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating trial params file: {str(e)}")
            return False


# ============= MODEL EXECUTION =============

class ModelExecutor:
    """Handles SUMMA and mizuRoute execution with routing support"""
    
    def __init__(self, config: Dict, logger: logging.Logger, calibration_target: CalibrationTarget):
        self.config = config
        self.logger = logger
        self.calibration_target = calibration_target
    
    def run_models(self, summa_dir: Path, mizuroute_dir: Path, settings_dir: Path,
                  mizuroute_settings_dir: Optional[Path] = None) -> bool:
        """Run SUMMA and mizuRoute if needed"""
        try:
            # Run SUMMA
            if not self._run_summa(settings_dir, summa_dir):
                return False
            
            # Run mizuRoute if needed
            if self.calibration_target.needs_routing():
                if mizuroute_settings_dir is None:
                    mizuroute_settings_dir = settings_dir.parent / "mizuRoute"
                
                # Handle lumped-to-distributed conversion if needed
                domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
                routing_delineation = self.config.get('ROUTING_DELINEATION', 'lumped')
                
                if domain_method == 'lumped' and routing_delineation == 'river_network':
                    if not self._convert_lumped_to_distributed(summa_dir, mizuroute_settings_dir):
                        return False
                
                if not self._run_mizuroute(mizuroute_settings_dir, mizuroute_dir):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error running models: {str(e)}")
            return False
    
    def _run_summa(self, settings_dir: Path, output_dir: Path) -> bool:
        """Run SUMMA simulation"""
        try:
            # Get SUMMA executable
            summa_path = self.config.get('SUMMA_INSTALL_PATH')
            if summa_path == 'default':
                summa_path = Path(self.config.get('CONFLUENCE_DATA_DIR')) / 'installs' / 'summa' / 'bin'
            else:
                summa_path = Path(summa_path)
            
            summa_exe = summa_path / self.config.get('SUMMA_EXE')
            file_manager = settings_dir / self.config.get('SETTINGS_SUMMA_FILEMANAGER', 'fileManager.txt')
            
            if not summa_exe.exists():
                self.logger.error(f"SUMMA executable not found: {summa_exe}")
                return False
            
            if not file_manager.exists():
                self.logger.error(f"File manager not found: {file_manager}")
                return False
            
            # Create log directory
            log_dir = output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Run SUMMA
            cmd = f"{summa_exe} -m {file_manager}"
            log_file = log_dir / f"summa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            with open(log_file, 'w') as f:
                result = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, 
                                      check=True, timeout=1800)
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"SUMMA simulation failed with exit code {e.returncode}")
            return False
        except subprocess.TimeoutExpired:
            self.logger.error("SUMMA simulation timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error running SUMMA: {str(e)}")
            return False
    
    def _run_mizuroute(self, settings_dir: Path, output_dir: Path) -> bool:
        """Run mizuRoute simulation"""
        try:
            # Get mizuRoute executable
            mizu_path = self.config.get('INSTALL_PATH_MIZUROUTE')
            if mizu_path == 'default':
                mizu_path = Path(self.config.get('CONFLUENCE_DATA_DIR')) / 'installs' / 'mizuRoute' / 'route' / 'bin'
            else:
                mizu_path = Path(mizu_path)
            
            mizu_exe = mizu_path / self.config.get('EXE_NAME_MIZUROUTE', 'mizuroute.exe')
            control_file = settings_dir / self.config.get('SETTINGS_MIZU_CONTROL_FILE', 'mizuroute.control')
            
            if not mizu_exe.exists():
                self.logger.error(f"mizuRoute executable not found: {mizu_exe}")
                return False
            
            if not control_file.exists():
                self.logger.error(f"mizuRoute control file not found: {control_file}")
                return False
            
            # Create log directory
            log_dir = output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Run mizuRoute
            cmd = f"{mizu_exe} {control_file}"
            log_file = log_dir / f"mizuroute_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
            with open(log_file, 'w') as f:
                result = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, 
                                      check=True, timeout=1800, cwd=str(settings_dir))
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"mizuRoute simulation failed with exit code {e.returncode}")
            return False
        except subprocess.TimeoutExpired:
            self.logger.error("mizuRoute simulation timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error running mizuRoute: {str(e)}")
            return False
    
    def _convert_lumped_to_distributed(self, summa_dir: Path, mizuroute_settings_dir: Path) -> bool:
        """Convert lumped SUMMA output for distributed routing"""
        try:
            # Find SUMMA timestep file
            timestep_files = list(summa_dir.glob("*timestep.nc"))
            if not timestep_files:
                return False
            
            summa_file = timestep_files[0]
            
            # Load topology to get segment information
            topology_file = mizuroute_settings_dir / self.config.get('SETTINGS_MIZU_TOPOLOGY', 'topology.nc')
            if not topology_file.exists():
                return False
            
            with xr.open_dataset(topology_file) as topo_ds:
                seg_ids = topo_ds['segId'].values
                n_segments = len(seg_ids)
            
            # Load and convert SUMMA output
            with xr.open_dataset(summa_file, decode_times=False) as summa_ds:
                # Find routing variable
                routing_var = self.config.get('SETTINGS_MIZU_ROUTING_VAR', 'averageRoutedRunoff')
                if routing_var not in summa_ds:
                    routing_var = 'basin__TotalRunoff'
                
                if routing_var not in summa_ds:
                    return False
                
                # Create mizuRoute forcing dataset
                mizuForcing = xr.Dataset()
                
                # Copy time coordinate
                mizuForcing['time'] = summa_ds['time']
                
                # Create GRU dimension
                mizuForcing['gru'] = xr.DataArray(seg_ids, dims=('gru',))
                mizuForcing['gruId'] = xr.DataArray(seg_ids, dims=('gru',))
                
                # Get and broadcast runoff data
                runoff_data = summa_ds[routing_var].values
                if len(runoff_data.shape) == 2:
                    runoff_data = runoff_data[:, 0] if runoff_data.shape[1] > 0 else runoff_data.flatten()
                
                # Broadcast to all segments
                tiled_data = np.tile(runoff_data[:, np.newaxis], (1, n_segments))
                
                mizuForcing['averageRoutedRunoff'] = xr.DataArray(
                    tiled_data, dims=('time', 'gru'),
                    attrs={'long_name': 'Broadcast runoff for distributed routing', 'units': 'm/s'}
                )
                
                # Copy global attributes
                mizuForcing.attrs.update(summa_ds.attrs)
            
            # Save converted file
            mizuForcing.to_netcdf(summa_file, format='NETCDF4')
            mizuForcing.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error converting lumped to distributed: {str(e)}")
            return False


# ============= RESULTS MANAGEMENT =============

class ResultsManager:
    """Handles optimization results, history tracking, and visualization"""
    
    def __init__(self, config: Dict, logger: logging.Logger, output_dir: Path):
        self.config = config
        self.logger = logger
        self.output_dir = output_dir
        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')
    
    def save_results(self, best_params: Dict, best_score: float, history: List[Dict], 
                    final_result: Optional[Dict] = None) -> bool:
        """Save optimization results to files"""
        try:
            # Save best parameters to CSV
            self._save_best_parameters_csv(best_params)
            
            # Save history to CSV
            self._save_history_csv(history)
            
            # Save metadata
            self._save_metadata(best_score, len(history), final_result)
            
            # Create visualization plots
            self._create_plots(history, best_params)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            return False
    
    def _save_best_parameters_csv(self, best_params: Dict) -> None:
        """Save best parameters to CSV file"""
        param_data = []
        
        for param_name, values in best_params.items():
            if isinstance(values, np.ndarray):
                if len(values) == 1:
                    param_data.append({
                        'parameter': param_name,
                        'value': values[0],
                        'type': 'scalar'
                    })
                else:
                    param_data.append({
                        'parameter': param_name,
                        'value': np.mean(values),
                        'type': 'array_mean',
                        'min': np.min(values),
                        'max': np.max(values),
                        'std': np.std(values)
                    })
            else:
                param_data.append({
                    'parameter': param_name,
                    'value': values,
                    'type': 'scalar'
                })
        
        param_df = pd.DataFrame(param_data)
        param_csv_path = self.output_dir / "best_parameters.csv"
        param_df.to_csv(param_csv_path, index=False)
        
        self.logger.info(f"Saved best parameters to: {param_csv_path}")
    
    def _save_history_csv(self, history: List[Dict]) -> None:
        """Save optimization history to CSV"""
        if not history:
            return
        
        history_data = []
        for gen_data in history:
            row = {
                'generation': gen_data.get('generation', 0),
                'best_score': gen_data.get('best_score'),
                'mean_score': gen_data.get('mean_score'),
                'std_score': gen_data.get('std_score'),
                'valid_individuals': gen_data.get('valid_individuals', 0)
            }
            
            # Add best parameters if available
            if gen_data.get('best_params'):
                for param_name, values in gen_data['best_params'].items():
                    if isinstance(values, np.ndarray):
                        row[f'best_{param_name}'] = np.mean(values) if len(values) > 1 else values[0]
                    else:
                        row[f'best_{param_name}'] = values
            
            history_data.append(row)
        
        history_df = pd.DataFrame(history_data)
        history_csv_path = self.output_dir / "optimization_history.csv"
        history_df.to_csv(history_csv_path, index=False)
        
        self.logger.info(f"Saved optimization history to: {history_csv_path}")
    
    def _save_metadata(self, best_score: float, num_generations: int, final_result: Optional[Dict]) -> None:
        """Save optimization metadata"""
        metadata = {
            'algorithm': 'Differential Evolution',
            'domain_name': self.domain_name,
            'experiment_id': self.experiment_id,
            'calibration_variable': self.config.get('CALIBRATION_VARIABLE', 'streamflow'),
            'target_metric': self.config.get('OPTIMIZATION_METRIC', 'KGE'),
            'best_score': best_score,
            'num_generations': num_generations,
            'population_size': self.config.get('POPULATION_SIZE', 50),
            'F': self.config.get('DE_SCALING_FACTOR', 0.5),
            'CR': self.config.get('DE_CROSSOVER_RATE', 0.9),
            'parallel_processes': self.config.get('MPI_PROCESSES', 1),
            'completed_at': datetime.now().isoformat()
        }
        
        if final_result:
            metadata.update(final_result)
        
        metadata_df = pd.DataFrame([metadata])
        metadata_csv_path = self.output_dir / "optimization_metadata.csv"
        metadata_df.to_csv(metadata_csv_path, index=False)
        
        self.logger.info(f"Saved metadata to: {metadata_csv_path}")
    
    def _create_plots(self, history: List[Dict], best_params: Dict) -> None:
        """Create optimization progress plots"""
        try:
            import matplotlib.pyplot as plt
            
            plots_dir = self.output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract progress data
            generations = [h['generation'] for h in history]
            best_scores = [h['best_score'] for h in history if h.get('best_score') is not None]
            
            if not best_scores:
                return
            
            # Progress plot
            plt.figure(figsize=(12, 6))
            plt.plot(generations[:len(best_scores)], best_scores, 'b-o', markersize=4)
            plt.xlabel('Generation')
            plt.ylabel(f"Performance ({self.config.get('OPTIMIZATION_METRIC', 'KGE')})")
            plt.title(f'DE Optimization Progress - {self.config.get("CALIBRATION_VARIABLE", "streamflow").title()} Calibration')
            plt.grid(True, alpha=0.3)
            
            # Mark best
            best_idx = np.nanargmax(best_scores)
            plt.plot(generations[best_idx], best_scores[best_idx], 'ro', markersize=10,
                    label=f'Best: {best_scores[best_idx]:.4f} at generation {generations[best_idx]}')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(plots_dir / "optimization_progress.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Parameter evolution plots for depth parameters
            if self.config.get('CALIBRATE_DEPTH', False):
                self._create_depth_parameter_plots(history, plots_dir)
            
            self.logger.info("Created optimization plots")
            
        except Exception as e:
            self.logger.error(f"Error creating plots: {str(e)}")
    
    def _create_depth_parameter_plots(self, history: List[Dict], plots_dir: Path) -> None:
        """Create depth parameter evolution plots"""
        try:
            import matplotlib.pyplot as plt
            
            # Extract depth parameters
            generations = []
            total_mults = []
            shape_factors = []
            
            for h in history:
                if h.get('best_params') and 'total_mult' in h['best_params'] and 'shape_factor' in h['best_params']:
                    generations.append(h['generation'])
                    
                    tm = h['best_params']['total_mult']
                    sf = h['best_params']['shape_factor']
                    
                    tm_val = tm[0] if isinstance(tm, np.ndarray) and len(tm) > 0 else tm
                    sf_val = sf[0] if isinstance(sf, np.ndarray) and len(sf) > 0 else sf
                    
                    total_mults.append(tm_val)
                    shape_factors.append(sf_val)
            
            if not generations:
                return
            
            # Create subplot figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Total multiplier plot
            ax1.plot(generations, total_mults, 'g-o', markersize=4)
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Total Depth Multiplier')
            ax1.set_title('Soil Depth Total Multiplier Evolution')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No change (1.0)')
            ax1.legend()
            
            # Shape factor plot
            ax2.plot(generations, shape_factors, 'm-o', markersize=4)
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Shape Factor')
            ax2.set_title('Soil Depth Shape Factor Evolution')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Uniform scaling (1.0)')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(plots_dir / "depth_parameter_evolution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating depth parameter plots: {str(e)}")


# ============= MAIN OPTIMIZER CLASSES =============

class BaseOptimizer(ABC):
    """
    Abstract base class for CONFLUENCE optimizers.
    
    Contains common functionality shared between different optimization algorithms
    like DDS and DE. Handles setup, parameter management, model execution, and
    results management.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """Initialize base optimizer with common components"""
        self.config = config
        self.logger = logger
        
        # Setup basic paths
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = self.config.get('EXPERIMENT_ID')
        
        # Algorithm-specific directory setup
        self.algorithm_name = self.get_algorithm_name().lower()
        self.optimization_dir = self.project_dir / "simulations" / f"run_{self.algorithm_name}"
        self.summa_sim_dir = self.optimization_dir / "SUMMA"
        self.mizuroute_sim_dir = self.optimization_dir / "mizuRoute"
        self.optimization_settings_dir = self.optimization_dir / "settings" / "SUMMA"
        self.output_dir = self.project_dir / "optimisation" / f"{self.algorithm_name}_{self.experiment_id}"
        
        # Initialize component managers
        self.parameter_manager = ParameterManager(config, logger, self.optimization_settings_dir)
        self._setup_optimization_directories()
        
        self.calibration_target = self._create_calibration_target()
        self.model_executor = ModelExecutor(config, logger, self.calibration_target)
        self.results_manager = ResultsManager(config, logger, self.output_dir)
        
        # Common algorithm parameters
        self.max_iterations = config.get('NUMBER_OF_ITERATIONS', 100)
        self.target_metric = config.get('OPTIMIZATION_METRIC', 'KGE')
        
        # Algorithm state variables
        self.best_params = None
        self.best_score = float('-inf')
        self.iteration_history = []
        
        # Parallel processing setup
        self.use_parallel = config.get('MPI_PROCESSES', 1) > 1
        self.num_processes = max(1, config.get('MPI_PROCESSES', 1))
        self.parallel_dirs = []
        self._consecutive_parallel_failures = 0
        
        if self.use_parallel:
            self._setup_parallel_processing()
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Return the name of the optimization algorithm"""
        pass
    
    @abstractmethod
    def _run_algorithm(self) -> Tuple[Dict, float, List]:
        """Run the specific optimization algorithm"""
        pass
    
    def _setup_optimization_directories(self) -> None:
        """Setup directory structure for optimization"""
        # Create all directories
        for directory in [self.optimization_dir, self.summa_sim_dir, self.mizuroute_sim_dir,
                         self.optimization_settings_dir, self.output_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create log directories
        (self.summa_sim_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.mizuroute_sim_dir / "logs").mkdir(parents=True, exist_ok=True)
        
        # Copy settings files
        self._copy_settings_files()
        
        # Update file managers for optimization
        self._update_optimization_file_managers()
    
    def _copy_settings_files(self) -> None:
        """Copy necessary settings files to optimization directory"""
        source_settings_dir = self.project_dir / "settings" / "SUMMA"
        
        if not source_settings_dir.exists():
            raise FileNotFoundError(f"Source settings directory not found: {source_settings_dir}")
        
        # SUMMA settings files to copy
        settings_files = [
            'fileManager.txt', 'modelDecisions.txt', 'outputControl.txt',
            'localParamInfo.txt', 'basinParamInfo.txt',
            'TBL_GENPARM.TBL', 'TBL_MPTABLE.TBL', 'TBL_SOILPARM.TBL', 'TBL_VEGPARM.TBL',
            'attributes.nc', 'coldState.nc', 'trialParams.nc', 'forcingFileList.txt'
        ]
        
        for file_name in settings_files:
            source_path = source_settings_dir / file_name
            dest_path = self.optimization_settings_dir / file_name
            
            if source_path.exists():
                shutil.copy2(source_path, dest_path)
        
        # Copy mizuRoute settings if they exist
        source_mizu_dir = self.project_dir / "settings" / "mizuRoute"
        dest_mizu_dir = self.optimization_dir / "settings" / "mizuRoute"
        
        if source_mizu_dir.exists():
            dest_mizu_dir.mkdir(parents=True, exist_ok=True)
            for mizu_file in source_mizu_dir.glob("*"):
                if mizu_file.is_file():
                    shutil.copy2(mizu_file, dest_mizu_dir / mizu_file.name)
    
    def _update_optimization_file_managers(self) -> None:
        """Update file managers for optimization runs"""
        # Update SUMMA file manager
        file_manager_path = self.optimization_settings_dir / 'fileManager.txt'
        if file_manager_path.exists():
            self._update_summa_file_manager(file_manager_path)
        
        # Update mizuRoute control file if it exists
        mizu_control_path = self.optimization_dir / "settings" / "mizuRoute" / "mizuroute.control"
        if mizu_control_path.exists():
            self._update_mizuroute_control_file(mizu_control_path)
    
    def _update_summa_file_manager(self, file_manager_path: Path, use_calibration_period: bool = True) -> None:
        """Update SUMMA file manager with spinup + calibration period"""
        with open(file_manager_path, 'r') as f:
            lines = f.readlines()
        
        if use_calibration_period:
            # Include spinup period before calibration
            calibration_period_str = self.config.get('CALIBRATION_PERIOD', '')
            spinup_period_str = self.config.get('SPINUP_PERIOD', '')
            
            if calibration_period_str and spinup_period_str:
                try:
                    spinup_dates = [d.strip() for d in spinup_period_str.split(',')]
                    cal_dates = [d.strip() for d in calibration_period_str.split(',')]
                    
                    if len(spinup_dates) >= 2 and len(cal_dates) >= 2:
                        spinup_start = datetime.strptime(spinup_dates[0], '%Y-%m-%d').replace(hour=1, minute=0)
                        cal_end = datetime.strptime(cal_dates[1], '%Y-%m-%d').replace(hour=23, minute=0)
                        
                        sim_start = spinup_start.strftime('%Y-%m-%d %H:%M')
                        sim_end = cal_end.strftime('%Y-%m-%d %H:%M')
                        
                        self.logger.info(f"Using spinup + calibration period: {sim_start} to {sim_end}")
                    else:
                        raise ValueError("Invalid period format")
                        
                except Exception as e:
                    self.logger.warning(f"Could not parse spinup+calibration periods: {str(e)}")
                    sim_start = self.config.get('EXPERIMENT_TIME_START', '1980-01-01 01:00')
                    sim_end = self.config.get('EXPERIMENT_TIME_END', '2018-12-31 23:00')
            else:
                sim_start = self.config.get('EXPERIMENT_TIME_START', '1980-01-01 01:00')
                sim_end = self.config.get('EXPERIMENT_TIME_END', '2018-12-31 23:00')
        else:
            sim_start = self.config.get('EXPERIMENT_TIME_START', '1980-01-01 01:00')
            sim_end = self.config.get('EXPERIMENT_TIME_END', '2018-12-31 23:00')
        
        # Update file manager
        updated_lines = []
        for line in lines:
            if 'simStartTime' in line:
                updated_lines.append(f"simStartTime         '{sim_start}'\n")
            elif 'simEndTime' in line:
                updated_lines.append(f"simEndTime           '{sim_end}'\n")
            elif 'outFilePrefix' in line:
                prefix = f'run_{self.algorithm_name}_final' if not use_calibration_period else f'run_{self.algorithm_name}_opt'
                updated_lines.append(f"outFilePrefix        '{prefix}_{self.experiment_id}'\n")
            elif 'outputPath' in line:
                output_path = str(self.summa_sim_dir).replace('\\', '/')
                updated_lines.append(f"outputPath           '{output_path}/'\n")
            elif 'settingsPath' in line:
                settings_path = str(self.optimization_settings_dir).replace('\\', '/')
                updated_lines.append(f"settingsPath         '{settings_path}/'\n")
            else:
                updated_lines.append(line)
        
        with open(file_manager_path, 'w') as f:
            f.writelines(updated_lines)
    
    def _update_mizuroute_control_file(self, control_path: Path) -> None:
        """Update mizuRoute control file for optimization"""
        with open(control_path, 'r') as f:
            lines = f.readlines()
        
        updated_lines = []
        for line in lines:
            if '<input_dir>' in line:
                input_path = str(self.summa_sim_dir).replace('\\', '/')
                updated_lines.append(f"<input_dir>             {input_path}/\n")
            elif '<output_dir>' in line:
                output_path = str(self.mizuroute_sim_dir).replace('\\', '/')
                updated_lines.append(f"<output_dir>            {output_path}/\n")
            elif '<case_name>' in line:
                updated_lines.append(f"<case_name>             run_{self.algorithm_name}_opt_{self.experiment_id}\n")
            elif '<fname_qsim>' in line:
                updated_lines.append(f"<fname_qsim>            run_{self.algorithm_name}_opt_{self.experiment_id}_timestep.nc\n")
            else:
                updated_lines.append(line)
        
        with open(control_path, 'w') as f:
            f.writelines(updated_lines)
    
    def _create_calibration_target(self) -> CalibrationTarget:
        """Factory method to create appropriate calibration target"""
        calibration_var = self.config.get('CALIBRATION_VARIABLE', 'streamflow')
        
        if calibration_var == 'streamflow':
            return StreamflowTarget(self.config, self.project_dir, self.logger)
        elif calibration_var == 'snow':
            return SnowTarget(self.config, self.project_dir, self.logger)
        else:
            raise ValueError(f"Unsupported calibration variable: {calibration_var}")
    
    def _setup_parallel_processing(self) -> None:
        """Setup parallel processing directories and files"""
        self.logger.info(f"Setting up parallel processing with {self.num_processes} processes")
        
        for proc_id in range(self.num_processes):
            # Create process-specific directories
            proc_base_dir = self.optimization_dir / f"parallel_proc_{proc_id:02d}"
            proc_summa_dir = proc_base_dir / "SUMMA"
            proc_mizuroute_dir = proc_base_dir / "mizuRoute"
            proc_summa_settings_dir = proc_base_dir / "settings" / "SUMMA"
            proc_mizu_settings_dir = proc_base_dir / "settings" / "mizuRoute"
            
            # Create directories
            for directory in [proc_base_dir, proc_summa_dir, proc_mizuroute_dir,
                             proc_summa_settings_dir, proc_mizu_settings_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            
            # Create log directories
            (proc_summa_dir / "logs").mkdir(parents=True, exist_ok=True)
            (proc_mizuroute_dir / "logs").mkdir(parents=True, exist_ok=True)
            
            # Copy settings files
            self._copy_settings_to_process_dir(proc_summa_settings_dir, proc_mizu_settings_dir)
            
            # Update file managers for this process
            self._update_process_file_managers(proc_id, proc_summa_dir, proc_mizuroute_dir,
                                             proc_summa_settings_dir, proc_mizu_settings_dir)
            
            # Store directory info
            self.parallel_dirs.append({
                'proc_id': proc_id,
                'base_dir': proc_base_dir,
                'summa_dir': proc_summa_dir,
                'mizuroute_dir': proc_mizuroute_dir,
                'summa_settings_dir': proc_summa_settings_dir,
                'mizuroute_settings_dir': proc_mizu_settings_dir
            })
    
    def _copy_settings_to_process_dir(self, proc_summa_settings_dir: Path, proc_mizu_settings_dir: Path) -> None:
        """Copy settings files to process-specific directory"""
        # Copy SUMMA settings
        if self.optimization_settings_dir.exists():
            for settings_file in self.optimization_settings_dir.glob("*"):
                if settings_file.is_file():
                    dest_file = proc_summa_settings_dir / settings_file.name
                    shutil.copy2(settings_file, dest_file)
        
        # Copy mizuRoute settings
        mizu_source_dir = self.optimization_dir / "settings" / "mizuRoute"
        if mizu_source_dir.exists():
            for settings_file in mizu_source_dir.glob("*"):
                if settings_file.is_file():
                    dest_file = proc_mizu_settings_dir / settings_file.name
                    shutil.copy2(settings_file, dest_file)
    
    def _update_process_file_managers(self, proc_id: int, summa_dir: Path, mizuroute_dir: Path,
                                    summa_settings_dir: Path, mizu_settings_dir: Path) -> None:
        """Update file managers for a specific process"""
        # Update SUMMA file manager
        file_manager = summa_settings_dir / 'fileManager.txt'
        if file_manager.exists():
            with open(file_manager, 'r') as f:
                lines = f.readlines()
            
            updated_lines = []
            for line in lines:
                if 'outFilePrefix' in line:
                    updated_lines.append(f"outFilePrefix        'proc_{proc_id:02d}_{self.algorithm_name}_opt_{self.experiment_id}'\n")
                elif 'outputPath' in line:
                    output_path = str(summa_dir).replace('\\', '/')
                    updated_lines.append(f"outputPath           '{output_path}/'\n")
                elif 'settingsPath' in line:
                    settings_path = str(summa_settings_dir).replace('\\', '/')
                    updated_lines.append(f"settingsPath         '{settings_path}/'\n")
                else:
                    updated_lines.append(line)
            
            with open(file_manager, 'w') as f:
                f.writelines(updated_lines)
        
        # Update mizuRoute control file
        control_file = mizu_settings_dir / 'mizuroute.control'
        if control_file.exists():
            with open(control_file, 'r') as f:
                lines = f.readlines()
            
            updated_lines = []
            for line in lines:
                if '<input_dir>' in line:
                    input_path = str(summa_dir).replace('\\', '/')
                    updated_lines.append(f"<input_dir>             {input_path}/\n")
                elif '<output_dir>' in line:
                    output_path = str(mizuroute_dir).replace('\\', '/')
                    updated_lines.append(f"<output_dir>            {output_path}/\n")
                elif '<case_name>' in line:
                    updated_lines.append(f"<case_name>             proc_{proc_id:02d}_{self.algorithm_name}_opt_{self.experiment_id}\n")
                elif '<fname_qsim>' in line:
                    updated_lines.append(f"<fname_qsim>            proc_{proc_id:02d}_{self.algorithm_name}_opt_{self.experiment_id}_timestep.nc\n")
                else:
                    updated_lines.append(line)
            
            with open(control_file, 'w') as f:
                f.writelines(updated_lines)
    
    def _cleanup_parallel_processing(self) -> None:
        """Cleanup parallel processing directories"""
        if not self.use_parallel:
            return
        
        self.logger.info("Cleaning up parallel working directories")
        cleanup_parallel = self.config.get('CLEANUP_PARALLEL_DIRS', True)
        if cleanup_parallel:
            try:
                for proc_dirs in self.parallel_dirs:
                    if proc_dirs['base_dir'].exists():
                        shutil.rmtree(proc_dirs['base_dir'])
            except Exception as e:
                self.logger.warning(f"Error during parallel cleanup: {str(e)}")
    
    def _evaluate_individual(self, normalized_params: np.ndarray) -> float:
        """Evaluate a single parameter set (sequential mode)"""
        try:
            # Denormalize parameters
            params = self.parameter_manager.denormalize_parameters(normalized_params)
            
            # Apply parameters to files
            if not self.parameter_manager.apply_parameters(params):
                return float('-inf')
            
            # Run models
            if not self.model_executor.run_models(
                self.summa_sim_dir, 
                self.mizuroute_sim_dir, 
                self.optimization_settings_dir
            ):
                return float('-inf')
            
            # Calculate performance metrics
            metrics = self.calibration_target.calculate_metrics(self.summa_sim_dir, self.mizuroute_sim_dir)
            if not metrics:
                return float('-inf')
            
            # Extract target metric
            score = self._extract_target_metric(metrics)
            
            # Apply negation for metrics where lower is better
            if self.target_metric.upper() in ['RMSE', 'MAE', 'PBIAS']:
                score = -score
            
            return score if score is not None and not np.isnan(score) else float('-inf')
            
        except Exception as e:
            self.logger.debug(f"Parameter evaluation failed: {str(e)}")
            return float('-inf')
    
    def _extract_target_metric(self, metrics: Dict[str, float]) -> Optional[float]:
        """Extract target metric from metrics dictionary"""
        # Try exact match
        if self.target_metric in metrics:
            return metrics[self.target_metric]
        
        # Try with calibration prefix
        calib_key = f"Calib_{self.target_metric}"
        if calib_key in metrics:
            return metrics[calib_key]
        
        # Try without prefix
        for key, value in metrics.items():
            if key.endswith(f"_{self.target_metric}"):
                return value
        
        # Fallback to first available metric
        return next(iter(metrics.values())) if metrics else None
    
    def _run_parallel_evaluations(self, evaluation_tasks: List[Dict]) -> List[Dict]:
        """Complete parallel evaluation with stale file handle recovery and fallback"""
        
        start_time = time.time()
        num_tasks = len(evaluation_tasks)
        
        # Track consecutive failures for recovery
        if not hasattr(self, '_consecutive_parallel_failures'):
            self._consecutive_parallel_failures = 0
        
        self.logger.info(f"Starting parallel evaluation of {num_tasks} tasks with {self.num_processes} processes")
        
        # RECOVERY LOGIC: If we've had consecutive failures, try recovery
        if self._consecutive_parallel_failures >= 50:
            self.logger.warning(f"Detected {self._consecutive_parallel_failures} consecutive parallel failures. Attempting recovery...")
            
            # Recovery measures
            gc.collect()
            time.sleep(3.0)  # Let file system settle
            
            # Reduce concurrent processes temporarily
            effective_processes = max(4, self.num_processes // 2)
            self.logger.info(f"Reducing concurrent processes from {self.num_processes} to {effective_processes} for recovery")
            
            # Touch critical files to refresh handles
            try:
                for proc_dirs in self.parallel_dirs[:effective_processes]:
                    settings_dir = Path(proc_dirs['summa_settings_dir'])
                    for critical_file in ['fileManager.txt', 'attributes.nc', 'coldState.nc', 'trialParams.nc']:
                        file_path = settings_dir / critical_file
                        if file_path.exists():
                            file_path.touch()
                            time.sleep(0.01)  # Small delay between touches
            except Exception as e:
                self.logger.debug(f"File handle refresh failed: {str(e)}")
        else:
            effective_processes = self.num_processes
        
        # FALLBACK LOGIC: If too many consecutive failures, fall back to sequential
        if self._consecutive_parallel_failures >= 5:
            self.logger.warning("Too many consecutive parallel failures. Falling back to sequential evaluation.")
            return self._run_sequential_fallback(evaluation_tasks)
        
        # Prepare worker tasks
        worker_tasks = []
        for task in evaluation_tasks:
            proc_dirs = self.parallel_dirs[task['proc_id']]
            
            task_data = {
                'individual_id': task['individual_id'],
                'params': task['params'],
                'proc_id': task['proc_id'],
                'evaluation_id': task['evaluation_id'],
                
                # Paths for worker
                'summa_exe': str(self._get_summa_exe_path()),
                'file_manager': str(proc_dirs['summa_settings_dir'] / 'fileManager.txt'),
                'summa_dir': str(proc_dirs['summa_dir']),
                'mizuroute_dir': str(proc_dirs['mizuroute_dir']),
                'summa_settings_dir': str(proc_dirs['summa_settings_dir']),
                'mizuroute_settings_dir': str(proc_dirs['mizuroute_settings_dir']),
                
                # Configuration for worker
                'config': self.config,
                'target_metric': self.target_metric,
                'calibration_variable': self.config.get('CALIBRATION_VARIABLE', 'streamflow'),
                'domain_name': self.domain_name,
                'project_dir': str(self.project_dir),
                'original_depths': self.parameter_manager.original_depths.tolist() if self.parameter_manager.original_depths is not None else None,
            }
            
            worker_tasks.append(task_data)
        
        results = []
        completed_count = 0
        
        try:
            # Execute with reduced concurrency and batching
            batch_size = min(effective_processes, 100)
            
            for batch_start in range(0, len(worker_tasks), batch_size):
                batch_end = min(batch_start + batch_size, len(worker_tasks))
                batch_tasks = worker_tasks[batch_start:batch_end]
                
                # Add pre-batch delay to reduce file system pressure
                if batch_start > 0:
                    time.sleep(1.0)
                
                # Execute batch with timeout and retry
                batch_results = self._execute_batch_mpi(batch_tasks, len(batch_tasks))
                results.extend(batch_results)
                completed_count += len(batch_tasks)
                
                # Check for stale file handle errors in this batch
                stale_errors = sum(1 for r in batch_results if r.get('error') and 'stale file handle' in str(r['error']).lower())
                if stale_errors > len(batch_tasks) * 0.5:  # More than 50% stale file handle errors
                    self.logger.warning(f"High rate of stale file handle errors in batch ({stale_errors}/{len(batch_tasks)})")
                    # Add extra recovery time
                    time.sleep(2.0)
            
            # Calculate success rate
            successful_count = sum(1 for r in results if r['score'] is not None)
            success_rate = successful_count / num_tasks if num_tasks > 0 else 0
            
            elapsed = time.time() - start_time
            self.logger.info(f"Parallel evaluation completed: {successful_count}/{num_tasks} successful "
                        f"({100*success_rate:.1f}%) in {elapsed/60:.1f} minutes")
            
            # Update failure counter based on success rate
            if success_rate >= 0.7:  # 70% or better success
                self._consecutive_parallel_failures = 0
            else:
                self._consecutive_parallel_failures += 1
                
            # If success rate is too low, warn about potential sequential fallback
            if success_rate < 0.5:
                self.logger.warning(f"Low success rate ({100*success_rate:.1f}%). Next failure may trigger sequential fallback.")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Critical error in parallel evaluation: {str(e)}")
            self._consecutive_parallel_failures += 1
            
            # Check if this is a stale file handle error
            if 'stale file handle' in str(e).lower() or 'errno 116' in str(e).lower():
                self.logger.error("Detected stale file handle error - file system may be overloaded")
                
            # Return failed results for all tasks
            return [
                {
                    'individual_id': task['individual_id'],
                    'params': task['params'],
                    'score': None,
                    'error': f'Critical parallel evaluation error: {str(e)}'
                }
                for task in evaluation_tasks
            ]

    def _execute_batch_mpi(self, batch_tasks: List[Dict], max_workers: int) -> List[Dict]:
        """Spawn MPI processes internally for parallel execution"""
        try:
            # Determine number of processes to use
            num_processes = min(max_workers, self.num_processes, len(batch_tasks))
            
            self.logger.info(f"Spawning {num_processes} MPI processes for batch execution")
            
            # Create temporary files for communication
            with tempfile.TemporaryDirectory(prefix='confluence_mpi_') as temp_dir:
                temp_dir = Path(temp_dir)
                
                # Save tasks to file
                tasks_file = temp_dir / 'tasks.pkl'
                with open(tasks_file, 'wb') as f:
                    pickle.dump(batch_tasks, f)
                
                # Create worker script
                worker_script = temp_dir / 'mpi_worker.py'
                self._create_mpi_worker_script(worker_script, tasks_file, temp_dir)
                
                # Run MPI command
                results_file = temp_dir / 'results.pkl'
                mpi_cmd = [
                    'mpirun', '-n', str(num_processes),
                    sys.executable, str(worker_script),
                    str(tasks_file), str(results_file)
                ]
                
                self.logger.debug(f"Running MPI command: {' '.join(mpi_cmd)}")
                
                # Execute MPI process
                start_time = time.time()
                result = subprocess.run(
                    mpi_cmd,
                    capture_output=True,
                    text=True,
                    timeout=7200,  # 2 hour timeout
                    env=os.environ.copy()
                )
                
                elapsed = time.time() - start_time
                
                if result.returncode != 0:
                    self.logger.error(f"MPI execution failed: {result.stderr}")
                    raise RuntimeError(f"MPI process failed with return code {result.returncode}")
                
                # Load results
                if results_file.exists():
                    with open(results_file, 'rb') as f:
                        results = pickle.load(f)
                    
                    successful_count = sum(1 for r in results if r.get('score') is not None)
                    self.logger.info(f"MPI batch completed in {elapsed:.1f}s: {successful_count}/{len(results)} successful")
                    
                    return results
                else:
                    raise RuntimeError("MPI results file not created")
        
        except subprocess.TimeoutExpired:
            self.logger.error("MPI execution timed out")
            return self._create_error_results(batch_tasks, "MPI execution timeout")
        
        except FileNotFoundError:
            self.logger.warning("mpirun not found")
            return self._create_error_results(batch_tasks, "mpirun not found")
        
        except Exception as e:
            self.logger.error(f"MPI spawn execution failed: {str(e)}")
            return self._create_error_results(batch_tasks, f"MPI spawn error: {str(e)}")

    def _create_mpi_worker_script(self, script_path: Path, tasks_file: Path, temp_dir: Path) -> None:
        """Create a standalone MPI worker script"""
        script_content = f'''#!/usr/bin/env python3
import sys
import pickle
import os
from pathlib import Path
from mpi4py import MPI

# Add CONFLUENCE path
sys.path.append(r"{Path(__file__).parent}")

# Import the worker function
from iterative_optimizer import _evaluate_parameters_worker_safe

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    tasks_file = Path(sys.argv[1])
    results_file = Path(sys.argv[2])
    
    # Load tasks on rank 0
    if rank == 0:
        with open(tasks_file, 'rb') as f:
            all_tasks = pickle.load(f)
        
        # Distribute tasks
        tasks_per_rank = len(all_tasks) // size
        extra_tasks = len(all_tasks) % size
        
        all_results = []
        
        for worker_rank in range(size):
            start_idx = worker_rank * tasks_per_rank + min(worker_rank, extra_tasks)
            end_idx = start_idx + tasks_per_rank + (1 if worker_rank < extra_tasks else 0)
            
            if worker_rank == 0:
                my_tasks = all_tasks[start_idx:end_idx]
            else:
                worker_tasks = all_tasks[start_idx:end_idx]
                comm.send(worker_tasks, dest=worker_rank, tag=1)
        
        # Execute rank 0 tasks
        my_results = []
        for task in my_tasks:
            try:
                result = _evaluate_parameters_worker_safe(task)
                my_results.append(result)
            except Exception as e:
                error_result = {{
                    'individual_id': task.get('individual_id', -1),
                    'params': task.get('params', {{}}),
                    'score': None,
                    'error': f'Rank 0 error: {{str(e)}}'
                }}
                my_results.append(error_result)
        
        all_results.extend(my_results)
        
        # Collect from workers
        for worker_rank in range(1, size):
            try:
                worker_results = comm.recv(source=worker_rank, tag=2)
                all_results.extend(worker_results)
            except Exception as e:
                print(f"Error receiving from worker {{worker_rank}}: {{e}}")
        
        # Save results
        with open(results_file, 'wb') as f:
            pickle.dump(all_results, f)
    
    else:
        # Worker process
        try:
            my_tasks = comm.recv(source=0, tag=1)
            my_results = []
            
            for task in my_tasks:
                try:
                    result = _evaluate_parameters_worker_safe(task)
                    my_results.append(result)
                except Exception as e:
                    error_result = {{
                        'individual_id': task.get('individual_id', -1),
                        'params': task.get('params', {{}}),
                        'score': None,
                        'error': f'Rank {{rank}} error: {{str(e)}}'
                    }}
                    my_results.append(error_result)
            
            comm.send(my_results, dest=0, tag=2)
            
        except Exception as e:
            print(f"Worker {{rank}} failed: {{e}}")

if __name__ == "__main__":
    main()
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)

    def _create_error_results(self, batch_tasks: List[Dict], error_msg: str) -> List[Dict]:
        """Create error results for all tasks"""
        return [
            {
                'individual_id': task_data.get('individual_id', -1),
                'params': task_data.get('params', {}),
                'score': None,
                'error': error_msg
            }
            for task_data in batch_tasks
        ]

    def _run_sequential_fallback(self, evaluation_tasks: List[Dict]) -> List[Dict]:
        """Fallback to sequential evaluation when parallel processing fails"""
        self.logger.info("Running sequential fallback evaluation")
        
        results = []
        
        for i, task in enumerate(evaluation_tasks):
            self.logger.info(f"Sequential evaluation {i+1}/{len(evaluation_tasks)}")
            
            try:
                # Extract parameters and evaluate sequentially
                params = task['params']
                normalized_params = self.parameter_manager.normalize_parameters(params)
                score = self._evaluate_individual(normalized_params)
                
                results.append({
                    'individual_id': task['individual_id'],
                    'params': params,
                    'score': score if score != float('-inf') else None,
                    'error': None if score != float('-inf') else 'Sequential evaluation failed'
                })
                
            except Exception as e:
                self.logger.error(f"Sequential evaluation {i+1} failed: {str(e)}")
                results.append({
                    'individual_id': task['individual_id'],
                    'params': task['params'],
                    'score': None,
                    'error': f'Sequential evaluation error: {str(e)}'
                })
        
        # Reset parallel failure counter after successful sequential run
        successful_count = sum(1 for r in results if r['score'] is not None)
        if successful_count > 0:
            self._consecutive_parallel_failures = max(0, self._consecutive_parallel_failures - 1)
        
        return results
    
    def _run_final_evaluation(self, best_params: Dict) -> Optional[Dict]:
        """Run final evaluation with best parameters over full period"""
        self.logger.info("Running final evaluation with best parameters")
        
        try:
            # Update file manager for full period
            self._update_file_manager_for_final_run()
            
            # Apply best parameters
            if not self.parameter_manager.apply_parameters(best_params):
                self.logger.error("Failed to apply best parameters for final run")
                return None
            
            # Run models
            if not self.model_executor.run_models(
                self.summa_sim_dir,
                self.mizuroute_sim_dir,
                self.optimization_settings_dir
            ):
                self.logger.error("Final model run failed")
                return None
            
            # Calculate metrics for full period
            metrics = self.calibration_target.calculate_metrics(self.summa_sim_dir, self.mizuroute_sim_dir, calibration_only=False)
            
            if metrics:
                self.logger.info("Final evaluation completed successfully")
                return {
                    'final_metrics': metrics,
                    'summa_success': True,
                    'mizuroute_success': self.calibration_target.needs_routing()
                }
            else:
                return {'summa_success': False, 'mizuroute_success': False}
                
        except Exception as e:
            self.logger.error(f"Error in final evaluation: {str(e)}")
            return None
        finally:
            # Reset file manager back to optimization mode
            self._update_summa_file_manager(self.optimization_settings_dir / 'fileManager.txt')
    
    def _update_file_manager_for_final_run(self) -> None:
        """Update file manager to use full experiment period"""
        file_manager_path = self.optimization_settings_dir / 'fileManager.txt'
        
        if not file_manager_path.exists():
            return
        
        with open(file_manager_path, 'r') as f:
            lines = f.readlines()
        
        # Use full experiment period
        sim_start = self.config.get('EXPERIMENT_TIME_START', '1980-01-01 01:00')
        sim_end = self.config.get('EXPERIMENT_TIME_END', '2018-12-31 23:00')
        
        updated_lines = []
        for line in lines:
            if 'simStartTime' in line:
                updated_lines.append(f"simStartTime         '{sim_start}'\n")
            elif 'simEndTime' in line:
                updated_lines.append(f"simEndTime           '{sim_end}'\n")
            elif 'outFilePrefix' in line:
                updated_lines.append(f"outFilePrefix        'run_{self.algorithm_name}_final_{self.experiment_id}'\n")
            else:
                updated_lines.append(line)
        
        with open(file_manager_path, 'w') as f:
            f.writelines(updated_lines)
    
    def _save_to_default_settings(self, best_params: Dict) -> bool:
        """Save best parameters to default model settings"""
        try:
            self.logger.info("Saving best parameters to default model settings")
            
            default_settings_dir = self.project_dir / "settings" / "SUMMA"
            if not default_settings_dir.exists():
                self.logger.error("Default settings directory not found")
                return False
            
            # Backup existing files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save hydraulic parameters to trialParams.nc
            hydraulic_params = {k: v for k, v in best_params.items() 
                              if k not in ['total_mult', 'shape_factor']}
            
            if hydraulic_params:
                trial_params_path = default_settings_dir / "trialParams.nc"
                if trial_params_path.exists():
                    backup_path = default_settings_dir / f"trialParams_backup_{timestamp}.nc"
                    shutil.copy2(trial_params_path, backup_path)
                
                # Generate new trialParams.nc
                param_manager = ParameterManager(self.config, self.logger, default_settings_dir)
                if param_manager._generate_trial_params_file(hydraulic_params):
                    self.logger.info("✅ Saved optimized hydraulic parameters")
                else:
                    self.logger.error("Failed to save hydraulic parameters")
                    return False
            
            # Save soil depths to coldState.nc
            if self.config.get('CALIBRATE_DEPTH', False) and 'total_mult' in best_params:
                coldstate_path = default_settings_dir / "coldState.nc"
                if coldstate_path.exists():
                    backup_path = default_settings_dir / f"coldState_backup_{timestamp}.nc"
                    shutil.copy2(coldstate_path, backup_path)
                
                # Copy optimized coldState
                optimized_coldstate = self.optimization_settings_dir / "coldState.nc"
                if optimized_coldstate.exists():
                    shutil.copy2(optimized_coldstate, coldstate_path)
                    self.logger.info("✅ Saved optimized soil depths")
            
            # Save mizuRoute parameters
            if self.config.get('CALIBRATE_MIZUROUTE', False):
                mizu_param_file = self.project_dir / "settings" / "mizuRoute" / "param.nml.default"
                if mizu_param_file.exists():
                    backup_path = mizu_param_file.parent / f"param.nml.backup_{timestamp}"
                    shutil.copy2(mizu_param_file, backup_path)
                    
                    # Copy optimized parameters
                    optimized_mizu = self.optimization_dir / "settings" / "mizuRoute" / "param.nml.default"
                    if optimized_mizu.exists():
                        shutil.copy2(optimized_mizu, mizu_param_file)
                        self.logger.info("✅ Saved optimized mizuRoute parameters")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving to default settings: {str(e)}")
            return False
    
    def _get_summa_exe_path(self) -> Path:
        """Get SUMMA executable path"""
        summa_path = self.config.get('SUMMA_INSTALL_PATH')
        if summa_path == 'default':
            summa_path = Path(self.config.get('CONFLUENCE_DATA_DIR')) / 'installs' / 'summa' / 'bin'
        else:
            summa_path = Path(summa_path)
        
        return summa_path / self.config.get('SUMMA_EXE')
    
    def run_optimization(self) -> Dict[str, Any]:
        """
        Main optimization method that delegates to algorithm-specific implementation
        """
        algorithm_name = self.get_algorithm_name()
        
        self.logger.info("=" * 60)
        self.logger.info(f"Starting {algorithm_name} optimization for {self.config.get('CALIBRATION_VARIABLE', 'streamflow')} calibration")
        self.logger.info(f"Target metric: {self.target_metric}")
        self.logger.info(f"Max iterations: {self.max_iterations}")
        
        if self.use_parallel:
            self.logger.info(f"Parallel processing: {self.num_processes} processes")
        
        self.logger.info("=" * 60)
        
        try:
            start_time = datetime.now()
            
            # Get initial parameters
            initial_params = self.parameter_manager.get_initial_parameters()
            if not initial_params:
                raise RuntimeError("Failed to get initial parameters")
            
            # Run algorithm-specific implementation
            best_params, best_score, history = self._run_algorithm()
            
            # Run final evaluation
            final_result = self._run_final_evaluation(best_params)
            
            # Save results
            self.results_manager.save_results(best_params, best_score, history, final_result)
            
            # Save to default model settings
            self._save_to_default_settings(best_params)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info("=" * 60)
            self.logger.info(f"{algorithm_name} OPTIMIZATION COMPLETED")
            self.logger.info(f"🏆 Best {self.target_metric}: {best_score:.6f}")
            self.logger.info(f"⏱️ Total time: {duration}")
            self.logger.info("=" * 60)
            
            return {
                'best_parameters': best_params,
                'best_score': best_score,
                'history': history,
                'final_result': final_result,
                'algorithm': algorithm_name,
                'calibration_variable': self.config.get('CALIBRATION_VARIABLE', 'streamflow'),
                'optimization_metric': self.target_metric,
                'duration': str(duration),
                'output_dir': str(self.output_dir)
            }
            
        except Exception as e:
            self.logger.error(f"{algorithm_name} optimization failed: {str(e)}")
            raise
        finally:
            self._cleanup_parallel_processing()


class DDSOptimizer(BaseOptimizer):
    """
    Dynamically Dimensioned Search (DDS) Optimizer for CONFLUENCE
    
    Implements the DDS algorithm with support for multi-start parallel execution.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        
        # DDS-specific parameters
        self.dds_r = config.get('DDS_R', 0.2)
        
        # DDS state variables
        self.population = None
        self.population_scores = None
    
    def get_algorithm_name(self) -> str:
        return "DDS"
    
    def _run_algorithm(self) -> Tuple[Dict, float, List]:
        """
        Run DDS algorithm with optional multi-start parallel support
        """
        self.logger.info("Running DDS algorithm")
        
        # Initialize DDS with single solution
        initial_params = self.parameter_manager.get_initial_parameters()
        self._initialize_dds(initial_params)
        
        # Check if we should use multi-start parallel DDS
        if self.use_parallel and self.num_processes > 1:
            return self._run_multi_start_parallel_dds()
        else:
            return self._run_single_dds()
    
    def _initialize_dds(self, initial_params: Dict[str, np.ndarray]) -> None:
        """Initialize DDS with a single solution"""
        self.logger.info("Initializing DDS with single solution")
        
        param_count = len(self.parameter_manager.all_param_names)
        
        # Initialize minimal population for compatibility (DDS only needs 1 solution)
        self.population = np.random.random((1, param_count))
        self.population_scores = np.full(1, np.nan, dtype=float)
        
        # Set the single solution to initial parameters
        if initial_params:
            initial_normalized = self.parameter_manager.normalize_parameters(initial_params)
            self.population[0] = np.clip(initial_normalized, 0, 1)
        
        # Evaluate initial solution
        self.logger.info("Evaluating initial solution...")
        self.population_scores[0] = self._evaluate_individual(self.population[0])
        
        # Initialize best solution
        if not np.isnan(self.population_scores[0]):
            self.best_score = self.population_scores[0]
            self.best_params = self.parameter_manager.denormalize_parameters(self.population[0])
        else:
            self.best_score = float('-inf')
            self.best_params = initial_params
        
        # Record initial "generation"
        self._record_generation(0)
        
        self.logger.info(f"Initial solution evaluated. Score: {self.best_score:.6f}")
    
    def _run_single_dds(self) -> Tuple[Dict, float, List]:
        """Run single-instance DDS algorithm"""
        self.logger.info("Single-instance DDS")
        
        # Use the single solution from initialization
        current_solution = self.population[0].copy()
        current_score = self.population_scores[0]
        
        # Handle case where initial evaluation failed
        if np.isnan(current_score) or current_score == float('-inf'):
            self.logger.warning("Initial solution evaluation failed, re-evaluating...")
            current_score = self._evaluate_individual(current_solution)
            if current_score != float('-inf'):
                self.population_scores[0] = current_score
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_params = self.parameter_manager.denormalize_parameters(current_solution)
        
        num_params = len(self.parameter_manager.all_param_names)
        
        # DDS main loop
        for iteration in range(1, self.max_iterations + 1):
            iteration_start_time = datetime.now()
            
            # Calculate probability of selecting each variable for perturbation
            prob_select = 1.0 - np.log(iteration) / np.log(self.max_iterations)
            prob_select = max(prob_select, 1.0 / num_params)
            
            # Create trial solution by perturbing current solution
            trial_solution = current_solution.copy()
            
            # Determine which variables to perturb
            variables_to_perturb = np.random.random(num_params) < prob_select
            if not np.any(variables_to_perturb):
                random_idx = np.random.randint(0, num_params)
                variables_to_perturb[random_idx] = True
            
            # Apply DDS perturbation to selected variables
            for i in range(num_params):
                if variables_to_perturb[i]:
                    perturbation = np.random.normal(0, self.dds_r)
                    trial_solution[i] = current_solution[i] + perturbation
                    
                    # Reflect at bounds [0,1]
                    if trial_solution[i] < 0:
                        trial_solution[i] = -trial_solution[i]
                    elif trial_solution[i] > 1:
                        trial_solution[i] = 2.0 - trial_solution[i]
                    
                    trial_solution[i] = np.clip(trial_solution[i], 0, 1)
            
            # Evaluate trial solution
            trial_score = self._evaluate_individual(trial_solution)
            
            # DDS selection: accept if better (greedy)
            improvement = False
            if trial_score > current_score:
                current_solution = trial_solution.copy()
                current_score = trial_score
                improvement = True
                
                # Update population for consistency
                self.population[0] = current_solution.copy()
                self.population_scores[0] = current_score
                
                # Update global best
                if trial_score > self.best_score:
                    self.best_score = trial_score
                    self.best_params = self.parameter_manager.denormalize_parameters(trial_solution)
                    self.logger.info(f"Iter {iteration:3d}: NEW BEST! {self.target_metric}={self.best_score:.6f}")
            
            # Record generation statistics
            self._record_dds_generation(iteration, current_score, trial_score, improvement, 
                                      np.sum(variables_to_perturb), prob_select)
            
            # Log progress
            iteration_duration = datetime.now() - iteration_start_time
            status = "IMPROVED" if improvement else "no change"
            self.logger.info(f"Iter {iteration:3d}/{self.max_iterations}: "
                            f"Current={current_score:.6f}, Best={self.best_score:.6f}, "
                            f"Perturbed={np.sum(variables_to_perturb)}/{num_params} vars, "
                            f"{status} [{iteration_duration.total_seconds():.1f}s]")
        
        return self.best_params, self.best_score, self.iteration_history
    
    def _run_multi_start_parallel_dds(self) -> Tuple[Dict, float, List]:
        """Multi-start parallel DDS following Tolson et al. (2014)"""
        num_starts = self.num_processes
        iterations_per_start = max(1, self.max_iterations // num_starts)
        
        self.logger.info(f"Multi-start parallel DDS: {num_starts} instances")
        self.logger.info(f"Iterations per instance: {iterations_per_start}")
        
        # Get initial parameters for base starting point
        initial_params = self.parameter_manager.get_initial_parameters()
        if not initial_params:
            raise RuntimeError("Failed to get initial parameters for multi-start DDS")
        
        base_normalized = self.parameter_manager.normalize_parameters(initial_params)
        param_count = len(self.parameter_manager.all_param_names)
        
        # Create tasks for parallel DDS instances
        dds_tasks = []
        for start_id in range(num_starts):
            # Create diverse starting points
            if start_id == 0:
                starting_solution = base_normalized.copy()
            else:
                # Create perturbed starting points for diversity
                if start_id % 3 == 1:
                    noise = np.random.normal(0, 0.15, param_count)
                    starting_solution = np.clip(base_normalized + noise, 0, 1)
                elif start_id % 3 == 2:
                    starting_solution = np.random.random(param_count)
                else:
                    starting_solution = 1.0 - base_normalized + np.random.normal(0, 0.1, param_count)
                    starting_solution = np.clip(starting_solution, 0, 1)
            
            task = {
                'start_id': start_id,
                'starting_solution': starting_solution,
                'max_iterations': iterations_per_start,
                'dds_r': self.dds_r,
                'random_seed': start_id * 12345,
                'proc_id': start_id % len(self.parallel_dirs),
                'evaluation_id': f"multi_dds_{start_id:02d}"
            }
            dds_tasks.append(task)
        
        self.logger.info("Starting parallel DDS instances...")
        
        # Run all DDS instances in parallel
        start_time = datetime.now()
        results = self._run_parallel_dds_instances(dds_tasks)
        elapsed = datetime.now() - start_time
        
        # Process results and find best
        valid_results = [r for r in results if r.get('best_score') is not None]
        
        if not valid_results:
            raise RuntimeError("All parallel DDS instances failed")
        
        # Find globally best result
        best_result = max(valid_results, key=lambda x: x['best_score'])
        
        # Combine histories from all instances
        combined_history = self._combine_dds_histories(valid_results, num_starts)
        
        # Log summary
        success_rate = len(valid_results) / num_starts
        best_scores = [r['best_score'] for r in valid_results]
        
        self.logger.info("=" * 50)
        self.logger.info("MULTI-START DDS SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Successful instances: {len(valid_results)}/{num_starts} ({100*success_rate:.1f}%)")
        self.logger.info(f"Best score across all: {best_result['best_score']:.6f}")
        self.logger.info(f"Total parallel time: {elapsed.total_seconds():.1f}s")
        self.logger.info("=" * 50)
        
        # Update class variables with best result
        self.best_score = best_result['best_score']
        self.best_params = best_result['best_params']
        self.iteration_history = combined_history
        
        return self.best_params, self.best_score, self.iteration_history
    
    def _run_parallel_dds_instances(self, dds_tasks: List[Dict]) -> List[Dict]:
        """Run multiple DDS instances in parallel"""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        results = []
        max_workers = min(len(dds_tasks), self.num_processes)
        
        self.logger.info(f"Starting {len(dds_tasks)} DDS workers with {max_workers} processes")
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all DDS instance tasks
                future_to_task = {}
                
                for task in dds_tasks:
                    # Prepare worker data
                    worker_data = self._prepare_dds_worker_data(task)
                    future = executor.submit(_run_dds_instance_worker, worker_data)
                    future_to_task[future] = task
                
                # Collect results as they complete
                for future in as_completed(future_to_task, timeout=7200):  # 2 hour timeout
                    task = future_to_task[future]
                    start_id = task['start_id']
                    
                    try:
                        result = future.result(timeout=3600)  # 1 hour per instance
                        results.append(result)
                        
                        if result.get('best_score') is not None:
                            self.logger.info(f"DDS instance {start_id} completed: {result['best_score']:.6f}")
                        else:
                            self.logger.warning(f"DDS instance {start_id} failed")
                            
                    except Exception as e:
                        self.logger.error(f"DDS instance {start_id} failed: {str(e)}")
                        results.append({
                            'start_id': start_id,
                            'best_score': None,
                            'error': str(e)
                        })
        
        except Exception as e:
            self.logger.error(f"Parallel DDS execution failed: {str(e)}")
            raise
        
        return results
    
    def _prepare_dds_worker_data(self, task: Dict) -> Dict:
        """Prepare data for DDS instance worker"""
        dds_task = task
        proc_id = task['proc_id']
        
        # Get the parallel directory for this process
        if hasattr(self, 'parallel_dirs') and proc_id < len(self.parallel_dirs):
            proc_dirs = self.parallel_dirs[proc_id]
        else:
            proc_dirs = {
                'summa_dir': self.summa_sim_dir,
                'mizuroute_dir': self.mizuroute_sim_dir,
                'summa_settings_dir': self.optimization_settings_dir,
                'mizuroute_settings_dir': self.optimization_dir / "settings" / "mizuRoute"
            }
        
        return {
            'dds_task': dds_task,
            'config': self.config,
            'target_metric': self.target_metric,
            'calibration_variable': self.config.get('CALIBRATION_VARIABLE', 'streamflow'),
            'domain_name': self.domain_name,
            'project_dir': str(self.project_dir),
            'param_bounds': self.parameter_manager.param_bounds,
            'all_param_names': self.parameter_manager.all_param_names,
            'original_depths': self.parameter_manager.original_depths.tolist() if self.parameter_manager.original_depths is not None else None,
            
            # Paths for worker
            'summa_exe': str(self._get_summa_exe_path()),
            'summa_dir': str(proc_dirs['summa_dir']),
            'mizuroute_dir': str(proc_dirs['mizuroute_dir']),
            'summa_settings_dir': str(proc_dirs['summa_settings_dir']),
            'mizuroute_settings_dir': str(proc_dirs['mizuroute_settings_dir']),
            'file_manager': str(proc_dirs['summa_settings_dir'] / 'fileManager.txt')
        }
    
    def _combine_dds_histories(self, results: List[Dict], num_starts: int) -> List[Dict]:
        """Combine histories from multiple DDS instances"""
        combined_history = []
        max_iterations = max(len(r.get('history', [])) for r in results if r.get('history'))
        
        for iteration in range(max_iterations):
            iteration_scores = []
            best_result_this_iter = None
            
            for result in results:
                history = result.get('history', [])
                if iteration < len(history):
                    iter_data = history[iteration]
                    if iter_data.get('best_score') is not None:
                        iteration_scores.append(iter_data['best_score'])
                        if (best_result_this_iter is None or 
                            iter_data['best_score'] > best_result_this_iter.get('best_score', float('-inf'))):
                            best_result_this_iter = iter_data
            
            if iteration_scores:
                combined_entry = {
                    'generation': iteration,
                    'algorithm': 'Multi-Start DDS',
                    'num_starts': num_starts,
                    'best_score': max(iteration_scores),
                    'mean_score': np.mean(iteration_scores),
                    'std_score': np.std(iteration_scores),
                    'worst_score': min(iteration_scores),
                    'valid_individuals': len(iteration_scores),
                    'best_params': best_result_this_iter.get('best_params') if best_result_this_iter else None
                }
                combined_history.append(combined_entry)
        
        return combined_history
    
    def _record_generation(self, generation: int) -> None:
        """Record generation statistics"""
        valid_scores = self.population_scores[~np.isnan(self.population_scores)]
        
        generation_stats = {
            'generation': generation,
            'algorithm': 'DDS',
            'best_score': self.best_score if self.best_score != float('-inf') else None,
            'best_params': self.best_params.copy() if self.best_params is not None else None,
            'mean_score': np.mean(valid_scores) if len(valid_scores) > 0 else None,
            'std_score': np.std(valid_scores) if len(valid_scores) > 0 else None,
            'worst_score': np.min(valid_scores) if len(valid_scores) > 0 else None,
            'valid_individuals': len(valid_scores),
            'population_scores': self.population_scores.copy()
        }
        
        self.iteration_history.append(generation_stats)
    
    def _record_dds_generation(self, iteration: int, current_score: float, trial_score: float, 
                              improvement: bool, num_perturbed: int, prob_select: float) -> None:
        """Record DDS generation statistics"""
        valid_scores = self.population_scores[~np.isnan(self.population_scores)]
        
        generation_stats = {
            'generation': iteration,
            'algorithm': 'DDS',
            'best_score': self.best_score if self.best_score != float('-inf') else None,
            'best_params': self.best_params.copy() if self.best_params is not None else None,
            'current_score': current_score if current_score != float('-inf') else None,
            'trial_score': trial_score if trial_score != float('-inf') else None,
            'improvement': improvement,
            'num_variables_perturbed': num_perturbed,
            'selection_probability': prob_select,
            'mean_score': np.mean(valid_scores) if len(valid_scores) > 0 else None,
            'std_score': np.std(valid_scores) if len(valid_scores) > 0 else None,
            'worst_score': np.min(valid_scores) if len(valid_scores) > 0 else None,
            'valid_individuals': len(valid_scores),
            'population_scores': self.population_scores.copy()
        }
        
        self.iteration_history.append(generation_stats)


class DEOptimizer(BaseOptimizer):
    """
    Differential Evolution (DE) Optimizer for CONFLUENCE
    
    Implements the DE algorithm with population-based optimization.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        
        # DE-specific parameters
        self.population_size = self._determine_population_size()
        self.F = config.get('DE_SCALING_FACTOR', 0.5)
        self.CR = config.get('DE_CROSSOVER_RATE', 0.9)
        
        # DE state variables
        self.population = None
        self.population_scores = None
    
    def get_algorithm_name(self) -> str:
        return "DE"
    
    def _determine_population_size(self) -> int:
        """Determine appropriate population size based on parameter count"""
        config_pop_size = self.config.get('POPULATION_SIZE')
        if config_pop_size:
            return config_pop_size
        
        # Calculate based on parameter count
        total_params = (len(self.config.get('PARAMS_TO_CALIBRATE', '').split(',')) +
                       len(self.config.get('BASIN_PARAMS_TO_CALIBRATE', '').split(',')) +
                       (2 if self.config.get('CALIBRATE_DEPTH', False) else 0) +
                       (len(self.config.get('MIZUROUTE_PARAMS_TO_CALIBRATE', '').split(',')) if self.config.get('CALIBRATE_MIZUROUTE', False) else 0))
        
        return max(15, min(4 * total_params, 50))
    
    def _run_algorithm(self) -> Tuple[Dict, float, List]:
        """Run DE algorithm"""
        self.logger.info("Running DE algorithm")
        
        # Initialize population
        initial_params = self.parameter_manager.get_initial_parameters()
        self._initialize_population(initial_params)
        
        # Run DE generations
        return self._run_de_algorithm()
    
    def _initialize_population(self, initial_params: Dict[str, np.ndarray]) -> None:
        """Initialize DE population"""
        self.logger.info("Initializing DE population")
        
        param_count = len(self.parameter_manager.all_param_names)
        
        # Initialize random population in normalized space [0,1]
        self.population = np.random.random((self.population_size, param_count))
        self.population_scores = np.full(self.population_size, np.nan)
        
        # Set first individual to initial parameters
        if initial_params:
            initial_normalized = self.parameter_manager.normalize_parameters(initial_params)
            self.population[0] = np.clip(initial_normalized, 0, 1)
        
        # Evaluate initial population
        self.logger.info("Evaluating initial population...")
        self._evaluate_population()
        
        # Find best individual
        best_idx = np.nanargmax(self.population_scores)
        if not np.isnan(self.population_scores[best_idx]):
            self.best_score = self.population_scores[best_idx]
            self.best_params = self.parameter_manager.denormalize_parameters(self.population[best_idx])
        
        # Record initial generation
        self._record_generation(0)
        
        self.logger.info(f"Initial population evaluated. Best score: {self.best_score:.6f}")
    
    def _run_de_algorithm(self) -> Tuple[Dict, float, List]:
        """Core DE algorithm implementation"""
        for generation in range(1, self.max_iterations + 1):
            generation_start_time = datetime.now()
            
            # Create trial population
            trial_population = self._create_trial_population()
            
            # Evaluate trial population
            trial_scores = self._evaluate_trial_population(trial_population)
            
            # Selection phase
            improvements = 0
            for i in range(self.population_size):
                if not np.isnan(trial_scores[i]) and trial_scores[i] > self.population_scores[i]:
                    self.population[i] = trial_population[i].copy()
                    self.population_scores[i] = trial_scores[i]
                    improvements += 1
                    
                    # Update global best
                    if trial_scores[i] > self.best_score:
                        self.best_score = trial_scores[i]
                        self.best_params = self.parameter_manager.denormalize_parameters(trial_population[i])
                        self.logger.info(f"Gen {generation:3d}: NEW BEST! {self.target_metric}={self.best_score:.6f}")
            
            # Record generation
            self._record_generation(generation)
            
            # Log progress
            generation_duration = datetime.now() - generation_start_time
            self.logger.info(f"Gen {generation:3d}/{self.max_iterations}: "
                           f"Best={self.best_score:.6f}, Improvements={improvements}/{self.population_size} "
                           f"[{generation_duration.total_seconds():.1f}s]")
        
        return self.best_params, self.best_score, self.iteration_history
    
    def _create_trial_population(self) -> np.ndarray:
        """Create trial population using DE mutation and crossover"""
        trial_population = np.zeros_like(self.population)
        
        for i in range(self.population_size):
            # Select three random individuals different from target
            candidates = list(range(self.population_size))
            candidates.remove(i)
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            
            # Mutation: V = X_r1 + F * (X_r2 - X_r3)
            mutant = self.population[r1] + self.F * (self.population[r2] - self.population[r3])
            mutant = np.clip(mutant, 0, 1)
            
            # Crossover
            trial = self.population[i].copy()
            j_rand = np.random.randint(len(self.parameter_manager.all_param_names))
            
            for j in range(len(self.parameter_manager.all_param_names)):
                if np.random.random() < self.CR or j == j_rand:
                    trial[j] = mutant[j]
            
            trial_population[i] = trial
        
        return trial_population
    
    def _evaluate_population(self) -> None:
        """Evaluate current population"""
        if self.use_parallel:
            self._evaluate_population_parallel()
        else:
            self._evaluate_population_sequential()
    
    def _evaluate_trial_population(self, trial_population: np.ndarray) -> np.ndarray:
        """Evaluate trial population"""
        trial_scores = np.full(self.population_size, np.nan)
        
        if self.use_parallel:
            trial_scores = self._evaluate_trial_population_parallel(trial_population)
        else:
            for i in range(self.population_size):
                trial_scores[i] = self._evaluate_individual(trial_population[i])
        
        return trial_scores
    
    def _evaluate_population_sequential(self) -> None:
        """Evaluate population sequentially"""
        for i in range(self.population_size):
            if np.isnan(self.population_scores[i]):
                self.population_scores[i] = self._evaluate_individual(self.population[i])
    
    def _evaluate_population_parallel(self) -> None:
        """Evaluate population in parallel"""
        evaluation_tasks = []
        
        for i in range(self.population_size):
            if np.isnan(self.population_scores[i]):
                params = self.parameter_manager.denormalize_parameters(self.population[i])
                task = {
                    'individual_id': i,
                    'params': params,
                    'proc_id': i % self.num_processes,
                    'evaluation_id': f"pop_eval_{i:03d}"
                }
                evaluation_tasks.append(task)
        
        if evaluation_tasks:
            results = self._run_parallel_evaluations(evaluation_tasks)
            
            for result in results:
                individual_id = result['individual_id']
                score = result['score']
                self.population_scores[individual_id] = score if score is not None else float('-inf')
                
                if score is not None and score > self.best_score:
                    self.best_score = score
                    self.best_params = result['params'].copy()
    
    def _evaluate_trial_population_parallel(self, trial_population: np.ndarray) -> np.ndarray:
        """Evaluate trial population in parallel"""
        evaluation_tasks = []
        
        for i in range(self.population_size):
            params = self.parameter_manager.denormalize_parameters(trial_population[i])
            task = {
                'individual_id': i,
                'params': params,
                'proc_id': i % self.num_processes,
                'evaluation_id': f"trial_{len(self.iteration_history):03d}_{i:03d}"
            }
            evaluation_tasks.append(task)
        
        results = self._run_parallel_evaluations(evaluation_tasks)
        
        trial_scores = np.full(self.population_size, np.nan)
        for result in results:
            individual_id = result['individual_id']
            trial_scores[individual_id] = result['score'] if result['score'] is not None else float('-inf')
        
        return trial_scores
    
    def _record_generation(self, generation: int) -> None:
        """Record generation statistics"""
        valid_scores = self.population_scores[~np.isnan(self.population_scores)]
        
        generation_stats = {
            'generation': generation,
            'algorithm': 'DE',
            'best_score': self.best_score if self.best_score != float('-inf') else None,
            'best_params': self.best_params.copy() if self.best_params is not None else None,
            'mean_score': np.mean(valid_scores) if len(valid_scores) > 0 else None,
            'std_score': np.std(valid_scores) if len(valid_scores) > 0 else None,
            'worst_score': np.min(valid_scores) if len(valid_scores) > 0 else None,
            'valid_individuals': len(valid_scores),
            'population_scores': self.population_scores.copy()
        }
        
        self.iteration_history.append(generation_stats)


class PSOOptimizer(BaseOptimizer):
    """
    Particle Swarm Optimization (PSO) Optimizer for CONFLUENCE
    
    Implements the PSO algorithm with swarm-based optimization where particles
    move through parameter space guided by their personal best and global best.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        
        # PSO-specific parameters
        self.swarm_size = config.get('SWRMSIZE', 20)
        self.c1 = config.get('PSO_COGNITIVE_PARAM', 1.5)  # Personal best influence
        self.c2 = config.get('PSO_SOCIAL_PARAM', 1.5)     # Global best influence
        self.w_initial = config.get('PSO_INERTIA_WEIGHT', 0.7)
        self.w_reduction_rate = config.get('PSO_INERTIA_REDUCTION_RATE', 0.99)
        
        # PSO state variables
        self.swarm_positions = None      # Current positions (parameters)
        self.swarm_velocities = None     # Current velocities
        self.swarm_scores = None         # Current fitness scores
        self.personal_best_positions = None  # Personal best positions
        self.personal_best_scores = None     # Personal best scores
        self.global_best_position = None     # Global best position
        self.global_best_score = float('-inf')  # Global best score
        self.current_inertia = self.w_initial   # Current inertia weight
    
    def get_algorithm_name(self) -> str:
        return "PSO"
    
    def _run_algorithm(self) -> Tuple[Dict, float, List]:
        """Run PSO algorithm"""
        self.logger.info(f"Running PSO algorithm with {self.swarm_size} particles")
        self.logger.info(f"PSO parameters: c1={self.c1}, c2={self.c2}, w_initial={self.w_initial}")
        
        # Initialize swarm
        initial_params = self.parameter_manager.get_initial_parameters()
        self._initialize_swarm(initial_params)
        
        # Run PSO iterations
        return self._run_pso_algorithm()
    
    def _initialize_swarm(self, initial_params: Dict[str, np.ndarray]) -> None:
        """Initialize PSO swarm with positions, velocities, and personal bests"""
        self.logger.info("Initializing PSO swarm")
        
        param_count = len(self.parameter_manager.all_param_names)
        
        # Initialize swarm positions randomly in normalized space [0,1]
        self.swarm_positions = np.random.random((self.swarm_size, param_count))
        self.swarm_scores = np.full(self.swarm_size, np.nan)
        
        # Set first particle to initial parameters if available
        if initial_params:
            initial_normalized = self.parameter_manager.normalize_parameters(initial_params)
            self.swarm_positions[0] = np.clip(initial_normalized, 0, 1)
        
        # Initialize velocities (small random values)
        velocity_scale = 0.1  # 10% of parameter space
        self.swarm_velocities = np.random.uniform(-velocity_scale, velocity_scale, 
                                                 (self.swarm_size, param_count))
        
        # Initialize personal bests (copy of initial positions)
        self.personal_best_positions = self.swarm_positions.copy()
        self.personal_best_scores = np.full(self.swarm_size, float('-inf'))
        
        # Evaluate initial swarm
        self.logger.info("Evaluating initial swarm...")
        self._evaluate_swarm()
        
        # Initialize personal bests and global best
        self._update_personal_bests()
        self._update_global_best()
        
        # Record initial generation
        self._record_generation(0)
        
        self.logger.info(f"Initial swarm evaluated. Global best score: {self.global_best_score:.6f}")
    
    def _run_pso_algorithm(self) -> Tuple[Dict, float, List]:
        """Core PSO algorithm implementation"""
        for iteration in range(1, self.max_iterations + 1):
            iteration_start_time = datetime.now()
            
            # Update inertia weight (linearly decreasing)
            self.current_inertia *= self.w_reduction_rate
            
            # Update particle velocities and positions
            self._update_velocities()
            self._update_positions()
            
            # Evaluate swarm at new positions
            self._evaluate_swarm()
            
            # Update personal bests and global best
            improvements = self._update_personal_bests()
            global_improved = self._update_global_best()
            
            if global_improved:
                self.logger.info(f"Iter {iteration:3d}: NEW GLOBAL BEST! {self.target_metric}={self.global_best_score:.6f}")
            
            # Record iteration statistics
            self._record_generation(iteration)
            
            # Log progress
            iteration_duration = datetime.now() - iteration_start_time
            avg_score = np.nanmean(self.swarm_scores)
            self.logger.info(f"Iter {iteration:3d}/{self.max_iterations}: "
                           f"Global={self.global_best_score:.6f}, Avg={avg_score:.6f}, "
                           f"PB_improvements={improvements}, w={self.current_inertia:.4f} "
                           f"[{iteration_duration.total_seconds():.1f}s]")
        
        # Convert global best back to parameter dictionary
        self.best_score = self.global_best_score
        self.best_params = self.parameter_manager.denormalize_parameters(self.global_best_position)
        
        return self.best_params, self.best_score, self.iteration_history
    
    def _update_velocities(self) -> None:
        """Update particle velocities using PSO equation"""
        param_count = len(self.parameter_manager.all_param_names)
        
        for i in range(self.swarm_size):
            # Generate random vectors for cognitive and social components
            r1 = np.random.random(param_count)
            r2 = np.random.random(param_count)
            
            # Calculate cognitive component (attraction to personal best)
            cognitive = self.c1 * r1 * (self.personal_best_positions[i] - self.swarm_positions[i])
            
            # Calculate social component (attraction to global best)
            social = self.c2 * r2 * (self.global_best_position - self.swarm_positions[i])
            
            # Update velocity: v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
            self.swarm_velocities[i] = (self.current_inertia * self.swarm_velocities[i] + 
                                      cognitive + social)
            
            # Velocity clamping to prevent excessive velocities
            velocity_max = 0.2  # Maximum velocity as fraction of parameter space
            self.swarm_velocities[i] = np.clip(self.swarm_velocities[i], 
                                             -velocity_max, velocity_max)
    
    def _update_positions(self) -> None:
        """Update particle positions using velocities"""
        # Update positions: x = x + v
        self.swarm_positions += self.swarm_velocities
        
        # Handle boundary conditions - reflect particles that go out of bounds
        for i in range(self.swarm_size):
            for j in range(len(self.parameter_manager.all_param_names)):
                if self.swarm_positions[i, j] < 0:
                    self.swarm_positions[i, j] = -self.swarm_positions[i, j]
                    self.swarm_velocities[i, j] = -self.swarm_velocities[i, j]
                elif self.swarm_positions[i, j] > 1:
                    self.swarm_positions[i, j] = 2.0 - self.swarm_positions[i, j]
                    self.swarm_velocities[i, j] = -self.swarm_velocities[i, j]
        
        # Final clipping to ensure positions stay in [0,1]
        self.swarm_positions = np.clip(self.swarm_positions, 0, 1)
    
    def _evaluate_swarm(self) -> None:
        """Evaluate all particles in the swarm"""
        if self.use_parallel:
            self._evaluate_swarm_parallel()
        else:
            self._evaluate_swarm_sequential()
    
    def _evaluate_swarm_sequential(self) -> None:
        """Evaluate swarm sequentially"""
        for i in range(self.swarm_size):
            self.swarm_scores[i] = self._evaluate_individual(self.swarm_positions[i])
    
    def _evaluate_swarm_parallel(self) -> None:
        """Evaluate swarm in parallel"""
        evaluation_tasks = []
        
        for i in range(self.swarm_size):
            params = self.parameter_manager.denormalize_parameters(self.swarm_positions[i])
            task = {
                'individual_id': i,
                'params': params,
                'proc_id': i % self.num_processes,
                'evaluation_id': f"pso_{len(self.iteration_history):03d}_{i:03d}"
            }
            evaluation_tasks.append(task)
        
        results = self._run_parallel_evaluations(evaluation_tasks)
        
        # Update scores from results
        for result in results:
            particle_id = result['individual_id']
            score = result['score']
            self.swarm_scores[particle_id] = score if score is not None else float('-inf')
    
    def _update_personal_bests(self) -> int:
        """Update personal best positions and scores"""
        improvements = 0
        
        for i in range(self.swarm_size):
            if not np.isnan(self.swarm_scores[i]) and self.swarm_scores[i] > self.personal_best_scores[i]:
                self.personal_best_scores[i] = self.swarm_scores[i]
                self.personal_best_positions[i] = self.swarm_positions[i].copy()
                improvements += 1
        
        return improvements
    
    def _update_global_best(self) -> bool:
        """Update global best position and score"""
        # Find best personal best score
        valid_scores = self.personal_best_scores[~np.isnan(self.personal_best_scores)]
        if len(valid_scores) == 0:
            return False
        
        best_idx = np.nanargmax(self.personal_best_scores)
        best_score = self.personal_best_scores[best_idx]
        
        # Update global best if improved
        if best_score > self.global_best_score:
            self.global_best_score = best_score
            self.global_best_position = self.personal_best_positions[best_idx].copy()
            return True
        
        return False
    
    def _record_generation(self, generation: int) -> None:
        """Record PSO generation statistics"""
        valid_scores = self.swarm_scores[~np.isnan(self.swarm_scores)]
        valid_pb_scores = self.personal_best_scores[~np.isnan(self.personal_best_scores)]
        
        # Calculate swarm diversity (average distance between particles)
        diversity = self._calculate_swarm_diversity()
        
        generation_stats = {
            'generation': generation,
            'algorithm': 'PSO',
            'global_best_score': self.global_best_score if self.global_best_score != float('-inf') else None,
            'best_params': self.parameter_manager.denormalize_parameters(self.global_best_position).copy() if hasattr(self, 'global_best_position') and self.global_best_position is not None else None,
            'mean_current_score': np.mean(valid_scores) if len(valid_scores) > 0 else None,
            'std_current_score': np.std(valid_scores) if len(valid_scores) > 0 else None,
            'worst_current_score': np.min(valid_scores) if len(valid_scores) > 0 else None,
            'mean_personal_best': np.mean(valid_pb_scores) if len(valid_pb_scores) > 0 else None,
            'std_personal_best': np.std(valid_pb_scores) if len(valid_pb_scores) > 0 else None,
            'valid_particles': len(valid_scores),
            'swarm_diversity': diversity,
            'current_inertia': self.current_inertia,
            'swarm_scores': self.swarm_scores.copy(),
            'personal_best_scores': self.personal_best_scores.copy()
        }
        
        # Maintain compatibility with base class expectations
        generation_stats['best_score'] = generation_stats['global_best_score']
        generation_stats['mean_score'] = generation_stats['mean_current_score']
        generation_stats['std_score'] = generation_stats['std_current_score']
        generation_stats['worst_score'] = generation_stats['worst_current_score']
        generation_stats['valid_individuals'] = generation_stats['valid_particles']
        generation_stats['population_scores'] = generation_stats['swarm_scores']
        
        self.iteration_history.append(generation_stats)
    
    def _calculate_swarm_diversity(self) -> float:
        """Calculate swarm diversity as average euclidean distance between particles"""
        try:
            if self.swarm_positions is None or len(self.swarm_positions) < 2:
                return 0.0
            
            total_distance = 0.0
            count = 0
            
            for i in range(self.swarm_size):
                for j in range(i + 1, self.swarm_size):
                    distance = np.linalg.norm(self.swarm_positions[i] - self.swarm_positions[j])
                    total_distance += distance
                    count += 1
            
            return total_distance / count if count > 0 else 0.0
            
        except Exception:
            return 0.0

class NSGA2Optimizer(BaseOptimizer):
    """
    Non-dominated Sorting Genetic Algorithm II (NSGA-II) for CONFLUENCE
    
    Implements the NSGA-II multi-objective optimization algorithm. Uses both
    NSE and KGE as objectives for streamflow calibration, producing a Pareto
    front of non-dominated solutions.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        
        # NSGA-II specific parameters
        self.population_size = self._determine_population_size()
        self.crossover_rate = config.get('NSGA2_CROSSOVER_RATE', 0.9)
        self.mutation_rate = config.get('NSGA2_MUTATION_RATE', 0.1)
        self.eta_c = config.get('NSGA2_ETA_C', 20)  # Crossover distribution index
        self.eta_m = config.get('NSGA2_ETA_M', 20)  # Mutation distribution index
        
        # Multi-objective setup
        self.objectives = ['NSE', 'KGE']  # Both NSE and KGE as objectives
        self.num_objectives = len(self.objectives)
        
        # NSGA-II state variables
        self.population = None
        self.population_objectives = None  # Shape: (pop_size, num_objectives)
        self.population_ranks = None
        self.population_crowding_distances = None
        self.pareto_front = None  # Best non-dominated solutions
        
        # Override single-objective variables for compatibility
        self.best_score = None  # Not applicable for multi-objective
        self.best_params = None  # Will store one representative solution
    
    def get_algorithm_name(self) -> str:
        return "NSGA2"
    
    def _determine_population_size(self) -> int:
        """Determine appropriate population size - NSGA-II typically needs larger populations"""
        config_pop_size = self.config.get('POPULATION_SIZE')
        if config_pop_size:
            return config_pop_size
        
        # NSGA-II typically needs larger populations for good diversity
        total_params = (len(self.config.get('PARAMS_TO_CALIBRATE', '').split(',')) +
                       len(self.config.get('BASIN_PARAMS_TO_CALIBRATE', '').split(',')) +
                       (2 if self.config.get('CALIBRATE_DEPTH', False) else 0) +
                       (len(self.config.get('MIZUROUTE_PARAMS_TO_CALIBRATE', '').split(',')) if self.config.get('CALIBRATE_MIZUROUTE', False) else 0))
        
        return max(50, min(8 * total_params, 100))  # Larger than single-objective
    
    def _run_algorithm(self) -> Tuple[Dict, float, List]:
        """Run NSGA-II algorithm"""
        self.logger.info(f"Running NSGA-II algorithm with {self.population_size} individuals")
        self.logger.info(f"Multi-objective optimization using: {', '.join(self.objectives)}")
        
        # Initialize population
        initial_params = self.parameter_manager.get_initial_parameters()
        self._initialize_population(initial_params)
        
        # Run NSGA-II generations
        return self._run_nsga2_algorithm()
    
    def _initialize_population(self, initial_params: Dict[str, np.ndarray]) -> None:
        """Initialize NSGA-II population"""
        self.logger.info("Initializing NSGA-II population")
        
        param_count = len(self.parameter_manager.all_param_names)
        
        # Initialize random population in normalized space [0,1]
        self.population = np.random.random((self.population_size, param_count))
        self.population_objectives = np.full((self.population_size, self.num_objectives), np.nan)
        
        # Set first individual to initial parameters if available
        if initial_params:
            initial_normalized = self.parameter_manager.normalize_parameters(initial_params)
            self.population[0] = np.clip(initial_normalized, 0, 1)
        
        # Evaluate initial population
        self.logger.info("Evaluating initial population for multi-objective optimization...")
        self._evaluate_population_multiobjective()
        
        # Perform initial non-dominated sorting and crowding distance calculation
        self._perform_nsga2_selection()
        
        # Find representative solution for compatibility
        self._update_representative_solution()
        
        # Record initial generation
        self._record_generation(0)
        
        front_1_size = np.sum(self.population_ranks == 0)
        self.logger.info(f"Initial population evaluated. Pareto front size: {front_1_size}")
    
    def _run_nsga2_algorithm(self) -> Tuple[Dict, float, List]:
        """Core NSGA-II algorithm implementation"""
        for generation in range(1, self.max_iterations + 1):
            generation_start_time = datetime.now()
            
            # Generate offspring population
            offspring = self._generate_offspring()
            
            # Evaluate offspring
            offspring_objectives = self._evaluate_offspring(offspring)
            
            # Combine parent and offspring populations
            combined_population = np.vstack([self.population, offspring])
            combined_objectives = np.vstack([self.population_objectives, offspring_objectives])
            
            # Environmental selection using NSGA-II
            selected_indices = self._environmental_selection(combined_population, combined_objectives)
            
            # Update population
            self.population = combined_population[selected_indices]
            self.population_objectives = combined_objectives[selected_indices]
            
            # Perform non-dominated sorting and crowding distance
            self._perform_nsga2_selection()
            
            # Update representative solution
            self._update_representative_solution()
            
            # Record generation statistics
            self._record_generation(generation)
            
            # Log progress
            generation_duration = datetime.now() - generation_start_time
            front_1_size = np.sum(self.population_ranks == 0)
            avg_nse = np.nanmean(self.population_objectives[:, 0])
            avg_kge = np.nanmean(self.population_objectives[:, 1])
            
            self.logger.info(f"Gen {generation:3d}/{self.max_iterations}: "
                           f"Pareto_front={front_1_size}, Avg_NSE={avg_nse:.4f}, "
                           f"Avg_KGE={avg_kge:.4f} [{generation_duration.total_seconds():.1f}s]")
        
        # Final Pareto front extraction
        pareto_indices = np.where(self.population_ranks == 0)[0]
        self.pareto_front = {
            'solutions': self.population[pareto_indices],
            'objectives': self.population_objectives[pareto_indices],
            'parameters': [self.parameter_manager.denormalize_parameters(sol) for sol in self.population[pareto_indices]]
        }
        
        self.logger.info(f"Optimization completed. Final Pareto front size: {len(pareto_indices)}")
        
        return self.best_params, self.best_score, self.iteration_history
    
    def _evaluate_population_multiobjective(self) -> None:
        """Evaluate population for multiple objectives"""
        if self.use_parallel:
            self._evaluate_population_parallel_multiobjective()
        else:
            self._evaluate_population_sequential_multiobjective()
    
    def _evaluate_population_sequential_multiobjective(self) -> None:
        """Evaluate population sequentially for multiple objectives"""
        for i in range(self.population_size):
            if np.any(np.isnan(self.population_objectives[i])):
                objectives = self._evaluate_individual_multiobjective(self.population[i])
                self.population_objectives[i] = objectives
    
    def _evaluate_population_parallel_multiobjective(self) -> None:
        """Evaluate population in parallel for multiple objectives"""
        evaluation_tasks = []
        
        for i in range(self.population_size):
            if np.any(np.isnan(self.population_objectives[i])):
                params = self.parameter_manager.denormalize_parameters(self.population[i])
                task = {
                    'individual_id': i,
                    'params': params,
                    'proc_id': i % self.num_processes,
                    'evaluation_id': f"nsga2_pop_{i:03d}"
                }
                evaluation_tasks.append(task)
        
        if evaluation_tasks:
            results = self._run_parallel_evaluations_multiobjective(evaluation_tasks)
            
            for result in results:
                individual_id = result['individual_id']
                objectives = result.get('objectives')
                if objectives is not None:
                    self.population_objectives[individual_id] = objectives
                else:
                    # Fill with worst possible values if evaluation failed
                    self.population_objectives[individual_id] = [-1.0, -1.0]  # Bad NSE and KGE
    
    def _evaluate_individual_multiobjective(self, normalized_params: np.ndarray) -> np.ndarray:
        """Evaluate individual for multiple objectives (NSE and KGE)"""
        try:
            # Denormalize parameters
            params = self.parameter_manager.denormalize_parameters(normalized_params)
            
            # Apply parameters to files
            if not self.parameter_manager.apply_parameters(params):
                return np.array([-1.0, -1.0])  # Bad NSE and KGE
            
            # Run models
            if not self.model_executor.run_models(
                self.summa_sim_dir, 
                self.mizuroute_sim_dir, 
                self.optimization_settings_dir
            ):
                return np.array([-1.0, -1.0])
            
            # Calculate performance metrics
            metrics = self.calibration_target.calculate_metrics(self.summa_sim_dir, self.mizuroute_sim_dir)
            if not metrics:
                return np.array([-1.0, -1.0])
            
            # Extract NSE and KGE
            nse = self._extract_specific_metric(metrics, 'NSE')
            kge = self._extract_specific_metric(metrics, 'KGE')
            
            # Handle NaN values
            if nse is None or np.isnan(nse):
                nse = -1.0
            if kge is None or np.isnan(kge):
                kge = -1.0
            
            return np.array([nse, kge])
            
        except Exception as e:
            self.logger.debug(f"Multi-objective evaluation failed: {str(e)}")
            return np.array([-1.0, -1.0])
    
    def _extract_specific_metric(self, metrics: Dict[str, float], metric_name: str) -> Optional[float]:
        """Extract specific metric from metrics dictionary"""
        # Try exact match
        if metric_name in metrics:
            return metrics[metric_name]
        
        # Try with calibration prefix
        calib_key = f"Calib_{metric_name}"
        if calib_key in metrics:
            return metrics[calib_key]
        
        # Try without prefix
        for key, value in metrics.items():
            if key.endswith(f"_{metric_name}"):
                return value
        
        return None
    
    def _evaluate_population_parallel_multiobjective(self) -> None:
        """Evaluate population in parallel for multiple objectives using proper MPI framework"""
        evaluation_tasks = []
        
        for i in range(self.population_size):
            if np.any(np.isnan(self.population_objectives[i])):
                params = self.parameter_manager.denormalize_parameters(self.population[i])
                task = {
                    'individual_id': i,
                    'params': params,
                    'proc_id': i % self.num_processes,
                    'evaluation_id': f"nsga2_pop_{i:03d}"
                }
                evaluation_tasks.append(task)
        
        if evaluation_tasks:
            # Use the base class MPI parallel evaluation framework
            results = self._run_parallel_evaluations(evaluation_tasks)
            
            # Process results for multi-objective
            for result in results:
                individual_id = result['individual_id']
                score = result.get('score')  # This is the target_metric (KGE by default)
                
                if score is not None:
                    # We got the target metric from parallel evaluation
                    # For NSGA-II, we need both NSE and KGE
                    # Since the parallel worker already ran the model and calculated metrics,
                    # we'll do a lightweight re-evaluation to get the other objective
                    
                    # Get the parameters that were just evaluated
                    params = result['params']
                    
                    # If target_metric is KGE, we have KGE and need NSE
                    # If target_metric is NSE, we have NSE and need KGE
                    if self.target_metric == 'KGE':
                        kge_score = score
                        # Do a quick local evaluation to get NSE
                        nse_score = self._get_alternative_metric(params, 'NSE')
                        objectives = np.array([nse_score, kge_score])
                    else:  # target_metric is NSE or other
                        nse_score = score if self.target_metric == 'NSE' else self._get_alternative_metric(params, 'NSE')
                        kge_score = self._get_alternative_metric(params, 'KGE')
                        objectives = np.array([nse_score, kge_score])
                    
                    self.population_objectives[individual_id] = objectives
                else:
                    # Evaluation failed - use bad values
                    self.population_objectives[individual_id] = np.array([-1.0, -1.0])

    def _get_alternative_metric(self, params: Dict, metric_name: str) -> float:
        """
        Get alternative metric without re-running the model.
        This is a lightweight method that should ideally read from cached results.
        For now, it does a quick local evaluation - in a full implementation,
        you'd cache the full metrics from the parallel evaluation.
        """
        try:
            # In an ideal implementation, this would read cached metrics from the parallel run
            # For now, we do a quick evaluation (this is still faster than fully sequential)
            normalized_params = self.parameter_manager.normalize_parameters(params)
            
            # Apply parameters to files
            if not self.parameter_manager.apply_parameters(params):
                return -1.0
            
            # Check if output files already exist from parallel run (they should)
            # and try to read metrics from them without re-running the model
            metrics = self.calibration_target.calculate_metrics(self.summa_sim_dir, self.mizuroute_sim_dir)
            if metrics:
                metric_value = self._extract_specific_metric(metrics, metric_name)
                return metric_value if metric_value is not None else -1.0
            else:
                return -1.0
                
        except Exception as e:
            self.logger.debug(f"Error getting alternative metric {metric_name}: {str(e)}")
            return -1.0

    def _run_parallel_evaluations_multiobjective(self, evaluation_tasks: List[Dict]) -> List[Dict]:
        """Run parallel evaluations for offspring using proper MPI framework"""
        # Use the base class MPI parallel evaluation framework
        results = self._run_parallel_evaluations(evaluation_tasks)
        
        # Convert to multi-objective format
        multiobjective_results = []
        for result in results:
            individual_id = result['individual_id']
            params = result['params']
            score = result.get('score')
            error = result.get('error')
            
            if score is not None and error is None:
                # Extract both objectives using the same approach as population evaluation
                if self.target_metric == 'KGE':
                    kge_score = score
                    nse_score = self._get_alternative_metric(params, 'NSE')
                    objectives = np.array([nse_score, kge_score])
                else:
                    nse_score = score if self.target_metric == 'NSE' else self._get_alternative_metric(params, 'NSE')
                    kge_score = self._get_alternative_metric(params, 'KGE')
                    objectives = np.array([nse_score, kge_score])
                
                multiobjective_results.append({
                    'individual_id': individual_id,
                    'params': params,
                    'objectives': objectives,
                    'error': None
                })
            else:
                multiobjective_results.append({
                    'individual_id': individual_id,
                    'params': params,
                    'objectives': None,
                    'error': error or 'Parallel evaluation failed'
                })
        
        return multiobjective_results
    
    def _generate_offspring(self) -> np.ndarray:
        """Generate offspring using tournament selection, crossover, and mutation"""
        offspring = np.zeros_like(self.population)
        
        for i in range(0, self.population_size, 2):
            # Tournament selection for parents
            parent1_idx = self._tournament_selection()
            parent2_idx = self._tournament_selection()
            
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child1, child2 = self._sbx_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1 = self._polynomial_mutation(child1)
            child2 = self._polynomial_mutation(child2)
            
            offspring[i] = child1
            if i + 1 < self.population_size:
                offspring[i + 1] = child2
        
        return offspring
    
    def _tournament_selection(self, tournament_size: int = 2) -> int:
        """Tournament selection based on NSGA-II criteria (rank and crowding distance)"""
        candidates = np.random.choice(self.population_size, tournament_size, replace=False)
        
        best_idx = candidates[0]
        for candidate in candidates[1:]:
            # Compare based on rank first, then crowding distance
            if (self.population_ranks[candidate] < self.population_ranks[best_idx] or
                (self.population_ranks[candidate] == self.population_ranks[best_idx] and
                 self.population_crowding_distances[candidate] > self.population_crowding_distances[best_idx])):
                best_idx = candidate
        
        return best_idx
    
    def _sbx_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX)"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(len(parent1)):
            if np.random.random() < 0.5:  # Per-variable crossover probability
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    u = np.random.random()
                    
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (self.eta_c + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (self.eta_c + 1))
                    
                    child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                    child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
        
        # Ensure bounds [0, 1]
        child1 = np.clip(child1, 0, 1)
        child2 = np.clip(child2, 0, 1)
        
        return child1, child2
    
    def _polynomial_mutation(self, individual: np.ndarray) -> np.ndarray:
        """Polynomial mutation"""
        mutated = individual.copy()
        
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                u = np.random.random()
                
                if u < 0.5:
                    delta = (2 * u) ** (1.0 / (self.eta_m + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1.0 / (self.eta_m + 1))
                
                mutated[i] = individual[i] + delta
        
        # Ensure bounds [0, 1]
        return np.clip(mutated, 0, 1)
    
    def _evaluate_offspring(self, offspring: np.ndarray) -> np.ndarray:
        """Evaluate offspring population"""
        offspring_objectives = np.full((len(offspring), self.num_objectives), np.nan)
        
        if self.use_parallel:
            # Parallel evaluation of offspring
            evaluation_tasks = []
            for i in range(len(offspring)):
                params = self.parameter_manager.denormalize_parameters(offspring[i])
                task = {
                    'individual_id': i,
                    'params': params,
                    'proc_id': i % self.num_processes,
                    'evaluation_id': f"nsga2_off_{len(self.iteration_history):03d}_{i:03d}"
                }
                evaluation_tasks.append(task)
            
            results = self._run_parallel_evaluations_multiobjective(evaluation_tasks)
            
            for result in results:
                individual_id = result['individual_id']
                objectives = result.get('objectives')
                if objectives is not None:
                    offspring_objectives[individual_id] = objectives
                else:
                    offspring_objectives[individual_id] = [-1.0, -1.0]
        else:
            # Sequential evaluation
            for i in range(len(offspring)):
                offspring_objectives[i] = self._evaluate_individual_multiobjective(offspring[i])
        
        return offspring_objectives
    
    def _environmental_selection(self, combined_population: np.ndarray, 
                               combined_objectives: np.ndarray) -> np.ndarray:
        """NSGA-II environmental selection"""
        # Non-dominated sorting
        fronts = self._non_dominated_sorting(combined_objectives)
        
        selected_indices = []
        
        # Add complete fronts until we exceed population size
        for front in fronts:
            if len(selected_indices) + len(front) <= self.population_size:
                selected_indices.extend(front)
            else:
                # Partially fill with best individuals from this front based on crowding distance
                remaining_slots = self.population_size - len(selected_indices)
                
                # Calculate crowding distance for this front
                crowding_distances = self._calculate_crowding_distance(combined_objectives[front])
                
                # Sort by crowding distance (descending)
                front_with_distances = list(zip(front, crowding_distances))
                front_with_distances.sort(key=lambda x: x[1], reverse=True)
                
                # Take the best ones
                for i in range(remaining_slots):
                    selected_indices.append(front_with_distances[i][0])
                
                break
        
        return np.array(selected_indices)
    
    def _perform_nsga2_selection(self) -> None:
        """Perform non-dominated sorting and crowding distance calculation"""
        # Non-dominated sorting
        fronts = self._non_dominated_sorting(self.population_objectives)
        
        # Assign ranks
        self.population_ranks = np.zeros(self.population_size)
        for rank, front in enumerate(fronts):
            self.population_ranks[front] = rank
        
        # Calculate crowding distances
        self.population_crowding_distances = np.zeros(self.population_size)
        for front in fronts:
            if len(front) > 0:
                front_distances = self._calculate_crowding_distance(self.population_objectives[front])
                self.population_crowding_distances[front] = front_distances
    
    def _non_dominated_sorting(self, objectives: np.ndarray) -> List[List[int]]:
        """Fast non-dominated sorting"""
        n = len(objectives)
        domination_counts = np.zeros(n)  # Number of solutions that dominate each solution
        dominated_solutions = [[] for _ in range(n)]  # Solutions dominated by each solution
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(objectives[j], objectives[i]):
                        domination_counts[i] += 1
            
            if domination_counts[i] == 0:
                fronts[0].append(i)
        
        current_front = 0
        while len(fronts[current_front]) > 0:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            
            current_front += 1
            fronts.append(next_front)
        
        # Remove empty last front
        return fronts[:-1]
    
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2 (for maximization of both NSE and KGE)"""
        # For NSE and KGE, higher is better
        return np.all(obj1 >= obj2) and np.any(obj1 > obj2)
    
    def _calculate_crowding_distance(self, front_objectives: np.ndarray) -> np.ndarray:
        """Calculate crowding distance for a front"""
        n = len(front_objectives)
        if n <= 2:
            return np.full(n, float('inf'))
        
        distances = np.zeros(n)
        
        for obj_idx in range(self.num_objectives):
            # Sort by objective
            sorted_indices = np.argsort(front_objectives[:, obj_idx])
            
            # Boundary points get infinite distance
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Calculate range
            obj_range = front_objectives[sorted_indices[-1], obj_idx] - front_objectives[sorted_indices[0], obj_idx]
            
            if obj_range > 0:
                for i in range(1, n - 1):
                    distances[sorted_indices[i]] += (
                        front_objectives[sorted_indices[i + 1], obj_idx] - 
                        front_objectives[sorted_indices[i - 1], obj_idx]
                    ) / obj_range
        
        return distances
    
    def _update_representative_solution(self) -> None:
        """Update representative solution for compatibility with base class"""
        # Find best solution in first front based on a composite metric
        front_1_indices = np.where(self.population_ranks == 0)[0]
        
        if len(front_1_indices) > 0:
            # Use a composite metric (e.g., sum of normalized objectives)
            front_1_objectives = self.population_objectives[front_1_indices]
            
            # Normalize objectives to [0,1] range
            nse_norm = (front_1_objectives[:, 0] + 1) / 2  # NSE from [-1,1] to [0,1]
            kge_norm = (front_1_objectives[:, 1] + 1) / 2  # KGE from [-1,1] to [0,1]
            
            composite_scores = nse_norm + kge_norm  # Simple sum
            best_idx_in_front = np.argmax(composite_scores)
            best_idx = front_1_indices[best_idx_in_front]
            
            self.best_params = self.parameter_manager.denormalize_parameters(self.population[best_idx])
            self.best_score = composite_scores[best_idx_in_front]
        else:
            # Fallback if no valid solutions
            self.best_params = None
            self.best_score = -2.0  # Worst possible composite score
    
    def _record_generation(self, generation: int) -> None:
        """Record NSGA-II generation statistics"""
        valid_nse = self.population_objectives[:, 0][~np.isnan(self.population_objectives[:, 0])]
        valid_kge = self.population_objectives[:, 1][~np.isnan(self.population_objectives[:, 1])]
        
        # Count individuals in each front
        front_counts = {}
        for rank in range(int(np.max(self.population_ranks)) + 1):
            front_counts[f'front_{rank}'] = np.sum(self.population_ranks == rank)
        
        generation_stats = {
            'generation': generation,
            'algorithm': 'NSGA-II',
            'best_score': self.best_score,  # Composite score for compatibility
            'best_params': self.best_params.copy() if self.best_params is not None else None,
            
            # Multi-objective specific stats
            'pareto_front_size': front_counts.get('front_0', 0),
            'mean_nse': np.mean(valid_nse) if len(valid_nse) > 0 else None,
            'std_nse': np.std(valid_nse) if len(valid_nse) > 0 else None,
            'best_nse': np.max(valid_nse) if len(valid_nse) > 0 else None,
            'mean_kge': np.mean(valid_kge) if len(valid_kge) > 0 else None,
            'std_kge': np.std(valid_kge) if len(valid_kge) > 0 else None,
            'best_kge': np.max(valid_kge) if len(valid_kge) > 0 else None,
            
            # Front distribution
            'front_counts': front_counts,
            
            # Compatibility with base class
            'mean_score': np.mean(valid_nse + valid_kge) / 2 if len(valid_nse) > 0 and len(valid_kge) > 0 else None,
            'std_score': np.std(valid_nse + valid_kge) / 2 if len(valid_nse) > 0 and len(valid_kge) > 0 else None,
            'worst_score': np.min(valid_nse + valid_kge) / 2 if len(valid_nse) > 0 and len(valid_kge) > 0 else None,
            'valid_individuals': len(valid_nse),
            
            # Store raw data
            'population_objectives': self.population_objectives.copy(),
            'population_ranks': self.population_ranks.copy(),
            'population_crowding_distances': self.population_crowding_distances.copy()
        }
        
        self.iteration_history.append(generation_stats)
    
    def get_pareto_front(self) -> Dict[str, Any]:
        """Get the final Pareto front of solutions"""
        if hasattr(self, 'pareto_front') and self.pareto_front is not None:
            return self.pareto_front
        
        # Extract current Pareto front
        front_1_indices = np.where(self.population_ranks == 0)[0]
        
        if len(front_1_indices) > 0:
            return {
                'solutions': self.population[front_1_indices],
                'objectives': self.population_objectives[front_1_indices],
                'parameters': [self.parameter_manager.denormalize_parameters(sol) for sol in self.population[front_1_indices]],
                'nse_values': self.population_objectives[front_1_indices, 0],
                'kge_values': self.population_objectives[front_1_indices, 1]
            }
        else:
            return {'solutions': [], 'objectives': [], 'parameters': [], 'nse_values': [], 'kge_values': []}



class SCEUAOptimizer(BaseOptimizer):
    """
    Shuffled Complex Evolution - University of Arizona (SCE-UA) Optimizer for CONFLUENCE
    
    Implements the SCE-UA algorithm (Duan et al. 1992, 1993). Uses complexes of points that evolve and shuffle
    to explore the parameter space efficiently.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        
        # SCE-UA specific parameters
        self.num_complexes = config.get('NUMBER_OF_COMPLEXES', 2)
        self.points_per_subcomplex = config.get('POINTS_PER_SUBCOMPLEX', 5)
        self.points_per_complex = self._determine_points_per_complex()
        self.num_evolution_steps = config.get('NUMBER_OF_EVOLUTION_STEPS', 20)
        self.evolution_stagnation = config.get('EVOLUTION_STAGNATION', 5)
        self.percent_change_threshold = config.get('PERCENT_CHANGE_THRESHOLD', 0.01)
        
        # Calculate total population size
        self.population_size = self.num_complexes * self.points_per_complex
        
        # SCE-UA state variables
        self.population = None
        self.population_scores = None
        self.complexes = None  # List of complex indices
        self.convergence_history = []  # Track convergence metrics
        self.stagnation_counter = 0
        
        self.logger.info(f"SCE-UA configuration:")
        self.logger.info(f"  Complexes: {self.num_complexes}")
        self.logger.info(f"  Points per complex: {self.points_per_complex}")
        self.logger.info(f"  Total population: {self.population_size}")
        self.logger.info(f"  Points per subcomplex: {self.points_per_subcomplex}")
        self.logger.info(f"  Evolution steps: {self.num_evolution_steps}")
    
    def get_algorithm_name(self) -> str:
        return "SCE-UA"
    
    def _determine_points_per_complex(self) -> int:
        """Determine points per complex based on parameter count and complexes"""
        # SCE-UA rule: points per complex should be 2n+1 where n is number of parameters
        param_count = len(self.parameter_manager.all_param_names)
        recommended = 2 * param_count + 1
        
        # But also consider the specified population size if available
        if self.config.get('POPULATION_SIZE'):
            specified_pop = self.config.get('POPULATION_SIZE')
            points_per_complex = max(recommended, specified_pop // self.num_complexes)
        else:
            points_per_complex = recommended
        
        # Ensure minimum requirements for subcomplexes
        min_points = self.points_per_subcomplex * 2  # At least 2 subcomplexes
        points_per_complex = max(points_per_complex, min_points)
        
        return points_per_complex
    
    def _run_algorithm(self) -> Tuple[Dict, float, List]:
        """Run SCE-UA algorithm"""
        self.logger.info("Running SCE-UA algorithm")
        
        # Initialize population and complexes
        initial_params = self.parameter_manager.get_initial_parameters()
        self._initialize_population(initial_params)
        
        # Run SCE-UA iterations
        return self._run_sceua_algorithm()
    
    def _initialize_population(self, initial_params: Dict[str, np.ndarray]) -> None:
        """Initialize SCE-UA population and organize into complexes"""
        self.logger.info(f"Initializing SCE-UA population with {self.population_size} points")
        
        param_count = len(self.parameter_manager.all_param_names)
        
        # Initialize random population in normalized space [0,1]
        self.population = np.random.random((self.population_size, param_count))
        self.population_scores = np.full(self.population_size, np.nan)
        
        # Set first individual to initial parameters if available
        if initial_params:
            initial_normalized = self.parameter_manager.normalize_parameters(initial_params)
            self.population[0] = np.clip(initial_normalized, 0, 1)
        
        # Evaluate initial population
        self.logger.info("Evaluating initial population...")
        self._evaluate_population()
        
        # Sort population by fitness (best first)
        self._sort_population()
        
        # Initialize complexes
        self._initialize_complexes()
        
        # Find best individual
        if not np.isnan(self.population_scores[0]):
            self.best_score = self.population_scores[0]
            self.best_params = self.parameter_manager.denormalize_parameters(self.population[0])
        
        # Record initial generation
        self._record_generation(0)
        
        self.logger.info(f"Initial population evaluated. Best score: {self.best_score:.6f}")
    
    def _run_sceua_algorithm(self) -> Tuple[Dict, float, List]:
        """Core SCE-UA algorithm implementation"""
        for iteration in range(1, self.max_iterations + 1):
            iteration_start_time = datetime.now()
            
            # Evolve each complex
            total_improvements = 0
            for complex_id in range(self.num_complexes):
                improvements = self._evolve_complex(complex_id)
                total_improvements += improvements
            
            # Shuffle complexes (redistribute points among complexes)
            self._shuffle_complexes()
            
            # Sort entire population
            self._sort_population()
            
            # Update global best
            if self.population_scores[0] > self.best_score:
                self.best_score = self.population_scores[0]
                self.best_params = self.parameter_manager.denormalize_parameters(self.population[0])
                self.logger.info(f"Iter {iteration:3d}: NEW BEST! {self.target_metric}={self.best_score:.6f}")
            
            # Check convergence
            converged, convergence_info = self._check_convergence(iteration)
            
            # Record iteration statistics
            self._record_generation(iteration, convergence_info, total_improvements)
            
            # Log progress
            iteration_duration = datetime.now() - iteration_start_time
            avg_score = np.nanmean(self.population_scores)
            worst_score = np.nanmin(self.population_scores[~np.isnan(self.population_scores)])
            
            self.logger.info(f"Iter {iteration:3d}/{self.max_iterations}: "
                           f"Best={self.best_score:.6f}, Avg={avg_score:.6f}, "
                           f"Worst={worst_score:.6f}, Improvements={total_improvements}, "
                           f"Stagnation={self.stagnation_counter}/{self.evolution_stagnation} "
                           f"[{iteration_duration.total_seconds():.1f}s]")
            
            # Early termination if converged
            if converged:
                self.logger.info(f"SCE-UA converged after {iteration} iterations")
                self.logger.info(f"Convergence reason: {convergence_info['reason']}")
                break
        
        return self.best_params, self.best_score, self.iteration_history
    
    def _initialize_complexes(self) -> None:
        """Initialize complexes by systematic sampling"""
        self.complexes = []
        
        for i in range(self.num_complexes):
            complex_indices = []
            # Systematic sampling: take every kth point starting from i
            for j in range(self.points_per_complex):
                idx = (i + j * self.num_complexes) % self.population_size
                complex_indices.append(idx)
            self.complexes.append(complex_indices)
    
    def _evolve_complex(self, complex_id: int) -> int:
        """Evolve a single complex using Competitive Complex Evolution (CCE)"""
        complex_indices = self.complexes[complex_id]
        improvements = 0
        
        # Extract complex population and scores
        complex_population = self.population[complex_indices]
        complex_scores = self.population_scores[complex_indices]
        
        # Sort complex by fitness (best first)
        sorted_idx = np.argsort(complex_scores)[::-1]
        complex_population = complex_population[sorted_idx]
        complex_scores = complex_scores[sorted_idx]
        
        # Determine number of subcomplexes
        num_subcomplexes = self.points_per_complex // self.points_per_subcomplex
        
        # Evolve each subcomplex
        for subcomplex_id in range(num_subcomplexes):
            # Partition into subcomplex using triangular probability distribution
            subcomplex_indices = self._partition_subcomplex(subcomplex_id, num_subcomplexes)
            
            # Evolve subcomplex
            new_point, new_score = self._evolve_subcomplex(
                complex_population[subcomplex_indices], 
                complex_scores[subcomplex_indices]
            )
            
            # Replace worst point in subcomplex if improvement
            worst_idx_in_subcomplex = subcomplex_indices[-1]  # Last index (worst)
            if new_score > complex_scores[worst_idx_in_subcomplex]:
                complex_population[worst_idx_in_subcomplex] = new_point
                complex_scores[worst_idx_in_subcomplex] = new_score
                improvements += 1
        
        # Update main population with evolved complex
        for i, orig_idx in enumerate(complex_indices):
            self.population[orig_idx] = complex_population[sorted_idx[i]]
            self.population_scores[orig_idx] = complex_scores[sorted_idx[i]]
        
        return improvements
    
    def _partition_subcomplex(self, subcomplex_id: int, num_subcomplexes: int) -> List[int]:
        """Partition complex into subcomplex using triangular probability distribution"""
        subcomplex_indices = []
        
        # Systematic sampling for subcomplex formation
        for i in range(self.points_per_subcomplex):
            idx = subcomplex_id + i * num_subcomplexes
            if idx < self.points_per_complex:
                subcomplex_indices.append(idx)
        
        # If we don't have enough points, add randomly from remaining
        while len(subcomplex_indices) < self.points_per_subcomplex:
            remaining_indices = [i for i in range(self.points_per_complex) 
                               if i not in subcomplex_indices]
            if remaining_indices:
                subcomplex_indices.append(np.random.choice(remaining_indices))
            else:
                break
        
        return sorted(subcomplex_indices)
    
    def _evolve_subcomplex(self, subcomplex_population: np.ndarray, 
                          subcomplex_scores: np.ndarray) -> Tuple[np.ndarray, float]:
        """Evolve subcomplex using Competitive Complex Evolution (CCE)"""
        param_count = len(self.parameter_manager.all_param_names)
        
        # Sort subcomplex (best first)
        sorted_idx = np.argsort(subcomplex_scores)[::-1]
        sorted_population = subcomplex_population[sorted_idx]
        sorted_scores = subcomplex_scores[sorted_idx]
        
        best_point = sorted_population[0]
        worst_point = sorted_population[-1]
        
        # Calculate centroid of all points except worst
        centroid = np.mean(sorted_population[:-1], axis=0)
        
        # Generate new point using reflection
        alpha = 1.0  # Reflection coefficient
        new_point = centroid + alpha * (centroid - worst_point)
        
        # Ensure bounds [0,1]
        new_point = np.clip(new_point, 0, 1)
        
        # Evaluate new point
        new_score = self._evaluate_individual(new_point)
        
        # If new point is not better than worst, try contraction
        if new_score <= sorted_scores[-1]:
            # Contraction
            beta = 0.5  # Contraction coefficient
            new_point = centroid + beta * (worst_point - centroid)
            new_point = np.clip(new_point, 0, 1)
            new_score = self._evaluate_individual(new_point)
            
            # If still not better, generate random point
            if new_score <= sorted_scores[-1]:
                new_point = np.random.random(param_count)
                new_score = self._evaluate_individual(new_point)
        
        return new_point, new_score
    
    def _shuffle_complexes(self) -> None:
        """Shuffle complexes by redistributing points among complexes"""
        # Sort entire population by fitness
        self._sort_population()
        
        # Redistribute points to complexes using systematic sampling
        self._initialize_complexes()
    
    def _sort_population(self) -> None:
        """Sort population by fitness (best first)"""
        valid_indices = ~np.isnan(self.population_scores)
        valid_scores = self.population_scores[valid_indices]
        
        if len(valid_scores) > 0:
            # Sort indices by score (descending - best first)
            sorted_idx = np.argsort(self.population_scores)[::-1]
            
            # Handle NaN values (put them at the end)
            nan_indices = np.where(np.isnan(self.population_scores))[0]
            valid_indices = np.where(~np.isnan(self.population_scores))[0]
            valid_sorted = valid_indices[np.argsort(self.population_scores[valid_indices])[::-1]]
            
            all_sorted = np.concatenate([valid_sorted, nan_indices])
            
            self.population = self.population[all_sorted]
            self.population_scores = self.population_scores[all_sorted]
    
    def _check_convergence(self, iteration: int) -> Tuple[bool, Dict[str, Any]]:
        """Check SCE-UA convergence criteria"""
        convergence_info = {
            'iteration': iteration,
            'percent_change': None,
            'stagnation_count': self.stagnation_counter,
            'reason': None
        }
        
        # Calculate percent change in best fitness
        if len(self.convergence_history) > 0:
            prev_best = self.convergence_history[-1]['best_score']
            current_best = self.best_score
            
            if prev_best != 0:
                percent_change = abs((current_best - prev_best) / prev_best) * 100
                convergence_info['percent_change'] = percent_change
                
                # Check percent change criterion
                if percent_change < self.percent_change_threshold:
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
        
        # Record convergence info
        convergence_info['best_score'] = self.best_score
        self.convergence_history.append(convergence_info)
        
        # Check convergence criteria
        if self.stagnation_counter >= self.evolution_stagnation:
            convergence_info['reason'] = f"Stagnation: {self.stagnation_counter} iterations with <{self.percent_change_threshold}% change"
            return True, convergence_info
        
        return False, convergence_info
    
    def _evaluate_population(self) -> None:
        """Evaluate population using inherited methods"""
        if self.use_parallel:
            self._evaluate_population_parallel()
        else:
            self._evaluate_population_sequential()
    
    def _evaluate_population_sequential(self) -> None:
        """Evaluate population sequentially"""
        for i in range(self.population_size):
            if np.isnan(self.population_scores[i]):
                self.population_scores[i] = self._evaluate_individual(self.population[i])
    
    def _evaluate_population_parallel(self) -> None:
        """Evaluate population in parallel"""
        evaluation_tasks = []
        
        for i in range(self.population_size):
            if np.isnan(self.population_scores[i]):
                params = self.parameter_manager.denormalize_parameters(self.population[i])
                task = {
                    'individual_id': i,
                    'params': params,
                    'proc_id': i % self.num_processes,
                    'evaluation_id': f"sceua_pop_{i:03d}"
                }
                evaluation_tasks.append(task)
        
        if evaluation_tasks:
            results = self._run_parallel_evaluations(evaluation_tasks)
            
            for result in results:
                individual_id = result['individual_id']
                score = result['score']
                self.population_scores[individual_id] = score if score is not None else float('-inf')
                
                if score is not None and score > self.best_score:
                    self.best_score = score
                    self.best_params = result['params'].copy()
    
    def _record_generation(self, generation: int, convergence_info: Dict = None, 
                          improvements: int = 0) -> None:
        """Record SCE-UA generation statistics"""
        valid_scores = self.population_scores[~np.isnan(self.population_scores)]
        
        # Calculate complex statistics
        complex_stats = self._calculate_complex_statistics()
        
        generation_stats = {
            'generation': generation,
            'algorithm': 'SCE-UA',
            'best_score': self.best_score if self.best_score != float('-inf') else None,
            'best_params': self.best_params.copy() if self.best_params is not None else None,
            'mean_score': np.mean(valid_scores) if len(valid_scores) > 0 else None,
            'std_score': np.std(valid_scores) if len(valid_scores) > 0 else None,
            'worst_score': np.min(valid_scores) if len(valid_scores) > 0 else None,
            'valid_individuals': len(valid_scores),
            'population_scores': self.population_scores.copy(),
            
            # SCE-UA specific statistics
            'num_complexes': self.num_complexes,
            'points_per_complex': self.points_per_complex,
            'improvements_this_iteration': improvements,
            'stagnation_counter': self.stagnation_counter,
            'complex_best_scores': complex_stats['best_scores'],
            'complex_mean_scores': complex_stats['mean_scores'],
            'complex_diversity': complex_stats['diversity']
        }
        
        # Add convergence information if available
        if convergence_info:
            generation_stats.update({
                'percent_change': convergence_info.get('percent_change'),
                'convergence_reason': convergence_info.get('reason')
            })
        
        self.iteration_history.append(generation_stats)
    
    def _calculate_complex_statistics(self) -> Dict[str, List]:
        """Calculate statistics for each complex"""
        complex_stats = {
            'best_scores': [],
            'mean_scores': [],
            'diversity': []
        }
        
        for complex_indices in self.complexes:
            complex_scores = self.population_scores[complex_indices]
            valid_scores = complex_scores[~np.isnan(complex_scores)]
            
            if len(valid_scores) > 0:
                complex_stats['best_scores'].append(np.max(valid_scores))
                complex_stats['mean_scores'].append(np.mean(valid_scores))
                
                # Calculate diversity as standard deviation of parameters
                complex_population = self.population[complex_indices]
                diversity = np.mean(np.std(complex_population, axis=0))
                complex_stats['diversity'].append(diversity)
            else:
                complex_stats['best_scores'].append(float('-inf'))
                complex_stats['mean_scores'].append(float('-inf'))
                complex_stats['diversity'].append(0.0)
        
        return complex_stats
    
    def get_convergence_history(self) -> List[Dict]:
        """Get detailed convergence history for analysis"""
        return self.convergence_history
    
    def get_complex_statistics(self) -> Dict:
        """Get current complex statistics for analysis"""
        return {
            'num_complexes': self.num_complexes,
            'points_per_complex': self.points_per_complex,
            'total_population': self.population_size,
            'complex_indices': self.complexes,
            'complex_stats': self._calculate_complex_statistics()
        }

# ============= WORKER FUNCTIONS FOR PARALLEL PROCESSING =============



def _evaluate_parameters_worker_safe(task_data: Dict) -> Dict:
    
    # Set up signal handler for clean termination
    def signal_handler(signum, frame):
        sys.exit(1)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Set process-specific environment for isolation
    process_id = os.getpid()
    
    # Force single-threaded execution and disable problematic file locking
    os.environ.update({
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'NETCDF_DISABLE_LOCKING': '1',
        'HDF5_USE_FILE_LOCKING': 'FALSE',
        'HDF5_DISABLE_VERSION_CHECK': '1',
    })
    
    # Add small random delay to stagger file system access
    initial_delay = random.uniform(0.1, 0.8)
    time.sleep(initial_delay)
    
    try:
        # Force garbage collection at start
        gc.collect()
        
        # Enhanced logging setup for debugging
        proc_id = task_data.get('proc_id', 0)
        individual_id = task_data.get('individual_id', -1)
        
        # Try the evaluation with basic retry for stale file handle errors
        max_retries = 2
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    retry_delay = 2.0 * (attempt + 1) + random.uniform(0, 1)
                    time.sleep(retry_delay)
                    gc.collect()
                
                # Call the worker function
                result = _evaluate_parameters_worker(task_data)
                
                # Force cleanup
                gc.collect()
                return result
                
            except Exception as e:
                error_str = str(e).lower()
                error_trace = traceback.format_exc()
                
                # Check for stale file handle or similar filesystem errors
                if any(term in error_str for term in ['stale file handle', 'errno 116', 'input/output error', 'errno 5']):
                    if attempt < max_retries - 1:  # Not the last attempt
                        continue
                
                # For other errors or final attempt, return the error with full traceback
                return {
                    'individual_id': individual_id,
                    'params': task_data.get('params', {}),
                    'score': None,
                    'error': f'Worker exception (attempt {attempt + 1}): {str(e)}\nTraceback:\n{error_trace}',
                    'proc_id': proc_id,
                    'debug_info': {
                        'attempt': attempt + 1,
                        'max_retries': max_retries,
                        'process_id': process_id
                    }
                }
        
        # If we get here, all retries failed
        return {
            'individual_id': individual_id,
            'params': task_data.get('params', {}),
            'score': None,
            'error': f'Worker failed after {max_retries} attempts',
            'proc_id': proc_id
        }
        
    except Exception as e:
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': None,
            'error': f'Critical worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1)
        }
    
    finally:
        # Final cleanup
        gc.collect()

def _evaluate_parameters_worker(task_data: Dict) -> Dict:
    debug_info = {
        'stage': 'initialization',
        'files_checked': [],
        'commands_run': [],
        'errors': []
    }
    
    try:
        # Extract task info
        individual_id = task_data['individual_id']
        params = task_data['params']
        proc_id = task_data['proc_id']
        
        debug_info['individual_id'] = individual_id
        debug_info['proc_id'] = proc_id
        
        # Setup process logger with more detail
        logger = logging.getLogger(f'worker_{proc_id}_{individual_id}')
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)  # More verbose logging
            handler = logging.StreamHandler()
            formatter = logging.Formatter(f'[P{proc_id:02d}-I{individual_id:03d}] %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        logger.info(f"Starting evaluation of individual {individual_id}")
        
        # Convert all paths to absolute Path objects early
        debug_info['stage'] = 'path_setup'
        
        summa_exe = Path(task_data['summa_exe']).resolve()
        file_manager = Path(task_data['file_manager']).resolve()
        summa_dir = Path(task_data['summa_dir']).resolve()
        mizuroute_dir = Path(task_data['mizuroute_dir']).resolve()
        summa_settings_dir = Path(task_data['summa_settings_dir']).resolve()
        
        # Log paths for debugging
        logger.debug(f"SUMMA exe: {summa_exe}")
        logger.debug(f"File manager: {file_manager}")
        logger.debug(f"SUMMA dir: {summa_dir}")
        logger.debug(f"Settings dir: {summa_settings_dir}")
        
        debug_info['paths'] = {
            'summa_exe': str(summa_exe),
            'file_manager': str(file_manager),
            'summa_dir': str(summa_dir),
            'summa_settings_dir': str(summa_settings_dir)
        }
        
        # Verify critical paths exist
        debug_info['stage'] = 'path_verification'
        
        critical_paths = {
            'SUMMA executable': summa_exe,
            'File manager': file_manager,
            'SUMMA directory': summa_dir,
            'Settings directory': summa_settings_dir
        }
        
        for name, path in critical_paths.items():
            debug_info['files_checked'].append(f"{name}: {path}")
            if not path.exists():
                error_msg = f'{name} not found: {path}'
                logger.error(error_msg)
                debug_info['errors'].append(error_msg)
                return {
                    'individual_id': individual_id,
                    'params': params,
                    'score': None,
                    'error': error_msg,
                    'debug_info': debug_info
                }
        
        logger.info("All critical paths verified")
        
        # Apply parameters to files using CONSISTENT method
        debug_info['stage'] = 'parameter_application'
        logger.info("Applying parameters to model files (consistent method)")
        
        if not _apply_parameters_worker(params, task_data, summa_settings_dir, logger, debug_info):
            error_msg = 'Failed to apply parameters'
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return {
                'individual_id': individual_id,
                'params': params,
                'score': None,
                'error': error_msg,
                'debug_info': debug_info
            }
        
        logger.info("Parameters applied successfully")
        
        # Run SUMMA with enhanced debugging
        debug_info['stage'] = 'summa_execution'
        logger.info("Starting SUMMA execution")
        
        if not _run_summa_worker(summa_exe, file_manager, summa_dir, logger, debug_info):
            error_msg = 'SUMMA simulation failed'
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return {
                'individual_id': individual_id,
                'params': params,
                'score': None,
                'error': error_msg,
                'debug_info': debug_info
            }
        
        logger.info("SUMMA execution completed successfully")
        
        # Run mizuRoute if needed
        debug_info['stage'] = 'mizuroute_check'
        calibration_var = task_data.get('calibration_variable', 'streamflow')
        if calibration_var == 'streamflow':
            config = task_data['config']
            needs_routing = _needs_mizuroute_routing_worker(config)
            
            if needs_routing:
                debug_info['stage'] = 'mizuroute_execution'
                logger.info("Starting mizuRoute execution")
                
                if not _run_mizuroute_worker(task_data, mizuroute_dir, logger, debug_info):
                    error_msg = 'mizuRoute simulation failed'
                    logger.error(error_msg)
                    debug_info['errors'].append(error_msg)
                    return {
                        'individual_id': individual_id,
                        'params': params,
                        'score': None,
                        'error': error_msg,
                        'debug_info': debug_info
                    }
                
                logger.info("mizuRoute execution completed successfully")
        
        # Calculate metrics using CONSISTENT method
        debug_info['stage'] = 'metrics_calculation'
        logger.info("Calculating performance metrics (consistent with sequential)")
        
        score = _calculate_metrics_worker(task_data, summa_dir, mizuroute_dir, logger)
        
        if score is None:
            error_msg = 'Failed to calculate metrics (consistent method)'
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return {
                'individual_id': individual_id,
                'params': params,
                'score': None,
                'error': error_msg,
                'debug_info': debug_info
            }
        
        logger.info(f"Evaluation completed successfully. Score: {score:.6f} (consistent)")
        
        return {
            'individual_id': individual_id,
            'params': params,
            'score': score,
            'error': None,
            'debug_info': debug_info
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = f'Worker exception at stage {debug_info.get("stage", "unknown")}: {str(e)}'
        debug_info['errors'].append(f"{error_msg}\nTraceback:\n{error_trace}")
        
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': None,
            'error': error_msg,
            'debug_info': debug_info,
            'full_traceback': error_trace
        }

def _apply_parameters_worker(params: Dict, task_data: Dict, settings_dir: Path, logger, debug_info: Dict) -> bool:
    """Apply parameters consistently with sequential approach"""
    try:
        config = task_data['config']
        logger.debug(f"Applying parameters: {list(params.keys())} (consistent method)")
        
        # Parse parameter lists EXACTLY as ParameterManager does
        local_params = [p.strip() for p in config.get('PARAMS_TO_CALIBRATE', '').split(',') if p.strip()]
        basin_params = [p.strip() for p in config.get('BASIN_PARAMS_TO_CALIBRATE', '').split(',') if p.strip()]
        depth_params = ['total_mult', 'shape_factor'] if config.get('CALIBRATE_DEPTH', False) else []
        mizuroute_params = []
        
        if config.get('CALIBRATE_MIZUROUTE', False):
            mizuroute_params_str = config.get('MIZUROUTE_PARAMS_TO_CALIBRATE', 'velo,diff')
            mizuroute_params = [p.strip() for p in mizuroute_params_str.split(',') if p.strip()]
        
    
        # 1. Handle soil depth parameters
        if depth_params and 'total_mult' in params and 'shape_factor' in params:
            logger.debug("Updating soil depths (consistent)")
            if not _update_soil_depths_worker(params, task_data, settings_dir, logger, debug_info):
                return False
        
        # 2. Handle mizuRoute parameters
        if mizuroute_params and any(p in params for p in mizuroute_params):
            logger.debug("Updating mizuRoute parameters (consistent)")
            if not _update_mizuroute_params_worker(params, task_data, logger, debug_info):
                return False
        
        # 3. Generate trial parameters file (same exclusion logic as ParameterManager)
        hydraulic_params = {k: v for k, v in params.items() 
                          if k not in depth_params + mizuroute_params}
        
        if hydraulic_params:
            logger.debug(f"Generating trial parameters file with: {list(hydraulic_params.keys())} (consistent)")
            if not _generate_trial_params_worker(hydraulic_params, settings_dir, logger, debug_info):
                return False
        
        logger.debug("Parameter application completed successfully (consistent)")
        return True
        
    except Exception as e:
        error_msg = f"Error applying parameters (consistent): {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False


def _calculate_metrics_worker(task_data: Dict, summa_dir: Path, mizuroute_dir: Path, logger) -> Optional[float]:
    """Calculate metrics consistently with sequential evaluation using CalibrationTarget.calculate_metrics()"""
    try:
        # Recreate the CalibrationTarget to use same metrics calculation
        calibration_var = task_data.get('calibration_variable', 'streamflow')
        config = task_data['config']
        project_dir = Path(task_data['project_dir'])
        
        # Create the same calibration target as sequential
        if calibration_var == 'streamflow':
            target = StreamflowTarget(config, project_dir, logger)
        elif calibration_var == 'snow':
            target = SnowTarget(config, project_dir, logger)
        else:
            logger.error(f"Unsupported calibration variable: {calibration_var}")
            return None
        
        logger.debug("Created calibration target, calculating metrics...")
        
        # Use the SAME metrics calculation as sequential (calibration period only)
        metrics = target.calculate_metrics(summa_dir, mizuroute_dir, calibration_only=True)
        
        if not metrics:
            logger.error("Failed to calculate metrics using target.calculate_metrics()")
            return None
        
        logger.debug(f"Calculated metrics: {list(metrics.keys())}")
        
        # Use the SAME target metric extraction logic as sequential
        target_metric = task_data['target_metric']
        
        # Try exact match first
        if target_metric in metrics:
            score = metrics[target_metric]
            logger.debug(f"Found exact match for {target_metric}")
        # Try with calibration prefix
        elif f"Calib_{target_metric}" in metrics:
            score = metrics[f"Calib_{target_metric}"]
            logger.debug(f"Found calibration prefixed {target_metric}")
        # Try without prefix (remove Calib_ or Eval_)
        else:
            score = None
            for key, value in metrics.items():
                if key.endswith(f"_{target_metric}"):
                    score = value
                    logger.debug(f"Found suffixed match: {key}")
                    break
            
            # Fallback to first available metric (same as sequential)
            if score is None:
                score = next(iter(metrics.values())) if metrics else None
                if score is not None:
                    logger.warning(f"Using fallback metric: {list(metrics.keys())[0]} = {score}")
        
        if score is None or np.isnan(score):
            logger.error(f"Could not extract {target_metric} from metrics: {list(metrics.keys())}")
            return None
        
        # Apply same negation logic as sequential
        if target_metric.upper() in ['RMSE', 'MAE', 'PBIAS']:
            score = -score
            logger.debug(f"Applied negation for {target_metric}: {score}")
        
        logger.debug(f"Final {target_metric}: {score:.6f} (using consistent target.calculate_metrics)")
        return score
        
    except Exception as e:
        logger.error(f"Error in consistent metrics calculation: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def _update_soil_depths_worker(params: Dict, task_data: Dict, settings_dir: Path, logger, debug_info: Dict) -> bool:
    """Enhanced soil depth update with better error handling"""
    try:
        original_depths_list = task_data.get('original_depths')
        if not original_depths_list:
            logger.warning("No original depths provided, skipping soil depth update")
            return True
        
        original_depths = np.array(original_depths_list)
        
        total_mult = params['total_mult'][0] if isinstance(params['total_mult'], np.ndarray) else params['total_mult']
        shape_factor = params['shape_factor'][0] if isinstance(params['shape_factor'], np.ndarray) else params['shape_factor']
        
        logger.debug(f"Updating soil depths: total_mult={total_mult:.3f}, shape_factor={shape_factor:.3f}")
        
        # Calculate new depths
        arr = original_depths.copy()
        n = len(arr)
        idx = np.arange(n)
        
        if shape_factor > 1:
            w = np.exp(idx / (n - 1) * np.log(shape_factor))
        elif shape_factor < 1:
            w = np.exp((n - 1 - idx) / (n - 1) * np.log(1 / shape_factor))
        else:
            w = np.ones(n)
        
        w /= w.mean()
        new_depths = arr * w * total_mult
        
        # Calculate heights
        heights = np.zeros(len(new_depths) + 1)
        for i in range(len(new_depths)):
            heights[i + 1] = heights[i] + new_depths[i]
        
        # Update coldState.nc
        coldstate_path = settings_dir / 'coldState.nc'
        if not coldstate_path.exists():
            error_msg = f"coldState.nc not found: {coldstate_path}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False
        
        debug_info['files_checked'].append(f"coldState.nc: {coldstate_path}")
        
        with nc.Dataset(coldstate_path, 'r+') as ds:
            if 'mLayerDepth' not in ds.variables or 'iLayerHeight' not in ds.variables:
                error_msg = "Required depth variables not found in coldState.nc"
                logger.error(error_msg)
                debug_info['errors'].append(error_msg)
                return False
            
            num_hrus = ds.dimensions['hru'].size
            for h in range(num_hrus):
                ds.variables['mLayerDepth'][:, h] = new_depths
                ds.variables['iLayerHeight'][:, h] = heights
        
        logger.debug("Soil depths updated successfully")
        return True
        
    except Exception as e:
        error_msg = f"Error updating soil depths: {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False


def _update_mizuroute_params_worker(params: Dict, task_data: Dict, logger, debug_info: Dict) -> bool:
    """Enhanced mizuRoute parameter update with better error handling"""
    try:
        config = task_data['config']
        mizuroute_params = [p.strip() for p in config.get('MIZUROUTE_PARAMS_TO_CALIBRATE', '').split(',') if p.strip()]
        
        mizuroute_settings_dir = Path(task_data['mizuroute_settings_dir'])
        param_file = mizuroute_settings_dir / "param.nml.default"
        
        if not param_file.exists():
            logger.warning(f"mizuRoute param file not found: {param_file}")
            return True
        
        debug_info['files_checked'].append(f"mizuRoute param file: {param_file}")
        
        with open(param_file, 'r') as f:
            content = f.read()
        
        updated_content = content
        for param_name in mizuroute_params:
            if param_name in params:
                param_value = params[param_name]
                pattern = rf'(\s+{param_name}\s*=\s*)[0-9.-]+'
                
                if param_name in ['tscale']:
                    replacement = rf'\g<1>{int(param_value)}'
                else:
                    replacement = rf'\g<1>{param_value:.6f}'
                
                updated_content = re.sub(pattern, replacement, updated_content)
                logger.debug(f"Updated {param_name} = {param_value}")
        
        with open(param_file, 'w') as f:
            f.write(updated_content)
        
        logger.debug("mizuRoute parameters updated successfully")
        return True
        
    except Exception as e:
        error_msg = f"Error updating mizuRoute params: {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False


def _generate_trial_params_worker(params: Dict, settings_dir: Path, logger, debug_info: Dict) -> bool:
    """Enhanced trial parameters generation with better error handling and file locking"""
    import time
    import random
    import os
    
    try:
        if not params:
            logger.debug("No hydraulic parameters to write")
            return True
        
        trial_params_path = settings_dir / 'trialParams.nc'
        attr_file_path = settings_dir / 'attributes.nc'
        
        if not attr_file_path.exists():
            error_msg = f"Attributes file not found: {attr_file_path}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False
        
        debug_info['files_checked'].append(f"attributes.nc: {attr_file_path}")
        
        # Add retry logic with file locking
        max_retries = 5
        base_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                # Create temporary file first, then move it
                temp_path = trial_params_path.with_suffix(f'.tmp_{os.getpid()}_{random.randint(1000,9999)}')
                
                logger.debug(f"Attempt {attempt + 1}: Writing trial parameters to {temp_path}")
                
                # Define parameter levels
                routing_params = ['routingGammaShape', 'routingGammaScale']
                basin_params = ['basin__aquiferBaseflowExp', 'basin__aquiferScaleFactor', 'basin__aquiferHydCond']
                gru_level_params = routing_params + basin_params
                
                with xr.open_dataset(attr_file_path) as ds:
                    num_hrus = ds.sizes.get('hru', 1)
                    num_grus = ds.sizes.get('gru', 1)
                    hru_ids = ds['hruId'].values if 'hruId' in ds else np.arange(1, num_hrus + 1)
                    gru_ids = ds['gruId'].values if 'gruId' in ds else np.array([1])
                
                logger.debug(f"Writing parameters for {num_hrus} HRUs, {num_grus} GRUs")
                
                # Write to temporary file with exclusive access
                with nc.Dataset(temp_path, 'w', format='NETCDF4') as output_ds:
                    # Create dimensions
                    output_ds.createDimension('hru', num_hrus)
                    output_ds.createDimension('gru', num_grus)
                    
                    # Create coordinate variables
                    hru_var = output_ds.createVariable('hruId', 'i4', ('hru',), fill_value=-9999)
                    hru_var[:] = hru_ids
                    
                    gru_var = output_ds.createVariable('gruId', 'i4', ('gru',), fill_value=-9999)
                    gru_var[:] = gru_ids
                    
                    # Add parameters
                    for param_name, param_values in params.items():
                        param_values_array = np.asarray(param_values)
                        
                        if param_values_array.ndim > 1:
                            param_values_array = param_values_array.flatten()
                        
                        if param_name in gru_level_params:
                            # GRU-level parameters
                            param_var = output_ds.createVariable(param_name, 'f8', ('gru',), fill_value=np.nan)
                            param_var.long_name = f"Trial value for {param_name}"
                            
                            if len(param_values_array) >= num_grus:
                                param_var[:] = param_values_array[:num_grus]
                            else:
                                param_var[:] = param_values_array[0]
                        else:
                            # HRU-level parameters
                            param_var = output_ds.createVariable(param_name, 'f8', ('hru',), fill_value=np.nan)
                            param_var.long_name = f"Trial value for {param_name}"
                            
                            if len(param_values_array) == num_hrus:
                                param_var[:] = param_values_array
                            elif len(param_values_array) == 1:
                                param_var[:] = param_values_array[0]
                            else:
                                param_var[:] = param_values_array[:num_hrus]
                        
                        logger.debug(f"Added parameter {param_name} with shape {param_var.shape}")
                
                # Atomically move temporary file to final location
                try:
                    os.chmod(temp_path, 0o664)  # Set appropriate permissions
                    temp_path.rename(trial_params_path)
                    logger.debug(f"Trial parameters file created successfully: {trial_params_path}")
                    debug_info['files_checked'].append(f"trialParams.nc (created): {trial_params_path}")
                    return True
                except Exception as move_error:
                    if temp_path.exists():
                        temp_path.unlink()
                    raise move_error
                
            except (OSError, IOError, PermissionError) as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                
                # Clean up temp file if it exists
                if 'temp_path' in locals() and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except:
                        pass
                
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                    time.sleep(delay)
                else:
                    error_msg = f"Failed to generate trial params after {max_retries} attempts: {str(e)}"
                    logger.error(error_msg)
                    debug_info['errors'].append(error_msg)
                    return False
        
        return False
        
    except Exception as e:
        error_msg = f"Error generating trial params: {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False


def _run_summa_worker(summa_exe: Path, file_manager: Path, summa_dir: Path, logger, debug_info: Dict) -> bool:
    """Enhanced SUMMA execution with better error handling and debugging"""
    try:
        # Create log directory
        log_dir = summa_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"summa_worker_{os.getpid()}.log"
        
        # Set environment for single-threaded execution
        env = os.environ.copy()
        env.update({
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1'
        })
        
        # Convert paths to strings for subprocess
        summa_exe_str = str(summa_exe)
        file_manager_str = str(file_manager)
        
        # Build command
        cmd = f"{summa_exe_str} -m {file_manager_str}"
        
        logger.info(f"Executing SUMMA command: {cmd}")
        logger.debug(f"Working directory: {summa_dir}")
        logger.debug(f"Log file: {log_file}")
        
        debug_info['commands_run'].append(f"SUMMA: {cmd}")
        debug_info['summa_log'] = str(log_file)
        
        # Verify executable permissions
        if not os.access(summa_exe, os.X_OK):
            error_msg = f"SUMMA executable is not executable: {summa_exe}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False
        
        # Run SUMMA with explicit working directory
        with open(log_file, 'w') as f:
            f.write(f"SUMMA Execution Log\n")
            f.write(f"Command: {cmd}\n")
            f.write(f"Working Directory: {summa_dir}\n")
            f.write(f"Environment: OMP_NUM_THREADS={env.get('OMP_NUM_THREADS', 'unset')}\n")
            f.write("=" * 50 + "\n")
            f.flush()
            
            result = subprocess.run(
                cmd,
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1800,  # 30 minute timeout
                env=env,
                cwd=str(summa_dir)  # Explicit working directory
            )
        
        # Check if output files were created
        timestep_files = list(summa_dir.glob("*timestep.nc"))
        if not timestep_files:
            # Look for any .nc files
            nc_files = list(summa_dir.glob("*.nc"))
            if not nc_files:
                error_msg = f"No SUMMA output files found in {summa_dir}"
                logger.error(error_msg)
                debug_info['errors'].append(error_msg)
                
                # Read log file for more details
                if log_file.exists():
                    with open(log_file, 'r') as f:
                        log_content = f.read()
                        debug_info['summa_log_content'] = log_content[-2000:]  # Last 2000 chars
                
                return False
        
        logger.info(f"SUMMA execution completed successfully. Output files: {len(timestep_files)} timestep files")
        debug_info['summa_output_files'] = [str(f) for f in timestep_files[:3]]  # First 3 files
        
        return True
        
    except subprocess.CalledProcessError as e:
        error_msg = f"SUMMA simulation failed with exit code {e.returncode}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        
        # Read log file for error details
        if 'log_file' in locals() and log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    log_content = f.read()
                    debug_info['summa_log_content'] = log_content[-2000:]  # Last 2000 chars
            except:
                pass
                
        return False
        
    except subprocess.TimeoutExpired:
        error_msg = "SUMMA simulation timed out (30 minutes)"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False
        
    except Exception as e:
        error_msg = f"Error running SUMMA: {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False


def _run_mizuroute_worker(task_data: Dict, mizuroute_dir: Path, logger, debug_info: Dict) -> bool:
    try:
        config = task_data['config']
        
        # Get mizuRoute executable
        mizu_path = config.get('INSTALL_PATH_MIZUROUTE', 'default')
        if mizu_path == 'default':
            mizu_path = Path(config.get('CONFLUENCE_DATA_DIR')) / 'installs' / 'mizuRoute' / 'route' / 'bin'
        else:
            mizu_path = Path(mizu_path)
        
        mizu_exe = mizu_path / config.get('EXE_NAME_MIZUROUTE', 'mizuroute.exe')
        control_file = Path(task_data['mizuroute_settings_dir']) / 'mizuroute.control'
        
        # Verify files exist
        if not mizu_exe.exists():
            error_msg = f"mizuRoute executable not found: {mizu_exe}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False
            
        if not control_file.exists():
            error_msg = f"mizuRoute control file not found: {control_file}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False
        
        debug_info['files_checked'].extend([
            f"mizuRoute exe: {mizu_exe}",
            f"mizuRoute control: {control_file}"
        ])
        
        # Create log directory
        log_dir = mizuroute_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"mizuroute_worker_{os.getpid()}.log"
        
        # Build command
        cmd = f"{mizu_exe} {control_file}"
        
        logger.info(f"Executing mizuRoute command: {cmd}")
        debug_info['commands_run'].append(f"mizuRoute: {cmd}")
        debug_info['mizuroute_log'] = str(log_file)
        
        # Run mizuRoute
        with open(log_file, 'w') as f:
            f.write(f"mizuRoute Execution Log\n")
            f.write(f"Command: {cmd}\n")
            f.write(f"Working Directory: {control_file.parent}\n")
            f.write("=" * 50 + "\n")
            f.flush()
            
            result = subprocess.run(
                cmd,
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1800,  # 30 minute timeout
                cwd=str(control_file.parent)
            )
        
        # Check for output files
        nc_files = list(mizuroute_dir.glob("*.nc"))
        if not nc_files:
            error_msg = f"No mizuRoute output files found in {mizuroute_dir}"
            logger.error(error_msg)
            debug_info['errors'].append(error_msg)
            return False
        
        logger.info(f"mizuRoute execution completed successfully. Output files: {len(nc_files)}")
        debug_info['mizuroute_output_files'] = [str(f) for f in nc_files[:3]]
        
        return True
        
    except Exception as e:
        error_msg = f"mizuRoute execution failed: {str(e)}"
        logger.error(error_msg)
        debug_info['errors'].append(error_msg)
        return False

def _needs_mizuroute_routing_worker(config: Dict) -> bool:
    """Check if mizuRoute routing is needed"""
    domain_method = config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
    routing_delineation = config.get('ROUTING_DELINEATION', 'lumped')
    
    if domain_method not in ['point', 'lumped']:
        return True
    
    if domain_method == 'lumped' and routing_delineation == 'river_network':
        return True
    
    return False

def _run_dds_instance_worker(worker_data: Dict) -> Dict:
    """
    Worker function that runs a complete DDS instance
    This runs in a separate process
    """
    import numpy as np
    import logging
    import os
    from pathlib import Path
    
    try:
        dds_task = worker_data['dds_task']
        start_id = dds_task['start_id']
        max_iterations = dds_task['max_iterations']
        dds_r = dds_task['dds_r']
        starting_solution = dds_task['starting_solution']
        
        # Set up process-specific random seed
        np.random.seed(dds_task['random_seed'])
        
        # Set up logger
        logger = logging.getLogger(f'dds_worker_{start_id}')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter(f'[DDS-{start_id:02d}] %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        logger.info(f"Starting DDS instance {start_id} with {max_iterations} iterations")
        
        # Initialize DDS state
        current_solution = starting_solution.copy()
        param_count = len(current_solution)
        
        # Evaluate initial solution
        current_score = _evaluate_single_solution_worker(current_solution, worker_data, logger)
        
        best_solution = current_solution.copy()
        best_score = current_score
        best_params = _denormalize_params_worker(best_solution, worker_data)
        
        history = []
        total_evaluations = 1
        
        # Record initial state
        history.append({
            'generation': 0,
            'best_score': best_score,
            'current_score': current_score,
            'best_params': best_params
        })
        
        # DDS main loop
        for iteration in range(1, max_iterations + 1):
            # Calculate selection probability
            prob_select = 1.0 - np.log(iteration) / np.log(max_iterations)
            prob_select = max(prob_select, 1.0 / param_count)
            
            # Create trial solution
            trial_solution = current_solution.copy()
            
            # Select variables to perturb
            variables_to_perturb = np.random.random(param_count) < prob_select
            if not np.any(variables_to_perturb):
                random_idx = np.random.randint(0, param_count)
                variables_to_perturb[random_idx] = True
            
            # Apply perturbations
            for i in range(param_count):
                if variables_to_perturb[i]:
                    perturbation = np.random.normal(0, dds_r)
                    trial_solution[i] = current_solution[i] + perturbation
                    
                    # Reflect at bounds
                    if trial_solution[i] < 0:
                        trial_solution[i] = -trial_solution[i]
                    elif trial_solution[i] > 1:
                        trial_solution[i] = 2.0 - trial_solution[i]
                    
                    trial_solution[i] = np.clip(trial_solution[i], 0, 1)
            
            # Evaluate trial solution
            trial_score = _evaluate_single_solution_worker(trial_solution, worker_data, logger)
            total_evaluations += 1
            
            # Selection (greedy)
            improvement = False
            if trial_score > current_score:
                current_solution = trial_solution.copy()
                current_score = trial_score
                improvement = True
                
                if trial_score > best_score:
                    best_solution = trial_solution.copy()
                    best_score = trial_score
                    best_params = _denormalize_params_worker(best_solution, worker_data)
                    logger.info(f"Iter {iteration}: NEW BEST! Score={best_score:.6f}")
            
            # Record iteration
            history.append({
                'generation': iteration,
                'best_score': best_score,
                'current_score': current_score,
                'trial_score': trial_score,
                'improvement': improvement,
                'num_variables_perturbed': np.sum(variables_to_perturb),
                'best_params': best_params
            })
        
        logger.info(f"DDS instance {start_id} completed: Best={best_score:.6f}, Evaluations={total_evaluations}")
        
        return {
            'start_id': start_id,
            'best_score': best_score,
            'best_params': best_params,
            'best_solution': best_solution,
            'history': history,
            'total_evaluations': total_evaluations,
            'final_current_score': current_score
        }
        
    except Exception as e:
        import traceback
        return {
            'start_id': worker_data['dds_task']['start_id'],
            'best_score': None,
            'error': f'DDS worker exception: {str(e)}\n{traceback.format_exc()}'
        }

def _evaluate_single_solution_worker(solution: np.ndarray, worker_data: Dict, logger) -> float:
    """Evaluate a single solution in the worker process"""
    try:
        # Denormalize parameters
        params = _denormalize_params_worker(solution, worker_data)
        
        # Create evaluation task
        task_data = {
            'individual_id': 0,
            'params': params,
            'config': worker_data['config'],
            'target_metric': worker_data['target_metric'],
            'calibration_variable': worker_data['calibration_variable'],
            'domain_name': worker_data['domain_name'],
            'project_dir': worker_data['project_dir'],
            'original_depths': worker_data['original_depths'],
            'summa_exe': worker_data['summa_exe'],
            'file_manager': worker_data['file_manager'],
            'summa_dir': worker_data['summa_dir'],
            'mizuroute_dir': worker_data['mizuroute_dir'],
            'summa_settings_dir': worker_data['summa_settings_dir'],
            'mizuroute_settings_dir': worker_data['mizuroute_settings_dir'],
            'proc_id': 0
        }
        
        # Use existing worker function
        result = _evaluate_parameters_worker_safe(task_data)
        
        score = result.get('score')
        if score is None:
            logger.warning(f"Evaluation failed: {result.get('error', 'Unknown error')}")
            return float('-inf')
        
        return score
        
    except Exception as e:
        logger.error(f"Error evaluating solution: {str(e)}")
        return float('-inf')

def _denormalize_params_worker(normalized_solution: np.ndarray, worker_data: Dict) -> Dict:
    """Denormalize parameters in worker process"""
    param_bounds = worker_data['param_bounds']
    param_names = worker_data['all_param_names']
    
    params = {}
    for i, param_name in enumerate(param_names):
        if param_name in param_bounds:
            bounds = param_bounds[param_name]
            value = bounds['min'] + normalized_solution[i] * (bounds['max'] - bounds['min'])
            params[param_name] = np.array([value])
    
    return params


# ============= MAIN EXECUTION =============

if __name__ == "__main__":
    
    # Load configuration
    config_path = Path("config.yaml")  # Replace with actual config path
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Run optimization
    optimizer = DEOptimizer(config, logger)
    results = optimizer.run_optimization()
    
    print(f"Optimization completed. Best {results['optimization_metric']}: {results['best_score']:.6f}")
    print(f"Results saved to: {results['output_dir']}")