#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SYMFLUENCE Calibration Targets

This module provides calibration targets for different hydrologic variables including:
- Streamflow (routed and non-routed)
- Snow (SWE, SCA, depth)
- Groundwater (depth, GRACE TWS)
- Evapotranspiration (ET, latent heat)
- Soil moisture (point, SMAP, ESA)

Each target handles data loading, processing, and metric calculation for its specific variable.
"""

import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple, Optional
from abc import ABC, abstractmethod
import re



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
        
        # Parse calibration timestep
        self.calibration_timestep = config.get('CALIBRATION_TIMESTEP', 'native').lower()
        if self.calibration_timestep not in ['native', 'hourly', 'daily']:
            self.logger.warning(
                f"Invalid CALIBRATION_TIMESTEP '{self.calibration_timestep}'. "
                "Using 'native'. Valid options: 'native', 'hourly', 'daily'"
            )
            self.calibration_timestep = 'native'
        
        if self.calibration_timestep != 'native':
            self.logger.info(f"Calibration will use {self.calibration_timestep} timestep")
    
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
            
            # Resample to calibration timestep if specified in config
            if self.calibration_timestep != 'native':
                self.logger.info(f"Resampling data to {self.calibration_timestep} timestep")
                obs_period = self._resample_to_timestep(obs_period, self.calibration_timestep)
                sim_period = self._resample_to_timestep(sim_period, self.calibration_timestep)
                
                self.logger.debug(f"After resampling - obs points: {len(obs_period)}, sim points: {len(sim_period)}")
            
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
    
    def _resample_to_timestep(self, data: pd.Series, target_timestep: str) -> pd.Series:
        """
        Resample time series data to target timestep
        
        Args:
            data: Time series data with DatetimeIndex
            target_timestep: Target timestep ('hourly' or 'daily')
            
        Returns:
            Resampled time series
        """
        if target_timestep == 'native' or data is None or len(data) == 0:
            return data
        
        try:
            # Infer current frequency
            inferred_freq = pd.infer_freq(data.index)
            if inferred_freq is None:
                # Try to infer from first few differences
                if len(data) > 1:
                    time_diff = data.index[1] - data.index[0]
                    self.logger.debug(f"Inferred time difference: {time_diff}")
                else:
                    self.logger.warning("Cannot infer frequency from single data point")
                    return data
            else:
                self.logger.debug(f"Inferred frequency: {inferred_freq}")
            
            # Determine current timestep
            time_diff = data.index[1] - data.index[0] if len(data) > 1 else pd.Timedelta(hours=1)
            
            # Check if already at target timestep
            if target_timestep == 'hourly' and pd.Timedelta(minutes=45) <= time_diff <= pd.Timedelta(minutes=75):
                self.logger.debug("Data already at hourly timestep")
                return data
            elif target_timestep == 'daily' and pd.Timedelta(hours=20) <= time_diff <= pd.Timedelta(hours=28):
                self.logger.debug("Data already at daily timestep")
                return data
            
            # Perform resampling
            if target_timestep == 'hourly':
                if time_diff < pd.Timedelta(hours=1):
                    # Upsampling: sub-hourly to hourly (mean aggregation)
                    self.logger.info(f"Aggregating {time_diff} data to hourly using mean")
                    resampled = data.resample('H').mean()
                elif time_diff > pd.Timedelta(hours=1):
                    # Downsampling: daily/coarser to hourly (interpolation)
                    self.logger.info(f"Interpolating {time_diff} data to hourly")
                    # First resample to hourly (creates NaNs)
                    resampled = data.resample('H').asfreq()
                    # Then interpolate
                    resampled = resampled.interpolate(method='time', limit_direction='both')
                else:
                    resampled = data
                    
            elif target_timestep == 'daily':
                if time_diff < pd.Timedelta(days=1):
                    # Upsampling: hourly/sub-daily to daily (mean aggregation)
                    self.logger.info(f"Aggregating {time_diff} data to daily using mean")
                    resampled = data.resample('D').mean()
                elif time_diff > pd.Timedelta(days=1):
                    # Downsampling: weekly/monthly to daily (interpolation)
                    self.logger.info(f"Interpolating {time_diff} data to daily")
                    resampled = data.resample('D').asfreq()
                    resampled = resampled.interpolate(method='time', limit_direction='both')
                else:
                    resampled = data
            else:
                resampled = data
            
            # Remove any NaN values introduced by resampling at edges
            resampled = resampled.dropna()
            
            self.logger.info(
                f"Resampled from {len(data)} to {len(resampled)} points "
                f"(target: {target_timestep})"
            )
            
            return resampled
            
        except Exception as e:
            self.logger.error(f"Error resampling to {target_timestep}: {str(e)}")
            self.logger.warning("Returning original data without resampling")
            return data

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


class ETTarget(CalibrationTarget):
    """Evapotranspiration calibration target for FluxNet data"""
    
    def __init__(self, config: Dict, project_dir: Path, logger: logging.Logger):
        super().__init__(config, project_dir, logger)
        
        # Determine ET variable type from config
        self.optimization_target = config.get('OPTIMISATION_TARGET', 'streamflow')
        if self.optimization_target not in ['et', 'latent_heat']:
            raise ValueError(f"Invalid ET optimization target: {self.optimization_target}")
        
        self.variable_name = self.optimization_target
        
        # Temporal aggregation method for high-frequency FluxNet data
        self.temporal_aggregation = config.get('ET_TEMPORAL_AGGREGATION', 'daily_mean')  # daily_mean, daily_sum
        
        # Quality control settings
        self.use_quality_control = config.get('ET_USE_QUALITY_CONTROL', True)
        self.max_quality_flag = config.get('ET_MAX_QUALITY_FLAG', 2)  # FluxNet QC flags: 0=best, 3=worst
        
        self.logger.info(f"Initialized ETTarget for {self.optimization_target.upper()} calibration")
        self.logger.info(f"Temporal aggregation: {self.temporal_aggregation}")
        if self.use_quality_control:
            self.logger.info(f"Quality control enabled: max flag = {self.max_quality_flag}")
    
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Get SUMMA daily output files containing ET variables"""
        # Look for daily output files (they contain ET variables)
        daily_files = list(sim_dir.glob("*_day.nc"))
        if daily_files:
            return daily_files
        
        # Fallback to timestep files if daily not available
        return list(sim_dir.glob("*timestep.nc"))
    
    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract ET data from SUMMA simulation files"""
        sim_file = sim_files[0]  # Use first file
        
        try:
            with xr.open_dataset(sim_file) as ds:
                if self.optimization_target == 'et':
                    return self._extract_et_data(ds)
                elif self.optimization_target == 'latent_heat':
                    return self._extract_latent_heat_data(ds)
                else:
                    raise ValueError(f"Unknown ET target: {self.optimization_target}")
                    
        except Exception as e:
            self.logger.error(f"Error extracting ET data from {sim_file}: {str(e)}")
            raise
    
    def _extract_et_data(self, ds: xr.Dataset) -> pd.Series:
        """Extract total ET from SUMMA output"""
        # Primary variable: scalarTotalET
        if 'scalarTotalET' in ds.variables:
            et_var = ds['scalarTotalET']
            self.logger.debug(f"Found scalarTotalET variable with shape: {et_var.shape}")
            
            # Extract data - handle spatial dimensions
            if len(et_var.shape) > 1:
                if 'hru' in et_var.dims:
                    if et_var.shape[et_var.dims.index('hru')] == 1:
                        sim_data = et_var.isel(hru=0).to_pandas()
                    else:
                        # Average across HRUs for distributed domain
                        sim_data = et_var.mean(dim='hru').to_pandas()
                        self.logger.info(f"Averaged ET across {et_var.shape[et_var.dims.index('hru')]} HRUs")
                else:
                    non_time_dims = [dim for dim in et_var.dims if dim != 'time']
                    if non_time_dims:
                        sim_data = et_var.isel({non_time_dims[0]: 0}).to_pandas()
                    else:
                        sim_data = et_var.to_pandas()
            else:
                sim_data = et_var.to_pandas()
            
            # Convert units: SUMMA outputs kg m-2 s-1, convert to mm/day
            sim_data = self._convert_et_units(sim_data, from_unit='kg_m2_s', to_unit='mm_day')
            
            self.logger.debug(f"Extracted ET data: {len(sim_data)} timesteps, range: {sim_data.min():.3f} to {sim_data.max():.3f} mm/day")
            return sim_data
            
        else:
            # Fallback: sum ET components
            self.logger.warning("scalarTotalET not found, summing ET components")
            return self._sum_et_components(ds)
    
    def _sum_et_components(self, ds: xr.Dataset) -> pd.Series:
        """Sum individual ET components to get total ET"""
        try:
            et_components = {}
            
            # ET component variables in SUMMA
            component_vars = {
                'canopy_transpiration': 'scalarCanopyTranspiration',
                'canopy_evaporation': 'scalarCanopyEvaporation', 
                'ground_evaporation': 'scalarGroundEvaporation',
                'snow_sublimation': 'scalarSnowSublimation',
                'canopy_sublimation': 'scalarCanopySublimation'
            }
            
            total_et = None
            
            for component_name, var_name in component_vars.items():
                if var_name in ds.variables:
                    component_var = ds[var_name]
                    
                    # Handle spatial dimensions
                    if len(component_var.shape) > 1:
                        if 'hru' in component_var.dims:
                            if component_var.shape[component_var.dims.index('hru')] == 1:
                                component_data = component_var.isel(hru=0)
                            else:
                                component_data = component_var.mean(dim='hru')
                        else:
                            non_time_dims = [dim for dim in component_var.dims if dim != 'time']
                            if non_time_dims:
                                component_data = component_var.isel({non_time_dims[0]: 0})
                            else:
                                component_data = component_var
                    else:
                        component_data = component_var
                    
                    if total_et is None:
                        total_et = component_data
                    else:
                        total_et = total_et + component_data
                    
                    self.logger.debug(f"Added {component_name} to total ET")
            
            if total_et is None:
                raise ValueError("No ET component variables found in SUMMA output")
            
            # Convert to pandas and apply unit conversion
            sim_data = total_et.to_pandas()
            sim_data = self._convert_et_units(sim_data, from_unit='kg_m2_s', to_unit='mm_day')
            
            self.logger.info("Calculated total ET from component sum")
            return sim_data
            
        except Exception as e:
            self.logger.error(f"Error summing ET components: {str(e)}")
            raise
    
    def _extract_latent_heat_data(self, ds: xr.Dataset) -> pd.Series:
        """Extract latent heat flux from SUMMA output"""
        # Primary variable: scalarLatHeatTotal
        if 'scalarLatHeatTotal' in ds.variables:
            lh_var = ds['scalarLatHeatTotal']
            self.logger.debug(f"Found scalarLatHeatTotal variable with shape: {lh_var.shape}")
            
            # Extract data - handle spatial dimensions
            if len(lh_var.shape) > 1:
                if 'hru' in lh_var.dims:
                    if lh_var.shape[lh_var.dims.index('hru')] == 1:
                        sim_data = lh_var.isel(hru=0).to_pandas()
                    else:
                        sim_data = lh_var.mean(dim='hru').to_pandas()
                        self.logger.info(f"Averaged latent heat across {lh_var.shape[lh_var.dims.index('hru')]} HRUs")
                else:
                    non_time_dims = [dim for dim in lh_var.dims if dim != 'time']
                    if non_time_dims:
                        sim_data = lh_var.isel({non_time_dims[0]: 0}).to_pandas()
                    else:
                        sim_data = lh_var.to_pandas()
            else:
                sim_data = lh_var.to_pandas()
            
            # SUMMA outputs W m-2, which matches FluxNet LE units
            self.logger.debug(f"Extracted latent heat data: {len(sim_data)} timesteps, range: {sim_data.min():.2f} to {sim_data.max():.2f} W/m²")
            return sim_data
            
        else:
            raise ValueError("scalarLatHeatTotal not found in SUMMA output")
    
    def _convert_et_units(self, et_data: pd.Series, from_unit: str, to_unit: str) -> pd.Series:
        """Convert ET units"""
        if from_unit == 'kg_m2_s' and to_unit == 'mm_day':
            # Convert kg m-2 s-1 to mm/day
            # 1 kg m-2 s-1 = 86400 mm/day (since 1 kg/m² = 1 mm water)
            converted = et_data * 86400.0
            self.logger.debug("Converted ET from kg m-2 s-1 to mm/day (factor: 86400)")
            return converted
        elif from_unit == to_unit:
            return et_data
        else:
            self.logger.warning(f"Unknown unit conversion: {from_unit} to {to_unit}")
            return et_data
    
    def get_observed_data_path(self) -> Path:
        """Get path to observed FluxNet data"""
        return self.project_dir / "observations" / "energy_fluxes" / "processed" / f"{self.domain_name}_fluxnet_processed.csv"
    
    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        """Identify the ET data column in FluxNet file"""
        if self.optimization_target == 'et':
            # Look for ET column
            for col in columns:
                if any(term in col.lower() for term in ['et_from_le', 'et', 'evapotranspiration']):
                    return col
            # Fallback to exact match
            if 'ET_from_LE_mm_per_day' in columns:
                return 'ET_from_LE_mm_per_day'
                
        elif self.optimization_target == 'latent_heat':
            # Look for latent heat flux column
            for col in columns:
                if any(term in col.lower() for term in ['le_f_mds', 'le_', 'latent']):
                    return col
            # Fallback to exact match
            if 'LE_F_MDS' in columns:
                return 'LE_F_MDS'
        
        return None
    
    def _load_observed_data(self) -> Optional[pd.Series]:
        """Load observed FluxNet data with quality control and temporal aggregation"""
        try:
            obs_path = self.get_observed_data_path()
            if not obs_path.exists():
                self.logger.error(f"Observed FluxNet data file not found: {obs_path}")
                return None
            
            obs_df = pd.read_csv(obs_path)
            self.logger.debug(f"Loaded FluxNet data with columns: {list(obs_df.columns)}")
            
            # Find date and data columns
            date_col = self._find_date_column(obs_df.columns)
            data_col = self._get_observed_data_column(obs_df.columns)
            
            if not date_col or not data_col:
                self.logger.error(f"Could not identify date/data columns in {obs_path}")
                self.logger.error(f"Available columns: {list(obs_df.columns)}")
                return None
            
            self.logger.debug(f"Using date column: '{date_col}', data column: '{data_col}'")
            
            # Process datetime
            obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], errors='coerce')
            obs_df = obs_df.dropna(subset=['DateTime'])
            obs_df.set_index('DateTime', inplace=True)
            
            # Extract data column
            obs_data = pd.to_numeric(obs_df[data_col], errors='coerce')
            
            # Apply quality control if enabled
            if self.use_quality_control:
                obs_data = self._apply_quality_control(obs_df, obs_data, data_col)
            
            # Remove remaining NaN values
            obs_data = obs_data.dropna()
            
            # Apply temporal aggregation (FluxNet is typically 30-min, we want daily)
            if self.temporal_aggregation == 'daily_mean':
                obs_daily = obs_data.resample('D').mean()
                self.logger.info(f"Aggregated to daily means: {len(obs_data)} → {len(obs_daily)} points")
            elif self.temporal_aggregation == 'daily_sum':
                obs_daily = obs_data.resample('D').sum() 
                self.logger.info(f"Aggregated to daily sums: {len(obs_data)} → {len(obs_daily)} points")
            else:
                obs_daily = obs_data  # No aggregation
                self.logger.info("No temporal aggregation applied")
            
            # Remove any remaining NaN from aggregation
            obs_daily = obs_daily.dropna()
            
            #self.logger.info(f"Final dataset: {len(obs_daily)} valid daily observations")
            #self.logger.debug(f"Observed data range: {obs_daily.min():.3f} to {obs_daily.max():.3f}")
            
            return obs_daily
            
        except Exception as e:
            self.logger.error(f"Error loading observed FluxNet data: {str(e)}")
            return None
    
    def _apply_quality_control(self, obs_df: pd.DataFrame, obs_data: pd.Series, data_col: str) -> pd.Series:
        """Apply FluxNet quality control filters"""
        try:
            # Find quality control column
            qc_col = None
            if self.optimization_target == 'et':
                # For ET derived from LE, use LE quality flags
                if 'LE_F_MDS_QC' in obs_df.columns:
                    qc_col = 'LE_F_MDS_QC'
            elif self.optimization_target == 'latent_heat':
                if 'LE_F_MDS_QC' in obs_df.columns:
                    qc_col = 'LE_F_MDS_QC'
            
            if qc_col and qc_col in obs_df.columns:
                qc_flags = pd.to_numeric(obs_df[qc_col], errors='coerce')
                
                # Apply quality filter
                quality_mask = qc_flags <= self.max_quality_flag
                valid_before = len(obs_data.dropna())
                obs_data = obs_data[quality_mask]
                valid_after = len(obs_data.dropna())
                
                self.logger.info(f"Quality control applied: {valid_before} → {valid_after} valid points")
                self.logger.debug(f"Rejected {valid_before - valid_after} points with QC > {self.max_quality_flag}")
            else:
                self.logger.warning(f"Quality control column '{qc_col}' not found, skipping QC")
            
            return obs_data
            
        except Exception as e:
            self.logger.warning(f"Error applying quality control: {str(e)}")
            return obs_data
    
    def _find_date_column(self, columns: List[str]) -> Optional[str]:
        """Find timestamp column in FluxNet data"""
        timestamp_candidates = ['timestamp', 'TIMESTAMP_START', 'TIMESTAMP_END', 'datetime', 'time']
        
        for candidate in timestamp_candidates:
            if candidate in columns:
                return candidate
        
        # Look for columns containing timestamp-like terms
        for col in columns:
            if any(term in col.lower() for term in ['timestamp', 'time', 'date']):
                return col
        
        return None
    
    def needs_routing(self) -> bool:
        """ET calibration doesn't need routing"""
        return False
    
    def _calculate_performance_metrics(self, observed: pd.Series, simulated: pd.Series) -> Dict[str, float]:
        """Calculate ET-specific performance metrics"""
        try:
            # Clean data
            observed = pd.to_numeric(observed, errors='coerce')
            simulated = pd.to_numeric(simulated, errors='coerce')
            
            valid = ~(observed.isna() | simulated.isna())
            observed = observed[valid]
            simulated = simulated[valid]
            
            if len(observed) == 0:
                return {'KGE': np.nan, 'NSE': np.nan, 'RMSE': np.nan, 'PBIAS': np.nan, 'MAE': np.nan}
            
            # Standard metrics
            base_metrics = super()._calculate_performance_metrics(observed, simulated)
            
            # Add ET-specific metrics
            et_metrics = self._calculate_et_specific_metrics(observed, simulated)
            base_metrics.update(et_metrics)
            
            return base_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating ET performance metrics: {str(e)}")
            return {'KGE': np.nan, 'NSE': np.nan, 'RMSE': np.nan, 'PBIAS': np.nan, 'MAE': np.nan}
    
    def _calculate_et_specific_metrics(self, observed: pd.Series, simulated: pd.Series) -> Dict[str, float]:
        """Calculate ET-specific metrics"""
        try:
            metrics = {}
            
            # Seasonal cycle analysis
            if len(observed) > 365:  # Need at least a year
                obs_seasonal = self._extract_seasonal_cycle(observed)
                sim_seasonal = self._extract_seasonal_cycle(simulated)
                
                # Seasonal amplitude comparison
                obs_amplitude = obs_seasonal.max() - obs_seasonal.min()
                sim_amplitude = sim_seasonal.max() - sim_seasonal.min()
                
                if obs_amplitude > 0:
                    amplitude_ratio = sim_amplitude / obs_amplitude
                    metrics['Seasonal_Amplitude_Ratio'] = amplitude_ratio
                
                # Growing season ET (May-September in Northern Hemisphere)
                growing_months = [5, 6, 7, 8, 9]
                obs_growing = obs_seasonal[obs_seasonal.index.isin(growing_months)]
                sim_growing = sim_seasonal[sim_seasonal.index.isin(growing_months)]
                
                if len(obs_growing) > 0 and len(sim_growing) > 0:
                    growing_bias = (sim_growing.mean() - obs_growing.mean()) / obs_growing.mean() * 100
                    metrics['Growing_Season_Bias_Pct'] = growing_bias
            
            # Water use efficiency proxy (if precipitation data available)
            # This would require additional data, skipping for now
            
            # High/low flow periods
            obs_high_threshold = observed.quantile(0.9)
            obs_low_threshold = observed.quantile(0.1)
            
            high_mask = observed >= obs_high_threshold
            low_mask = observed <= obs_low_threshold
            
            if high_mask.any():
                high_bias = (simulated[high_mask].mean() - observed[high_mask].mean()) / observed[high_mask].mean() * 100
                metrics['High_ET_Bias_Pct'] = high_bias
            
            if low_mask.any():
                low_bias = (simulated[low_mask].mean() - observed[low_mask].mean()) / observed[low_mask].mean() * 100
                metrics['Low_ET_Bias_Pct'] = low_bias
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating ET-specific metrics: {str(e)}")
            return {}
    
    def _extract_seasonal_cycle(self, data: pd.Series) -> pd.Series:
        """Extract average seasonal cycle from ET data"""
        try:
            # Group by month and calculate mean
            monthly_means = data.groupby(data.index.month).mean()
            return monthly_means
            
        except Exception:
            return pd.Series()


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
        """Extract streamflow from mizuRoute output using worker script approach"""
        with xr.open_dataset(sim_file) as ds:
            # Debug logging
            self.logger.debug(f"DEBUG: mizuRoute file variables: {list(ds.variables.keys())}")
            
            # Find streamflow variable (same as worker script)
            streamflow_vars = ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff']
            
            for var_name in streamflow_vars:
                if var_name in ds.variables:
                    var = ds[var_name]
                    self.logger.debug(f"DEBUG: Using variable {var_name}, shape: {var.shape}")
                    
                    # Use the SAME approach as worker script - find segment with highest average runoff
                    if 'seg' in var.dims:
                        # Calculate mean runoff across time for each segment to find outlet
                        segment_means = var.mean(dim='time').values
                        outlet_seg_idx = np.argmax(segment_means)
                        
                        self.logger.debug(f"DEBUG: Found outlet at segment index {outlet_seg_idx} with mean runoff {segment_means[outlet_seg_idx]:.3f} m³/s")
                        
                        # Extract time series for outlet segment
                        result = var.isel(seg=outlet_seg_idx).to_pandas()
                        
                    elif 'reachID' in var.dims:
                        # Calculate mean runoff across time for each reach to find outlet
                        reach_means = var.mean(dim='time').values
                        outlet_reach_idx = np.argmax(reach_means)
                        
                        self.logger.debug(f"DEBUG: Found outlet at reach index {outlet_reach_idx} with mean runoff {reach_means[outlet_reach_idx]:.3f} m³/s")
                        
                        # Extract time series for outlet reach
                        result = var.isel(reachID=outlet_reach_idx).to_pandas()
                        
                    else:
                        self.logger.error(f"DEBUG: Unexpected dimensions for {var_name}: {var.dims}")
                        continue
                    
                    # Debug the extracted data
                    self.logger.debug(f"DEBUG: Extracted {len(result)} timesteps")
                    
                    return result
            
            raise ValueError("No suitable streamflow variable found in mizuRoute output")

    def _find_outlet_reach_id(self, mizuroute_ds: xr.Dataset) -> int:
        """Find the reach ID closest to the pour point with comprehensive fallbacks"""
        
        # First get available reaches for fallbacks
        try:
            available_reaches = mizuroute_ds['reachID'].values
            self.logger.info(f"DEBUG: Found {len(available_reaches)} available reaches")
        except Exception as e:
            self.logger.error(f"Cannot access reachID from mizuRoute dataset: {str(e)}")
            raise ValueError("mizuRoute dataset missing reachID variable")
        
        # APPROACH 1: Geographic approach (if pour point available)
        try:
            import geopandas as gpd
            from shapely.geometry import Point
            
            # Load pour point shapefile
            pour_point_path = self.config.get('POUR_POINT_SHP_PATH', 'default')
            pour_point_name = self.config.get('POUR_POINT_SHP_NAME', 'default')
            
            if pour_point_path == 'default':
                pour_point_path = self.project_dir / "shapefiles" / "pour_point"
            else:
                pour_point_path = Path(pour_point_path)
            
            if pour_point_name == 'default':
                pour_point_name = f"{self.domain_name}_pourPoint.shp"
            
            pour_point_file = pour_point_path / pour_point_name
            
            if pour_point_file.exists():
                self.logger.info("DEBUG: Attempting geographic approach...")
                # For now, falling through to discharge approach
                raise Exception("Geographic approach not available, using discharge approach")
            else:
                raise Exception("Pour point file not found")
                
        except Exception as e:
            self.logger.info(f"DEBUG: Geographic approach failed: {str(e)}")
        
        # APPROACH 2: Find outlet reach by topology (downSegId == -1)
        try:
            self.logger.info("DEBUG: Attempting topology-based outlet identification...")
            if 'downSegId' in mizuroute_ds.variables:
                down_seg_ids = mizuroute_ds['downSegId'].values
                outlet_mask = down_seg_ids == -1
                if np.any(outlet_mask):
                    outlet_reach_id = available_reaches[outlet_mask][0]
                    self.logger.info(f"DEBUG: Found outlet reach by topology: {outlet_reach_id}")
                    return int(outlet_reach_id)
            else:
                self.logger.info("DEBUG: No downSegId variable in dataset")
        except Exception as e:
            self.logger.warning(f"DEBUG: Topology approach failed: {str(e)}")
        
        # APPROACH 3: Find reach with highest discharge
        try:
            self.logger.info("DEBUG: Attempting discharge-based outlet identification...")
            streamflow_vars = ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff']
            
            for var_name in streamflow_vars:
                if var_name in mizuroute_ds.variables:
                    self.logger.info(f"DEBUG: Using {var_name} to find highest discharge reach")
                    flow_data = mizuroute_ds[var_name]
                    
                    # Calculate mean discharge for each reach over the simulation period
                    if 'seg' in flow_data.dims:
                        # Calculate mean discharge per segment
                        mean_discharge = flow_data.mean(dim='time')
                        max_discharge_idx = int(mean_discharge.argmax().values)
                        highest_discharge_reach = available_reaches[max_discharge_idx]
                        max_discharge_value = float(mean_discharge.max().values)
                        
                        self.logger.info(f"DEBUG: Reach {highest_discharge_reach} has highest mean discharge: {max_discharge_value:.6f}")
                        return int(highest_discharge_reach)
                        
                    elif 'reachID' in flow_data.dims:
                        # Direct indexing by reachID
                        mean_discharge = flow_data.mean(dim='time')
                        max_discharge_reach_id = int(mean_discharge.idxmax(dim='reachID').values)
                        max_discharge_value = float(mean_discharge.max().values)
                        
                        self.logger.info(f"DEBUG: Reach {max_discharge_reach_id} has highest mean discharge: {max_discharge_value:.6f}")
                        return int(max_discharge_reach_id)
                        
                    else:
                        self.logger.warning(f"DEBUG: {var_name} has unexpected dimensions: {flow_data.dims}")
                        
            self.logger.warning("DEBUG: No suitable streamflow variables found for discharge analysis")
            
        except Exception as e:
            self.logger.warning(f"DEBUG: Discharge approach failed: {str(e)}")
        
        # APPROACH 4: Use last reach (positional fallback)
        try:
            fallback_reach = int(available_reaches[-1])
            self.logger.info(f"DEBUG: Using last reach as positional fallback: {fallback_reach}")
            return fallback_reach
        except Exception as e:
            self.logger.warning(f"DEBUG: Positional fallback failed: {str(e)}")
        
        # APPROACH 5: Ultimate fallback - first reach
        try:
            ultimate_fallback = int(available_reaches[0])
            self.logger.info(f"DEBUG: Using first reach as ultimate fallback: {ultimate_fallback}")
            return ultimate_fallback
        except Exception as e:
            self.logger.error(f"DEBUG: Even ultimate fallback failed: {str(e)}")
            raise ValueError("Cannot determine any valid reach ID from mizuRoute dataset")
    
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


class SoilMoistureTarget(CalibrationTarget):
    """Soil moisture calibration target supporting point, SMAP, and ESA data"""
    
    def __init__(self, config: Dict, project_dir: Path, logger: logging.Logger):
        super().__init__(config, project_dir, logger)
        
        # Determine soil moisture variable type from config
        self.optimization_target = config.get('OPTIMISATION_TARGET', 'streamflow')
        if self.optimization_target not in ['sm_point', 'sm_smap', 'sm_esa']:
            raise ValueError(f"Invalid soil moisture optimization target: {self.optimization_target}")
        
        self.variable_name = self.optimization_target
        
        # Configuration for different observation types
        if self.optimization_target == 'sm_point':
            # For point data, specify which depth to use
            self.target_depth = config.get('SM_TARGET_DEPTH', 'auto')  # 'auto' or specific depth like '0.1016'
            self.depth_tolerance = config.get('SM_DEPTH_TOLERANCE', 0.05)  # meters
        elif self.optimization_target == 'sm_smap':
            # For SMAP, specify surface or rootzone
            self.smap_layer = config.get('SMAP_LAYER', 'surface_sm')  # 'surface_sm' or 'rootzone_sm'
            self.temporal_aggregation = config.get('SM_TEMPORAL_AGGREGATION', 'daily_mean')
        elif self.optimization_target == 'sm_esa':
            # ESA is typically surface soil moisture
            self.temporal_aggregation = config.get('SM_TEMPORAL_AGGREGATION', 'daily_mean')
        
        # Quality control
        self.use_quality_control = config.get('SM_USE_QUALITY_CONTROL', True)
        self.min_valid_pixels = config.get('SM_MIN_VALID_PIXELS', 10)  # For satellite data
        
        self.logger.info(f"Initialized SoilMoistureTarget for {self.optimization_target.upper()} calibration")
        if self.optimization_target == 'sm_point':
            self.logger.info(f"Target depth: {self.target_depth}")
        elif self.optimization_target == 'sm_smap':
            self.logger.info(f"SMAP layer: {self.smap_layer}")
    
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Get SUMMA daily output files containing soil moisture variables"""
        # Look for daily output files (they contain soil layer variables)
        daily_files = list(sim_dir.glob("*_day.nc"))
        if daily_files:
            return daily_files
        
        # Fallback to timestep files if daily not available
        return list(sim_dir.glob("*timestep.nc"))
    
    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract soil moisture data from SUMMA simulation files"""
        sim_file = sim_files[0]  # Use first file
        
        try:
            with xr.open_dataset(sim_file) as ds:
                if self.optimization_target == 'sm_point':
                    return self._extract_point_soil_moisture(ds)
                elif self.optimization_target == 'sm_smap':
                    return self._extract_smap_soil_moisture(ds)
                elif self.optimization_target == 'sm_esa':
                    return self._extract_esa_soil_moisture(ds)
                else:
                    raise ValueError(f"Unknown soil moisture target: {self.optimization_target}")
                    
        except Exception as e:
            self.logger.error(f"Error extracting soil moisture data from {sim_file}: {str(e)}")
            raise
    
    def _extract_point_soil_moisture(self, ds: xr.Dataset) -> pd.Series:
        """Extract soil moisture at specific depth for point comparison"""
        if 'mLayerVolFracLiq' not in ds.variables:
            raise ValueError("mLayerVolFracLiq variable not found in SUMMA output")
        
        soil_moisture_var = ds['mLayerVolFracLiq']
        
        # Get layer depths
        if 'mLayerDepth' in ds.variables:
            layer_depths = ds['mLayerDepth']
        else:
            raise ValueError("mLayerDepth variable not found - needed for depth matching")
        
        self.logger.debug(f"Found soil moisture variable with shape: {soil_moisture_var.shape}")
        self.logger.debug(f"Layer dimensions: {soil_moisture_var.dims}")
        
        # Handle spatial dimensions first
        if 'hru' in soil_moisture_var.dims:
            if soil_moisture_var.shape[soil_moisture_var.dims.index('hru')] == 1:
                soil_moisture_data = soil_moisture_var.isel(hru=0)
                layer_depths_data = layer_depths.isel(hru=0)
            else:
                # Average across HRUs
                soil_moisture_data = soil_moisture_var.mean(dim='hru')
                layer_depths_data = layer_depths.mean(dim='hru')
                self.logger.info(f"Averaged soil moisture across {soil_moisture_var.shape[soil_moisture_var.dims.index('hru')]} HRUs")
        else:
            soil_moisture_data = soil_moisture_var
            layer_depths_data = layer_depths
        
        # Find target depth layer
        target_layer_idx = self._find_target_layer(layer_depths_data)
        
        if target_layer_idx is None:
            raise ValueError(f"Could not find suitable layer for target depth {self.target_depth}")
        
        # Extract soil moisture for target layer
        layer_dim = [dim for dim in soil_moisture_data.dims if 'mid' in dim.lower() or 'layer' in dim.lower()][0]
        sim_data = soil_moisture_data.isel({layer_dim: target_layer_idx}).to_pandas()
        
        self.logger.info(f"Extracted soil moisture from layer {target_layer_idx} (depth ~{self.target_depth})")
        self.logger.debug(f"Soil moisture range: {sim_data.min():.3f} to {sim_data.max():.3f}")
        
        return sim_data
    
    def _find_target_layer(self, layer_depths: xr.DataArray) -> Optional[int]:
        """Find the layer index that best matches the target depth"""
        try:
            if self.target_depth == 'auto':
                # Use surface layer (typically index 0)
                return 0
            
            # Parse target depth from string (e.g., '0.1016' -> 0.1016 meters)
            try:
                target_depth_m = float(self.target_depth)
            except (ValueError, TypeError):
                self.logger.error(f"Could not parse target depth: {self.target_depth}")
                return 0  # Default to surface
            
            # Calculate cumulative depths (center of each layer)
            if len(layer_depths.shape) == 3:  # (time, layer, hru) or similar
                # Use first time step for layer structure
                depths_sample = layer_depths.isel(time=0).values
            elif len(layer_depths.shape) == 2:  # (time, layer)
                depths_sample = layer_depths.isel(time=0).values
            else:
                depths_sample = layer_depths.values
            
            # Calculate cumulative depths to center of each layer
            cumulative_depths = np.cumsum(depths_sample) - depths_sample / 2
            
            # Find closest layer
            depth_differences = np.abs(cumulative_depths - target_depth_m)
            best_layer_idx = np.argmin(depth_differences)
            
            best_depth = cumulative_depths[best_layer_idx]
            depth_error = abs(best_depth - target_depth_m)
            
            if depth_error > self.depth_tolerance:
                self.logger.warning(f"Best matching layer depth {best_depth:.3f}m differs from target {target_depth_m:.3f}m by {depth_error:.3f}m")
            else:
                self.logger.info(f"Found matching layer: index {best_layer_idx}, depth {best_depth:.3f}m (target: {target_depth_m:.3f}m)")
            
            return int(best_layer_idx)
            
        except Exception as e:
            self.logger.error(f"Error finding target layer: {str(e)}")
            return 0  # Default to surface layer
    
    def _extract_smap_soil_moisture(self, ds: xr.Dataset) -> pd.Series:
        """Extract soil moisture for SMAP comparison (surface or rootzone)"""
        if 'mLayerVolFracLiq' not in ds.variables:
            raise ValueError("mLayerVolFracLiq variable not found in SUMMA output")
        
        soil_moisture_var = ds['mLayerVolFracLiq']
        
        # Handle spatial dimensions
        if 'hru' in soil_moisture_var.dims:
            if soil_moisture_var.shape[soil_moisture_var.dims.index('hru')] == 1:
                soil_moisture_data = soil_moisture_var.isel(hru=0)
            else:
                soil_moisture_data = soil_moisture_var.mean(dim='hru')
                self.logger.info(f"Averaged soil moisture across {soil_moisture_var.shape[soil_moisture_var.dims.index('hru')]} HRUs")
        else:
            soil_moisture_data = soil_moisture_var
        
        if self.smap_layer == 'surface_sm':
            # Surface layer (top ~5cm, typically first layer)
            layer_dim = [dim for dim in soil_moisture_data.dims if 'mid' in dim.lower() or 'layer' in dim.lower()][0]
            sim_data = soil_moisture_data.isel({layer_dim: 0}).to_pandas()
            self.logger.info("Extracted surface soil moisture (layer 0) for SMAP comparison")
            
        elif self.smap_layer == 'rootzone_sm':
            # Rootzone average (top ~1m, multiple layers)
            # Get layer depths to determine rootzone
            if 'mLayerDepth' in ds.variables:
                layer_depths = ds['mLayerDepth']
                if 'hru' in layer_depths.dims:
                    if layer_depths.shape[layer_depths.dims.index('hru')] == 1:
                        depths_data = layer_depths.isel(hru=0)
                    else:
                        depths_data = layer_depths.mean(dim='hru')
                else:
                    depths_data = layer_depths
                
                # Find layers within top 1m
                rootzone_depth = 1.0  # meters
                rootzone_layers = self._find_rootzone_layers(depths_data, rootzone_depth)
                
                # Average soil moisture across rootzone layers
                layer_dim = [dim for dim in soil_moisture_data.dims if 'mid' in dim.lower() or 'layer' in dim.lower()][0]
                rootzone_sm = soil_moisture_data.isel({layer_dim: rootzone_layers}).mean(dim=layer_dim)
                sim_data = rootzone_sm.to_pandas()
                
                self.logger.info(f"Extracted rootzone soil moisture (layers {rootzone_layers[0]}-{rootzone_layers[-1]}) for SMAP comparison")
            else:
                # Fallback: average top 3 layers
                layer_dim = [dim for dim in soil_moisture_data.dims if 'mid' in dim.lower() or 'layer' in dim.lower()][0]
                top_layers = soil_moisture_data.isel({layer_dim: slice(0, 3)}).mean(dim=layer_dim)
                sim_data = top_layers.to_pandas()
                self.logger.warning("mLayerDepth not found, using top 3 layers for rootzone")
        else:
            raise ValueError(f"Unknown SMAP layer: {self.smap_layer}")
        
        self.logger.debug(f"SMAP soil moisture range: {sim_data.min():.3f} to {sim_data.max():.3f}")
        return sim_data
    
    def _find_rootzone_layers(self, layer_depths: xr.DataArray, rootzone_depth: float) -> List[int]:
        """Find layer indices within the rootzone depth"""
        try:
            # Use first time step for layer structure
            if len(layer_depths.shape) >= 2:
                depths_sample = layer_depths.isel(time=0).values
            else:
                depths_sample = layer_depths.values
            
            # Calculate cumulative depths
            cumulative_depths = np.cumsum(depths_sample)
            
            # Find layers within rootzone
            rootzone_mask = cumulative_depths <= rootzone_depth
            rootzone_indices = np.where(rootzone_mask)[0]
            
            if len(rootzone_indices) == 0:
                # Include at least the first layer
                rootzone_indices = [0]
            
            return rootzone_indices.tolist()
            
        except Exception as e:
            self.logger.error(f"Error finding rootzone layers: {str(e)}")
            return [0, 1, 2]  # Default to top 3 layers
    
    def _extract_esa_soil_moisture(self, ds: xr.Dataset) -> pd.Series:
        """Extract surface soil moisture for ESA comparison"""
        if 'mLayerVolFracLiq' not in ds.variables:
            raise ValueError("mLayerVolFracLiq variable not found in SUMMA output")
        
        soil_moisture_var = ds['mLayerVolFracLiq']
        
        # Handle spatial dimensions
        if 'hru' in soil_moisture_var.dims:
            if soil_moisture_var.shape[soil_moisture_var.dims.index('hru')] == 1:
                soil_moisture_data = soil_moisture_var.isel(hru=0)
            else:
                soil_moisture_data = soil_moisture_var.mean(dim='hru')
                self.logger.info(f"Averaged soil moisture across {soil_moisture_var.shape[soil_moisture_var.dims.index('hru')]} HRUs")
        else:
            soil_moisture_data = soil_moisture_var
        
        # ESA is surface soil moisture (top layer)
        layer_dim = [dim for dim in soil_moisture_data.dims if 'mid' in dim.lower() or 'layer' in dim.lower()][0]
        sim_data = soil_moisture_data.isel({layer_dim: 0}).to_pandas()
        
        self.logger.info("Extracted surface soil moisture (layer 0) for ESA comparison")
        self.logger.debug(f"ESA soil moisture range: {sim_data.min():.3f} to {sim_data.max():.3f}")
        
        return sim_data
    
    def get_observed_data_path(self) -> Path:
        """Get path to observed soil moisture data"""
        if self.optimization_target == 'sm_point':
            return self.project_dir / "observations" / "soil_moisture" / "point" / "processed" / f"{self.domain_name}_sm_processed.csv"
        elif self.optimization_target == 'sm_smap':
            return self.project_dir / "observations" / "soil_moisture" / "smap" / "processed" / f"{self.domain_name}_smap_processed.csv"
        elif self.optimization_target == 'sm_esa':
            return self.project_dir / "observations" / "soil_moisture" / "esa_sm" / "processed" / f"{self.domain_name}_esa_processed.csv"
        else:
            raise ValueError(f"Unknown soil moisture target: {self.optimization_target}")
    
    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        """Identify the soil moisture data column in observed data file"""
        if self.optimization_target == 'sm_point':
            # Find the best matching depth column
            if self.target_depth == 'auto':
                # Look for surface layer (smallest depth)
                depth_columns = [col for col in columns if col.startswith('sm_')]
                if depth_columns:
                    # Extract depths and find shallowest
                    depths = []
                    for col in depth_columns:
                        try:
                            # Parse depth from column name like 'sm_0.1016_0.1016'
                            depth_str = col.split('_')[1]
                            depth = float(depth_str)
                            depths.append((depth, col))
                        except (IndexError, ValueError):
                            continue
                    
                    if depths:
                        depths.sort()  # Sort by depth
                        shallowest_col = depths[0][1]
                        self.target_depth = str(depths[0][0])  # Update target depth
                        self.logger.info(f"Auto-selected shallowest depth column: {shallowest_col} (depth: {self.target_depth}m)")
                        return shallowest_col
            else:
                # Look for specific depth
                target_depth_str = str(self.target_depth)
                for col in columns:
                    if col.startswith('sm_') and target_depth_str in col:
                        return col
                
                # If exact match not found, find closest
                self.logger.warning(f"Exact depth {self.target_depth} not found, searching for closest match")
                return self._find_closest_depth_column(columns, float(self.target_depth))
                
        elif self.optimization_target == 'sm_smap':
            # Look for SMAP column
            if self.smap_layer in columns:
                return self.smap_layer
            # Fallback search
            for col in columns:
                if 'surface_sm' in col.lower() or 'rootzone_sm' in col.lower():
                    return col
                    
        elif self.optimization_target == 'sm_esa':
            # Look for ESA column
            for col in columns:
                if any(term in col.lower() for term in ['esa', 'soil_moisture', 'sm']):
                    return col
            if 'ESA' in columns:
                return 'ESA'
        
        return None
    
    def _find_closest_depth_column(self, columns: List[str], target_depth: float) -> Optional[str]:
        """Find the column with depth closest to target"""
        try:
            depth_columns = []
            for col in columns:
                if col.startswith('sm_'):
                    try:
                        depth_str = col.split('_')[1]
                        depth = float(depth_str)
                        depth_diff = abs(depth - target_depth)
                        depth_columns.append((depth_diff, col, depth))
                    except (IndexError, ValueError):
                        continue
            
            if depth_columns:
                depth_columns.sort()  # Sort by difference
                best_col = depth_columns[0][1]
                best_depth = depth_columns[0][2]
                depth_error = depth_columns[0][0]
                
                self.logger.info(f"Closest depth match: {best_col} (depth: {best_depth}m, error: {depth_error:.3f}m)")
                return best_col
            
        except Exception as e:
            self.logger.error(f"Error finding closest depth column: {str(e)}")
        
        return None
    
    def _load_observed_data(self) -> Optional[pd.Series]:
        """Load observed soil moisture data with proper processing for each type"""
        try:
            obs_path = self.get_observed_data_path()
            if not obs_path.exists():
                self.logger.error(f"Observed soil moisture data file not found: {obs_path}")
                return None
            
            obs_df = pd.read_csv(obs_path)
            self.logger.debug(f"Loaded soil moisture data with columns: {list(obs_df.columns)}")
            
            # Find date and data columns
            date_col = self._find_date_column(obs_df.columns)
            data_col = self._get_observed_data_column(obs_df.columns)
            
            if not date_col or not data_col:
                self.logger.error(f"Could not identify date/data columns in {obs_path}")
                self.logger.error(f"Available columns: {list(obs_df.columns)}")
                return None
            
            self.logger.debug(f"Using date column: '{date_col}', data column: '{data_col}'")
            
            # Process data based on target type
            if self.optimization_target == 'sm_point':
                obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], errors='coerce')
            elif self.optimization_target == 'sm_smap':
                obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], errors='coerce')
            elif self.optimization_target == 'sm_esa':
                obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], format='%d/%m/%Y', errors='coerce')
            
            # Remove rows with invalid dates
            obs_df = obs_df.dropna(subset=['DateTime'])
            obs_df.set_index('DateTime', inplace=True)
            
            # Extract data column and clean
            obs_series = pd.to_numeric(obs_df[data_col], errors='coerce')
            
            # Apply quality control for satellite data
            if self.optimization_target == 'sm_smap' and self.use_quality_control:
                obs_series = self._apply_smap_quality_control(obs_df, obs_series)
            
            # Remove NaN values
            obs_series = obs_series.dropna()
            
            # Apply temporal aggregation if needed
            if hasattr(self, 'temporal_aggregation') and self.temporal_aggregation == 'daily_mean':
                if len(obs_series) > len(obs_series.resample('D').mean().dropna()):
                    obs_series = obs_series.resample('D').mean().dropna()
                    self.logger.info("Applied daily mean aggregation to observations")
            
            self.logger.info(f"Loaded {len(obs_series)} valid soil moisture observations")
            self.logger.debug(f"Observed SM range: {obs_series.min():.3f} to {obs_series.max():.3f}")
            
            return obs_series
            
        except Exception as e:
            self.logger.error(f"Error loading observed soil moisture data: {str(e)}")
            return None
    
    def _apply_smap_quality_control(self, obs_df: pd.DataFrame, obs_series: pd.Series) -> pd.Series:
        """Apply quality control to SMAP data"""
        try:
            # Check for valid pixel count
            if 'valid_px' in obs_df.columns:
                valid_pixels = pd.to_numeric(obs_df['valid_px'], errors='coerce')
                quality_mask = valid_pixels >= self.min_valid_pixels
                
                valid_before = len(obs_series.dropna())
                obs_series = obs_series[quality_mask]
                valid_after = len(obs_series.dropna())
                
                self.logger.info(f"SMAP quality control: {valid_before} → {valid_after} valid points")
                self.logger.debug(f"Rejected {valid_before - valid_after} points with <{self.min_valid_pixels} valid pixels")
            
            return obs_series
            
        except Exception as e:
            self.logger.warning(f"Error applying SMAP quality control: {str(e)}")
            return obs_series
    
    def _find_date_column(self, columns: List[str]) -> Optional[str]:
        """Find date/time column in soil moisture data"""
        date_candidates = ['timestamp', 'date', 'time', 'DateTime', 'TIMESTAMP_START']
        
        for candidate in date_candidates:
            if candidate in columns:
                return candidate
        
        # Look for columns containing date-like terms
        for col in columns:
            if any(term in col.lower() for term in ['date', 'time', 'timestamp']):
                return col
        
        return None
    
    def needs_routing(self) -> bool:
        """Soil moisture calibration doesn't need routing"""
        return False
    
    def _calculate_performance_metrics(self, observed: pd.Series, simulated: pd.Series) -> Dict[str, float]:
        """Calculate soil moisture-specific performance metrics"""
        try:
            # Clean data
            observed = pd.to_numeric(observed, errors='coerce')
            simulated = pd.to_numeric(simulated, errors='coerce')
            
            valid = ~(observed.isna() | simulated.isna())
            observed = observed[valid]
            simulated = simulated[valid]
            
            if len(observed) == 0:
                return {'KGE': np.nan, 'NSE': np.nan, 'RMSE': np.nan, 'PBIAS': np.nan, 'MAE': np.nan}
            
            # Standard metrics
            base_metrics = super()._calculate_performance_metrics(observed, simulated)
            
            # Add soil moisture-specific metrics
            sm_metrics = self._calculate_soil_moisture_specific_metrics(observed, simulated)
            base_metrics.update(sm_metrics)
            
            return base_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating soil moisture performance metrics: {str(e)}")
            return {'KGE': np.nan, 'NSE': np.nan, 'RMSE': np.nan, 'PBIAS': np.nan, 'MAE': np.nan}
    
    def _calculate_soil_moisture_specific_metrics(self, observed: pd.Series, simulated: pd.Series) -> Dict[str, float]:
        """Calculate soil moisture-specific metrics"""
        try:
            metrics = {}
            
            # Dynamic range preservation
            obs_range = observed.max() - observed.min()
            sim_range = simulated.max() - simulated.min()
            
            if obs_range > 0:
                range_ratio = sim_range / obs_range
                metrics['Range_Ratio'] = range_ratio
            
            # Wet/dry period analysis
            obs_median = observed.median()
            wet_mask = observed >= obs_median
            dry_mask = observed < obs_median
            
            if wet_mask.any():
                wet_bias = (simulated[wet_mask].mean() - observed[wet_mask].mean()) / observed[wet_mask].mean() * 100
                metrics['Wet_Period_Bias_Pct'] = wet_bias
            
            if dry_mask.any():
                dry_bias = (simulated[dry_mask].mean() - observed[dry_mask].mean()) / observed[dry_mask].mean() * 100
                metrics['Dry_Period_Bias_Pct'] = dry_bias
            
            # Temporal persistence (lag-1 autocorrelation)
            if len(observed) > 10:
                obs_autocorr = observed.autocorr(lag=1)
                sim_autocorr = simulated.autocorr(lag=1)
                
                if not pd.isna(obs_autocorr) and not pd.isna(sim_autocorr):
                    metrics['Obs_Autocorr'] = obs_autocorr
                    metrics['Sim_Autocorr'] = sim_autocorr
                    metrics['Autocorr_Diff'] = abs(sim_autocorr - obs_autocorr)
            
            # Seasonal cycle analysis (if sufficient data)
            if len(observed) > 365:
                obs_seasonal = self._extract_seasonal_cycle(observed)
                sim_seasonal = self._extract_seasonal_cycle(simulated)
                
                if len(obs_seasonal) > 0 and len(sim_seasonal) > 0:
                    # Seasonal amplitude
                    obs_amplitude = obs_seasonal.max() - obs_seasonal.min()
                    sim_amplitude = sim_seasonal.max() - sim_seasonal.min()
                    
                    if obs_amplitude > 0:
                        seasonal_amplitude_ratio = sim_amplitude / obs_amplitude
                        metrics['Seasonal_Amplitude_Ratio'] = seasonal_amplitude_ratio
                    
                    # Seasonal timing (month of max/min)
                    obs_max_month = obs_seasonal.idxmax()
                    sim_max_month = sim_seasonal.idxmax()
                    
                    phase_diff = abs(obs_max_month - sim_max_month)
                    if phase_diff > 6:  # Handle wrap-around
                        phase_diff = 12 - phase_diff
                    
                    metrics['Seasonal_Phase_Diff_Months'] = phase_diff
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating soil moisture-specific metrics: {str(e)}")
            return {}
    
    def _extract_seasonal_cycle(self, data: pd.Series) -> pd.Series:
        """Extract average seasonal cycle from soil moisture data"""
        try:
            # Group by month and calculate mean
            monthly_means = data.groupby(data.index.month).mean()
            return monthly_means
            
        except Exception:
            return pd.Series()

class SnowTarget(CalibrationTarget):
    """Snow calibration target supporting both SWE and SCA"""
    
    def __init__(self, config: Dict, project_dir: Path, logger: logging.Logger):
        super().__init__(config, project_dir, logger)
        
        # Determine snow variable type from config
        self.optimization_target = config.get('OPTIMISATION_TARGET', config.get('CALIBRATION_VARIABLE', 'streamflow'))
        if self.optimization_target not in ['swe', 'sca', 'snow_depth']:
            # Check if it's a snow-related target but with different naming
            calibration_var = config.get('CALIBRATION_VARIABLE', '').lower()
            if 'swe' in calibration_var or 'snow' in calibration_var:
                self.optimization_target = 'swe'
            else:
                raise ValueError(f"Invalid snow optimization target: {self.optimization_target}")
        
        self.variable_name = self.optimization_target
        self.logger.info(f"Initialized SnowTarget for {self.optimization_target.upper()} calibration")
    
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Get SUMMA daily output files containing snow variables"""
        # Look for daily output files (they contain snow variables)
        daily_files = list(sim_dir.glob("*_day.nc"))
        if daily_files:
            return daily_files
        
        # Fallback to timestep files if daily not available
        return list(sim_dir.glob("*timestep.nc"))
    
    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract snow data from SUMMA simulation files"""
        sim_file = sim_files[0]  # Use first file
        
        try:
            with xr.open_dataset(sim_file) as ds:
                if self.optimization_target == 'swe':
                    return self._extract_swe_data(ds)
                elif self.optimization_target == 'sca':
                    return self._extract_sca_data(ds)
                else:
                    raise ValueError(f"Unknown snow target: {self.optimization_target}")
                    
        except Exception as e:
            self.logger.error(f"Error extracting snow data from {sim_file}: {str(e)}")
            raise
    
    def _extract_swe_data(self, ds: xr.Dataset) -> pd.Series:
        """Extract SWE data from SUMMA output"""
        if 'scalarSWE' not in ds.variables:
            raise ValueError("scalarSWE variable not found in SUMMA output")
        
        swe_var = ds['scalarSWE']
        self.logger.debug(f"Found scalarSWE variable with shape: {swe_var.shape}")
        
        # Extract data - handle spatial dimensions
        if len(swe_var.shape) > 1:
            if 'hru' in swe_var.dims:
                # Take first HRU or average across HRUs
                if swe_var.shape[swe_var.dims.index('hru')] == 1:
                    sim_data = swe_var.isel(hru=0).to_pandas()
                else:
                    # Average across HRUs for distributed domain
                    sim_data = swe_var.mean(dim='hru').to_pandas()
                    self.logger.info(f"Averaged SWE across {swe_var.shape[swe_var.dims.index('hru')]} HRUs")
            else:
                # Handle other dimensions
                non_time_dims = [dim for dim in swe_var.dims if dim != 'time']
                if non_time_dims:
                    sim_data = swe_var.isel({non_time_dims[0]: 0}).to_pandas()
                else:
                    sim_data = swe_var.to_pandas()
        else:
            sim_data = swe_var.to_pandas()
        
        # SUMMA outputs SWE in kg/m² which is equivalent to mm water equivalent
        # No unit conversion needed for simulated data
        self.logger.debug(f"Extracted SWE data: {len(sim_data)} timesteps, range: {sim_data.min():.2f} to {sim_data.max():.2f} kg/m² (mm)")
        
        return sim_data
    
    def _extract_sca_data(self, ds: xr.Dataset) -> pd.Series:
        """Extract SCA data from SUMMA output"""
        # Try different possible SCA variables
        sca_vars = ['scalarGroundSnowFraction', 'scalarSWE']  # Can derive SCA from SWE
        
        for var_name in sca_vars:
            if var_name in ds.variables:
                sca_var = ds[var_name]
                self.logger.debug(f"Found {var_name} variable with shape: {sca_var.shape}")
                
                # Extract data - handle spatial dimensions
                if len(sca_var.shape) > 1:
                    if 'hru' in sca_var.dims:
                        if sca_var.shape[sca_var.dims.index('hru')] == 1:
                            sim_data = sca_var.isel(hru=0).to_pandas()
                        else:
                            # Average across HRUs for distributed domain
                            sim_data = sca_var.mean(dim='hru').to_pandas()
                            self.logger.info(f"Averaged {var_name} across {sca_var.shape[sca_var.dims.index('hru')]} HRUs")
                    else:
                        non_time_dims = [dim for dim in sca_var.dims if dim != 'time']
                        if non_time_dims:
                            sim_data = sca_var.isel({non_time_dims[0]: 0}).to_pandas()
                        else:
                            sim_data = sca_var.to_pandas()
                else:
                    sim_data = sca_var.to_pandas()
                
                # Convert to SCA fraction if using SWE
                if var_name == 'scalarSWE':
                    # Convert SWE to binary snow cover (0 or 1)
                    swe_threshold = 1.0  # kg/m² threshold for snow presence
                    sim_data = (sim_data > swe_threshold).astype(float)
                    self.logger.info(f"Converted SWE to binary SCA using threshold {swe_threshold} kg/m²")
                
                self.logger.debug(f"Extracted SCA data: {len(sim_data)} timesteps, range: {sim_data.min():.3f} to {sim_data.max():.3f}")
                return sim_data
        
        raise ValueError("No suitable SCA variable found in SUMMA output")
    
    def get_observed_data_path(self) -> Path:
        """Get path to observed snow data"""
        if self.optimization_target == 'swe':
            return self.project_dir / "observations" / "snow" / "swe" / "processed" / f"{self.domain_name}_swe_processed.csv"
        elif self.optimization_target == 'sca':
            return self.project_dir / "observations" / "snow" / "sca" / "processed" / f"{self.domain_name}_sca_processed.csv"
        else:
            raise ValueError(f"Unknown snow target: {self.optimization_target}")
    
    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        """Identify the snow data column in observed data file"""
        if self.optimization_target == 'swe':
            # Look for SWE column
            for col in columns:
                if any(term in col.lower() for term in ['swe', 'snow_water_equivalent']):
                    return col
            # Fallback to exact match
            if 'SWE' in columns:
                return 'SWE'
                
        elif self.optimization_target == 'sca':
            # Look for snow cover ratio column
            for col in columns:
                if any(term in col.lower() for term in ['snow_cover_ratio', 'sca', 'snow_cover']):
                    return col
            # Fallback to exact match
            if 'snow_cover_ratio' in columns:
                return 'snow_cover_ratio'
        
        return None
    
    def _load_observed_data(self) -> Optional[pd.Series]:
        """Load observed snow data with proper date parsing, unit handling, and robust data cleaning"""
        try:
            obs_path = self.get_observed_data_path()
            if not obs_path.exists():
                self.logger.error(f"Observed snow data file not found: {obs_path}")
                return None
            
            obs_df = pd.read_csv(obs_path)
            self.logger.debug(f"Loaded observed data with columns: {list(obs_df.columns)}")
            
            # Find date and data columns
            date_col = self._find_date_column(obs_df.columns)
            data_col = self._get_observed_data_column(obs_df.columns)
            
            if not date_col or not data_col:
                self.logger.error(f"Could not identify date/data columns in {obs_path}")
                self.logger.error(f"Available columns: {list(obs_df.columns)}")
                return None
            
            self.logger.debug(f"Using date column: '{date_col}', data column: '{data_col}'")
            
            # Process data
            if self.optimization_target == 'swe':
                obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], format='%d/%m/%Y', errors='coerce')
            else:  # sca
                obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], errors='coerce')
            
            # Remove rows with invalid dates
            invalid_dates = obs_df['DateTime'].isna()
            if invalid_dates.any():
                self.logger.debug(f"Removing {invalid_dates.sum()} rows with invalid dates")
                obs_df = obs_df.dropna(subset=['DateTime'])
            
            obs_df.set_index('DateTime', inplace=True)
            
            # Extract data column and convert to numeric, handling various missing value indicators
            obs_series = obs_df[data_col].copy()
            
            # Replace common missing value indicators with NaN
            missing_indicators = ['', ' ', 'NA', 'na', 'N/A', 'n/a', 'NULL', 'null', '-', '--', '---', 'missing', 'Missing', 'MISSING']
            for indicator in missing_indicators:
                obs_series = obs_series.replace(indicator, np.nan)
            
            # Convert to numeric, coercing errors to NaN
            obs_series = pd.to_numeric(obs_series, errors='coerce')
            
            # Log data quality info
            total_points = len(obs_series)
            nan_count = obs_series.isna().sum()
            zero_count = (obs_series == 0).sum()
            valid_count = total_points - nan_count
            
            self.logger.debug(f"Data quality: {total_points} total, {valid_count} valid, {nan_count} NaN, {zero_count} zeros")
            
            # Handle unit conversion for SWE if needed (before removing negative values)
            if self.optimization_target == 'swe':
                obs_series = self._convert_swe_units(obs_series)
            
            # Remove negative values (not physically meaningful for SWE)
            if self.optimization_target == 'swe':
                negative_mask = obs_series < 0
                if negative_mask.any():
                    negative_count = negative_mask.sum()
                    self.logger.warning(f"Removing {negative_count} negative SWE values")
                    obs_series[negative_mask] = np.nan
            
            # Remove NaN values for final dataset
            obs_series = obs_series.dropna()
            
            if len(obs_series) == 0:
                self.logger.error("No valid observations remaining after data cleaning")
                return None
            
            # Log final statistics
            self.logger.debug(f"Final dataset: {len(obs_series)} valid observations")
            self.logger.debug(f"Observed data range: {obs_series.min():.3f} to {obs_series.max():.3f}")
            
            # Check for reasonable data range after unit conversion
            if self.optimization_target == 'swe':
                if obs_series.max() > 10000:  # Very high SWE values might indicate unit issues
                    self.logger.warning(f"Very high SWE values detected (max: {obs_series.max():.1f} kg/m²). Please verify units.")
                elif obs_series.max() < 1:  # Very low SWE values might indicate unit issues
                    self.logger.warning(f"Very low SWE values detected (max: {obs_series.max():.3f} kg/m²). Please verify units.")
                #else:

                    #self.logger.info(f"SWE values appear reasonable (max: {obs_series.max():.1f} kg/m²)")
            
            return obs_series
            
        except Exception as e:
            self.logger.error(f"Error loading observed snow data: {str(e)}")
            return None
    
    def _find_date_column(self, columns: List[str]) -> Optional[str]:
        """Find date column in observed data"""
        date_candidates = ['date', 'Date', 'DATE', 'datetime', 'DateTime', 'time', 'Time']
        
        for candidate in date_candidates:
            if candidate in columns:
                return candidate
        
        # Look for columns containing date-like terms
        for col in columns:
            if any(term in col.lower() for term in ['date', 'time']):
                return col
        
        return None
        
    def _convert_swe_units(self, obs_swe: pd.Series) -> pd.Series:
        """Convert SWE units from inches to kg/m² (mm water equivalent) - FORCE INCHES"""
        max_value = obs_swe.max()
        
        if pd.isna(max_value) or max_value <= 0:
            self.logger.warning("No valid SWE values found in observed data")
            return obs_swe
        
        # FORCE INCHES CONVERSION - REMOVE AUTO-DETECTION
        #self.logger.info(f"FORCING conversion from inches to kg/m² (factor: 25.4)")
        #self.logger.info(f"Original data range: {obs_swe.min():.1f} to {obs_swe.max():.1f} inches")
        
        # Convert inches to kg/m² (1 inch = 25.4 mm = 25.4 kg/m²)
        converted = obs_swe * 25.4
        
        #self.logger.info(f"Converted data range: {converted.min():.1f} to {converted.max():.1f} kg/m²")
        
        return converted


    def debug_swe_data_comparison(self, observed: pd.Series, simulated: pd.Series) -> Dict[str, Any]:
        """Debug function to analyze observed vs simulated SWE data"""
        debug_info = {}
        
        # Basic statistics
        debug_info['observed_stats'] = {
            'count': len(observed),
            'min': observed.min(),
            'max': observed.max(),
            'mean': observed.mean(),
            'std': observed.std(),
            'median': observed.median()
        }
        
        debug_info['simulated_stats'] = {
            'count': len(simulated),
            'min': simulated.min(),
            'max': simulated.max(),
            'mean': simulated.mean(),
            'std': simulated.std(),
            'median': simulated.median()
        }
        
        # Scale comparison
        obs_range = observed.max() - observed.min()
        sim_range = simulated.max() - simulated.min()
        debug_info['scale_ratio'] = sim_range / obs_range if obs_range > 0 else np.inf
        debug_info['mean_ratio'] = simulated.mean() / observed.mean() if observed.mean() > 0 else np.inf
        
        # Time alignment check
        common_times = observed.index.intersection(simulated.index)
        debug_info['time_overlap'] = {
            'common_points': len(common_times),
            'obs_total': len(observed),
            'sim_total': len(simulated),
            'overlap_pct': len(common_times) / min(len(observed), len(simulated)) * 100
        }
        
        if len(common_times) > 0:
            obs_common = observed.loc[common_times]
            sim_common = simulated.loc[common_times]
            
            # Sample values for inspection
            debug_info['sample_comparison'] = {
                'first_10_obs': obs_common.head(10).tolist(),
                'first_10_sim': sim_common.head(10).tolist(),
                'correlation': obs_common.corr(sim_common)
            }
            
            # Check for obvious issues
            debug_info['potential_issues'] = []
            
            if abs(debug_info['mean_ratio']) > 100 or abs(debug_info['mean_ratio']) < 0.01:
                debug_info['potential_issues'].append(f"Large scale difference: sim/obs mean ratio = {debug_info['mean_ratio']:.2f}")
            
            if debug_info['sample_comparison']['correlation'] < -0.5:
                debug_info['potential_issues'].append("Strong negative correlation - possible inverted relationship")
            
            if np.isnan(debug_info['sample_comparison']['correlation']):
                debug_info['potential_issues'].append("No correlation - data might be constant or misaligned")
            
            # Check for zero/constant issues
            obs_zeros = (obs_common == 0).sum()
            sim_zeros = (sim_common == 0).sum()
            
            if obs_zeros > len(obs_common) * 0.8:
                debug_info['potential_issues'].append(f"Observed data mostly zeros: {obs_zeros}/{len(obs_common)} points")
            
            if sim_zeros > len(sim_common) * 0.8:
                debug_info['potential_issues'].append(f"Simulated data mostly zeros: {sim_zeros}/{len(sim_common)} points")
        
        return debug_info


    def needs_routing(self) -> bool:
        """Snow calibration doesn't need routing"""
        return False
        
    def _calculate_performance_metrics(self, observed: pd.Series, simulated: pd.Series) -> Dict[str, float]:
        """Calculate snow-specific performance metrics with debugging"""
        try:
            # Clean data
            observed = pd.to_numeric(observed, errors='coerce')
            simulated = pd.to_numeric(simulated, errors='coerce')
            
            valid = ~(observed.isna() | simulated.isna())
            observed = observed[valid]
            simulated = simulated[valid]
            
            if len(observed) == 0:
                return {'KGE': np.nan, 'NSE': np.nan, 'RMSE': np.nan, 'PBIAS': np.nan, 'MAE': np.nan}
            
            # DEBUG: Add detailed comparison analysis
            debug_info = self.debug_swe_data_comparison(observed, simulated)
            
            # Log debug information
            #self.logger.error("=== SWE DATA DEBUG ANALYSIS ===")
            #self.logger.error(f"Observed stats: min={debug_info['observed_stats']['min']:.2f}, max={debug_info['observed_stats']['max']:.2f}, mean={debug_info['observed_stats']['mean']:.2f}")
            #self.logger.error(f"Simulated stats: min={debug_info['simulated_stats']['min']:.2f}, max={debug_info['simulated_stats']['max']:.2f}, mean={debug_info['simulated_stats']['mean']:.2f}")
            #self.logger.error(f"Scale ratio (sim/obs): {debug_info['scale_ratio']:.2f}")
            #self.logger.error(f"Mean ratio (sim/obs): {debug_info['mean_ratio']:.2f}")
            #self.logger.error(f"Time overlap: {debug_info['time_overlap']['common_points']} points ({debug_info['time_overlap']['overlap_pct']:.1f}%)")
            
            #if 'sample_comparison' in debug_info:
            #    self.logger.error(f"Correlation: {debug_info['sample_comparison']['correlation']:.4f}")
            #    self.logger.error(f"First 3 observed: {debug_info['sample_comparison']['first_10_obs'][:3]}")
            #    self.logger.error(f"First 3 simulated: {debug_info['sample_comparison']['first_10_sim'][:3]}")
            
            #if debug_info['potential_issues']:
            #    self.logger.error("POTENTIAL ISSUES DETECTED:")
            #    for issue in debug_info['potential_issues']:
            #        self.logger.error(f"  - {issue}")
            
            #self.logger.error("=== END DEBUG ANALYSIS ===")
            
            # Standard metrics
            base_metrics = super()._calculate_performance_metrics(observed, simulated)
            
            # Add snow-specific metrics
            if self.optimization_target == 'swe':
                swe_metrics = self._calculate_swe_metrics(observed, simulated)
                base_metrics.update(swe_metrics)
            
            return base_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating snow performance metrics: {str(e)}")
            return {'KGE': np.nan, 'NSE': np.nan, 'RMSE': np.nan, 'PBIAS': np.nan, 'MAE': np.nan}
    
    def _calculate_sca_metrics(self, observed: pd.Series, simulated: pd.Series) -> Dict[str, float]:
        """Calculate SCA-specific categorical metrics"""
        try:
            # Convert to binary if not already
            obs_binary = (observed > 0.5).astype(int)
            sim_binary = (simulated > 0.5).astype(int)
            
            # Contingency table
            true_pos = np.sum((obs_binary == 1) & (sim_binary == 1))
            false_pos = np.sum((obs_binary == 0) & (sim_binary == 1))
            false_neg = np.sum((obs_binary == 1) & (sim_binary == 0))
            true_neg = np.sum((obs_binary == 0) & (sim_binary == 0))
            
            # Categorical metrics
            pod = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else np.nan  # Probability of Detection
            far = false_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else np.nan  # False Alarm Ratio
            csi = true_pos / (true_pos + false_pos + false_neg) if (true_pos + false_pos + false_neg) > 0 else np.nan  # Critical Success Index
            accuracy = (true_pos + true_neg) / len(observed) if len(observed) > 0 else np.nan
            
            return {
                'POD': pod,
                'FAR': far, 
                'CSI': csi,
                'Accuracy': accuracy
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating SCA metrics: {str(e)}")
            return {}
    
    def _calculate_swe_metrics(self, observed: pd.Series, simulated: pd.Series) -> Dict[str, float]:
        """Calculate SWE-specific metrics"""
        try:
            metrics = {}
            
            # Peak SWE timing and magnitude
            if len(observed) > 10:  # Need reasonable amount of data
                obs_peak_idx = observed.idxmax()
                sim_peak_idx = simulated.idxmax()
                
                obs_peak_value = observed.max()
                sim_peak_value = simulated.max()
                
                # Peak timing difference (days)
                if pd.notna(obs_peak_idx) and pd.notna(sim_peak_idx):
                    peak_timing_diff = abs((sim_peak_idx - obs_peak_idx).days)
                    metrics['Peak_Timing_Error_Days'] = peak_timing_diff
                
                # Peak magnitude error
                if pd.notna(obs_peak_value) and pd.notna(sim_peak_value):
                    peak_magnitude_error = abs(sim_peak_value - obs_peak_value) / obs_peak_value * 100
                    metrics['Peak_Magnitude_Error_Pct'] = peak_magnitude_error
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating SWE metrics: {str(e)}")
            return {}

class GroundwaterTarget(CalibrationTarget):
    """Groundwater calibration target supporting both depth and GRACE TWS"""
    
    def __init__(self, config: Dict, project_dir: Path, logger: logging.Logger):
        super().__init__(config, project_dir, logger)
        
        # Determine groundwater variable type from config
        self.optimization_target = config.get('OPTIMISATION_TARGET', 'streamflow')
        if self.optimization_target not in ['gw_depth', 'gw_grace']:
            raise ValueError(f"Invalid groundwater optimization target: {self.optimization_target}")
        
        self.variable_name = self.optimization_target
        
        # GRACE processing center preference
        self.grace_center = config.get('GRACE_PROCESSING_CENTER', 'csr')  # Options: jpl, csr, gsfc
        
        self.logger.info(f"Initialized GroundwaterTarget for {self.optimization_target.upper()} calibration")
        if self.optimization_target == 'gw_grace':
            self.logger.info(f"Using GRACE {self.grace_center.upper()} processing center data")
    
    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Get SUMMA daily output files containing groundwater variables"""
        # Look for daily output files (they contain groundwater variables)
        daily_files = list(sim_dir.glob("*_day.nc"))
        if daily_files:
            return daily_files
        
        # Fallback to timestep files if daily not available
        return list(sim_dir.glob("*timestep.nc"))
    
    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract groundwater data from SUMMA simulation files"""
        sim_file = sim_files[0]  # Use first file
        
        try:
            with xr.open_dataset(sim_file) as ds:
                if self.optimization_target == 'gw_depth':
                    return self._extract_groundwater_depth(ds)
                elif self.optimization_target == 'gw_grace':
                    return self._extract_total_water_storage(ds)
                else:
                    raise ValueError(f"Unknown groundwater target: {self.optimization_target}")
                    
        except Exception as e:
            self.logger.error(f"Error extracting groundwater data from {sim_file}: {str(e)}")
            raise
    
    def _extract_groundwater_depth(self, ds: xr.Dataset) -> pd.Series:
        """Extract groundwater depth from SUMMA output"""
        # Primary variable: scalarAquiferStorage (water needed to reach bottom of soil profile)
        if 'scalarAquiferStorage' in ds.variables:
            gw_var = ds['scalarAquiferStorage']
            self.logger.debug(f"Found scalarAquiferStorage variable with shape: {gw_var.shape}")
            
            # Extract data - handle spatial dimensions
            if len(gw_var.shape) > 1:
                if 'hru' in gw_var.dims:
                    if gw_var.shape[gw_var.dims.index('hru')] == 1:
                        sim_data = gw_var.isel(hru=0).to_pandas()
                    else:
                        # Average across HRUs for distributed domain
                        sim_data = gw_var.mean(dim='hru').to_pandas()
                        self.logger.info(f"Averaged aquifer storage across {gw_var.shape[gw_var.dims.index('hru')]} HRUs")
                else:
                    non_time_dims = [dim for dim in gw_var.dims if dim != 'time']
                    if non_time_dims:
                        sim_data = gw_var.isel({non_time_dims[0]: 0}).to_pandas()
                    else:
                        sim_data = gw_var.to_pandas()
            else:
                sim_data = gw_var.to_pandas()
            
            # scalarAquiferStorage is in meters - this represents depth to water table
            # Convert to positive depth (if negative values, make positive)
            sim_data = sim_data.abs()
            
            self.logger.debug(f"Extracted groundwater depth: {len(sim_data)} timesteps, range: {sim_data.min():.2f} to {sim_data.max():.2f} m")
            return sim_data
            
        else:
            # Fallback: try to derive from soil moisture layers
            self.logger.warning("scalarAquiferStorage not found, attempting to derive from soil layers")
            return self._derive_groundwater_depth(ds)
    
    def _derive_groundwater_depth(self, ds: xr.Dataset) -> pd.Series:
        """Derive groundwater depth from soil moisture layers"""
        try:
            if 'mLayerVolFracLiq' not in ds.variables or 'mLayerDepth' not in ds.variables:
                raise ValueError("Required soil layer variables not found for groundwater depth derivation")
            
            vol_frac_liq = ds['mLayerVolFracLiq']
            layer_depths = ds['mLayerDepth']
            
            # Calculate cumulative depth and find water table
            # This is a simplified approach - water table is where saturation drops below threshold
            saturation_threshold = 0.8  # 80% saturation threshold
            
            # Take first HRU if multiple
            if 'hru' in vol_frac_liq.dims and vol_frac_liq.shape[vol_frac_liq.dims.index('hru')] > 1:
                vol_frac_liq = vol_frac_liq.isel(hru=0)
                layer_depths = layer_depths.isel(hru=0)
            elif 'hru' in vol_frac_liq.dims:
                vol_frac_liq = vol_frac_liq.isel(hru=0)
                layer_depths = layer_depths.isel(hru=0)
            
            # Calculate cumulative depths
            cumulative_depths = layer_depths.cumsum(dim=layer_depths.dims[-2])  # Sum over layer dimension
            
            # Find water table depth (simplified)
            # Take mean depth of saturated layers as proxy for water table depth
            saturated_mask = vol_frac_liq > saturation_threshold
            
            # Calculate average depth of saturated zone
            water_table_depths = []
            for t in range(len(vol_frac_liq.time)):
                time_sat_mask = saturated_mask.isel(time=t)
                time_cum_depths = cumulative_depths.isel(time=t)
                
                if time_sat_mask.any():
                    # Find deepest saturated layer
                    sat_depths = time_cum_depths.where(time_sat_mask, drop=True)
                    if len(sat_depths) > 0:
                        water_table_depths.append(float(sat_depths.max()))
                    else:
                        water_table_depths.append(0.0)  # Surface
                else:
                    water_table_depths.append(float(time_cum_depths.max()))  # Deep water table
            
            sim_data = pd.Series(water_table_depths, index=vol_frac_liq.time.to_pandas())
            
            self.logger.info("Derived groundwater depth from soil moisture layers")
            self.logger.debug(f"Derived depth range: {sim_data.min():.2f} to {sim_data.max():.2f} m")
            
            return sim_data
            
        except Exception as e:
            self.logger.error(f"Error deriving groundwater depth: {str(e)}")
            raise
    
    def _extract_total_water_storage(self, ds: xr.Dataset) -> pd.Series:
        """Extract total water storage for GRACE comparison"""
        try:
            # Components of total water storage
            storage_components = {}
            
            # 1. Snow water equivalent
            if 'scalarSWE' in ds.variables:
                storage_components['swe'] = ds['scalarSWE']
            
            # 2. Soil water storage
            if 'scalarTotalSoilWat' in ds.variables:
                storage_components['soil'] = ds['scalarTotalSoilWat']
            
            # 3. Aquifer storage
            if 'scalarAquiferStorage' in ds.variables:
                storage_components['aquifer'] = ds['scalarAquiferStorage']
            
            # 4. Canopy water (if available)
            if 'scalarCanopyWat' in ds.variables:
                storage_components['canopy'] = ds['scalarCanopyWat']
            elif 'scalarCanopyLiq' in ds.variables:
                storage_components['canopy'] = ds['scalarCanopyLiq']
            
            if not storage_components:
                raise ValueError("No water storage components found for TWS calculation")
            
            self.logger.info(f"Found TWS components: {list(storage_components.keys())}")
            
            # Sum all components to get total water storage
            total_storage = None
            for component_name, component_data in storage_components.items():
                # Handle spatial dimensions
                if len(component_data.shape) > 1:
                    if 'hru' in component_data.dims:
                        if component_data.shape[component_data.dims.index('hru')] == 1:
                            component_series = component_data.isel(hru=0)
                        else:
                            # Average across HRUs
                            component_series = component_data.mean(dim='hru')
                    else:
                        non_time_dims = [dim for dim in component_data.dims if dim != 'time']
                        if non_time_dims:
                            component_series = component_data.isel({non_time_dims[0]: 0})
                        else:
                            component_series = component_data
                else:
                    component_series = component_data
                
                if total_storage is None:
                    total_storage = component_series
                else:
                    total_storage = total_storage + component_series
                
                self.logger.debug(f"Added {component_name} component to TWS")
            
            # Convert to pandas Series
            sim_data = total_storage.to_pandas()
            
            # Convert units to match GRACE (typically cm equivalent water height)
            # Most SUMMA variables are in kg/m² or m, convert to cm
            sim_data = self._convert_tws_units(sim_data)
            
            self.logger.debug(f"Extracted TWS: {len(sim_data)} timesteps, range: {sim_data.min():.2f} to {sim_data.max():.2f} cm")
            
            return sim_data
            
        except Exception as e:
            self.logger.error(f"Error calculating total water storage: {str(e)}")
            raise
    
    def _convert_tws_units(self, tws_data: pd.Series) -> pd.Series:
        """Convert TWS units to cm equivalent water height"""
        # Check data range to infer units
        data_range = tws_data.max() - tws_data.min()
        
        if data_range > 1000:  # Likely in kg/m² (mm)
            # Convert kg/m² to cm (divide by 10)
            converted = tws_data / 10.0
            self.logger.info("Converted TWS from kg/m² to cm (factor: 0.1)")
        elif data_range > 10:  # Likely in meters
            # Convert meters to cm (multiply by 100)
            converted = tws_data * 100.0
            self.logger.info("Converted TWS from meters to cm (factor: 100)")
        else:  # Likely already in cm or similar
            converted = tws_data
            self.logger.info("TWS appears to be in cm, no conversion applied")
        
        return converted
    
    def get_observed_data_path(self) -> Path:
        """Get path to observed groundwater data"""
        if self.optimization_target == 'gw_depth':
            return self.project_dir / "observations" / "groundwater" / "depth" / "processed" / f"{self.domain_name}_gw_processed.csv"
        elif self.optimization_target == 'gw_grace':
            return self.project_dir / "observations" / "groundwater" / "grace" / "processed" / f"{self.domain_name}_grace_processed.csv"
        else:
            raise ValueError(f"Unknown groundwater target: {self.optimization_target}")
    
    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        """Identify the groundwater data column in observed data file"""
        if self.optimization_target == 'gw_depth':
            # Look for depth column
            for col in columns:
                if any(term in col.lower() for term in ['depth', 'depth_m', 'water_level']):
                    return col
            # Fallback to exact match
            if 'Depth_m' in columns:
                return 'Depth_m'
                
        elif self.optimization_target == 'gw_grace':
            # Look for GRACE TWS column based on processing center
            grace_columns = {
                'jpl': ['grace_jpl_tws'],
                'csr': ['grace_csr_tws'],
                'gsfc': ['grace_gsfc_tws']
            }
            
            preferred_cols = grace_columns.get(self.grace_center, ['grace_csr_tws'])
            
            for col in preferred_cols:
                if col in columns:
                    return col
            
            # Fallback to any GRACE column
            for col in columns:
                if 'grace' in col.lower() and 'tws' in col.lower():
                    return col
        
        return None
    
    def _load_observed_data(self) -> Optional[pd.Series]:
        """Load observed groundwater data with proper date parsing"""
        try:
            obs_path = self.get_observed_data_path()
            if not obs_path.exists():
                self.logger.error(f"Observed groundwater data file not found: {obs_path}")
                return None
            
            obs_df = pd.read_csv(obs_path)
            self.logger.debug(f"Loaded observed data with columns: {list(obs_df.columns)}")
            
            # Find date and data columns
            date_col = self._find_date_column(obs_df.columns)
            data_col = self._get_observed_data_column(obs_df.columns)
            
            if not date_col or not data_col:
                self.logger.error(f"Could not identify date/data columns in {obs_path}")
                self.logger.error(f"Available columns: {list(obs_df.columns)}")
                return None
            
            self.logger.debug(f"Using date column: '{date_col}', data column: '{data_col}'")
            
            # Process data based on target type
            if self.optimization_target == 'gw_depth':
                # Handle timezone-aware datetime
                obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], errors='coerce')
            else:  # gw_grace
                # Handle MM/DD/YYYY format
                obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], format='%m/%d/%Y', errors='coerce')
            
            # Remove rows with invalid dates
            obs_df = obs_df.dropna(subset=['DateTime'])
            obs_df.set_index('DateTime', inplace=True)
            
            # Extract data column and clean
            obs_series = pd.to_numeric(obs_df[data_col], errors='coerce')
            
            # Remove NaN values
            obs_series = obs_series.dropna()
            
            # Convert timezone-naive for consistency
            if obs_series.index.tz is not None:
                obs_series.index = obs_series.index.tz_convert('UTC').tz_localize(None)
            
            self.logger.info(f"Loaded {len(obs_series)} valid observations")
            self.logger.debug(f"Observed data range: {obs_series.min():.3f} to {obs_series.max():.3f}")
            
            return obs_series
            
        except Exception as e:
            self.logger.error(f"Error loading observed groundwater data: {str(e)}")
            return None
    
    def _find_date_column(self, columns: List[str]) -> Optional[str]:
        """Find date column in observed data"""
        date_candidates = ['time', 'Time', 'date', 'Date', 'DATE', 'datetime', 'DateTime']
        
        for candidate in date_candidates:
            if candidate in columns:
                return candidate
        
        # Look for columns containing date-like terms
        for col in columns:
            if any(term in col.lower() for term in ['date', 'time']):
                return col
        
        return None
    
    def needs_routing(self) -> bool:
        """Groundwater calibration doesn't need routing"""
        return False
    
    def _calculate_performance_metrics(self, observed: pd.Series, simulated: pd.Series) -> Dict[str, float]:
        """Calculate groundwater-specific performance metrics"""
        try:
            # Clean data
            observed = pd.to_numeric(observed, errors='coerce')
            simulated = pd.to_numeric(simulated, errors='coerce')
            
            valid = ~(observed.isna() | simulated.isna())
            observed = observed[valid]
            simulated = simulated[valid]
            
            if len(observed) == 0:
                return {'KGE': np.nan, 'NSE': np.nan, 'RMSE': np.nan, 'PBIAS': np.nan, 'MAE': np.nan}
            
            # Standard metrics
            base_metrics = super()._calculate_performance_metrics(observed, simulated)
            
            # Add groundwater-specific metrics
            if self.optimization_target == 'gw_depth':
                depth_metrics = self._calculate_depth_metrics(observed, simulated)
                base_metrics.update(depth_metrics)
            elif self.optimization_target == 'gw_grace':
                tws_metrics = self._calculate_tws_metrics(observed, simulated)
                base_metrics.update(tws_metrics)
            
            return base_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating groundwater performance metrics: {str(e)}")
            return {'KGE': np.nan, 'NSE': np.nan, 'RMSE': np.nan, 'PBIAS': np.nan, 'MAE': np.nan}
    
    def _calculate_depth_metrics(self, observed: pd.Series, simulated: pd.Series) -> Dict[str, float]:
        """Calculate groundwater depth-specific metrics"""
        try:
            metrics = {}
            
            # Range preservation
            obs_range = observed.max() - observed.min()
            sim_range = simulated.max() - simulated.min()
            
            if obs_range > 0:
                range_ratio = sim_range / obs_range
                metrics['Range_Ratio'] = range_ratio
                
                # Range bias (how well does model capture variability)
                range_bias = (sim_range - obs_range) / obs_range * 100
                metrics['Range_Bias_Pct'] = range_bias
            
            # Trend analysis (are long-term trends captured?)
            if len(observed) > 30:  # Need sufficient data
                obs_trend = self._calculate_trend(observed)
                sim_trend = self._calculate_trend(simulated)
                
                metrics['Obs_Trend'] = obs_trend
                metrics['Sim_Trend'] = sim_trend
                
                if abs(obs_trend) > 1e-6:  # Avoid division by near-zero
                    trend_ratio = sim_trend / obs_trend
                    metrics['Trend_Ratio'] = trend_ratio
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating depth metrics: {str(e)}")
            return {}
    
    def _calculate_tws_metrics(self, observed: pd.Series, simulated: pd.Series) -> Dict[str, float]:
        """Calculate GRACE TWS-specific metrics"""
        try:
            metrics = {}
            
            # Seasonal cycle analysis
            if len(observed) > 365:  # Need at least a year of data
                obs_seasonal = self._extract_seasonal_cycle(observed)
                sim_seasonal = self._extract_seasonal_cycle(simulated)
                
                # Seasonal amplitude comparison
                obs_amplitude = obs_seasonal.max() - obs_seasonal.min()
                sim_amplitude = sim_seasonal.max() - sim_seasonal.min()
                
                if obs_amplitude > 0:
                    amplitude_ratio = sim_amplitude / obs_amplitude
                    metrics['Seasonal_Amplitude_Ratio'] = amplitude_ratio
                
                # Seasonal timing (month of maximum)
                obs_max_month = obs_seasonal.idxmax()
                sim_max_month = sim_seasonal.idxmax()
                
                phase_diff = abs(obs_max_month - sim_max_month)
                if phase_diff > 6:  # Handle wrap-around
                    phase_diff = 12 - phase_diff
                
                metrics['Seasonal_Phase_Diff_Months'] = phase_diff
            
            # Anomaly correlation (detrended correlation)
            obs_anomaly = observed - observed.rolling(window=12, center=True).mean()
            sim_anomaly = simulated - simulated.rolling(window=12, center=True).mean()
            
            valid_anomaly = ~(obs_anomaly.isna() | sim_anomaly.isna())
            if valid_anomaly.sum() > 10:
                anomaly_corr = obs_anomaly[valid_anomaly].corr(sim_anomaly[valid_anomaly])
                metrics['Anomaly_Correlation'] = anomaly_corr
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating TWS metrics: {str(e)}")
            return {}
    
    def _calculate_trend(self, data: pd.Series) -> float:
        """Calculate linear trend in data (units per year)"""
        try:
            if len(data) < 10:
                return 0.0
            
            # Convert index to numeric (days since start)
            numeric_time = (data.index - data.index[0]).days.values
            numeric_time = numeric_time / 365.25  # Convert to years
            
            # Linear regression
            coeffs = np.polyfit(numeric_time, data.values, 1)
            trend_per_year = coeffs[0]
            
            return trend_per_year
            
        except Exception:
            return 0.0
    
    def _extract_seasonal_cycle(self, data: pd.Series) -> pd.Series:
        """Extract average seasonal cycle from data"""
        try:
            # Group by month and calculate mean
            monthly_means = data.groupby(data.index.month).mean()
            return monthly_means
            
        except Exception:
            return pd.Series()


