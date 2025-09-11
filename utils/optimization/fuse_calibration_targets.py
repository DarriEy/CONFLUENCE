#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FUSE Calibration Target

Provides calibration target classes specifically designed for FUSE model output.
Handles FUSE-specific file patterns and data structures.
"""

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
import sys

# Import evaluation functions
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.evaluation.calculate_sim_stats import get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE


class FUSECalibrationTarget:
    """Base class for FUSE calibration targets"""
    
    def __init__(self, config: Dict[str, Any], project_dir: Path, logger: logging.Logger):
        self.config = config
        self.project_dir = project_dir
        self.logger = logger
        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')
        
        # Get observation data
        self._load_observations()
        
        # Get catchment area for unit conversion
        self.catchment_area_km2 = self._get_catchment_area()
    
    def _load_observations(self):
        """Load observation data (to be implemented by subclasses)"""
        pass
    
    def _get_catchment_area(self) -> float:
        """Get catchment area for unit conversion with improved error handling"""
        try:
            # Try multiple shapefile sources in order of preference
            area_sources = [
                ('RIVER_BASINS_PATH', 'RIVER_BASINS_NAME', 'shapefiles/river_basins', 'riverBasins', 'GRU_area'),
                ('CATCHMENT_PATH', 'CATCHMENT_SHP_NAME', 'shapefiles/catchment', 'HRUs', 'HRU_area'),
                ('CATCHMENT_PATH', 'CATCHMENT_SHP_NAME', 'shapefiles/catchment', 'HRUs', 'GRU_area')
            ]
            
            for path_key, name_key, default_subdir, file_pattern, area_col in area_sources:
                try:
                    area = self._try_get_area_from_shapefile(path_key, name_key, default_subdir, file_pattern, area_col)
                    if area and area > 0:
                        self.logger.info(f"Successfully calculated catchment area: {area:.2f} km2 from {area_col}")
                        return area
                except Exception as e:
                    self.logger.debug(f"Failed to get area from {path_key}/{area_col}: {str(e)}")
                    continue
            
            # If all methods fail, look for any available shapefiles
            self.logger.warning("Standard area calculation failed, searching for available shapefiles...")
            return self._search_for_any_shapefile_area()
            
        except Exception as e:
            self.logger.warning(f"Error calculating catchment area: {str(e)}")
        
        # Default fallback
        self.logger.warning("Using default catchment area: 1000 km2")
        return 1000.0
    
    def _try_get_area_from_shapefile(self, path_key: str, name_key: str, default_subdir: str, 
                                   file_pattern: str, area_col: str) -> Optional[float]:
        """Try to get area from a specific shapefile configuration"""
        # Get shapefile path
        shapefile_path = self.config.get(path_key)
        if shapefile_path == 'default':
            shapefile_path = self.project_dir / default_subdir
        else:
            shapefile_path = Path(shapefile_path)
        
        # Get shapefile name
        shapefile_name = self.config.get(name_key)
        if shapefile_name == 'default':
            domain_method = self.config.get('DOMAIN_DISCRETIZATION', 'GRUs')
            shapefile_name = f"{self.domain_name}_{file_pattern}_{domain_method}.shp"
        
        shapefile_file = shapefile_path / shapefile_name
        self.logger.debug(f"Trying shapefile: {shapefile_file}")
        
        if shapefile_file.exists():
            gdf = gpd.read_file(shapefile_file)
            self.logger.debug(f"Shapefile columns: {list(gdf.columns)}")
            
            # Check if area column exists
            if area_col in gdf.columns:
                total_area_m2 = gdf[area_col].sum()
                return total_area_m2 / 1e6  # Convert to km2
            else:
                # Calculate area from geometry
                if gdf.crs and not gdf.crs.is_geographic:
                    total_area_m2 = gdf.geometry.area.sum()
                else:
                    # Project to appropriate UTM zone for area calculation
                    gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
                    total_area_m2 = gdf_utm.geometry.area.sum()
                
                return total_area_m2 / 1e6  # Convert to km2
        
        return None
    
    def _search_for_any_shapefile_area(self) -> float:
        """Search for any available shapefiles in the project directory"""
        shapefile_dirs = ['shapefiles/river_basins', 'shapefiles/catchment', 'shapefiles']
        
        for subdir in shapefile_dirs:
            search_dir = self.project_dir / subdir
            if search_dir.exists():
                shapefiles = list(search_dir.glob("*.shp"))
                self.logger.info(f"Found shapefiles in {search_dir}: {[f.name for f in shapefiles]}")
                
                for shapefile in shapefiles:
                    try:
                        gdf = gpd.read_file(shapefile)
                        self.logger.debug(f"Checking {shapefile.name}, columns: {list(gdf.columns)}")
                        
                        # Look for any area column
                        area_cols = [col for col in gdf.columns if 'area' in col.lower()]
                        if area_cols:
                            area_col = area_cols[0]
                            total_area_m2 = gdf[area_col].sum()
                            area_km2 = total_area_m2 / 1e6
                            self.logger.info(f"Using area from {shapefile.name}, column {area_col}: {area_km2:.2f} km2")
                            return area_km2
                        else:
                            # Calculate from geometry
                            if gdf.crs and not gdf.crs.is_geographic:
                                total_area_m2 = gdf.geometry.area.sum()
                            else:
                                gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
                                total_area_m2 = gdf_utm.geometry.area.sum()
                            
                            area_km2 = total_area_m2 / 1e6
                            self.logger.info(f"Calculated area from {shapefile.name} geometry: {area_km2:.2f} km2")
                            return area_km2
                            
                    except Exception as e:
                        self.logger.debug(f"Error reading {shapefile}: {str(e)}")
                        continue
        
        return 1000.0  # Final fallback
    
    def _parse_datetime_robust(self, datetime_series: pd.Series, datetime_col: str) -> pd.Series:
        """
        Robust datetime parsing that tries multiple formats and handles different date conventions.
        
        Args:
            datetime_series: Series containing datetime strings
            datetime_col: Name of the datetime column (for logging)
            
        Returns:
            pd.Series: Parsed datetime series
        """
        # First check if it's already datetime
        if pd.api.types.is_datetime64_any_dtype(datetime_series):
            self.logger.info(f"Column '{datetime_col}' is already datetime format")
            return datetime_series
        
        # Try different parsing strategies
        parsing_strategies = [
            # Strategy 1: Use dayfirst=True for DD/MM/YYYY format
            lambda x: pd.to_datetime(x, dayfirst=True, errors='coerce'),
            
            # Strategy 2: Use format='mixed' to infer format for each element
            lambda x: pd.to_datetime(x, format='mixed', errors='coerce'),
            
            # Strategy 3: Try common formats explicitly
            lambda x: pd.to_datetime(x, format='%d/%m/%Y', errors='coerce'),
            lambda x: pd.to_datetime(x, format='%m/%d/%Y', errors='coerce'),
            lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors='coerce'),
            lambda x: pd.to_datetime(x, format='%d-%m-%Y', errors='coerce'),
            lambda x: pd.to_datetime(x, format='%m-%d-%Y', errors='coerce'),
            
            # Strategy 4: Default pandas parsing
            lambda x: pd.to_datetime(x, errors='coerce'),
        ]
        
        # Sample a few values to log what we're working with
        sample_values = datetime_series.dropna().iloc[:5].tolist()
        self.logger.info(f"Sample datetime values from '{datetime_col}': {sample_values}")
        
        for i, strategy in enumerate(parsing_strategies):
            try:
                parsed_series = strategy(datetime_series)
                
                # Check if parsing was successful (not too many NaT values)
                if parsed_series.notna().sum() > len(parsed_series) * 0.8:  # At least 80% success
                    successful_count = parsed_series.notna().sum()
                    self.logger.info(f"Successfully parsed {successful_count}/{len(parsed_series)} dates using strategy {i+1}")
                    
                    if parsed_series.notna().sum() < len(parsed_series):
                        failed_indices = parsed_series[parsed_series.isna()].index[:5]  # Show first 5 failures
                        failed_values = [datetime_series.iloc[idx] for idx in failed_indices]
                        self.logger.warning(f"Failed to parse some dates, examples: {failed_values}")
                    
                    return parsed_series
                else:
                    successful_count = parsed_series.notna().sum()
                    self.logger.debug(f"Strategy {i+1} only parsed {successful_count}/{len(parsed_series)} dates, trying next strategy")
                    
            except Exception as e:
                self.logger.debug(f"Strategy {i+1} failed with error: {str(e)}")
                continue
        
        # If all strategies fail, raise an error with helpful information
        self.logger.error(f"All datetime parsing strategies failed for column '{datetime_col}'")
        self.logger.error(f"Sample values: {sample_values}")
        raise ValueError(f"Could not parse datetime column '{datetime_col}'. Sample values: {sample_values}")
        
    def calculate_metrics(self, fuse_sim_dir: Path, mizuroute_dir: Optional[Path] = None) -> Optional[Dict[str, float]]:
        """Calculate metrics from FUSE simulation output"""
        try:
            self.logger.debug(f"Starting metrics calculation in: {fuse_sim_dir}")
            
            # Find FUSE simulation file (prefer runs_def.nc for calibration)
            sim_file = None
            available_files = []
            
            for pattern in ['runs_def.nc','runs_pre.nc', 'runs_best.nc']:
                potential_files = list(fuse_sim_dir.glob(f"*_{pattern}"))
                available_files.extend([f.name for f in potential_files])
                if potential_files:
                    sim_file = potential_files[0]
                    
                    # DEBUG: Check file timestamp
                    import os
                    mtime = os.path.getmtime(sim_file)
                    self.logger.debug(f"Using simulation file: {sim_file.name}, timestamp: {mtime}")
                    break
            
            self.logger.debug(f"Available simulation files: {available_files}")
            
            if not sim_file:
                self.logger.error(f"No FUSE simulation files found in {fuse_sim_dir}")
                return None
            
            self.logger.debug(f"Reading FUSE simulation from: {sim_file}")
            
            # Read FUSE simulation output
            with xr.open_dataset(sim_file) as ds:
                self.logger.debug(f"Dataset variables: {list(ds.variables.keys())}")
                
                # Get the appropriate runoff variable
                if 'q_routed' in ds.variables:
                    sim_var = 'q_routed'
                elif 'q_instnt' in ds.variables:
                    sim_var = 'q_instnt'
                else:
                    self.logger.error("No runoff variable found in FUSE output")
                    return None
                
                self.logger.debug(f"Using variable: {sim_var}")
                
                # Extract simulated data (assuming single parameter set)
                simulated = ds[sim_var].isel(param_set=0, latitude=0, longitude=0)
                simulated_df = simulated.to_pandas()
            
            # DEBUG: Log simulation statistics before unit conversion
            self.logger.debug(f"Raw simulation mean: {simulated_df.mean():.6f} mm/day")
            self.logger.debug(f"Raw simulation std: {simulated_df.std():.6f} mm/day")
            self.logger.debug(f"Raw simulation min: {simulated_df.min():.6f} mm/day")
            self.logger.debug(f"Raw simulation max: {simulated_df.max():.6f} mm/day")
            
            # Convert FUSE output from mm/day to cms if needed
            if self._needs_unit_conversion():
                # Q(cms) = Q(mm/day) * Area(km2) / 86.4
                simulated_df = simulated_df * self.catchment_area_km2 / 86.4
                self.logger.debug(f"Converted FUSE output to cms using area: {self.catchment_area_km2:.2f} km2")
            
            # DEBUG: Log simulation statistics after unit conversion
            self.logger.debug(f"Final simulation mean: {simulated_df.mean():.6f} cms")
            self.logger.debug(f"Final simulation std: {simulated_df.std():.6f} cms")
            
            # Calculate metrics with observations
            return self._calculate_performance_metrics(simulated_df)
            
        except Exception as e:
            self.logger.error(f"Error calculating FUSE metrics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _needs_unit_conversion(self) -> bool:
        """Check if unit conversion is needed (to be overridden by subclasses)"""
        return True
    
    def _calculate_performance_metrics(self, simulated_df: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement _calculate_performance_metrics")


class FUSEStreamflowTarget(FUSECalibrationTarget):
    """FUSE calibration target for streamflow"""
    
    def _load_observations(self):
        """Load streamflow observations with improved error handling"""
        try:
            # Try multiple possible paths for observations
            obs_paths = self._get_possible_observation_paths()
            
            obs_file_path = None
            for path in obs_paths:
                if path.exists():
                    obs_file_path = path
                    self.logger.info(f"Found streamflow observations at: {obs_file_path}")
                    break
            
            if not obs_file_path:
                # List available files for debugging
                self._debug_available_files()
                raise FileNotFoundError(f"No streamflow observation files found. Tried: {[str(p) for p in obs_paths]}")
            
            # Read observations with flexible column detection
            dfObs = pd.read_csv(obs_file_path)
            self.logger.info(f"Loaded CSV with columns: {list(dfObs.columns)}")
            
            # Find datetime and discharge columns
            datetime_col, discharge_col = self._identify_columns(dfObs)
            
            # Process datetime index with robust parsing
            dfObs = self._process_datetime_index(dfObs, datetime_col)
            
            # Extract discharge data
            self.observed_streamflow = dfObs[discharge_col].resample('d').mean()
            
            self.logger.info(f"Loaded streamflow observations: {len(self.observed_streamflow)} days")
            self.logger.info(f"Observation period: {self.observed_streamflow.index.min()} to {self.observed_streamflow.index.max()}")
            
        except Exception as e:
            self.logger.error(f"Error loading streamflow observations: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.observed_streamflow = None
    
    def _get_possible_observation_paths(self) -> List[Path]:
        """Get list of possible observation file paths"""
        obs_file_path = self.config.get('OBSERVATIONS_PATH')
        paths = []
        
        if obs_file_path and obs_file_path != 'default':
            paths.append(Path(obs_file_path))
        
        # Standard path
        paths.append(self.project_dir / 'observations' / 'streamflow' / 'preprocessed' / 
                    f"{self.domain_name}_streamflow_processed.csv")
        
        # Alternative paths
        alt_paths = [
            self.project_dir / 'observations' / 'streamflow' / f"{self.domain_name}_streamflow.csv",
            self.project_dir / 'observations' / f"{self.domain_name}_streamflow_processed.csv",
            self.project_dir / 'observations' / f"{self.domain_name}_streamflow.csv",
            self.project_dir / 'observations' / 'streamflow' / 'raw' / f"{self.domain_name}_streamflow.csv"
        ]
        paths.extend(alt_paths)
        
        return paths
    
    def _debug_available_files(self):
        """Debug function to list available observation files"""
        obs_dir = self.project_dir / 'observations'
        if obs_dir.exists():
            self.logger.info(f"Available files in observations directory:")
            for file_path in obs_dir.rglob("*.csv"):
                self.logger.info(f"  {file_path.relative_to(self.project_dir)}")
        else:
            self.logger.error(f"Observations directory does not exist: {obs_dir}")
    
    def _identify_columns(self, df: pd.DataFrame) -> Tuple[str, str]:
        """Identify datetime and discharge columns"""
        datetime_col = None
        discharge_col = None
        
        # Find datetime column
        datetime_candidates = ['datetime', 'date', 'time', 'Date', 'DateTime', 'TIME']
        for col in datetime_candidates:
            if col in df.columns:
                datetime_col = col
                break
        
        if not datetime_col:
            # Look for columns with date-like names
            for col in df.columns:
                if any(term in col.lower() for term in ['date', 'time']):
                    datetime_col = col
                    break
        
        # Find discharge column
        discharge_candidates = ['discharge_cms', 'discharge', 'streamflow', 'flow', 'Q', 'q']
        for col in discharge_candidates:
            if col in df.columns:
                discharge_col = col
                break
        
        if not discharge_col:
            # Look for columns with flow-like names
            for col in df.columns:
                if any(term in col.lower() for term in ['discharge', 'flow', 'q_', 'streamflow']):
                    discharge_col = col
                    break
        
        if not datetime_col:
            raise ValueError(f"Could not identify datetime column. Available columns: {list(df.columns)}")
        
        if not discharge_col:
            raise ValueError(f"Could not identify discharge column. Available columns: {list(df.columns)}")
        
        self.logger.info(f"Using datetime column: '{datetime_col}', discharge column: '{discharge_col}'")
        return datetime_col, discharge_col
    
    def _process_datetime_index(self, df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
        """Process datetime index with robust error handling"""
        try:
            # Use the robust datetime parsing method
            df[datetime_col] = self._parse_datetime_robust(df[datetime_col], datetime_col)
            
            # Set as index
            df.set_index(datetime_col, inplace=True)
            
            # Ensure timezone-naive for compatibility
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing datetime index: {str(e)}")
            raise
    
    def _calculate_performance_metrics(self, simulated_df: pd.Series) -> Dict[str, float]:
        """Calculate streamflow performance metrics"""
        if self.observed_streamflow is None:
            self.logger.error("No streamflow observations available")
            return {}
        
        try:
            # Align time series
            common_index = self.observed_streamflow.index.intersection(simulated_df.index)
            if len(common_index) == 0:
                self.logger.error("No overlapping time period between observations and simulations")
                return {}
            
            obs_aligned = self.observed_streamflow.loc[common_index].dropna()
            sim_aligned = simulated_df.loc[common_index].dropna()
            
            # Further align after dropping NaNs
            final_index = obs_aligned.index.intersection(sim_aligned.index)
            obs_values = obs_aligned.loc[final_index].values
            sim_values = sim_aligned.loc[final_index].values
            
            if len(obs_values) == 0:
                self.logger.error("No valid data points for metric calculation")
                return {}
            
            self.logger.debug(f"Calculating metrics for {len(obs_values)} data points")
            
            # Calculate metrics
            metrics = {
                'KGE': get_KGE(obs_values, sim_values, transfo=1),
                'KGEp': get_KGEp(obs_values, sim_values, transfo=1),
                'NSE': get_NSE(obs_values, sim_values, transfo=1),
                'MAE': get_MAE(obs_values, sim_values, transfo=1),
                'RMSE': get_RMSE(obs_values, sim_values, transfo=1)
            }
            
            # Add calibration period prefix for compatibility
            calib_metrics = {f"Calib_{k}": v for k, v in metrics.items()}
            metrics.update(calib_metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating streamflow metrics: {str(e)}")
            return {}


class FUSESnowTarget:
    """FUSE calibration target for snow variables - FIXED VERSION"""
    
    def __init__(self, config: Dict[str, Any], project_dir: Path, logger: logging.Logger):
        self.config = config
        self.project_dir = project_dir
        self.logger = logger
        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')
        
        # Get observation data
        self._load_observations()
        
        # Get catchment area for unit conversion
        self.catchment_area_km2 = self._get_catchment_area()
    
    def _load_observations(self):
        """Load snow observations with improved error handling"""
        try:
            # Try to load snow observations first
            snow_loaded = self._try_load_snow_observations()
            
            if not snow_loaded:
                self.logger.warning("No snow observations found - using streamflow as proxy")
                self._load_streamflow_observations()
                
        except Exception as e:
            self.logger.error(f"Error loading snow observations: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.observed_swe = None
            # Fallback to streamflow
            self._load_streamflow_observations()
    
    def _try_load_snow_observations(self) -> bool:
        """Try to load snow observations from various sources"""
        # Get snow observation paths
        snow_paths = self._get_possible_snow_paths()
        
        for obs_file_path in snow_paths:
            if obs_file_path and obs_file_path.exists():
                try:
                    self.logger.info(f"Trying to load snow observations from: {obs_file_path}")
                    
                    # Read snow observations
                    dfObs = pd.read_csv(obs_file_path)
                    self.logger.info(f"Snow CSV columns: {list(dfObs.columns)}")
                    
                    # Find datetime and SWE columns
                    datetime_col, swe_col = self._identify_snow_columns(dfObs)
                    
                    if datetime_col and swe_col:
                        # Process datetime index with robust parsing
                        dfObs = self._process_snow_datetime_index(dfObs, datetime_col)
                        
                        # Extract SWE data
                        self.observed_swe = dfObs[swe_col].resample('d').mean()
                        
                        self.logger.info(f"Loaded SWE observations: {len(self.observed_swe)} days")
                        self.logger.info(f"SWE period: {self.observed_swe.index.min()} to {self.observed_swe.index.max()}")
                        return True
                        
                except Exception as e:
                    self.logger.error(f"Error loading snow file {obs_file_path}: {str(e)}")
                    continue
        
        self.observed_swe = None
        return False
    
    def _get_possible_snow_paths(self) -> List[Path]:
        """Get list of possible snow observation file paths"""
        obs_file_path = self.config.get('SNOW_OBSERVATIONS_PATH', 'default')
        paths = []
        
        if obs_file_path and obs_file_path != 'default':
            paths.append(Path(obs_file_path))
        
        # Try different possible paths
        possible_paths = [
            self.project_dir / 'observations' / 'snow' / 'swe' / 'processed' / f"{self.domain_name}_swe_processed.csv",
            self.project_dir / 'observations' / 'snow' / 'preprocessed' / f"{self.domain_name}_snow_processed.csv",
            self.project_dir / 'observations' / 'snow' / f"{self.domain_name}_snow.csv",
            self.project_dir / 'observations' / 'swe' / f"{self.domain_name}_swe.csv",
            self.project_dir / 'observations' / f"{self.domain_name}_swe.csv",
            self.project_dir / 'observations' / f"{self.domain_name}_snow.csv"
        ]
        
        paths.extend(possible_paths)
        return paths
    
    def _identify_snow_columns(self, df: pd.DataFrame) -> Tuple[str, str]:
        """Identify datetime and SWE columns in snow data"""
        datetime_col = None
        swe_col = None
        
        # Find datetime column
        datetime_candidates = ['Date', 'date', 'datetime', 'time', 'DateTime', 'TIME']
        for col in datetime_candidates:
            if col in df.columns:
                datetime_col = col
                break
        
        if not datetime_col:
            # Look for columns with date-like names
            for col in df.columns:
                if any(term in col.lower() for term in ['date', 'time']):
                    datetime_col = col
                    break
        
        # Find SWE column
        swe_candidates = ['swe', 'SWE', 'snow_water_equivalent', 'snow_depth', 'snow']
        for col in df.columns:
            if any(term in col.lower() for term in swe_candidates):
                swe_col = col
                break
        
        self.logger.info(f"Snow data - datetime column: '{datetime_col}', SWE column: '{swe_col}'")
        return datetime_col, swe_col
    
    def _process_snow_datetime_index(self, df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
        """Process datetime index for snow data with robust parsing"""
        try:
            # Use robust datetime parsing
            df[datetime_col] = pd.to_datetime(df[datetime_col], dayfirst=True, errors='coerce')
            
            # Set as index
            df.set_index(datetime_col, inplace=True)
            
            # Ensure timezone-naive for compatibility
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Ensure we have a proper DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.DatetimeIndex(df.index)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing snow datetime index: {str(e)}")
            raise
    
    def _load_streamflow_observations(self):
        """Load streamflow observations as fallback for snow calibration"""
        try:
            # Try multiple possible paths for observations
            obs_paths = [
                self.project_dir / 'observations' / 'streamflow' / 'preprocessed' / 
                f"{self.domain_name}_streamflow_processed.csv",
                self.project_dir / 'observations' / 'streamflow' / f"{self.domain_name}_streamflow.csv",
                self.project_dir / 'observations' / f"{self.domain_name}_streamflow_processed.csv"
            ]
            
            obs_file_path = None
            for path in obs_paths:
                if path.exists():
                    obs_file_path = path
                    break
            
            if obs_file_path:
                dfObs = pd.read_csv(obs_file_path)
                self.logger.info(f"Streamflow CSV columns: {list(dfObs.columns)}")
                
                # Find datetime and discharge columns
                datetime_col = None
                discharge_col = None
                
                for col in ['datetime', 'date', 'time', 'Date', 'DateTime']:
                    if col in dfObs.columns:
                        datetime_col = col
                        break
                
                for col in ['discharge_cms', 'discharge', 'streamflow', 'flow', 'Q']:
                    if col in dfObs.columns:
                        discharge_col = col
                        break
                
                if datetime_col and discharge_col:
                    dfObs[datetime_col] = pd.to_datetime(dfObs[datetime_col], dayfirst=True, errors='coerce')
                    dfObs.set_index(datetime_col, inplace=True)
                    self.observed_streamflow = dfObs[discharge_col].resample('d').mean()
                    self.logger.info(f"Loaded streamflow observations as snow proxy: {len(self.observed_streamflow)} days")
                else:
                    self.logger.error("Could not identify datetime and discharge columns")
                    self.observed_streamflow = None
            else:
                self.logger.error("No streamflow observation files found")
                self.observed_streamflow = None
                
        except Exception as e:
            self.logger.error(f"Error loading streamflow observations: {str(e)}")
            self.observed_streamflow = None
    
    def _get_catchment_area(self) -> float:
        """Get catchment area for unit conversion - simplified version"""
        try:
            # Try to get area from river basins shapefile
            basin_path = self.project_dir / 'shapefiles' / 'river_basins' / f"{self.domain_name}_riverBasins.shp"
            if basin_path.exists():
                gdf = gpd.read_file(basin_path)
                if 'GRU_area' in gdf.columns:
                    total_area_m2 = gdf['GRU_area'].sum()
                    return total_area_m2 / 1e6  # Convert to km2
                else:
                    # Calculate from geometry
                    if gdf.crs and not gdf.crs.is_geographic:
                        total_area_m2 = gdf.geometry.area.sum()
                    else:
                        gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
                        total_area_m2 = gdf_utm.geometry.area.sum()
                    return total_area_m2 / 1e6
            
            # Fallback default
            self.logger.warning("Using default catchment area: 1000 km2")
            return 1000.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating catchment area: {str(e)}")
            return 1000.0
    
    def _needs_unit_conversion(self) -> bool:
        """SWE in FUSE is in mm water equivalent; no area conversion needed"""
        return False


    def calculate_metrics(self, fuse_sim_dir: Path, mizuroute_dir: Optional[Path] = None) -> Optional[Dict[str, float]]:
        try:
            self.logger.debug(f"Starting SWE metrics calculation in: {fuse_sim_dir}")
            
            # Find FUSE simulation file
            sim_file = None
            available_files = []
            
            for pattern in ['runs_def.nc', 'runs_best.nc', 'runs_sce.nc']:
                potential_files = list(fuse_sim_dir.glob(f"*_{pattern}"))
                available_files.extend([f.name for f in potential_files])
                if potential_files:
                    sim_file = potential_files[0]
                    
                    # DEBUG: Check file timestamp
                    import os
                    mtime = os.path.getmtime(sim_file)
                    self.logger.debug(f"Using simulation file: {sim_file.name}, timestamp: {mtime}")
                    break
            
            self.logger.debug(f"Available simulation files: {available_files}")
            
            if not sim_file:
                self.logger.error(f"No simulation files found in {fuse_sim_dir}")
                return None

            # Read FUSE simulation output
            self.logger.debug(f"Opening dataset: {sim_file}")
            with xr.open_dataset(sim_file) as ds:
                self.logger.debug(f"Dataset variables: {list(ds.variables.keys())}")
                
                # FIXED: Better SWE variable detection with fallbacks
                sim_var = None
                
                # Primary candidates for SWE variables
                swe_candidates = [
                    'swe_tot',       # Total SWE
                    'swe_z01',       # SWE in zone 1
                    'swe',           # Generic SWE
                    'SWE',           # Uppercase SWE
                    'snowpack',      # Alternative snow variable
                    'snow_depth',    # Snow depth (could be converted)
                    'snow'           # Generic snow
                ]
                
                # Find the first available SWE variable
                for candidate in swe_candidates:
                    if candidate in ds.variables:
                        sim_var = candidate
                        self.logger.debug(f"Using SWE variable: {sim_var}")
                        break
                
                # If no SWE variable found, check for any snow-related variable
                if sim_var is None:
                    snow_vars = [var for var in ds.variables.keys() 
                                if any(term in var.lower() for term in ['snow', 'swe'])]
                    
                    if snow_vars:
                        sim_var = snow_vars[0]
                        self.logger.warning(f"No standard SWE variable found. Using: {sim_var}")
                        self.logger.warning(f"Available snow-related variables: {snow_vars}")
                    else:
                        self.logger.error("No SWE or snow-related variables found in FUSE output")
                        self.logger.error(f"Available variables: {list(ds.variables.keys())}")
                        return None
                
                # Extract the simulation data
                simulated = ds[sim_var].isel(param_set=0, latitude=0, longitude=0).to_pandas()
                self.logger.debug(f"Successfully extracted {sim_var} data: {len(simulated)} time steps")

            # DEBUG: Log SWE simulation statistics
            self.logger.debug(f"SWE simulation mean: {simulated.mean():.6f} mm")
            self.logger.debug(f"SWE simulation std: {simulated.std():.6f} mm")
            self.logger.debug(f"SWE simulation min: {simulated.min():.6f} mm")
            self.logger.debug(f"SWE simulation max: {simulated.max():.6f} mm")

            # No unit conversion for SWE (already in mm water equivalent)
            return self._calculate_swe_metrics(simulated)

        except Exception as e:
            self.logger.error(f"Error calculating SWE metrics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _calculate_swe_metrics(self, simulated_swe: pd.Series) -> Dict[str, float]:
        """Calculate SWE metrics using observed SWE"""
        if not hasattr(self, 'observed_swe') or self.observed_swe is None:
            self.logger.error("SWE observations not available for snow calibration")
            return {}

        try:
            # Align time series and drop NaNs
            common_index = self.observed_swe.index.intersection(simulated_swe.index)
            if len(common_index) == 0:
                self.logger.error("No overlapping time period between SWE observations and simulations")
                return {}

            obs = self.observed_swe.loc[common_index].dropna()
            sim = simulated_swe.loc[common_index].dropna()
            idx = obs.index.intersection(sim.index)

            if len(idx) == 0:
                self.logger.error("No valid overlapping points after dropping NaNs")
                return {}

            obs_vals = obs.loc[idx].values
            sim_vals = sim.loc[idx].values

            self.logger.info(f"Calculating SWE metrics for {len(obs_vals)} data points")

            # Core metrics
            metrics = {
                'KGE': get_KGE(obs_vals, sim_vals, transfo=1),
                'KGEp': get_KGEp(obs_vals, sim_vals, transfo=1),
                'NSE': get_NSE(obs_vals, sim_vals, transfo=1),
                'MAE': get_MAE(obs_vals, sim_vals, transfo=1),
                'RMSE': get_RMSE(obs_vals, sim_vals, transfo=1),
            }

            # Optional: add seasonal/peak metrics
            try:
                # Peak SWE day difference (in days)
                obs_peak_day = idx[np.argmax(obs_vals)]
                sim_peak_day = idx[np.argmax(sim_vals)]
                metrics['PeakDayDiff_days'] = float((sim_peak_day - obs_peak_day).days)
                
                # Bias (%)
                metrics['Bias_pct'] = float(100.0 * (sim_vals.mean() - obs_vals.mean()) / (obs_vals.mean() + 1e-9))
            except Exception as e:
                self.logger.debug(f"Could not compute extra SWE metrics: {e}")

            # Add Calib_ prefix copies for downstream compatibility
            metrics.update({f"Calib_{k}": v for k, v in metrics.items()})
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in SWE metrics calculation: {str(e)}")
            return {}