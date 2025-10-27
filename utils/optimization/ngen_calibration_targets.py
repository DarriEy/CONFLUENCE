#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NextGen (ngen) Calibration Targets

Provides calibration target classes for ngen model outputs.
Currently supports streamflow calibration with plans for snow, ET, etc.

Author: CONFLUENCE Development Team
Date: 2025
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


class NgenCalibrationTarget:
    """Base class for ngen calibration targets"""
    
    def __init__(self, config: Dict[str, Any], project_dir: Path, logger: logging.Logger):
        """
        Initialize calibration target.
        
        Args:
            config: Configuration dictionary
            project_dir: Path to project directory
            logger: Logger object
        """
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
        """Get catchment area for unit conversion"""
        try:
            # Try river basins shapefile first
            river_basins_path = self.config.get('RIVER_BASINS_PATH', 'default')
            if river_basins_path == 'default':
                river_basins_path = self.project_dir / 'shapefiles' / 'river_basins'
            else:
                river_basins_path = Path(river_basins_path)
            
            river_basins_name = self.config.get('RIVER_BASINS_NAME', 'default')
            if river_basins_name == 'default':
                river_basins_name = f"{self.domain_name}_riverBasins.shp"
            
            shapefile = river_basins_path / river_basins_name
            
            if shapefile.exists():
                gdf = gpd.read_file(shapefile)
                area_col = self.config.get('RIVER_BASINS_AREA_COL', 'GRU_area')
                if area_col in gdf.columns:
                    total_area = gdf[area_col].sum()
                    self.logger.info(f"Catchment area from river basins: {total_area:.2f} km²")
                    return total_area
            
            # Try catchment shapefile as fallback
            catchment_path = self.config.get('CATCHMENT_PATH', 'default')
            if catchment_path == 'default':
                catchment_path = self.project_dir / 'shapefiles' / 'catchment'
            else:
                catchment_path = Path(catchment_path)
            
            catchment_name = self.config.get('CATCHMENT_SHP_NAME', 'default')
            if catchment_name == 'default':
                catchment_name = f"{self.domain_name}_HRUs.shp"
            
            shapefile = catchment_path / catchment_name
            
            if shapefile.exists():
                gdf = gpd.read_file(shapefile)
                # Try different possible area column names
                for area_col in ['HRU_area', 'GRU_area', 'area_km2', 'Area_km2']:
                    if area_col in gdf.columns:
                        total_area = gdf[area_col].sum()
                        self.logger.info(f"Catchment area from HRUs: {total_area:.2f} km²")
                        return total_area
            
            # If no shapefile available, use default or config value
            default_area = self.config.get('CATCHMENT_AREA_KM2', 100.0)
            self.logger.warning(f"Could not find area from shapefiles, using default: {default_area} km²")
            return default_area
            
        except Exception as e:
            self.logger.error(f"Error getting catchment area: {e}")
            return 100.0  # Default fallback
    
    def calculate_metrics(self, experiment_id: str = None) -> Dict[str, float]:
        """Calculate performance metrics (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement calculate_metrics()")


class NgenStreamflowTarget(NgenCalibrationTarget):
    """Calibration target for streamflow from ngen nexus outputs"""
    
    def __init__(self, config: Dict[str, Any], project_dir: Path, logger: logging.Logger):
        """
        Initialize streamflow calibration target.
        
        Args:
            config: Configuration dictionary
            project_dir: Path to project directory
            logger: Logger object
        """
        super().__init__(config, project_dir, logger)
        
        # Streamflow-specific settings
        self.station_id = config.get('STATION_ID', None)
        self.calibration_period = self._parse_calibration_period()
        
    def _load_observations(self):
        """Load observed streamflow data"""
        obs_file = self.config.get('OBSERVED_STREAMFLOW_FILE', None)
        
        if obs_file is None or obs_file == 'default':
            obs_file = self.project_dir / 'observations' / 'streamflow' / 'preprocessed' / f"{self.domain_name}_streamflow_processed.csv"
        else:
            obs_file = Path(obs_file)
        
        if not obs_file.exists():
            self.logger.warning(f"Observed streamflow file not found: {obs_file}")
            self.obs_data = None
            return
        
        try:
            # Read observed data
            self.obs_data = pd.read_csv(obs_file)
            
            # Standardize column names
            date_col = None
            flow_col = None
            
            for col in self.obs_data.columns:
                col_lower = col.lower()
                if col_lower in ['date', 'datetime', 'time']:
                    date_col = col
                elif col_lower in ['discharge', 'streamflow', 'flow', 'q', 'q_obs']:
                    flow_col = col
            
            if date_col is None or flow_col is None:
                self.logger.error("Could not identify date and flow columns in observed data")
                self.obs_data = None
                return
            
            # Rename columns to standard names
            self.obs_data = self.obs_data.rename(columns={
                date_col: 'datetime',
                flow_col: 'streamflow_cms'
            })
            
            # Convert datetime
            self.obs_data['datetime'] = pd.to_datetime(self.obs_data['datetime'])
            
            # Remove NaN values
            self.obs_data = self.obs_data.dropna(subset=['streamflow_cms'])
            
            self.logger.info(f"Loaded {len(self.obs_data)} observed streamflow records")
            self.logger.info(f"Observation period: {self.obs_data['datetime'].min()} to {self.obs_data['datetime'].max()}")
            
        except Exception as e:
            self.logger.error(f"Error loading observed streamflow: {e}")
            self.obs_data = None
    
    def _parse_calibration_period(self) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Parse calibration period from config"""
        start_str = self.config.get('CALIBRATION_PERIOD_START', None)
        end_str = self.config.get('CALIBRATION_PERIOD_END', None)
        
        start_date = pd.to_datetime(start_str) if start_str else None
        end_date = pd.to_datetime(end_str) if end_str else None
        
        return start_date, end_date
    
    def _load_simulated_streamflow(self, experiment_id: str) -> Optional[pd.DataFrame]:
        """
        Load simulated streamflow from ngen nexus outputs.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            DataFrame with simulated streamflow or None
        """
        # Look for nexus output files
        output_dir = self.project_dir / 'simulations' / experiment_id / 'ngen'
        
        if not output_dir.exists():
            self.logger.error(f"Simulation output directory not found: {output_dir}")
            return None
        
        # Find nexus output files
        nexus_files = list(output_dir.glob('nex-*_output.csv'))
        
        if not nexus_files:
            self.logger.error(f"No nexus output files found in {output_dir}")
            return None
        
        # Read all nexus outputs and combine
        all_streamflow = []
        
        for nexus_file in nexus_files:
            try:
                df = pd.read_csv(nexus_file)
                
                # Find flow column
                flow_col = None
                for col_name in ['flow', 'Flow', 'Q_OUT', 'streamflow', 'discharge']:
                    if col_name in df.columns:
                        flow_col = col_name
                        break
                
                if flow_col is None:
                    continue
                
                # Find time column
                time_col = None
                for col_name in ['Time', 'time', 'datetime', 'date']:
                    if col_name in df.columns:
                        time_col = col_name
                        break
                
                if time_col is None:
                    continue
                
                # Extract data
                nexus_id = nexus_file.stem.replace('_output', '')
                streamflow_df = pd.DataFrame({
                    'datetime': pd.to_datetime(df[time_col]) if 'Time' in df.columns else pd.to_datetime(df[time_col]),
                    'streamflow_cms': df[flow_col],
                    'nexus_id': nexus_id
                })
                
                all_streamflow.append(streamflow_df)
                
            except Exception as e:
                self.logger.error(f"Error reading {nexus_file}: {e}")
                continue
        
        if not all_streamflow:
            return None
        
        # Combine all nexus outputs (sum for total catchment outflow)
        combined = pd.concat(all_streamflow, ignore_index=True)
        
        # Group by time and sum flows (if multiple nexuses)
        combined = combined.groupby('datetime')['streamflow_cms'].sum().reset_index()
        
        return combined
    
    def calculate_metrics(self, experiment_id: str = None) -> Dict[str, float]:
        """
        Calculate streamflow performance metrics.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Dictionary of performance metrics
        """
        if experiment_id is None:
            experiment_id = self.experiment_id
        
        if self.obs_data is None:
            self.logger.error("No observed data available for metric calculation")
            return {'KGE': -999.0, 'NSE': -999.0, 'MAE': 999.0, 'RMSE': 999.0}
        
        # Load simulated data
        sim_data = self._load_simulated_streamflow(experiment_id)
        
        if sim_data is None:
            self.logger.error("No simulated data available for metric calculation")
            return {'KGE': -999.0, 'NSE': -999.0, 'MAE': 999.0, 'RMSE': 999.0}
        
        # Merge observed and simulated on datetime
        merged = pd.merge(self.obs_data, sim_data, on='datetime', suffixes=('_obs', '_sim'))
        
        # Apply calibration period filter if specified
        start_date, end_date = self.calibration_period
        if start_date is not None:
            merged = merged[merged['datetime'] >= start_date]
        if end_date is not None:
            merged = merged[merged['datetime'] <= end_date]
        
        if len(merged) == 0:
            self.logger.error("No overlapping data between observed and simulated")
            return {'KGE': -999.0, 'NSE': -999.0, 'MAE': 999.0, 'RMSE': 999.0}
        
        # Remove any remaining NaN values
        merged = merged.dropna(subset=['streamflow_cms_obs', 'streamflow_cms_sim'])
        
        if len(merged) < 10:
            self.logger.warning(f"Only {len(merged)} overlapping timesteps for metric calculation")
            return {'KGE': -999.0, 'NSE': -999.0, 'MAE': 999.0, 'RMSE': 999.0}
        
        # Calculate metrics
        obs = merged['streamflow_cms_obs'].values
        sim = merged['streamflow_cms_sim'].values
        
        try:
            kge = get_KGE(obs, sim)
            kgep = get_KGEp(obs, sim)
            nse = get_NSE(obs, sim)
            mae = get_MAE(obs, sim)
            rmse = get_RMSE(obs, sim)
            
            metrics = {
                'KGE': float(kge),
                'KGEp': float(kgep),
                'NSE': float(nse),
                'MAE': float(mae),
                'RMSE': float(rmse),
                'n_points': len(merged)
            }
            
            self.logger.info(f"Calculated metrics: KGE={kge:.3f}, NSE={nse:.3f}, n={len(merged)}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {'KGE': -999.0, 'NSE': -999.0, 'MAE': 999.0, 'RMSE': 999.0}
