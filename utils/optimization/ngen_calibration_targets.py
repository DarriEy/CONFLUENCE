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
        self.obs_data = self._load_observations()
        
        # Get catchment area for unit conversion
        self.catchment_area_km2 = self._get_catchment_area()
    
    def _load_observations(self):
        """Load observation data (to be implemented by subclasses)"""
        return None
    
    def _get_catchment_area(self) -> float:
        """
        Return catchment area in km².
        Priority/order:
        1) config['catchment']['area_km2']
        2) sum of area from a configured catchment shapefile path
        3) auto-discover *HRUs_GRUs.shp under domain/<domain>/shapefiles/catchment/
        We project to an equal-area CRS if needed before computing geometry areas.
        """
        from pathlib import Path
        import geopandas as gpd
        import numpy as np

        # direct override in config
        cfg_area = (self.config.get("catchment", {}) or {}).get("area_km2")
        if cfg_area:
            self.logger.info("Using catchment area from config: %.3f km²", float(cfg_area))
            return float(cfg_area)

        # project_dir is already the domain directory, use it directly
        domain_dir = self.project_dir
        if "paths" in self.config and "domain_dir" in self.config["paths"]:
            domain_dir = Path(self.config["paths"]["domain_dir"])

        # explicit shapefile path in config?
        shp_cfg = (self.config.get("shapefiles", {}) or {}).get("catchment")
        candidates = []
        if shp_cfg:
            candidates.append(Path(shp_cfg))

        # convention used in your logs
        shp_dir = domain_dir / "shapefiles" / "catchment"
        if shp_dir.exists():
            # prefer HRUs_GRUs style files first
            candidates += sorted(shp_dir.glob("*HRUs_GRUs.shp"))
            # otherwise, any polygon shapefile
            candidates += [p for p in sorted(shp_dir.glob("*.shp")) if p not in candidates]

        shp_path = next((p for p in candidates if p.exists()), None)
        if shp_path is None:
            self.logger.warning(
                "Could not find catchment shapefile. Looked for: %s",
                [str(p) for p in candidates] or [str(shp_dir / '<*.shp>')]
            )
            self.logger.warning("Using default catchment area: 100.0 km²")
            return 100.0

        self.logger.info("Reading catchment shapefile: %s", shp_path)
        gdf = gpd.read_file(shp_path)

        # If there's a precomputed area column, use it (common: 'area', 'AREA', 'Area_m2', etc.)
        area_cols = [c for c in gdf.columns if c.lower() in ("area", "area_m2", "areasqkm", "area_km2")]
        if area_cols:
            col = area_cols[0]
            vals = gdf[col].astype(float)
            # detect unit by magnitude; prefer km² columns if indicated
            if col.lower() in ("areasqkm", "area_km2"):
                area_km2 = float(np.nansum(vals))
            else:
                # assume m² if not labelled as km²
                area_km2 = float(np.nansum(vals)) / 1e6
            self.logger.info("Catchment area from attribute '%s': %.3f km²", col, area_km2)
            return area_km2

        # Otherwise compute from geometry; reproject if necessary to equal-area
        if gdf.crs is None or gdf.crs.is_geographic:
            # CONUS equal-area; switch if your domain is elsewhere
            equal_area = "EPSG:5070"
            self.logger.info("Projecting geometries to %s for area calculation", equal_area)
            gdf = gdf.to_crs(equal_area)

        area_m2 = gdf.geometry.area.sum()
        area_km2 = float(area_m2 / 1e6)
        self.logger.info("Catchment area from geometry: %.3f km²", area_km2)
        return area_km2

    
    def calculate_metrics(self, experiment_id: str = None, output_dir: Optional[Path] = None) -> Dict[str, float]:
        """Calculate performance metrics (to be implemented by subclasses)
        
        Args:
            experiment_id: Experiment identifier
            output_dir: Optional output directory (for parallel mode)
        """
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
            
    def _load_observations(self) -> pd.DataFrame:
        """
        Load observed streamflow and return a DataFrame with datetime and streamflow_cms columns.
        Priority:
        1) config['observations']['streamflow']['path']
        2) domain_{domain}/observations/streamflow/*_streamflow_obs.csv
        3) any .csv under observations/streamflow/
        The loader tries common column names and units (cfs/cms).
        
        Returns:
            DataFrame with columns ['datetime', 'streamflow_cms']
        """
        import re
        from pathlib import Path

        # ---- resolve domain + base dirs ----
        # project_dir is already the full domain directory (e.g., domain_USA_01073000_headwater)
        # so we should use it directly
        domain_dir = None
        # Allow explicit project/domain override
        if "paths" in self.config and "domain_dir" in self.config["paths"]:
            domain_dir = Path(self.config["paths"]["domain_dir"])
        else:
            # Use project_dir directly - it's already the domain directory
            domain_dir = self.project_dir

        obs_override = (
            self.config.get("observations", {})
                    .get("streamflow", {})
                    .get("path")
        )

        # ---- candidate paths ----
        candidates = []
        if obs_override:
            candidates.append(Path(obs_override))

        obs_dir = domain_dir / "observations" / "streamflow" / 'preprocessed'
        if obs_dir.exists():
            # prefer *_streamflow_obs.csv first (your convention), then any csv
            candidates += sorted(obs_dir.glob("*_streamflow_processed.csv"))
            candidates += [p for p in sorted(obs_dir.glob("*.csv"))
                        if p not in candidates]

        # ---- pick the first that exists ----
        obs_path = next((p for p in candidates if p.exists()), None)
        if obs_path is None:
            self.logger.warning(
                "Observed streamflow file not found. Looked for: %s",
                [str(p) for p in candidates] or [str(obs_dir / "<*.csv>")]
            )
            # keep behavior consistent with previous code: default to empty series
            return pd.Series(dtype="float64", name="obs")

        self.logger.info("Loading observed streamflow: %s", obs_path)

        # ---- read CSV with flexible parsing ----
        # Try common datetime column names
        dt_cols_try = ["time", "datetime", "date_time", "DateTime", "date", "timestamp"]
        df = pd.read_csv(obs_path)

        # find datetime column
        dt_col = next((c for c in df.columns if c in dt_cols_try), None)
        if dt_col is None:
            # try fuzzy match
            dt_col = next((c for c in df.columns if re.search("date|time", c, re.I)), None)
        if dt_col is None:
            raise ValueError(
                f"Could not find a datetime column in {obs_path}. "
                f"Tried: {dt_cols_try}"
            )

        # find value column (cms/cfs/flow/value/discharge)
        val_cols_try = [
            "discharge_cms", "flow_cms", "Q_cms", "cms",
            "discharge_cfs", "flow_cfs", "Q_cfs", "cfs",
            "discharge", "flow", "value", "Q"
        ]
        val_col = next((c for c in df.columns if c in val_cols_try), None)
        if val_col is None:
            # heuristic: pick the first non-datetime numeric column
            numeric_cols = [c for c in df.columns if c != dt_col and pd.api.types.is_numeric_dtype(df[c])]
            if not numeric_cols:
                raise ValueError(f"No numeric streamflow column found in {obs_path}.")
            val_col = numeric_cols[0]

        # parse times -> UTC index
        ts = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
        s = pd.Series(df[val_col].astype(float).values, index=ts, name="obs").sort_index()
        s = s[~s.index.duplicated(keep="first")].dropna()

        # ---- unit handling (convert to m³/s) ----
        col_lower = val_col.lower()
        if "cfs" in col_lower:
            s = s * 0.028316846592  # ft³/s -> m³/s
        # if cms or otherwise, assume already m³/s

        # ---- align to model window if available ----
        start = (self.config.get("forcing", {}) or {}).get("start_time")
        end   = (self.config.get("forcing", {}) or {}).get("end_time")
        if start:
            start = pd.to_datetime(start, utc=True)
        if end:
            end = pd.to_datetime(end, utc=True)

        if start or end:
            s = s.loc[slice(start or s.index.min(), end or s.index.max())]

        # optional resample to model timestep
        dt_min = (self.config.get("model", {}) or {}).get("timestep_minutes") \
                or (self.config.get("ngen", {}) or {}).get("timestep_minutes")
        if dt_min:
            s = s.resample(f"{int(dt_min)}min").mean().interpolate(limit=2)

        # Convert Series to DataFrame with proper columns for merging
        obs_df = s.reset_index()
        obs_df.columns = ['datetime', 'streamflow_cms']
        
        return obs_df

    
    def _parse_calibration_period(self) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Parse calibration period from config"""
        start_str = self.config.get('CALIBRATION_PERIOD_START', None)
        end_str = self.config.get('CALIBRATION_PERIOD_END', None)
        
        start_date = pd.to_datetime(start_str) if start_str else None
        end_date = pd.to_datetime(end_str) if end_str else None
        
        return start_date, end_date
    
    def _load_simulated_streamflow(self, experiment_id: str, output_dir: Optional[Path] = None) -> Optional[pd.DataFrame]:
        """
        Load simulated streamflow from ngen nexus outputs.
        
        Args:
            experiment_id: Experiment identifier
            output_dir: Optional output directory (for parallel mode). If provided,
                       uses this directory instead of the default path.
            
        Returns:
            DataFrame with simulated streamflow or None
        """
        # Use provided output_dir (parallel mode) or default path (serial mode)
        if output_dir is not None:
            # Parallel mode - use provided directory
            sim_dir = output_dir
        else:
            # Serial mode - use standard path
            sim_dir = self.project_dir / 'simulations' / experiment_id / 'ngen'
        
        if not sim_dir.exists():
            self.logger.error(f"Simulation output directory not found: {sim_dir}")
            return None
        
        # Find nexus output files
        nexus_files = list(sim_dir.glob('nex-*_output.csv'))
        
        if not nexus_files:
            self.logger.error(f"No nexus output files found in {sim_dir}")
            return None
        
        # Read all nexus outputs and combine
        all_streamflow = []
        
        for nexus_file in nexus_files:
            try:
                # Read CSV - ngen output has no header, format is: index, datetime, flow
                df = pd.read_csv(nexus_file, header=None, names=['index', 'datetime', 'flow'])
                
                # Check if we got data
                if df.empty:
                    self.logger.warning(f"Empty file: {nexus_file}")
                    continue
                
                # Extract data - convert datetime to UTC to match observations
                nexus_id = nexus_file.stem.replace('_output', '')
                streamflow_df = pd.DataFrame({
                    'datetime': pd.to_datetime(df['datetime'], utc=True),
                    'streamflow_cms': df['flow'],
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
    
    def calculate_metrics(self, experiment_id: str = None, output_dir: Optional[Path] = None) -> Dict[str, float]:
        """
        Calculate streamflow performance metrics.
        
        Args:
            experiment_id: Experiment identifier
            output_dir: Optional output directory (for parallel mode). If provided,
                       uses this directory to find ngen outputs instead of default path.
            
        Returns:
            Dictionary of performance metrics
        """
        if experiment_id is None:
            experiment_id = self.experiment_id
        
        if self.obs_data is None:
            self.logger.error("No observed data available for metric calculation")
            return {'KGE': -999.0, 'NSE': -999.0, 'MAE': 999.0, 'RMSE': 999.0}
        
        # Load simulated data with optional output_dir for parallel mode
        sim_data = self._load_simulated_streamflow(experiment_id, output_dir=output_dir)
        
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