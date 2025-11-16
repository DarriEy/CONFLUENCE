"""
Cloud Data Utilities for SYMFLUENCE
====================================

Direct access to cloud-hosted forcing datasets via S3/Zarr/GCS
without requiring intermediate file downloads or hydrofabric preprocessing.

Activated via: DATA_ACCESS: cloud

Supported datasets:
- AORC: Analysis of Record for Calibration (CONUS, 1km, hourly, 1979-present)
  Storage: AWS S3 (s3://noaa-nws-aorc-v1-1-1km) - Zarr format
  
- ERA5: ECMWF Reanalysis (Global, 31km, hourly, 1940-present)
  Storage: Google Cloud (gs://gcp-public-data-arco-era5) - ARCO Zarr format
  
- EM-Earth: Ensemble Meteorological Dataset (Global, 11km, daily, 1950-2019)
  Storage: AWS S3 (s3://emearth/) - NetCDF format
  Activated via: SUPPLEMENT_FORCING: true
  
- HRRR: High-Resolution Rapid Refresh (CONUS, 3km, hourly+subhourly, 2014-present)
  Storage: AWS S3 (s3://hrrrzarr/) - Zarr format
  
- CONUS404: WRF Reanalysis (CONUS+, 4km, hourly, 1979-2022)
  Storage: AWS S3 (s3://hytest/conus404/) - Zarr format via HyTEST

Requirements:
- s3fs: For AWS S3 access (AORC, EM-Earth, HRRR, CONUS404)
- gcsfs: For Google Cloud Storage access (ERA5)
- intake-xarray: For CONUS404 catalog access (optional but recommended)

Author: SYMFLUENCE Development Team
Date: 2025-01-14
"""

import xarray as xr
import s3fs
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
import re
import math
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.windows import from_bounds
import requests
import datetime as dt
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import logging


def _process_era5_chunk(
    chunk_info: Tuple[int, Tuple, str, Dict, Path, List[str], int, float, float, float, float]
) -> Tuple[int, Optional[Path], str]:
    """
    Process a single ERA5 chunk for parallel execution.
    
    This function is designed to be called by ProcessPoolExecutor workers.
    
    Parameters
    ----------
    chunk_info : tuple
        Contains (chunk_idx, chunk_times, zarr_store, config_subset, output_dir, 
                 available_vars, step, lat_min_raw, lat_max_raw, lon_min, lon_max)
    
    Returns
    -------
    tuple
        (chunk_idx, output_file_path or None, status_message)
    """
    try:
        import xarray as xr
        import pandas as pd
        import numpy as np
        import gcsfs
        from pathlib import Path
        
        # Unpack chunk information
        (chunk_idx, (chunk_start, chunk_end), zarr_store, config_subset, 
         output_dir, available_vars, step, lat_min_raw, lat_max_raw, 
         lon_min, lon_max) = chunk_info
        
        total_chunks = config_subset.get('total_chunks', 1)
        domain_name = config_subset.get('DOMAIN_NAME', 'domain')
        
        # Open the Zarr store (each worker opens independently)
        gcs = gcsfs.GCSFileSystem(token="anon")
        mapper = gcs.get_mapper(zarr_store)
        ds = xr.open_zarr(mapper, consolidated=True, chunks={})
        
        # Load coordinates
        ds = ds.assign_coords(
            longitude=ds.longitude.load(),
            latitude=ds.latitude.load(),
            time=ds.time.load(),
        )
        
        # 1) Subset in time
        ds_t = ds.sel(time=slice(chunk_start, chunk_end))
        
        if "time" not in ds_t.dims or ds_t.sizes["time"] < 2:
            return (chunk_idx, None, f"Chunk {chunk_idx}/{total_chunks}: Fewer than 2 timesteps, skipped")
        
        # 2) Subset in space
        ds_ts = ds_t.sel(
            latitude=slice(lat_max_raw, lat_min_raw),
            longitude=slice(lon_min, lon_max),
        )
        
        # 3) Apply temporal thinning
        if step > 1 and "time" in ds_ts.dims:
            ds_ts = ds_ts.isel(time=slice(0, None, step))
        
        if "time" not in ds_ts.dims or ds_ts.sizes["time"] < 2:
            return (chunk_idx, None, f"Chunk {chunk_idx}/{total_chunks}: Fewer than 2 timesteps after thinning, skipped")
        
        # 4) Select variables
        chunk_vars = [v for v in available_vars if v in ds_ts.data_vars]
        if not chunk_vars:
            return (chunk_idx, None, f"Chunk {chunk_idx}/{total_chunks}: No variables available, skipped")
        
        ds_chunk_raw = ds_ts[chunk_vars]
        
        # 5) Convert to SUMMA schema
        ds_chunk = _era5_to_summa_schema_standalone(ds_chunk_raw)
        
        if "time" not in ds_chunk.dims or ds_chunk.sizes["time"] < 1:
            return (chunk_idx, None, f"Chunk {chunk_idx}/{total_chunks}: No timesteps after SUMMA conversion, skipped")
        
        # Build filename
        file_year = chunk_start.year
        file_month = chunk_start.month
        chunk_file = output_dir / f"domain_{domain_name}_ERA5_merged_{file_year}{file_month:02d}.nc"
        
        # Set up encoding
        encoding = {}
        for var in ds_chunk.data_vars:
            encoding[var] = {
                "zlib": True,
                "complevel": 1,
                "chunksizes": (
                    min(168, ds_chunk.sizes["time"]),
                    ds_chunk.sizes["latitude"],
                    ds_chunk.sizes["longitude"],
                ),
            }
        
        # Write to NetCDF
        ds_chunk.to_netcdf(chunk_file, encoding=encoding, compute=True)
        
        return (chunk_idx, chunk_file, f"Chunk {chunk_idx}/{total_chunks}: Success")
        
    except Exception as e:
        return (chunk_idx, None, f"Chunk {chunk_idx}/{total_chunks}: Error - {str(e)}")


def _era5_to_summa_schema_standalone(ds_chunk):
    """
    Standalone version of ERA5 to SUMMA schema conversion for use in parallel workers.
    This duplicates the logic from CloudForcingDownloader._era5_to_summa_schema
    but doesn't require class instance.
    """
    import xarray as xr
    import numpy as np
    
    if "time" not in ds_chunk.dims or ds_chunk.sizes["time"] < 2:
        return ds_chunk
    
    # Ensure time is sorted
    ds_chunk = ds_chunk.sortby("time")
    
    # Drop first time step for finite differences
    ds_base = ds_chunk.isel(time=slice(1, None))
    
    # Preserve coords
    out = xr.Dataset(coords={c: ds_base.coords[c] for c in ds_base.coords})
    
    # Simple renames / direct copies
    if "surface_pressure" in ds_base:
        airpres = ds_base["surface_pressure"].astype("float32")
        airpres.name = "airpres"
        airpres.attrs.update({
            "units": "Pa",
            "long_name": "air pressure",
            "standard_name": "air_pressure",
        })
        out["airpres"] = airpres
    
    if "2m_temperature" in ds_base:
        airtemp = ds_base["2m_temperature"].astype("float32")
        airtemp.name = "airtemp"
        airtemp.attrs.update({
            "units": "K",
            "long_name": "air temperature",
            "standard_name": "air_temperature",
        })
        out["airtemp"] = airtemp
    
    # Wind speed from U/V components
    if "10m_u_component_of_wind" in ds_base and "10m_v_component_of_wind" in ds_base:
        u = ds_base["10m_u_component_of_wind"]
        v = ds_base["10m_v_component_of_wind"]
        windspd = np.sqrt(u ** 2 + v ** 2).astype("float32")
        windspd.name = "windspd"
        windspd.attrs.update({
            "units": "m s-1",
            "long_name": "wind speed",
            "standard_name": "wind_speed",
        })
        out["windspd"] = windspd
    
    # Specific humidity from dew point + surface pressure
    if "2m_dewpoint_temperature" in ds_base and "surface_pressure" in ds_base:
        Td = ds_base["2m_dewpoint_temperature"]
        p = ds_base["surface_pressure"]
        
        # Convert to Celsius for Magnus formula
        Td_C = Td - 273.15
        
        # Saturation vapor pressure (Magnus, over water)
        es_hPa = 6.112 * np.exp((17.67 * Td_C) / (Td_C + 243.5))
        es = es_hPa * 100.0
        
        eps = 0.622
        denom = xr.where((p - es) <= 1.0, 1.0, p - es)
        r = eps * es / denom
        q = (r / (1.0 + r)).astype("float32")
        
        spechum = q
        spechum.name = "spechum"
        spechum.attrs.update({
            "units": "kg kg-1",
            "long_name": "specific humidity",
            "standard_name": "specific_humidity",
        })
        out["spechum"] = spechum
    
    # Helper for accumulated → rate conversion
    time = ds_chunk["time"]
    dt = (time.diff("time") / np.timedelta64(1, "s")).astype("float32")
    
    def _accum_to_rate(var_name, out_name, units, long_name, standard_name):
        if var_name not in ds_chunk:
            return
        
        accum = ds_chunk[var_name]
        diff = accum.diff("time")
        rate = (diff / dt).astype("float32")
        rate.name = out_name
        rate.attrs.update({
            "units": units,
            "long_name": long_name,
            "standard_name": standard_name,
        })
        out[out_name] = rate
    
    # Precipitation: m -> m s-1
    _accum_to_rate(
        "total_precipitation",
        "pptrate",
        "m s-1",
        "precipitation rate",
        "precipitation_rate",
    )
    
    # Shortwave radiation: J m-2 -> W m-2
    _accum_to_rate(
        "surface_solar_radiation_downwards",
        "SWRadAtm",
        "W m-2",
        "surface downwelling shortwave radiation",
        "surface_downwelling_shortwave_flux_in_air",
    )
    
    # Longwave radiation: J m-2 -> W m-2
    _accum_to_rate(
        "surface_thermal_radiation_downwards",
        "LWRadAtm",
        "W m-2",
        "surface downwelling longwave radiation",
        "surface_downwelling_longwave_flux_in_air",
    )
    
    return out




class CloudForcingDownloader:
    """
    Download forcing data directly from cloud storage (AWS S3, Google Cloud Storage).
    
    This class provides methods to access cloud-optimized forcing datasets
    without requiring local file downloads or hydrofabric preprocessing.
    
    Supported storage:
    - AWS S3: AORC (s3fs)
    - Google Cloud Storage: ERA5 ARCO (gcsfs)
    """
    
    def __init__(self, config: Dict, logger):
        """
        Initialize the CloudForcingDownloader.
        
        Parameters
        ----------
        config : dict
            SYMFLUENCE configuration dictionary
        logger : logging.Logger
            Logger instance for tracking progress
        """
        self.config = config
        self.logger = logger
        
        # Parse bounding box
        self.bbox = self._parse_bbox(config['BOUNDING_BOX_COORDS'])
        
        # Parse time period
        self.start_date = pd.to_datetime(config['EXPERIMENT_TIME_START'])
        self.end_date = pd.to_datetime(config['EXPERIMENT_TIME_END'])
        
        # Get forcing dataset
        self.dataset_name = config.get('FORCING_DATASET', '').upper()

        self.supplement_data = config.get('SUPPLEMENT_FORCING', False)
        
        # Initialize S3 filesystem
        self.fs = s3fs.S3FileSystem(anon=True)
        
    def _parse_bbox(self, bbox_string: str) -> Dict[str, float]:
        """
        Parse bounding box string into dictionary.
        
        Parameters
        ----------
        bbox_string : str
            Bounding box in format 'lat_max/lon_max/lat_min/lon_min'
            
        Returns
        -------
        dict
            Dictionary with keys: lat_min, lat_max, lon_min, lon_max
        """
        coords = bbox_string.split('/')
        return {
            'lat_min': float(coords[2]),
            'lat_max': float(coords[0]),
            'lon_min': float(coords[3]),
            'lon_max': float(coords[1])
        }
    
    @property
    def domain_dir(self) -> Path:
        """
        Base directory for the current SYMFLUENCE domain, e.g.
        SYMFLUENCE_DATA_DIR / f\"domain_{DOMAIN_NAME}\".
        """
        base = Path(self.config["SYMFLUENCE_DATA_DIR"])
        domain_name = self.config.get("DOMAIN_NAME", "domain")
        d = base / f"domain_{domain_name}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _attribute_dir(self, subdir: str) -> Path:
        """
        Helper to create attribute subdirs such as:
          - attributes/elevation
          - attributes/soilclass
          - attributes/landclass
        """
        d = self.domain_dir / "attributes" / subdir
        d.mkdir(parents=True, exist_ok=True)
        return d

    def download_forcing_data(self, output_dir: Path) -> Path:
        """
        Download forcing data based on configured dataset.

        Parameters
        ----------
        output_dir : Path
            Directory to save downloaded forcing data

        Returns
        -------
        Path
            Path to the downloaded forcing file or directory

        Raises
        ------
        ValueError
            If dataset is not supported for cloud access
        """
        self.logger.info(f"Starting cloud data download for {self.dataset_name}")

        # Check to see if we need supplementary forcing data and download if we do
        if self.supplement_data:
            self.logger.info('Supplementing data, dowloading EM-Earth')
            self._download_emearth(output_dir)
        else:
            self.logger.info('No supplementary data needed')

        # Download selected forcing datset
        if self.dataset_name == 'NEX-GDDP-CMIP6':
            return self._download_nex_gddp_cmip6_all(output_dir)
        elif self.dataset_name == 'GLDAS':
            return self._download_gldas(output_dir)
        elif self.dataset_name == 'AORC':
            return self._download_aorc(output_dir)
        elif self.dataset_name == 'ERA5':
            return self._download_era5(output_dir)
        elif self.dataset_name == 'HRRR':
            return self._download_hrrr(output_dir)
        elif self.dataset_name == 'CONUS404':
            return self._download_conus404(output_dir)
        else:
            raise ValueError(
                f"Dataset '{self.dataset_name}' is not supported for cloud access. "
                f"Supported datasets: NEX-GDDP-CMIP6, GLDAS, AORC, ERA5, EM-EARTH, HRRR, CONUS404"
            )

    def _download_gldas(self, output_dir: Path) -> Path:
        """
        Placeholder for GLDAS cloud access.

        We currently *do not* support anonymous GLDAS downloads from GES DISC
        in this cloud-data path. GES DISC now serves GLDAS through an
        authenticated HTML / OPeNDAP interface rather than direct, anonymous
        NetCDF URLs, so naive xr.open_dataset(<https URL>) will return HTML and
        fail with syntax errors.

        For now, use:
          - DATA_ACCESS: local  with pre-downloaded GLDAS files, or
          - another cloud-hosted forcing dataset (AORC, ERA5, CONUS404,
            NEX-GDDP-CMIP6, etc.)
        """
        # --- derive time_slice for consistency with other downloaders ---
        time_slice = self.config.get("time_slice", None)

        if time_slice is not None:
            start_date_str, end_date_str = time_slice
        else:
            exp_start = self.config.get("EXPERIMENT_TIME_START")
            exp_end   = self.config.get("EXPERIMENT_TIME_END")

            if exp_start is None or exp_end is None:
                raise ValueError(
                    "GLDAS: neither 'time_slice' nor EXPERIMENT_TIME_START/END "
                    "are set in the config."
                )

            start_date = pd.to_datetime(exp_start)
            end_date   = pd.to_datetime(exp_end)
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str   = end_date.strftime("%Y-%m-%d")

            self.logger.info(
                f"GLDAS: derived time_slice from experiment window: "
                f"{start_date_str} to {end_date_str}"
            )

        # At this point wiring is correct, but we deliberately stop: no stable
        # anonymous GLDAS cloud endpoint is available for generic xarray access.
        msg = (
            "GLDAS cloud access is not implemented in SYMFLUENCE.\n"
            "NASA GES DISC serves GLDAS via authenticated HTTPS/OPeNDAP and via "
            "platforms like Google Earth Engine, not an anonymous S3 bucket.\n"
            "Use DATA_ACCESS: local with pre-downloaded GLDAS files or switch "
            "to another cloud-hosted forcing dataset."
        )
        self.logger.error(msg)
        raise NotImplementedError(msg)

    def _download_nex_gddp_cmip6_all(self, output_dir: Path) -> Path:
        """
        Download (a subset of) models / scenarios / variables for NEX-GDDP-CMIP6
        using the NCCS THREDDS NetCDF Subset Service (NCSS), subsetting to the
        experiment time window and bounding box.

        This version:
          * restricts to years that overlap the requested time_slice,
          * for `historical` scenario, clips years to <= 2014 (NEX-GDDP-CMIP6 coverage),
          * calls NCSS with server-side time + spatial subset (per year, per variable),
          * for each (model, scenario, member) builds a combined dataset over time,
          * stacks all (model, scenario, member) combos into an `ensemble` dimension
            with coordinates `model(ensemble)`, `scenario(ensemble)`, `member(ensemble)`,
          * writes ONE monthly NetCDF4 file per month containing ALL variables and
            ALL ensembles,
          * cleans up the NCSS cache.
        """

        import datetime as dt
        import shutil
        from typing import List

        import pandas as pd
        import xarray as xr
        import requests

        # --- time window ---
        cfg_time_slice = self.config.get("time_slice", None)
        if cfg_time_slice is not None:
            start_date_str, end_date_str = cfg_time_slice
        else:
            exp_start = pd.to_datetime(self.config["EXPERIMENT_TIME_START"])
            exp_end   = pd.to_datetime(self.config["EXPERIMENT_TIME_END"])
            start_date_str = exp_start.strftime("%Y-%m-%d")
            end_date_str   = exp_end.strftime("%Y-%m-%d")
            self.logger.info(
                f"NEX-GDDP-CMIP6: derived time_slice from experiment window: "
                f"{start_date_str} to {end_date_str}"
            )

        start_dt = pd.to_datetime(start_date_str).date()
        end_dt   = pd.to_datetime(end_date_str).date()

        # --- spatial window ---
        bbox = self.bbox  # dict: lat_min/lat_max/lon_min/lon_max

        # Ensure proper ordering for NCSS
        lat_min = min(bbox["lat_min"], bbox["lat_max"])
        lat_max = max(bbox["lat_min"], bbox["lat_max"])
        lon_min = min(bbox["lon_min"], bbox["lon_max"])
        lon_max = max(bbox["lon_min"], bbox["lon_max"])

        # Optional sub-selection hooks from config
        cfg_models    = self.config.get("NEX_MODELS", None)     # e.g. ["ACCESS-CM2"]
        cfg_scenarios = self.config.get("NEX_SCENARIOS", None)  # e.g. ["historical", "ssp245"]
        cfg_variables = self.config.get("NEX_VARIABLES", None)  # e.g. ["pr", "tasmax"]
        cfg_members   = self.config.get("NEX_ENSEMBLES", ["r1i1p1f1"])

        if cfg_variables is not None:
            variables = cfg_variables
        else:
            variables = [
                "hurs", "huss", "pr", "rlds", "rsds",
                "sfcWind", "tas", "tasmax", "tasmin",
            ]

        if not cfg_models:
            raise ValueError("NEX_MODELS must be set for THREDDS/NCSS access.")
        if cfg_scenarios is None:
            cfg_scenarios = ["historical"]

        ncss_base = "https://ds.nccs.nasa.gov/thredds/ncss/grid"

        self.logger.info("Fetching NEX-GDDP-CMIP6 from NCCS THREDDS (NCSS)")
        self.logger.info(f"  Time slice: {start_date_str} to {end_date_str}")
        self.logger.info(f"  Bbox: {bbox}")
        self.logger.info(f"  Variables: {variables}")
        self.logger.info(f"  Models: {cfg_models}")
        self.logger.info(f"  Scenarios: {cfg_scenarios}")
        self.logger.info(f"  Ensembles: {cfg_members}")
        self.logger.info(f"  NCSS base URL: {ncss_base}")

        downloaded_paths: List[Path] = []

        # root cache dir for NCSS subset files
        cache_root = output_dir / "_nex_ncss_cache"
        cache_root.mkdir(parents=True, exist_ok=True)

        def _year_chunks(s: dt.date, e: dt.date):
            """Yield (year, chunk_start, chunk_end) between dates s and e."""
            for year in range(s.year, e.year + 1):
                y_start = dt.date(year, 1, 1)
                y_end   = dt.date(year, 12, 31)
                chunk_start = max(s, y_start)
                chunk_end   = min(e, y_end)
                if chunk_start <= chunk_end:
                    yield year, chunk_start, chunk_end

        # We'll build one Dataset per (model, scenario, member),
        # then stack them along a new 'ensemble' dimension.
        ensemble_datasets = []

        for model_name in cfg_models:
            for scenario_name in cfg_scenarios:
                # Clip end date for historical to 2014-12-31 (NEX-GDDP-CMIP6 coverage)
                if scenario_name == "historical":
                    scenario_end_dt = min(end_dt, dt.date(2014, 12, 31))
                else:
                    scenario_end_dt = end_dt

                # If the clipped window is empty, skip this scenario
                if start_dt > scenario_end_dt:
                    self.logger.warning(
                        "No temporal overlap between experiment window %s–%s "
                        "and available data for scenario %s.",
                        start_dt, scenario_end_dt, scenario_name,
                    )
                    continue

                years_in_window = list(range(start_dt.year, scenario_end_dt.year + 1))
                self.logger.info(
                    f"  Scenario {scenario_name}: effective years {years_in_window}"
                )

                for member in cfg_members:
                    all_nc_files_for_ens: List[str] = []

                    for var in variables:
                        self.logger.info(
                            f"NEX-GDDP-CMIP6 NCSS: model={model_name}, "
                            f"scenario={scenario_name}, member={member}, var={var}"
                        )

                        var_cache_dir = (
                            cache_root / model_name / scenario_name / member / var
                        )
                        var_cache_dir.mkdir(parents=True, exist_ok=True)

                        for year, chunk_start, chunk_end in _year_chunks(start_dt, scenario_end_dt):

                            # Always use v2.0 filenames, as per working example
                            fname = (
                                f"{var}_day_{model_name}_{scenario_name}_"
                                f"{member}_gn_{year}_v2.0.nc"
                            )

                            dataset_path = (
                                f"AMES/NEX/GDDP-CMIP6/"
                                f"{model_name}/{scenario_name}/{member}/{var}/"
                                f"{fname}"
                            )
                            ncss_url = f"{ncss_base}/{dataset_path}"

                            out_nc = (
                                var_cache_dir /
                                f"{fname.replace('.nc', '')}_"
                                f"{chunk_start:%Y%m%d}-{chunk_end:%Y%m%d}.nc"
                            )

                            # Log one example URL for debugging
                            if (
                                year == years_in_window[0]
                                and var == variables[0]
                                and scenario_name == cfg_scenarios[0]
                                and model_name == cfg_models[0]
                            ):
                                self.logger.info(
                                    f"Example NCSS URL: {ncss_url}"
                                )

                            if out_nc.exists():
                                self.logger.info(
                                    f"Using cached NCSS subset {out_nc} for {var}, year {year}"
                                )
                                all_nc_files_for_ens.append(str(out_nc))
                                continue

                            params = {
                                "var": var,
                                "north": lat_max,
                                "south": lat_min,
                                "west": lon_min,
                                "east": lon_max,
                                "horizStride": 1,
                                "time_start": f"{chunk_start.isoformat()}T12:00:00Z",
                                "time_end":   f"{chunk_end.isoformat()}T12:00:00Z",
                                "accept": "netcdf4-classic",
                            }

                            self.logger.info(
                                "Requesting NCSS subset: "
                                f"{model_name}/{scenario_name}/{member}/{var}/{year} "
                                f"{chunk_start}–{chunk_end} (v2.0)"
                            )

                            try:
                                resp = requests.get(
                                    ncss_url,
                                    params=params,
                                    stream=True,
                                    timeout=600,
                                )
                            except Exception as e:
                                self.logger.warning(
                                    f"NCSS request failed for "
                                    f"{model_name}/{scenario_name}/{member}/{var}/{year} "
                                    f"(v2.0): {e}"
                                )
                                continue

                            if resp.status_code == 404:
                                self.logger.warning(
                                    "NCSS 404 for %s (v2.0) – no data for this year.",
                                    ncss_url,
                                )
                                continue
                            elif resp.status_code != 200:
                                self.logger.warning(
                                    "NCSS request returned status %s for %s/%s/%s/%s/%s",
                                    resp.status_code, model_name, scenario_name,
                                    member, var, year,
                                )
                                continue

                            # stream to disk
                            with open(out_nc, "wb") as f:
                                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                                    if chunk:
                                        f.write(chunk)

                            self.logger.info(
                                f"  -> wrote NCSS subset to {out_nc}"
                            )
                            all_nc_files_for_ens.append(str(out_nc))

                    if not all_nc_files_for_ens:
                        self.logger.warning(
                            "No NCSS subset files created for "
                            f"{model_name}/{scenario_name}/{member} "
                            f"in {start_dt}–{scenario_end_dt}"
                        )
                        continue

                    # --- Build combined dataset for this (model, scenario, member) ---
                    self.logger.info(
                        "Opening %d NCSS subset files for %s/%s/%s (all vars)",
                        len(all_nc_files_for_ens), model_name, scenario_name, member
                    )

                    ds_ens = xr.open_mfdataset(
                        all_nc_files_for_ens,
                        engine="netcdf4",
                        combine="by_coords",
                        parallel=False,
                    )

                    # Rechunk so NetCDF writing sees uniform chunk sizes along time
                    ds_ens = ds_ens.chunk({"time": -1})

                    # Add an 'ensemble' dimension (length 1) plus coords
                    ens_index = len(ensemble_datasets)
                    ds_ens = ds_ens.expand_dims(ensemble=[ens_index])
                    ds_ens = ds_ens.assign_coords(
                        model=("ensemble", [model_name]),
                        scenario=("ensemble", [scenario_name]),
                        member=("ensemble", [member]),
                    )

                    ensemble_datasets.append(ds_ens)

        if not ensemble_datasets:
            # Clean up cache before failing
            if cache_root.exists():
                self.logger.info(f"Removing NCSS cache directory: {cache_root}")
                shutil.rmtree(cache_root)

            raise RuntimeError(
                "NEX-GDDP-CMIP6 (THREDDS/NCSS): no data written for the requested "
                "models/scenarios/ensembles/variables/time window. "
                "Check NEX_MODELS, NEX_SCENARIOS, NEX_ENSEMBLES, "
                "time_slice/experiment dates, and bbox."
            )

        # --- Stack all ensembles along the 'ensemble' dimension ---
        self.logger.info(
            "Combining %d ensemble datasets into a single dataset", len(ensemble_datasets)
        )
        ds_all = xr.concat(ensemble_datasets, dim="ensemble")

        # --- Write ONE monthly NetCDF per month with ALL vars & ALL ensembles ---
        time_vals = pd.to_datetime(ds_all["time"].values)
        t_start = time_vals[0]
        t_end   = time_vals[-1]

        month_starts = pd.date_range(
            t_start.replace(day=1), t_end, freq="MS"
        )

        base_prefix = "NEXGDDP_all"

        for ms in month_starts:
            me = (ms + pd.offsets.MonthEnd(0))
            ds_m = ds_all.sel(time=slice(ms, me))

            if "time" not in ds_m.dims or ds_m.sizes["time"] == 0:
                continue

            month_fname = f"{base_prefix}_{ms.year:04d}{ms.month:02d}.nc"
            month_path = output_dir / month_fname

            self.logger.info(
                f"Writing monthly NetCDF4 (all vars, all ensembles): {month_path} "
                f"(time={ds_m.sizes['time']}, ensemble={ds_m.sizes['ensemble']})"
            )
            ds_m.to_netcdf(month_path, engine="netcdf4")

            downloaded_paths.append(month_path)

        ds_all.close()

        # Clean up NCSS cache now that all monthly NetCDFs are written
        if cache_root.exists():
            self.logger.info(f"Removing NCSS cache directory: {cache_root}")
            shutil.rmtree(cache_root)

        return output_dir


    def _download_aorc(self, output_dir: Path) -> Path:
        """
        Download AORC data from S3 Zarr store.
        
        AORC bucket structure:
        s3://noaa-nws-aorc-v1-1-1km/YEAR.zarr
        """
        self.logger.info("Downloading AORC data from S3")
        self.logger.info(f"  Bounding box (config): {self.bbox}")
        self.logger.info(f"  Time period: {self.start_date} to {self.end_date}")
        
        # Determine which years we need
        years = range(self.start_date.year, self.end_date.year + 1)
        
        datasets = []
        for year in years:
            self.logger.info(f"  Processing year {year}...")
            
            try:
                # Open Zarr store for this year
                zarr_path = f'noaa-nws-aorc-v1-1-1km/{year}.zarr'
                store = s3fs.S3Map(zarr_path, s3=self.fs)
                ds = xr.open_zarr(store)
                
                # -------------------------------
                # Handle longitude convention
                # -------------------------------
                lat_min = self.bbox['lat_min']
                lat_max = self.bbox['lat_max']

                # Config may have lon_min > lon_max; fix that safely
                lon1 = self.bbox['lon_min']
                lon2 = self.bbox['lon_max']
                lon_min_cfg, lon_max_cfg = sorted([lon1, lon2])
                
                lon_vals = ds['longitude'].values
                lon_min_ds = float(lon_vals.min())
                lon_max_ds = float(lon_vals.max())
                self.logger.info(
                    f"    AORC dataset longitude range: [{lon_min_ds}, {lon_max_ds}]"
                )

                # If dataset is 0–360 and bbox is in -180–180, convert bbox
                if lon_max_ds > 180.0:
                    lon_min = (lon_min_cfg + 360.0) % 360.0
                    lon_max = (lon_max_cfg + 360.0) % 360.0
                    self.logger.info(
                        "    Converting bbox lon from [-180, 180] to [0, 360): "
                        f"{lon_min_cfg}–{lon_max_cfg} → {lon_min}–{lon_max}"
                    )
                else:
                    # Dataset already uses -180–180 (as here)
                    lon_min = lon_min_cfg
                    lon_max = lon_max_cfg
                    self.logger.info(
                        "    Using bbox lon directly in dataset convention (sorted): "
                        f"{lon_min}–{lon_max}"
                    )

                # -------------------------------
                # Subset by bounding box and time
                # -------------------------------
                ds_subset = ds.sel(
                    latitude=slice(lat_min, lat_max),
                    longitude=slice(lon_min, lon_max)
                )
                
                # Filter by time range for this year
                year_start = max(self.start_date, pd.Timestamp(f'{year}-01-01'))
                year_end = min(self.end_date, pd.Timestamp(f'{year}-12-31 23:59:59'))
                ds_subset = ds_subset.sel(time=slice(year_start, year_end))
                
                if len(ds_subset.time) > 0:
                    datasets.append(ds_subset)
                    self.logger.info(
                        f"    ✓ Extracted {len(ds_subset.time)} timesteps "
                        f"and grid {len(ds_subset.latitude)} x {len(ds_subset.longitude)}"
                    )
                else:
                    self.logger.warning(
                        f"    No timesteps after subsetting for year {year} "
                        f"(lat {lat_min}–{lat_max}, lon {lon_min}–{lon_max})"
                    )
                
            except Exception as e:
                self.logger.error(f"    ✗ Error processing year {year}: {str(e)}")
                raise
        
        if not datasets:
            raise ValueError("No data extracted for the specified time period")
        
        # Combine all years
        self.logger.info("Combining data across years...")
        ds_combined = xr.concat(datasets, dim='time')
        
        # Log data summary
        self.logger.info("Data extraction summary:")
        self.logger.info(f"  Dimensions: {dict(ds_combined.dims)}")
        self.logger.info(f"  Variables: {list(ds_combined.data_vars)}")
        self.logger.info(f"  Time steps: {len(ds_combined.time)}")
        self.logger.info(
            f"  Grid size: {len(ds_combined.latitude)} x {len(ds_combined.longitude)}"
        )
        
        # Add metadata
        ds_combined.attrs['source'] = 'NOAA AORC v1.1'
        ds_combined.attrs['source_url'] = 's3://noaa-nws-aorc-v1-1-1km'
        ds_combined.attrs['downloaded_by'] = 'SYMFLUENCE cloud_data_utils'
        ds_combined.attrs['download_date'] = pd.Timestamp.now().isoformat()
        ds_combined.attrs['bbox'] = str(self.bbox)
        
        # Save to NetCDF
        output_dir.mkdir(parents=True, exist_ok=True)
        domain_name = self.config.get('DOMAIN_NAME', 'domain')
        output_file = output_dir / f'{domain_name}_AORC_{self.start_date.year}-{self.end_date.year}.nc'
        
        self.logger.info(f"Saving data to: {output_file}")
        ds_combined.to_netcdf(output_file)
        
        self.logger.info(f"✓ AORC data download complete: {output_file}")
        return output_file

    
    def _download_era5(self, output_dir: Path) -> Path:
        """
        Download ERA5 data from Google Cloud ARCO-ERA5, in monthly chunks,
        and write NetCDF files that already match the SUMMA forcing schema
        expected by the agnostic preprocessor.

        This function:
          * reads ARCO-ERA5 Zarr with ARCO variable names,
          * subsets to the user bbox and experiment window,
          * applies optional temporal thinning,
          * converts variables to SUMMA-style names/units:
              - airpres   [Pa]
              - LWRadAtm  [W m-2]
              - SWRadAtm  [W m-2]
              - pptrate   [m s-1]
              - airtemp   [K]
              - spechum   [kg kg-1] (from dewpoint + pressure)
              - windspd   [m s-1]   (from 10m U/V),
          * writes one NetCDF file per monthly chunk.

        The resulting NetCDFs can be consumed directly by the agnostic
        preprocessor without additional renaming or unit conversion.
        """
        self.logger.info("Downloading ERA5 data from Google Cloud ARCO-ERA5")
        self.logger.info(f"  Bounding box: {self.bbox}")
        self.logger.info(f"  Time period: {self.start_date} to {self.end_date}")
        domain_name = self.config.get("DOMAIN_NAME", "domain")

        try:
            import gcsfs
            import xarray as xr
            import pandas as pd
            from pandas.tseries.offsets import MonthEnd
        except ImportError as e:
            raise ImportError(
                "gcsfs and xarray are required for ERA5 cloud access. "
                "Install with: pip install gcsfs xarray"
            ) from e

        try:
            # Anonymous GCS filesystem
            gcs = gcsfs.GCSFileSystem(token="anon")

            # Allow override via config, otherwise use default ARCO path
            default_store = (
                "gcp-public-data-arco-era5/ar/"
                "full_37-1h-0p25deg-chunk-1.zarr-v3"
            )
            zarr_store = self.config.get("ERA5_ZARR_PATH", default_store)

            self.logger.info(f"Opening ARCO-ERA5 Zarr store: {zarr_store}")

            # dask-backed, using store-defined chunks
            mapper = gcs.get_mapper(zarr_store)
            ds = xr.open_zarr(
                mapper,
                consolidated=True,
                chunks={},  # keep lazy / dask-backed
            )

            self.logger.info("Successfully opened ERA5 Zarr store")

            # Load coordinate axes (small) to avoid lazy coord ops
            ds = ds.assign_coords(
                longitude=ds.longitude.load(),
                latitude=ds.latitude.load(),
                time=ds.time.load(),
            )
            self.logger.info("ERA5 coordinate axes loaded into memory.")

            all_vars = list(ds.data_vars)
            sample_vars = all_vars[:20]
            self.logger.info(
                f"  Available variables (sample, first 20): {sample_vars}"
            )

            # ----------------- Robust bbox handling -----------------------------
            raw_lon1 = float(self.bbox["lon_min"])
            raw_lon2 = float(self.bbox["lon_max"])
            raw_lat1 = float(self.bbox["lat_min"])
            raw_lat2 = float(self.bbox["lat_max"])

            if "longitude" not in ds.coords or "latitude" not in ds.coords:
                raise ValueError(
                    "ERA5 dataset does not have 'longitude' and 'latitude' coordinates"
                )

            lon_min_raw, lon_max_raw = sorted([raw_lon1, raw_lon2])

            # Convert to 0..360 if needed (ARCO-ERA5 uses 0..360)
            if lon_min_raw < 0 or lon_max_raw < 0:
                lon_min = (lon_min_raw + 360) % 360
                lon_max = (lon_max_raw + 360) % 360
            else:
                lon_min, lon_max = lon_min_raw, lon_max_raw

            lat_min_raw, lat_max_raw = sorted([raw_lat1, raw_lat2])

            self.logger.info(
                f"Normalised bbox: lat [{lat_min_raw}, {lat_max_raw}], "
                f"lon [{lon_min}, {lon_max}] (0–360 frame)"
            )

            # ------------------- Temporal thinning factor -----------------------
            step = int(self.config.get("ERA5_TIME_STEP_HOURS", 1))
            if step < 1:
                step = 1

            # For accumulated fields (precip/radiation) we take finite differences and
            # drop the first step. To ensure the *output* starts exactly at
            # EXPERIMENT_TIME_START, we request one extra ERA5 step before the
            # experiment window.
            era5_start = self.start_date - pd.Timedelta(hours=step)
            era5_end = self.end_date

            # ------------------- Build monthly chunks ---------------------------
            first_month_start = era5_start.replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )
            current_month_start = first_month_start

            chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
            while current_month_start <= era5_end:
                month_end = current_month_start + MonthEnd(1)
                month_end = month_end.replace(
                    hour=23, minute=0, second=0, microsecond=0
                )

                # Use the expanded window [era5_start, era5_end] for selection
                chunk_start = max(era5_start, current_month_start)
                chunk_end = min(era5_end, month_end)

                if chunk_start <= chunk_end:
                    chunks.append((chunk_start, chunk_end))

                # Move to next calendar month
                current_month_start = (
                    current_month_start.replace(day=28)
                    + pd.Timedelta(days=4)
                ).replace(
                    day=1, hour=0, minute=0, second=0, microsecond=0
                )


            if not chunks:
                raise ValueError(
                    "ERA5: no monthly chunks found for the requested time period. "
                    "Check EXPERIMENT_TIME_START/END."
                )

            self.logger.info(
                f"ERA5: processing in {len(chunks)} monthly chunk(s)"
            )

            # ------------------- Variable selection -----------------------------
            default_required_vars = [
                "2m_temperature",
                "2m_dewpoint_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "surface_pressure",
                "total_precipitation",
                "surface_solar_radiation_downwards",
                "surface_thermal_radiation_downwards",
            ]

            requested_vars = self.config.get(
                "ERA5_VARS", default_required_vars
            )
            self.logger.debug(f"Requested ERA5 variables: {requested_vars}")

            available_vars = [
                v for v in requested_vars if v in ds.data_vars
            ]
            if not available_vars:
                raise ValueError(
                    "None of the requested variables found in ARCO-ERA5 dataset. "
                    f"Requested: {requested_vars}. "
                    f"Available (sample): {sample_vars}"
                )
            self.logger.debug(f"Variables present in dataset: {available_vars}")

            # ------------------- Output directory -------------------------------
            output_dir.mkdir(parents=True, exist_ok=True)
            domain_name = self.config.get("DOMAIN_NAME", "domain")

            # ------------------- Parallel processing setup -------------------
            # Determine number of parallel workers
            n_workers = self.config.get('ERA5_PARALLEL_WORKERS', 
                                       self.config.get('MPI_PROCESSES', 4))
            n_workers = max(1, int(n_workers))  # At least 1 worker
            n_workers = min(n_workers, len(chunks))  # No more workers than chunks
            
            self.logger.info(f"Processing {len(chunks)} chunks using {n_workers} parallel workers")

            # Prepare config subset for workers (only serializable data)
            config_subset = {
                'DOMAIN_NAME': domain_name,
                'total_chunks': len(chunks),
            }
            
            # Prepare chunk information for parallel processing
            chunk_infos = []
            for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
                chunk_info = (
                    i,  # chunk index
                    (chunk_start, chunk_end),
                    zarr_store,
                    config_subset,
                    output_dir,
                    available_vars,
                    step,
                    lat_min_raw,
                    lat_max_raw,
                    lon_min,
                    lon_max,
                )
                chunk_infos.append(chunk_info)
            
            # ------------------- Process chunks in parallel -------------------
            chunk_files: list[Path] = []
            completed = 0
            total_chunks = len(chunks)
            
            if n_workers == 1:
                # Sequential processing (for debugging or single-core systems)
                self.logger.info("Using sequential processing (n_workers=1)")
                for chunk_info in chunk_infos:
                    chunk_idx, chunk_file, status = _process_era5_chunk(chunk_info)
                    self.logger.info(f"[ERA5 {chunk_idx}/{total_chunks}] {status}")
                    if chunk_file is not None:
                        chunk_files.append(chunk_file)
                    completed += 1
                    if completed % 5 == 0 or completed == total_chunks:
                        self.logger.info(f"Progress: {completed}/{total_chunks} chunks completed")
            else:
                # Parallel processing
                self.logger.info(f"Using parallel processing with {n_workers} workers")
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    # Submit all chunks
                    future_to_chunk = {
                        executor.submit(_process_era5_chunk, chunk_info): chunk_info[0]
                        for chunk_info in chunk_infos
                    }
                    
                    # Process results as they complete
                    for future in as_completed(future_to_chunk):
                        chunk_idx = future_to_chunk[future]
                        try:
                            chunk_idx_result, chunk_file, status = future.result()
                            self.logger.info(f"[ERA5 {chunk_idx_result}/{total_chunks}] {status}")
                            if chunk_file is not None:
                                chunk_files.append(chunk_file)
                        except Exception as e:
                            self.logger.error(f"[ERA5 {chunk_idx}/{total_chunks}] Exception: {str(e)}")
                        
                        completed += 1
                        if completed % 5 == 0 or completed == total_chunks:
                            self.logger.info(f"Progress: {completed}/{total_chunks} chunks completed")

            # Sort chunk files by name for consistent ordering
            chunk_files.sort()

            if not chunk_files:
                raise ValueError(
                    "ERA5: after subsetting and SUMMA conversion, no data was written. "
                    "Check bbox, time window, and variable availability."
                )

            # ------------------- Final summary ----------------------------------
            self.logger.info("ERA5 data extraction (chunked, SUMMA-format) summary:")
            self.logger.info(f"  Output directory: {output_dir}")
            self.logger.info(f"  Chunk files written: {len(chunk_files)}")
            self.logger.info(f"  Temporal thinning (hours): {step}")
            preview = [str(p.name) for p in chunk_files[:5]]
            self.logger.info(f"  Example files: {preview}")

            # If only one file, return that; otherwise return the directory
            if len(chunk_files) == 1:
                self.logger.info(
                    f"✓ ERA5 data download complete (single SUMMA-format chunk): "
                    f"{chunk_files[0]}"
                )
                return chunk_files[0]
            else:
                self.logger.info(
                    f"✓ ERA5 data download complete: {len(chunk_files)} files in {output_dir}"
                )
                return output_dir

        except ImportError:
            raise ImportError(
                "gcsfs package is required for ERA5 cloud access. "
                "Install with: pip install gcsfs"
            )
        except Exception as e:
            self.logger.error(f"Error downloading ERA5 data: {str(e)}")
            raise



    # ------------------------------------------------------------------
    # SoilGrids soil classes via WCS → attributes/soilclasses
    # ------------------------------------------------------------------
    def download_soilgrids_soilclasses(self) -> Path:
        """
        Fetch SoilGrids soil 'classes' (or any SoilGrids coverage) for the
        domain bbox using WCS, and save a GeoTIFF in:

          domain_dir / attributes / soilclasses /
              <DOMAIN_NAME>_soilgrids_soilclasses.tif

        Requires in config:
          SOILGRIDS_WCS_MAP: e.g. "/map/soilgrids.map"
          SOILGRIDS_COVERAGE_ID: e.g. "clay_0-5cm_mean" or a USDA/WRB class

        The function does NOT reproject; it requests WGS84 (EPSG:4326) from
        the WCS server directly.
        """
        soil_dir = self._attribute_dir("soilclass")

        wcs_map   = self.config.get("SOILGRIDS_WCS_MAP")
        coverage  = self.config.get("SOILGRIDS_COVERAGE_ID")

        if wcs_map is None or coverage is None:
            raise ValueError(
                "SoilGrids WCS requires SOILGRIDS_WCS_MAP and "
                "SOILGRIDS_COVERAGE_ID in the config."
            )

        bbox = self.bbox  # dict: {'lat_min', 'lat_max', 'lon_min', 'lon_max'}
        lon_min = bbox["lon_min"]
        lon_max = bbox["lon_max"]
        lat_min = bbox["lat_min"]
        lat_max = bbox["lat_max"]

        self.logger.info(
            "Requesting SoilGrids WCS coverage "
            f"{coverage} for bbox "
            f"lon [{lon_min}, {lon_max}], lat [{lat_min}, {lat_max}]"
        )

        # SoilGrids WCS endpoint
        base_url = "https://maps.isric.org/mapserv"

        # WCS 2.0.1 request parameters.
        # We ask for EPSG:4326 as both subset and output CRS.
        params = [
            ("map", wcs_map),
            ("SERVICE", "WCS"),
            ("VERSION", "2.0.1"),
            ("REQUEST", "GetCoverage"),
            ("COVERAGEID", coverage),
            ("FORMAT", "GEOTIFF_INT16"),
            ("SUBSETTINGCRS", "http://www.opengis.net/def/crs/EPSG/0/4326"),
            ("OUTPUTCRS",     "http://www.opengis.net/def/crs/EPSG/0/4326"),
            ("SUBSET", f"Lat({lat_min},{lat_max})"),
            ("SUBSET", f"Lon({lon_min},{lon_max})"),
        ]

        resp = requests.get(base_url, params=params, stream=True)
        try:
            resp.raise_for_status()
        except Exception as e:
            # Log response text for debugging (likely WCS error XML)
            self.logger.error(
                f"SoilGrids WCS GetCoverage failed: {e}. "
                f"Response: {resp.text[:500]}..."
            )
            raise

        out_path = soil_dir / f"domain_{self.config.get('DOMAIN_NAME','domain')}_soil_classes.tif"
        self.logger.info(f"Writing SoilGrids soilclasses GeoTIFF to {out_path}")

        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)

        return out_path


    # ------------------------------------------------------------------
    # MODIS MCD12Q1.061 land cover (local raster) → attributes/landcover
    # ------------------------------------------------------------------
    def download_modis_landcover(self) -> Path:
        """
        Acquire MODIS MCD12Q1 v061-style land cover for the domain and write a
        MAF-style landclass GeoTIFF:

          domain_dir / attributes / landclass / landclass.tif

        Behaviour
        ---------
        1) If LANDCOVER_LOCAL_FILE is set in config:
           - Treat it as an existing EPSG:4326 raster (local or /vsicurl/http),
             crop to bbox, and write landclass.tif.

        2) If LANDCOVER_LOCAL_FILE is NOT set:
           - Use annual MCD12Q1 v061 land cover COGs from Zenodo (IGBP classes),
             one file per year, at native 500 m resolution (EPSG:4326).
           - Crop each year to bbox, stack across years, and compute the
             per-pixel MODE of land class (ignoring nodata).
           - Write the mode field to landclass.tif (MAF-style).
        """
        lc_dir = self._attribute_dir("landclass")

        # ------------------------------------------------------------------
        # Bounding box
        # ------------------------------------------------------------------
        bbox = self.bbox  # e.g. {'lat_min', 'lat_max', 'lon_min', 'lon_max'}

        lon_min = float(bbox["lon_min"])
        lon_max = float(bbox["lon_max"])
        lat_min = float(bbox["lat_min"])
        lat_max = float(bbox["lat_max"])

        # Normalise in case user reversed coordinates
        if lon_min > lon_max:
            self.logger.info(
                f"Swapping lon_min ({lon_min}) and lon_max ({lon_max}) - they were reversed"
            )
            lon_min, lon_max = lon_max, lon_min
        if lat_min > lat_max:
            self.logger.info(
                f"Swapping lat_min ({lat_min}) and lat_max ({lat_max}) - they were reversed"
            )
            lat_min, lat_max = lat_max, lat_min

        self.logger.info(
            "Preparing MODIS landcover for bbox "
            f"lon [{lon_min}, {lon_max}], lat [{lat_min}, {lat_max}]"
        )

        from rasterio.windows import from_bounds

        # ------------------------------------------------------------------
        # Case 1: user supplied a local/remote raster
        # ------------------------------------------------------------------
        src_path = self.config.get("LANDCOVER_LOCAL_FILE", None)
        is_remote = False

        if src_path is not None:
            # Allow vsicurl/http here too
            if (
                isinstance(src_path, str)
                and (
                    src_path.startswith("/vsicurl/")
                    or src_path.startswith("http://")
                    or src_path.startswith("https://")
                )
            ):
                is_remote = True
            else:
                src_path = Path(src_path)
                if not src_path.exists():
                    raise FileNotFoundError(
                        f"LANDCOVER_LOCAL_FILE does not exist: {src_path}"
                    )

            self.logger.info(
                f"LANDCOVER_LOCAL_FILE set; cropping provided raster: {src_path}"
            )

            with rasterio.open(src_path) as src:
                if src.crs is None or src.crs.to_epsg() != 4326:
                    self.logger.warning(
                        "Landcover source raster is not EPSG:4326. "
                        "Cropping may be incorrect unless it was reprojected beforehand."
                    )

                window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
                out_transform = src.window_transform(window)
                out_data = src.read(1, window=window)

                out_meta = src.meta.copy()
                out_meta.update(
                    {
                        "driver": "GTiff",
                        "height": out_data.shape[0],
                        "width": out_data.shape[1],
                        "transform": out_transform,
                        "count": 1,
                    }
                )

            # MAF-style output name
            out_path = lc_dir / "landclass.tif"
            self.logger.info(f"Writing cropped landclass raster to {out_path}")

            with rasterio.open(out_path, "w", **out_meta) as dst:
                dst.write(out_data, 1)

            return out_path

        # ------------------------------------------------------------------
        # Case 2: multi-year MODIS MCD12Q1 from Zenodo COGs → per-pixel mode
        # ------------------------------------------------------------------
        self.logger.info(
            "LANDCOVER_LOCAL_FILE not set; using multi-year MODIS MCD12Q1 v061 "
            "COGs from Zenodo to compute per-pixel MODE of land cover."
        )

        # Years to include in the mode calculation (configurable)
        default_years = list(range(2001, 2020))  # 2001–2019
        years = self.config.get("MODIS_LANDCOVER_YEARS", default_years)

        # Base URL + filename pattern (can be overridden in config)
        # OpenGeoHub MCD12Q1 v061 LC_Type1 (IGBP) mosaics, 500 m, EPSG:4326
        base_url = self.config.get(
            "MODIS_LANDCOVER_BASE_URL",
            "https://zenodo.org/records/8367523/files",
        )
        filename_template = self.config.get(
            "MODIS_LANDCOVER_TEMPLATE",
            "lc_mcd12q1v061.t1_c_500m_s_{year}0101_{year}1231_go_epsg.4326_v20230818.tif",
        )


        arrays = []
        out_meta = None
        nodata_val = None
        window = None
        out_transform = None
        first_shape = None

        for year in years:
            fname = filename_template.format(year=year)
            url = f"/vsicurl/{base_url}/{fname}"

            self.logger.info(f"  Opening MODIS landcover COG for year {year}: {url}")

            try:
                with rasterio.open(url) as src:
                    if src.crs is None or src.crs.to_epsg() != 4326:
                        self.logger.warning(
                            f"Year {year} MODIS landcover raster is not EPSG:4326; "
                            "bbox cropping assumes EPSG:4326."
                        )

                    # Use the first year to define the crop window & transform
                    if window is None:
                        window = from_bounds(
                            lon_min, lat_min, lon_max, lat_max, src.transform
                        )
                        out_transform = src.window_transform(window)
                        out_meta = src.meta.copy()

                    data = src.read(1, window=window)

                    if first_shape is None:
                        first_shape = data.shape
                    elif data.shape != first_shape:
                        self.logger.warning(
                            f"Year {year} landcover window shape {data.shape} "
                            f"does not match first shape {first_shape}; skipping year."
                        )
                        continue

                    arrays.append(data)

                    # Capture nodata value from first successful year
                    if nodata_val is None:
                        nodata_val = src.nodata
                        if nodata_val is None:
                            nodata_val = 255  # fallback for MODIS MCD12Q1
                            self.logger.info(
                                "MODIS landcover nodata not set; defaulting to 255."
                            )

            except Exception as e:
                self.logger.warning(
                    f"  Could not read MODIS COG for year {year}: {e}; skipping."
                )
                continue

        if not arrays:
            raise RuntimeError(
                "No MODIS landcover rasters could be opened for the requested years. "
                "Check network access, base URL, and year list."
            )

        self.logger.info(
            f"Computing per-pixel mode of MODIS landcover across {len(arrays)} year(s): "
            f"{years}"
        )

        stack = np.stack(arrays, axis=0)  # (nyears, ny, nx)

        if nodata_val is None:
            nodata_val = 255

        def _mode_1d(vals: np.ndarray) -> int:
            """Mode of a 1D array ignoring nodata; returns nodata if all missing."""
            valid = vals[vals != nodata_val]
            if valid.size == 0:
                return nodata_val
            # np.unique is fine for small number of years
            uniq, counts = np.unique(valid, return_counts=True)
            # In case of ties, np.argmax picks the first occurrence
            return int(uniq[np.argmax(counts)])

        # Apply along the "years" axis → result shape (ny, nx)
        mode_data = np.apply_along_axis(_mode_1d, 0, stack)
        mode_data = mode_data.astype(out_meta["dtype"] if "dtype" in out_meta else stack.dtype)

        # Update metadata for output raster
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mode_data.shape[0],
                "width": mode_data.shape[1],
                "transform": out_transform,
                "count": 1,
                "nodata": nodata_val,
            }
        )

        out_path = lc_dir / f"domain_{self.config.get('DOMAIN_NAME')}_land_classes.tif"
        self.logger.info(
            f"Writing MODIS landcover MODE raster (MAF-style) to {out_path}"
        )

        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(mode_data, 1)

        self.logger.info(
            "✓ MODIS landcover mode computation complete; "
            f"landclass written to {out_path}"
        )
        return out_path




    def _download_emearth(self, output_dir: Path) -> Path:
        """
        Download EM-Earth data from AWS S3.

        Structure for daily deterministic (global):
            s3://emearth/nc/deterministic_raw_daily/{variable}/EM_Earth_deterministic_daily_{variable}_{YYYYMM}.nc

        Config options
        --------------
        EM_EARTH_DATA_TYPE : str, optional
            'deterministic' (default) or 'ensemble'
        EM_PRCP : str, optional
            Precipitation variable to use: 'prcp' (default) or 'prcp_corrected'
        EM_EARTH_REGION_FOLDER : str, optional
            Region subfolder (e.g., 'NorthAmerica') or 'global' (default)
        SUPPLEMENT_FORCING : bool
            If True, treat EM-Earth as supplemental forcing and save under a dedicated subdir.

        Parameters
        ----------
        output_dir : Path
            Directory to save downloaded data

        Returns
        -------
        Path
            Path to the saved NetCDF file
        """
        import pandas as pd
        import xarray as xr

        self.logger.info("Downloading EM-Earth data from AWS S3")
        self.logger.info(f"  Bounding box: {self.bbox}")
        self.logger.info(f"  Time period: {self.start_date} to {self.end_date}")

        # Determine data type
        emearth_type_cfg = self.config.get("EM_EARTH_DATA_TYPE", "deterministic")
        emearth_type = str(emearth_type_cfg).lower()
        if emearth_type not in ("deterministic", "ensemble"):
            self.logger.warning(
                f"Unknown EM_EARTH_DATA_TYPE='{emearth_type_cfg}', defaulting to 'deterministic'"
            )
            emearth_type = "deterministic"
        data_type = emearth_type
        self.logger.info(f"  Using {data_type} EM-Earth data")

        # Set base folder (daily deterministic vs ensemble)
        if data_type == "deterministic":
            base_folder = "nc/deterministic_raw_daily"
        else:
            base_folder = "nc/probabilistic_daily"
        self.logger.info(f"  Using folder path: {base_folder}")

        # Precipitation variable selection
        precip_var = self.config.get("EM_PRCP", "prcp")
        self.logger.info(f"  Using EM-Earth precip variable: {precip_var}")

        # Variables list to download
        em_earth_vars = [precip_var, "tmean", "trange", "tdew"]

        # Region handling (global default)
        region_folder_cfg = self.config.get("EM_EARTH_REGION_FOLDER", "global")
        region_folder = str(region_folder_cfg)
        use_region_subfolder = region_folder.lower() not in ("global", "")
        if use_region_subfolder:
            self.logger.info(f"  Using region subfolder: {region_folder}")
        else:
            self.logger.info("  No region subfolder (global)")

        # Determine year/month range (dataset supports 1950-2019) :contentReference[oaicite:1]{index=1}
        min_year = 1950
        max_year = 2019
        start_year = max(self.start_date.year, min_year)
        end_year = min(self.end_date.year, max_year)
        if self.start_date.year < min_year:
            self.logger.warning(f"EM-Earth only supports from year {min_year}; truncating start year to {min_year}")
        if self.end_date.year > max_year:
            self.logger.warning(f"EM-Earth only supports through year {max_year}; truncating end year to {max_year}")
        years = range(start_year, end_year + 1)

        all_datasets = {}

        for var in em_earth_vars:
            self.logger.info(f"  Processing variable: {var}")
            var_datasets = []

            for year in years:
                for month in range(1, 13):
                    ym = f"{year}{month:02d}"
                    filename = f"EM_Earth_deterministic_daily_{var}_{ym}.nc" if data_type == "deterministic" else f"EM_Earth_probabilistic_daily_{var}_{ym}.nc"

                    if use_region_subfolder:
                        s3_key = f"emearth/{base_folder}/{var}/{region_folder}/{filename}"
                    else:
                        s3_key = f"emearth/{base_folder}/{var}/{filename}"

                    self.logger.debug(f"    Trying key: {s3_key}")

                    try:
                        if not self.fs.exists(s3_key):
                            self.logger.warning(f"    File not found for year-month {ym}, var {var}: {s3_key}")
                            continue

                        with self.fs.open(s3_key, "rb") as f:
                            ds = xr.open_dataset(f, engine="h5netcdf")

                        # Subset spatially
                        ds_subset = ds.sel(
                            lat=slice(self.bbox["lat_min"], self.bbox["lat_max"]),
                            lon=slice(self.bbox["lon_min"], self.bbox["lon_max"])
                        )

                        # Subset temporally for that year/month
                        ym_start = pd.Timestamp(f"{year}-{month:02d}-01")
                        ym_end = ym_start + pd.offsets.MonthEnd(0)
                        real_start = max(self.start_date, ym_start)
                        real_end = min(self.end_date, ym_end)
                        ds_subset = ds_subset.sel(time=slice(real_start, real_end))

                        if len(ds_subset.time) > 0:
                            ds_subset = ds_subset.load()
                            var_datasets.append(ds_subset)
                            self.logger.info(f"    ✓ Year-month {ym}: {len(ds_subset.time)} timesteps")
                        else:
                            self.logger.debug(f"    Year-month {ym}: no timesteps in requested period")

                    except Exception as e:
                        self.logger.warning(f"    ⚠ Error retrieving {s3_key}: {e}")
                        continue

            if var_datasets:
                combined = xr.concat(var_datasets, dim="time")
                all_datasets[var] = combined
            else:
                self.logger.warning(f"  No data found for variable {var}")

        if not all_datasets:
            raise ValueError("No EM-Earth data could be downloaded for the specified period")

        self.logger.info("Merging all EM-Earth variables...")
        ds_final = xr.merge(list(all_datasets.values()))

        # Logging summary
        self.logger.info("EM-Earth data extraction summary:")
        self.logger.info(f"  Data type: {data_type}")
        self.logger.info(f"  Dimensions: {dict(ds_final.dims)}")
        self.logger.info(f"  Variables: {list(ds_final.data_vars)}")
        self.logger.info(f"  Time steps: {len(ds_final.time)}")
        self.logger.info(f"  Grid size: {len(ds_final.lat)} x {len(ds_final.lon)}")

        # Add metadata
        ds_final.attrs["source"] = "EM-Earth (University of Saskatchewan)"
        ds_final.attrs["source_url"] = "s3://emearth"
        ds_final.attrs["data_type"] = data_type
        ds_final.attrs["downloaded_by"] = "SYMFLUENCE cloud_data_utils"
        ds_final.attrs["download_date"] = pd.Timestamp.now().isoformat()
        ds_final.attrs["bbox"] = str(self.bbox)

        # Save
        if self.supplement_data:
            output_subdir = output_dir / "raw_data_em_earth"
            output_subdir.mkdir(parents=True, exist_ok=True)
            save_dir = output_subdir
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_dir = output_dir

        domain_name = self.config.get("DOMAIN_NAME", "domain")
        output_file = save_dir / f"{domain_name}_EM-Earth_{data_type}_{start_year}-{end_year}.nc"

        self.logger.info(f"Saving EM-Earth data to: {output_file}")
        ds_final.to_netcdf(output_file)

        self.logger.info(f"✓ EM-Earth data download complete: {output_file}")
        return output_file


    def _download_hrrr(self, output_dir: Path) -> Path:
        """
        Download HRRR hourly analysis data from AWS S3 Zarr store.

        Uses the HRRR-Zarr surface dataset from s3://hrrrzarr/sfc, grabbing
        analysis (anl, F00) files for each hour in the requested date range.

        The function:
        * Opens each variable as a lazy, dask-backed Xarray dataset.
        * Merges variables per hour, subsetting to the requested bbox.
        * Concatenates all hours along 'time' and writes a single NetCDF.

        Config options (optional)
        -------------------------
        HRRR_VARS : list[str]
            List of HRRR parameter names to download (e.g., ["TMP", "APCP"]).
            Defaults to a hydrology-relevant subset.
        HRRR_TIME_STEP_HOURS : int
            Temporal thinning factor applied after concatenation (1 = hourly,
            3 = every 3rd hour, etc.).
        """
        self.logger.info("Downloading HRRR HOURLY analysis data from AWS S3 Zarr")
        self.logger.info(f"  Bounding box: {self.bbox}")
        self.logger.info(f"  Time period: {self.start_date} to {self.end_date}")

        import xarray as xr
        import numpy as np
        import pandas as pd
        import s3fs

        # Ensure we have an S3 filesystem (you already set self.fs earlier)
        if not hasattr(self, "fs") or self.fs is None:
            self.fs = s3fs.S3FileSystem(anon=True)

        # Default hydrology-focused variable set (matches your existing mapping)
        hrrr_variables = {
            "TMP": "2m_above_ground",   # 2m Temperature [K]
            "SPFH": "2m_above_ground",  # 2m Specific humidity [kg/kg]
            "PRES": "surface",          # Surface pressure [Pa]
            "UGRD": "10m_above_ground", # 10m U wind component [m/s]
            "VGRD": "10m_above_ground", # 10m V wind component [m/s]
            "DSWRF": "surface",         # Downward shortwave radiation flux [W/m2]
            "DLWRF": "surface",         # Downward longwave radiation flux [W/m2]
            "APCP": "surface",          # Total precipitation [kg/m2]
        }

        # Optional variable override from config
        requested_hrrr_vars = self.config.get("HRRR_VARS")
        if requested_hrrr_vars:
            requested_hrrr_vars = set(requested_hrrr_vars)
            hrrr_variables = {
                k: v for k, v in hrrr_variables.items() if k in requested_hrrr_vars
            }

        self.logger.info(f"  HRRR variables to download: {list(hrrr_variables.keys())}")

        all_datasets = []
        successful_hours = 0
        failed_hours = 0

        # We will cache the spatial y/x slice on first successful hour
        xy_slice = getattr(self, "_hrrr_xy_slice", None)

        current_date = self.start_date.date()
        end_date = self.end_date.date()

        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")
            self.logger.info(f"  Processing date: {date_str}")

            # HRRR analysis runs every hour (00-23 UTC)
            for hour in range(0, 24):
                current_datetime = pd.Timestamp(f"{date_str} {hour:02d}:00:00")
                if current_datetime < self.start_date or current_datetime > self.end_date:
                    continue

                hour_str = f"{hour:02d}"
                self.logger.debug(f"    Attempting {date_str} {hour_str}z analysis")

                try:
                    zarr_base = f"hrrrzarr/sfc/{date_str}/{date_str}_{hour_str}z_anl.zarr"

                    var_datasets = []
                    vars_found = []

                    for var_name, level in hrrr_variables.items():
                        try:
                            # Two paths per variable, as in your original code
                            var_path1 = f"{zarr_base}/{level}/{var_name}/{level}"
                            var_path2 = f"{zarr_base}/{level}/{var_name}"

                            store1 = s3fs.S3Map(var_path1, s3=self.fs)
                            store2 = s3fs.S3Map(var_path2, s3=self.fs)

                            # Lazy open; dask-backed
                            ds_var = xr.open_mfdataset(
                                [store1, store2],
                                engine="zarr",
                                consolidated=False,
                            )

                            var_datasets.append(ds_var)
                            vars_found.append(var_name)

                        except Exception as e:
                            # Variable not available for this hour; just skip it
                            self.logger.debug(
                                f"      Variable {var_name} not available "
                                f"for {date_str} {hour_str}z: {e}"
                            )
                            continue

                    if not var_datasets:
                        failed_hours += 1
                        self.logger.debug(
                            f"    ✗ {hour_str}z - no HRRR variables available at this hour"
                        )
                        continue

                    # Merge variables for this hour
                    ds_hour = xr.merge(var_datasets)

                    # Subset to the requested bbox using cached y/x slice if available
                    if "latitude" in ds_hour.coords and "longitude" in ds_hour.coords:
                        if xy_slice is None:
                            lat_mask = (
                                (ds_hour.latitude >= self.bbox["lat_min"])
                                & (ds_hour.latitude <= self.bbox["lat_max"])
                            )
                            lon_mask = (
                                (ds_hour.longitude >= self.bbox["lon_min"])
                                & (ds_hour.longitude <= self.bbox["lon_max"])
                            )
                            spatial_mask = lat_mask & lon_mask

                            y_idx, x_idx = np.where(spatial_mask)
                            if len(y_idx) == 0:
                                self.logger.warning(
                                    f"      No HRRR gridpoints within bbox for "
                                    f"{date_str} {hour_str}z"
                                )
                                failed_hours += 1
                                continue

                            y_min, y_max = y_idx.min(), y_idx.max()
                            x_min, x_max = x_idx.min(), x_idx.max()
                            xy_slice = (slice(y_min, y_max + 1), slice(x_min, x_max + 1))
                            # Cache for later hours
                            setattr(self, "_hrrr_xy_slice", xy_slice)

                        # Apply cached slice
                        y_slice, x_slice = xy_slice
                        ds_subset = ds_hour.isel(y=y_slice, x=x_slice)
                    else:
                        # Fallback if lat/lon are not present (unlikely)
                        ds_subset = ds_hour

                    # IMPORTANT: do NOT .load() here. Keep lazy; let to_netcdf compute.
                    all_datasets.append(ds_subset)
                    successful_hours += 1

                    self.logger.info(
                        f"    ✓ {hour_str}z - {len(vars_found)} variables "
                        f"(total successful hours: {successful_hours})"
                    )

                except Exception as e:
                    failed_hours += 1
                    self.logger.warning(
                        f"    ⚠ Error loading HRRR for {date_str} {hour_str}z: {e}"
                    )
                    continue

            current_date += pd.Timedelta(days=1)

        if not all_datasets:
            raise ValueError(
                "No HRRR data could be downloaded for the specified period. "
                f"Attempted {successful_hours + failed_hours} hours; all failed."
            )

        self.logger.info("Combining HRRR hourly data across all time steps...")
        ds_final = xr.concat(all_datasets, dim="time")

        # Sort chronologically
        ds_final = ds_final.sortby("time")

        # Optional temporal thinning after concatenation
        step = int(self.config.get("HRRR_TIME_STEP_HOURS", 1))
        if step > 1 and "time" in ds_final.dims and ds_final.sizes["time"] > 1:
            self.logger.info(f"Applying HRRR temporal thinning: every {step} hours")
            ds_final = ds_final.isel(time=slice(0, None, step))

        # Summary logging (still lazy)
        self.logger.info("HRRR HOURLY data extraction summary:")
        self.logger.info("  Temporal resolution: HOURLY")
        self.logger.info(f"  Successful hours: {successful_hours}")
        self.logger.info(f"  Failed hours: {failed_hours}")
        self.logger.info(f"  Dimensions: {dict(ds_final.dims)}")
        self.logger.info(f"  Variables: {list(ds_final.data_vars)}")
        if "time" in ds_final.dims:
            self.logger.info(f"  Time steps: {ds_final.sizes['time']}")
            self.logger.info(
                f"  Time range: {ds_final.time.min().values} "
                f"to {ds_final.time.max().values}"
            )

        # Metadata
        ds_final.attrs["source"] = "NOAA HRRR (High-Resolution Rapid Refresh)"
        ds_final.attrs["source_url"] = "s3://hrrrzarr"
        ds_final.attrs["spatial_resolution"] = "3km"
        ds_final.attrs["temporal_resolution"] = "hourly"
        ds_final.attrs["data_type"] = "analysis"  # F00 analyses
        ds_final.attrs["bbox"] = str(self.bbox)
        ds_final.attrs["successful_hours"] = successful_hours
        ds_final.attrs["failed_hours"] = failed_hours

        # Output
        output_dir.mkdir(parents=True, exist_ok=True)
        domain_name = self.config.get("DOMAIN_NAME", "domain")
        start_str = self.start_date.strftime("%Y%m%d")
        end_str = self.end_date.strftime("%Y%m%d")
        output_file = output_dir / f"{domain_name}_HRRR_hourly_{start_str}-{end_str}.nc"

        self.logger.info(f"Saving HRRR hourly data to: {output_file}")
        # This is where the actual remote I/O happens
        ds_final.to_netcdf(output_file)

        self.logger.info(f"✓ HRRR hourly data download complete: {output_file}")
        self.logger.info(f"  Final dataset has {len(ds_final.time)} hourly time steps")
        return output_file


    def _derive_time_slice_from_experiment(self):
        """
        Fallback for GLDAS: build a time_slice from EXPERIMENT_TIME_START/END
        when time_slice is not explicitly provided in the config.
        """
        start_str = self.config.get("EXPERIMENT_TIME_START")
        end_str   = self.config.get("EXPERIMENT_TIME_END")

        if not start_str or not end_str:
            raise ValueError(
                "Cannot derive GLDAS time_slice: "
                "EXPERIMENT_TIME_START and EXPERIMENT_TIME_END must be set in config."
            )

        # Your config uses 'YYYY-MM-DD HH:MM'
        fmt = "%Y-%m-%d %H:%M"
        start_dt = datetime.strptime(start_str.strip("'\""), fmt)
        end_dt   = datetime.strptime(end_str.strip("'\""), fmt)

        # Common, safe choice: ISO-8601 strings; tweak if GLDAS code expects
        # a different format or a list instead of a dict
        return {
            "start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end":   end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    
    def _download_conus404(self, output_dir: Path) -> Path:
        """
        Download CONUS404 data from HyTEST via intake catalog or direct S3 Zarr.

        Preferred: HyTEST intake catalog using the CONUS404 subcatalog, which
        exposes OSN (Open Storage Network) datasets that do not require auth.

        Parameters
        ----------
        output_dir : Path
            Directory to save downloaded data

        Returns
        -------
        Path
            Path to the saved NetCDF file
        """
        self.logger.info("Downloading CONUS404 data from AWS S3 Zarr (HyTEST)")
        self.logger.info(f"  Bounding box: {self.bbox}")
        self.logger.info(f"  Time period: {self.start_date} to {self.end_date}")

        try:
            # Try to use intake catalog first (preferred method)
            try:
                import intake

                # HyTEST catalog URL
                catalog_url = (
                    "https://raw.githubusercontent.com/hytest-org/hytest/main/"
                    "dataset_catalog/hytest_intake_catalog.yml"
                )
                self.logger.info("  Loading HyTEST intake catalog...")
                hytest_cat = intake.open_catalog(catalog_url)

                # Drill into the CONUS404 subcatalog
                if "conus404-catalog" not in list(hytest_cat):
                    available_top = list(hytest_cat)
                    msg = (
                        "conus404-catalog not found in HyTEST catalog. "
                        f"Available top-level entries: {available_top}"
                    )
                    self.logger.error("  " + msg)
                    raise KeyError(msg)

                conus404_cat = hytest_cat["conus404-catalog"]

                # Use hourly data by default (can be configured)
                temporal_res = self.config.get("CONUS404_TEMPORAL_RES", "hourly")

                if temporal_res == "daily":
                    dataset_name = "conus404-daily-osn"
                else:
                    # OSN = Open Storage Network (no auth needed)
                    dataset_name = "conus404-hourly-osn"

                self.logger.info(f"  Opening {dataset_name} dataset from conus404-catalog...")

                if dataset_name not in list(conus404_cat):
                    available_ds = list(conus404_cat)
                    msg = (
                        f"{dataset_name} not found in HyTEST conus404-catalog. "
                        f"Available datasets: {available_ds}"
                    )
                    self.logger.error("  " + msg)
                    raise KeyError(msg)

                ds = conus404_cat[dataset_name].to_dask()

            except ImportError:
                # Fallback to direct S3 access if intake not available
                self.logger.info("  intake-xarray not available, using direct S3 access...")

                temporal_res = self.config.get("CONUS404_TEMPORAL_RES", "hourly")
                if temporal_res == "daily":
                    zarr_path = "hytest/conus404/conus404_daily.zarr"
                else:
                    zarr_path = "hytest/conus404/conus404_hourly.zarr"

                # Open with anonymous access via OSN/S3-compatible endpoint
                store = s3fs.S3Map(zarr_path, s3=self.fs)
                ds = xr.open_zarr(store, consolidated=True)

            self.logger.info("  Successfully opened CONUS404 dataset")
            self.logger.info(f"  Available variables: {list(ds.data_vars)[:10]}...")  # Show first 10

            # Subset by bounding box
            # CONUS404 uses 'y' and 'x' coordinates (projection)
            # We need to work with lat/lon if available
            self.logger.info("  Subsetting spatial domain...")

            self.logger.info(f"  Subsetting spatial domain...")

            if "lat" in ds.coords and "lon" in ds.coords:
                # Use lat/lon subsetting with a pre-computed boolean mask
                self.logger.info("  Using lat/lon coordinates for subsetting...")

                lat = ds["lat"]
                lon = ds["lon"]

                # Build boolean mask as dask array, then compute to NumPy
                self.logger.info("  Computing spatial mask for bounding box...")
                mask = (
                    (lat >= self.bbox["lat_min"])
                    & (lat <= self.bbox["lat_max"])
                    & (lon >= self.bbox["lon_min"])
                    & (lon <= self.bbox["lon_max"])
                ).compute()

                # Now apply mask with drop=True
                ds_subset = ds.where(mask, drop=True)

            else:
                # Use x/y subsetting (may need adjustment)
                self.logger.warning("  Using x/y coordinates - results may need verification")
                ds_subset = ds

            # Subset by time
            self.logger.info("  Subsetting temporal domain...")
            ds_subset = ds_subset.sel(time=slice(self.start_date, self.end_date))

            # Select key meteorological variables for hydrological forcing
            key_vars = [
                "T2",        # 2m temperature
                "Q2",        # 2m mixing ratio
                "PSFC",      # Surface pressure
                "U10",       # 10m U wind
                "V10",       # 10m V wind
                "GLW",       # Downward longwave at surface
                "SWDOWN",    # Downward shortwave at surface
                "RAINRATE",  # Precipitation rate (or RAINNC if available)
            ]

            # Filter to only variables that exist
            available_vars = [var for var in key_vars if var in ds_subset.data_vars]

            if not available_vars:
                self.logger.warning("  Key variables not found, using all available variables")
                ds_final = ds_subset
            else:
                ds_final = ds_subset[available_vars]
                self.logger.info(f"  Selected variables: {available_vars}")

            # Load the data (this triggers the actual download)
            self.logger.info(
                "  Loading data from cloud storage (this may take several minutes)..."
            )
            ds_final = ds_final.load()

            # Log data summary
            self.logger.info("CONUS404 data extraction summary:")
            self.logger.info(f"  Dimensions: {dict(ds_final.dims)}")
            self.logger.info(f"  Variables: {list(ds_final.data_vars)}")
            self.logger.info(f"  Time steps: {len(ds_final.time)}")

            # Add metadata
            ds_final.attrs["source"] = "CONUS404 (NCAR/USGS)"
            ds_final.attrs["source_url"] = "s3://hytest/conus404"
            ds_final.attrs["resolution"] = "4km"
            ds_final.attrs["model"] = "WRF 3.9.1.1"
            ds_final.attrs["downloaded_by"] = "SYMFLUENCE cloud_data_utils"
            ds_final.attrs["download_date"] = pd.Timestamp.now().isoformat()
            ds_final.attrs["bbox"] = str(self.bbox)

            # Save to NetCDF
            output_dir.mkdir(parents=True, exist_ok=True)
            domain_name = self.config.get("DOMAIN_NAME", "domain")
            output_file = (
                output_dir
                / f"{domain_name}_CONUS404_{self.start_date.year}-{self.end_date.year}.nc"
            )

            self.logger.info(f"Saving CONUS404 data to: {output_file}")
            ds_final.to_netcdf(output_file)

            self.logger.info(f"✓ CONUS404 data download complete: {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(f"Error downloading CONUS404 data: {str(e)}")
            raise

    # ------------------------------------------------------------------
    # Copernicus DEM GLO-30 (30 m) via AWS COGs
    # ------------------------------------------------------------------
    def _copdem30_tile_names_for_bbox(self, bbox):
        """
        Given bbox = (lon_min, lat_min, lon_max, lat_max) in EPSG:4326,
        return a list of Copernicus DEM tile base names like:
          Copernicus_DSM_COG_10_N40_00_W111_00_DEM
        """
        lon_min, lat_min, lon_max, lat_max = bbox

        lon_start = math.floor(lon_min)
        lon_end   = math.floor(lon_max - 1e-9)
        lat_start = math.floor(lat_min)
        lat_end   = math.floor(lat_max - 1e-9)

        tile_names = []
        for lat in range(lat_start, lat_end + 1):
            for lon in range(lon_start, lon_end + 1):
                ns = "N" if lat >= 0 else "S"
                ew = "E" if lon >= 0 else "W"
                lat_deg = abs(int(lat))
                lon_deg = abs(int(lon))
                name = f"Copernicus_DSM_COG_10_{ns}{lat_deg:02d}_00_{ew}{lon_deg:03d}_00_DEM"
                tile_names.append(name)

        return tile_names

    def download_copernicus_dem30(self) -> Path:
        """
        Download + subset Copernicus DEM GLO-30 (30 m) for the domain bbox
        and write a clipped GeoTIFF to:
          domain_dir / attributes / elevation / <domain>_copdem30m.tif
        """
        elev_dir = self._attribute_dir("elevation")
        bbox = self.bbox  # (lon_min, lat_min, lon_max, lat_max)

        # AWS bucket & tile naming (public, no-sign-request)
        bucket = "copernicus-dem-30m"
        tile_names = self._copdem30_tile_names_for_bbox(bbox)
        self.logger.info(f"Copernicus DEM: tiles for bbox {bbox}: {tile_names}")

        # Open tiles and mosaic
        sources = []
        for name in tile_names:
            key = f"{name}/{name}.tif"
            url = f"/vsis3/{bucket}/{key}"  # GDAL/rasterio-style vsis3 path
            try:
                self.logger.info(f"Trying Copernicus DEM tile: {url}")
                src = rasterio.open(
                    url,
                    sharing=False,
                    # use environment var AWS_NO_SIGN_REQUEST=YES at runtime
                )
                sources.append(src)
            except Exception as e:
                self.logger.warning(f"Could not open Copernicus DEM tile {url}: {e}")
                continue

        if not sources:
            raise RuntimeError("No Copernicus DEM tiles could be opened for bbox; "
                               "check bbox and tile naming.")

        mosaic, transform = rio_merge(sources)

        # Write clipped raster to elevation directory
        out_path = elev_dir / 'dem' / f"domain_{self.config.get('DOMAIN_NAME','domain')}_elv.tif"
        meta = sources[0].meta.copy()
        meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "count": 1,
            }
        )

        self.logger.info(f"Writing Copernicus DEM mosaic to {out_path}")
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mosaic[0, :, :], 1)

        # Close sources
        for src in sources:
            src.close()

        return out_path


    # ------------------------------------------------------------------
    # NASADEM / SRTM (local tiles) → attributes/elevation
    # ------------------------------------------------------------------
    def download_nasadem_local(self) -> Path:
        """
        Import NASADEM/SRTM tiles from a *local* tile directory (populated
        separately using Earthdata credentials), mosaic for the bbox, and
        write a clipped GeoTIFF into attributes/elevation.

        Config key:
          NASADEM_LOCAL_DIR: directory with NASADEM_HGT_*.zip or *_hgt files
        """
        elev_dir = self._attribute_dir("elevation")
        tiles_dir = Path(self.config["NASADEM_LOCAL_DIR"])
        if not tiles_dir.exists():
            raise FileNotFoundError(f"NASADEM_LOCAL_DIR does not exist: {tiles_dir}")

        # Get bounding box coordinates
        lon_min = self.bbox['lon_min']
        lat_min = self.bbox['lat_min']
        lon_max = self.bbox['lon_max']
        lat_max = self.bbox['lat_max']

        # Figure out which integer-degree tiles we need
        lon_start = math.floor(lon_min)
        lon_end   = math.floor(lon_max - 1e-9)
        lat_start = math.floor(lat_min)
        lat_end   = math.floor(lat_max - 1e-9)

        rasters = []
        for lat in range(lat_start, lat_end + 1):
            for lon in range(lon_start, lon_end + 1):
                ns = "n" if lat >= 0 else "s"
                ew = "e" if lon >= 0 else "w"
                lat_deg = abs(int(lat))
                lon_deg = abs(int(lon))

                base = f"NASADEM_HGT_{ns}{lat_deg:02d}{ew}{lon_deg:03d}"
                # You might have unzipped .hgt or pre-converted GeoTIFF
                candidates = list(tiles_dir.glob(base + "*.tif")) + \
                             list(tiles_dir.glob(base + "*.hgt"))

                if not candidates:
                    self.logger.warning(f"No local NASADEM tile found matching {base}* in {tiles_dir}")
                    continue

                src_path = candidates[0]
                self.logger.info(f"Using NASADEM tile {src_path}")
                src = rasterio.open(src_path)
                rasters.append(src)

        if not rasters:
            raise RuntimeError("No local NASADEM tiles found for bbox; check NASADEM_LOCAL_DIR.")

        mosaic, transform = rio_merge(rasters)
        out_path = elev_dir / f"{self.config.get('DOMAIN_NAME','domain')}_nasadem30m.tif"

        meta = rasters[0].meta.copy()
        meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "count": mosaic.shape[0],
            }
        )

        self.logger.info(f"Writing NASADEM mosaic to {out_path}")
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mosaic)

        for r in rasters:
            r.close()

        return out_path

    # ------------------------------------------------------------------
    # Copernicus DEM GLO-30 (cloud) → attributes/elevation
    # ------------------------------------------------------------------
    def download_copernicus_dem(self) -> Path:
        """
        Download Copernicus DEM GLO-30 from AWS S3 (30m global DEM).
        
        No registration required. Cloud Optimized GeoTIFF (COG) format.
        Available on AWS: s3://copernicus-dem-30m/
        
        Resolution: 1 arc-second (~30m)
        Coverage: Global (land areas)
        
        Returns
        -------
        Path
            Path to the mosaicked and clipped Copernicus DEM GeoTIFF
        """
        elev_dir = self._attribute_dir("elevation")
        
        # Get bounding box
        lon_min = self.bbox['lon_min']
        lat_min = self.bbox['lat_min']
        lon_max = self.bbox['lon_max']
        lat_max = self.bbox['lat_max']
        
        # Normalize bbox (ensure min < max)
        if lon_min > lon_max:
            self.logger.warning(f"Swapping lon_min ({lon_min}) and lon_max ({lon_max}) - they were reversed")
            lon_min, lon_max = lon_max, lon_min
        if lat_min > lat_max:
            self.logger.warning(f"Swapping lat_min ({lat_min}) and lat_max ({lat_max}) - they were reversed")
            lat_min, lat_max = lat_max, lat_min
        
        self.logger.info(f"Downloading Copernicus DEM for bbox: [{lon_min}, {lat_min}, {lon_max}, {lat_max}]")
        
        # Figure out which 1-degree tiles we need
        lon_start = math.floor(lon_min)
        lon_end = math.floor(lon_max)
        lat_start = math.floor(lat_min)
        lat_end = math.floor(lat_max)
        
        # Initialize S3 filesystem
        import s3fs
        s3 = s3fs.S3FileSystem(anon=True)
        
        rasters = []
        temp_files = []
        
        for lat in range(lat_start, lat_end + 1):
            for lon in range(lon_start, lon_end + 1):
                # Copernicus tile naming: Copernicus_DSM_COG_10_N44_00_W112_00_DEM
                ns = "N" if lat >= 0 else "S"
                ew = "E" if lon >= 0 else "W"
                lat_deg = abs(int(lat))
                lon_deg = abs(int(lon))
                
                tile_name = f"Copernicus_DSM_COG_10_{ns}{lat_deg:02d}_00_{ew}{lon_deg:03d}_00_DEM"
                s3_path = f"copernicus-dem-30m/{tile_name}/{tile_name}.tif"
                
                self.logger.info(f"Fetching tile: {tile_name}")
                
                try:
                    # Download tile to temporary file
                    temp_file = elev_dir / f"temp_{tile_name}.tif"
                    temp_files.append(temp_file)
                    
                    # Read from S3 and write locally
                    with s3.open(s3_path, 'rb') as s3_file:
                        with open(temp_file, 'wb') as local_file:
                            local_file.write(s3_file.read())
                    
                    # Open with rasterio
                    src = rasterio.open(temp_file)
                    rasters.append(src)
                    
                except Exception as e:
                    self.logger.warning(f"Could not fetch tile {tile_name}: {e}")
                    continue
        
        if not rasters:
            raise RuntimeError(f"No Copernicus DEM tiles found for bbox: {self.bbox}")
        
        # Mosaic tiles
        self.logger.info(f"Mosaicking {len(rasters)} Copernicus DEM tiles")
        mosaic, transform = rio_merge(rasters)
        
        # Clip to exact bbox
        from rasterio.mask import mask
        from shapely.geometry import box
        
        bbox_geom = box(lon_min, lat_min, lon_max, lat_max)
        
        out_path = elev_dir / 'dem' / f"domain_{self.config.get('DOMAIN_NAME','domain')}_elv.tif"
        
        meta = rasters[0].meta.copy()
        meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "count": mosaic.shape[0],
            "compress": "lzw"
        })
        
        self.logger.info(f"Writing Copernicus DEM to {out_path}")
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mosaic)
        
        # Cleanup
        for r in rasters:
            r.close()
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()
        
        self.logger.info(f"✓ Copernicus DEM downloaded: {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # FABDEM (cloud) → attributes/elevation
    # ------------------------------------------------------------------
    def download_fabdem(self) -> Path:
        """
        Download FABDEM (Forest And Buildings removed DEM) based on Copernicus.
        
        FABDEM removes vegetation and building bias from Copernicus DEM,
        making it more suitable for hydrological applications.
        
        No registration required. Available via University of Bristol.
        
        Resolution: 1 arc-second (~30m)
        Coverage: Global (60°S to 80°N)
        
        Returns
        -------
        Path
            Path to the mosaicked and clipped FABDEM GeoTIFF
        """
        elev_dir = self._attribute_dir("elevation")
        
        # Get bounding box
        lon_min = self.bbox['lon_min']
        lat_min = self.bbox['lat_min']
        lon_max = self.bbox['lon_max']
        lat_max = self.bbox['lat_max']
        
        # Normalize bbox (ensure min < max)
        if lon_min > lon_max:
            self.logger.warning(f"Swapping lon_min ({lon_min}) and lon_max ({lon_max}) - they were reversed")
            lon_min, lon_max = lon_max, lon_min
        if lat_min > lat_max:
            self.logger.warning(f"Swapping lat_min ({lat_min}) and lat_max ({lat_max}) - they were reversed")
            lat_min, lat_max = lat_max, lat_min
        
        self.logger.info(f"Downloading FABDEM for bbox: [{lon_min}, {lat_min}, {lon_max}, {lat_max}]")
        
        # FABDEM tiles are organized by 1-degree tiles
        lon_start = math.floor(lon_min)
        lon_end = math.floor(lon_max)
        lat_start = math.floor(lat_min)
        lat_end = math.floor(lat_max)
        
        # FABDEM base URL (adjust based on actual hosting)
        # Note: FABDEM is hosted on multiple platforms - using the public HTTP access
        base_url = "https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn"
        
        rasters = []
        temp_files = []
        
        for lat in range(lat_start, lat_end + 1):
            for lon in range(lon_start, lon_end + 1):
                # FABDEM tile naming convention
                ns = "N" if lat >= 0 else "S"
                ew = "E" if lon >= 0 else "W"
                lat_deg = abs(int(lat))
                lon_deg = abs(int(lon))
                
                # FABDEM uses format like: N44W112_FABDEM_V1-2.tif
                tile_name = f"{ns}{lat_deg:02d}{ew}{lon_deg:03d}_FABDEM_V1-2.tif"
                
                # Try multiple possible URLs/paths
                possible_urls = [
                    f"{base_url}/{tile_name}",
                    f"https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn/FABDEM_V1-2/{tile_name}",
                    # Add more URL patterns as needed
                ]
                
                downloaded = False
                for url in possible_urls:
                    try:
                        self.logger.info(f"Attempting to download FABDEM tile: {tile_name}")
                        
                        response = requests.get(url, stream=True, timeout=300)
                        if response.status_code == 200:
                            temp_file = elev_dir / f"temp_{tile_name}"
                            temp_files.append(temp_file)
                            
                            with open(temp_file, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            
                            src = rasterio.open(temp_file)
                            rasters.append(src)
                            downloaded = True
                            self.logger.info(f"✓ Downloaded {tile_name}")
                            break
                            
                    except Exception as e:
                        continue
                
                if not downloaded:
                    self.logger.warning(f"Could not download FABDEM tile {tile_name} - trying next tile")
        
        if not rasters:
            # Fallback: Try accessing via OpenTopography API or alternative source
            self.logger.warning("Direct FABDEM download failed. Attempting alternative access method...")
            return self._download_fabdem_alternative()
        
        # Mosaic tiles
        self.logger.info(f"Mosaicking {len(rasters)} FABDEM tiles")
        mosaic, transform = rio_merge(rasters)
        
        out_path = elev_dir / f"{self.config.get('DOMAIN_NAME','domain')}_fabdem30m.tif"
        
        meta = rasters[0].meta.copy()
        meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "count": mosaic.shape[0],
            "compress": "lzw"
        })
        
        self.logger.info(f"Writing FABDEM to {out_path}")
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mosaic)
        
        # Cleanup
        for r in rasters:
            r.close()
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()
        
        self.logger.info(f"✓ FABDEM downloaded: {out_path}")
        return out_path

    def _era5_to_summa_schema(self, ds_chunk):
        """
        Convert an ERA5 ARCO-ERA5 chunk (ARCO variable names) into the
        SUMMA-style forcing schema expected by the agnostic pre-processor.

        Input variables (if present):
            2m_temperature                      [K]
            2m_dewpoint_temperature            [K]
            10m_u_component_of_wind            [m s-1]
            10m_v_component_of_wind            [m s-1]
            surface_pressure                   [Pa]
            total_precipitation                [m]   (accumulated)
            surface_solar_radiation_downwards  [J m-2] (accumulated)
            surface_thermal_radiation_downwards[J m-2] (accumulated)

        Output variables:
            airpres   [Pa]
            LWRadAtm  [W m-2]
            SWRadAtm  [W m-2]
            pptrate   [m s-1]
            airtemp   [K]
            spechum   [kg kg-1]
            windspd   [m s-1]

        Notes
        -----
        * For accumulated variables (precipitation + radiation) we take a
          temporal derivative using finite differences and drop the first
          time step so that all variables share the same time coordinate.
        * Specific humidity is derived from dew point temperature and surface
          pressure assuming saturation at the dew point.
        """
        if "time" not in ds_chunk.dims or ds_chunk.sizes["time"] < 2:
            self.logger.warning(
                "ERA5 chunk has fewer than 2 time steps; skipping SUMMA "
                "conversion and returning original chunk."
            )
            return ds_chunk

        # Ensure time is sorted
        ds_chunk = ds_chunk.sortby("time")

        # We'll ultimately drop the first time step (needed for finite
        # differences of accumulated fields).
        ds_base = ds_chunk.isel(time=slice(1, None))

        # Preserve coords (time/lat/lon, etc.) after dropping first step
        out = xr.Dataset(coords={c: ds_base.coords[c] for c in ds_base.coords})

        # ------------------------------------------------------------------
        # Simple renames / direct copies
        # ------------------------------------------------------------------
        if "surface_pressure" in ds_base:
            airpres = ds_base["surface_pressure"].astype("float32")
            airpres.name = "airpres"
            airpres.attrs.update(
                {
                    "units": "Pa",
                    "long_name": "air pressure",
                    "standard_name": "air_pressure",
                }
            )
            out["airpres"] = airpres

        if "2m_temperature" in ds_base:
            airtemp = ds_base["2m_temperature"].astype("float32")
            airtemp.name = "airtemp"
            airtemp.attrs.update(
                {
                    "units": "K",
                    "long_name": "air temperature",
                    "standard_name": "air_temperature",
                }
            )
            out["airtemp"] = airtemp

        # ------------------------------------------------------------------
        # Wind speed from U/V components
        # ------------------------------------------------------------------
        if (
            "10m_u_component_of_wind" in ds_base
            and "10m_v_component_of_wind" in ds_base
        ):
            u = ds_base["10m_u_component_of_wind"]
            v = ds_base["10m_v_component_of_wind"]
            windspd = np.sqrt(u ** 2 + v ** 2).astype("float32")
            windspd.name = "windspd"
            windspd.attrs.update(
                {
                    "units": "m s-1",
                    "long_name": "wind speed",
                    "standard_name": "wind_speed",
                }
            )
            out["windspd"] = windspd

        # ------------------------------------------------------------------
        # Specific humidity from dew point + surface pressure
        # ------------------------------------------------------------------
        if "2m_dewpoint_temperature" in ds_base and "surface_pressure" in ds_base:
            Td = ds_base["2m_dewpoint_temperature"]        # K
            p = ds_base["surface_pressure"]                # Pa

            # Convert to Celsius for Magnus formula
            Td_C = Td - 273.15

            # Saturation vapor pressure (Magnus, over water)
            # es in hPa, then convert to Pa
            es_hPa = 6.112 * np.exp((17.67 * Td_C) / (Td_C + 243.5))
            es = es_hPa * 100.0

            eps = 0.622
            # Guard against p - es <= 0
            denom = xr.where((p - es) <= 1.0, 1.0, p - es)
            r = eps * es / denom             # mixing ratio
            q = (r / (1.0 + r)).astype("float32")

            spechum = q
            spechum.name = "spechum"
            spechum.attrs.update(
                {
                    "units": "kg kg-1",
                    "long_name": "specific humidity",
                    "standard_name": "specific_humidity",
                }
            )
            out["spechum"] = spechum

        # ------------------------------------------------------------------
        # Helper for accumulated → rate conversion
        # ------------------------------------------------------------------
        time = ds_chunk["time"]
        dt = (time.diff("time") / np.timedelta64(1, "s")).astype("float32")

        def _accum_to_rate(var_name, out_name, units, long_name, standard_name):
            if var_name not in ds_chunk:
                return

            accum = ds_chunk[var_name]
            diff = accum.diff("time")
            rate = (diff / dt).astype("float32")
            # After diff, the time coord is aligned with ds_base.time
            rate.name = out_name
            rate.attrs.update(
                {
                    "units": units,
                    "long_name": long_name,
                    "standard_name": standard_name,
                }
            )
            out[out_name] = rate

        # Precipitation: m -> m s-1
        _accum_to_rate(
            "total_precipitation",
            "pptrate",
            "m s-1",
            "precipitation rate",
            "precipitation_rate",
        )

        # Shortwave radiation: J m-2 -> W m-2
        _accum_to_rate(
            "surface_solar_radiation_downwards",
            "SWRadAtm",
            "W m-2",
            "surface downwelling shortwave radiation",
            "surface_downwelling_shortwave_flux_in_air",
        )

        # Longwave radiation: J m-2 -> W m-2
        _accum_to_rate(
            "surface_thermal_radiation_downwards",
            "LWRadAtm",
            "W m-2",
            "surface downwelling longwave radiation",
            "surface_downwelling_longwave_flux_in_air",
        )

        # Ensure required variables exist (warn if missing)
        required = [
            "airpres",
            "LWRadAtm",
            "SWRadAtm",
            "pptrate",
            "airtemp",
            "spechum",
            "windspd",
        ]
        missing = [v for v in required if v not in out.data_vars]
        if missing:
            self.logger.warning(
                f"ERA5 SUMMA conversion: missing variables in output: {missing}"
            )

        return out


    def _download_fabdem_alternative(self) -> Path:
        """
        Alternative FABDEM download using OpenTopography or COG access.
        
        This is a fallback if direct Bristol download fails.
        """
        elev_dir = self._attribute_dir("elevation")
        
        # Get bounding box
        lon_min = self.bbox['lon_min']
        lat_min = self.bbox['lat_min']
        lon_max = self.bbox['lon_max']
        lat_max = self.bbox['lat_max']
        
        # Normalize bbox (ensure min < max)
        if lon_min > lon_max:
            lon_min, lon_max = lon_max, lon_min
        if lat_min > lat_max:
            lat_min, lat_max = lat_max, lat_min
        
        self.logger.info("Attempting FABDEM download via OpenTopography API")
        
        # OpenTopography API endpoint for FABDEM
        api_url = "https://portal.opentopography.org/API/globaldem"
        
        params = {
            'demtype': 'FABDEM',
            'south': lat_min,
            'north': lat_max,
            'west': lon_min,
            'east': lon_max,
            'outputFormat': 'GTiff',
            'API_Key': self.config.get('OPENTOPOGRAPHY_API_KEY', '')  # Optional
        }
        
        out_path = elev_dir / f"{self.config.get('DOMAIN_NAME','domain')}_fabdem30m.tif"
        
        try:
            response = requests.get(api_url, params=params, stream=True, timeout=600)
            response.raise_for_status()
            
            with open(out_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.logger.info(f"✓ FABDEM downloaded via OpenTopography: {out_path}")
            return out_path
            
        except Exception as e:
            self.logger.error(f"FABDEM alternative download failed: {e}")
            raise RuntimeError(
                "Could not download FABDEM. Try setting DEM_SOURCE to 'copernicus' "
                "or 'merit_hydro', or provide an OPENTOPOGRAPHY_API_KEY in config."
            )

    def download_global_soilclasses(self) -> Path:
        """
        Download / subset the USDA global soil texture class map used by MAF
        and save to:
            domain_dir / attributes / soilclasses / <domain>_soilclasses.tif

        Source:
        HydroShare resource:
            1361509511e44adfba814f6950c6e742
        File inside resource:
            usda_mode_soilclass_250m_ll.tif

        We assume this resource is public and accessible via a direct file URL.
        No HydroShare credentials are required.

        Behaviour
        ---------
        - Read the global map via GDAL's /vsicurl/ HTTP access.
        - Crop to the domain bbox (EPSG:4326).
        - Write a single-band GeoTIFF with the cropped soil classes.
        """
        soil_dir = self._attribute_dir("soilclass")

        # Domain bbox
        lon_min = float(self.bbox["lon_min"])
        lon_max = float(self.bbox["lon_max"])
        lat_min = float(self.bbox["lat_min"])
        lat_max = float(self.bbox["lat_max"])

        # Normalise in case bbox is reversed
        if lon_min > lon_max:
            self.logger.info(
                f"Swapping lon_min ({lon_min}) and lon_max ({lon_max}) - they were reversed"
            )
            lon_min, lon_max = lon_max, lon_min
        if lat_min > lat_max:
            self.logger.info(
                f"Swapping lat_min ({lat_min}) and lat_max ({lat_max}) - they were reversed"
            )
            lat_min, lat_max = lat_max, lat_min

        self.logger.info(
            "Preparing global USDA soilclass map for bbox "
            f"lon [{lon_min}, {lon_max}], lat [{lat_min}, {lat_max}]"
        )

        # Hard-coded HydroShare resource + file used by MAF
        resource_id = "1361509511e44adfba814f6950c6e742"
        filename = "usda_mode_soilclass_250m_ll.tif"

        # Direct public file URL (no auth), accessed via GDAL /vsicurl/
        base_url = (
            f"https://www.hydroshare.org/resource/{resource_id}/data/contents"
        )
        hs_url = f"/vsicurl/{base_url}/{filename}"

        from rasterio.windows import from_bounds

        self.logger.info(f"Opening USDA soilclass global map from {hs_url}")

        with rasterio.open(hs_url) as src:
            if src.crs is None or src.crs.to_epsg() != 4326:
                self.logger.warning(
                    "USDA soilclass global map is not EPSG:4326; "
                    "cropping assumes lat/lon in EPSG:4326."
                )

            # Window over our bbox
            window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
            out_transform = src.window_transform(window)
            data = src.read(1, window=window)

            meta = src.meta.copy()
            meta.update(
                {
                    "driver": "GTiff",
                    "height": data.shape[0],
                    "width": data.shape[1],
                    "transform": out_transform,
                    "count": 1,
                }
            )

        out_path = soil_dir / f"domain_{self.config.get('DOMAIN_NAME','domain')}_soil_classes.tif"
        self.logger.info(f"Writing cropped USDA soilclass raster to {out_path}")

        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data, 1)

        self.logger.info(
            f"✓ USDA global soilclass map cropped & saved: {out_path}"
        )
        return out_path



    def _crop_with_gdal(self, global_vrt: Path, lon_min: float, lat_min: float,
                       lon_max: float, lat_max: float, out_path: Path):
        """
        Crop global VRT to domain extent using GDAL Translate.
        Follows the CWARHM approach.
        """
        try:
            from osgeo import gdal
            gdal.UseExceptions()
        except ImportError:
            raise ImportError(
                "GDAL Python bindings required. Install with: "
                "conda install -c conda-forge gdal  OR  pip install gdal"
            )
        
        self.logger.info(f"Cropping with GDAL...")
        
        # GDAL projWin format: [ulx, uly, lrx, lry] = [lon_min, lat_max, lon_max, lat_min]
        bbox = (lon_min, lat_max, lon_max, lat_min)
        
        translate_options = gdal.TranslateOptions(
            format='GTiff',
            projWin=bbox,
            creationOptions=['COMPRESS=DEFLATE', 'TILED=YES', 'BIGTIFF=IF_SAFER']
        )
        
        ds = gdal.Translate(str(out_path), str(global_vrt), options=translate_options)
        
        if ds is None:
            raise RuntimeError("GDAL Translate failed")
        
        ds = None  # Close dataset
        
        # Verify
        with rasterio.open(out_path) as src:
            self.logger.info(f"Cropped shape: {src.shape}, CRS: {src.crs}")



def get_aorc_variable_mapping() -> Dict[str, str]:
    """
    Get mapping from AORC variable names to SUMMA/standard names.
    
    Returns
    -------
    dict
        Mapping of AORC variables to standard names
    """
    return {
        'APCP_surface': 'pptrate',           # Precipitation rate [kg/m2 or mm]
        'TMP_2maboveground': 'airtemp',      # Air temperature [K]
        'SPFH_2maboveground': 'spechum',     # Specific humidity [kg/kg]
        'PRES_surface': 'airpres',           # Surface pressure [Pa]
        'DLWRF_surface': 'LWRadAtm',         # Downward longwave radiation [W/m2]
        'DSWRF_surface': 'SWRadAtm',         # Downward shortwave radiation [W/m2]
        'UGRD_10maboveground': 'wind_u',     # U-component of wind [m/s]
        'VGRD_10maboveground': 'wind_v'      # V-component of wind [m/s]
    }


def get_era5_variable_mapping() -> Dict[str, str]:
    """
    Get mapping from ERA5 variable names to SUMMA/standard names.
    
    Returns
    -------
    dict
        Mapping of ERA5 variables to standard names
    """
    return {
        't2m': 'airtemp',      # 2m temperature [K]
        'u10': 'wind_u',       # 10m U wind component [m/s]
        'v10': 'wind_v',       # 10m V wind component [m/s]
        'sp': 'airpres',       # Surface pressure [Pa]
        'd2m': 'dewpoint',     # 2m dewpoint temperature [K]
        'q': 'spechum',        # Specific humidity [kg/kg]
        'tp': 'pptrate',       # Total precipitation [m]
        'ssrd': 'SWRadAtm',    # Surface solar radiation downwards [J/m2]
        'strd': 'LWRadAtm',    # Surface thermal radiation downwards [J/m2]
    }


def get_emearth_variable_mapping() -> Dict[str, str]:
    """
    Get mapping from EM-Earth variable names to SUMMA/standard names.
    
    Returns
    -------
    dict
        Mapping of EM-Earth variables to standard names
    """
    return {
        "prcp": "pptrate",             # Precipitation [mm/day]
        "prcp_corrected": "pptrate",   # Bias-corrected precipitation [mm/day]
        "tmean": "airtemp",           # Mean air temperature [°C]
        "trange": "temp_range",       # Temperature range [°C]
        "tdew": "dewpoint",           # Dewpoint temperature [°C]
    }


def get_hrrr_variable_mapping() -> Dict[str, str]:
    """
    Get mapping from HRRR variable names to SUMMA/standard names.
    
    Returns
    -------
    dict
        Mapping of HRRR variables to standard names
    """
    return {
        'TMP': 'airtemp',      # Temperature [K]
        'SPFH': 'spechum',     # Specific humidity [kg/kg]
        'PRES': 'airpres',     # Surface pressure [Pa]
        'UGRD': 'wind_u',      # U wind component [m/s]
        'VGRD': 'wind_v',      # V wind component [m/s]
        'DSWRF': 'SWRadAtm',   # Downward shortwave radiation [W/m2]
        'DLWRF': 'LWRadAtm',   # Downward longwave radiation [W/m2]
        'APCP': 'pptrate',     # Accumulated precipitation [kg/m2]
    }


def get_conus404_variable_mapping() -> Dict[str, str]:
    """
    Get mapping from CONUS404 variable names to SUMMA/standard names.
    
    Returns
    -------
    dict
        Mapping of CONUS404 variables to standard names
    """
    return {
        'T2': 'airtemp',       # 2m temperature [K]
        'Q2': 'spechum',       # 2m mixing ratio [kg/kg]
        'PSFC': 'airpres',     # Surface pressure [Pa]
        'U10': 'wind_u',       # 10m U wind component [m/s]
        'V10': 'wind_v',       # 10m V wind component [m/s]
        'GLW': 'LWRadAtm',     # Downward longwave at surface [W/m2]
        'SWDOWN': 'SWRadAtm',  # Downward shortwave at surface [W/m2]
        'RAINRATE': 'pptrate', # Precipitation rate [mm/s]
        'RAINNC': 'pptrate',   # Accumulated grid-scale precipitation [mm]
    }


def check_cloud_access_availability(dataset_name: str, logger) -> bool:
    """
    Check if a dataset is available for cloud access.
    
    Parameters
    ----------
    dataset_name : str
        Name of the forcing dataset
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    bool
        True if dataset supports cloud access
    """
    supported_datasets = ['AORC', 'ERA5', 'EM-EARTH', 'HRRR', 'CONUS404', 'GLDAS', 'NEX-GDDP-CMIP6']
    
    if dataset_name.upper() in supported_datasets:
        logger.info(f"✓ {dataset_name} supports cloud data access")
        return True
    else:
        logger.warning(
            f"✗ {dataset_name} does not support cloud access. "
            f"Supported datasets: {', '.join(supported_datasets)}"
        )
        return False