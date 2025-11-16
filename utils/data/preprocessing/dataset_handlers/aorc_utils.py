# aorc_utils.py

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon

from .base_dataset import BaseDatasetHandler
from .dataset_registry import DatasetRegistry


@DatasetRegistry.register('aorc')
class AORCHandler(BaseDatasetHandler):
    """
    Handler for AORC (Analysis of Record for Calibration) dataset.

    This assumes AORC has been downloaded via CloudForcingDownloader into
    forcing/raw_data/domain_<DOMAIN_NAME>_AORC_<startYear>-<endYear>.nc
    and stored on a regular lat/lon grid with coordinates:
        latitude, longitude
    """

    # ---------- Variable mapping / standardization ----------

    def get_variable_mapping(self) -> Dict[str, str]:
        """
        AORC → SUMMA/standard variable mapping.

        We align to the same standard names used by RDRS/CASR/ERA5 so that
        the easymore remapper can request:
            ['airpres', 'LWRadAtm', 'SWRadAtm', 'pptrate',
             'airtemp', 'spechum', 'windspd']
        """
        return {
            'APCP_surface': 'pptrate',            # Precip / accumulation
            'TMP_2maboveground': 'airtemp',       # 2m air temperature [K]
            'SPFH_2maboveground': 'spechum',      # 2m specific humidity [kg/kg]
            'PRES_surface': 'airpres',            # surface pressure [Pa]
            'DLWRF_surface': 'LWRadAtm',          # longwave down [W/m2]
            'DSWRF_surface': 'SWRadAtm',          # shortwave down [W/m2]
            'UGRD_10maboveground': 'windspd_u',   # 10m U wind [m/s]
            'VGRD_10maboveground': 'windspd_v',   # 10m V wind [m/s]
        }

    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Process AORC dataset:
        - rename variables to standard names
        - derive wind speed from u/v
        - ensure clean attributes
        """
        var_map = self.get_variable_mapping()
        existing = {old: new for old, new in var_map.items() if old in ds.variables}
        ds = ds.rename(existing)

        # ---- Derive wind speed magnitude if components are present ----
        if 'windspd_u' in ds and 'windspd_v' in ds:
            u = ds['windspd_u']
            v = ds['windspd_v']
            windspd = np.sqrt(u**2 + v**2)
            windspd.name = 'windspd'
            windspd.attrs = {
                'units': 'm s-1',
                'long_name': 'wind speed',
                'standard_name': 'wind_speed'
            }
            ds['windspd'] = windspd

        # ---- Set / clean attributes similar to other handlers ----
        if 'airpres' in ds:
            ds['airpres'].attrs = {
                'units': 'Pa',
                'long_name': 'air pressure',
                'standard_name': 'air_pressure'
            }

        if 'airtemp' in ds:
            ds['airtemp'].attrs = {
                'units': 'K',
                'long_name': 'air temperature',
                'standard_name': 'air_temperature'
            }

        if 'pptrate' in ds:
            # remove any inherited encoding-style attrs to avoid xarray conflicts
            ds['pptrate'].attrs.pop('missing_value', None)
            ds['pptrate'].attrs = {
                'units': 'm s-1',
                'long_name': 'precipitation rate',
                'standard_name': 'precipitation_rate'
            }

        if 'spechum' in ds:
            ds['spechum'].attrs = {
                'units': 'kg kg-1',
                'long_name': 'specific humidity',
                'standard_name': 'specific_humidity'
            }

        if 'LWRadAtm' in ds:
            ds['LWRadAtm'].attrs = {
                'units': 'W m-2',
                'long_name': 'downward longwave radiation at the surface',
                'standard_name': 'surface_downwelling_longwave_flux_in_air'
            }

        if 'SWRadAtm' in ds:
            ds['SWRadAtm'].attrs = {
                'units': 'W m-2',
                'long_name': 'downward shortwave radiation at the surface',
                'standard_name': 'surface_downwelling_shortwave_flux_in_air'
            }

        return ds

    # ---------- Coordinate info ----------

    def get_coordinate_names(self) -> Tuple[str, str]:
        """
        AORC cloud path uses latitude/longitude coordinates, same as ERA5. :contentReference[oaicite:4]{index=4}
        """
        return ('latitude', 'longitude')

    # ---------- Merging (standardization) ----------

    def needs_merging(self) -> bool:
        """
        We *do* mark AORC as needing "merging", but it's really a
        standardization step: we take the raw cloud-downloaded file(s),
        apply variable mapping, and write processed NetCDF to forcing/merged_path.

        This hooks into forcingResampler's existing logic that calls
        dataset_handler.merge_forcings() when needs_merging() is True. :contentReference[oaicite:5]{index=5}
        """
        return True

    def merge_forcings(
        self,
        raw_forcing_path: Path,
        merged_forcing_path: Path,
        start_year: int,
        end_year: int,
    ) -> None:
        """
        "Merge" AORC forcings: for each cloud-downloaded file in raw_data,
        apply process_dataset() and save a standardized file to merged_path.

        No temporal slicing or monthly splitting here: we just 1:1 transform.
        """
        self.logger.info("Standardizing AORC forcing files (no temporal merging)")

        merged_forcing_path.mkdir(parents=True, exist_ok=True)

        # CloudForcingDownloader._download_aorc currently writes:
        #   <DOMAIN_NAME>_AORC_<startYear>-<endYear>.nc
        # e.g., paradise_AORC_2000-2002.nc
        # but we also support potential future patterns.
        patterns = [
            f"{self.domain_name}_AORC_*.nc",          # paradise_AORC_2000-2002.nc (current)
            f"domain_{self.domain_name}_AORC_*.nc",   # domain_paradise_AORC_*.nc (future-proof)
            "*AORC*.nc",                              # last-resort catch-all
        ]

        files: list[Path] = []
        for pattern in patterns:
            candidates = sorted(raw_forcing_path.glob(pattern))
            if candidates:
                self.logger.info(
                    f"Found {len(candidates)} AORC file(s) in {raw_forcing_path} "
                    f"with pattern '{pattern}'"
                )
                files = candidates
                break

        if not files:
            self.logger.error(
                f"No AORC files found in {raw_forcing_path} with any of patterns: {patterns}"
            )
            raise FileNotFoundError(
                f"No AORC forcing files found in {raw_forcing_path}"
            )

        for f in files:
            self.logger.info(f"Processing AORC file: {f}")
            try:
                ds = xr.open_dataset(f)
            except Exception as e:
                self.logger.error(f"Error opening AORC file {f}: {e}")
                continue

            try:
                # Dataset-specific standardization
                ds_proc = self.process_dataset(ds)

                # Common BaseDatasetHandler helpers (time encoding, metadata, attrs)
                ds_proc = self.setup_time_encoding(ds_proc)
                ds_proc = self.add_metadata(
                    ds_proc,
                    "AORC data standardized for SUMMA-compatible forcing (SYMFLUENCE)",
                )
                ds_proc = self.clean_variable_attributes(ds_proc)

                # 🔧 Fix: xarray complains if 'missing_value' is left in attrs,
                # because it's treated as an encoding key. Strip it explicitly.
                for var_name in ds_proc.data_vars:
                    if "missing_value" in ds_proc[var_name].attrs:
                        self.logger.debug(
                            f"Removing 'missing_value' attribute from variable '{var_name}'"
                        )
                        ds_proc[var_name].attrs.pop("missing_value", None)

                out_name = merged_forcing_path / f"{f.stem}_processed.nc"
                ds_proc.to_netcdf(out_name)
                self.logger.info(f"Saved processed AORC forcing: {out_name}")
            except Exception as e:
                self.logger.error(f"Error processing AORC dataset from {f}: {e}")
            finally:
                ds.close()

        self.logger.info("AORC forcing standardization completed")

    # ---------- Shapefile for AORC grid ----------

    def create_shapefile(
        self,
        shapefile_path: Path,
        merged_forcing_path: Path,
        dem_path: Path,
        elevation_calculator,
    ) -> Path:
        """
        Create AORC grid shapefile.

        We mirror the ERA5/CASR/RDRS logic:
        - open a processed AORC file in merged_forcing_path
        - build polygons from latitude/longitude
        - compute elevation via provided elevation_calculator
        """
        self.logger.info("Creating AORC grid shapefile")

        output_shapefile = shapefile_path / f"forcing_{self.config['FORCING_DATASET']}.shp"

        # Choose a file to infer grid from
        aorc_files = list(merged_forcing_path.glob('*.nc'))
        if not aorc_files:
            raise FileNotFoundError(f"No AORC processed files found in {merged_forcing_path}")

        aorc_file = aorc_files[0]
        self.logger.info(f"Using AORC file for grid: {aorc_file}")

        with xr.open_dataset(aorc_file) as ds:
            var_lat, var_lon = self.get_coordinate_names()

            # Try coords first, then data_vars, then fail
            if var_lat in ds.coords:
                lat = ds.coords[var_lat].values
            elif var_lat in ds.variables:
                lat = ds[var_lat].values
            else:
                raise KeyError(
                    f"Latitude coordinate '{var_lat}' not found in AORC file {aorc_file}. "
                    f"Available coords: {list(ds.coords)}; variables: {list(ds.data_vars)}"
                )

            if var_lon in ds.coords:
                lon = ds.coords[var_lon].values
            elif var_lon in ds.variables:
                lon = ds[var_lon].values
            else:
                raise KeyError(
                    f"Longitude coordinate '{var_lon}' not found in AORC file {aorc_file}. "
                    f"Available coords: {list(ds.coords)}; variables: {list(ds.data_vars)}"
                )

        # 1D or 2D grid handling
        geometries = []
        ids = []
        lats = []
        lons = []

        if lat.ndim == 1 and lon.ndim == 1:
            # Regular lat/lon grid (ERA5-style) :contentReference[oaicite:10]{index=10}
            half_dlat = abs(lat[1] - lat[0]) / 2 if len(lat) > 1 else 0.005
            half_dlon = abs(lon[1] - lon[0]) / 2 if len(lon) > 1 else 0.005

            for i, center_lon in enumerate(lon):
                for j, center_lat in enumerate(lat):
                    verts = [
                        [float(center_lon) - half_dlon, float(center_lat) - half_dlat],
                        [float(center_lon) - half_dlon, float(center_lat) + half_dlat],
                        [float(center_lon) + half_dlon, float(center_lat) + half_dlat],
                        [float(center_lon) + half_dlon, float(center_lat) - half_dlat],
                        [float(center_lon) - half_dlon, float(center_lat) - half_dlat],
                    ]
                    geometries.append(Polygon(verts))
                    ids.append(i * len(lat) + j)
                    lats.append(float(center_lat))
                    lons.append(float(center_lon))
        else:
            # 2D lat/lon grid (CASR/RDRS-style) :contentReference[oaicite:11]{index=11}
            ny, nx = lat.shape
            total_cells = ny * nx
            self.logger.info(f"AORC grid dimensions (2D): ny={ny}, nx={nx}, total={total_cells}")

            cell_count = 0
            for i in range(ny):
                for j in range(nx):
                    # Corner indices with edge-safe fallbacks
                    lat_corners = [
                        lat[i, j],
                        lat[i, j+1] if j+1 < nx else lat[i, j],
                        lat[i+1, j+1] if i+1 < ny and j+1 < nx else lat[i, j],
                        lat[i+1, j] if i+1 < ny else lat[i, j],
                    ]
                    lon_corners = [
                        lon[i, j],
                        lon[i, j+1] if j+1 < nx else lon[i, j],
                        lon[i+1, j+1] if i+1 < ny and j+1 < nx else lon[i, j],
                        lon[i+1, j] if i+1 < ny else lon[i, j],
                    ]

                    geometries.append(Polygon(zip(lon_corners, lat_corners)))
                    ids.append(i * nx + j)
                    lats.append(float(lat[i, j]))
                    lons.append(float(lon[i, j]))

                    cell_count += 1
                    if cell_count % 5000 == 0 or cell_count == total_cells:
                        self.logger.info(f"Created {cell_count}/{total_cells} AORC grid cells")

        gdf = gpd.GeoDataFrame(
            {
                'geometry': geometries,
                'ID': ids,
                self.config.get('FORCING_SHAPE_LAT_NAME'): lats,
                self.config.get('FORCING_SHAPE_LON_NAME'): lons,
            },
            crs='EPSG:4326',
        )

        # Elevation using the safe helper used by other handlers
        self.logger.info("Calculating elevation values for AORC grid")
        elevations = elevation_calculator(gdf, dem_path, batch_size=50)
        gdf['elev_m'] = elevations

        shapefile_path.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_shapefile)
        self.logger.info(f"AORC shapefile created at {output_shapefile}")

        return output_shapefile
