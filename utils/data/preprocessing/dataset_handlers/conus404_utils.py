# conus404_utils.py

from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon

from .base_dataset import BaseDatasetHandler
from .dataset_registry import DatasetRegistry


@DatasetRegistry.register("conus404")
class CONUS404Handler(BaseDatasetHandler):
    """
    Handler for CONUS404 WRF reanalysis.

    Cloud downloader currently writes a domain file:
        <DOMAIN_NAME>_CONUS404_<startYear>-<endYear>.nc

    Variables (from cloud_downloader) typically include:
        T2       – 2m temperature [K]
        Q2       – 2m mixing ratio or specific humidity [kg/kg]
        PSFC     – surface pressure [Pa]
        U10,V10  – 10m wind components [m/s]
        GLW      – downward longwave radiation [W/m2]
        SWDOWN   – downward shortwave radiation [W/m2]
        RAINRATE – precipitation rate [mm/s or kg/m2/s, depending on source]

    This handler:
      - renames those to SYMFLUENCE/SUMMA standard names
      - derives wind speed magnitude
      - cleans attrs
      - writes processed files into forcing/merged_path
      - creates a grid shapefile from lat/lon
    """

    # --------------------- Variable mapping ---------------------

    def get_variable_mapping(self) -> Dict[str, str]:
        """
        Map raw CONUS404 variables to standard forcing names expected by SUMMA.

        Note:
        - We map RAINRATE directly to pptrate (precipitation rate).
        - If your CONUS404 RAINRATE is in mm/s or kg m-2 s-1, we treat it
          as equivalent to m s-1 after scaling in process_dataset().
        """
        return {
            "T2": "airtemp",
            "Q2": "spechum",      # may be mixing ratio; we adjust if needed
            "PSFC": "airpres",
            "GLW": "LWRadAtm",
            "SWDOWN": "SWRadAtm",
            "RAINRATE": "pptrate",
            "U10": "windspd_u",
            "V10": "windspd_v",
        }

    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Process CONUS404 dataset:
          - rename core meteorological variables
          - detect SW/LW/precip fields from typical HyTEST/WRF names
          - convert precip flux/rrate to pptrate [m s-1]
          - derive wind speed from U10/V10
          - standardize attributes
        """
        # --- Core met variables ---
        core_map = {
            "T2": "airtemp",
            "Q2": "spechum",
            "PSFC": "airpres",
            "U10": "windspd_u",
            "V10": "windspd_v",
        }
        rename_map = {old: new for old, new in core_map.items() if old in ds.data_vars}
        ds = ds.rename(rename_map)

        # ============================
        # Shortwave radiation → SWRadAtm
        # ============================
        # If already standardized, just grab it
        if "SWRadAtm" in ds.data_vars:
            sw = ds["SWRadAtm"]
        else:
            sw_candidates = [
                # HyTEST / CONUS404 accum / downward SW at bottom:
                "ACSWDNB", "ACSWDNLSM", "ACSWDNT",
                # More generic / WRF-style:
                "SWDOWN", "SWDOWN_surface",
                "RSDS", "rsds",
            ]
            sw_src = next((v for v in sw_candidates if v in ds.data_vars), None)
            if sw_src is None:
                self.logger.error("No shortwave radiation variable found in CONUS404 dataset")
                raise KeyError("SWRadAtm")

            # Rename to canonical name
            ds = ds.rename({sw_src: "SWRadAtm"})
            sw = ds["SWRadAtm"]

        sw.attrs.update(
            {
                "units": sw.attrs.get("units", "W m-2"),
                "long_name": "downward shortwave radiation at the surface",
                "standard_name": "surface_downwelling_shortwave_flux_in_air",
            }
        )

        # ============================
        # Longwave radiation → LWRadAtm
        # ============================
        if "LWRadAtm" in ds.data_vars:
            lw = ds["LWRadAtm"]
        else:
            lw_candidates = [
                # HyTEST / CONUS404:
                "ACLWDNB",
                # More generic / WRF-style:
                "GLW", "GLW_surface",
                "RLDS", "rlds",
            ]
            lw_src = next((v for v in lw_candidates if v in ds.data_vars), None)
            if lw_src is None:
                self.logger.error("No longwave radiation variable found in CONUS404 dataset")
                raise KeyError("LWRadAtm")

            ds = ds.rename({lw_src: "LWRadAtm"})
            lw = ds["LWRadAtm"]

        lw.attrs.update(
            {
                "units": lw.attrs.get("units", "W m-2"),
                "long_name": "downward longwave radiation at the surface",
                "standard_name": "surface_downwelling_longwave_flux_in_air",
            }
        )

        # ============================
        # Precipitation rate → pptrate [m s-1]
        # ============================
        if "pptrate" in ds.data_vars:
            p = ds["pptrate"]
        else:
            pr_candidates = [
                # HyTEST / CONUS404 rate we derived in cloud_downloader:
                "ACDRIPR",
                # More generic / WRF / reanalysis names:
                "RAINRATE", "RAINRATE_surface",
                "PRATE", "prate", "pr",
                "PRECIP", "precip",
            ]
            pr_src = next((v for v in pr_candidates if v in ds.data_vars), None)
            if pr_src is None:
                self.logger.error("No precipitation flux/rate variable found in CONUS404 dataset")
                raise KeyError("pptrate")

            ds = ds.rename({pr_src: "pptrate"})
            p = ds["pptrate"]

        units = p.attrs.get("units", "").lower()

        # Common cases: HyTEST precip is typically in mm/s or kg m-2 s-1
        if "kg m-2 s-1" in units or "kg m-2 s^-1" in units:
            ds["pptrate"] = p / 1000.0
            ds["pptrate"].attrs["units"] = "m s-1"
        elif "mm" in units:
            ds["pptrate"] = p / 1000.0
            ds["pptrate"].attrs["units"] = "m s-1"
        else:
            # Assume already m/s if unknown
            ds["pptrate"].attrs.setdefault("units", "m s-1")

        ds["pptrate"].attrs.update(
            {
                "long_name": "precipitation rate",
                "standard_name": "precipitation_rate",
            }
        )

        # ============================
        # Wind speed from components
        # ============================
        if "windspd_u" in ds and "windspd_v" in ds:
            u = ds["windspd_u"]
            v = ds["windspd_v"]
            windspd = np.sqrt(u**2 + v**2)
            windspd.name = "windspd"
            windspd.attrs = {
                "units": "m s-1",
                "long_name": "wind speed",
                "standard_name": "wind_speed",
            }
            ds["windspd"] = windspd
        else:
            self.logger.error("Missing U10 and/or V10 for wind speed in CONUS404 dataset")
            raise KeyError("windspd")

        # ============================
        # Attributes for other core variables
        # ============================
        if "airtemp" in ds:
            ds["airtemp"].attrs.update(
                {
                    "units": "K",
                    "long_name": "air temperature",
                    "standard_name": "air_temperature",
                }
            )
        if "spechum" in ds:
            ds["spechum"].attrs.update(
                {
                    "units": "kg kg-1",
                    "long_name": "specific humidity",
                    "standard_name": "specific_humidity",
                }
            )
        if "airpres" in ds:
            ds["airpres"].attrs.update(
                {
                    "units": "Pa",
                    "long_name": "air pressure",
                    "standard_name": "air_pressure",
                }
            )

        # Common metadata + clean attributes
        ds = self.setup_time_encoding(ds)
        ds = self.add_metadata(
            ds,
            "CONUS404 data standardized for SUMMA-compatible forcing (SYMFLUENCE)",
        )
        ds = self.clean_variable_attributes(ds)

        return ds



    # --------------------- Coordinates -------------------------

    def get_coordinate_names(self) -> Tuple[str, str]:
        """
        CONUS404 (HyTEST Zarr) exposes 2D lat/lon grids named 'lat' and 'lon'.
        """
        return ("lat", "lon")

    # --------------------- Merging / standardization ----------

    def needs_merging(self) -> bool:
        """
        We mark CONUS404 as needing 'merging' in the same sense as AORC:
        a standardization pass over raw cloud-downloaded files.
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
        Standardize CONUS404 forcings:

          - Look for domain-level CONUS404 NetCDF files in raw_forcing_path
          - Apply process_dataset()
          - Save processed files into merged_forcing_path
        """
        self.logger.info("Standardizing CONUS404 forcing files (no temporal merging)")

        merged_forcing_path.mkdir(parents=True, exist_ok=True)

        patterns: List[str] = [
            f"{self.domain_name}_CONUS404_*.nc",
            f"domain_{self.domain_name}_CONUS404_*.nc",
            "*CONUS404*.nc",
        ]

        files: List[Path] = []
        for pattern in patterns:
            candidates = sorted(raw_forcing_path.glob(pattern))
            if candidates:
                self.logger.info(
                    f"Found {len(candidates)} CONUS404 file(s) in {raw_forcing_path} "
                    f"with pattern '{pattern}'"
                )
                files = candidates
                break

        if not files:
            msg = f"No CONUS404 forcing files found in {raw_forcing_path} with patterns {patterns}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        for f in files:
            self.logger.info(f"Processing CONUS404 file: {f}")
            try:
                ds = xr.open_dataset(f)
            except Exception as e:
                self.logger.error(f"Error opening CONUS404 file {f}: {e}")
                continue

            try:
                ds_proc = self.process_dataset(ds)

                out_name = merged_forcing_path / f"{f.stem}_processed.nc"
                ds_proc.to_netcdf(out_name)
                self.logger.info(f"Saved processed CONUS404 forcing: {out_name}")
            except Exception as e:
                self.logger.error(f"Error processing CONUS404 dataset from {f}: {e}")
            finally:
                ds.close()

        self.logger.info("CONUS404 forcing standardization completed")

    # --------------------- Shapefile creation -----------------

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
        self.logger.info("Creating CONUS404 grid shapefile")

        output_shapefile = shapefile_path / f"forcing_{self.config['FORCING_DATASET']}.shp"

        # 🔧 Only use CONUS404 processed files, not ANY .nc
        patterns = [
            f"{self.domain_name}_CONUS404_*_processed.nc",
            f"domain_{self.domain_name}_CONUS404_*_processed.nc",
            "*CONUS404*_processed.nc",
        ]
        conus_files = []
        for pattern in patterns:
            candidates = sorted(merged_forcing_path.glob(pattern))
            if candidates:
                self.logger.info(
                    f"Found {len(candidates)} CONUS404 processed file(s) "
                    f"in {merged_forcing_path} with pattern '{pattern}'"
                )
                conus_files = candidates
                break

        if not conus_files:
            raise FileNotFoundError(
                f"No CONUS404 processed files found in {merged_forcing_path} "
                f"with patterns {patterns}"
            )

        conus_file = conus_files[0]
        self.logger.info(f"Using CONUS404 file for grid: {conus_file}")

        with xr.open_dataset(conus_file) as ds:
            var_lat, var_lon = self.get_coordinate_names()

            if var_lat in ds.coords:
                lat = ds.coords[var_lat].values
            elif var_lat in ds.variables:
                lat = ds[var_lat].values
            else:
                raise KeyError(
                    f"Latitude coordinate '{var_lat}' not found in CONUS404 file {conus_file}. "
                    f"Available coords: {list(ds.coords)}; variables: {list(ds.data_vars)}"
                )

            if var_lon in ds.coords:
                lon = ds.coords[var_lon].values
            elif var_lon in ds.variables:
                lon = ds[var_lon].values
            else:
                raise KeyError(
                    f"Longitude coordinate '{var_lon}' not found in CONUS404 file {conus_file}. "
                    f"Available coords: {list(ds.coords)}; variables: {list(ds.data_vars)}"
                )

        geometries = []
        ids = []
        lats = []
        lons = []

        if lat.ndim == 1 and lon.ndim == 1:
            # Regular 1D lat/lon axes
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
            # 2D lat/lon grid
            ny, nx = lat.shape
            total_cells = ny * nx
            self.logger.info(f"CONUS404 grid dimensions (2D): ny={ny}, nx={nx}, total={total_cells}")

            cell_count = 0
            for i in range(ny):
                for j in range(nx):
                    lat_corners = [
                        lat[i, j],
                        lat[i, j + 1] if j + 1 < nx else lat[i, j],
                        lat[i + 1, j + 1] if i + 1 < ny and j + 1 < nx else lat[i, j],
                        lat[i + 1, j] if i + 1 < ny else lat[i, j],
                    ]
                    lon_corners = [
                        lon[i, j],
                        lon[i, j + 1] if j + 1 < nx else lon[i, j],
                        lon[i + 1, j + 1] if i + 1 < ny and j + 1 < nx else lon[i, j],
                        lon[i + 1, j] if i + 1 < ny else lon[i, j],
                    ]

                    geometries.append(Polygon(zip(lon_corners, lat_corners)))
                    ids.append(i * nx + j)
                    lats.append(float(lat[i, j]))
                    lons.append(float(lon[i, j]))

                    cell_count += 1
                    if cell_count % 5000 == 0 or cell_count == total_cells:
                        self.logger.info(f"Created {cell_count}/{total_cells} CONUS404 grid cells")

        gdf = gpd.GeoDataFrame(
            {
                "geometry": geometries,
                "ID": ids,
                self.config.get("FORCING_SHAPE_LAT_NAME"): lats,
                self.config.get("FORCING_SHAPE_LON_NAME"): lons,
            },
            crs="EPSG:4326",
        )

        self.logger.info("Calculating elevation values for CONUS404 grid")
        elevations = elevation_calculator(gdf, dem_path, batch_size=50)
        gdf["elev_m"] = elevations

        shapefile_path.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_shapefile)
        self.logger.info(f"CONUS404 shapefile created at {output_shapefile}")

        return output_shapefile
