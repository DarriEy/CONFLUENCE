# hrrr_utils.py

from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon

from .base_dataset import BaseDatasetHandler
from .dataset_registry import DatasetRegistry


@DatasetRegistry.register("hrrr")
class HRRRHandler(BaseDatasetHandler):
    """
    Handler for HRRR (High-Resolution Rapid Refresh) forcing data.

    Assumptions (similar to AORC, NCEP-style fields in NetCDF):
      - Coordinates:
          time, latitude, longitude
      - Typical variables:
          APCP_surface         – accumulated precip or precip rate
          TMP_2maboveground    – 2m air temp [K]
          SPFH_2maboveground   – 2m specific humidity [kg/kg]
          PRES_surface         – surface pressure [Pa]
          DLWRF_surface        – down longwave [W/m²]
          DSWRF_surface        – down shortwave [W/m²]
          UGRD_10maboveground  – 10m U wind [m/s]
          VGRD_10maboveground  – 10m V wind [m/s]

    This mirrors AORC’s mapping so that the rest of the pipeline
    (SUMMA, easymore, etc.) can treat HRRR in the same way.
    """

    # ------------ variable mapping ------------

    def get_variable_mapping(self) -> Dict[str, str]:
        """
        Map raw HRRR variables → SYMFLUENCE/SUMMA standard names.
        """
        return {
            "APCP_surface": "pptrate",
            "TMP_2maboveground": "airtemp",
            "SPFH_2maboveground": "spechum",
            "PRES_surface": "airpres",
            "DLWRF_surface": "LWRadAtm",
            "DSWRF_surface": "SWRadAtm",
            "UGRD_10maboveground": "windspd_u",
            "VGRD_10maboveground": "windspd_v",
        }

    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Process HRRR dataset:
          - rename variables
          - derive wind speed from components
          - enforce units / attrs
        """
        var_map = self.get_variable_mapping()
        rename_map = {old: new for old, new in var_map.items() if old in ds.data_vars}
        ds = ds.rename(rename_map)

        # Wind speed magnitude
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

        # Precipitation rate
        if "pptrate" in ds:
            p = ds["pptrate"]
            # remove problematic encoding attrs if present
            p.attrs.pop("missing_value", None)

            units = p.attrs.get("units", "").lower()
            # Handle common cases; adjust if your HRRR derives rate differently
            if "mm" in units:
                # mm/s → m/s
                ds["pptrate"] = p / 1000.0
                ds["pptrate"].attrs["units"] = "m s-1"
            elif "kg m-2 s-1" in units or "kg m-2 s^-1" in units:
                # 1 kg/m²/s ≈ 1 mm/s = 1e-3 m/s
                ds["pptrate"] = p / 1000.0
                ds["pptrate"].attrs["units"] = "m s-1"
            else:
                ds["pptrate"].attrs.setdefault("units", "m s-1")

            ds["pptrate"].attrs.update(
                {
                    "long_name": "precipitation rate",
                    "standard_name": "precipitation_rate",
                }
            )

        # Temperature
        if "airtemp" in ds:
            ds["airtemp"].attrs.update(
                {
                    "units": "K",
                    "long_name": "air temperature",
                    "standard_name": "air_temperature",
                }
            )

        # Specific humidity
        if "spechum" in ds:
            ds["spechum"].attrs.update(
                {
                    "units": "kg kg-1",
                    "long_name": "specific humidity",
                    "standard_name": "specific_humidity",
                }
            )

        # Pressure
        if "airpres" in ds:
            ds["airpres"].attrs.update(
                {
                    "units": "Pa",
                    "long_name": "air pressure",
                    "standard_name": "air_pressure",
                }
            )

        # Radiation
        if "LWRadAtm" in ds:
            ds["LWRadAtm"].attrs.update(
                {
                    "units": "W m-2",
                    "long_name": "downward longwave radiation at the surface",
                    "standard_name": "surface_downwelling_longwave_flux_in_air",
                }
            )

        if "SWRadAtm" in ds:
            ds["SWRadAtm"].attrs.update(
                {
                    "units": "W m-2",
                    "long_name": "downward shortwave radiation at the surface",
                    "standard_name": "surface_downwelling_shortwave_flux_in_air",
                }
            )

        # Common metadata + clean attrs via base helpers
        ds = self.setup_time_encoding(ds)
        ds = self.add_metadata(
            ds,
            "HRRR data standardized for SUMMA-compatible forcing (SYMFLUENCE)",
        )
        ds = self.clean_variable_attributes(ds)

        return ds

    # ------------ coordinates ------------

    def get_coordinate_names(self) -> Tuple[str, str]:
        """
        HRRR NetCDF subset from the cloud downloader is assumed to have:
            latitude, longitude
        as coordinates.
        """
        return ("latitude", "longitude")

    # ------------ merging / standardization ------------

    def needs_merging(self) -> bool:
        """
        Treat HRRR like AORC/CONUS404: run a standardization pass
        over the raw cloud-downloaded file(s).
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
        Standardize HRRR forcings:

          - Find HRRR NetCDF files in raw_forcing_path
          - Apply process_dataset()
          - Save processed file(s) to merged_forcing_path
        """
        self.logger.info("Standardizing HRRR forcing files (no temporal merging)")

        merged_forcing_path.mkdir(parents=True, exist_ok=True)

        patterns: List[str] = [
            f"{self.domain_name}_HRRR_*.nc",
            f"domain_{self.domain_name}_HRRR_*.nc",
            "*HRRR*.nc",
        ]

        files: List[Path] = []
        for pattern in patterns:
            candidates = sorted(raw_forcing_path.glob(pattern))
            if candidates:
                self.logger.info(
                    f"Found {len(candidates)} HRRR file(s) in {raw_forcing_path} "
                    f"with pattern '{pattern}'"
                )
                files = candidates
                break

        if not files:
            msg = f"No HRRR forcing files found in {raw_forcing_path} with patterns {patterns}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        for f in files:
            self.logger.info(f"Processing HRRR file: {f}")
            try:
                ds = xr.open_dataset(f)
            except Exception as e:
                self.logger.error(f"Error opening HRRR file {f}: {e}")
                continue

            try:
                ds_proc = self.process_dataset(ds)
                out_name = merged_forcing_path / f"{f.stem}_processed.nc"
                ds_proc.to_netcdf(out_name)
                self.logger.info(f"Saved processed HRRR forcing: {out_name}")
            except Exception as e:
                self.logger.error(f"Error processing HRRR dataset from {f}: {e}")
            finally:
                ds.close()

        self.logger.info("HRRR forcing standardization completed")

    # ------------ shapefile creation ------------

    def create_shapefile(
        self,
        shapefile_path: Path,
        merged_forcing_path: Path,
        dem_path: Path,
        elevation_calculator,
    ) -> Path:
        """
        Create HRRR grid shapefile from latitude/longitude.

        Mirrors the AORC/CONUS404 logic but assumes regular lat/lon grid.
        """
        self.logger.info("Creating HRRR grid shapefile")

        output_shapefile = shapefile_path / f"forcing_{self.config['FORCING_DATASET']}.shp"

        hrrr_files = list(merged_forcing_path.glob("*.nc"))
        if not hrrr_files:
            raise FileNotFoundError(f"No HRRR processed files found in {merged_forcing_path}")

        hrrr_file = hrrr_files[0]
        self.logger.info(f"Using HRRR file for grid: {hrrr_file}")

        with xr.open_dataset(hrrr_file) as ds:
            var_lat, var_lon = self.get_coordinate_names()

            if var_lat in ds.coords:
                lat = ds.coords[var_lat].values
            elif var_lat in ds.variables:
                lat = ds[var_lat].values
            else:
                raise KeyError(
                    f"Latitude coordinate '{var_lat}' not found in HRRR file {hrrr_file}. "
                    f"Available coords: {list(ds.coords)}; variables: {list(ds.data_vars)}"
                )

            if var_lon in ds.coords:
                lon = ds.coords[var_lon].values
            elif var_lon in ds.variables:
                lon = ds[var_lon].values
            else:
                raise KeyError(
                    f"Longitude coordinate '{var_lon}' not found in HRRR file {hrrr_file}. "
                    f"Available coords: {list(ds.coords)}; variables: {list(ds.data_vars)}"
                )

        geometries = []
        ids = []
        lats = []
        lons = []

        if lat.ndim == 1 and lon.ndim == 1:
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
            ny, nx = lat.shape
            total_cells = ny * nx
            self.logger.info(f"HRRR grid dimensions (2D): ny={ny}, nx={nx}, total={total_cells}")

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
                        self.logger.info(f"Created {cell_count}/{total_cells} HRRR grid cells")

        gdf = gpd.GeoDataFrame(
            {
                "geometry": geometries,
                "ID": ids,
                self.config.get("FORCING_SHAPE_LAT_NAME"): lats,
                self.config.get("FORCING_SHAPE_LON_NAME"): lons,
            },
            crs="EPSG:4326",
        )

        self.logger.info("Calculating elevation values for HRRR grid")
        elevations = elevation_calculator(gdf, dem_path, batch_size=50)
        gdf["elev_m"] = elevations

        shapefile_path.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_shapefile)
        self.logger.info(f"HRRR shapefile created at {output_shapefile}")

        return output_shapefile
