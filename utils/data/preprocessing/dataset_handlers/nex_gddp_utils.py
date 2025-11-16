# nex_gddp_utils.py

from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon

from .base_dataset import BaseDatasetHandler
from .dataset_registry import DatasetRegistry


@DatasetRegistry.register("nex-gddp-cmip6")
class NEXGDDPCMIP6Handler(BaseDatasetHandler):
    """
    Handler for NEX-GDDP-CMIP6 downscaled climate data.

    Assumptions (align with typical NEX-GDDP-CMIP6 conventions):
      - Variables:
          pr      – precipitation flux [kg m-2 s-1]
          tas     – near-surface air temperature [K]
          huss    – near-surface specific humidity [1]
          ps      – surface air pressure [Pa] (may be absent in some configs)
          rlds    – surface downwelling longwave radiation [W m-2]
          rsds    – surface downwelling shortwave radiation [W m-2]
          sfcWind – near-surface wind speed [m/s]

      - Coordinates: lat, lon (1D or 2D), time, possibly an ensemble/realization
        dimension that we currently do not collapse; we just preserve it.

    This handler:
      - renames variables to SUMMA standard names
      - converts pr [kg m-2 s-1] → pptrate [m s-1]
      - ensures attributes are consistent
      - writes processed files into forcing/merged_path
      - builds a grid shapefile from lat/lon
    """

    # --------------------- Variable mapping ---------------------

    def get_variable_mapping(self) -> Dict[str, str]:
        return {
            "pr": "pptrate",
            "tas": "airtemp",
            "huss": "spechum",
            "ps": "airpres",
            "rlds": "LWRadAtm",
            "rsds": "SWRadAtm",
            "sfcWind": "windspd",
        }

    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Process NEX-GDDP-CMIP6 dataset into SUMMA-friendly forcing:
          - rename variables
          - convert precipitation to m s-1
          - standardize metadata
        """
        var_map = self.get_variable_mapping()
        rename_map = {old: new for old, new in var_map.items() if old in ds.data_vars}
        ds = ds.rename(rename_map)

        # ---- Precipitation flux: kg m-2 s-1 → m s-1 ----
        if "pptrate" in ds:
            p = ds["pptrate"]
            units = p.attrs.get("units", "").lower()

            # Many CMIP6 datasets use "kg m-2 s-1"
            if "kg m-2 s-1" in units or "kg m-2 s^-1" in units:
                ds["pptrate"] = p / 1000.0
                ds["pptrate"].attrs["units"] = "m s-1"
            elif "mm" in units:
                # mm/s → m/s
                ds["pptrate"] = p / 1000.0
                ds["pptrate"].attrs["units"] = "m s-1"
            else:
                # Fall back: treat as already m s-1
                ds["pptrate"].attrs.setdefault("units", "m s-1")

            ds["pptrate"].attrs.update(
                {
                    "long_name": "precipitation rate",
                    "standard_name": "precipitation_rate",
                }
            )

        # ---- Temperature, humidity, pressure ----
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

        # ---- Radiation ----
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

        # ---- Wind speed ----
        if "windspd" in ds:
            ds["windspd"].attrs.update(
                {
                    "units": "m s-1",
                    "long_name": "wind speed",
                    "standard_name": "wind_speed",
                }
            )

        # Add common metadata and clean attributes
        ds = self.setup_time_encoding(ds)
        ds = self.add_metadata(
            ds,
            "NEX-GDDP-CMIP6 data standardized for SUMMA-compatible forcing (SYMFLUENCE)",
        )
        ds = self.clean_variable_attributes(ds)

        return ds

    # --------------------- Coordinates -------------------------

    def get_coordinate_names(self) -> Tuple[str, str]:
        """
        NEX-GDDP-CMIP6 NetCDF typically uses 'lat' and 'lon' for coordinates.
        """
        return ("lat", "lon")

    # --------------------- Merging / standardization ----------

    def needs_merging(self) -> bool:
        """
        As with AORC and CONUS404, we treat this as needing a ‘standardization’
        pass over the raw files, even if there is only one big file.
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
        Standardize NEX-GDDP-CMIP6 forcings:

          - Find NEX-GDDP-CMIP6 NetCDF files in raw_forcing_path
          - Apply process_dataset()
          - Save processed files into merged_forcing_path

        We do not (yet) merge across different models/scenarios/ensembles;
        each file is processed independently and later remapped.
        """
        self.logger.info("Standardizing NEX-GDDP-CMIP6 forcing files")

        merged_forcing_path.mkdir(parents=True, exist_ok=True)

        patterns: List[str] = [
            f"{self.domain_name}_NEX-GDDP-CMIP6_*.nc",
            "*NEX-GDDP-CMIP6*.nc",
            "*nex-gddp*.nc",
        ]

        files: List[Path] = []
        for pattern in patterns:
            candidates = sorted(raw_forcing_path.glob(pattern))
            if candidates:
                self.logger.info(
                    f"Found {len(candidates)} NEX-GDDP-CMIP6 file(s) in {raw_forcing_path} "
                    f"with pattern '{pattern}'"
                )
                files = candidates
                break

        if not files:
            msg = f"No NEX-GDDP-CMIP6 forcing files found in {raw_forcing_path} with patterns {patterns}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        for f in files:
            self.logger.info(f"Processing NEX-GDDP-CMIP6 file: {f}")
            try:
                ds = xr.open_dataset(f)
            except Exception as e:
                self.logger.error(f"Error opening NEX-GDDP-CMIP6 file {f}: {e}")
                continue

            try:
                ds_proc = self.process_dataset(ds)
                out_name = merged_forcing_path / f"{f.stem}_processed.nc"
                ds_proc.to_netcdf(out_name)
                self.logger.info(f"Saved processed NEX-GDDP-CMIP6 forcing: {out_name}")
            except Exception as e:
                self.logger.error(f"Error processing NEX-GDDP-CMIP6 dataset from {f}: {e}")
            finally:
                ds.close()

        self.logger.info("NEX-GDDP-CMIP6 forcing standardization completed")

    # --------------------- Shapefile creation -----------------

    def create_shapefile(
        self,
        shapefile_path: Path,
        merged_forcing_path: Path,
        dem_path: Path,
        elevation_calculator,
    ) -> Path:
        """
        Create NEX-GDDP-CMIP6 grid shapefile from lat/lon coordinates.
        """
        self.logger.info("Creating NEX-GDDP-CMIP6 grid shapefile")

        output_shapefile = shapefile_path / f"forcing_{self.config['FORCING_DATASET']}.shp"

        nex_files = list(merged_forcing_path.glob("*.nc"))
        if not nex_files:
            raise FileNotFoundError(f"No NEX-GDDP-CMIP6 processed files found in {merged_forcing_path}")

        nex_file = nex_files[0]
        self.logger.info(f"Using NEX-GDDP-CMIP6 file for grid: {nex_file}")

        with xr.open_dataset(nex_file) as ds:
            var_lat, var_lon = self.get_coordinate_names()

            if var_lat in ds.coords:
                lat = ds.coords[var_lat].values
            elif var_lat in ds.variables:
                lat = ds[var_lat].values
            else:
                raise KeyError(
                    f"Latitude coordinate '{var_lat}' not found in NEX-GDDP-CMIP6 file {nex_file}. "
                    f"Available coords: {list(ds.coords)}; variables: {list(ds.data_vars)}"
                )

            if var_lon in ds.coords:
                lon = ds.coords[var_lon].values
            elif var_lon in ds.variables:
                lon = ds[var_lon].values
            else:
                raise KeyError(
                    f"Longitude coordinate '{var_lon}' not found in NEX-GDDP-CMIP6 file {nex_file}. "
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
            self.logger.info(f"NEX-GDDP-CMIP6 grid dimensions (2D): ny={ny}, nx={nx}, total={total_cells}")

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
                        self.logger.info(f"Created {cell_count}/{total_cells} NEX-GDDP-CMIP6 grid cells")

        gdf = gpd.GeoDataFrame(
            {
                "geometry": geometries,
                "ID": ids,
                self.config.get("FORCING_SHAPE_LAT_NAME"): lats,
                self.config.get("FORCING_SHAPE_LON_NAME"): lons,
            },
            crs="EPSG:4326",
        )

        self.logger.info("Calculating elevation values for NEX-GDDP-CMIP6 grid")
        elevations = elevation_calculator(gdf, dem_path, batch_size=50)
        gdf["elev_m"] = elevations

        shapefile_path.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_shapefile)
        self.logger.info(f"NEX-GDDP-CMIP6 shapefile created at {output_shapefile}")

        return output_shapefile
