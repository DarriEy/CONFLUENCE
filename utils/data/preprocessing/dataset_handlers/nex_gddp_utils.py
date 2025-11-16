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
        Process a single NEX-GDDP-CMIP6 dataset:

          - Rename native NEX variables to SYMFLUENCE / SUMMA names
          - Convert units where needed
          - Ensure presence of the standard forcing variables expected downstream.
        """
        # --- NEW: clean problematic attrs that conflict with xarray encodings ---
        # NEX-GDDP files often carry 'missing_value' in attrs *and* encodings.
        # xarray will error on to_netcdf if both exist, so we strip the attr copy.
        ds = ds.copy()  # avoid mutating caller’s dataset
        for vname, var in ds.variables.items():
            if "missing_value" in var.attrs:
                # optional debug log if you like
                # self.logger.debug(f"Removing 'missing_value' from attrs of {vname}")
                var.attrs.pop("missing_value", None)

        var_map = self.get_variable_mapping()

        # Only rename variables that exist in this dataset
        rename_map = {
            src: tgt for src, tgt in var_map.items()
            if src in ds.data_vars
        }
        if rename_map:
            self.logger.debug(
                f"Renaming NEX-GDDP-CMIP6 variables: {rename_map}"
            )
            ds = ds.rename(rename_map)

        # ---------------- Variable-specific handling ----------------

        # Precipitation: pr -> pptrate [kg m-2 s-1] (equivalent to mm/s)
        if "pptrate" in ds.data_vars:
            pr = ds["pptrate"]
            # Ensure we end up with float and clean metadata
            ds["pptrate"] = pr.astype("float32")
            ds["pptrate"].attrs.update(
                long_name="Total precipitation rate",
                units="kg m-2 s-1",
                standard_name="precipitation_flux",
            )

        # Longwave radiation: rlds -> LWRadAtm [W m-2]
        if "LWRadAtm" in ds.data_vars:
            lw = ds["LWRadAtm"]
            ds["LWRadAtm"] = lw.astype("float32")
            ds["LWRadAtm"].attrs.update(
                long_name="Downwelling longwave radiation at surface",
                units="W m-2",
            )

        # Shortwave radiation: rsds -> SWRadAtm [W m-2]
        if "SWRadAtm" in ds.data_vars:
            sw = ds["SWRadAtm"]
            ds["SWRadAtm"] = sw.astype("float32")
            ds["SWRadAtm"].attrs.update(
                long_name="Downwelling shortwave radiation at surface",
                units="W m-2",
            )

        # Near-surface air temperature: tas -> airtemp [K]
        if "airtemp" in ds.data_vars:
            ta = ds["airtemp"]
            ds["airtemp"] = ta.astype("float32")
            ds["airtemp"].attrs.update(
                long_name="Near-surface air temperature",
                units="K",
            )

        # Near-surface specific humidity / relative humidity:
        #   - if hurs (relative humidity) is available, we keep it as-is for now
        #   - if huss is available in some variants, you could map it to spechum
        if "hurs" in ds.data_vars and "spechum" not in ds.data_vars:
            hurs = ds["hurs"].astype("float32")
            ds["spechum"] = hurs
            ds["spechum"].attrs.update(
                long_name="Near-surface relative humidity",
                units="percent",
            )

        # Wind speed at 10m: sfcWind -> windspd [m s-1]
        if "windspd" in ds.data_vars:
            ws = ds["windspd"]
            ds["windspd"] = ws.astype("float32")
            ds["windspd"].attrs.update(
                long_name="Near-surface wind speed",
                units="m s-1",
            )

        # Keep only the standardized forcing variables plus coordinates
        keep_vars = [
            v for v in [
                "pptrate",
                "LWRadAtm",
                "SWRadAtm",
                "airtemp",
                "spechum",
                "windspd",
            ]
            if v in ds.data_vars
        ]

        if not keep_vars:
            self.logger.warning(
                "NEX-GDDP-CMIP6 process_dataset produced no forcing variables; "
                "dataset will be empty."
            )
            return ds

        ds_out = ds[keep_vars]

        # Make sure lat/lon coords are preserved
        for coord in ("lat", "lon"):
            if coord in ds.coords and coord not in ds_out.coords:
                ds_out = ds_out.assign_coords({coord: ds.coords[coord]})

        return ds_out


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

        # Support both the "domain + dataset" naming convention *and*
        # the NEXGDDP_all_YYYYMM.nc files produced by the downloader.
        patterns: List[str] = [
            f"{self.domain_name}_NEX-GDDP-CMIP6_*.nc",  # future / more explicit naming
            "NEXGDDP_all_*.nc",                         # current downloader output
            "*NEXGDDP*.nc",                             # any other NEXGDDP-style names
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
            msg = (
                f"No NEX-GDDP-CMIP6 forcing files found in {raw_forcing_path} "
                f"with patterns {patterns}"
            )
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
                self.logger.error(
                    f"Error processing NEX-GDDP-CMIP6 dataset from {f}: {e}"
                )
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
