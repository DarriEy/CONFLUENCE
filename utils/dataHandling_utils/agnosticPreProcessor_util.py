import os
from pathlib import Path
import easymore # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import xarray as xr # type: ignore
import geopandas as gpd # type: ignore
import shutil
from rasterio.mask import mask # type: ignore
from shapely.geometry import Polygon # type: ignore
import rasterstats # type: ignore
from pyproj import CRS, Transformer # type: ignore
import pyproj # type: ignore
import shapefile # type: ignore
import rasterio # type: ignore
from rasterstats import zonal_stats # type: ignore

class forcingResampler:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"
        self.shapefile_path = self.project_dir / 'shapefiles' / 'forcing'
        dem_name = self.config['DEM_NAME']
        if dem_name == "default":
            dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"

        self.dem_path = self._get_default_path('DEM_PATH', f"attributes/elevation/dem/{dem_name}")
        self.forcing_basin_path = self.project_dir / 'forcing' / 'basin_averaged_data'
        self.catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')
        self.catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        if self.catchment_name == 'default':
            self.catchment_name = f"{self.config['DOMAIN_NAME']}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp"
        self.forcing_dataset = self.config.get('FORCING_DATASET').lower()
        self.merged_forcing_path = self._get_default_path('FORCING_PATH', 'forcing/raw_data')
        if self.forcing_dataset == 'rdrs':
            self.merged_forcing_path = self._get_default_path('FORCING_PATH', 'forcing/merged_path')
            
    def _get_default_path(self, path_key, default_subpath):
        path_value = self.config.get(path_key)
        if path_value == 'default' or path_value is None:
            return self.project_dir / default_subpath
        return Path(path_value)

    def run_resampling(self):
        self.logger.info("Starting forcing data resampling process")
        self.create_shapefile()
        self.remap_forcing()
        self.logger.info("Forcing data resampling process completed")

    def create_shapefile(self):
        self.logger.info(f"Creating {self.forcing_dataset.upper()} shapefile")

        if self.forcing_dataset == 'rdrs':
            self._create_rdrs_shapefile()
        elif self.forcing_dataset.lower() == 'era5':
            self._create_era5_shapefile()
        elif self.forcing_dataset == 'carra':
            self._create_carra_shapefile()
        else:
            self.logger.error(f"Unsupported forcing dataset: {self.forcing_dataset}")
            raise ValueError(f"Unsupported forcing dataset: {self.forcing_dataset}")

    def _create_rdrs_shapefile(self):
        forcing_file = next((f for f in os.listdir(self.merged_forcing_path) if f.endswith('.nc') and f.startswith('RDRS_monthly_')), None)
        if not forcing_file:
            self.logger.error("No RDRS monthly file found")
            return

        with xr.open_dataset(self.merged_forcing_path / forcing_file) as ds:
            rlat, rlon = ds.rlat.values, ds.rlon.values
            lat, lon = ds.lat.values, ds.lon.values

        geometries, ids, lats, lons = [], [], [], []
        for i in range(len(rlat)):
            for j in range(len(rlon)):
                rlat_corners = [rlat[i], rlat[i], rlat[i+1] if i+1 < len(rlat) else rlat[i], rlat[i+1] if i+1 < len(rlat) else rlat[i]]
                rlon_corners = [rlon[j], rlon[j+1] if j+1 < len(rlon) else rlon[j], rlon[j+1] if j+1 < len(rlon) else rlon[j], rlon[j]]
                lat_corners = [lat[i,j], lat[i,j+1] if j+1 < len(rlon) else lat[i,j], 
                            lat[i+1,j+1] if i+1 < len(rlat) and j+1 < len(rlon) else lat[i,j], 
                            lat[i+1,j] if i+1 < len(rlat) else lat[i,j]]
                lon_corners = [lon[i,j], lon[i,j+1] if j+1 < len(rlon) else lon[i,j], 
                            lon[i+1,j+1] if i+1 < len(rlat) and j+1 < len(rlon) else lon[i,j], 
                            lon[i+1,j] if i+1 < len(rlat) else lon[i,j]]
                geometries.append(Polygon(zip(lon_corners, lat_corners)))
                ids.append(i * len(rlon) + j)
                lats.append(lat[i,j])
                lons.append(lon[i,j])

        gdf = gpd.GeoDataFrame({
            'geometry': geometries,
            'ID': ids,
            self.config.get('FORCING_SHAPE_LAT_NAME'): lats,
            self.config.get('FORCING_SHAPE_LON_NAME'): lons,
        }, crs='EPSG:4326')

        zs = rasterstats.zonal_stats(gdf, str(self.dem_path), stats=['mean'])
        gdf['elev_m'] = [item['mean'] for item in zs]
        gdf.dropna(subset=['elev_m'], inplace=True)

        self.shapefile_path.mkdir(parents=True, exist_ok=True)
        output_shapefile = self.shapefile_path / f"forcing_{self.config['FORCING_DATASET']}.shp"
        gdf.to_file(output_shapefile)
        self.logger.info(f"RDRS shapefile created and saved to {output_shapefile}")

    def _create_era5_shapefile(self):
        self.logger.info("Creating ERA5 shapefile")

        # Find an .nc file in the forcing path
        forcing_files = list(self.merged_forcing_path.glob('*.nc'))
        if not forcing_files:
            raise FileNotFoundError("No ERA5 forcing files found")
        forcing_file = forcing_files[0]

        # Set the dimension variable names
        source_name_lat = "latitude"
        source_name_lon = "longitude"

        # Open the file and get the dimensions and spatial extent of the domain
        with xr.open_dataset(forcing_file) as src:
            lat = src[source_name_lat].values
            lon = src[source_name_lon].values

        # Find the spacing
        half_dlat = abs(lat[1] - lat[0])/2
        half_dlon = abs(lon[1] - lon[0])/2

        # Create lists to store the data
        geometries = []
        ids = []
        lats = []
        lons = []

        for i, center_lon in enumerate(lon):
            for j, center_lat in enumerate(lat):
                vertices = [
                    [float(center_lon)-half_dlon, float(center_lat)-half_dlat],
                    [float(center_lon)-half_dlon, float(center_lat)+half_dlat],
                    [float(center_lon)+half_dlon, float(center_lat)+half_dlat],
                    [float(center_lon)+half_dlon, float(center_lat)-half_dlat],
                    [float(center_lon)-half_dlon, float(center_lat)-half_dlat]
                ]
                geometries.append(Polygon(vertices))
                ids.append(i * len(lat) + j)
                lats.append(float(center_lat))
                lons.append(float(center_lon))

        # Create the GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'geometry': geometries,
            'ID': ids,
            self.config.get('FORCING_SHAPE_LAT_NAME'): lats,
            self.config.get('FORCING_SHAPE_LON_NAME'): lons,
        }, crs='EPSG:4326')

        # Calculate zonal statistics (mean elevation) for each grid cell
        zs = rasterstats.zonal_stats(gdf, str(self.dem_path), stats=['mean'])

        # Add mean elevation to the GeoDataFrame
        gdf['elev_m'] = [item['mean'] for item in zs]

        # Drop columns that are on the edge and don't have elevation data
        gdf.dropna(subset=['elev_m'], inplace=True)

        # Save the shapefile
        self.shapefile_path.mkdir(parents=True, exist_ok=True)
        output_shapefile = self.shapefile_path / f"forcing_{self.config['FORCING_DATASET']}.shp"
        gdf.to_file(output_shapefile)

        self.logger.info(f"ERA5 shapefile created and saved to {output_shapefile}")

    def _create_carra_shapefile(self):
        self.logger.info("Creating CARRA grid shapefile")

        # Find a processed CARRA file
        carra_files = list(self.merged_forcing_path.glob('*.nc'))
        if not carra_files:
            raise FileNotFoundError("No processed CARRA files found")
        carra_file = carra_files[0]

        # Read CARRA data
        with xr.open_dataset(carra_file) as ds:
            lats = ds.latitude.values
            lons = ds.longitude.values

        # Define CARRA projection
        carra_proj = pyproj.CRS('+proj=stere +lat_0=90 +lat_ts=90 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +a=6378137 +b=6356752.3142 +units=m +no_defs')
        wgs84 = pyproj.CRS('EPSG:4326')

        transformer = Transformer.from_crs(carra_proj, wgs84, always_xy=True)
        transformer_to_carra = Transformer.from_crs(wgs84, carra_proj, always_xy=True)

        # Create shapefile
        self.shapefile_path.mkdir(parents=True, exist_ok=True)
        output_shapefile = self.shapefile_path / f"forcing_{self.config['FORCING_DATASET']}.shp"

        with shapefile.Writer(str(output_shapefile)) as w:
            w.autoBalance = 1
            w.field("ID", 'N')
            w.field(self.config.get('FORCING_SHAPE_LAT_NAME'), 'F', decimal=6)
            w.field(self.config.get('FORCING_SHAPE_LON_NAME'), 'F', decimal=6)

            for i in range(len(lons)):
                # Convert lat/lon to CARRA coordinates
                x, y = transformer_to_carra.transform(lons[i], lats[i])
                
                # Define grid cell (assuming 2.5 km resolution)
                half_dx = 1250  # meters
                half_dy = 1250  # meters
                
                vertices = [
                    (x - half_dx, y - half_dy),
                    (x - half_dx, y + half_dy),
                    (x + half_dx, y + half_dy),
                    (x + half_dx, y - half_dy),
                    (x - half_dx, y - half_dy)
                ]
                
                # Convert vertices back to lat/lon
                lat_lon_vertices = [transformer.transform(vx, vy) for vx, vy in vertices]
                
                w.poly([lat_lon_vertices])
                center_lon, center_lat = transformer.transform(x, y)
                w.record(i, center_lat, center_lon)

        # Add elevation data to the shapefile
        shp = gpd.read_file(output_shapefile)
        shp = shp.set_crs('EPSG:4326')

        # Calculate zonal statistics (mean elevation) for each grid cell
        zs = rasterstats.zonal_stats(shp, str(self.dem_path), stats=['mean'])

        # Add mean elevation to the GeoDataFrame
        shp['elev_m'] = [item['mean'] for item in zs]

        # Save the updated shapefile
        shp.to_file(output_shapefile)

        self.logger.info(f"CARRA grid shapefile created and saved to {output_shapefile}")

    def remap_forcing(self):
        self.logger.info("Starting forcing remapping process")
        self._create_one_weighted_forcing_file()
        #self._create_all_weighted_forcing_files()
        self.logger.info("Forcing remapping process completed")

    def _create_one_weighted_forcing_file(self):
        self.logger.info("Creating one weighted forcing file")
        
        forcing_path = self.merged_forcing_path
        forcing_files = sorted([f for f in forcing_path.glob('*.nc')])
        shp_id = self.config.get('CATCHMENT_SHP_HRUID')

        for file in forcing_files:
            esmr = easymore.Easymore()

            esmr.author_name = 'SUMMA public workflow scripts'
            esmr.license = 'Copernicus data use license: https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf'
            esmr.case_name = f"{self.config['DOMAIN_NAME']}_{self.config['FORCING_DATASET']}"

            esmr.source_shp = self.project_dir / 'shapefiles' / 'forcing' / f"forcing_{self.config['FORCING_DATASET']}.shp"
            esmr.source_shp_lat = self.config.get('FORCING_SHAPE_LAT_NAME')
            esmr.source_shp_lon = self.config.get('FORCING_SHAPE_LON_NAME')

            esmr.target_shp = self.catchment_path / self.catchment_name
            esmr.target_shp_ID = shp_id
            esmr.target_shp_lat = self.config.get('CATCHMENT_SHP_LAT')
            esmr.target_shp_lon = self.config.get('CATCHMENT_SHP_LON')

            
            var_lat = 'lat' if self.forcing_dataset == 'rdrs' else 'latitude'
            var_lon = 'lon' if self.forcing_dataset == 'rdrs' else 'longitude'
        
            esmr.source_nc = str(file)
            esmr.var_names = ['airpres', 'LWRadAtm', 'SWRadAtm', 'pptrate', 'airtemp', 'spechum', 'windspd']
            esmr.var_lat = var_lat 
            esmr.var_lon = var_lon
            esmr.var_time = 'time'

            esmr.temp_dir = str(self.project_dir / 'forcing' / 'temp_easymore') + '/'
            esmr.output_dir = str(self.forcing_basin_path) + '/'

            esmr.remapped_dim_id = 'hru'
            esmr.remapped_var_id = 'hruId'
            esmr.format_list = ['f4']
            esmr.fill_value_list = ['-9999']

            esmr.save_csv = False
            esmr.remap_csv = ''
            esmr.sort_ID = False

            esmr.nc_remapper()

            # Move files to prescribed locations
            remap_file = f"{esmr.case_name}_remapping.nc"
            self.intersect_path = self.project_dir / 'shapefiles' / 'catchment_intersection' / 'with_forcing'
            self.intersect_path.mkdir(parents=True, exist_ok=True)
            os.rename(os.path.join(esmr.temp_dir, remap_file), self.intersect_path / remap_file)

            for file in Path(esmr.temp_dir).glob(f"{esmr.case_name}_intersected_shapefile.*"):
                os.rename(file, self.intersect_path / file.name)

            # Remove temporary directory
            shutil.rmtree(esmr.temp_dir, ignore_errors=True)

    def _create_all_weighted_forcing_files(self):
        self.logger.info("Creating all weighted forcing files")

        forcing_path = self.merged_forcing_path
        var_lat = 'lat' if self.forcing_dataset == 'rdrs' else 'latitude'
        var_lon = 'lon' if self.forcing_dataset == 'rdrs' else 'longitude'

        esmr = easymore.Easymore()

        esmr.author_name = 'SUMMA public workflow scripts'
        esmr.license = 'Copernicus data use license: https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf'
        esmr.case_name = f"{self.config['DOMAIN_NAME']}_{self.config['FORCING_DATASET']}"

        esmr.var_names = ['airpres', 'LWRadAtm', 'SWRadAtm', 'pptrate', 'airtemp', 'spechum', 'windspd']
        esmr.var_lat = var_lat
        esmr.var_lon = var_lon
        esmr.var_time = 'time'

        esmr.temp_dir = ''
        esmr.output_dir = str(self.forcing_basin_path) + '/'

        esmr.remapped_dim_id = 'hru'
        esmr.remapped_var_id = 'hruId'
        esmr.format_list = ['f4']
        esmr.fill_value_list = ['-9999']

        esmr.save_csv = False
        esmr.remap_csv = ''# str(self.intersect_path / f"{self.config.get('DOMAIN_NAME')}_{self.config.get('FORCING_DATASET')}_remapping.csv")
        esmr.sort_ID = False
        esmr.overwrite_existing_remap = False

        forcing_files = sorted([f for f in forcing_path.glob('*.nc')])
        for file in forcing_files[1:]:
            try:
                esmr.source_nc = str(file)
                esmr.nc_remapper()
            except Exception as e:
                self.logger.info(f'Issue with file: {file}, Error: {str(e)}')

        self.logger.info("All weighted forcing files created")

class geospatialStatistics:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"
        self.catchment_path = self._get_file_path('CATCHMENT_PATH', 'shapefiles/catchment')
        self.catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        if self.catchment_name == 'default':
            self.catchment_name = f"{self.config['DOMAIN_NAME']}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp"
        dem_name = self.config['DEM_NAME']
        if dem_name == "default":
            dem_name = f"domain_{self.config['DOMAIN_NAME']}_elv.tif"

        self.dem_path = self._get_file_path('DEM_PATH', f"attributes/elevation/dem/{dem_name}")
        self.soil_path = self._get_file_path('SOIL_CLASS_PATH', 'attributes/soilclass')
        self.land_path = self._get_file_path('LAND_CLASS_PATH', 'attributes/landclass')

    def _get_file_path(self, file_type, file_def_path):
        if self.config.get(f'{file_type}') == 'default':
            return self.project_dir / file_def_path
        else:
            return Path(self.config.get(f'{file_type}'))

    def get_nodata_value(self, raster_path):
        with rasterio.open(raster_path) as src:
            nodata = src.nodatavals[0]
            if nodata is None:
                nodata = -9999
            return nodata

    def calculate_elevation_stats(self):
        self.logger.info("Calculating elevation statistics")
        catchment_gdf = gpd.read_file(self.catchment_path / self.catchment_name)
        nodata_value = self.get_nodata_value(self.dem_path)

        with rasterio.open(self.dem_path) as src:
            affine = src.transform
            dem_data = src.read(1)

        stats = zonal_stats(catchment_gdf, dem_data, affine=affine, stats=['mean'], nodata=nodata_value)
        result_df = pd.DataFrame(stats).rename(columns={'mean': 'elev_mean_new'})
        
        if 'elev_mean' in catchment_gdf.columns:
            self.logger.info("Updating existing 'elev_mean' column")
            catchment_gdf['elev_mean'] = result_df['elev_mean_new']
        else:
            self.logger.info("Adding new 'elev_mean' column")
            catchment_gdf['elev_mean'] = result_df['elev_mean_new']

        result_df = result_df.drop(columns=['elev_mean_new'])

        intersect_path = self._get_file_path('INTERSECT_DEM_PATH', 'shapefiles/catchment_intersection/with_dem')
        intersect_name = self.config.get('INTERSECT_DEM_NAME')
        intersect_path.mkdir(parents=True, exist_ok=True)
        catchment_gdf.to_file(intersect_path / intersect_name)
        
        self.logger.info(f"Elevation statistics saved to {intersect_path / intersect_name}")

    def calculate_soil_stats(self):
        self.logger.info("Calculating soil statistics")
        catchment_gdf = gpd.read_file(self.catchment_path / self.catchment_name)
        soil_name = self.config['SOIL_CLASS_NAME']
        if soil_name == 'default':
            soil_name = f"domain_{self.config['DOMAIN_NAME']}_soil_classes.tif"
        soil_raster = self.soil_path / soil_name
        nodata_value = self.get_nodata_value(soil_raster)

        with rasterio.open(soil_raster) as src:
            affine = src.transform
            soil_data = src.read(1)

        stats = zonal_stats(catchment_gdf, soil_data, affine=affine, stats=['count'], 
                            categorical=True, nodata=nodata_value)
        result_df = pd.DataFrame(stats).fillna(0)

        def rename_column(x):
            if x == 'count':
                return x
            try:
                return f'USGS_{int(float(x))}'
            except ValueError:
                return x

        result_df = result_df.rename(columns=rename_column)
        for col in result_df.columns:
            if col != 'count':
                result_df[col] = result_df[col].astype(int)

        catchment_gdf = catchment_gdf.join(result_df)
        
        intersect_path = self._get_file_path('INTERSECT_SOIL_PATH', 'shapefiles/catchment_intersection/with_soilgrids')
        intersect_name = self.config.get('INTERSECT_SOIL_NAME')
        intersect_path.mkdir(parents=True, exist_ok=True)
        catchment_gdf.to_file(intersect_path / intersect_name)
        
        self.logger.info(f"Soil statistics saved to {intersect_path / intersect_name}")

    def calculate_land_stats(self):
        self.logger.info("Calculating land statistics")
        catchment_gdf = gpd.read_file(self.catchment_path / self.catchment_name)
        land_name = self.config['LAND_CLASS_NAME']
        if land_name == 'default':
            land_name = f"domain_{self.config['DOMAIN_NAME']}_land_classes.tif"
        land_raster = self.land_path / land_name
        nodata_value = self.get_nodata_value(land_raster)

        with rasterio.open(land_raster) as src:
            affine = src.transform
            land_data = src.read(1)

        stats = zonal_stats(catchment_gdf, land_data, affine=affine, stats=['count'], 
                            categorical=True, nodata=nodata_value)
        result_df = pd.DataFrame(stats).fillna(0)

        def rename_column(x):
            if x == 'count':
                return x
            try:
                return f'IGBP_{int(float(x))}'
            except ValueError:
                return x

        result_df = result_df.rename(columns=rename_column)
        for col in result_df.columns:
            if col != 'count':
                result_df[col] = result_df[col].astype(int)

        catchment_gdf = catchment_gdf.join(result_df)
        
        intersect_path = self._get_file_path('INTERSECT_LAND_PATH', 'shapefiles/catchment_intersection/with_landclass')
        intersect_name = self.config.get('INTERSECT_LAND_NAME')
        intersect_path.mkdir(parents=True, exist_ok=True)
        catchment_gdf.to_file(intersect_path / intersect_name)
        
        self.logger.info(f"Land statistics saved to {intersect_path / intersect_name}")

    def run_statistics(self):
        self.calculate_soil_stats()
        self.calculate_land_stats()
        self.calculate_elevation_stats()
        self.logger.info("All geospatial statistics calculated successfully")