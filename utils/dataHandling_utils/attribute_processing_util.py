from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import os
import logging
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
from rasterstats import zonal_stats
from osgeo import gdal, osr
import glob
import xarray as xr
from dateutil.relativedelta import relativedelta
from scipy.stats import circmean, circstd, skew, kurtosis
from scipy.optimize import curve_fit
import rasterio

class attributeProcessor:
    """
    Processes catchment characteristic attributes for hydrological modeling.
    
    This class adapts CAMELS-SPAT functionality to extract catchment attributes
    from various geospatial datasets for use in CONFLUENCE hydrological modeling.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """Initialize the attribute processor."""
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
        # Create attributes directory
        self.attributes_dir = self.project_dir / 'attributes'
        self.attributes_dir.mkdir(parents=True, exist_ok=True)
        
        # Get the catchment shapefile
        self.catchment_path = self._get_catchment_path()
        self.is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
        self.case = 'lumped' if self.is_lumped else 'distributed'
        
        # Equal area coordinate reference system (for area calculations)
        self.equal_area_crs = 'ESRI:102008'
        
        # Initialize attribute processing directories
        self._init_attribute_dirs()
    
    def _init_attribute_dirs(self):
        """Initialize directories for different attribute types."""
        # Create subdirectories for different attribute types
        attribute_types = [
            'soil', 'climate', 'topography', 'landcover', 
            'geology', 'hydrology', 'openwater'
        ]
        
        for attr_type in attribute_types:
            attr_dir = self.attributes_dir / attr_type
            attr_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_catchment_path(self) -> Path:
        """Get the path to the catchment shapefile."""
        catchment_path = self.config.get('CATCHMENT_PATH')
        catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        
        if catchment_path == 'default':
            catchment_path = self.project_dir / 'shapefiles' / 'catchment'
        else:
            catchment_path = Path(catchment_path)
        
        if catchment_name == 'default':
            # Find the catchment shapefile based on domain discretization
            discretization = self.config.get('DOMAIN_DISCRETIZATION')
            catchment_file = f"{self.domain_name}_HRUs_{discretization}.shp"
        else:
            catchment_file = catchment_name
        
        return catchment_path / catchment_file
    
    def process_attributes(self):
        """Process all catchment attributes and save to output files."""
        self.logger.info("Starting attribute processing")
        
        # Determine if we are processing lumped or distributed attributes
        attribute_type = 'lumped' if self.is_lumped else 'distributed'
        self.logger.info(f"Processing {attribute_type} attributes")
        
        try:
            # Initialize attribute collection
            if self.is_lumped:
                l_values = []  # For lumped case, single list of values
                l_index = []   # List to track attribute metadata
            else:
                # Read catchment to determine number of HRUs
                catchment = gpd.read_file(self.catchment_path)
                num_hrus = len(catchment)
                l_values = [[] for _ in range(num_hrus)]  # List of lists for distributed case
                l_index = []  # Single index list for all HRUs
            
            # Process attributes from different sources
            self.logger.info("Processing soil attributes")
            l_values, l_index = self._process_soil_attributes(l_values, l_index)
            
            self.logger.info("Processing topographic attributes")
            l_values, l_index = self._process_topographic_attributes(l_values, l_index)
            
            self.logger.info("Processing land cover attributes")
            l_values, l_index = self._process_landcover_attributes(l_values, l_index)
            
            self.logger.info("Processing climate attributes")
            l_values, l_index = self._process_climate_attributes(l_values, l_index)
            
            # Convert to DataFrame and save
            attributes_df = self._convert_to_dataframe(l_values, l_index)
            
            # Save attributes
            output_file = self.attributes_dir / f"{self.domain_name}_attributes.csv"
            attributes_df.to_csv(output_file)
            self.logger.info(f"Attributes saved to {output_file}")
            
            return attributes_df
            
        except Exception as e:
            self.logger.error(f"Error processing attributes: {str(e)}")
            raise
    
    def _convert_to_dataframe(self, l_values, l_index) -> pd.DataFrame:
        """Convert attribute lists to DataFrame."""
        if self.is_lumped:
            # For lumped domain
            attributes_df = pd.DataFrame([l_values], columns=[f"{category}.{name}" for category, name, _, _ in l_index])
            attributes_df.insert(0, 'Basin_ID', self.domain_name)
        else:
            # For distributed domain
            # Create a list of attribute dictionaries for each HRU
            attribute_dicts = []
            catchment = gpd.read_file(self.catchment_path)
            hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
            
            for i, hru_values in enumerate(l_values):
                attr_dict = {f"{category}.{name}": value for (category, name, _, _), value in zip(l_index, hru_values)}
                attr_dict['HRU_ID'] = catchment.iloc[i][hru_id_field]
                attr_dict['Basin_ID'] = self.domain_name
                attribute_dicts.append(attr_dict)
            
            attributes_df = pd.DataFrame(attribute_dicts)
        
        return attributes_df
    
    def _process_soil_attributes(self, l_values, l_index):
        """Process soil attributes from SOILGRIDS."""
        # Get SOILGRIDS data path
        soilgrids_path = self._get_data_path('ATTRIBUTES_SOILGRIDS_PATH', 'soilgrids')
        
        # Process SOILGRIDS data if available
        if soilgrids_path.exists():
            self.logger.info(f"Processing SOILGRIDS data from {soilgrids_path}")
            l_values, l_index = self.attributes_from_soilgrids(
                soilgrids_path,
                'soilgrids',
                str(self.catchment_path),
                l_values,
                l_index,
                case=self.case
            )
        else:
            self.logger.warning(f"SOILGRIDS data not found at {soilgrids_path}")
        
        # Process Pelletier soil data if available
        pelletier_path = self._get_data_path('ATTRIBUTES_PELLETIER_PATH', 'pelletier')
        if pelletier_path.exists():
            self.logger.info(f"Processing Pelletier soil data from {pelletier_path}")
            l_values, l_index = self.attributes_from_pelletier(
                pelletier_path,
                'pelletier',
                str(self.catchment_path),
                l_values,
                l_index,
                case=self.case
            )
        
        return l_values, l_index
    
    def _process_topographic_attributes(self, l_values, l_index):
        """Process topographic attributes from MERIT and other sources."""
        merit_path = self._get_data_path('ATTRIBUTES_MERIT_PATH', 'merit')
        
        if merit_path.exists():
            # Get river network path
            river_network_path = self._get_river_network_path()
            
            # Prepare basin area information
            if self.is_lumped:
                basin = gpd.read_file(self.catchment_path)
                basin_area = basin.to_crs(self.equal_area_crs).area.sum() / 10**6  # km²
                area_info = {'Basin_area_km2': basin_area}
            else:
                area_info = None
            
            # Process MERIT attributes
            l_values, l_index, _ = self.attributes_from_merit(
                merit_path,
                'merit',
                str(self.catchment_path),
                river_network_path,
                area_info,
                l_values,
                l_index,
                equal_area_crs=self.equal_area_crs,
                case=self.case
            )
        
        return l_values, l_index
    
    def _process_landcover_attributes(self, l_values, l_index):
        """Process land cover attributes from various sources."""
        # Process MODIS land cover data
        modis_path = self._get_data_path('ATTRIBUTES_MODIS_PATH', 'modis')
        if modis_path.exists():
            l_values, l_index = self.attributes_from_modis_land(
                modis_path,
                'modis',
                str(self.catchment_path),
                l_values,
                l_index,
                case=self.case
            )
        
        # Process GLCLU2019 data
        glclu_path = self._get_data_path('ATTRIBUTES_GLCLU_PATH', 'glclu2019')
        if glclu_path.exists():
            l_values, l_index = self.attributes_from_glclu2019(
                glclu_path,
                'glclu2019',
                str(self.catchment_path),
                l_values,
                l_index,
                case=self.case
            )
        
        return l_values, l_index
    
    def _process_climate_attributes(self, l_values, l_index):
        """Process climate attributes from ERA5, RDRS, and WorldClim."""
        # Process WorldClim data
        worldclim_path = self._get_data_path('ATTRIBUTES_WORLDCLIM_PATH', 'worldclim')
        if worldclim_path.exists():
            # Check for PET data, generate if needed
            pet_path = worldclim_path / 'raw' / 'pet'
            if not pet_path.exists() or len(list(pet_path.glob('*.tif'))) == 0:
                self.logger.info("Generating PET data from WorldClim temperature and radiation")
                self.oudin_pet_from_worldclim(worldclim_path, 'raw')
            
            # Check for aridity and snow fraction data, generate if needed
            aridity_path = worldclim_path / 'raw' / 'aridity2'
            if not aridity_path.exists() or len(list(aridity_path.glob('*.tif'))) == 0:
                self.logger.info("Generating aridity and snow fraction data from WorldClim")
                self.aridity_and_fraction_snow_from_worldclim(worldclim_path, 'raw')
            
            # Process WorldClim attributes
            l_values, l_index = self.attributes_from_worldclim(
                worldclim_path,
                'raw',
                str(self.catchment_path),
                l_values,
                l_index,
                case=self.case
            )
        
        return l_values, l_index
    
    def _get_data_path(self, config_key, default_subfolder):
        """Get the path to a data source."""
        path = self.config.get(config_key)
        
        if path == 'default':
            path = self.data_dir / 'geospatial-data' / default_subfolder
        else:
            path = Path(path)
        
        return path
    
    def _get_river_network_path(self) -> str:
        """Get the path to the river network shapefile."""
        river_network_path = self.config.get('RIVER_NETWORK_SHP_PATH')
        river_network_name = self.config.get('RIVER_NETWORK_SHP_NAME')
        
        if river_network_path == 'default':
            river_network_path = self.project_dir / 'shapefiles' / 'river_network'
        else:
            river_network_path = Path(river_network_path)
        
        if river_network_name == 'default':
            river_network_name = f"{self.domain_name}_riverNetwork.shp"
        
        return str(river_network_path / river_network_name)

    # =====================================================================
    # CAMELS-SPAT Function Adaptations
    # =====================================================================
    
    # Utility methods for processing attributes
    def harmonic_mean(self, x):
        """Calculate harmonic mean, handling masked arrays."""
        return np.ma.count(x) / np.sum(1/x)
    
    def read_scale_and_offset(self, geotiff_path):
        """Read scale and offset values from a GeoTIFF file."""
        geotiff_path = str(geotiff_path)
        dataset = gdal.Open(geotiff_path)
        if dataset is None:
            raise FileNotFoundError(f"File not found: {geotiff_path}")
        scale = dataset.GetRasterBand(1).GetScale()
        offset = dataset.GetRasterBand(1).GetOffset()
        dataset = None
        return scale, offset
    
    def check_zonal_stats_outcomes(self, zonal_out, new_val=np.nan):
        """Check for None values in zonal_out and replace with specified value."""
        for ix in range(0, len(zonal_out)):
            for key, val in zonal_out[ix].items():
                if val is None:
                    zonal_out[ix][key] = new_val
        return zonal_out
    
    def update_values_list(self, l_values, stats, zonal_out, scale, offset, case='lumped'):
        """Update values list with zonal statistics results."""
        # Update scale and offset to usable values
        if scale is None: scale = 1  # If scale is undefined that means we simply multiply by 1
        if offset is None: offset = 0  # Undefined offset > add 0

        # Deal with the occasional None that we get when raster data input to zonal-stats is missing
        zonal_out = self.zonal_out_none2nan(zonal_out)
        
        # We loop through the calculated stats in a pre-determined order
        if case == 'lumped':  # lumped case
            if 'min' in stats:  l_values.append(zonal_out[0]['min'] * scale + offset)
            if 'mean' in stats: l_values.append(zonal_out[0]['mean'] * scale + offset)
            if 'harmonic_mean' in stats: l_values.append(zonal_out[0]['harmonic_mean'] * scale + offset)
            if 'circ_mean' in stats: l_values.append(zonal_out[0]['circ_mean'] * scale + offset)
            if 'max' in stats:  l_values.append(zonal_out[0]['max'] * scale + offset)
            if 'std' in stats:  l_values.append(zonal_out[0]['std'] * scale + offset)
            if 'circ_std' in stats:  l_values.append(zonal_out[0]['circ_std'] * scale + offset)
        
        # distributed case, multiple polygons
        elif case == 'distributed':
            # confirm that l_values has as many nested lists as we have zonal stats outputs
            num_nested_lists = sum(1 for item in l_values if isinstance(item, list))
            assert num_nested_lists == len(zonal_out), f"zonal_out length does not match expected list length {num_nested_lists}"
            
            # now loop over the zonal outputs and append to relevant lists
            for i in range(0, num_nested_lists):
                if 'min' in stats:  l_values[i].append(zonal_out[i]['min'] * scale + offset)
                if 'mean' in stats: l_values[i].append(zonal_out[i]['mean'] * scale + offset)
                if 'harmonic_mean' in stats: l_values[i].append(zonal_out[i]['harmonic_mean'] * scale + offset)
                if 'circ_mean' in stats: l_values[i].append(zonal_out[i]['circ_mean'] * scale + offset)
                if 'max' in stats:  l_values[i].append(zonal_out[i]['max'] * scale + offset)
                if 'std' in stats:  l_values[i].append(zonal_out[i]['std'] * scale + offset)
                if 'circ_std' in stats:  l_values[i].append(zonal_out[i]['circ_std'] * scale + offset)

        return l_values
    
    def zonal_out_none2nan(self, zonal_out):
        """Replace None values in zonal_out with NaN."""
        for zonal_dict in zonal_out:
            for key in zonal_dict:
                if zonal_dict[key] is None:
                    zonal_dict[key] = np.nan
        return zonal_out
    
    def calc_circmean(self, x):
        """Calculate circular mean, handling masked arrays."""
        return circmean(x.compressed(), high=360) 

    def calc_circstd(self, x):
        """Calculate circular standard deviation, handling masked arrays."""
        return circstd(x.compressed(), high=360)
    
    def get_geotif_data_as_array(self, file, band=1):
        """Get GeoTIFF data as a NumPy array."""
        file = str(file)
        ds = gdal.Open(file)  # open the file
        band = ds.GetRasterBand(band)  # get the data band
        data = band.ReadAsArray()  # convert to numpy array
        return data
    
    def get_geotif_nodata_value(self, tif):
        """Get the no-data value from a GeoTIFF file."""
        with rasterio.open(tif) as src:
            nodata_value = src.nodata
        return nodata_value
    
    def check_and_set_nodata_value_on_pelletier_file(self, tif, nodata=255):
        """Check and set no-data value on Pelletier file if not already set."""
        if not self.get_geotif_nodata_value(tif):  # no no-data value set yet
            data = self.get_geotif_data_as_array(tif)
            if data.max() == nodata:  # but we do need to set a new no-data value here
                ds = gdal.Open(tif)
                band = ds.GetRasterBand(1)
                band.SetNoDataValue(255)
                ds.FlushCache()
                ds = None
    
    def check_scale_and_offset(self, tif):
        """Check scale and offset values of a GeoTIFF file."""
        scale, offset = self.read_scale_and_offset(tif)
        if scale is None: scale = 1
        if offset is None: offset = 0
        if not (scale == 1) and not (offset == 0):
            self.logger.warning(f'Scale or offset not 1 or 0 respectively: {scale}, {offset}')
        return True
    
    def get_categorical_dict(self, source):
        """Get dictionaries for categorical variables."""
        if source == 'GLCLU 2019':
            cat_dict = {1: 'true_desert', 2: 'semi_arid', 3: 'dense_short_vegetation',
                        4: 'open_tree_cover', 5: 'dense_tree_cover', 6: 'tree_cover_gain',
                        7: 'tree_cover_loss', 8: 'salt_pan', 9: 'wetland_sparse_vegetation',
                        10: 'wetland_dense_short_vegetation', 11: 'wetland_open_tree_cover',
                        12: 'wetland_dense_tree_cover', 13: 'wetland_tree_cover_gain',
                        14: 'wetland_tree_cover_loss', 15: 'ice', 16: 'water',
                        17: 'cropland', 18: 'built_up', 19: 'ocean', 20: 'no_data'}
        elif source == 'MCD12Q1.061':
            cat_dict = {1: 'evergreen_needleleaf_forest', 2: 'evergreen_broadleaf_forest',
                        3: 'deciduous_needleleaf_forest', 4: 'deciduous_broadleaf_forest',
                        5: 'mixed_forest', 6: 'closed_shrubland', 7: 'open_shrubland',
                        8: 'woody_savanna', 9: 'savanna', 10: 'grassland',
                        11: 'permanent_wetland', 12: 'cropland', 13: 'urban_and_built_up',
                        14: 'cropland_natural_mosaic', 15: 'permanent_snow_ice',
                        16: 'barren', 17: 'water', 255: 'unclassified'}
        elif source == 'LGRIP30':
            cat_dict = {0: 'water', 1: 'non_cropland', 2: 'irrigated_cropland', 3: 'rainfed_cropland'}
        
        return cat_dict
    
    def update_values_list_with_categorical(self, l_values, l_index, zonal_out, source, prefix='', case='lumped'):
        """Maps a zonal histogram of categorical classes onto descriptions and adds to lists."""
        # Get the category definitions
        cat_dict = self.get_categorical_dict(source)

        # Separately handle lumped and distributed cases
        if case == 'lumped':
            # Find the total number of classified pixels
            total_pixels = 0
            for land_id, count in zonal_out[0].items():
                total_pixels += count
            
            # Loop over all categories and see what we have in this catchment
            for land_id, text in cat_dict.items():
                land_prct = 0
                if land_id in zonal_out[0].keys():
                    land_prct = zonal_out[0][land_id] / total_pixels
                l_values.append(land_prct)
                l_index.append(('Land cover', f'{prefix}{text}_fraction', '-', f'{source}'))
        else:  # distributed case
            num_nested_lists = sum(1 for item in l_values if isinstance(item, list))
            assert num_nested_lists == len(zonal_out), f"zonal_out length does not match expected list length"

            for i in range(0, num_nested_lists):
                # Find the total number of classified pixels
                total_pixels = 0
                for land_id, count in zonal_out[i].items():
                    total_pixels += count
                
                # Process all categories for this HRU
                tmp_index = []
                for land_id, text in cat_dict.items():
                    land_prct = 0
                    if land_id in zonal_out[i].keys():
                        land_prct = zonal_out[i][land_id] / total_pixels
                    l_values[i].append(land_prct)
                    tmp_index.append(('Land cover', f'{prefix}{text}_fraction', '-', f'{source}'))

            # Add index values only once
            for item in tmp_index:
                l_index.append(item)

        return l_values, l_index
    
    def zonal_stats_unit_conversion(self, zonal_out, stat_to_convert, variable, scale, offset):
        """Takes a zonal_stats output and converts units of variables."""
        # Constants
        j_per_kj = 1000  # [J kJ-1]
        seconds_per_day = 24*60*60  # [s day-1]

        # Handle scale and offset
        if scale is None: scale = 1
        if offset is None: offset = 0

        if (scale != 1) or (offset != 0):
            self.logger.error(f'Code needed to deal with scale {scale} and offset {offset}')
            return zonal_out

        # Select conversion factor
        if variable == 'srad':
            # From [kJ m-2 day-1] to [W m-2]:
            # [kJ m-2 day-1] * 1/[s day-1] * [J kJ-1] = [J m-2 s-1] = [W m-2]
            c_factor = 1/seconds_per_day * j_per_kj

        # Apply conversion to each list element
        for list_id in range(0, len(zonal_out)):
            zonal_dict = zonal_out[list_id]

            # Loop over dictionary entries
            for key, val in zonal_dict.items():
                if key in stat_to_convert:
                    zonal_out[list_id][key] = zonal_out[list_id][key] * c_factor

        return zonal_out
    
    def write_geotif_sameDomain(self, src_file, des_file, des_data, nodata_value=None):
        """Write a GeoTIFF file with the same domain as a source file."""
        # Enforce data type
        src_file = str(src_file)
        
        # Load the source file to get attributes
        src_ds = gdal.Open(src_file)
        
        # Get the geotransform
        des_transform = src_ds.GetGeoTransform()

        # Get metadata from source
        scale_factor = src_ds.GetRasterBand(1).GetScale()
        offset = src_ds.GetRasterBand(1).GetOffset()
        
        # Get dimensions
        ncols = des_data.shape[1]
        nrows = des_data.shape[0]
        
        # Create output file
        driver = gdal.GetDriverByName("GTiff")
        dst_ds = driver.Create(des_file, ncols, nrows, 1, gdal.GDT_Float32, options=['COMPRESS=DEFLATE'])
        dst_ds.GetRasterBand(1).WriteArray(des_data)

        # Set metadata
        if scale_factor: 
            dst_ds.GetRasterBand(1).SetScale(scale_factor)
        if offset: 
            dst_ds.GetRasterBand(1).SetOffset(offset)
        
        # Set the nodata value
        if nodata_value is not None:
            dst_ds.GetRasterBand(1).SetNoDataValue(nodata_value)

        # Set georeferencing
        dst_ds.SetGeoTransform(des_transform)
        wkt = src_ds.GetProjection()
        srs = osr.SpatialReference()
        srs.ImportFromWkt(wkt)
        dst_ds.SetProjection(srs.ExportToWkt())
        
        # Clean up
        src_ds = None
        dst_ds = None

    # Soil attribute functions
    def attributes_from_soilgrids(self, geo_folder, dataset, shp_str, l_values, l_index, case='lumped'):
        """Calculates attributes from SOILGRIDS maps."""
        self.logger.info(f"Extracting SOILGRIDS attributes using case: {case}")

        # File specification
        sub_folders = ['bdod', 'cfvo', 'clay', 'sand', 'silt', 'soc', 'porosity']
        units = ['cg cm^-3', 'cm^3 dm^-3', 'g kg^-1', 'g kg^-1', 'g kg^-1', 'dg kg^-1', '-']
        depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
        fields = ['mean', 'uncertainty']
        stats = ['mean', 'min', 'max', 'std']
        
        # Process standard attributes
        for sub_folder, unit in zip(sub_folders, units):
            for depth in depths:
                for field in fields:
                    tif = str(geo_folder / dataset / 'raw' / f'{sub_folder}' / f'{sub_folder}_{depth}_{field}.tif')
                    if not os.path.exists(tif): 
                        continue  # porosity has no uncertainty field
                    
                    zonal_out = zonal_stats(shp_str, tif, stats=stats, all_touched=True)
                    scale, offset = self.read_scale_and_offset(tif)
                    l_values = self.update_values_list(l_values, stats, zonal_out, scale, offset, case=case)
                    l_index += [('Soil', f'{sub_folder}_{depth}_{field}_min',  f'{unit}', 'SOILGRIDS'),
                                ('Soil', f'{sub_folder}_{depth}_{field}_mean', f'{unit}', 'SOILGRIDS'),
                                ('Soil', f'{sub_folder}_{depth}_{field}_max',  f'{unit}', 'SOILGRIDS'),
                                ('Soil', f'{sub_folder}_{depth}_{field}_std',  f'{unit}', 'SOILGRIDS')]
           
        # Process conductivity with harmonic mean
        sub_folder = 'conductivity'
        unit = 'cm hr^-1'
        field = 'mean'  # no uncertainty maps for these
        stats = ['min', 'max', 'std']

        for depth in depths:
            tif = str(geo_folder / dataset / 'raw' / f'{sub_folder}' / f'{sub_folder}_{depth}_{field}.tif')
            if not os.path.exists(tif): 
                continue
            
            zonal_out = zonal_stats(shp_str, tif, stats=stats, 
                                  add_stats={'harmonic_mean': self.harmonic_mean}, 
                                  all_touched=True)
            scale, offset = self.read_scale_and_offset(tif)
            l_values = self.update_values_list(l_values, ['min', 'max', 'std', 'harmonic_mean'], 
                                             zonal_out, scale, offset, case=case)
            l_index += [('Soil', f'{sub_folder}_{depth}_{field}_min',  f'{unit}', 'SOILGRIDS'),
                        ('Soil', f'{sub_folder}_{depth}_{field}_harmonic_mean', f'{unit}', 'SOILGRIDS'),
                        ('Soil', f'{sub_folder}_{depth}_{field}_max',  f'{unit}', 'SOILGRIDS'),
                        ('Soil', f'{sub_folder}_{depth}_{field}_std',  f'{unit}', 'SOILGRIDS')]

        return l_values, l_index
    
    def attributes_from_pelletier(self, geo_folder, dataset, shp_str, l_values, l_index, case='lumped'):
        """Calculates statistics for Pelletier maps."""
        # General
        files = ['upland_hill-slope_regolith_thickness.tif',
                'upland_hill-slope_soil_thickness.tif',
                'upland_valley-bottom_and_lowland_sedimentary_deposit_thickness.tif',
                'average_soil_and_sedimentary-deposit_thickness.tif']
        attrs = ['regolith_thickness', 'soil_thickness',
                'sedimentary_thickness', 'average_thickness']
        units = ['m', 'm', 'm', 'm']
        stats = ['mean', 'min', 'max', 'std']

        for file, att, unit in zip(files, attrs, units):
            tif = str(geo_folder / dataset / 'raw' / file)
            self.check_and_set_nodata_value_on_pelletier_file(tif)
            zonal_out = zonal_stats(shp_str, tif, stats=stats, all_touched=True)
            zonal_out = self.check_zonal_stats_outcomes(zonal_out, new_val=0)
            scale, offset = self.read_scale_and_offset(tif)
            l_values = self.update_values_list(l_values, stats, zonal_out, scale, offset, case=case)
            l_index += [('Soil', f'{att}_min',  f'{unit}', 'Pelletier'),
                        ('Soil', f'{att}_mean', f'{unit}', 'Pelletier'),
                        ('Soil', f'{att}_max',  f'{unit}', 'Pelletier'),
                        ('Soil', f'{att}_std',  f'{unit}', 'Pelletier')]
            
        return l_values, l_index
    
    # Merit attributes
    def attributes_from_merit(self, geo_folder, dataset, shp_str, riv_str, row, l_values, l_index, equal_area_crs='ESRI:102008', case='lumped'):
        """Calculates topographic attributes from MERIT data."""
        # Load the shapefile
        basin = gpd.read_file(shp_str)

        ## Known values
        if case == 'lumped':
            clon = basin.to_crs(equal_area_crs).centroid.to_crs('EPSG:4326').x.iloc[0]  # lon
            clat = basin.to_crs(equal_area_crs).centroid.to_crs('EPSG:4326').y.iloc[0]  # lat
            
            if row is not None:
                area = row.get('Basin_area_km2')
                slat = row.get('Station_lat')
                slon = row.get('Station_lon')
            else:
                # Calculate area if not provided
                area = basin.to_crs(equal_area_crs).area.sum() / 10**6  # km²
                slat = None
                slon = None
                
            l_values.append(clat)
            l_index.append(('Topography', 'centroid_lat', 'degrees', 'MERIT Hydro'))
            l_values.append(clon)
            l_index.append(('Topography', 'centroid_lon', 'degrees', 'MERIT Hydro'))
            l_values.append(area)
            l_index.append(('Topography', 'basin_area', 'km^2', 'MERIT Hydro'))
            
            if slat is not None:
                l_values.append(slat)
                l_index.append(('Topography', 'gauge_lat', 'degrees', 'USGS/WSC'))
            if slon is not None:
                l_values.append(slon)
                l_index.append(('Topography', 'gauge_lon', 'degrees', 'USGS/WSC'))
        
        elif case == 'distributed':
            num_poly = len(basin)
            for i_poly in range(num_poly):
                clon = basin.to_crs(equal_area_crs).centroid.to_crs('EPSG:4326').x.iloc[i_poly]  # lon
                clat = basin.to_crs(equal_area_crs).centroid.to_crs('EPSG:4326').y.iloc[i_poly]  # lat
                area = (basin.to_crs(equal_area_crs).area / 10**6).iloc[i_poly]
                l_values[i_poly].append(clat)
                l_values[i_poly].append(clon)
                l_values[i_poly].append(area)
            l_index.append(('Topography', 'centroid_lat', 'degrees', 'MERIT Hydro'))
            l_index.append(('Topography', 'centroid_lon', 'degrees', 'MERIT Hydro'))
            l_index.append(('Topography', 'basin_area', 'km^2', 'MERIT Hydro'))

        ## RASTERS
        # Slope and elevation from zonal stats
        files = [str(geo_folder / dataset / 'raw' / 'merit_hydro_elv.tif'),
                str(geo_folder / dataset / 'slope' / 'merit_hydro_slope.tif')]
        attrs = ['elev', 'slope']
        units = ['m.a.s.l.', 'degrees']
        stats = ['mean', 'min', 'max', 'std']
        
        for tif, att, unit in zip(files, attrs, units):
            zonal_out = zonal_stats(shp_str, tif, stats=stats, all_touched=True)
            scale, offset = self.read_scale_and_offset(tif)
            l_values = self.update_values_list(l_values, stats, zonal_out, scale, offset, case=case)
            l_index += [('Topography', f'{att}_min',  f'{unit}', 'MERIT Hydro'),
                        ('Topography', f'{att}_mean', f'{unit}', 'MERIT Hydro'),
                        ('Topography', f'{att}_max',  f'{unit}', 'MERIT Hydro'),
                        ('Topography', f'{att}_std',  f'{unit}', 'MERIT Hydro')]

        # Aspect needs circular stats
        stats = ['min', 'max']
        tif = str(geo_folder / dataset / 'aspect' / 'merit_hydro_aspect.tif')
        zonal_out = zonal_stats(shp_str, tif, stats=stats, add_stats={'circ_mean': self.calc_circmean,
                                                                      'circ_std': self.calc_circstd}, all_touched=True)
        scale, offset = self.read_scale_and_offset(tif)
        l_values = self.update_values_list(l_values, ['min', 'max', 'circ_mean', 'circ_std'], zonal_out, scale, offset, case=case)
        l_index += [('Topography', 'aspect_min',   'degrees', 'MERIT Hydro'),
                    ('Topography', 'aspect_mean',  'degrees', 'MERIT Hydro'),
                    ('Topography', 'aspect_max',   'degrees', 'MERIT Hydro'),
                    ('Topography', 'aspect_std',   'degrees', 'MERIT Hydro')]
        
        ## VECTOR - calculate river attributes
        merit_comids = None
        if os.path.exists(riv_str):
            try:
                river = gpd.read_file(riv_str)
                # Process river network attributes based on case
                if case == 'lumped':
                    # Calculate stream lengths, density, etc.
                    stream_lengths = []
                    stream_total = np.nan
                    min_length = np.nan
                    mean_length = np.nan
                    max_length = np.nan
                    std_length = np.nan
                    riv_order = np.nan
                    density = np.nan
                    elongation = np.nan
                    
                    if 'COMID' in river.columns:
                        river = river.set_index('COMID')
                        
                        if len(river) > 0 and 'new_len_km' in river.columns:
                            # Find stream lengths
                            headwaters = river[(river.get('maxup', 0) == 0) | river['maxup'].isna()]
                            for COMID in headwaters.index:
                                stream_length = 0
                                curr_id = COMID
                                while curr_id in river.index and 'NextDownID' in river.columns:
                                    stream_length += river.loc[curr_id]['new_len_km']
                                    curr_id = river.loc[curr_id]['NextDownID']
                                stream_lengths.append(stream_length)
                            
                            # Calculate statistics
                            if stream_lengths:
                                stream_lengths = np.array(stream_lengths)
                                stream_total = river['new_len_km'].sum()
                                min_length = stream_lengths.min()
                                mean_length = stream_lengths.mean()
                                max_length = stream_lengths.max()
                                std_length = stream_lengths.std()
                                if 'order' in river.columns:
                                    riv_order = river['order'].max()
                                density = stream_total/area
                                elongation = 2*np.sqrt(area/np.pi)/max_length
                    
                    # Add river attributes to lists
                    l_values.append(min_length)
                    l_index.append(('Topography', 'stream_length_min', 'km', 'MERIT Hydro Basins'))
                    l_values.append(mean_length)
                    l_index.append(('Topography', 'stream_length_mean', 'km', 'MERIT Hydro Basins'))
                    l_values.append(max_length)
                    l_index.append(('Topography', 'stream_length_max', 'km', 'MERIT Hydro Basins'))
                    l_values.append(std_length)
                    l_index.append(('Topography', 'stream_length_std', 'km', 'MERIT Hydro Basins'))
                    l_values.append(stream_total)
                    l_index.append(('Topography', 'segment_length_total', 'km', 'MERIT Hydro Basins'))
                    l_values.append(riv_order)
                    l_index.append(('Topography', 'stream_order_max', '-', 'MERIT Hydro Basins'))
                    l_values.append(density)
                    l_index.append(('Topography', 'stream_density', 'km^-1', 'MERIT Hydro, MERIT Hydro Basins'))
                    l_values.append(elongation)
                    l_index.append(('Topography', 'elongation_ratio', '-', 'MERIT Hydro, MERIT Hydro Basins'))
            
                else:  # distributed case
                    if 'COMID' in basin.columns:
                        merit_comids = basin['COMID'].values
                        
                    # Calculate stream attributes for each HRU
                    for i_poly in range(len(basin)):
                        if 'COMID' in basin.columns and 'COMID' in river.columns:
                            comid = basin.iloc[i_poly]['COMID']
                            river_subset = river[river['COMID'] == comid]
                            
                            if len(river_subset) > 0:
                                # Extract river attributes
                                stream_length = river_subset['new_len_km'].values[0] if 'new_len_km' in river_subset.columns else np.nan
                                slope = river_subset['slope'].values[0] if 'slope' in river_subset.columns else np.nan
                                upstream_area = river_subset['uparea'].values[0] if 'uparea' in river_subset.columns else np.nan
                                hru_area = basin.to_crs(equal_area_crs).area.iloc[i_poly] / 10**6
                                density = stream_length/hru_area if not np.isnan(stream_length) else np.nan
                                elongation = 2*np.sqrt(hru_area/np.pi)/stream_length if not np.isnan(stream_length) else np.nan
                            else:
                                stream_length = np.nan
                                slope = np.nan
                                upstream_area = np.nan
                                density = np.nan
                                elongation = np.nan
                        else:
                            stream_length = np.nan
                            slope = np.nan
                            upstream_area = np.nan
                            density = np.nan
                            elongation = np.nan
                        
                        # Add to values list
                        l_values[i_poly].append(stream_length)
                        l_values[i_poly].append(slope)
                        l_values[i_poly].append(upstream_area)
                        l_values[i_poly].append(density)
                        l_values[i_poly].append(elongation)
                    
                    # Add attribute names to index
                    l_index += [('Topography', 'stream_length',    'km',     'MERIT Hydro Basins'),
                                ('Topography', 'stream_slope',     'm m^-1', 'MERIT Hydro Basins'),
                                ('Topography', 'upstream_area',    'km^2',   'MERIT Hydro Basins'),
                                ('Topography', 'stream_density',   'km^-1',  'MERIT Hydro, MERIT Hydro Basins'),
                                ('Topography', 'elongation_ratio', '-',      'MERIT Hydro, MERIT Hydro Basins')]
            except Exception as e:
                self.logger.error(f"Error processing river network: {str(e)}")
                # Add placeholder values if error occurs
                if case == 'lumped':
                    for _ in range(8):
                        l_values.append(np.nan)
                else:
                    for i_poly in range(len(basin)):
                        for _ in range(5):
                            l_values[i_poly].append(np.nan)
                    
                    l_index += [('Topography', 'stream_length',    'km',     'MERIT Hydro Basins'),
                                ('Topography', 'stream_slope',     'm m^-1', 'MERIT Hydro Basins'),
                                ('Topography', 'upstream_area',    'km^2',   'MERIT Hydro Basins'),
                                ('Topography', 'stream_density',   'km^-1',  'MERIT Hydro, MERIT Hydro Basins'),
                                ('Topography', 'elongation_ratio', '-',      'MERIT Hydro, MERIT Hydro Basins')]
        
        return l_values, l_index, merit_comids
    
    # Land cover attributes
    def attributes_from_modis_land(self, geo_folder, dataset, shp_str, l_values, l_index, case='lumped'):
        """Calculates percentage occurrence of all classes in MODIS IGBP map."""
        tif_path = geo_folder / dataset / 'raw' / '2001_2022_mode_MCD12Q1_LC_Type1.tif'
        if not tif_path.exists():
            self.logger.warning(f"MODIS land cover file not found at {tif_path}")
            return l_values, l_index
            
        zonal_out = zonal_stats(str(tif_path), shp_str, categorical=True, all_touched=True)
        self.check_scale_and_offset(tif_path)
        l_values, l_index = self.update_values_list_with_categorical(l_values, l_index, zonal_out, 'MCD12Q1.061', prefix='lc2_', case=case)
        return l_values, l_index
    
    def attributes_from_glclu2019(self, geo_folder, dataset, shp_str, l_values, l_index, case='lumped'):
        """Calculates percentage occurrence of all classes in GLCLU2019 map."""
        tif_path = geo_folder / dataset / 'raw' / 'glclu2019_map.tif'
        if not tif_path.exists():
            self.logger.warning(f"GLCLU2019 file not found at {tif_path}")
            return l_values, l_index
            
        zonal_out = zonal_stats(str(tif_path), shp_str, categorical=True, all_touched=True)
        self.check_scale_and_offset(tif_path)
        l_values, l_index = self.update_values_list_with_categorical(l_values, l_index, zonal_out, 'GLCLU 2019', prefix='lc1_', case=case)
        return l_values, l_index
    
    # WorldClim methods
    def attributes_from_worldclim(self, geo_folder, dataset, shp_str, l_values, l_index, case='lumped'):
        """Calculates mean and stdv for tifs of monthly WorldClim values."""
        # Define file locations
        clim_folder = geo_folder / dataset / 'raw'
        sub_folders = ['prec', 'srad', 'tavg', 'tmax', 'tmin', 'vapr', 'wind', 'pet', 'aridity2', 'fracsnow2']
        sub_folder_units = ['mm', 'W m^-2', 'C', 'C', 'C', 'kPa', 'm s^-1', 'mm', '-', '-']

        # Get the annual values
        l_values, l_index = self.get_annual_worldclim_attributes(clim_folder, shp_str, 'prec', 'tavg', 'pet', 'snow2', l_values, l_index, case)

        # Loop over the files and calculate the stats
        for sub_folder, sub_folder_unit in zip(sub_folders, sub_folder_units):
            folder_path = clim_folder / sub_folder
            if not folder_path.exists():
                continue
                
            month_files = sorted(glob.glob(str(folder_path / '*.tif')))
            for month_file in month_files:
                stats = ['mean', 'std']
                zonal_out = zonal_stats(shp_str, month_file, stats=stats, all_touched=True)
                
                scale, offset = self.read_scale_and_offset(month_file)
                if sub_folder == 'srad':
                    zonal_out = self.zonal_stats_unit_conversion(zonal_out, stats, 'srad', scale, offset)

                l_values = self.update_values_list(l_values, stats, zonal_out, scale, offset, case=case)
                
                month = os.path.basename(month_file).split('_')[3].split('.')[0]
                var = os.path.basename(month_file).split('_')[2]
                source = 'WorldClim'
                if var == 'pet': 
                    source = 'WorldClim (derived, Oudin et al., 2005)'
                    var = 'pet2'  # distinguish from other PET sources
                l_index += [('Climate', f'{var}_mean_month_{month}', f'{sub_folder_unit}', source),
                            ('Climate', f'{var}_std_month_{month}', f'{sub_folder_unit}', source)]

        return l_values, l_index
    
    def get_annual_worldclim_attributes(self, clim_folder, shp_str, prec_folder, tavg_folder, pet_folder, snow_folder, l_values, l_index, case='lumped'):
        """Calculates annual WorldClim statistics."""
        # Create the output folder
        ann_folder = clim_folder / 'annual'
        ann_folder.mkdir(exist_ok=True, parents=True)

        # General settings
        stats = ['mean', 'std']

        num_nested_lists = 0
        if case == 'distributed':
            num_nested_lists = len(l_values)
        
        # --- P
        prec_files = sorted(glob.glob(str(clim_folder / prec_folder / '*.tif')))
        if prec_files:
            annual_prec = self.create_annual_worldclim_map(prec_files, ann_folder, 'prec_sum.tif', 'prec')
            zonal_out = zonal_stats(shp_str, annual_prec, stats=stats, all_touched=True)
            
            # Update scale and offset
            scale, offset = self.read_scale_and_offset(annual_prec)
            if scale is None: scale = 1
            if offset is None: offset = 0

            # Update lists
            for stat in stats:
                if case == 'lumped':
                    l_values.append(zonal_out[0][stat] * scale + offset)
                else:
                    for i in range(0, num_nested_lists):
                        l_values[i].append(zonal_out[i][stat] * scale + offset)
                l_index.append(('Climate', f'prec_{stat}', 'mm', 'WorldClim'))

            # --- PET
            pet_files = sorted(glob.glob(str(clim_folder / pet_folder / '*.tif')))
            if pet_files:
                annual_pet = self.create_annual_worldclim_map(pet_files, ann_folder, 'pet_sum.tif', 'pet')
                zonal_out = zonal_stats(shp_str, annual_pet, stats=stats, all_touched=True)
                
                scale, offset = self.read_scale_and_offset(annual_pet)
                if scale is None: scale = 1
                if offset is None: offset = 0
                    
                for stat in stats:
                    if case == 'lumped':
                        l_values.append(zonal_out[0][stat] * scale + offset)
                    else:
                        for i in range(0, num_nested_lists):
                            l_values[i].append(zonal_out[i][stat] * scale + offset)
                    l_index.append(('Climate', f'pet2_{stat}', 'mm', 'WorldClim'))
            
            # --- T
            tavg_files = sorted(glob.glob(str(clim_folder / tavg_folder / '*.tif')))
            if tavg_files:
                annual_tavg = self.create_annual_worldclim_map(tavg_files, ann_folder, 't_avg.tif', 'tavg')
                zonal_out = zonal_stats(shp_str, annual_tavg, stats=stats, all_touched=True)
                
                scale, offset = self.read_scale_and_offset(annual_tavg)
                if scale is None: scale = 1
                if offset is None: offset = 0
                    
                for stat in stats:
                    if case == 'lumped':
                        l_values.append(zonal_out[0][stat] * scale + offset)
                    else:
                        for i in range(0, num_nested_lists):
                            l_values[i].append(zonal_out[i][stat] * scale + offset)
                    l_index.append(('Climate', f'tavg_{stat}', 'C', 'WorldClim'))

            # --- Snow, Aridity, Seasonality
            snow_files = sorted(glob.glob(str(clim_folder / snow_folder / '*.tif')))
            if snow_files and pet_files:
                annual_snow = self.create_annual_worldclim_map(snow_files, ann_folder, 'snow_sum.tif', 'snow')
                annual_ari = self.derive_annual_worldclim_aridity(annual_prec, annual_pet, ann_folder, 'aridity.tif')
                annual_fs = self.derive_annual_worldclim_fracsnow(annual_prec, annual_snow, ann_folder, 'fracsnow.tif')
                annual_seas = self.derive_annual_worldclim_seasonality(prec_files, tavg_files, ann_folder, 'seasonality.tif')
                
                # Process these derived attributes
                derived_maps = [
                    (annual_ari, 'aridity2', '-'),
                    (annual_seas, 'seasonality2', '-'),
                    (annual_fs, 'fracsnow2', '-')
                ]
                
                for map_path, attr_name, unit in derived_maps:
                    zonal_out = zonal_stats(shp_str, map_path, stats=stats, all_touched=True)
                    scale, offset = self.read_scale_and_offset(map_path)
                    if scale is None: scale = 1
                    if offset is None: offset = 0
                    
                    for stat in stats:
                        if case == 'lumped':
                            l_values.append(zonal_out[0][stat] * scale + offset)
                        else:
                            for i in range(0, num_nested_lists):
                                l_values[i].append(zonal_out[i][stat] * scale + offset)
                        l_index.append(('Climate', f'{attr_name}_{stat}', unit, 'WorldClim'))
        
        return l_values, l_index
    
    def create_annual_worldclim_map(self, files, output_folder, output_name, var):
        """Creates an annual map from monthly WorldClim data."""
        # Load the data and mask no-data values
        datasets = [self.get_geotif_data_as_array(file) for file in files]
        no_datas = [self.get_geotif_nodata_value(file) for file in files]
        masked_data = [np.ma.masked_array(data, mask=data==no_data) for data, no_data in zip(datasets, no_datas)]

        # Create stacked array for computations
        stacked_array = np.ma.dstack(masked_data)

        # Create the annual value depending on variable
        if var in ['prec', 'pet', 'snow']:
            res_array = np.ma.sum(stacked_array, axis=2)  # Sum across months
        elif var in ['tavg']:
            res_array = np.ma.mean(stacked_array, axis=2)  # Mean across months

        # Apply no-data value and write to file
        res_array = res_array.filled(no_datas[0])
        output_path = str(output_folder/output_name)
        self.write_geotif_sameDomain(files[0], output_path, res_array, nodata_value=no_datas[0])
        
        return output_path
    
    def derive_annual_worldclim_aridity(self, annual_prec, annual_pet, output_folder, output_name):
        """Creates an annual aridity map from annual P and PET maps."""
        # Load the data and mask no-data values
        p_data = self.get_geotif_data_as_array(annual_prec)
        no_data = self.get_geotif_nodata_value(annual_prec)
        masked_p = np.ma.masked_array(p_data, mask=p_data==no_data)
        
        pet_data = self.get_geotif_data_as_array(annual_pet)
        pet_no_data = self.get_geotif_nodata_value(annual_pet)
        masked_pet = np.ma.masked_array(pet_data, mask=pet_data==pet_no_data)

        # Build in a failsafe for zero P
        masked_p = np.where(masked_p == 0, 1, masked_p)  # avoid divide by zero
        
        # Calculate aridity (PET/P)
        ari = np.ma.divide(masked_pet, masked_p)

        # Apply no-data value and write to file
        ari = ari.filled(no_data)
        output_path = str(output_folder/output_name)
        self.write_geotif_sameDomain(annual_prec, output_path, ari, nodata_value=no_data)
        
        return output_path
    
    def derive_annual_worldclim_fracsnow(self, annual_prec, annual_snow, output_folder, output_name):
        """Creates an annual fraction snow map from annual P and snow maps."""
        # Load the data and mask no-data values
        p_data = self.get_geotif_data_as_array(annual_prec)
        no_data = self.get_geotif_nodata_value(annual_prec)
        masked_p = np.ma.masked_array(p_data, mask=p_data==no_data)
        
        snow_data = self.get_geotif_data_as_array(annual_snow)
        snow_no_data = self.get_geotif_nodata_value(annual_snow)
        masked_snow = np.ma.masked_array(snow_data, mask=snow_data==snow_no_data)

        # Build in a failsafe for zero P
        masked_p = np.where(masked_p == 0, 1, masked_p)
        
        # Calculate fraction snow
        fsnow = np.ma.divide(masked_snow, masked_p)

        # Apply no-data value and write to file
        fsnow = fsnow.filled(no_data)
        output_path = str(output_folder/output_name)
        self.write_geotif_sameDomain(annual_prec, output_path, fsnow, nodata_value=no_data)
        
        return output_path
    
    def derive_annual_worldclim_seasonality(self, prec_files, tavg_files, output_folder, output_name):
        """Calculates a map of seasonality values using WorldClim precipitation and temperature."""
        # Define sine functions for P and T
        def sine_function_temp(x, mean, delta, phase, period=1):
            """Sine function for temperature."""
            result = mean + delta * np.sin(2*np.pi*(x-phase)/period)
            return result

        def sine_function_prec(x, mean, delta, phase, period=1):
            """Sine function for precipitation."""
            result = mean * (1 + delta * np.sin(2*np.pi*(x-phase)/period))
            result = np.where(result < 0, 0, result)
            return result

        # Functions to fit sinusoids along time dimension
        def fit_temp_sines(t):
            """Fit temperature sine curves."""
            # Short-circuit the computation if we have only masked values
            if np.ma.getmaskarray(t).all():
                return [np.nan, np.nan, np.nan]
            
            # Define the x-coordinate as daily steps as fractions of a year
            x = np.arange(0, len(t))/len(t)  # days as fraction of a year
            
            # Fit the temperature sine
            t_mean = float(t.mean())
            t_delta = float(t.max()-t.min())
            t_phase = 0.5
            initial_guess = [t_mean, t_delta, t_phase]
            try:
                t_pars, _ = curve_fit(sine_function_temp, x, t, p0=initial_guess)
                return t_pars
            except:
                return [np.nan, np.nan, np.nan]

        def fit_prec_sines(p):
            """Fit precipitation sine curves."""
            # Short-circuit the computation if we have only masked values
            if np.ma.getmaskarray(p).all():
                return [np.nan, np.nan, np.nan]
            
            # Define the x-coordinate as daily steps as fractions of a year
            x = np.arange(0, len(p))/len(p)  # days as fraction of a year

            # Fit the precipitation sine
            p_mean = float(p.mean())
            p_delta = float(p.max()-p.min()) / float(p.mean()) if p.mean() != 0 else 0
            p_phase = 0.5
            initial_guess = [p_mean, p_delta, p_phase]
            try:
                p_pars, _ = curve_fit(sine_function_prec, x, p, p0=initial_guess)
                return p_pars
            except:
                return [np.nan, np.nan, np.nan]

        # Get the precip data
        datasets = [self.get_geotif_data_as_array(file) for file in prec_files]
        no_datas = [self.get_geotif_nodata_value(file) for file in prec_files]
        masked_data = [np.ma.masked_array(data, mask=data==no_data) for data, no_data in zip(datasets, no_datas)]
        stacked_prec = np.ma.dstack(masked_data)

        # Get the tavg data
        datasets = [self.get_geotif_data_as_array(file) for file in tavg_files]
        no_datas = [self.get_geotif_nodata_value(file) for file in tavg_files]
        masked_data = [np.ma.masked_array(data, mask=data==no_data) for data, no_data in zip(datasets, no_datas)]
        stacked_tavg = np.ma.dstack(masked_data)

        # Fit the sine curves in the third dimension
        t_pars = np.apply_along_axis(fit_temp_sines, axis=2, arr=stacked_tavg)
        p_pars = np.apply_along_axis(fit_prec_sines, axis=2, arr=stacked_prec)

        # Calculate the dimensionless seasonality index in space as per Eq. 14 in Woods (2009)
        seasonality = p_pars[:,:,1]*np.sign(t_pars[:,:,1])*np.cos(2*np.pi*(p_pars[:,:,2]-t_pars[:,:,2])/1)

        # Apply no-data value and write to file
        output_path = str(output_folder/output_name)
        self.write_geotif_sameDomain(prec_files[0], output_path, seasonality, nodata_value=np.nan)
         
        return output_path
    
    def aridity_and_fraction_snow_from_worldclim(self, geo_folder, dataset, overwrite=False):
        """Calculates aridity and fraction snow maps from WorldClim data."""
        # Find files
        clim_folder = geo_folder / dataset / 'raw'
        prc_files = sorted(glob.glob(str(clim_folder / 'prec' / '*.tif')))  # [mm]
        pet_files = sorted(glob.glob(str(clim_folder / 'pet' / '*.tif')))  # [mm]
        tmp_files = sorted(glob.glob(str(clim_folder / 'tavg' / '*.tif')))  # [C]
        
        # Make the output locations
        ari_folder = clim_folder / 'aridity2'
        ari_folder.mkdir(parents=True, exist_ok=True)
        snow_folder = clim_folder / 'snow2'
        snow_folder.mkdir(parents=True, exist_ok=True)    
        fsnow_folder = clim_folder / 'fracsnow2'
        fsnow_folder.mkdir(parents=True, exist_ok=True)

        # Loop over files and calculate aridity
        for prc_file, pet_file, tmp_file in zip(prc_files, pet_files, tmp_files):
            # Define output file names
            ari_name = os.path.basename(prc_file).replace('prec', 'aridity2')
            ari_file = str(ari_folder / ari_name)
            fsnow_name = os.path.basename(prc_file).replace('prec', 'fracsnow2')
            fsnow_file = str(fsnow_folder / fsnow_name)
            snow_name = os.path.basename(prc_file).replace('prec', 'snow2')
            snow_file = str(snow_folder / snow_name)

            # Check if already processed
            if os.path.isfile(ari_file) and os.path.isfile(fsnow_file) and os.path.isfile(snow_file) and not overwrite:
                continue  # skip already processed files

            # Load data
            prc_path = str(clim_folder / 'prec' / os.path.basename(prc_file))
            pet_path = str(clim_folder / 'pet' / os.path.basename(pet_file))
            tmp_path = str(clim_folder / 'tavg' / os.path.basename(tmp_file))
            prc = self.get_geotif_data_as_array(prc_path)  # [mm]
            pet = self.get_geotif_data_as_array(pet_path)  # [mm]
            tmp = self.get_geotif_data_as_array(tmp_path)  # [C]

            # Calculate variables
            snow = np.where(tmp < 0, prc, 0)  # areas with temp < 0°C get snow instead of rain
            if (prc == 0).any():
                prc[prc == 0] = 1  # add 1 mm to avoid divide by zero errors
            ari = pet/prc  # [-]
            frac_snow = snow/prc  # [-]

            # Write to disk
            self.write_geotif_sameDomain(prc_path, ari_file, ari)
            self.write_geotif_sameDomain(prc_path, fsnow_file, frac_snow)
            self.write_geotif_sameDomain(prc_path, snow_file, snow)
        
        self.logger.info("Created aridity and fraction snow maps from WorldClim data")
    
    def oudin_pet_from_worldclim(self, geo_folder, dataset, debug=False, overwrite=False):
        """
        Calculates PET estimates from WorldClim data, using the Oudin (2005) formulation.
        
        Args:
            geo_folder: Path to geospatial data folder
            dataset: Dataset name
            debug: Flag for debug output
            overwrite: Flag to overwrite existing files
        """
        # Constants
        lh = 2.45  # latent heat flux, MJ kg-1
        rw = 1000  # rho water, kg m-3
        days_per_month = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # days month-1
        mm_per_m = 1000  # mm m-1
        
        # Find files
        clim_folder = geo_folder / dataset / 'raw'
        srad_files = sorted(glob.glob(str(clim_folder / 'srad' / '*.tif')))
        tavg_files = sorted(glob.glob(str(clim_folder / 'tavg' / '*.tif')))

        # Make the output location
        pet_folder = clim_folder / 'pet'
        pet_folder.mkdir(parents=True, exist_ok=True)

        # Loop over files and calculate PET
        for srad_file, tavg_file in zip(srad_files, tavg_files):
            # Define output file name
            srad_basename = os.path.basename(srad_file)
            pet_name = srad_basename.replace('srad', 'pet')
            pet_file = str(pet_folder / pet_name)

            # Check if already processed
            if os.path.isfile(pet_file) and not overwrite:
                continue  # skip already processed files

            # Define month
            month = srad_basename.split('_')[-1].split('.')[0]  # 'wc2.1_30s_srad_01.tif' > '01'
            month_ix = int(month)-1  # zero-based indexing
            
            # Load data
            srad_path = str(clim_folder / 'srad' / srad_basename)
            tavg_path = str(clim_folder / 'tavg' / os.path.basename(tavg_file))
            srad = self.get_geotif_data_as_array(srad_path) / 1000  # [kJ m-2 day-1] / 1000 = [MJ m-2 day-1]
            tavg = self.get_geotif_data_as_array(tavg_path)
            
            # Oudin et al, 2005, Eq. 3
            pet = np.where(tavg+5 > 0, (srad / (lh*rw)) * ((tavg+5)/100) * mm_per_m, 0)  # m day-1 > mm day-1
            pet_month = pet * days_per_month[month_ix]  # mm month-1
            if debug: 
                self.logger.debug(f'Calculating monthly PET for month {month} at day-index {month_ix}')

            # Write to disk
            self.write_geotif_sameDomain(srad_path, pet_file, pet_month)
        
        self.logger.info("Created PET estimates from WorldClim data using Oudin method")
    
    def process_all_attributes(self):
        """Process all catchment attributes including the newly added ones."""
        self.logger.info("Starting attribute processing")
        
        # Determine if we are processing lumped or distributed attributes
        attribute_type = 'lumped' if self.is_lumped else 'distributed'
        self.logger.info(f"Processing {attribute_type} attributes")
        
        try:
            # Initialize attribute collection
            if self.is_lumped:
                l_values = []  # For lumped case, single list of values
                l_index = []   # List to track attribute metadata
            else:
                # Read catchment to determine number of HRUs
                catchment = gpd.read_file(self.catchment_path)
                num_hrus = len(catchment)
                l_values = [[] for _ in range(num_hrus)]  # List of lists for distributed case
                l_index = []  # Single index list for all HRUs
            
            # Process attributes from different sources
            self.logger.info("Processing soil attributes")
            l_values, l_index = self._process_soil_attributes(l_values, l_index)
            
            self.logger.info("Processing topographic attributes")
            l_values, l_index = self._process_topographic_attributes(l_values, l_index)
            
            self.logger.info("Processing land cover attributes")
            l_values, l_index = self._process_landcover_attributes(l_values, l_index)
            
            self.logger.info("Processing climate attributes")
            l_values, l_index = self._process_climate_attributes(l_values, l_index)
            
            # Add new attribute processing calls
            self.logger.info("Processing geology attributes")
            l_values, l_index = self._process_geology_attributes(l_values, l_index)
            
            self.logger.info("Processing hydrology attributes")
            l_values, l_index = self._process_hydrology_attributes(l_values, l_index)
            
            self.logger.info("Processing open water attributes")
            l_values, l_index = self._process_openwater_attributes(l_values, l_index)
            
            # Convert to DataFrame and save
            attributes_df = self._convert_to_dataframe(l_values, l_index)
            
            # Save attributes
            output_file = self.attributes_dir / f"{self.domain_name}_attributes.csv"
            attributes_df.to_csv(output_file)
            self.logger.info(f"Attributes saved to {output_file}")
            
            return attributes_df
            
        except Exception as e:
            self.logger.error(f"Error processing attributes: {str(e)}")
            raise

    def _process_geology_attributes(self, l_values, l_index):
        """Process geology attributes from various sources."""
        # Get GLiM data path
        glim_path = self._get_data_path('ATTRIBUTES_GLIM_PATH', 'glim')
        
        # Process GLiM data if available
        if glim_path.exists():
            self.logger.info(f"Processing GLiM geological data from {glim_path}")
            l_values, l_index = self.attributes_from_glim(
                glim_path,
                'glim',
                str(self.catchment_path),
                l_values,
                l_index,
                case=self.case
            )
        else:
            self.logger.warning(f"GLiM geological data not found at {glim_path}")
        
        # Process USGS groundwater data if available
        gw_path = self._get_data_path('ATTRIBUTES_GROUNDWATER_PATH', 'groundwater')
        if gw_path.exists():
            self.logger.info(f"Processing groundwater data from {gw_path}")
            l_values, l_index = self.attributes_from_groundwater(
                gw_path,
                'groundwater',
                str(self.catchment_path),
                l_values,
                l_index,
                case=self.case
            )
        
        return l_values, l_index

    def _process_hydrology_attributes(self, l_values, l_index):
        """Process hydrology attributes from streamflow and gauge data."""
        # Get streamflow statistics 
        streamflow_path = self._get_data_path('ATTRIBUTES_STREAMFLOW_PATH', 'streamflow')
        
        if streamflow_path.exists():
            self.logger.info(f"Processing streamflow characteristics from {streamflow_path}")
            l_values, l_index = self.attributes_from_streamflow(
                streamflow_path,
                str(self.catchment_path),
                l_values,
                l_index,
                case=self.case
            )
        else:
            self.logger.warning(f"Streamflow data not found at {streamflow_path}")
        
        # Process stream gauge information if available
        gauge_path = self.project_dir / "observations" / "streamflow" / "preprocessed"
        gauge_file = gauge_path / f"{self.domain_name}_streamflow_processed.csv"
        
        if gauge_file.exists():
            self.logger.info(f"Processing streamflow gauge data from {gauge_file}")
            l_values, l_index = self.attributes_from_gauge_data(
                gauge_file,
                l_values,
                l_index,
                case=self.case
            )
        
        return l_values, l_index

    def _process_openwater_attributes(self, l_values, l_index):
        """Process open water attributes from lakes and wetlands data."""
        # Process Global Lakes and Wetlands Database
        glwd_path = self._get_data_path('ATTRIBUTES_GLWD_PATH', 'glwd')
        
        if glwd_path.exists():
            self.logger.info(f"Processing lakes and wetlands data from {glwd_path}")
            l_values, l_index = self.attributes_from_glwd(
                glwd_path,
                'glwd',
                str(self.catchment_path),
                l_values,
                l_index,
                case=self.case
            )
        else:
            self.logger.warning(f"Lakes and wetlands data not found at {glwd_path}")
        
        # Process HydroLAKES if available
        hydrolakes_path = self._get_data_path('ATTRIBUTES_HYDROLAKES_PATH', 'hydrolakes')
        if hydrolakes_path.exists():
            self.logger.info(f"Processing HydroLAKES data from {hydrolakes_path}")
            l_values, l_index = self.attributes_from_hydrolakes(
                hydrolakes_path,
                'hydrolakes',
                str(self.catchment_path),
                l_values,
                l_index,
                case=self.case
            )
        
        return l_values, l_index

    # Geological attribute processing methods
    def attributes_from_glim(self, geo_folder, dataset, shp_str, l_values, l_index, case='lumped'):
        """Calculate geological attributes from GLiM (Global Lithological Map)."""
        self.logger.info(f"Extracting GLiM attributes using case: {case}")
        
        # Path to GLiM classification raster
        glim_tif = geo_folder / dataset / 'raw' / 'glim_class.tif'
        
        if not glim_tif.exists():
            self.logger.warning(f"GLiM raster not found at {glim_tif}")
            return l_values, l_index
        
        # Geological classes from GLiM
        geology_classes = {
            1: 'metamorphic',
            2: 'sedimentary',
            3: 'igneous_plutonic',
            4: 'igneous_volcanic',
            5: 'unconsolidated_sediments',
            6: 'mixed_sedimentary',
            7: 'carbonate_rocks',
            8: 'siliciclastic_sedimentary',
            9: 'evaporites',
            10: 'pyroclastic',
            11: 'water_bodies',
            12: 'ice_and_glaciers',
            13: 'no_data'
        }
        
        # Calculate zonal statistics for categorical data
        zonal_out = zonal_stats(shp_str, str(glim_tif), categorical=True, all_touched=True)
        self.check_scale_and_offset(glim_tif)
        
        # Process results differently for lumped vs distributed case
        if case == 'lumped':
            # Find the total number of classified pixels
            total_pixels = 0
            for geo_id, count in zonal_out[0].items():
                if geo_id in geology_classes:  # Only count valid classes
                    total_pixels += count
            
            # Loop over all geology classes and calculate fractions
            for geo_id, geo_name in geology_classes.items():
                geo_fraction = 0
                if geo_id in zonal_out[0].keys():
                    geo_fraction = zonal_out[0][geo_id] / total_pixels if total_pixels > 0 else 0
                l_values.append(geo_fraction)
                l_index.append(('Geology', f'{geo_name}_fraction', '-', 'GLiM'))
        else:
            # Distributed case
            num_nested_lists = sum(1 for item in l_values if isinstance(item, list))
            assert num_nested_lists == len(zonal_out), f"zonal_out length does not match expected list length"
            
            # Process each polygon
            for i in range(num_nested_lists):
                # Calculate total pixels for this polygon
                total_pixels = 0
                for geo_id, count in zonal_out[i].items():
                    if geo_id in geology_classes:
                        total_pixels += count
                
                # Calculate fractions for each geology class
                for geo_id, geo_name in geology_classes.items():
                    geo_fraction = 0
                    if geo_id in zonal_out[i].keys():
                        geo_fraction = zonal_out[i][geo_id] / total_pixels if total_pixels > 0 else 0
                    l_values[i].append(geo_fraction)
            
            # Add index entries (only once)
            for geo_id, geo_name in geology_classes.items():
                l_index.append(('Geology', f'{geo_name}_fraction', '-', 'GLiM'))
        
        return l_values, l_index

    def attributes_from_groundwater(self, geo_folder, dataset, shp_str, l_values, l_index, case='lumped'):
        """Calculates groundwater attributes from USGS or other groundwater data."""
        self.logger.info(f"Extracting groundwater attributes using case: {case}")
        
        # Files and attributes
        files = [
            'gw_depth.tif',  # Groundwater depth
            'gw_capacity.tif',  # Aquifer storage capacity
            'gw_transmissivity.tif'  # Transmissivity
        ]
        attrs = ['gw_depth', 'gw_capacity', 'gw_transmissivity']
        units = ['m', 'm^3/m^2', 'm^2/day']
        stats = ['mean', 'min', 'max', 'std']
        
        for file, attr, unit in zip(files, attrs, units):
            tif = geo_folder / dataset / 'raw' / file
            
            if not tif.exists():
                self.logger.warning(f"Groundwater file not found: {tif}")
                continue
            
            # Calculate zonal statistics
            zonal_out = zonal_stats(shp_str, str(tif), stats=stats, all_touched=True)
            zonal_out = self.check_zonal_stats_outcomes(zonal_out)
            scale, offset = self.read_scale_and_offset(tif)
            
            # Update values and index
            l_values = self.update_values_list(l_values, stats, zonal_out, scale, offset, case=case)
            l_index += [('Geology', f'{attr}_min',  f'{unit}', 'Groundwater'),
                        ('Geology', f'{attr}_mean', f'{unit}', 'Groundwater'),
                        ('Geology', f'{attr}_max',  f'{unit}', 'Groundwater'),
                        ('Geology', f'{attr}_std',  f'{unit}', 'Groundwater')]
        
        return l_values, l_index

    # Hydrology attribute processing methods
    def attributes_from_streamflow(self, streamflow_path, shp_str, l_values, l_index, case='lumped'):
        """Calculate hydrological attributes from streamflow statistics."""
        # If this is a distributed case without a gauge for each HRU, we may need to 
        # relate HRUs to gauges or derive statistics from modeling results
        if case == 'distributed' and not self.config.get('HRU_GAUGE_MAPPING', False):
            self.logger.warning("Distributed streamflow attributes require HRU-gauge mapping or model results")
            return l_values, l_index
        
        # Find the streamflow statistics file
        stats_file = streamflow_path / f"{self.domain_name}_flow_metrics.csv"
        
        if not stats_file.exists():
            # Try to calculate flow metrics if available streamflow data exists
            obs_file = self.project_dir / "observations" / "streamflow" / "preprocessed" / f"{self.domain_name}_streamflow_processed.csv"
            
            if obs_file.exists():
                self.logger.info(f"Calculating flow metrics from {obs_file}")
                stats_df = self.calculate_flow_metrics(obs_file)
                stats_df.to_csv(stats_file)
            else:
                self.logger.warning(f"No streamflow statistics or observations found for {self.domain_name}")
                return l_values, l_index
        else:
            stats_df = pd.read_csv(stats_file)
        
        # Process flow metrics
        if case == 'lumped':
            # Add each flow metric to our attributes
            for col in stats_df.columns:
                if col != 'Basin_ID':  # Skip identifier column
                    value = stats_df.iloc[0][col]
                    attr_parts = col.split('_')
                    if len(attr_parts) >= 2:
                        metric_name = '_'.join(attr_parts[:-1])
                        unit = self._get_flow_metric_unit(metric_name)
                        l_values.append(value)
                        l_index.append(('Hydrology', col, unit, 'Streamflow Analysis'))
        else:
            # Handle distributed case if we have a mapping
            gauge_mapping = self.config.get('HRU_GAUGE_MAPPING', {})
            if gauge_mapping:
                num_nested_lists = len(l_values)
                catchment = gpd.read_file(shp_str)
                hru_id_field = self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')
                
                for i in range(num_nested_lists):
                    hru_id = catchment.iloc[i][hru_id_field]
                    gauge_id = gauge_mapping.get(hru_id, None)
                    
                    if gauge_id and gauge_id in stats_df['Basin_ID'].values:
                        gauge_stats = stats_df[stats_df['Basin_ID'] == gauge_id].iloc[0]
                        
                        for col in stats_df.columns:
                            if col != 'Basin_ID':
                                l_values[i].append(gauge_stats[col])
                    else:
                        # No associated gauge for this HRU
                        for _ in range(len(stats_df.columns) - 1):
                            l_values[i].append(np.nan)
                
                # Add column names to index
                for col in stats_df.columns:
                    if col != 'Basin_ID':
                        attr_parts = col.split('_')
                        if len(attr_parts) >= 2:
                            metric_name = '_'.join(attr_parts[:-1])
                            unit = self._get_flow_metric_unit(metric_name)
                            l_index.append(('Hydrology', col, unit, 'Streamflow Analysis'))
        
        return l_values, l_index

    def _get_flow_metric_unit(self, metric_name):
        """Get the appropriate unit for a flow metric."""
        # Define units for different types of metrics
        units = {
            'q_mean': 'm^3/s',
            'q_min': 'm^3/s',
            'q_max': 'm^3/s',
            'q_std': 'm^3/s',
            'yield': 'mm/yr',
            'runoff_ratio': '-',
            'baseflow_index': '-',
            'flow_duration': 'm^3/s',
            'q_': 'm^3/s',  # Default for flow quantiles
            'high_flow_freq': 'events/yr',
            'high_flow_duration': 'days',
            'low_flow_freq': 'events/yr',
            'low_flow_duration': 'days',
            'zero_flow_freq': 'events/yr',
            'slope_fdc': '-',
            'seasonality': '-',
            'day_of_max': 'day of year',
            'day_of_min': 'day of year'
        }
        
        # Find the matching unit
        for key, unit in units.items():
            if metric_name.startswith(key):
                return unit
        
        return '-'  # Default unit if not found

    def calculate_flow_metrics(self, streamflow_file):
        """Calculate comprehensive flow metrics from a streamflow time series."""
        # Load streamflow data
        flow_df = pd.read_csv(streamflow_file, parse_dates=True)
        
        if 'Date' in flow_df.columns:
            flow_df = flow_df.set_index('Date')
        elif 'date' in flow_df.columns:
            flow_df = flow_df.set_index('date')
        
        # Check for necessary columns
        if 'flow' not in flow_df.columns and 'Discharge' not in flow_df.columns:
            raise ValueError("Streamflow data must contain 'flow' or 'Discharge' column")
        
        flow_col = 'flow' if 'flow' in flow_df.columns else 'Discharge'
        
        # Get drainage area
        if self.is_lumped:
            basin = gpd.read_file(self.catchment_path)
            basin_area_km2 = basin.to_crs(self.equal_area_crs).area.sum() / 10**6  # km²
        else:
            basin_area_km2 = self.config.get('BASIN_AREA_KM2', None)
            if basin_area_km2 is None:
                self.logger.warning("Basin area not found, some metrics may be unavailable")
                basin_area_km2 = np.nan
        
        # Calculate basic statistics
        metrics = {}
        metrics['Basin_ID'] = self.domain_name
        
        # Flow statistics
        flow_data = flow_df[flow_col].values
        metrics['q_mean_annual'] = np.mean(flow_data)
        metrics['q_min_annual'] = np.min(flow_data)
        metrics['q_max_annual'] = np.max(flow_data)
        metrics['q_std_annual'] = np.std(flow_data)
        metrics['q_cv_annual'] = metrics['q_std_annual'] / metrics['q_mean_annual'] if metrics['q_mean_annual'] > 0 else np.nan
        
        # Flow quantiles
        for q in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
            metrics[f'q_{int(q*100)}_percent'] = np.percentile(flow_data, q*100)
        
        # Calculate mean daily flow for each day of the year
        if hasattr(flow_df.index, 'dayofyear'):
            daily_means = flow_df.groupby(flow_df.index.dayofyear)[flow_col].mean()
            
            # Seasonality metrics
            max_day = daily_means.idxmax()
            min_day = daily_means.idxmin()
            metrics['day_of_max_annual'] = max_day
            metrics['day_of_min_annual'] = min_day
            
            # Seasonality index - half the difference between max and min flows
            # divided by the mean annual flow (Poff & Ward, 1989)
            max_mean = daily_means.max()
            min_mean = daily_means.min()
            seasonality = 0.5 * (max_mean - min_mean) / metrics['q_mean_annual'] if metrics['q_mean_annual'] > 0 else np.nan
            metrics['seasonality_index'] = seasonality
        
        # Baseflow separation using Lyne-Hollick digital filter
        if len(flow_data) > 30:
            try:
                # Simple recursive filter method
                alpha = 0.925  # Filter parameter
                baseflow = np.zeros_like(flow_data)
                baseflow[0] = flow_data[0] / 2  # Initialize
                
                for i in range(1, len(flow_data)):
                    baseflow[i] = alpha * baseflow[i-1] + (1-alpha) * flow_data[i]
                    baseflow[i] = min(baseflow[i], flow_data[i])  # Ensure baseflow <= total flow
                
                metrics['baseflow_index'] = np.sum(baseflow) / np.sum(flow_data) if np.sum(flow_data) > 0 else np.nan
            except Exception as e:
                self.logger.warning(f"Baseflow separation failed: {str(e)}")
                metrics['baseflow_index'] = np.nan
        
        # Convert to dataframe
        metrics_df = pd.DataFrame([metrics])
        
        return metrics_df

    def attributes_from_gauge_data(self, gauge_file, l_values, l_index, case='lumped'):
        """Extract attributes from gauge station metadata."""
        try:
            # Check if the source gauge data contains metadata
            provider = self.config.get('STREAMFLOW_DATA_PROVIDER', 'WSC')
            
            # Only process for lumped case - gauge metadata doesn't apply to individual HRUs
            if case == 'lumped':
                if provider == 'WSC':
                    # Water Survey of Canada format
                    with open(gauge_file, 'r') as f:
                        # Look for metadata in first few lines
                        metadata = {}
                        for i, line in enumerate(f):
                            if i > 20:  # Only check first 20 lines
                                break
                            if ':' in line:
                                key, value = line.split(':', 1)
                                metadata[key.strip()] = value.strip()
                    
                    # Extract relevant metadata if available
                    if 'Station Name' in metadata:
                        l_values.append(metadata['Station Name'])
                        l_index.append(('Hydrology', 'gauge_name', '-', 'WSC'))
                    
                    if 'Drainage Area' in metadata:
                        area_str = metadata['Drainage Area']
                        area_value = float(''.join(c for c in area_str if c.isdigit() or c == '.'))
                        l_values.append(area_value)
                        l_index.append(('Hydrology', 'gauge_drainage_area', 'km^2', 'WSC'))
                    
                    if 'Regulation' in metadata:
                        regulated = 'regulated' in metadata['Regulation'].lower()
                        l_values.append(regulated)
                        l_index.append(('Hydrology', 'is_regulated', '-', 'WSC'))
                
                elif provider == 'USGS':
                    # USGS format typically has different metadata
                    try:
                        with open(gauge_file, 'r') as f:
                            header = [next(f) for _ in range(20)]
                        
                        # Try to extract USGS metadata
                        metadata = {}
                        for line in header:
                            if '#' in line and ':' in line:
                                key, value = line.split(':', 1)
                                metadata[key.strip('#').strip()] = value.strip()
                        
                        if 'STATION_NAME' in metadata:
                            l_values.append(metadata['STATION_NAME'])
                            l_index.append(('Hydrology', 'gauge_name', '-', 'USGS'))
                        
                        if 'DRAINAGE_AREA' in metadata:
                            area_str = metadata['DRAINAGE_AREA']
                            area_value = float(''.join(c for c in area_str if c.isdigit() or c == '.'))
                            l_values.append(area_value)
                            l_index.append(('Hydrology', 'gauge_drainage_area', 'km^2', 'USGS'))
                    except:
                        self.logger.warning("Failed to extract USGS gauge metadata")
        
        except Exception as e:
            self.logger.warning(f"Failed to extract gauge metadata: {str(e)}")
        
        return l_values, l_index

    # Open water/lakes attribute processing methods
    def attributes_from_glwd(self, geo_folder, dataset, shp_str, l_values, l_index, case='lumped'):
        """Calculate open water attributes from Global Lakes and Wetlands Database."""
        self.logger.info(f"Extracting GLWD attributes using case: {case}")
        
        # GLWD classification file
        glwd_tif = geo_folder / dataset / 'raw' / 'glwd_class.tif'
        
        if not glwd_tif.exists():
            self.logger.warning(f"GLWD raster not found at {glwd_tif}")
            return l_values, l_index
        
        # GLWD classes
        water_classes = {
            1: 'lake_permanent', 
            2: 'lake_seasonal',
            3: 'river_permanent',
            4: 'floodplain_permanent',
            5: 'floodplain_seasonal',
            6: 'swamp_forest_permanent',
            7: 'swamp_forest_seasonal',
            8: 'wetland_permanent',
            9: 'wetland_seasonal',
            10: 'coastal_wetland',
            11: 'no_water',
            12: 'wetland_mixed',
            255: 'no_data'
        }
        
        # Calculate zonal statistics for categorical data
        zonal_out = zonal_stats(shp_str, str(glwd_tif), categorical=True, all_touched=True)
        self.check_scale_and_offset(glwd_tif)
        
        # Different processing for lumped vs distributed cases
        if case == 'lumped':
            # Find the total number of classified pixels
            total_pixels = 0
            for water_id, count in zonal_out[0].items():
                if water_id in water_classes and water_id != 255:  # Skip no_data class
                    total_pixels += count
            
            # Calculate water body percentages
            for water_id, water_name in water_classes.items():
                if water_id != 255:  # Skip no_data class
                    water_fraction = 0
                    if water_id in zonal_out[0].keys():
                        water_fraction = zonal_out[0][water_id] / total_pixels if total_pixels > 0 else 0
                    l_values.append(water_fraction)
                    l_index.append(('Open Water', f'{water_name}_fraction', '-', 'GLWD'))
            
            # Calculate total water and wetland fractions
            lake_ids = [1, 2]
            river_ids = [3]
            wetland_ids = [4, 5, 6, 7, 8, 9, 10, 12]
            
            lake_fraction = sum(zonal_out[0].get(i, 0) for i in lake_ids) / total_pixels if total_pixels > 0 else 0
            river_fraction = sum(zonal_out[0].get(i, 0) for i in river_ids) / total_pixels if total_pixels > 0 else 0
            wetland_fraction = sum(zonal_out[0].get(i, 0) for i in wetland_ids) / total_pixels if total_pixels > 0 else 0
            water_fraction = lake_fraction + river_fraction
            
            l_values.append(lake_fraction)
            l_index.append(('Open Water', 'lake_fraction', '-', 'GLWD'))
            l_values.append(river_fraction)
            l_index.append(('Open Water', 'river_fraction', '-', 'GLWD'))
            l_values.append(wetland_fraction)
            l_index.append(('Open Water', 'wetland_fraction', '-', 'GLWD'))
            l_values.append(water_fraction)
            l_index.append(('Open Water', 'open_water_fraction', '-', 'GLWD'))
        
        else:
            # Distributed case
            num_nested_lists = sum(1 for item in l_values if isinstance(item, list))
            assert num_nested_lists == len(zonal_out), f"zonal_out length does not match expected list length"
            
            # Process each polygon separately
            for i in range(num_nested_lists):
                # Calculate total pixels for this HRU
                total_pixels = 0
                for water_id, count in zonal_out[i].items():
                    if water_id in water_classes and water_id != 255:
                        total_pixels += count
                
                # Water type fractions for this HRU
                hru_water_fractions = []
                for water_id, water_name in water_classes.items():
                    if water_id != 255:  # Skip no_data class
                        water_fraction = 0
                        if water_id in zonal_out[i].keys():
                            water_fraction = zonal_out[i][water_id] / total_pixels if total_pixels > 0 else 0
                        hru_water_fractions.append(water_fraction)
                
                # Calculate total water and wetland fractions for this HRU
                lake_ids = [1, 2]
                river_ids = [3]
                wetland_ids = [4, 5, 6, 7, 8, 9, 10, 12]
                
                lake_fraction = sum(zonal_out[i].get(j, 0) for j in lake_ids) / total_pixels if total_pixels > 0 else 0
                river_fraction = sum(zonal_out[i].get(j, 0) for j in river_ids) / total_pixels if total_pixels > 0 else 0
                wetland_fraction = sum(zonal_out[i].get(j, 0) for j in wetland_ids) / total_pixels if total_pixels > 0 else 0
                water_fraction = lake_fraction + river_fraction
                
                # Add all fractions to this HRU's values
                l_values[i].extend(hru_water_fractions)
                l_values[i].append(lake_fraction)
                l_values[i].append(river_fraction)
                l_values[i].append(wetland_fraction)
                l_values[i].append(water_fraction)
            
            # Add all water class names to the index (only once)
            for water_id, water_name in water_classes.items():
                if water_id != 255:  # Skip no_data class
                    l_index.append(('Open Water', f'{water_name}_fraction', '-', 'GLWD'))
            
            # Add summary water fractions to the index
            l_index.append(('Open Water', 'lake_fraction', '-', 'GLWD'))
            l_index.append(('Open Water', 'river_fraction', '-', 'GLWD'))
            l_index.append(('Open Water', 'wetland_fraction', '-', 'GLWD'))
            l_index.append(('Open Water', 'open_water_fraction', '-', 'GLWD'))
        
        return l_values, l_index

    def attributes_from_hydrolakes(self, geo_folder, dataset, shp_str, l_values, l_index, case='lumped'):
        """Calculate lake attributes from HydroLAKES database."""
        self.logger.info(f"Extracting HydroLAKES attributes using case: {case}")
        
        # Check if catchment shapefile exists
        basin = gpd.read_file(shp_str)
        
        # Path to HydroLAKES shapefile
        hydrolakes_shp = geo_folder / dataset / 'raw' / 'HydroLAKES_polys.shp'
        
        if not hydrolakes_shp.exists():
            self.logger.warning(f"HydroLAKES shapefile not found at {hydrolakes_shp}")
            return l_values, l_index
        
        try:
            # Read HydroLAKES data
            lakes = gpd.read_file(str(hydrolakes_shp))
            
            # Ensure same CRS
            if basin.crs != lakes.crs:
                lakes = lakes.to_crs(basin.crs)
            
            # Get lakes that intersect with the catchment
            intersect_lakes = gpd.overlay(basin, lakes, how='intersection')
            
            if len(intersect_lakes) == 0:
                self.logger.info("No lakes found within the catchment")
                # Add zero values for all lake metrics
                if case == 'lumped':
                    l_values.extend([0, 0, 0, 0, 0, 0])
                    l_index.extend([
                        ('Open Water', 'lake_count', 'count', 'HydroLAKES'),
                        ('Open Water', 'lake_area_total', 'km^2', 'HydroLAKES'),
                        ('Open Water', 'lake_area_max', 'km^2', 'HydroLAKES'),
                        ('Open Water', 'lake_area_mean', 'km^2', 'HydroLAKES'),
                        ('Open Water', 'lake_volume_total', 'km^3', 'HydroLAKES'),
                        ('Open Water', 'lake_residence_time_mean', 'days', 'HydroLAKES')
                    ])
                else:
                    for i in range(len(l_values)):
                        l_values[i].extend([0, 0, 0, 0, 0, 0])
                    
                    # Add index entries only once
                    l_index.extend([
                        ('Open Water', 'lake_count', 'count', 'HydroLAKES'),
                        ('Open Water', 'lake_area_total', 'km^2', 'HydroLAKES'),
                        ('Open Water', 'lake_area_max', 'km^2', 'HydroLAKES'),
                        ('Open Water', 'lake_area_mean', 'km^2', 'HydroLAKES'),
                        ('Open Water', 'lake_volume_total', 'km^3', 'HydroLAKES'),
                        ('Open Water', 'lake_residence_time_mean', 'days', 'HydroLAKES')
                    ])
                
                return l_values, l_index
            
            # Calculate area in km² using equal area projection
            intersect_lakes['area_km2'] = intersect_lakes.to_crs(self.equal_area_crs).area / 10**6
            
            if case == 'lumped':
                # Count number of lakes
                lake_count = len(intersect_lakes)
                
                # Calculate lake area statistics
                lake_area_total = intersect_lakes['area_km2'].sum()
                lake_area_max = intersect_lakes['area_km2'].max()
                lake_area_mean = intersect_lakes['area_km2'].mean()
                
                # Calculate lake volume if available
                if 'Vol_total' in intersect_lakes.columns:
                    # Convert to km³
                    lake_volume_total = intersect_lakes['Vol_total'].sum() / 10**9
                else:
                    lake_volume_total = np.nan
                
                # Calculate residence time if available
                if 'Res_time' in intersect_lakes.columns:
                    lake_residence_time_mean = intersect_lakes['Res_time'].mean()
                else:
                    lake_residence_time_mean = np.nan
                
                # Add to values list
                l_values.extend([
                    lake_count,
                    lake_area_total,
                    lake_area_max,
                    lake_area_mean,
                    lake_volume_total,
                    lake_residence_time_mean
                ])
                
                # Add to index
                l_index.extend([
                    ('Open Water', 'lake_count', 'count', 'HydroLAKES'),
                    ('Open Water', 'lake_area_total', 'km^2', 'HydroLAKES'),
                    ('Open Water', 'lake_area_max', 'km^2', 'HydroLAKES'),
                    ('Open Water', 'lake_area_mean', 'km^2', 'HydroLAKES'),
                    ('Open Water', 'lake_volume_total', 'km^3', 'HydroLAKES'),
                    ('Open Water', 'lake_residence_time_mean', 'days', 'HydroLAKES')
                ])
                
            else:
                # For distributed case, calculate lake attributes for each HRU
                for i in range(len(basin)):
                    hru = basin.iloc[[i]]
                    hru_lakes = gpd.overlay(hru, lakes, how='intersection')
                    
                    if len(hru_lakes) == 0:
                        # No lakes in this HRU
                        l_values[i].extend([0, 0, 0, 0, 0, 0])
                    else:
                        # Calculate area in km²
                        hru_lakes['area_km2'] = hru_lakes.to_crs(self.equal_area_crs).area / 10**6
                        
                        # Lake statistics for this HRU
                        lake_count = len(hru_lakes)
                        lake_area_total = hru_lakes['area_km2'].sum()
                        lake_area_max = hru_lakes['area_km2'].max()
                        lake_area_mean = hru_lakes['area_km2'].mean()
                        
                        # Volume and residence time if available
                        if 'Vol_total' in hru_lakes.columns:
                            lake_volume_total = hru_lakes['Vol_total'].sum() / 10**9
                        else:
                            lake_volume_total = np.nan
                        
                        if 'Res_time' in hru_lakes.columns:
                            lake_residence_time_mean = hru_lakes['Res_time'].mean()
                        else:
                            lake_residence_time_mean = np.nan
                        
                        # Add to this HRU's values
                        l_values[i].extend([
                            lake_count,
                            lake_area_total,
                            lake_area_max,
                            lake_area_mean,
                            lake_volume_total,
                            lake_residence_time_mean
                        ])
                
                # Add index entries only once
                l_index.extend([
                    ('Open Water', 'lake_count', 'count', 'HydroLAKES'),
                    ('Open Water', 'lake_area_total', 'km^2', 'HydroLAKES'),
                    ('Open Water', 'lake_area_max', 'km^2', 'HydroLAKES'),
                    ('Open Water', 'lake_area_mean', 'km^2', 'HydroLAKES'),
                    ('Open Water', 'lake_volume_total', 'km^3', 'HydroLAKES'),
                    ('Open Water', 'lake_residence_time_mean', 'days', 'HydroLAKES')
                ])
        
        except Exception as e:
            self.logger.error(f"Error processing HydroLAKES: {str(e)}")
            # Add placeholder values
            if case == 'lumped':
                l_values.extend([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            else:
                for i in range(len(l_values)):
                    l_values[i].extend([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            
            # Add index entries
            l_index.extend([
                ('Open Water', 'lake_count', 'count', 'HydroLAKES'),
                ('Open Water', 'lake_area_total', 'km^2', 'HydroLAKES'),
                ('Open Water', 'lake_area_max', 'km^2', 'HydroLAKES'),
                ('Open Water', 'lake_area_mean', 'km^2', 'HydroLAKES'),
                ('Open Water', 'lake_volume_total', 'km^3', 'HydroLAKES'),
                ('Open Water', 'lake_residence_time_mean', 'days', 'HydroLAKES')
            ])
        
        return l_values, l_index