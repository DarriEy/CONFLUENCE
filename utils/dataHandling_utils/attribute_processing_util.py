from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import os
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from rasterstats import zonal_stats
from osgeo import gdal

class attributeProcessor:
    """
    Processes catchment characteristic attributes for hydrological modeling.
    
    This implementation focuses on processing soil attributes from SOILGRIDS data.
    Additional data sources can be added incrementally.
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the attribute processor.
        
        Args:
            config: Configuration dictionary containing CONFLUENCE settings
            logger: Logger instance for recording processing information
        """
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
        
        # Initialize attribute processing
        self.process_attributes()
    
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
        """Process all catchment attributes."""
        self.logger.info("Starting attribute processing")
        
        # Determine if we are processing lumped or distributed attributes
        is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
        attribute_type = 'lumped' if is_lumped else 'distributed'
        self.logger.info(f"Processing {attribute_type} attributes")
        
        try:
            # Process attributes from different sources
            attributes_df = self._process_soil_attributes()
            
            # Save attributes
            output_file = self.attributes_dir / f"{self.domain_name}_attributes.csv"
            attributes_df.to_csv(output_file)
            self.logger.info(f"Attributes saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error processing attributes: {str(e)}")
            raise
    
    def _process_soil_attributes(self) -> pd.DataFrame:
        """
        Process soil attributes from SOILGRIDS.
        
        Returns:
            DataFrame containing processed soil attributes
        """
        self.logger.info("Processing soil attributes from SOILGRIDS")
        
        # Determine if we are processing lumped or distributed attributes
        is_lumped = self.config.get('DOMAIN_DEFINITION_METHOD') == 'lumped'
        
        # Load catchment shapefile
        catchment = gpd.read_file(self.catchment_path)
        
        # Initialize attribute collection lists
        l_values = [] if is_lumped else [[] for _ in range(len(catchment))]
        l_index = []
        
        # Get SOILGRIDS data path
        soilgrids_path = self._get_soilgrids_path()
        
        # Process SOILGRIDS data if available
        if soilgrids_path.exists():
            self.logger.info(f"Processing SOILGRIDS data from {soilgrids_path}")
            case = 'lumped' if is_lumped else 'distributed'
            l_values, l_index = self.attributes_from_soilgrids(
                soilgrids_path,
                'soilgrids',
                str(self.catchment_path),
                l_values,
                l_index,
                case=case
            )
        else:
            self.logger.warning(f"SOILGRIDS data not found at {soilgrids_path}")
        
        # Convert to DataFrame
        if is_lumped:
            # For lumped domain
            attributes_df = pd.DataFrame([l_values], columns=[f"{category}.{name}" for category, name, _, _ in l_index])
            attributes_df.insert(0, 'Basin_ID', self.domain_name)
        else:
            # For distributed domain
            # Create a list of attribute dictionaries for each HRU
            attribute_dicts = []
            for i, hru_values in enumerate(l_values):
                attr_dict = {f"{category}.{name}": value for (category, name, _, _), value in zip(l_index, hru_values)}
                attr_dict['HRU_ID'] = catchment.iloc[i][self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')]
                attr_dict['Basin_ID'] = self.domain_name
                attribute_dicts.append(attr_dict)
            attributes_df = pd.DataFrame(attribute_dicts)
        
        return attributes_df
    
    def _get_soilgrids_path(self) -> Path:
        """Get the path to the SOILGRIDS data."""
        soilgrids_path = self.config.get('ATTRIBUTES_SOILGRIDS_PATH')
        
        if soilgrids_path == 'default':
            soilgrids_path = self.data_dir / 'geospatial-data' / 'soilgrids'
        else:
            soilgrids_path = Path(soilgrids_path)
        
        return soilgrids_path
    
    # Utility functions
    def harmonic_mean(self, x):
        """Calculate harmonic mean, handling masked arrays."""
        return np.ma.count(x) / (1/x).sum()
    
    def read_scale_and_offset(self, geotiff_path):
        """Read scale and offset values from a GeoTIFF file."""
        # Enforce data type
        geotiff_path = str(geotiff_path)
        
        # Open the GeoTIFF file
        dataset = gdal.Open(geotiff_path)

        if dataset is None:
            raise FileNotFoundError(f"File not found: {geotiff_path}")

        # Get the scale and offset values
        scale = dataset.GetRasterBand(1).GetScale()
        offset = dataset.GetRasterBand(1).GetOffset()

        # Close the dataset
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

        # Deal with the occassional None that we get when raster data input to zonal-stats is missing
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
                    self.logger.debug(f'Replaced None in {key} with NaN')
        return zonal_out
    
    # SOILGRIDS attribute processing
    def attributes_from_soilgrids(self, geo_folder, dataset, shp_str, l_values, l_index, case='lumped'):
        """
        Calculates attributes from SOILGRIDS maps.
        
        Args:
            geo_folder: Path to geospatial data folder
            dataset: Dataset name
            shp_str: Path to shapefile as string
            l_values: List to store attribute values
            l_index: List to store attribute index information
            case: Either 'lumped' or 'distributed'
            
        Returns:
            Updated l_values and l_index lists
        """
        self.logger.info(f"Extracting SOILGRIDS attributes using case: {case}")

        # File specification
        sub_folders = ['bdod',     'cfvo',       'clay',    'sand',    'silt',    'soc',      'porosity']
        units       = ['cg cm^-3', 'cm^3 dm^-3', 'g kg^-1', 'g kg^-1', 'g kg^-1', 'dg kg^-1', '-']
        depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
        fields = ['mean', 'uncertainty']
        stats = ['mean', 'min', 'max', 'std']
        
        # Loop over the files and calculate stats
        for sub_folder, unit in zip(sub_folders, units):
            for depth in depths:
                for field in fields:
                    tif = str(geo_folder / dataset / 'raw' / f'{sub_folder}' / f'{sub_folder}_{depth}_{field}.tif')
                    if not os.path.exists(tif): 
                        self.logger.debug(f"File not found: {tif}")
                        continue  # porosity has no uncertainty field
                    
                    self.logger.debug(f"Processing: {tif}")
                    zonal_out = zonal_stats(shp_str, tif, stats=stats, all_touched=True)
                    scale, offset = self.read_scale_and_offset(tif)
                    l_values = self.update_values_list(l_values, stats, zonal_out, scale, offset, case=case)
                    l_index += [('Soil', f'{sub_folder}_{depth}_{field}_min',  f'{unit}', 'SOILGRIDS'),
                                ('Soil', f'{sub_folder}_{depth}_{field}_mean', f'{unit}', 'SOILGRIDS'),
                                ('Soil', f'{sub_folder}_{depth}_{field}_max',  f'{unit}', 'SOILGRIDS'),
                                ('Soil', f'{sub_folder}_{depth}_{field}_std',  f'{unit}', 'SOILGRIDS')]
           
        # --- Specific processing for the conductivity fields
        # For conductivity we want to have a harmonic mean because that should be more
        # representative as a spatial average.
        
        # Process the data fields
        sub_folder = 'conductivity'
        unit       = 'cm hr^-1'
        depths = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
        field  = 'mean'  # no uncertainty maps for these
        stats = ['min', 'max', 'std']

        for depth in depths:
            tif = str(geo_folder / dataset / 'raw' / f'{sub_folder}' / f'{sub_folder}_{depth}_{field}.tif')
            if not os.path.exists(tif): 
                self.logger.debug(f"File not found: {tif}")
                continue
            
            self.logger.debug(f"Processing conductivity: {tif}")
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

        self.logger.info(f"Completed SOILGRIDS attribute extraction: {len(l_index)} attributes")
        return l_values, l_index


