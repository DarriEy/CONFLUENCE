import os
import sys
from pathlib import Path
from typing import Dict, Any
import xarray as xr # type: ignore

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.configHandling_utils.logging_utils import get_function_logger # type: ignore
from utils.models_utils.summaflow import ( # type: ignore
    write_summa_forcing,
    write_summa_attribute,
    write_summa_paramtrial,
    write_summa_initial_conditions,
    write_summa_filemanager,
    copy_summa_static_files
)

class SummaPreProcessor:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"
        self.summa_setup_dir = self.project_dir / "settings" / "summa_setup"
        
        # Add these new attributes
        self.geofabric_mapping = self.config.get('GEOFABRIC_MAPPING', {})
        self.landcover_mapping = self.config.get('LANDCOVER_MAPPING', {})
        self.soil_mapping = self.config.get('SOIL_MAPPING', {})
        self.write_mizuroute_domain = self.config.get('WRITE_MIZUROUTE_DOMAIN', False)

    @get_function_logger
    def run_preprocessing(self):
        self.logger.info("Starting SUMMA preprocessing")
        
        self.summa_setup_dir.mkdir(parents=True, exist_ok=True)
        
        # Write SUMMA attribute file
        attr = self.write_summa_attribute()
        
        # Write SUMMA forcing file
        forcing = self.write_summa_forcing(attr)
        
        # Write SUMMA parameter trial file
        self.write_summa_paramtrial(attr)
        
        # Write SUMMA initial conditions file
        self.write_summa_initial_conditions(attr)
        
        # Write SUMMA file manager
        self.write_summa_filemanager(forcing)
        
        # Copy SUMMA static files
        self.copy_summa_static_files()
        
        self.logger.info("SUMMA preprocessing completed")

    def write_summa_attribute(self):
        subbasins_name = self.config.get('CATCHMENT_SHP_NAME')
        if subbasins_name == 'default':
            subbasins_name = f"{self.config['DOMAIN_NAME']}_HRUs_{self.config['DOMAIN_DISCRETIZATION']}.shp"

        subbasins_shapefile = self.project_dir / "shapefiles" / "catchment" / subbasins_name

        rivers_name = self.config.get('RIVER_NETWORK_SHP_NAME')
        if rivers_name == 'default':
            rivers_name = f"{self.config['DOMAIN_NAME']}_riverNetwork_delineate.shp"

        rivers_shapefile = self.project_dir / "shapefiles" / "river_network" / rivers_name
        gistool_output = self.project_dir / "attributes"
        
        return write_summa_attribute(
            self.summa_setup_dir,
            subbasins_shapefile,
            rivers_shapefile,
            gistool_output,
            self.config.get('MINIMUM_LAND_FRACTION'),
            self.config.get('HRU_DISCRETIZATION'),
            self.geofabric_mapping,
            self.landcover_mapping,
            self.soil_mapping,
            self.write_mizuroute_domain
        )

    def write_summa_forcing(self, attr):
        easymore_output = self.project_dir / "forcing" / "basin_averaged_data"
        timeshift = self.config.get('FORCING_TIMESHIFT', 0)
        forcing_units = self.config.get('FORCING_UNITS', {})
        return write_summa_forcing(self.summa_setup_dir, timeshift, forcing_units, easymore_output, attr, self.geofabric_mapping)

    def write_summa_paramtrial(self, attr):
        write_summa_paramtrial(attr, self.summa_setup_dir)

    def write_summa_initial_conditions(self, attr):
        write_summa_initial_conditions(attr, self.config.get('SOIL_LAYER_DEPTH'), self.summa_setup_dir)

    def write_summa_filemanager(self, forcing):
        write_summa_filemanager(self.summa_setup_dir, forcing)

    def copy_summa_static_files(self):
        copy_summa_static_files(self.summa_setup_dir)