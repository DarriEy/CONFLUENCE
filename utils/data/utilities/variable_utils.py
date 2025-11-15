import pint_xarray # type: ignore
import pint # type: ignore
import xarray as xr # type: ignore
import yaml # type: ignore
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class VariableHandler:
    """
    Handles variable name mapping and unit conversion between different datasets and models.
    
    Attributes:
        variable_mappings (Dict): Dataset to model variable name mappings
        unit_registry (pint_xarray.UnitRegistry): Unit conversion registry
        logger (logging.Logger): SYMFLUENCE logger instance
    """
    
    # Dataset variable name mappings
    DATASET_MAPPINGS = {
        'ERA5': {
            'airtemp': {'standard_name': 'air_temperature', 'units': 'K'},
            'airpres': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'spechum': {'standard_name': 'specific_humidity', 'units': '1'},
            'windspd': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'LWRadAtm': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'SWRadAtm': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'pptrate': {'standard_name': 'precipitation_flux', 'units': 'mm/s'}
        },
        'CARRA': {
            '2m_temperature': {'standard_name': 'air_temperature', 'units': 'K'},
            'surface_pressure': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            '2m_specific_humidity': {'standard_name': 'specific_humidity', 'units': '1'},
            '10m_u_component_of_wind': {'standard_name': 'eastward_wind', 'units': 'm/s'},
            '10m_v_component_of_wind': {'standard_name': 'northward_wind', 'units': 'm/s'},
            'thermal_surface_radiation_downwards': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'surface_net_solar_radiation': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'total_precipitation': {'standard_name': 'precipitation_flux', 'units': 'mm/s'}
        },
        'RDRS': {
            'RDRS_v2.1_P_TT_1.5m': {'standard_name': 'air_temperature', 'units': 'K'},
            'RDRS_v2.1_P_P0_SFC': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'RDRS_v2.1_P_HU_1.5m': {'standard_name': 'specific_humidity', 'units': '1'},
            'RDRS_v2.1_P_UVC_10m': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'RDRS_v2.1_P_FI_SFC': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'RDRS_v2.1_P_FB_SFC': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'RDRS_v2.1_A_PR0_SFC': {'standard_name': 'precipitation_flux', 'units': 'mm/s'}
        },
        'CASR': {
            'CaSR_v3.1_A_TT_1.5m': {'standard_name': 'air_temperature', 'units': 'K'},
            'CaSR_v3.1_P_P0_SFC': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'CaSR_v3.1_P_HU_1.5m': {'standard_name': 'specific_humidity', 'units': '1'},
            'CaSR_v3.1_P_UVC_10m': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'CaSR_v3.1_P_FI_SFC': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'CaSR_v3.1_P_FB_SFC': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'CaSR_v3.1_P_PR0_SFC': {'standard_name': 'precipitation_flux', 'units': 'm'}
        },
        'DayMet': {
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/s'},
            'dayl': {'standard_name': 'day_length', 'units': 's/day'},
            'prcp': {'standard_name': 'precipitation_flux', 'units': 'mm/day'},
            'srad': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'swe': {'standard_name': 'snow_water_equivalent', 'units': 'kg/m^2'},
            'tmax': {'standard_name': 'air_temperature_max', 'units': 'degC'},
            'tmin': {'standard_name': 'air_temperature_min', 'units': 'degC'},
            'vp': {'standard_name': 'water_vapor_pressure', 'units': 'Pa'}
        },
        'NEX-GDDP': {
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/day'},
            'tas': {'standard_name': 'air_temperature', 'units': 'K'},
            'tasmax': {'standard_name': 'air_temperature_max', 'units': 'K'},
            'tasmin': {'standard_name': 'air_temperature_min', 'units': 'K'},
            'hurs': {'standard_name': 'relative_humidity', 'units': '%'},
            'huss': {'standard_name': 'specific_humidity', 'units': '1'},
            'rlds': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'rsds': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'sfcWind': {'standard_name': 'wind_speed', 'units': 'm/s'}
        },
        'GWF-I': {
            'PSFC': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'Q2': {'standard_name': 'specific_humidity', 'units': '1'},
            'T2': {'standard_name': 'air_temperature', 'units': 'K'},
            'U10': {'standard_name': 'eastward_wind', 'units': 'm/s'},
            'V10': {'standard_name': 'northward_wind', 'units': 'm/s'},
            'PREC_ACC_NC': {'standard_name': 'precipitation_flux', 'units': 'mm/hr'},  
            'SWDOWN': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'GLW': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'}
        },
        'GWF-II': {
            'PSFC': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'Q2': {'standard_name': 'specific_humidity', 'units': '1'},
            'T2': {'standard_name': 'air_temperature', 'units': 'K'},
            'U10': {'standard_name': 'eastward_wind', 'units': 'm/s'},
            'V10': {'standard_name': 'northward_wind', 'units': 'm/s'},
            'PREC_ACC_NC': {'standard_name': 'precipitation_flux', 'units': 'mm/hr'},  
            'SWDOWN': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'GLW': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'}
        },
        'CCRN-CanRCM4': {
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/s'},
            'ta': {'standard_name': 'air_temperature', 'units': 'K'},
            'ps': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'hus': {'standard_name': 'specific_humidity', 'units': '1'},
            'wind': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'rsds': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'lsds': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'}
        },
        'CCRN-WFDEI': {
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/s'},
            'ta': {'standard_name': 'air_temperature', 'units': 'K'},
            'ps': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'hus': {'standard_name': 'specific_humidity', 'units': '1'},
            'wind': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'rsds': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'lsds': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'}
        },
        'Ouranos-ESPO': {
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/s'},
            'tasmax': {'standard_name': 'air_temperature_max', 'units': 'K'},
            'tasmin': {'standard_name': 'air_temperature_min', 'units': 'K'}
        },
        'Ouranos-MRCC5': {
            'tas': {'standard_name': 'air_temperature', 'units': 'K'},
            'ps': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'huss': {'standard_name': 'specific_humidity', 'units': '1'},
            'uas': {'standard_name': 'eastward_wind', 'units': 'm/s'},
            'vas': {'standard_name': 'northward_wind', 'units': 'm/s'},
            'rlds': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'rsds': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/s'}
        },
        'AGCD': {
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/day'},
            'tmax': {'standard_name': 'air_temperature_max', 'units': 'degC'},
            'tmin': {'standard_name': 'air_temperature_min', 'units': 'degC'}
        }

    }

    # Model variable requirements
    MODEL_REQUIREMENTS = {
        'SUMMA': {
            'airtemp': {'standard_name': 'air_temperature', 'units': 'K'},
            'airpres': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'spechum': {'standard_name': 'specific_humidity', 'units': '1'},
            'windspd': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'LWRadAtm': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'SWRadAtm': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'pptrate': {'standard_name': 'precipitation_flux', 'units': 'mm/s'}
        },
        'FUSE': {
            'temp': {'standard_name': 'air_temperature', 'units': 'degC'},
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/day'}
        },
        'GR': {
            'temp': {'standard_name': 'air_temperature', 'units': 'degC'},
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/day'}
        }, 
        'HYPE': {
            'temp': {'standard_name': 'air_temperature', 'units': 'degC'},
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/day'}
        },
        'MESH': {
            'airtemp': {'standard_name': 'air_temperature', 'units': 'K'},
            'airpres': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'spechum': {'standard_name': 'specific_humidity', 'units': '1'},
            'windspd': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'LWRadAtm': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'SWRadAtm': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'pptrate': {'standard_name': 'precipitation_flux', 'units': 'mm/s'}
        }
    }

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, dataset: str, model: str):
        """
        Initialize VariableHandler with configuration settings.
        
        Args:
            config: SYMFLUENCE configuration dictionary
            logger: SYMFLUENCE logger instance
        """
        self.config = config
        self.logger = logger
        self.dataset = dataset if dataset is not None else config.get('FORCING_DATASET')
        self.model = model if model is not None else config.get('HYDROLOGICAL_MODEL')
        
        # Initialize pint for unit handling
        self.ureg = pint.UnitRegistry()
        pint_xarray.setup_registry(self.ureg)
        
        # Validate dataset and model are supported
        if self.dataset not in self.DATASET_MAPPINGS:
            self.logger.error(f"Unsupported dataset: {self.dataset}")
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        if self.model not in self.MODEL_REQUIREMENTS:
            self.logger.error(f"Unsupported model: {self.model}")
            raise ValueError(f"Unsupported model: {self.model}")

    def get_dataset_variables(self, dataset: Optional[str] = None) -> str:
        """
        Get the forcing variable keys for a specified dataset as a comma-separated string.
        
        Args:
            dataset (Optional[str]): Name of the dataset. If None, uses the instance's dataset.
            
        Returns:
            str: Comma-separated string of variable keys for the specified dataset
            
        Raises:
            ValueError: If the specified dataset is not supported
        """
        # Use instance dataset if none provided
        dataset_name = dataset if dataset is not None else self.dataset
        
        # Check if dataset exists in mappings
        if dataset_name not in self.DATASET_MAPPINGS:
            self.logger.error(f"Unsupported dataset: {dataset_name}")
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        return ','.join(self.DATASET_MAPPINGS[dataset_name].keys())
    
    def process_forcing_data(self, data: xr.Dataset) -> xr.Dataset:
        """Process forcing data by mapping variable names and converting units."""
        self.logger.info("Starting forcing data unit processing")
        
        processed_data = data.copy()
        
        # Get dataset and model mappings
        dataset_map = self.DATASET_MAPPINGS[self.dataset]
        model_map = self.MODEL_REQUIREMENTS[self.model]
        
        # Process each required model variable
        for model_var, model_req in model_map.items():
            # Find corresponding dataset variable
            dataset_var = self._find_matching_variable(model_req['standard_name'], dataset_map)
            
            if dataset_var is None:
                self.logger.error(f"Required variable {model_var} not found in dataset {self.dataset}")
                raise ValueError(f"Required variable {model_var} not found in dataset {self.dataset}")
            
            # Rename variable
            if dataset_var in processed_data:
                self.logger.debug(f"Processing {dataset_var} -> {model_var}")
                
                # Get units from our mappings
                source_units = dataset_map[dataset_var]['units']
                target_units = model_req['units']
                
                # Convert units if needed
                if source_units != target_units:
                    self.logger.debug(f"Converting units for {dataset_var}: {source_units} -> {target_units}")
                    try:
                        processed_data[dataset_var] = self._convert_units(
                            processed_data[dataset_var], 
                            source_units, 
                            target_units
                        )
                    except Exception as e:
                        self.logger.error(f"Unit conversion failed for {dataset_var}: {str(e)}")
                        raise
                
                # Rename after conversion
                processed_data = processed_data.rename({dataset_var: model_var})
        
        self.logger.info("Forcing data unit processing completed")
        return processed_data

    def _find_matching_variable(self, standard_name: str, dataset_map: Dict) -> Optional[str]:
        """Find dataset variable matching the required standard_name."""
        for var, attrs in dataset_map.items():
            if attrs['standard_name'] == standard_name:
                return var
        self.logger.warning(f"No matching variable found for standard_name: {standard_name}")
        return None

    def _convert_units(self, data: xr.DataArray, from_units: str, to_units: str) -> xr.DataArray:
        """
        Convert variable units using pint-xarray.
        
        Args:
            data: DataArray to convert
            from_units: Source units
            to_units: Target units
            
        Returns:
            DataArray with converted units
        """
        try:
            # Special case for precipitation flux conversions
            if ('kg/m2/s' in from_units or 'kilogram / meter ** 2 / second' in from_units) and 'mm/day' in to_units:
                # 1 kg/m² = 1 mm of water
                # Convert kg/m²/s to mm/s, then to mm/day
                converted = data * 86400  # multiply by seconds per day
                return converted
            
            # Regular unit conversion
            data = data.pint.quantify(from_units)
            converted = data.pint.to(to_units)
            return converted.pint.dequantify()
        except Exception as e:
            self.logger.error(f"Unit conversion failed: {from_units} -> {to_units}: {str(e)}")
            raise

    def save_mappings(self, filepath: Path):
        """Save current mappings to YAML file."""
        self.logger.info(f"Saving variable mappings to: {filepath}")
        mappings = {
            'dataset_mappings': self.DATASET_MAPPINGS,
            'model_requirements': self.MODEL_REQUIREMENTS
        }
        
        try:
            with open(filepath, 'w') as f:
                yaml.dump(mappings, f)
            self.logger.info("Variable mappings saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save mappings: {str(e)}")
            raise

    @classmethod
    def load_mappings(cls, filepath: Path, logger: logging.Logger):
        """Load mappings from YAML file."""
        logger.info(f"Loading variable mappings from: {filepath}")
        try:
            with open(filepath, 'r') as f:
                mappings = yaml.safe_load(f)
                
            cls.DATASET_MAPPINGS = mappings['dataset_mappings']
            cls.MODEL_REQUIREMENTS = mappings['model_requirements']
            logger.info("Variable mappings loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load mappings: {str(e)}")
            raise