#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NGEN Parameter Manager for Calibration

Handles NGEN model parameter bounds, normalization, and configuration updates
for calibration with modules like NOAH-OWP
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging


class NgenParameterManager:
    """Handles NGEN parameter bounds, normalization, and file updates"""
    
    def __init__(self, config: Dict, logger: logging.Logger, ngen_settings_dir: Path):
        self.config = config
        self.logger = logger
        self.ngen_settings_dir = ngen_settings_dir
        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')
        
        # Parse modules to calibrate
        modules_str = config.get('NGEN_MODULES_TO_CALIBRATE', 'NOAH')
        self.modules_to_calibrate = [m.strip() for m in modules_str.split(',') if m.strip()]
        
        # Get parameters for each module
        self.module_params = {}
        self.param_bounds = {}
        
        for module in self.modules_to_calibrate:
            params = self._get_module_parameters(module)
            self.module_params[module] = params
            bounds = self._get_module_parameter_bounds(module, params)
            self.param_bounds.update(bounds)
        
        self.logger.info(f"NgenParameterManager initialized")
        self.logger.info(f"Calibrating modules: {self.modules_to_calibrate}")
        self.logger.info(f"Total parameters to calibrate: {len(self.param_bounds)}")
    
    def _get_module_parameters(self, module: str) -> List[str]:
        """Get list of parameters to calibrate for a specific module"""
        config_key = f'NGEN_{module.upper()}_PARAMS_TO_CALIBRATE'
        
        if config_key in self.config:
            params_str = self.config[config_key]
            return [p.strip() for p in params_str.split(',') if p.strip()]
        else:
            # Return default parameters for known modules
            return self._get_default_module_parameters(module)
    
    def _get_default_module_parameters(self, module: str) -> List[str]:
        """Get default calibration parameters for a module"""
        defaults = {
            'NOAH': ['rain_snow_thresh', 'ZREF', 'noah_refdk', 'noah_refkdt', 
                    'noah_czil', 'noah_z0', 'noah_frzk', 'noah_salp'],
            'CFE': ['maxsmc', 'satdk', 'slope', 'bb', 'multiplier'],
            'TOPMODEL': ['szm', 'td', 'm', 'sr0'],
        }
        return defaults.get(module.upper(), [])
    
    def _get_module_parameter_bounds(self, module: str, params: List[str]) -> Dict[str, Dict[str, float]]:
        """Get parameter bounds for a specific module"""
        if module.upper() == 'NOAH':
            return self._get_noah_parameter_bounds(params)
        elif module.upper() == 'CFE':
            return self._get_cfe_parameter_bounds(params)
        elif module.upper() == 'TOPMODEL':
            return self._get_topmodel_parameter_bounds(params)
        else:
            self.logger.warning(f"Unknown module {module}, using default bounds")
            return {param: {'min': 0.1, 'max': 10.0} for param in params}
    
    def _get_noah_parameter_bounds(self, params: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Get parameter bounds for NOAH-OWP parameters
        
        Bounds based on:
        - NOAH-MP documentation
        - WRF NOAH LSM user guide  
        - Literature calibration studies
        """
        noah_bounds = {
            # Temperature threshold for rain/snow partitioning (degrees Celsius)
            # Literature range: 0-4°C, commonly calibrated around 0-2°C
            'rain_snow_thresh': {'min': -2.0, 'max': 4.0},
            
            # Reference height for forcing data (meters)
            # Standard values: 2m (screen height) to 10m (tower height)
            'ZREF': {'min': 2.0, 'max': 10.0},
            
            # Reference saturated hydraulic conductivity (m/s)
            # Default: 2.0e-6 (silty clay loam Ksat)
            # Calibration range: 1.0e-6 to 5.0e-6 m/s
            'noah_refdk': {'min': 1.0e-6, 'max': 5.0e-6},
            
            # Infiltration/runoff partitioning parameter (dimensionless)
            # Default: 3.0, tuneable range 1.0-5.0
            # Higher values decrease surface runoff
            'noah_refkdt': {'min': 1.0, 'max': 5.0},
            
            # Zilitinkevich coefficient for roughness length ratio (dimensionless)
            # Default: 0.1, typical range 0.075-0.2
            # Controls ratio of heat to momentum roughness length
            'noah_czil': {'min': 0.075, 'max': 0.2},
            
            # Roughness length (meters)
            # Vegetation dependent, typical range 0.01-2.0 m
            # Bare soil ~0.01 m, short grass ~0.05 m, forest ~1-2 m
            'noah_z0': {'min': 0.01, 'max': 2.0},
            
            # Frozen ground parameter (dimensionless)
            # Default: 0.15 for light clay
            # Range: 0.05-0.3
            'noah_frzk': {'min': 0.05, 'max': 0.3},
            
            # Shape parameter for snow cover distribution (dimensionless)
            # Default: 2.6, typical range 1.0-4.0
            'noah_salp': {'min': 1.0, 'max': 4.0},
            
            # Additional common NOAH parameters
            'noah_sbeta': {'min': -3.0, 'max': -1.0},  # Soil heat flux parameter
            'noah_fxexp': {'min': 1.0, 'max': 3.0},     # Bare soil evap exponent
            'noah_smcref': {'min': 0.2, 'max': 0.4},    # Reference soil moisture
            'noah_smcwlt': {'min': 0.02, 'max': 0.15},  # Wilting point
            'noah_smcdry': {'min': 0.01, 'max': 0.1},   # Dry soil moisture
        }
        
        # Return only the requested parameters
        bounds = {}
        for param in params:
            if param in noah_bounds:
                bounds[param] = noah_bounds[param]
            else:
                self.logger.warning(f"No bounds defined for parameter {param}, using default [0.1, 10.0]")
                bounds[param] = {'min': 0.1, 'max': 10.0}
        
        return bounds
    
    def _get_cfe_parameter_bounds(self, params: List[str]) -> Dict[str, Dict[str, float]]:
        """Get parameter bounds for CFE (Conceptual Functional Equivalent) model"""
        cfe_bounds = {
            'maxsmc': {'min': 0.3, 'max': 0.7},      # Maximum soil moisture content
            'satdk': {'min': 1.0e-6, 'max': 1.0e-4}, # Saturated hydraulic conductivity
            'slope': {'min': 0.0, 'max': 1.0},       # Slope parameter
            'bb': {'min': 2.0, 'max': 14.0},         # Pore size distribution index
            'multiplier': {'min': 0.1, 'max': 10.0}, # General multiplier
        }
        
        bounds = {}
        for param in params:
            if param in cfe_bounds:
                bounds[param] = cfe_bounds[param]
            else:
                self.logger.warning(f"No bounds defined for CFE parameter {param}, using default")
                bounds[param] = {'min': 0.1, 'max': 10.0}
        
        return bounds
    
    def _get_topmodel_parameter_bounds(self, params: List[str]) -> Dict[str, Dict[str, float]]:
        """Get parameter bounds for TOPMODEL"""
        topmodel_bounds = {
            'szm': {'min': 0.01, 'max': 0.5},    # Maximum storage deficit
            'td': {'min': 0.1, 'max': 10.0},     # Unsaturated zone time delay
            'm': {'min': 0.001, 'max': 0.1},     # Exponential decay parameter
            'sr0': {'min': 0.001, 'max': 0.1},   # Initial subsurface flow
        }
        
        bounds = {}
        for param in params:
            if param in topmodel_bounds:
                bounds[param] = topmodel_bounds[param]
            else:
                self.logger.warning(f"No bounds defined for TOPMODEL parameter {param}, using default")
                bounds[param] = {'min': 0.1, 'max': 10.0}
        
        return bounds
    
    @property
    def all_param_names(self) -> List[str]:
        """Get list of all parameter names to calibrate"""
        return list(self.param_bounds.keys())
    
    def get_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """Get parameter bounds for calibration"""
        return self.param_bounds.copy()
    
    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from default or mid-range of bounds"""
        params = {}
        
        for param_name, bounds in self.param_bounds.items():
            # Use middle of bounds as initial value
            params[param_name] = (bounds['min'] + bounds['max']) / 2.0
        
        return params
    
    def normalize_parameters(self, params: Dict[str, float]) -> np.ndarray:
        """Normalize parameters to [0, 1] range for optimization"""
        normalized = []
        
        for param_name in self.all_param_names:
            if param_name not in params:
                self.logger.warning(f"Parameter {param_name} not found, using mid-range value")
                normalized.append(0.5)
                continue
                
            value = params[param_name]
            bounds = self.param_bounds[param_name]
            norm_value = (value - bounds['min']) / (bounds['max'] - bounds['min'])
            normalized.append(np.clip(norm_value, 0.0, 1.0))
        
        return np.array(normalized)
    
    def denormalize_parameters(self, normalized_params: np.ndarray) -> Dict[str, float]:
        """Denormalize parameters from [0, 1] range to actual values"""
        params = {}
        
        for i, param_name in enumerate(self.all_param_names):
            bounds = self.param_bounds[param_name]
            value = (normalized_params[i] * (bounds['max'] - bounds['min']) + 
                    bounds['min'])
            params[param_name] = float(value)
        
        return params
    
    def validate_parameters(self, params: Dict[str, float]) -> bool:
        """Validate that parameters are within bounds"""
        for param_name, value in params.items():
            if param_name in self.param_bounds:
                bounds = self.param_bounds[param_name]
                if not (bounds['min'] <= value <= bounds['max']):
                    self.logger.warning(
                        f"Parameter {param_name}={value} outside bounds "
                        f"[{bounds['min']}, {bounds['max']}]"
                    )
                    return False
        
        return True
    
    def update_ngen_configs(self, params: Dict[str, float]) -> bool:
        """
        Update NGEN configuration files with new parameter values
        
        This updates the catchment-specific BMI configuration files
        """
        try:
            # Find all catchment config files
            config_dir = self.ngen_settings_dir / 'model_configs'
            
            if not config_dir.exists():
                self.logger.error(f"Config directory not found: {config_dir}")
                return False
            
            # Update each module's configuration
            for module in self.modules_to_calibrate:
                module_params = {k: v for k, v in params.items() 
                               if k in self.module_params[module]}
                
                if not module_params:
                    continue
                
                success = self._update_module_configs(module, module_params, config_dir)
                if not success:
                    self.logger.error(f"Failed to update {module} configs")
                    return False
            
            self.logger.info("Successfully updated all NGEN configuration files")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating NGEN configs: {str(e)}")
            return False
    
    def _update_module_configs(self, module: str, params: Dict[str, float], 
                               config_dir: Path) -> bool:
        """Update configuration files for a specific module"""
        try:
            # Find config files for this module
            pattern = f"*{module.lower()}*.json"
            config_files = list(config_dir.glob(pattern))
            
            if not config_files:
                self.logger.warning(f"No config files found for {module}")
                return True  # Not an error if module not used
            
            for config_file in config_files:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Update parameters in the config
                if 'config' in config_data:
                    config_section = config_data['config']
                elif 'parameters' in config_data:
                    config_section = config_data['parameters']
                else:
                    config_section = config_data
                
                for param_name, param_value in params.items():
                    # Handle nested parameter structures
                    param_key = param_name.replace('noah_', '').replace('cfe_', '')
                    config_section[param_key] = param_value
                
                # Write updated config
                with open(config_file, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                self.logger.debug(f"Updated {config_file.name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating {module} configs: {str(e)}")
            return False
