#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NextGen (ngen) Parameter Manager

Handles ngen parameter bounds, normalization, denormalization, and 
configuration file updates for model calibration.

Author: CONFLUENCE Development Team
Date: 2025
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging


class NgenParameterManager:
    """Manages ngen calibration parameters across CFE, NOAH-OWP, and PET modules"""
    
    def __init__(self, config: Dict, logger: logging.Logger, ngen_settings_dir: Path):
        """
        Initialize ngen parameter manager.
        
        Args:
            config: Configuration dictionary
            logger: Logger object
            ngen_settings_dir: Path to ngen settings directory
        """
        self.config = config
        self.logger = logger
        self.ngen_settings_dir = ngen_settings_dir
        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')
        
        # Parse which modules to calibrate
        self.modules_to_calibrate = self._parse_modules_to_calibrate()
        
        # Parse parameters to calibrate for each module
        self.params_to_calibrate = self._parse_parameters_to_calibrate()
        
        # Define parameter bounds for all supported ngen parameters
        self.param_bounds = self._get_default_ngen_bounds()
        
        # Path to ngen configuration files
        self.data_dir = Path(config.get('CONFLUENCE_DATA_DIR'))
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.ngen_sim_dir = self.project_dir / 'simulations' / self.experiment_id / 'ngen'
        self.ngen_setup_dir = self.project_dir / 'settings' / 'ngen'
        
        # Configuration file paths
        self.realization_config = self.ngen_setup_dir / 'realization.json'
        self.cfe_config = self.ngen_setup_dir / 'CFE' / 'cfe_config.json'
        self.pet_config = self.ngen_setup_dir / 'PET' / 'pet_config.json'
        self.noah_config = self.ngen_setup_dir / 'NOAH' / 'noah_config.json'
        
        self.logger.info(f"NgenParameterManager initialized")
        self.logger.info(f"Calibrating modules: {self.modules_to_calibrate}")
        self.logger.info(f"Total parameters to calibrate: {len(self.all_param_names)}")
    
    def _parse_modules_to_calibrate(self) -> List[str]:
        """Parse which ngen modules to calibrate from config"""
        modules_str = self.config.get('NGEN_MODULES_TO_CALIBRATE', 'CFE')
        modules = [m.strip().upper() for m in modules_str.split(',') if m.strip()]
        
        # Validate modules
        valid_modules = ['CFE', 'NOAH', 'PET']
        for module in modules:
            if module not in valid_modules:
                self.logger.warning(f"Unknown module '{module}', skipping")
                modules.remove(module)
        
        return modules if modules else ['CFE']  # Default to CFE
    
    def _parse_parameters_to_calibrate(self) -> Dict[str, List[str]]:
        """Parse parameters to calibrate for each module"""
        params = {}
        
        # CFE parameters
        if 'CFE' in self.modules_to_calibrate:
            cfe_params_str = self.config.get('NGEN_CFE_PARAMS_TO_CALIBRATE', 
                                            'maxsmc,satdk,bb,slop')
            params['CFE'] = [p.strip() for p in cfe_params_str.split(',') if p.strip()]
        
        # NOAH-OWP parameters
        if 'NOAH' in self.modules_to_calibrate:
            noah_params_str = self.config.get('NGEN_NOAH_PARAMS_TO_CALIBRATE', 
                                             'refkdt,slope,smcmax,dksat')
            params['NOAH'] = [p.strip() for p in noah_params_str.split(',') if p.strip()]
        
        # PET parameters
        if 'PET' in self.modules_to_calibrate:
            pet_params_str = self.config.get('NGEN_PET_PARAMS_TO_CALIBRATE', 
                                            'wind_speed_measurement_height_m')
            params['PET'] = [p.strip() for p in pet_params_str.split(',') if p.strip()]
        
        return params
    
    def _get_default_ngen_bounds(self) -> Dict[str, Dict[str, float]]:
        """
        Get default parameter bounds for ngen modules.
        
        Returns:
            Dictionary of parameter bounds with 'min' and 'max' values
        """
        bounds = {}
        
        # CFE (Conceptual Functional Equivalent) parameters
        bounds['maxsmc'] = {'min': 0.2, 'max': 0.7}      # Maximum soil moisture content (fraction)
        bounds['wltsmc'] = {'min': 0.01, 'max': 0.3}     # Wilting point soil moisture (fraction)
        bounds['satdk'] = {'min': 1e-7, 'max': 1e-4}     # Saturated hydraulic conductivity (m/s)
        bounds['satpsi'] = {'min': 0.01, 'max': 1.0}     # Saturated soil potential (m)
        bounds['bb'] = {'min': 2.0, 'max': 14.0}         # Pore size distribution index (-)
        bounds['mult'] = {'min': 500.0, 'max': 5000.0}   # Multiplier parameter (mm)
        bounds['slop'] = {'min': 0.0, 'max': 1.0}        # TOPMODEL slope parameter (-)
        bounds['smcmax'] = {'min': 0.3, 'max': 0.6}      # Maximum soil moisture (m3/m3)
        bounds['alpha_fc'] = {'min': 0.25, 'max': 0.95}  # Field capacity coefficient (-)
        bounds['expon'] = {'min': 1.0, 'max': 8.0}       # Exponent parameter (-)
        bounds['Klf'] = {'min': 0.0, 'max': 1.0}         # Lateral flow coefficient (-)
        bounds['Kn'] = {'min': 0.001, 'max': 0.5}        # Nash cascade coefficient (1/h)
        
        # NOAH-OWP parameters
        bounds['refkdt'] = {'min': 0.5, 'max': 5.0}      # Surface runoff parameter (-)
        bounds['slope'] = {'min': 0.1, 'max': 1.0}       # Slope (-)
        bounds['smcmax'] = {'min': 0.3, 'max': 0.6}      # Max soil moisture (m3/m3) - duplicate with CFE
        bounds['dksat'] = {'min': 1e-7, 'max': 1e-4}     # Saturated hydraulic conductivity (m/s)
        bounds['psisat'] = {'min': 0.01, 'max': 1.0}     # Saturated soil potential (m)
        bounds['bexp'] = {'min': 2.0, 'max': 14.0}       # Pore size distribution (-)
        bounds['smcwlt'] = {'min': 0.01, 'max': 0.3}     # Wilting point (m3/m3)
        bounds['smcref'] = {'min': 0.1, 'max': 0.5}      # Reference soil moisture (m3/m3)
        
        # PET (Potential Evapotranspiration) parameters
        bounds['wind_speed_measurement_height_m'] = {'min': 2.0, 'max': 10.0}  # Wind measurement height (m)
        bounds['humidity_measurement_height_m'] = {'min': 2.0, 'max': 10.0}    # Humidity measurement height (m)
        bounds['vegetation_height_m'] = {'min': 0.01, 'max': 2.0}              # Vegetation height (m)
        bounds['zero_plane_displacement_height_m'] = {'min': 0.0, 'max': 1.5}  # Zero plane displacement (m)
        bounds['momentum_transfer_roughness_length_m'] = {'min': 0.001, 'max': 0.5}  # Roughness length (m)
        
        return bounds
    
    @property
    def all_param_names(self) -> List[str]:
        """Get list of all parameter names to calibrate across all modules"""
        all_params = []
        for module, params in self.params_to_calibrate.items():
            # Prefix parameters with module name to avoid conflicts
            all_params.extend([f"{module}.{p}" for p in params])
        return all_params
    
    def get_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """Get parameter bounds for all parameters to calibrate"""
        bounds = {}
        
        for module, params in self.params_to_calibrate.items():
            for param in params:
                full_param_name = f"{module}.{param}"
                if param in self.param_bounds:
                    bounds[full_param_name] = self.param_bounds[param]
                else:
                    self.logger.warning(
                        f"No bounds defined for parameter {param}, using default [0.1, 10.0]"
                    )
                    bounds[full_param_name] = {'min': 0.1, 'max': 10.0}
        
        return bounds
    
    def get_default_parameters(self) -> Dict[str, float]:
        """Get default parameter values (middle of bounds)"""
        bounds = self.get_parameter_bounds()
        params = {}
        
        for param_name, param_bounds in bounds.items():
            params[param_name] = (param_bounds['min'] + param_bounds['max']) / 2.0
        
        return params
    
    def normalize_parameters(self, params: Dict[str, float]) -> np.ndarray:
        """
        Normalize parameters to [0, 1] range for optimization.
        
        Args:
            params: Dictionary of parameter names and values
            
        Returns:
            Normalized parameter array
        """
        bounds = self.get_parameter_bounds()
        normalized = []
        
        for param_name in self.all_param_names:
            value = params[param_name]
            param_bounds = bounds[param_name]
            norm_value = (value - param_bounds['min']) / (param_bounds['max'] - param_bounds['min'])
            normalized.append(np.clip(norm_value, 0.0, 1.0))
        
        return np.array(normalized)
    
    def denormalize_parameters(self, normalized_params: np.ndarray) -> Dict[str, float]:
        """
        Denormalize parameters from [0, 1] range to actual values.
        
        Args:
            normalized_params: Normalized parameter array
            
        Returns:
            Dictionary of denormalized parameters
        """
        bounds = self.get_parameter_bounds()
        params = {}
        
        for i, param_name in enumerate(self.all_param_names):
            param_bounds = bounds[param_name]
            value = (normalized_params[i] * (param_bounds['max'] - param_bounds['min']) + 
                    param_bounds['min'])
            params[param_name] = float(value)
        
        return params
    
    def validate_parameters(self, params: Dict[str, float]) -> bool:
        """
        Validate that parameters are within bounds.
        
        Args:
            params: Dictionary of parameters to validate
            
        Returns:
            True if all parameters valid, False otherwise
        """
        bounds = self.get_parameter_bounds()
        
        for param_name, value in params.items():
            if param_name in bounds:
                param_bounds = bounds[param_name]
                if not (param_bounds['min'] <= value <= param_bounds['max']):
                    self.logger.warning(
                        f"Parameter {param_name}={value} outside bounds "
                        f"[{param_bounds['min']}, {param_bounds['max']}]"
                    )
                    return False
        
        return True
    
    def update_config_files(self, params: Dict[str, float]) -> bool:
        """
        Update ngen configuration files with new parameter values.
        
        Args:
            params: Dictionary of parameters (with module.param naming)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Group parameters by module
            module_params = {}
            for param_name, value in params.items():
                if '.' in param_name:
                    module, param = param_name.split('.', 1)
                    if module not in module_params:
                        module_params[module] = {}
                    module_params[module][param] = value
            
            # Update each module's config file
            success = True
            if 'CFE' in module_params:
                success = success and self._update_cfe_config(module_params['CFE'])
            
            if 'NOAH' in module_params:
                success = success and self._update_noah_config(module_params['NOAH'])
            
            if 'PET' in module_params:
                success = success and self._update_pet_config(module_params['PET'])
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error updating ngen config files: {e}")
            return False
    
    def _update_cfe_config(self, params: Dict[str, float]) -> bool:
        """Update CFE configuration file"""
        try:
            if not self.cfe_config.exists():
                self.logger.error(f"CFE config not found: {self.cfe_config}")
                return False
            
            # Read existing config
            with open(self.cfe_config, 'r') as f:
                config = json.load(f)
            
            # Update parameters
            for param_name, value in params.items():
                if param_name in config:
                    config[param_name] = value
                else:
                    self.logger.warning(f"CFE parameter {param_name} not found in config")
            
            # Write updated config
            with open(self.cfe_config, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.debug(f"Updated CFE config with {len(params)} parameters")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating CFE config: {e}")
            return False
    
    def _update_noah_config(self, params: Dict[str, float]) -> bool:
        """Update NOAH-OWP configuration file"""
        try:
            if not self.noah_config.exists():
                self.logger.error(f"NOAH config not found: {self.noah_config}")
                return False
            
            # Read existing config
            with open(self.noah_config, 'r') as f:
                config = json.load(f)
            
            # Update parameters
            for param_name, value in params.items():
                if param_name in config:
                    config[param_name] = value
                else:
                    self.logger.warning(f"NOAH parameter {param_name} not found in config")
            
            # Write updated config
            with open(self.noah_config, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.debug(f"Updated NOAH config with {len(params)} parameters")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating NOAH config: {e}")
            return False
    
    def _update_pet_config(self, params: Dict[str, float]) -> bool:
        """Update PET configuration file"""
        try:
            if not self.pet_config.exists():
                self.logger.error(f"PET config not found: {self.pet_config}")
                return False
            
            # Read existing config
            with open(self.pet_config, 'r') as f:
                config = json.load(f)
            
            # Update parameters
            for param_name, value in params.items():
                if param_name in config:
                    config[param_name] = value
                else:
                    self.logger.warning(f"PET parameter {param_name} not found in config")
            
            # Write updated config
            with open(self.pet_config, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.debug(f"Updated PET config with {len(params)} parameters")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating PET config: {e}")
            return False
