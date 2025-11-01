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
import re


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
        self.cfe_txt_dir = self.ngen_setup_dir / 'CFE'
        self.pet_config = self.ngen_setup_dir / 'PET' / 'pet_config.json'
        self.noah_config = self.ngen_setup_dir / 'NOAH' / 'noah_config.json'
        self.noah_dir = self.ngen_setup_dir / 'NOAH'
        self.pet_dir  = self.ngen_setup_dir / 'PET'

        # expected JSONs (may not exist; that's fine)
        self.noah_config = self.noah_dir / 'noah_config.json'
        self.pet_config  = self.pet_dir  / 'pet_config.json'

        # BMI text dirs
        self.cfe_txt_dir  = self.ngen_setup_dir / 'CFE'
        self.pet_txt_dir  = self.pet_dir
        
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
        bounds['K_lf'] = {'min': 0.001, 'max': 1.0}      # Lateral flow coefficient (1/hour)
        bounds['K_nash'] = {'min': 0.001, 'max': 0.5}    # Nash cascade coefficient (1/h)
        bounds['Klf'] = {'min': 0.001, 'max': 1.0}       # Alias for K_lf
        bounds['Kn'] = {'min': 0.001, 'max': 0.5}        # Alias for K_nash

        # Add missing groundwater parameter bounds
        bounds['Cgw'] = {'min': 0.0001, 'max': 0.01}     # Groundwater coefficient (m/h)
        bounds['max_gw_storage'] = {'min': 0.01, 'max': 0.5}  # Max GW storage (m)
        
        # NOAH-OWP parameters
        bounds['refkdt'] = {'min': 0.5, 'max': 5.0}      # Surface runoff parameter (-)
        bounds['slope'] = {'min': 0.1, 'max': 1.0}       # Slope (-)
        bounds['smcmax'] = {'min': 0.3, 'max': 0.6}      # Max soil moisture (m3/m3) - duplicate with CFE
        bounds['dksat'] = {'min': 1e-7, 'max': 1e-4}     # Saturated hydraulic conductivity (m/s)
        bounds['psisat'] = {'min': 0.01, 'max': 1.0}     # Saturated soil potential (m)
        bounds['bexp'] = {'min': 2.0, 'max': 14.0}       # Pore size distribution (-)
        bounds['smcwlt'] = {'min': 0.01, 'max': 0.3}     # Wilting point (m3/m3)
        bounds['smcref'] = {'min': 0.1, 'max': 0.5}      # Reference soil moisture (m3/m3)
        bounds['noah_refdk'] = {'min': 1e-7, 'max': 1e-3}     # Reference saturated hydraulic conductivity (m/s)
        bounds['noah_refkdt'] = {'min': 0.5, 'max': 5.0}      # Surface runoff parameter (-)
        bounds['noah_czil'] = {'min': 0.02, 'max': 0.2}       # Zilitinkevich coefficient for roughness (-)
        bounds['noah_z0'] = {'min': 0.001, 'max': 1.0}        # Roughness length (m)
        bounds['noah_frzk'] = {'min': 0.0, 'max': 10.0}       # Frozen ground parameter (-)
        bounds['noah_salp'] = {'min': -2.0, 'max': 2.0}       # Shape parameter for surface runoff (-)
        
        # PET (Potential Evapotranspiration) parameters
        bounds['wind_speed_measurement_height_m'] = {'min': 2.0, 'max': 10.0}  # Wind measurement height (m)
        bounds['humidity_measurement_height_m'] = {'min': 2.0, 'max': 10.0}    # Humidity measurement height (m)
        bounds['vegetation_height_m'] = {'min': 0.01, 'max': 2.0}              # Vegetation height (m)
        bounds['zero_plane_displacement_height_m'] = {'min': 0.0, 'max': 1.5}  # Zero plane displacement (m)
        bounds['momentum_transfer_roughness_length_m'] = {'min': 0.001, 'max': 0.5}  # Roughness length (m)
        bounds['rain_snow_thresh'] = {'min': -2.0, 'max': 3.0}                 # Rain/snow temperature threshold (°C)
        bounds['ZREF'] = {'min': 2.0, 'max': 10.0}                             # Reference height (m)
        bounds['pet_albedo'] = {'min': 0.05, 'max': 0.95}                      # Surface albedo (fraction)
        bounds['pet_z0_mom'] = {'min': 0.001, 'max': 0.5}                      # Momentum roughness length (m)
        bounds['pet_z0_heat'] = {'min': 0.0001, 'max': 0.1}                    # Heat roughness length (m)
        bounds['pet_veg_h'] = {'min': 0.01, 'max': 30.0}                       # Vegetation height (m)
        bounds['pet_d0'] = {'min': 0.0, 'max': 20.0}                           # Zero plane displacement height (m)
        
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
        """Update CFE configuration: prefer JSON, fallback to BMI .txt.
        Preserves units in [brackets] for BMI text files."""
        try:
            # --- Preferred path: JSON file ---
            if self.cfe_config.exists():
                with open(self.cfe_config, 'r') as f:
                    cfg = json.load(f)
                updated = 0
                for k, v in params.items():
                    if k in cfg:
                        cfg[k] = v
                        updated += 1
                    else:
                        self.logger.warning(f"CFE parameter {k} not found in JSON config")
                with open(self.cfe_config, 'w') as f:
                    json.dump(cfg, f, indent=2)
                self.logger.debug(f"Updated CFE JSON with {updated} parameters")
                return True

            # --- Fallback: BMI text file ---
            # Choose the right file: prefer the one matching {{id}} if known
            candidates = []
            if getattr(self, "hydro_id", None):
                # e.g., cat-1_bmi_config_cfe_pass.txt
                pattern = f"{self.hydro_id}_bmi_config_cfe_*.txt"
                candidates = list(self.cfe_txt_dir.glob(pattern))

            # If we didn't find any by id, fall back to a single *.txt in CFE/
            if not candidates:
                candidates = list(self.cfe_txt_dir.glob("*.txt"))

            if len(candidates) == 0:
                self.logger.error(f"CFE config not found (no JSON, no BMI .txt in {self.cfe_txt_dir})")
                return False
            if len(candidates) > 1:
                # Ambiguity guard: don’t risk updating multiple files unexpectedly
                self.logger.error(f"Multiple BMI .txt files in {self.cfe_txt_dir}; please set NGEN_ACTIVE_CATCHMENT_ID or prune files")
                return False

            path = candidates[0]
            lines = path.read_text().splitlines()

            # Map DE parameter names to BMI keys in the file
            keymap = {
                # Soil parameters
                "bb": "soil_params.b",
                "satdk": "soil_params.satdk",
                "slop": "soil_params.slop",
                "maxsmc": "soil_params.smcmax",
                "smcmax": "soil_params.smcmax",
                "wltsmc": "soil_params.wltsmc",
                "satpsi": "soil_params.satpsi",
                "expon": "soil_params.expon",  # Note: also exists as root-level param
                
                # Groundwater parameters
                "Cgw": "Cgw",
                "max_gw_storage": "max_gw_storage",
                
                # Routing parameters 
                "K_nash": "K_nash",
                "K_lf": "K_lf",
                "Kn": "K_nash",      # Alias for K_nash
                "Klf": "K_lf",       # Alias for K_lf
                
                # Other CFE parameters
                "alpha_fc": "alpha_fc",
                "refkdt": "refkdt",
            }

            # Helper: write numeric value preserving any trailing [units]
            num_units_re = re.compile(r"""
                ^\s*             # leading space
                (?P<num>[+-]?(?:\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)  # number (incl. sci)
                (?P<tail>\s*(\[[^\]]*\])?.*)$   # optional units and remainder
            """, re.VERBOSE)

            def render_value(original_rhs: str, new_val: float) -> str:
                m = num_units_re.match(original_rhs.strip())
                if m:
                    tail = m.group('tail') or ''
                    # Use compact but stable float formatting
                    return f"{new_val:.8g}{tail}"
                # If parsing fails, just replace entirely with the new number
                return f"{new_val:.8g}"

            updated = set()
            for i, line in enumerate(lines):
                if "=" not in line or line.strip().startswith("#"):
                    continue
                k, rhs = line.split("=", 1)
                k = k.strip()
                rhs_keep = rhs.rstrip("\n")
                # Match parameters by mapped BMI key
                for p, bmi_k in keymap.items():
                    if p in params and k == bmi_k:
                        new_rhs = render_value(rhs_keep, params[p])
                        lines[i] = f"{k}={new_rhs}"
                        updated.add(p)

            # Warn about any requested params we couldn’t find in the BMI file
            for p in params:
                if p not in updated and p in keymap:
                    self.logger.warning(f"CFE parameter {p} not found in BMI config {path.name}")

            path.write_text("\n".join(lines) + "\n")
            self.logger.debug(f"Updated CFE BMI text ({path.name}) with {len(updated)} parameters")
            return True

        except Exception as e:
            self.logger.error(f"Error updating CFE config: {e}")
            return False


        
    def _update_noah_config(self, params: Dict[str, float]) -> bool:
        """
        Update NOAH configuration for calibration:
        1) Prefer JSON if present.
        2) Fallback to NOAH BMI input file ({{id}}.input).
        3) Optionally update TBL parameters in NOAH/parameters (if mappings supplied).
        """
        try:
            # ---------- 1) JSON path ----------
            if self.noah_config.exists():
                with open(self.noah_config, 'r') as f:
                    cfg = json.load(f)
                updated = 0
                for k, v in params.items():
                    if k in cfg:
                        cfg[k] = v
                        updated += 1
                    else:
                        self.logger.warning(f"NOAH parameter {k} not in JSON config")
                with open(self.noah_config, 'w') as f:
                    json.dump(cfg, f, indent=2)
                self.logger.debug(f"Updated NOAH JSON with {updated} parameters")
                return True

            # ---------- 2) BMI input fallback ----------
            # Select the right *.input (prefer the one matching the active id)
            if not self.noah_dir.exists():
                self.logger.error(f"NOAH directory missing: {self.noah_dir}")
                return False

            input_candidates = []
            if getattr(self, "hydro_id", None):
                input_candidates = list(self.noah_dir.glob(f"{self.hydro_id}.input"))
            if not input_candidates:
                input_candidates = list(self.noah_dir.glob("*.input"))

            if len(input_candidates) == 0:
                self.logger.error(f"NOAH config not found: no JSON and no *.input under {self.noah_dir}")
                return False
            if len(input_candidates) > 1:
                self.logger.error(f"Multiple NOAH *.input files; set NGEN_ACTIVE_CATCHMENT_ID to disambiguate")
                return False

            ipath = input_candidates[0]
            text = ipath.read_text()

            # Map DE param names -> (&section, key) within the NOAH input.
            # Start with a small set; add more as you choose to calibrate them.
            keymap = {
                # de_name : (section, key)
                "rain_snow_thresh": ("forcing", "rain_snow_thresh"),
                "ZREF"            : ("forcing", "ZREF"),
                "dt"              : ("timing", "dt"),  # careful: affects timestep!
            }

            import re
            sec_re_template = r"(?s)&\s*{section}\b(.*?)/"
            key_re_template = r"(^|\n)(\s*{key}\s*=\s*)([^,\n/]+)"

            def replace_in_namelist(txt: str, section: str, key: str, new_val: float) -> tuple[str, bool]:
                sec_re = re.compile(sec_re_template.format(section=re.escape(section)))
                m_sec = sec_re.search(txt)
                if not m_sec:
                    return txt, False
                block = m_sec.group(1)
                key_re = re.compile(key_re_template.format(key=re.escape(key)), re.MULTILINE)

                def _sub(mm):
                    prefix = mm.group(2)
                    rhs = mm.group(3).strip()
                    # Preserve quotes if present (rare for numeric keys, but safe)
                    if rhs.startswith('"') and rhs.endswith('"'):
                        return f"{mm.group(1)}{prefix}\"{new_val:.8g}\""
                    else:
                        return f"{mm.group(1)}{prefix}{new_val:.8g}"

                new_block, n = key_re.subn(_sub, block, count=1)
                if n == 0:
                    return txt, False
                return txt[:m_sec.start(1)] + new_block + txt[m_sec.end(1):], True

            updated_inputs = 0
            for p, (sec, key) in keymap.items():
                if p in params:
                    text, ok = replace_in_namelist(text, sec, key, params[p])
                    if ok:
                        updated_inputs += 1
                    else:
                        self.logger.warning(f"NOAH param {p} ({sec}.{key}) not found in {ipath.name}")

            if updated_inputs > 0:
                ipath.write_text(text)
                self.logger.debug(f"Updated NOAH input ({ipath.name}) with {updated_inputs} parameter(s)")
                return True

            # ---------- 3) Optional: TBL updates ----------
            # If nothing changed in *.input, try TBL mappings if provided.
            # Expect a structure like:
            # self.noah_tbl_map = {
            #   "REFKDT": ("MPTABLE.TBL", "REFKDT", 1),    # (file, variable, column_index or None)
            #   "REFDK" : ("MPTABLE.TBL", "REFDK", 1),
            #   "SLOPE" : ("SOILPARM.TBL", "SLOPE",  <col>),
            # }
            tbl_map: Dict[str, Tuple[str, str, Optional[int]]] = getattr(self, "noah_tbl_map", {})

            if not tbl_map:
                # Nothing to do is not an error; we may just not be calibrating NOAH today.
                return True

            # Stage per-run parameters dir (recommended)
            params_dir = self.noah_dir / "parameters"
            if not params_dir.exists():
                self.logger.error(f"NOAH parameters directory missing: {params_dir}")
                return False

            # Implement minimal editor: update numeric for a row that starts with var name.
            def edit_tbl_value(tbl_path: Path, var: str, col: Optional[int], new_val: float) -> bool:
                if not tbl_path.exists():
                    self.logger.error(f"TBL not found: {tbl_path}")
                    return False
                lines = tbl_path.read_text().splitlines()
                changed = False
                for i, line in enumerate(lines):
                    if not line.strip() or line.strip().startswith("#"):
                        continue
                    # naive match: variable name at start (allow whitespace)
                    if line.lstrip().startswith(var):
                        parts = line.split()
                        # If first token is the var name, replace either the single value or a specific column
                        if parts[0] == var:
                            if col is None:
                                # single-valued variable
                                if len(parts) >= 2:
                                    parts[1] = f"{new_val:.8g}"
                                    lines[i] = " ".join(parts)
                                    changed = True
                                    break
                            else:
                                # multi-column; ensure index in bounds
                                idx = int(col)
                                if idx < len(parts):
                                    parts[idx] = f"{new_val:.8g}"
                                    lines[i] = " ".join(parts)
                                    changed = True
                                    break
                if changed:
                    tbl_path.write_text("\n".join(lines) + "\n")
                return changed

            updated_tbls = 0
            for p, (fname, var, col) in tbl_map.items():
                if p not in params:
                    continue
                tbl_path = params_dir / fname
                if edit_tbl_value(tbl_path, var, col, params[p]):
                    updated_tbls += 1
                else:
                    self.logger.warning(f"NOAH TBL param {p} ({fname}:{var}[{col}]) not found/updated")

            if updated_tbls > 0:
                self.logger.debug(f"Updated NOAH TBLs with {updated_tbls} parameter(s)")
                return True

            # Nothing updated; not fatal (maybe NOAH not being calibrated this run)
            return True

        except Exception as e:
            self.logger.error(f"Error updating NOAH config: {e}")
            return False

        
    def _update_pet_config(self, params: Dict[str, float]) -> bool:
        """
        Update PET configuration:
        1) Prefer JSON if present.
        2) Fallback to PET BMI text file: PET/{{id}}_pet_config.txt (or the only *.txt).
        """
        keymap = {
                    "rain_snow_thresh": ("forcing", "rain_snow_thresh"),
                    "ZREF"            : ("forcing", "ZREF"),
                }
        try:
            # ---------- 1) JSON ----------
            if self.pet_config.exists():
                with open(self.pet_config, 'r') as f:
                    cfg = json.load(f)
                up = 0
                for k, v in params.items():
                    if k in cfg:
                        cfg[k] = v; up += 1
                    else:
                        self.logger.warning(f"PET parameter {k} not in JSON config")
                with open(self.pet_config, 'w') as f:
                    json.dump(cfg, f, indent=2)
                self.logger.debug(f"Updated PET JSON with {up} parameter(s)")
                return True

            # ---------- 2) BMI text ----------
            # pick file by hydro_id if present, else a single *.txt under PET/
            if not self.pet_txt_dir.exists():
                self.logger.error(f"PET directory missing: {self.pet_txt_dir}")
                return False

            candidates = []
            if getattr(self, "hydro_id", None):
                candidates = list(self.pet_txt_dir.glob(f"{self.hydro_id}_pet_config.txt"))
            if not candidates:
                candidates = list(self.pet_txt_dir.glob("*.txt"))

            if len(candidates) == 0:
                self.logger.error(f"PET config not found: no JSON and no *.txt in {self.pet_txt_dir}")
                return False
            if len(candidates) > 1:
                self.logger.error(f"Multiple PET *.txt configs; set NGEN_ACTIVE_CATCHMENT_ID to disambiguate")
                return False

            path = candidates[0]
            lines = path.read_text().splitlines()

            import re
            num_units_re = re.compile(r"""
                ^\s*
                (?P<num>[+-]?(?:\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)
                (?P<tail>\s*(\[[^\]]*\])?.*)$
            """, re.VERBOSE)

            def render_value(rhs: str, new_val: float) -> str:
                m = num_units_re.match(rhs.strip())
                if m:
                    tail = m.group('tail') or ''
                    return f"{new_val:.8g}{tail}"
                return f"{new_val:.8g}"

            # Map DE param names -> keys in PET text (adjust to your file)
            keymap = {
                # "albedo": "albedo",
                # "z0m"   : "surface_roughness",
                # add the keys that actually exist in your PET file
            }

            updated = set()
            for i, line in enumerate(lines):
                if "=" not in line or line.strip().startswith("#"):
                    continue
                k, rhs = line.split("=", 1)
                key = k.strip()
                if not key:
                    continue
                for p, txt_key in keymap.items():
                    if p in params and key == txt_key:
                        lines[i] = f"{key}={render_value(rhs, params[p])}"
                        updated.add(p)

            for p in params:
                if p in keymap and p not in updated:
                    self.logger.warning(f"PET parameter {p} not found in {path.name}")

            if updated:
                path.write_text("\n".join(lines) + "\n")
                self.logger.debug(f"Updated PET BMI text ({path.name}) with {len(updated)} parameter(s)")
            return True

        except Exception as e:
            self.logger.error(f"Error updating PET config: {e}")
            return False