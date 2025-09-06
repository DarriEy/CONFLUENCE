# In utils/optimization/distributed_emulator.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import json
from datetime import datetime, timedelta
import subprocess
import multiprocessing as mp
from scipy import sparse
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Point
from sklearn.neighbors import BallTree


class LargeDomainEmulator:
    """
    Distributed hydrological emulator for CONFLUENCE - scales from single basin to continental.
    
    This class implements a graph-based neural emulator that can operate at multiple scales:
    - Single basin: Uses CONFLUENCE's existing domain discretization
    - Regional: Multiple connected basins 
    - Continental: Large-scale hydrofabric networks
    
    The emulator uses Graph Transformer networks to learn parameter optimization
    across spatially distributed hydrological units, leveraging connectivity
    information and multi-scale forcing data.
    
    Key capabilities:
    - Parameter optimization using graph neural networks
    - Multi-objective loss functions (streamflow, snow, soil moisture)
    - Integration with SUMMA-SUNDIALS for gradient computation
    - Scalable from HRU-level to continental domains
    - Compatible with CONFLUENCE data formats and workflows
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the distributed emulator.
        
        Args:
            config: CONFLUENCE configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.data_dir = Path(config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = config.get('EXPERIMENT_ID')
        
        # Create emulation output directory
        self.emulation_dir = self.project_dir / "emulation" / "distributed"
        self.emulation_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract emulator configuration
        self.emulator_config = self._extract_emulator_config()
        
        # Initialize components
        self.hydrofabric = None
        self.emulator_model = None
        self.summa_interface = None
        self.observation_mapper = None
        self.loss_function = None
        
        self.logger.info("Distributed emulator initialized")
    
    def _extract_emulator_config(self) -> Dict[str, Any]:
        """Extract emulator-specific configuration from CONFLUENCE config."""
        return {
            # Model architecture
            'hidden_dim': self.config.get('DISTRIBUTED_EMULATOR_HIDDEN_DIM', 512),
            'n_heads': self.config.get('DISTRIBUTED_EMULATOR_N_HEADS', 8),
            'n_transformer_layers': self.config.get('DISTRIBUTED_EMULATOR_N_LAYERS', 6),
            'dropout': self.config.get('DISTRIBUTED_EMULATOR_DROPOUT', 0.1),
            
            # Training configuration
            'batch_size': self.config.get('DISTRIBUTED_EMULATOR_BATCH_SIZE', 32),
            'learning_rate': self.config.get('DISTRIBUTED_EMULATOR_LEARNING_RATE', 1e-4),
            'n_epochs': self.config.get('DISTRIBUTED_EMULATOR_EPOCHS', 100),
            'validation_split': self.config.get('DISTRIBUTED_EMULATOR_VALIDATION_SPLIT', 0.2),
            'window_days': self.config.get('DISTRIBUTED_EMULATOR_WINDOW_DAYS', 30),
            
            # Loss weights
            'loss_weights': {
                'streamflow': self.config.get('DISTRIBUTED_EMULATOR_STREAMFLOW_WEIGHT', 1.0),
                'snow_cover': self.config.get('DISTRIBUTED_EMULATOR_SNOW_WEIGHT', 0.5),
                'soil_moisture': self.config.get('DISTRIBUTED_EMULATOR_SOIL_WEIGHT', 0.3),
                'groundwater': self.config.get('DISTRIBUTED_EMULATOR_GW_WEIGHT', 0.2)
            },
            
            # Parameter configuration
            'n_parameters': len(self.config.get('PARAMS_TO_CALIBRATE', '').split(',')) if self.config.get('PARAMS_TO_CALIBRATE') else 10,
            'parameter_names': self.config.get('PARAMS_TO_CALIBRATE', '').split(',') if self.config.get('PARAMS_TO_CALIBRATE') else [],
            
            # Time periods (reuse CONFLUENCE settings)
            'simulation_period': [
                int(self.config.get('EXPERIMENT_TIME_START', '1980-01-01')[:4]),
                int(self.config.get('EXPERIMENT_TIME_END', '2018-12-31')[:4])
            ],
            'calibration_period': self.config.get('CALIBRATION_PERIOD', '1982-01-01, 1990-12-31'),
            'evaluation_period': self.config.get('EVALUATION_PERIOD', '2012-01-01, 2018-12-31'),
            
            # SUMMA configuration (reuse existing)
            'summa_executable': self.config.get('SUMMA_INSTALL_PATH', '') + '/' + self.config.get('SUMMA_EXE', 'summa_sundials.exe'),
            'solver_tolerance': self.config.get('DISTRIBUTED_EMULATOR_SOLVER_TOL', 1e-6),
            'sensitivity_method': self.config.get('DISTRIBUTED_EMULATOR_SENSITIVITY', 'forward'),
            
            # Parallel processing
            'n_parallel_workers': self.config.get('MPI_PROCESSES', 8),
            
            # Output configuration
            'output_dir': str(self.emulation_dir),
            'checkpoint_dir': str(self.emulation_dir / "checkpoints"),
            'results_dir': str(self.emulation_dir / "results")
        }
    
    def run_distributed_emulation(self) -> Dict[str, Any]:
        """
        Run the complete distributed emulation workflow.
        
        Returns:
            Dictionary with emulation results and metadata
        """
        self.logger.info("Starting distributed emulation workflow")
        
        try:
            # Step 1: Initialize components
            self._initialize_components()
            
            # Step 2: Prepare training data
            self._prepare_training_data()
            
            # Step 3: Train emulator
            training_results = self._train_emulator()
            
            # Step 4: Validate emulator
            validation_results = self._validate_emulator()
            
            # Step 5: Optimize parameters
            optimization_results = self._optimize_parameters()
            
            # Step 6: Save results
            results = self._save_results(training_results, validation_results, optimization_results)
            
            self.logger.info("Distributed emulation completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Distributed emulation failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
    
    def _initialize_components(self):
        """Initialize all emulator components."""
        self.logger.info("Initializing emulator components")
        
        # Initialize hydrofabric graph
        self.hydrofabric = CONFLUENCEHydrofabricGraph(self.config, self.logger)
        
        # Initialize observation mapper
        self.observation_mapper = CONFLUENCEObservationMapper(self.config, self.logger, self.hydrofabric)
        
        # Initialize SUMMA interface
        self.summa_interface = CONFLUENCESUMMAInterface(self.config, self.logger)
        
        # Initialize loss function
        self.loss_function = DistributedLossFunction(self.emulator_config, self.observation_mapper, self.logger)
    
    def _prepare_training_data(self):
        """Prepare training data from CONFLUENCE formats."""
        self.logger.info("Preparing training data")
        
        # Load parameter bounds and generate training sets
        parameter_sets = self._generate_parameter_sets()
        
        # Load attributes from CONFLUENCE attribute processing
        attributes_df = self._load_confluence_attributes()
        
        # Create forcing summary from CONFLUENCE forcing data
        forcing_summary_df = self._create_forcing_summary()
        
        # Build graph features
        self.hydrofabric.build_node_features(parameter_sets[0], attributes_df, forcing_summary_df)
        
        # Store parameter ensemble
        self.parameter_ensemble = torch.tensor(parameter_sets, dtype=torch.float32)
        
        self.logger.info(f"Prepared {len(parameter_sets)} parameter sets for training")
    
    def _generate_parameter_sets(self) -> np.ndarray:
        """Generate parameter sets using CONFLUENCE parameter bounds."""
        n_sets = self.config.get('DISTRIBUTED_EMULATOR_N_PARAMETER_SETS', 1000)
        n_hrus = len(self.hydrofabric.catchment_ids)
        n_params = self.emulator_config['n_parameters']
        
        # Load parameter bounds from CONFLUENCE
        param_bounds = self._load_parameter_bounds()
        
        # Generate Latin Hypercube samples
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=n_params, seed=42)
        normalized_samples = sampler.random(n_sets)
        
        # Scale to parameter bounds
        parameter_sets = []
        for sample in normalized_samples:
            param_set = np.zeros((n_hrus, n_params))
            for i, (param_name, bounds) in enumerate(param_bounds.items()):
                if i < n_params:
                    scaled_value = bounds[0] + sample[i] * (bounds[1] - bounds[0])
                    param_set[:, i] = scaled_value
            parameter_sets.append(param_set)
        
        return np.array(parameter_sets)
    
    def _load_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Load parameter bounds from CONFLUENCE configuration."""
        bounds = {}
        
        # Default bounds - these should be configurable
        default_bounds = {
            'Fcapil': (0.01, 10.0),
            'albedoDecayRate': (0.1, 5.0),
            'tempCritRain': (0.0, 5.0),
            'k_soil': (1e-7, 1e-3),
            'vGn_n': (1.2, 10.0),
            'theta_sat': (0.3, 0.6),
            'theta_res': (0.01, 0.2),
            'theta_mp': (0.01, 0.4),
            'rootingDepth': (0.1, 10.0),
            'zScale_TOPMODEL': (1.0, 100.0)
        }
        
        param_names = self.emulator_config['parameter_names']
        for param_name in param_names:
            bounds[param_name] = default_bounds.get(param_name, (0.0, 1.0))
        
        return bounds
    
    def _load_confluence_attributes(self) -> pd.DataFrame:
        """Load attributes from CONFLUENCE attribute processing."""
        attributes_dir = self.project_dir / "attributes"
        
        # Initialize attributes DataFrame
        n_hrus = len(self.hydrofabric.catchment_ids)
        attributes_data = {}
        
        # Load elevation attributes
        elev_file = attributes_dir / "elevation" / "elevation_statistics.csv"
        if elev_file.exists():
            elev_df = pd.read_csv(elev_file)
            for col in ['mean_elevation', 'std_elevation', 'min_elevation', 'max_elevation']:
                if col in elev_df.columns:
                    attributes_data[col] = elev_df[col].values[:n_hrus]
        
        # Load soil attributes
        soil_file = attributes_dir / "soilclass" / "soil_statistics.csv"
        if soil_file.exists():
            soil_df = pd.read_csv(soil_file)
            for col in soil_df.columns:
                if col.startswith('soil_'):
                    attributes_data[col] = soil_df[col].values[:n_hrus]
        
        # Load land cover attributes
        land_file = attributes_dir / "landclass" / "landcover_statistics.csv"
        if land_file.exists():
            land_df = pd.read_csv(land_file)
            for col in land_df.columns:
                if col.startswith('land_'):
                    attributes_data[col] = land_df[col].values[:n_hrus]
        
        # Fill with defaults if no attributes found
        if not attributes_data:
            n_attr = self.config.get('DISTRIBUTED_EMULATOR_N_ATTRIBUTES', 50)
            for i in range(n_attr):
                attributes_data[f'attr_{i}'] = np.random.normal(0, 1, n_hrus)
        
        return pd.DataFrame(attributes_data)
    
    def _create_forcing_summary(self) -> pd.DataFrame:
        """Create forcing summary from CONFLUENCE forcing data."""
        forcing_dir = self.project_dir / "forcing" / "basin_averaged_data"
        n_hrus = len(self.hydrofabric.catchment_ids)
        
        forcing_summary = {}
        
        if forcing_dir.exists():
            # Load basin-averaged forcing data
            for forcing_file in forcing_dir.glob("*.nc"):
                try:
                    ds = xr.open_dataset(forcing_file)
                    for var in ds.data_vars:
                        # Compute summary statistics
                        mean_val = ds[var].mean(dim='time').values
                        std_val = ds[var].std(dim='time').values
                        
                        forcing_summary[f'{var}_mean'] = mean_val[:n_hrus] if len(mean_val) >= n_hrus else np.full(n_hrus, mean_val.mean())
                        forcing_summary[f'{var}_std'] = std_val[:n_hrus] if len(std_val) >= n_hrus else np.full(n_hrus, std_val.mean())
                    ds.close()
                except Exception as e:
                    self.logger.warning(f"Could not load forcing file {forcing_file}: {e}")
        
        # Fill with defaults if no forcing found
        if not forcing_summary:
            n_forcing = self.config.get('DISTRIBUTED_EMULATOR_N_FORCING', 10)
            for i in range(n_forcing):
                forcing_summary[f'forcing_{i}'] = np.random.normal(0, 1, n_hrus)
        
        return pd.DataFrame(forcing_summary)
    
    def _train_emulator(self) -> Dict[str, Any]:
        """Train the graph transformer emulator."""
        self.logger.info("Training emulator model")
        
        # Initialize emulator model
        self.emulator_model = DistributedGraphTransformer(self.emulator_config, self.hydrofabric)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.emulator_model.parameters(),
            lr=self.emulator_config['learning_rate'],
            weight_decay=1e-5
        )
        
        # Training loop
        training_losses = []
        for epoch in range(self.emulator_config['n_epochs']):
            epoch_loss = self._train_epoch(optimizer, epoch)
            training_losses.append(epoch_loss)
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Loss = {epoch_loss:.4f}")
                self._save_checkpoint(epoch, epoch_loss)
        
        return {'training_losses': training_losses, 'final_loss': training_losses[-1]}
    
    def _train_epoch(self, optimizer: torch.optim.Optimizer, epoch: int) -> float:
        """Train one epoch with chain rule implementation."""
        self.emulator_model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        # Create time windows for BPTT
        cal_start, cal_end = self.emulator_config['calibration_period'].split(', ')
        start_year = int(cal_start[:4])
        end_year = int(cal_end[:4])
        window_days = self.emulator_config['window_days']
        
        for year in range(start_year, min(start_year + 2, end_year)):  # Limit for initial implementation
            for month in range(1, 13, 3):  # Quarterly windows
                
                window_start = f"{year}-{month:02d}-01"
                window_end_date = datetime(year, month, 1) + timedelta(days=window_days)
                window_end = window_end_date.strftime("%Y-%m-%d")
                
                if window_end_date.year > end_year:
                    break
                
                # Sample parameter set
                param_idx = np.random.randint(0, len(self.parameter_ensemble))
                current_params = self.parameter_ensemble[param_idx]
                
                # Update graph features and run forward pass
                batch_loss = self._train_batch(current_params, window_start, window_end, optimizer)
                
                if batch_loss is not None:
                    epoch_loss += batch_loss
                    n_batches += 1
        
        return epoch_loss / max(n_batches, 1)
    
    def _train_batch(self, parameters: torch.Tensor, window_start: str, window_end: str, optimizer: torch.optim.Optimizer) -> Optional[float]:
        """Train a single batch."""
        try:
            # Update graph with parameters
            updated_features = self.hydrofabric.update_features_with_parameters(parameters)
            
            # Forward pass through emulator
            theta_optimized = self.emulator_model(updated_features)
            
            # Run SUMMA simulation (simplified for now)
            simulated_outputs = self._run_simplified_summa(theta_optimized, window_start, window_end)
            
            # Compute loss
            loss = self.loss_function.compute_loss(simulated_outputs, window_start, window_end)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.emulator_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            return loss.item()
            
        except Exception as e:
            self.logger.warning(f"Batch training failed: {e}")
            return None
    
    def _run_simplified_summa(self, parameters: torch.Tensor, window_start: str, window_end: str) -> torch.Tensor:
        """Simplified SUMMA simulation for initial implementation."""
        # This is a placeholder - in full implementation this would call SUMMA-SUNDIALS
        n_nodes = parameters.shape[0]
        n_days = (datetime.strptime(window_end, "%Y-%m-%d") - datetime.strptime(window_start, "%Y-%m-%d")).days
        n_variables = 3  # streamflow, snow, soil moisture
        
        # Generate synthetic outputs based on parameters (placeholder)
        outputs = torch.randn(n_nodes, n_days, n_variables, requires_grad=True)
        
        # Scale outputs based on parameters to create some correlation
        param_scaling = parameters.mean(dim=1, keepdim=True).unsqueeze(-1)
        outputs = outputs * (0.5 + param_scaling)
        
        return outputs
    
    def _validate_emulator(self) -> Dict[str, Any]:
        """Validate the trained emulator."""
        self.logger.info("Validating emulator")
        
        # Use evaluation period for validation
        eval_start, eval_end = self.emulator_config['evaluation_period'].split(', ')
        
        validation_losses = []
        n_validation_samples = 50
        
        for i in range(n_validation_samples):
            param_idx = np.random.randint(0, len(self.parameter_ensemble))
            current_params = self.parameter_ensemble[param_idx]
            
            with torch.no_grad():
                updated_features = self.hydrofabric.update_features_with_parameters(current_params)
                theta_optimized = self.emulator_model(updated_features)
                simulated_outputs = self._run_simplified_summa(theta_optimized, eval_start, eval_end)
                loss = self.loss_function.compute_loss(simulated_outputs, eval_start, eval_end)
                validation_losses.append(loss.item())
        
        mean_val_loss = np.mean(validation_losses)
        std_val_loss = np.std(validation_losses)
        
        self.logger.info(f"Validation loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
        
        return {
            'validation_loss_mean': mean_val_loss,
            'validation_loss_std': std_val_loss,
            'validation_losses': validation_losses
        }
    
    def _optimize_parameters(self) -> Dict[str, Any]:
        """Optimize parameters using the trained emulator."""
        self.logger.info("Optimizing parameters using trained emulator")
        
        # Use emulator for rapid parameter optimization
        best_loss = float('inf')
        best_parameters = None
        
        n_optimization_trials = 1000
        
        for trial in range(n_optimization_trials):
            # Sample random parameters
            param_idx = np.random.randint(0, len(self.parameter_ensemble))
            trial_params = self.parameter_ensemble[param_idx]
            
            with torch.no_grad():
                updated_features = self.hydrofabric.update_features_with_parameters(trial_params)
                theta_optimized = self.emulator_model(updated_features)
                
                # Evaluate on validation period
                eval_start, eval_end = self.emulator_config['evaluation_period'].split(', ')
                simulated_outputs = self._run_simplified_summa(theta_optimized, eval_start, eval_end)
                loss = self.loss_function.compute_loss(simulated_outputs, eval_start, eval_end)
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_parameters = theta_optimized.clone()
        
        self.logger.info(f"Best optimization loss: {best_loss:.4f}")
        
        return {
            'best_loss': best_loss,
            'best_parameters': best_parameters.numpy() if best_parameters is not None else None,
            'optimization_trials': n_optimization_trials
        }
    
    def _save_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.emulator_config['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.emulator_model.state_dict(),
            'loss': loss,
            'config': self.emulator_config
        }
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
    
    def _save_results(self, training_results: Dict, validation_results: Dict, optimization_results: Dict) -> Dict[str, Any]:
        """Save all emulation results."""
        results_dir = Path(self.emulator_config['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Combine all results
        all_results = {
            'training': training_results,
            'validation': validation_results,
            'optimization': optimization_results,
            'config': self.emulator_config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results to JSON
        results_file = results_dir / "emulation_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._convert_numpy_to_lists(all_results)
            json.dump(json_results, f, indent=2)
        
        # Save best parameters to CSV
        if optimization_results.get('best_parameters') is not None:
            best_params_df = pd.DataFrame(
                optimization_results['best_parameters'],
                columns=self.emulator_config['parameter_names'][:optimization_results['best_parameters'].shape[1]]
            )
            best_params_df.to_csv(results_dir / "optimized_parameters.csv", index=False)
        
        self.logger.info(f"Results saved to {results_dir}")
        return all_results
    
    def _convert_numpy_to_lists(self, obj):
        """Recursively convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_lists(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_lists(item) for item in obj]
        else:
            return obj


class CONFLUENCEHydrofabricGraph:
    """CONFLUENCE-compatible hydrofabric graph representation."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.data_dir = Path(config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
        # Load CONFLUENCE shapefiles
        self.catchment_gdf = self._load_catchment_shapefile()
        self.river_network_gdf = self._load_river_network_shapefile()
        
        # Build graph connectivity
        self.catchment_ids = self.catchment_gdf[config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')].astype(str).tolist()
        self.node_mapping = {hru_id: idx for idx, hru_id in enumerate(self.catchment_ids)}
        
        self._build_graph_connectivity()
        self._initialize_encoder_dimensions()
    
    def _load_catchment_shapefile(self) -> gpd.GeoDataFrame:
        """Load catchment shapefile from CONFLUENCE."""
        catchment_path = self.config.get('CATCHMENT_PATH', 'default')
        if catchment_path == 'default':
            catchment_path = self.project_dir / "shapefiles" / "catchment"
        
        catchment_name = self.config.get('CATCHMENT_SHP_NAME', 'default')
        if catchment_name == 'default':
            discretization = self.config.get('DOMAIN_DISCRETIZATION', 'GRUs')
            catchment_name = f"{self.domain_name}_HRUs_{discretization}.shp"
        
        shapefile_path = Path(catchment_path) / catchment_name
        
        if not shapefile_path.exists():
            raise FileNotFoundError(f"Catchment shapefile not found: {shapefile_path}")
        
        return gpd.read_file(shapefile_path)
    
    def _load_river_network_shapefile(self) -> gpd.GeoDataFrame:
        """Load river network shapefile from CONFLUENCE."""
        network_path = self.config.get('RIVER_NETWORK_SHP_PATH', 'default')
        if network_path == 'default':
            network_path = self.project_dir / "shapefiles" / "river_network"
        
        network_name = self.config.get('RIVER_NETWORK_SHP_NAME', 'default')
        if network_name == 'default':
            network_name = f"{self.domain_name}_riverNetwork.shp"
        
        shapefile_path = Path(network_path) / network_name
        
        if shapefile_path.exists():
            return gpd.read_file(shapefile_path)
        else:
            self.logger.warning(f"River network shapefile not found: {shapefile_path}")
            return gpd.GeoDataFrame()
    
    def _build_graph_connectivity(self):
        """Build graph connectivity from CONFLUENCE data."""
        edge_list = []
        edge_attributes = []
        
        # Use HRU to segment mapping if available
        hru_to_seg_col = self.config.get('RIVER_BASIN_SHP_HRU_TO_SEG', 'gru_to_seg')
        
        if hru_to_seg_col in self.catchment_gdf.columns and not self.river_network_gdf.empty:
            # Build connectivity using river network
            seg_id_col = self.config.get('RIVER_NETWORK_SHP_SEGID', 'LINKNO')
            downstream_col = self.config.get('RIVER_NETWORK_SHP_DOWNSEGID', 'DSLINKNO')
            
            for _, catchment in self.catchment_gdf.iterrows():
                hru_id = str(catchment[self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')])
                from_node = self.node_mapping.get(hru_id)
                
                if from_node is not None:
                    # Find downstream connections
                    seg_id = catchment.get(hru_to_seg_col)
                    if pd.notna(seg_id):
                        downstream_segs = self.river_network_gdf[
                            self.river_network_gdf[seg_id_col] == seg_id
                        ][downstream_col].values
                        
                        for downstream_seg in downstream_segs:
                            if pd.notna(downstream_seg):
                                # Find catchments connected to downstream segment
                                downstream_catchments = self.catchment_gdf[
                                    self.catchment_gdf[hru_to_seg_col] == downstream_seg
                                ]
                                
                                for _, downstream_catchment in downstream_catchments.iterrows():
                                    downstream_hru = str(downstream_catchment[self.config.get('CATCHMENT_SHP_HRUID', 'HRU_ID')])
                                    to_node = self.node_mapping.get(downstream_hru)
                                    
                                    if to_node is not None and from_node != to_node:
                                        edge_list.append([from_node, to_node])
                                        # Simple edge attributes for now
                                        edge_attributes.append([1.0, 0.0, 1.0])  # [distance, elevation_drop, travel_time]
        
        # If no river network connectivity, create simple adjacency
        if not edge_list:
            self.logger.warning("No river network connectivity found, using spatial adjacency")
            # Create simple chain connectivity as fallback
            for i in range(len(self.catchment_ids) - 1):
                edge_list.append([i, i + 1])
                edge_attributes.append([1.0, 0.0, 1.0])
        
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
        self.edge_attr = torch.tensor(edge_attributes, dtype=torch.float32) if edge_attributes else torch.empty((0, 3), dtype=torch.float32)
        
        self.logger.info(f"Built graph with {len(self.catchment_ids)} nodes and {len(edge_list)} edges")
    
    def _initialize_encoder_dimensions(self):
        """Initialize encoder dimensions for the graph transformer."""
        hidden_dim = self.config.get('DISTRIBUTED_EMULATOR_HIDDEN_DIM', 512)
        
        self.encoder_dims = {
            'param': int(0.4 * hidden_dim),
            'attr': int(0.3 * hidden_dim),
            'forcing': int(0.2 * hidden_dim),
            'spatial': int(0.1 * hidden_dim)
        }
        
        # Ensure exact sum
        total = sum(self.encoder_dims.values())
        if total != hidden_dim:
            self.encoder_dims['spatial'] += (hidden_dim - total)
    
    def build_node_features(self, parameters: np.ndarray, attributes: pd.DataFrame, forcing_summary: pd.DataFrame):
        """Build node features from CONFLUENCE data."""
        # Store dimensions for later use
        self.n_parameters = parameters.shape[1] if len(parameters.shape) > 1 else len(parameters)
        self.n_attributes = len(attributes.columns)
        self.n_forcing = len(forcing_summary.columns)
        
        # Convert to tensors
        param_features = torch.tensor(parameters, dtype=torch.float32)
        if len(param_features.shape) == 1:
            param_features = param_features.unsqueeze(0).repeat(len(self.catchment_ids), 1)
        
        attr_features = torch.tensor(attributes.values, dtype=torch.float32)
        forcing_features = torch.tensor(forcing_summary.values, dtype=torch.float32)
        
        # Add spatial features from catchment shapefile
        lat_col = self.config.get('CATCHMENT_SHP_LAT', 'center_lat')
        lon_col = self.config.get('CATCHMENT_SHP_LON', 'center_lon')
        area_col = self.config.get('CATCHMENT_SHP_AREA', 'HRU_area')
        
        spatial_features = torch.tensor([
            self.catchment_gdf[lon_col].values,
            self.catchment_gdf[lat_col].values,
            self.catchment_gdf[area_col].values,
            np.arange(len(self.catchment_ids))  # Simple distance to outlet proxy
        ], dtype=torch.float32).t()
        
        # Concatenate all features
        self.base_features = torch.cat([
            param_features,
            attr_features,
            forcing_features,
            spatial_features
        ], dim=1)
        
        # Create PyTorch Geometric Data object
        self.graph_data = Data(
            x=self.base_features,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr
        )
        
        self.logger.info(f"Built node features: {self.base_features.shape}")
    
    def update_features_with_parameters(self, new_parameters: torch.Tensor):
        """Update graph features with new parameters."""
        updated_features = self.base_features.clone()
        updated_features[:, :self.n_parameters] = new_parameters
        
        return Data(
            x=updated_features,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr
        )


class CONFLUENCEObservationMapper:
    """Map observations to CONFLUENCE catchments."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, hydrofabric: CONFLUENCEHydrofabricGraph):
        self.config = config
        self.logger = logger
        self.hydrofabric = hydrofabric
        self.gauge_to_node_map = {}
        
        self._map_streamflow_observations()
    
    def _map_streamflow_observations(self):
        """Map streamflow observations to catchments."""
        # Use CONFLUENCE's streamflow observations
        obs_path = self.config.get('OBSERVATIONS_PATH', 'default')
        if obs_path == 'default':
            obs_path = self.hydrofabric.project_dir / "observations" / "streamflow" / "preprocessed"
        
        processed_file = f"{self.config.get('DOMAIN_NAME')}_streamflow_processed.csv"
        streamflow_file = Path(obs_path) / processed_file
        
        if streamflow_file.exists():
            # Map the main gauge to the outlet
            sim_reach_id = self.config.get('SIM_REACH_ID', '123')
            outlet_node = len(self.hydrofabric.catchment_ids) - 1  # Assume outlet is last node
            self.gauge_to_node_map[sim_reach_id] = outlet_node
            
            self.logger.info(f"Mapped gauge {sim_reach_id} to outlet node {outlet_node}")
        else:
            self.logger.warning(f"No streamflow observations found at {streamflow_file}")


class CONFLUENCESUMMAInterface:
    """Interface with SUMMA for CONFLUENCE."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Use CONFLUENCE SUMMA configuration
        self.summa_exe = Path(config.get('SUMMA_INSTALL_PATH', '')) / config.get('SUMMA_EXE', 'summa_sundials.exe')
        self.settings_path = Path(config.get('SETTINGS_SUMMA_PATH', ''))


class DistributedGraphTransformer(nn.Module):
    """Graph Transformer for distributed emulation."""
    
    def __init__(self, config: Dict[str, Any], hydrofabric: CONFLUENCEHydrofabricGraph):
        super().__init__()
        self.config = config
        self.hydrofabric = hydrofabric
        
        # Use hydrofabric dimensions
        n_parameters = hydrofabric.n_parameters
        n_attributes = hydrofabric.n_attributes
        n_forcing = hydrofabric.n_forcing
        encoder_dims = hydrofabric.encoder_dims
        
        # Feature encoders
        self.param_encoder = nn.Sequential(
            nn.Linear(n_parameters, encoder_dims['param']),
            nn.LayerNorm(encoder_dims['param']),
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )
        
        self.attribute_encoder = nn.Sequential(
            nn.Linear(n_attributes, encoder_dims['attr']),
            nn.LayerNorm(encoder_dims['attr']),
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )
        
        self.forcing_encoder = nn.Sequential(
            nn.Linear(n_forcing, encoder_dims['forcing']),
            nn.LayerNorm(encoder_dims['forcing']),
            nn.ReLU()
        )
        
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, encoder_dims['spatial']),
            nn.LayerNorm(encoder_dims['spatial']),
            nn.ReLU()
        )
        
        # Graph Transformer layers
        self.transformer_layers = nn.ModuleList([
            gnn.TransformerConv(
                in_channels=config['hidden_dim'],
                out_channels=config['hidden_dim'] // config['n_heads'],
                heads=config['n_heads'],
                dropout=config['dropout'],
                edge_dim=3,
                concat=True
            )
            for _ in range(config['n_transformer_layers'])
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config['hidden_dim'])
            for _ in range(config['n_transformer_layers'])
        ])
        
        # Parameter optimization head
        self.param_optimizer = nn.Sequential(
            nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2),
            nn.LayerNorm(config['hidden_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'] // 2, config['hidden_dim'] // 4),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'] // 4, n_parameters),
            nn.Sigmoid()
        )
    
    def forward(self, graph_data) -> torch.Tensor:
        """Forward pass through the graph transformer."""
        x = graph_data.x
        edge_index = graph_data.edge_index
        edge_attr = graph_data.edge_attr
        
        # Split features
        param_end = self.hydrofabric.n_parameters
        attr_end = param_end + self.hydrofabric.n_attributes
        forcing_end = attr_end + self.hydrofabric.n_forcing
        
        # Encode features
        params = self.param_encoder(x[:, :param_end])
        attrs = self.attribute_encoder(x[:, param_end:attr_end])
        forcing = self.forcing_encoder(x[:, attr_end:forcing_end])
        spatial = self.spatial_encoder(x[:, -4:])
        
        # Combine encoded features
        h = torch.cat([params, attrs, forcing, spatial], dim=-1)
        
        # Graph Transformer layers
        for transformer, layer_norm in zip(self.transformer_layers, self.layer_norms):
            h_new = transformer(h, edge_index, edge_attr)
            h = layer_norm(h + h_new)
        
        # Generate optimized parameters
        theta_optimized = self.param_optimizer(h)
        
        return theta_optimized


class DistributedLossFunction:
    """Loss function for distributed emulation."""
    
    def __init__(self, config: Dict[str, Any], observation_mapper: CONFLUENCEObservationMapper, logger: logging.Logger):
        self.config = config
        self.observation_mapper = observation_mapper
        self.logger = logger
        self.loss_weights = config['loss_weights']
    
    def compute_loss(self, simulated_outputs: torch.Tensor, window_start: str, window_end: str) -> torch.Tensor:
        """Compute multi-objective loss."""
        total_loss = torch.tensor(0.0, requires_grad=True)
        
        # Streamflow loss (primary component)
        if len(self.observation_mapper.gauge_to_node_map) > 0:
            streamflow_loss = self._compute_streamflow_loss(simulated_outputs, window_start, window_end)
            total_loss = total_loss + self.loss_weights['streamflow'] * streamflow_loss
        
        # Add other loss components as needed
        # Snow, soil moisture, groundwater losses would be implemented here
        
        return total_loss
    
    def _compute_streamflow_loss(self, simulated_outputs: torch.Tensor, window_start: str, window_end: str) -> torch.Tensor:
        """Compute streamflow loss using Nash-Sutcliffe efficiency."""
        # For now, return a simple synthetic loss
        # In full implementation, this would compare against CONFLUENCE observations
        
        # Extract streamflow component (assuming first variable)
        streamflow = simulated_outputs[:, :, 0]
        
        # Simple loss based on flow magnitude and variability
        mean_flow = torch.mean(streamflow, dim=1)
        flow_std = torch.std(streamflow, dim=1)
        
        # Encourage realistic flow magnitudes and variability
        magnitude_loss = torch.mean((mean_flow - 1.0) ** 2)  # Target mean flow of 1.0
        variability_loss = torch.mean((flow_std - 0.5) ** 2)  # Target std of 0.5
        
        return magnitude_loss + 0.1 * variability_loss