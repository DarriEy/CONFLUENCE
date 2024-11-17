import os
import sys
import glob
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd # type: ignore
import geopandas as gpd # type: ignore
from datetime import datetime
import torch # type: ignore
import torch.nn as nn # type: ignore
import numpy as np # type: ignore
import torch.optim as optim # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
import xarray as xr # type: ignore
import psutil # type: ignore
import torch.optim as optim # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.dates as mdates # type: ignore
from matplotlib.gridspec import GridSpec # type: ignore
from torch.optim.lr_scheduler import OneCycleLR # type: ignore
from torch.utils.data import TensorDataset, DataLoader # type: ignore
import yaml # type: ignore

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.evaluation_util.calculate_sim_stats import get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE, get_KGEnp # type: ignore

class FLUXPreProcessor:
    """
    Preprocessor for the FLUX (Flow Learning Using Complex Networks) model.
    Handles data preparation and preprocessing for FLUX model runs.
    
    Attributes:
        config (Dict[str, Any]): Configuration settings for FLUX
        logger (Any): Logger object for recording processing information
        project_dir (Path): Directory for the current project
        domain_name (str): Name of the domain being processed
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.flux_setup_dir = self.project_dir / "settings" / "FLUX"

    def run_preprocessing(self):
        """Run the complete FLUX preprocessing workflow."""
        self.logger.info("Starting FLUX preprocessing")
        try:
            self.create_directories()
            forcing_df, streamflow_df = self.prepare_data()
            self.save_preprocessed_data(forcing_df, streamflow_df)
            self.logger.info("FLUX preprocessing completed successfully")
        except Exception as e:
            self.logger.error(f"Error during FLUX preprocessing: {str(e)}")
            raise

    def create_directories(self):
        """Create necessary directories for FLUX setup."""
        dirs = [
            self.project_dir / 'models' / 'FLUX',
            self.project_dir / 'settings' / 'FLUX',
            self.project_dir / 'simulations' / self.config.get('EXPERIMENT_ID') / 'FLUX'
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare forcing and streamflow data for FLUX model.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Processed forcing and streamflow data
        """
        self.logger.info("Preparing data for FLUX model")

        # Load forcing data
        forcing_path = self.project_dir / 'forcing' / 'basin_averaged_data'
        forcing_files = glob.glob(str(forcing_path / '*.nc'))
        
        if not forcing_files:
            raise FileNotFoundError(f"No forcing files found in {forcing_path}")
        
        forcing_files.sort()
        datasets = [xr.open_dataset(file) for file in forcing_files]
        combined_ds = xr.concat(datasets, dim='time')
        forcing_df = combined_ds.to_dataframe().reset_index()
        
        required_vars = ['hruId', 'time', 'pptrate', 'SWRadAtm', 'LWRadAtm', 'airpres', 'airtemp', 'spechum', 'windspd']
        missing_vars = [var for var in required_vars if var not in forcing_df.columns]
        if missing_vars:
            raise ValueError(f"Missing required variables in forcing data: {missing_vars}")

        forcing_df['time'] = pd.to_datetime(forcing_df['time'])
        forcing_df = forcing_df.set_index(['time', 'hruId']).sort_index()

        # Load streamflow data
        streamflow_path = self.project_dir / 'observations' / 'streamflow' / 'preprocessed' / f"{self.config.get('DOMAIN_NAME')}_streamflow_processed.csv"
        streamflow_df = pd.read_csv(streamflow_path, parse_dates=['datetime'])
        
        # Keep the original column name 'discharge_cms'
        streamflow_df = streamflow_df.set_index('datetime')
        
        # Align timestamps
        start_date = max(forcing_df.index.get_level_values('time').min(),
                        streamflow_df.index.min())
        end_date = min(forcing_df.index.get_level_values('time').max(),
                    streamflow_df.index.max())

        forcing_df = forcing_df.loc[pd.IndexSlice[start_date:end_date, :], :]
        streamflow_df = streamflow_df.loc[start_date:end_date]

        self.logger.info(f"Prepared forcing data with shape: {forcing_df.shape}")
        self.logger.info(f"Prepared streamflow data with shape: {streamflow_df.shape}")

        return forcing_df, streamflow_df

    def save_preprocessed_data(self, forcing_df: pd.DataFrame, streamflow_df: pd.DataFrame):
        """Save preprocessed data for FLUX model."""
        prep_dir = self.project_dir / 'forcing' / 'FLUX_input'
        prep_dir.mkdir(parents=True, exist_ok=True)

        # Save forcing data
        forcing_file = prep_dir / f"{self.domain_name}_forcing_processed.nc"
        forcing_ds = forcing_df.to_xarray()
        forcing_ds.to_netcdf(forcing_file)

        # Save streamflow data
        streamflow_file = prep_dir / f"{self.domain_name}_streamflow_processed.csv"
        streamflow_df.to_csv(streamflow_file)

        self.logger.info(f"Saved preprocessed forcing data to: {forcing_file}")
        self.logger.info(f"Saved preprocessed streamflow data to: {streamflow_file}")

class FLUXRunner:
    """
    Runner class for the FLUX (Flow Learning Using Complex Networks) model.
    Handles model execution, training, prediction, and result management.
    
    Attributes:
        config (Dict[str, Any]): Configuration settings for FLUX
        logger (Any): Logger object for recording run information
        device (torch.device): Device to run the model on (CPU or CUDA)
        project_dir (Path): Directory for the current project
        domain_name (str): Name of the domain being processed
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = StandardScaler()
        self.lookback = config.get('FLASH_LOOKBACK', 30)
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
    def run_flux(self) -> Optional[Path]:
        """
        Run the complete FLUX workflow including training and prediction.
        
        Returns:
            Optional[Path]: Path to the output directory if successful, None otherwise
        """
        self.logger.info("Starting FLUX model run")

        try:
            # Load preprocessed data
            forcing_df, streamflow_df = self._load_preprocessed_data()
            
            # Define path to save model
            model_save_path = self.project_dir / 'models' / 'FLUX' / 'flux_model.pt'

            # Preprocess data
            X, y, common_dates, hru_ids, features_avg = self._preprocess_data(forcing_df, streamflow_df)
            
            input_size = X.shape[2]
            hidden_size = self.config.get('FLASH_HIDDEN_SIZE', 64)
            num_layers = self.config.get('FLASH_NUM_LAYERS', 2)

            if self.config.get('FLASH_LOAD', False):
                # Load pre-trained model
                self.logger.info("Loading pre-trained FLUX model")
                model_state = self._load_model(model_save_path)
                self._create_model(input_size, hidden_size, num_layers)
                self.model.load_state_dict(model_state)
            else:
                # Train new model
                self._create_model(input_size, hidden_size, num_layers)
                self._train_model(X, y,
                                epochs=self.config.get('FLASH_EPOCHS', 100),
                                batch_size=self.config.get('FLASH_BATCH_SIZE', 32),
                                learning_rate=self.config.get('FLASH_LEARNING_RATE', 0.001))
                self._save_model(model_save_path)

            # Run simulation
            results = self._simulate(forcing_df, streamflow_df)
            
            # Save and visualize results
            output_path = self._save_results(results)
            self._visualize_results(results, streamflow_df)
            
            self.logger.info("FLUX model run completed successfully")
            return output_path

        except Exception as e:
            self.logger.error(f"Error during FLUX model run: {str(e)}")
            raise

    def _create_model(self, input_size: int, hidden_size: int, num_layers: int):
        """Create the LSTM model."""
        self.logger.info(f"Creating FLUX model architecture")
        
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.2):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
                self.dropout = nn.Dropout(dropout_rate)
                self.ln = nn.LayerNorm(hidden_size)
                self.fc = nn.Linear(hidden_size, 1)
                
                self.init_weights()

            def init_weights(self):
                for name, param in self.lstm.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)
                nn.init.xavier_uniform_(self.fc.weight)
                nn.init.constant_(self.fc.bias, 0.0)

            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                out, _ = self.lstm(x, (h0, c0))
                out = self.ln(out[:, -1, :])
                out = self.dropout(out)
                out = self.fc(out)
                return out

        dropout_rate = float(self.config.get('FLASH_DROPOUT', 0.2))
        self.model = LSTMModel(input_size, hidden_size, num_layers, dropout_rate).to(self.device)
        self.logger.info(f"FLUX model created: {self.model}")

    def _train_model(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 100, 
                    batch_size: int = 32, learning_rate: float = 0.001):
        """Train the FLUX model."""
        self.logger.info(f"Training FLUX model")

        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        criterion = nn.SmoothL1Loss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, 
                              weight_decay=float(self.config.get('FLASH_L2_REGULARIZATION', 1e-6)))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=10, verbose=True)

        best_val_loss = float('inf')
        patience = self.config.get('FLASH_LEARNING_PATIENCE')
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for i in range(0, X_train.size(0), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                if torch.isnan(loss):
                    self.logger.warning(f"NaN loss encountered in epoch {epoch}")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

            if (epoch + 1) % 10 == 0:
                self.logger.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}')

    def _save_model(self, path: Path):
        """Save the trained model."""
        self.logger.info(f"Saving FLUX model to {path}")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'lookback': self.lookback,
        }, path)

    def _load_model(self, path: Path):
        """Load a trained model."""
        self.logger.info(f"Loading FLUX model from {path}")
        return torch.load(path, map_location=self.device).get('model_state_dict')
    
    def _save_results(self, results: pd.DataFrame) -> Path:
        """Save model results."""
        output_dir = self.project_dir / 'simulations' / self.config.get('EXPERIMENT_ID') / 'FLUX'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to xarray Dataset
        ds = xr.Dataset(
            {
                "streamflow": (["time"], results['predicted_streamflow'].values)
            },
            coords={
                "time": results.index
            }
        )
        
        # Add attributes
        ds.streamflow.attrs['units'] = 'm3 s-1'
        ds.streamflow.attrs['long_name'] = 'Simulated streamflow'
        
        # Save as NetCDF
        output_file = output_dir / f"{self.config.get('EXPERIMENT_ID')}_FLUX_output.nc"
        ds.to_netcdf(output_file)
        
        self.logger.info(f"FLUX results saved to {output_file}")
        return output_file

    def _visualize_results(self, results: pd.DataFrame, obs_df: pd.DataFrame):
        """Create visualization of model results."""
        self.logger.info("Visualizing FLUX model results")
        
        # Prepare data
        sim_dates = results.index
        sim_flow = results['predicted_streamflow']
        obs_flow = obs_df.reindex(sim_dates)['streamflow']
        
        # Calculate metrics
        metrics = {
            '': self._calculate_metrics(obs_flow, sim_flow)
        }

        # Create figure
        fig = plt.figure(figsize=(15, 8))
        
        # Plot streamflow
        plt.plot(sim_dates, sim_flow, label='FLUX simulated', color='blue')
        plt.plot(sim_dates, obs_flow, label='Observed', color='red')
        plt.ylabel('Streamflow (m³/s)')
        plt.title('Observed vs Simulated Streamflow')
        plt.legend()
        plt.grid(True)

        # Add metrics text to upper left corner
        metrics_text = "Performance Metrics:\n"
        for period, metric_values in metrics.items():
            metrics_text += f"{period}:\n"
            metrics_text += "\n".join([f"  {k}: {v:.3f}" for k, v in metric_values.items()]) + "\n"
        
        plt.text(0.01, 0.99, metrics_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontsize=8, 
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))

        # Format x-axis
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())
        plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        # Save the figure
        output_dir = self.project_dir / 'plots' / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_path = output_dir / f'{self.config.get("EXPERIMENT_ID")}_FLUX_simulation_results.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Results visualization saved to {fig_path}")

    def _calculate_metrics(self, obs: pd.Series, sim: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics."""
        # Align and clean data
        aligned_data = pd.concat([obs, sim], axis=1, keys=['obs', 'sim']).dropna()
        obs = aligned_data['obs']
        sim = aligned_data['sim']
        
        return {
            'RMSE': get_RMSE(obs.values, sim.values, transfo=1),
            'KGE': get_KGE(obs.values, sim.values, transfo=1),
            'KGEp': get_KGEp(obs.values, sim.values, transfo=1),
            'NSE': get_NSE(obs.values, sim.values, transfo=1),
            'MAE': get_MAE(obs.values, sim.values, transfo=1)
        }

    def _load_preprocessed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load preprocessed forcing and streamflow data."""
        prep_dir = self.project_dir / 'forcing' / 'FLUX_input'
        
        # Load forcing data
        forcing_file = prep_dir / f"{self.domain_name}_forcing_processed.nc"
        forcing_df = xr.open_dataset(forcing_file).to_dataframe()
        
        # Load streamflow data
        streamflow_file = prep_dir / f"{self.domain_name}_streamflow_processed.csv"
        streamflow_df = pd.read_csv(streamflow_file, index_col=0, parse_dates=True)
        
        return forcing_df, streamflow_df

    def _preprocess_data(self, forcing_df: pd.DataFrame, streamflow_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, pd.DatetimeIndex, pd.Index, pd.DataFrame]:
        self.logger.info("Preprocessing data for FLUX model")
        
        # Align the data
        common_dates = forcing_df.index.get_level_values('time').intersection(streamflow_df.index)
        forcing_df = forcing_df.loc[pd.IndexSlice[common_dates, :], :]
        streamflow_df = streamflow_df.loc[common_dates]

        # Prepare features
        features = forcing_df.reset_index()
        feature_columns = features.columns.drop(['time', 'hruId', 'hru', 'latitude', 'longitude'] 
                                            if 'time' in features.columns and 'hruId' in features.columns else [])
        
        # Average features across all HRUs
        features_avg = forcing_df.groupby('time')[feature_columns].mean()

        # Scale features
        self.feature_scaler = StandardScaler()
        scaled_features = self.feature_scaler.fit_transform(features_avg)
        scaled_features = np.clip(scaled_features, -10, 10)

        # Prepare targets - using log transform for streamflow
        targets = streamflow_df['discharge_cms'].values.reshape(-1, 1)
        targets = np.log1p(targets)  # Apply log1p transform to handle zeros and small values
        
        # Scale log-transformed targets
        self.target_scaler = StandardScaler()
        scaled_targets = self.target_scaler.fit_transform(targets)
        scaled_targets = np.clip(scaled_targets, -10, 10)

        # Create sequences
        X, y = [], []
        for i in range(len(scaled_features) - self.lookback):
            X.append(scaled_features[i:(i + self.lookback)])
            y.append(scaled_targets[i + self.lookback])

        # Convert to PyTorch tensors
        X = torch.FloatTensor(np.array(X)).to(self.device)
        y = torch.FloatTensor(np.array(y)).to(self.device)

        self.logger.info(f"Preprocessed data shape: X: {X.shape}, y: {y.shape}")
        return X, y, pd.DatetimeIndex(common_dates), pd.Index(['average']), features_avg

    def _simulate(self, forcing_df: pd.DataFrame, streamflow_df: pd.DataFrame) -> pd.DataFrame:
        """Run simulation with trained model."""
        self.logger.info("Running simulation with FLUX model")
        X, _, common_dates, _, features_avg = self._preprocess_data(forcing_df, streamflow_df)
        
        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
        
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch_predictions = self.model(batch[0])
                predictions.append(batch_predictions.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        predictions = self.target_scaler.inverse_transform(predictions)
        predictions = np.expm1(predictions)  # Inverse of log1p transform
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=1e15, neginf=0.0)
        
        pred_df = pd.DataFrame(predictions, columns=['predicted_streamflow'], 
                             index=common_dates[self.lookback:])
        
        result = features_avg.join(pred_df, how='outer')
        
        self.logger.info(f"Simulation completed with shape: {result.shape}")
        return result
    

class FLUXPostProcessor:
    """
    Postprocessor for FLUX (Flow Learning Using Complex Networks) model outputs.
    Handles extraction, processing, and saving of simulation results.
    
    Attributes:
        config (Dict[str, Any]): Configuration settings for FLUX
        logger (Any): Logger object for recording processing information
        project_dir (Path): Directory for the current project
        domain_name (str): Name of the domain being processed
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.results_dir = self.project_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract simulated streamflow from FLUX output and save to CSV.
        
        Returns:
            Optional[Path]: Path to the saved CSV file if successful, None otherwise
        """
        try:
            self.logger.info("Extracting FLUX streamflow results")
            
            # Define paths
            sim_path = self.project_dir / 'simulations' / self.config['EXPERIMENT_ID'] / 'FLUX' / f"{self.config['EXPERIMENT_ID']}_FLUX_output.nc"
            
            # Read simulation results
            ds = xr.open_dataset(sim_path)
            
            # Extract streamflow
            q_sim = ds['streamflow'].to_pandas()
            
            # Get catchment area from river basins shapefile
            basin_name = self.config.get('RIVER_BASINS_NAME')
            if basin_name == 'default':
                basin_name = f"{self.domain_name}_riverBasins_delineate.shp"
            basin_path = self._get_file_path('RIVER_BASINS_PATH', 'shapefiles/river_basins', basin_name)
            basin_gdf = gpd.read_file(basin_path)
            
            # Calculate total area in km2
            area_km2 = basin_gdf['GRU_area'].sum() / 1e6
            self.logger.info(f"Total catchment area: {area_km2:.2f} km2")
            
            # Create DataFrame with FLUX-prefixed column name
            results_df = pd.DataFrame({
                'FLUX_discharge_cms': q_sim
            }, index=q_sim.index)
            
            # Add metadata as attributes
            results_df.attrs = {
                'model': 'FLUX',
                'domain': self.domain_name,
                'experiment_id': self.config['EXPERIMENT_ID'],
                'catchment_area_km2': area_km2,
                'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'units': 'm3/s'
            }
            
            # Save to CSV
            output_file = self.results_dir / f"{self.config['EXPERIMENT_ID']}_FLUX_results.csv"
            results_df.to_csv(output_file)
            
            self.logger.info(f"Results saved to: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error extracting streamflow: {str(e)}")
            raise

    def extract_model_parameters(self) -> Optional[Path]:
        """
        Extract and save FLUX model parameters and hyperparameters.
        
        Returns:
            Optional[Path]: Path to the saved parameter file if successful, None otherwise
        """
        try:
            self.logger.info("Extracting FLUX model parameters")
            
            # Load model state
            model_path = self.project_dir / 'models' / 'FLUX' / 'flux_model.pt'
            model_state = torch.load(model_path, map_location='cpu')
            
            # Extract hyperparameters from config
            hyperparameters = {
                'hidden_size': self.config.get('FLASH_HIDDEN_SIZE', 64),
                'num_layers': self.config.get('FLASH_NUM_LAYERS', 2),
                'dropout_rate': self.config.get('FLASH_DROPOUT', 0.2),
                'lookback': self.config.get('FLASH_LOOKBACK', 30),
                'batch_size': self.config.get('FLASH_BATCH_SIZE', 32),
                'learning_rate': self.config.get('FLASH_LEARNING_RATE', 0.001),
                'l2_regularization': self.config.get('FLASH_L2_REGULARIZATION', 1e-6)
            }
            
            # Create dictionary with all parameters
            param_dict = {
                'hyperparameters': hyperparameters,
                'model_state': {k: v.numpy().tolist() for k, v in model_state['model_state_dict'].items()},
                'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'experiment_id': self.config['EXPERIMENT_ID']
            }
            
            # Save parameters
            output_file = self.results_dir / f"{self.config['EXPERIMENT_ID']}_FLUX_parameters.yaml"
            with open(output_file, 'w') as f:
                yaml.dump(param_dict, f, default_flow_style=False)
            
            self.logger.info(f"Model parameters saved to: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error extracting model parameters: {str(e)}")
            raise

    def create_performance_report(self) -> Optional[Path]:
        """
        Create a comprehensive performance report including metrics and visualizations.
        
        Returns:
            Optional[Path]: Path to the saved report if successful, None otherwise
        """
        try:
            self.logger.info("Creating FLUX performance report")
            
            # Load simulated and observed data
            sim_path = self.project_dir / 'simulations' / self.config['EXPERIMENT_ID'] / 'FLUX' / f"{self.config['EXPERIMENT_ID']}_FLUX_output.nc"
            obs_path = self.project_dir / 'observations' / 'streamflow' / 'preprocessed' / f"{self.domain_name}_streamflow_processed.csv"
            
            sim_ds = xr.open_dataset(sim_path)
            obs_df = pd.read_csv(obs_path, parse_dates=['datetime']).set_index('datetime')
            
            # Align data
            sim_flow = sim_ds['streamflow'].to_pandas()
            obs_flow = obs_df['discharge_cms'].reindex(sim_flow.index)
            
            # Calculate performance metrics
            metrics = {
                'RMSE': get_RMSE(obs_flow.values, sim_flow.values, transfo=1),
                'KGE': get_KGE(obs_flow.values, sim_flow.values, transfo=1),
                'KGEp': get_KGEp(obs_flow.values, sim_flow.values, transfo=1),
                'NSE': get_NSE(obs_flow.values, sim_flow.values, transfo=1),
                'MAE': get_MAE(obs_flow.values, sim_flow.values, transfo=1)
            }
            
            # Create report
            report_dict = {
                'model': 'FLUX',
                'experiment_id': self.config['EXPERIMENT_ID'],
                'domain': self.domain_name,
                'simulation_period': {
                    'start': sim_flow.index[0].strftime('%Y-%m-%d'),
                    'end': sim_flow.index[-1].strftime('%Y-%m-%d')
                },
                'performance_metrics': metrics,
                'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save report
            output_file = self.results_dir / f"{self.config['EXPERIMENT_ID']}_FLUX_performance_report.yaml"
            with open(output_file, 'w') as f:
                yaml.dump(report_dict, f, default_flow_style=False)
            
            self.logger.info(f"Performance report saved to: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error creating performance report: {str(e)}")
            raise

    def run_postprocessing(self) -> Dict[str, Path]:
        """
        Run all postprocessing steps.
        
        Returns:
            Dict[str, Path]: Dictionary of output files generated
        """
        self.logger.info("Starting FLUX postprocessing")
        
        outputs = {}
        
        try:
            # Extract streamflow
            streamflow_file = self.extract_streamflow()
            if streamflow_file:
                outputs['streamflow'] = streamflow_file
            
            # Extract model parameters
            params_file = self.extract_model_parameters()
            if params_file:
                outputs['parameters'] = params_file
            
            # Create performance report
            report_file = self.create_performance_report()
            if report_file:
                outputs['report'] = report_file
            
            self.logger.info("FLUX postprocessing completed successfully")
            return outputs
            
        except Exception as e:
            self.logger.error(f"Error during postprocessing: {str(e)}")
            raise

    def _get_file_path(self, file_type: str, file_def_path: str, file_name: str) -> Path:
        """Helper method to get file paths from config or defaults."""
        if self.config.get(file_type) == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(self.config.get(file_type))