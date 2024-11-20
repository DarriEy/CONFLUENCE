import glob
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd # type: ignore
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
from torch.utils.data import TensorDataset, DataLoader # type: ignore
import geopandas as gpd # type: ignore
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.evaluation_util.calculate_sim_stats import get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE, get_KGEnp # type: ignore

class FLASH:
    """
    FLASH: Flow and Snow Hydrological LSTM

    A class implementing an LSTM-based model for hydrological predictions,
    specifically for streamflow and snow water equivalent (SWE).

    This class handles data preprocessing, model creation, training, prediction,
    and result saving for the FLASH model.

    Attributes:
        config (Dict[str, Any]): Configuration settings for the model.
        logger (Any): Logger object for recording model information.
        device (torch.device): Device to run the model on (CPU or CUDA).
        model (nn.Module): The LSTM model.
        scaler (StandardScaler): Scaler for normalizing input features.
        lookback (int): Number of time steps to look back for input sequences.
        project_dir (Path): Directory for the current project.
    """

    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = StandardScaler()
        self.lookback = config.get('FLASH_LOOKBACK', 30)
        self.project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"
        self.logger.info(f"Initialized FLASH model with device: {self.device}")


    def preprocess_data(self, forcing_df: pd.DataFrame, streamflow_df: pd.DataFrame, snow_df: Optional[pd.DataFrame] = None) -> Tuple[torch.Tensor, torch.Tensor, pd.DatetimeIndex, pd.Index, pd.DataFrame]:
        self.logger.info("Preprocessing data for FLASH model")
        
        # Align the data
        common_dates = forcing_df.index.get_level_values('time').intersection(streamflow_df.index)
        if snow_df is not None:
            common_dates = common_dates.intersection(snow_df.index)
        
        forcing_df = forcing_df.loc[pd.IndexSlice[common_dates, :], :]
        streamflow_df = streamflow_df.loc[common_dates]
        if snow_df is not None:
            snow_df = snow_df.loc[common_dates]

        # Prepare features (forcing data)
        features = forcing_df.reset_index()
        feature_columns = features.columns.drop(['time', 'hruId', 'hru', 'latitude', 'longitude'] if 'time' in features.columns and 'hruId' in features.columns else [])
        
        # Average features across all HRUs for each timestep
        features_avg = forcing_df.groupby('time')[feature_columns].mean()

        # Scale features
        self.feature_scaler = StandardScaler()
        scaled_features = self.feature_scaler.fit_transform(features_avg)
        scaled_features = np.clip(scaled_features, -10, 10)

        # Prepare targets (streamflow and optionally snow)
        if snow_df is not None:
            targets = pd.concat([streamflow_df['streamflow'], snow_df['snw']], axis=1)
            targets.columns = ['streamflow', 'SWE']
            self.output_size = 2
            self.target_names = ['streamflow', 'SWE']
        else:
            targets = pd.DataFrame(streamflow_df['streamflow'], columns=['streamflow'])
            self.output_size = 1
            self.target_names = ['streamflow']
        
        # Scale targets
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

    def create_model(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        """
        Create the LSTM model for FLASH.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden units in the LSTM layers.
            num_layers (int): Number of LSTM layers.
            output_size (int): Number of output variables (2 for streamflow and basin-averaged SWE).
        """
        self.logger.info(f"Creating FLASH model with input_size: {input_size}, hidden_size: {hidden_size}, num_layers: {num_layers}, output_size: {output_size}")
        
        
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
                self.dropout = nn.Dropout(dropout_rate)
                self.ln = nn.LayerNorm(hidden_size)
                self.fc = nn.Linear(hidden_size, output_size)
                
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
                out = self.dropout(out)  # Apply dropout here
                out = self.fc(out)
                return out
        
        dropout_rate = float(self.config.get('FLASH_DROPOUT', 0.2))
        self.model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_rate).to(self.device)
        self.logger.info(f"FLASH model created: {self.model}")

    def train_model(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001):
        self.logger.info(f"Training FLASH model with {epochs} epochs, batch_size: {batch_size}, learning_rate: {learning_rate}")

        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        criterion = nn.SmoothL1Loss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=float(self.config.get('FLASH_L2_REGULARIZATION', 1e-6)))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        best_val_loss = float('inf')
        patience = self.config.get('FLASH_LEARNING_PATIENCE')
        patience_counter = 0

        for epoch in range(epochs):
            self.logger.info(f'Training on epoch number: {epoch}')
            self.model.train()
            total_loss = 0

            for i in range(0, X_train.size(0), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                if torch.isnan(loss):
                    self.logger.warning(f"NaN loss encountered in epoch {epoch}, batch {i//batch_size}")
                    continue
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()

                total_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val)

            # Step the scheduler with the validation loss
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

        self.logger.info("FLASH model training completed")

    def predict(self, X: torch.Tensor, batch_size: int = 1000) -> np.ndarray:
        self.logger.info(f"Making predictions with FLASH model")
        self.logger.info(f"Input tensor shape: {X.shape}")
        self.log_memory_usage()

        predictions = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size):
                batch = X[i:i+batch_size]
                batch_predictions = self.model(batch)
                predictions.append(batch_predictions.cpu().numpy())
                
                if i % 10000 == 0:
                    self.logger.info(f"Processed {i}/{X.shape[0]} samples")
                    self.log_memory_usage()

        predictions = np.concatenate(predictions, axis=0)
        self.logger.info(f"Predictions shape: {predictions.shape}")
        self.log_memory_usage()
        return predictions

    def log_memory_usage(self):
        process = psutil.Process()
        memory_info = process.memory_info()
        self.logger.info(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")

    def simulate(self, forcing_df: pd.DataFrame, streamflow_df: pd.DataFrame, snow_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        self.logger.info("Running full simulation with FLASH model")
        X, _, common_dates, _, features_avg = self.preprocess_data(forcing_df, streamflow_df, snow_df)
        
        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
        
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch_predictions = self.model(batch[0])
                predictions.append(batch_predictions.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        
        # Inverse transform the predictions
        predictions = self.target_scaler.inverse_transform(predictions)
        
        # Handle NaN values
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=1e15, neginf=-1e15)
        
        # Create column names based on number of targets
        if self.output_size == 2:
            columns = ['predicted_streamflow', 'predicted_SWE']
        else:
            columns = ['predicted_streamflow']
        
        # Create a DataFrame for predictions
        pred_df = pd.DataFrame(predictions, columns=columns, index=common_dates[self.lookback:])
        
        # Join predictions with the original averaged features
        result = features_avg.join(pred_df, how='outer')
        
        self.logger.info(f"Shape of final result: {result.shape}")
        self.logger.info("Simulation completed")
        return result
    
    def save_model(self, path: Path):
        self.logger.info(f"Saving FLASH model to {path}")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'lookback': self.lookback,
            'output_size': self.output_size,
            'target_names': self.target_names
        }, path)
        self.logger.info("Model saved successfully")


    def load_model(self, path: Path):
        self.logger.info(f"Loading FLASH model from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.output_size = checkpoint['output_size']
        self.target_names = checkpoint['target_names']
        self.feature_scaler = checkpoint['feature_scaler']
        self.target_scaler = checkpoint['target_scaler']
        self.lookback = checkpoint['lookback']
        return checkpoint['model_state_dict']

    def run_flash(self):
        self.logger.info("Starting FLASH model run")

        try:
            # Load data
            forcing_df, streamflow_df, snow_df = self._load_data()
            
            # Define path to save model
            model_save_path = self.project_dir / 'models' / 'flash_model.pt'

            # Check if snow data should be used based on config
            use_snow = self.config.get('FLASH_USE_SNOW', True)
            snow_df_input = snow_df if use_snow else None

            # Preprocess data
            X, y, common_dates, hru_ids, features_avg = self.preprocess_data(
                forcing_df, streamflow_df, snow_df_input
            )
            
            input_size = X.shape[2]
            hidden_size = self.config.get('FLASH_HIDDEN_SIZE', 64)
            num_layers = self.config.get('FLASH_NUM_LAYERS', 2)

            if self.config.get('FLASH_LOAD', False):
                # Load pre-trained model
                self.logger.info("Loading pre-trained FLASH model")
                model_state = self.load_model(model_save_path)
                self.create_model(input_size, hidden_size, num_layers, self.output_size)
                self.model.load_state_dict(model_state)
            else:
                # Create and train model
                self.create_model(input_size, hidden_size, num_layers, self.output_size)
                self.train_model(X, y,
                                epochs=self.config.get('FLASH_EPOCHS', 100),
                                batch_size=self.config.get('FLASH_BATCH_SIZE', 32),
                                learning_rate=self.config.get('FLASH_LEARNING_RATE', 0.001))
                self.save_model(model_save_path)

            # Run simulation
            results = self.simulate(forcing_df, streamflow_df, snow_df_input)
            
            # Visualize results
            self.visualize_results(results, streamflow_df, snow_df_input)
            
            # Save results
            self._save_results(results)
            
            self.logger.info("FLASH model run completed successfully")
        except Exception as e:
            self.logger.error(f"Error during FLASH model run: {str(e)}")
            raise

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.logger.info("Loading data for FLASH model")
        
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
        streamflow_df = streamflow_df.set_index('datetime').rename(columns={'discharge_cms': 'streamflow'})
        streamflow_df.index = pd.to_datetime(streamflow_df.index)

        # Load snow data
        snow_path = self.project_dir / 'observations' / 'snow' / 'preprocessed'
        snow_files = glob.glob(str(snow_path / f"{self.config.get('DOMAIN_NAME')}_filtered_snow_observations.csv"))
        if not snow_files:
            raise FileNotFoundError(f"No snow observation files found in {snow_path}")
        
        snow_df = pd.concat([pd.read_csv(file, parse_dates=['datetime']) for file in snow_files])
        
        # Aggregate snow data across all stations
        snow_df = snow_df.groupby('datetime')['snw'].mean().reset_index()
        snow_df['datetime'] = pd.to_datetime(snow_df['datetime'])
        snow_df = snow_df.set_index('datetime')

        # Ensure all datasets cover the same time period
        start_date = max(forcing_df.index.get_level_values('time').min(),
                        streamflow_df.index.min(),
                        snow_df.index.min())
        end_date = min(forcing_df.index.get_level_values('time').max(),
                    streamflow_df.index.max(),
                    snow_df.index.max())

        forcing_df = forcing_df.loc[pd.IndexSlice[start_date:end_date, :], :]
        streamflow_df = streamflow_df.loc[start_date:end_date]
        snow_df = snow_df.loc[start_date:end_date]
        snow_df = snow_df.resample('h').interpolate(method = 'linear')

        self.logger.info(f"Loaded forcing data with shape: {forcing_df.shape}")
        self.logger.info(f"Loaded streamflow data with shape: {streamflow_df.shape}")
        self.logger.info(f"Loaded snow data with shape: {snow_df.shape}")
        
        return forcing_df, streamflow_df, snow_df

    def _save_results(self, results: pd.DataFrame):
        self.logger.info("Saving FLASH model results")
        
        # Prepare the output directory
        output_dir = self.project_dir / 'simulations' / self.config.get('EXPERIMENT_ID') / 'FLASH'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize dataset dictionary with streamflow
        data_vars = {
            "predicted_streamflow": (["time"], results['predicted_streamflow'].values)
        }
        
        # Add SWE if enabled in config
        if self.config.get('FLASH_USE_SNOW', True):
            data_vars["scalarSWE"] = (["time"], results['predicted_SWE'].values)
        
        # Create xarray Dataset
        ds = xr.Dataset(
            data_vars,
            coords={
                "time": results.index
            }
        )
        
        # Add attributes
        ds.predicted_streamflow.attrs['units'] = 'm3 s-1'
        ds.predicted_streamflow.attrs['long_name'] = 'Routed streamflow'
        
        if self.config.get('FLASH_USE_SNOW', True):
            ds.scalarSWE.attrs['units'] = 'mm'
            ds.scalarSWE.attrs['long_name'] = 'Snow Water Equivalent'
        
        # Save as NetCDF
        output_file = output_dir / f"{self.config.get('EXPERIMENT_ID')}_FLASH_output.nc"
        ds.to_netcdf(output_file)
        
        self.logger.info(f"FLASH results saved to {output_file}")

    def visualize_results(self, results_df: pd.DataFrame, obs_streamflow: pd.DataFrame, obs_snow: pd.DataFrame):
        self.logger.info("Visualizing results")

        # Prepare data
        sim_dates = results_df.index
        sim_streamflow = results_df['predicted_streamflow']
        obs_streamflow = obs_streamflow.reindex(sim_dates)
        
        # Calculate metrics for streamflow
        metrics_streamflow = {
            '': self.calculate_metrics(obs_streamflow, sim_streamflow)
        }

        # Determine figure size and layout based on whether snow is used
        use_snow = self.config.get('FLASH_USE_SNOW', True)
        if use_snow:
            fig = plt.figure(figsize=(15, 16))
            gs = GridSpec(2, 1, height_ratios=[1, 1])
        else:
            fig = plt.figure(figsize=(15, 8))
            gs = GridSpec(1, 1)

        # Streamflow subplot
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(sim_dates, sim_streamflow, label='FLASH simulated', color='blue')
        ax1.plot(sim_dates, obs_streamflow, label='Observed', color='red')
        ax1.set_ylabel('Streamflow (m³/s)')
        ax1.set_title('Observed vs Simulated Streamflow')
        ax1.legend()
        ax1.grid(True)

        # Add metrics text to upper left corner of streamflow plot
        metrics_text = "Performance Metrics:\n\n"
        metrics_text += "Streamflow Metrics:\n"
        for period, metrics in metrics_streamflow.items():
            metrics_text += f"{period}:\n"
            metrics_text += "\n".join([f"  {k}: {v:.3f}" for k, v in metrics.items()]) + "\n\n"
        
        if use_snow:
            # Add snow metrics and plot
            sim_swe = results_df['predicted_SWE']
            obs_snow = obs_snow.reindex(sim_dates)
            metrics_swe = {
                '': self.calculate_metrics(obs_snow, sim_swe)
            }
            metrics_text += "Snow Water Equivalent Metrics:\n"
            for period, metrics in metrics_swe.items():
                metrics_text += f"{period}:\n"
                metrics_text += "\n".join([f"  {k}: {v:.3f}" for k, v in metrics.items()]) + "\n\n"
            
            # Snow subplot
            ax2 = fig.add_subplot(gs[1])
            ax2.plot(sim_dates, sim_swe, label='FLASH simulated', color='blue')
            ax2.plot(sim_dates, obs_snow, label='Observed', color='red')
            ax2.set_ylabel('Snow Water Equivalent (mm)')
            ax2.set_title('Observed vs Simulated Snow Water Equivalent')
            ax2.legend()
            ax2.grid(True)
            
            # Format x-axis for both plots
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                ax.xaxis.set_minor_locator(mdates.MonthLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            # Format x-axis for single plot
            ax1.xaxis.set_major_locator(mdates.YearLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax1.xaxis.set_minor_locator(mdates.MonthLocator())
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        ax1.text(0.01, 0.99, metrics_text, transform=ax1.transAxes, verticalalignment='top', 
                fontsize=8, bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))

        plt.tight_layout()

        # Save the figure
        output_dir = self.project_dir / 'plots' / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_path = output_dir / f'{self.config.get("EXPERIMENT_ID")}_FLASH_simulation_results.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Results visualization saved to {fig_path}")

    def calculate_metrics(self, obs: pd.Series, sim: pd.Series) -> Dict[str, float]:
        # Ensure obs and sim have the same length and are aligned
        aligned_data = pd.concat([obs, sim], axis=1, keys=['obs', 'sim']).dropna()
        obs = aligned_data['obs']
        sim = aligned_data['sim']
        
        return {
            'RMSE': get_RMSE(obs.values, sim.values, transfo=1),
            'KGE': get_KGE(obs.values, sim.values, transfo=1),
            'KGEp': get_KGEp(obs.values, sim.values, transfo=1),
            'NSE': get_NSE(obs.values, sim.values, transfo=1),
            'MAE': get_MAE(obs.values, sim.values, transfo=1),
            'KGEnp': get_KGEnp(obs.values, sim.values, transfo=1)
        }
    
class FLASHPostProcessor:
    """
    Postprocessor for FLASH model outputs.
    Handles extraction, processing, and saving of simulation results.
    
    Attributes:
        config (Dict[str, Any]): Configuration settings for FLASH
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
        Extract simulated streamflow from FLASH output and save to CSV.
        
        Returns:
            Optional[Path]: Path to the saved CSV file if successful, None otherwise
        """
        try:
            self.logger.info("Extracting FLASH streamflow results")
            
            # Define paths
            sim_path = self.project_dir / 'simulations' / self.config['EXPERIMENT_ID'] / 'FLASH' / f'{self.config["EXPERIMENT_ID"]}_FLASH_output.nc'
            
            # Read simulation results
            ds = xr.open_dataset(sim_path)
            
            # Extract streamflow
            q_sim = ds['predicted_streamflow'].to_pandas()
            
            # Get catchment area from river basins shapefile
            basin_name = self.config.get('RIVER_BASINS_NAME')
            if basin_name == 'default':
                basin_name = f"{self.domain_name}_riverBasins_delineate.shp"
            basin_path = self._get_file_path('RIVER_BASINS_PATH', 'shapefiles/river_basins', basin_name)
            basin_gdf = gpd.read_file(basin_path)
            
            # Calculate total area in km2
            area_km2 = basin_gdf['GRU_area'].sum() / 1e6
            self.logger.info(f"Total catchment area: {area_km2:.2f} km2")
            
            # Create DataFrame with FLASH-prefixed column name
            results_df = pd.DataFrame({
                'FLASH_discharge_cms': q_sim
            }, index=q_sim.index)
            
            # Add metadata as attributes
            results_df.attrs = {
                'model': 'FLASH',
                'domain': self.domain_name,
                'experiment_id': self.config['EXPERIMENT_ID'],
                'catchment_area_km2': area_km2,
                'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'units': 'm3/s'
            }
            
            # Load existing results if file exists, otherwise create new DataFrame
            output_file = self.results_dir / f"{self.config['EXPERIMENT_ID']}_results.csv"
            if output_file.exists():
                existing_df = pd.read_csv(output_file, index_col=0, parse_dates=True)
                # Reindex the new results to match existing index
                results_df = results_df.reindex(existing_df.index)
                # Merge existing results with new results
                results_df = pd.concat([existing_df, results_df], axis=1)
            
            # Save to CSV
            results_df.to_csv(output_file)
            
            self.logger.info(f"Results saved to: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error extracting streamflow: {str(e)}")
            raise
            
    def _get_file_path(self, file_type: str, file_def_path: str, file_name: str) -> Path:
        """Helper method to get file paths from config or defaults."""
        if self.config.get(file_type) == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(self.config.get(file_type))