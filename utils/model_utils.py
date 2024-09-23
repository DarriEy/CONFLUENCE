import os
import glob
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
import pandas as pd # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import numpy as np # type: ignore
import torch.optim as optim # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
import xarray as xr # type: ignore
import psutil # type: ignore
import traceback

class MizuRouteRunner:
    """
    A class to run the mizuRoute model.

    This class handles the execution of the mizuRoute model, including setting up paths,
    running the model, and managing log files.

    Attributes:
        config (Dict[str, Any]): Configuration settings for the model run.
        logger (Any): Logger object for recording run information.
        root_path (Path): Root path for the project.
        domain_name (str): Name of the domain being processed.
        project_dir (Path): Directory for the current project.
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.root_path = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.root_path / f"domain_{self.domain_name}"

    def run_mizuroute(self):
        """
        Run the mizuRoute model.

        This method sets up the necessary paths, executes the mizuRoute model,
        and handles any errors that occur during the run.
        """
        self.logger.info("Starting mizuRoute run")

        # Set up paths and filenames
        mizu_path = self.config.get('INSTALL_PATH_MIZUROUTE')
        
        if mizu_path == 'default':
            mizu_path = self.root_path / 'installs/mizuRoute/route/bin/'
        else:
            mizu_path = Path(mizu_path)

        mizu_exe = self.config.get('EXE_NAME_MIZUROUTE')
        settings_path = self._get_config_path('SETTINGS_MIZU_PATH', 'settings/mizuRoute/')
        control_file = self.config.get('SETTINGS_MIZU_CONTROL_FILE')
        
        experiment_id = self.config.get('EXPERIMENT_ID')
        mizu_log_path = self._get_config_path('EXPERIMENT_LOG_MIZUROUTE', f"simulations/{experiment_id}/mizuRoute/mizuRoute_logs/")
        mizu_log_name = "mizuRoute_log.txt"
        
        mizu_out_path = self._get_config_path('EXPERIMENT_OUTPUT_MIZUROUTE', f"simulations/{experiment_id}/mizuRoute/")

        # Backup settings if required
        if self.config.get('EXPERIMENT_BACKUP_SETTINGS') == 'yes':
            backup_path = mizu_out_path / "run_settings"
            self._backup_settings(settings_path, backup_path)

        # Run mizuRoute
        os.makedirs(mizu_log_path, exist_ok=True)
        mizu_command = f"{mizu_path / mizu_exe} {settings_path / control_file}"
        
        try:
            with open(mizu_log_path / mizu_log_name, 'w') as log_file:
                subprocess.run(mizu_command, shell=True, check=True, stdout=log_file, stderr=subprocess.STDOUT)
            self.logger.info("mizuRoute run completed successfully")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"mizuRoute run failed with error: {e}")
            raise

    def _get_config_path(self, config_key: str, default_suffix: str) -> Path:
        path = self.config.get(config_key)
        if path == 'default':
            return self.project_dir / default_suffix
        return Path(path)

    def _backup_settings(self, source_path: Path, backup_path: Path):
        backup_path.mkdir(parents=True, exist_ok=True)
        os.system(f"cp -R {source_path}/. {backup_path}")
        self.logger.info(f"Settings backed up to {backup_path}")


class SummaRunner:
    """
    A class to run the SUMMA (Structure for Unifying Multiple Modeling Alternatives) model.

    This class handles the execution of the SUMMA model, including setting up paths,
    running the model, and managing log files.

    Attributes:
        config (Dict[str, Any]): Configuration settings for the model run.
        logger (Any): Logger object for recording run information.
        root_path (Path): Root path for the project.
        domain_name (str): Name of the domain being processed.
        project_dir (Path): Directory for the current project.
    """
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.root_path = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.root_path / f"domain_{self.domain_name}"

    def run_summa(self):
        """
        Run the SUMMA model.

        This method sets up the necessary paths, executes the SUMMA model,
        and handles any errors that occur during the run.
        """
        self.logger.info("Starting SUMMA run")

        # Set up paths and filenames
        summa_path = self.config.get('SUMMA_INSTALL_PATH')
        
        if summa_path == 'default':
            summa_path = self.root_path / 'installs/summa/bin/'
        else:
            summa_path = Path(summa_path)
            
        summa_exe = self.config.get('SUMMA_EXE')
        settings_path = self._get_config_path('SETTINGS_SUMMA_PATH', 'settings/SUMMA/')
        filemanager = self.config.get('SETTINGS_SUMMA_FILEMANAGER')
        
        experiment_id = self.config.get('EXPERIMENT_ID')
        summa_log_path = self._get_config_path('EXPERIMENT_LOG_SUMMA', f"simulations/{experiment_id}/SUMMA/SUMMA_logs/")
        summa_log_name = "summa_log.txt"
        
        summa_out_path = self._get_config_path('EXPERIMENT_OUTPUT_SUMMA', f"simulations/{experiment_id}/SUMMA/")

        # Backup settings if required
        if self.config.get('EXPERIMENT_BACKUP_SETTINGS') == 'yes':
            backup_path = summa_out_path / "run_settings"
            self._backup_settings(settings_path, backup_path)

        # Run SUMMA
        os.makedirs(summa_log_path, exist_ok=True)
        summa_command = f"{str(summa_path / summa_exe)} -m {str(settings_path / filemanager)}"
        
        try:
            with open(summa_log_path / summa_log_name, 'w') as log_file:
                subprocess.run(summa_command, shell=True, check=True, stdout=log_file, stderr=subprocess.STDOUT)
            self.logger.info("SUMMA run completed successfully")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"SUMMA run failed with error: {e}")
            raise

    def _get_config_path(self, config_key: str, default_suffix: str) -> Path:
        path = self.config.get(config_key)
        if path == 'default':
            return self.project_dir / default_suffix
        return Path(path)

    def _backup_settings(self, source_path: Path, backup_path: Path):
        backup_path.mkdir(parents=True, exist_ok=True)
        os.system(f"cp -R {source_path}/. {backup_path}")
        self.logger.info(f"Settings backed up to {backup_path}")

from sklearn.preprocessing import StandardScaler # type: ignore
import torch.optim as optim # type: ignore
from torch.optim.lr_scheduler import ReduceLROnPlateau # type: ignore
import torch.nn.functional as F # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.dates as mdates # type: ignore
from matplotlib.gridspec import GridSpec # type: ignore
import torch.nn as nn # type: ignore
from torch.optim.lr_scheduler import OneCycleLR # type: ignore
from torch.utils.data import TensorDataset, DataLoader # type: ignore

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


    def preprocess_data(self, forcing_df: pd.DataFrame, streamflow_df: pd.DataFrame, snow_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, pd.DatetimeIndex, pd.Index]:
        self.logger.info("Preprocessing data for FLASH model")
        
        # Align the data
        common_dates = forcing_df.index.get_level_values('time').intersection(streamflow_df.index).intersection(snow_df.index)
        forcing_df = forcing_df.loc[pd.IndexSlice[common_dates, :], :]
        streamflow_df = streamflow_df.loc[common_dates]
        snow_df = snow_df.loc[common_dates]

        self.logger.info(f"Shape of aligned forcing_df: {forcing_df.shape}")
        self.logger.info(f"Shape of aligned streamflow_df: {streamflow_df.shape}")
        self.logger.info(f"Shape of aligned snow_df: {snow_df.shape}")

        # Prepare features (forcing data)
        features = forcing_df.reset_index()
        hru_ids = features['hruId'].unique()
        features['hruId'] = features['hruId'].astype('category').cat.codes  # Convert hruId to numeric
        features.set_index(['time', 'hruId'], inplace=True)
        feature_columns = features.columns.drop(['time', 'hruId'] if 'time' in features.columns and 'hruId' in features.columns else [])
        
         # Scale features
        self.feature_scaler = StandardScaler()
        scaled_features = self.feature_scaler.fit_transform(features[feature_columns])
        scaled_features = np.clip(scaled_features, -10, 10)  # Clip to avoid extreme values


        self.logger.info(f"Shape of scaled_features: {scaled_features.shape}")

        # Prepare targets (streamflow and snow)
        targets = pd.concat([streamflow_df['streamflow'], snow_df['snw']], axis=1)
        targets.columns = ['streamflow', 'SWE']
        
        # Scale targets
        self.target_scaler = StandardScaler()
        scaled_targets = self.target_scaler.fit_transform(targets)
        scaled_targets = np.clip(scaled_targets, -10, 10)  # Clip to avoid extreme values



        self.logger.info(f"Shape of targets: {targets.shape}")

        # Create sequences
        X, y = [], []
        for hru in forcing_df.index.get_level_values('hruId').unique():
            hru_data = scaled_features[forcing_df.index.get_level_values('hruId') == hru]
            target_data = np.tile(scaled_targets, (len(hru_ids), 1))  # Repeat targets for each HRU
            for i in range(len(hru_data) - self.lookback):
                X.append(hru_data[i:(i + self.lookback)])
                y.append(target_data[i + self.lookback])

        # Convert to PyTorch tensors
        X = torch.FloatTensor(np.array(X)).to(self.device)
        y = torch.FloatTensor(np.array(y)).to(self.device)

        self.logger.info(f"Preprocessed data shape: X: {X.shape}, y: {y.shape}")
        return X, y, common_dates, hru_ids
    
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
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
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
                out = self.fc(out)
                return out
        self.model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(self.device)
        self.logger.info(f"FLASH model created: {self.model}")

    def train_model(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001):
        self.logger.info(f"Training FLASH model with {epochs} epochs, batch_size: {batch_size}, learning_rate: {learning_rate}")

        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        criterion = nn.SmoothL1Loss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Calculate the exact number of steps
        steps_per_epoch = (len(X_train) - 1) // batch_size + 1
        total_steps = steps_per_epoch * epochs

        self.logger.info(f"Train data shape: {X_train.shape}")
        self.logger.info(f"Steps per epoch: {steps_per_epoch}")
        self.logger.info(f"Total steps: {total_steps}")
                
        scheduler = OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps)

        best_val_loss = float('inf')
        patience = 10
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
                
                # Gradient clipping and noise addition
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.data.clamp_(-1, 1)
                        param.grad.data.add_(torch.randn_like(param.grad) * 0.01)
                
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val)

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

    def simulate(self, forcing_df: pd.DataFrame, streamflow_df: pd.DataFrame, snow_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Running full simulation with FLASH model")
        X, _, common_dates, hru_ids = self.preprocess_data(forcing_df, streamflow_df, snow_df)
        
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

        
        self.logger.info(f"Shape of predictions: {predictions.shape}")
        self.logger.info(f"Shape of forcing_df: {forcing_df.shape}")
        
        # Get the dates for which we have predictions
        prediction_dates = common_dates[self.lookback:]
        
        self.logger.info(f"Number of prediction dates: {len(prediction_dates)}")
        self.logger.info(f"Number of HRUs in predictions: {len(hru_ids)}")
        
        # Create a DataFrame for predictions
        pred_df = pd.DataFrame(predictions, columns=['predicted_streamflow', 'predicted_SWE'])
        pred_df['time'] = np.repeat(prediction_dates, len(hru_ids))
        pred_df['hruId'] = np.tile(hru_ids, len(prediction_dates))
        pred_df.set_index(['time', 'hruId'], inplace=True)
        
        self.logger.info(f"Shape of pred_df: {pred_df.shape}")
        
        # Combine predictions with original data
        result = forcing_df.copy()
        result = result.join(pred_df, how='left')
        
        self.logger.info(f"Shape of final result: {result.shape}")
        self.logger.info("Simulation completed")
        return result
    
    def save_model(self, path: Path):
        """
        Save the trained FLASH model to a file.

        Args:
            path (Path): Path to save the model.
        """
        self.logger.info(f"Saving FLASH model to {path}")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'lookback': self.lookback,
        }, path)
        self.logger.info("Model saved successfully")

    def load_model(self, path: Path):
        """
        Load a trained FLASH model from a file.

        Args:
            path (Path): Path to the saved model file.
        """
        self.logger.info(f"Loading FLASH model from {path}")
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.lookback = checkpoint['lookback']
        self.model.eval()
        self.logger.info("Model loaded successfully")

    def run_flash(self):
        self.logger.info("Starting FLASH model run")

        try:
            # Load data
            forcing_df, streamflow_df, snow_df = self._load_data()
            
            # Preprocess data
            X, y, common_dates, hru_ids = self.preprocess_data(forcing_df, streamflow_df, snow_df)
            
            # Create and train model
            input_size = X.shape[2]
            hidden_size = self.config.get('FLASH_HIDDEN_SIZE', 64)
            num_layers = self.config.get('FLASH_NUM_LAYERS', 2)
            output_size = 2  # streamflow and SWE
            
            self.create_model(input_size, hidden_size, num_layers, output_size)
            self.train_model(X, y,
                            epochs=self.config.get('FLASH_EPOCHS', 100),
                            batch_size=self.config.get('FLASH_BATCH_SIZE', 32),
                            learning_rate=self.config.get('FLASH_LEARNING_RATE', 0.001))
            
            # Run simulation
            results = self.simulate(forcing_df, streamflow_df, snow_df)
            
            # Visualize results
            self.visualize_results(results, streamflow_df, snow_df)
            
            # Save results
            self._save_results(results)
            
            # Save model
            model_save_path = self.project_dir / 'models' / 'flash_model.pt'
            self.save_model(model_save_path)
            
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
        """
        Save the results of the FLASH model simulation.

        Args:
            results (pd.DataFrame): Simulation results to be saved.
        """
        self.logger.info("Saving FLASH model results")
        
        # Prepare the output directory
        output_dir = self.project_dir / 'simulations' / self.config.get('EXPERIMENT_ID') / 'FLASH'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Reshape results to have time and hru dimensions
        hru_ids = results.index.get_level_values('hruId').unique()
        times = results.index.get_level_values('time').unique()
        
        streamflow = results['predicted_streamflow'].values.reshape(len(times), len(hru_ids))
        swe = results['predicted_SWE'].values.reshape(len(times), len(hru_ids))
        
        # Create xarray Dataset
        ds = xr.Dataset(
            {
                "IRFroutedRunoff": (["time", "hru"], streamflow),
                "scalarSWE": (["time", "hru"], swe)
            },
            coords={
                "time": times,
                "hru": hru_ids
            }
        )
        
        # Add attributes
        ds.IRFroutedRunoff.attrs['units'] = 'm3 s-1'
        ds.IRFroutedRunoff.attrs['long_name'] = 'Routed streamflow'
        ds.scalarSWE.attrs['units'] = 'mm'
        ds.scalarSWE.attrs['long_name'] = 'Snow Water Equivalent'
        
        # Save as NetCDF
        output_file = output_dir / f"{self.config.get('EXPERIMENT_ID')}_FLASH_output.nc"
        ds.to_netcdf(output_file)
        
        self.logger.info(f"FLASH results saved to {output_file}")

    def visualize_results(self, results_df: pd.DataFrame, obs_streamflow: pd.DataFrame, obs_snow: pd.DataFrame):
        """
        Visualize the observed and simulated snow and streamflow.

        Args:
        results_df (pd.DataFrame): DataFrame containing the simulation results
        obs_streamflow (pd.DataFrame): DataFrame containing observed streamflow data
        obs_snow (pd.DataFrame): DataFrame containing observed snow data
        """
        self.logger.info("Visualizing results")

        # Prepare data
        sim_dates = results_df.index.get_level_values('time').unique()
        sim_streamflow = results_df.groupby('time')['predicted_streamflow'].mean()
        sim_swe = results_df.groupby('time')['predicted_SWE'].mean()

        obs_streamflow = obs_streamflow.reindex(sim_dates)
        obs_snow = obs_snow.reindex(sim_dates)
        self.logger.info(obs_snow)
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 1, height_ratios=[1, 1])

        # Streamflow subplot
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(sim_dates, sim_streamflow, label='Simulated', color='blue')
        ax1.plot(sim_dates, obs_streamflow, label='Observed', color='red')
        ax1.set_ylabel('Streamflow (m³/s)')
        ax1.set_title('Observed vs Simulated Streamflow')
        ax1.legend()
        ax1.grid(True)

        # Snow subplot
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(sim_dates, sim_swe, label='Simulated', color='blue')
        ax2.plot(sim_dates, obs_snow, label='Observed', color='red')
        ax2.set_ylabel('Snow Water Equivalent (mm)')
        ax2.set_title('Observed vs Simulated Snow Water Equivalent')
        ax2.legend()
        ax2.grid(True)

        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        # Save the figure
        output_dir = self.project_dir / 'plots' / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_path = output_dir / f'{self.config.get("EXPERIMENT_ID")}_simulation_results.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Results visualization saved to {fig_path}")