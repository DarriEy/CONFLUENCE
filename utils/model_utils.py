import os
import glob
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
import pandas as pd # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import numpy as np
import torch.optim as optim # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
import xarray as xr # type: ignore

class MizuRouteRunner:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.root_path = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.root_path / f"domain_{self.domain_name}"

    def run_mizuroute(self):
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
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.root_path = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.root_path / f"domain_{self.domain_name}"

    def run_summa(self):
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

class FLASH:
    '''
    Welcome to Flash: The Flow and Snow Hydrological LSTM

    '''
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = StandardScaler()
        self.lookback = config.get('FLASH_LOOKBACK', 30)
        self.project_dir = Path(self.config.get('CONFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}"
        self.logger.info(f"Initialized FLASH model with device: {self.device}")


    def preprocess_data(self, forcing_df: pd.DataFrame, streamflow_df: pd.DataFrame, snow_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        self.logger.info("Preprocessing data for FLASH model")
        
        # Align the data
        common_dates = forcing_df.index.get_level_values('time').intersection(streamflow_df.index).intersection(snow_df.index.get_level_values('datetime'))
        forcing_df = forcing_df.loc[common_dates]
        streamflow_df = streamflow_df.loc[common_dates]
        snow_df = snow_df.loc[common_dates]

        # Prepare features (forcing data)
        features = forcing_df.drop('hruId', axis=1, errors='ignore')
        scaled_features = self.scaler.fit_transform(features)

        # Prepare targets (streamflow and snow)
        targets = pd.concat([streamflow_df['streamflow'], snow_df['snw'].unstack(level='station_id').mean(axis=1)], axis=1)
        targets.columns = ['streamflow', 'SWE']

        # Create sequences
        X, y = [], []
        for hru in forcing_df.index.get_level_values('hruId').unique():
            hru_data = scaled_features[forcing_df.index.get_level_values('hruId') == hru]
            for i in range(len(hru_data) - self.lookback):
                X.append(hru_data[i:(i + self.lookback)])
                y.append(targets.iloc[i + self.lookback])

        # Convert to PyTorch tensors
        X = torch.FloatTensor(np.array(X)).to(self.device)
        y = torch.FloatTensor(np.array(y)).to(self.device)

        self.logger.info(f"Preprocessed data shape: X: {X.shape}, y: {y.shape}")
        return X, y

    def create_model(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        self.logger.info(f"Creating FLASH model with input_size: {input_size}, hidden_size: {hidden_size}, num_layers: {num_layers}, output_size: {output_size}")
        
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])
                return out

        self.model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(self.device)
        self.logger.info(f"FLASH model created: {self.model}")

    def train_model(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001):
        self.logger.info(f"Training FLASH model with {epochs} epochs, batch_size: {batch_size}, learning_rate: {learning_rate}")

        # Split data into training and validation sets
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for i in range(0, X_train.size(0), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val)

            if (epoch + 1) % 10 == 0:
                self.logger.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}')

        self.logger.info("FLASH model training completed")

    def predict(self, X: torch.Tensor) -> np.ndarray:
        self.logger.info(f"Making predictions with FLASH model")
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        return predictions.cpu().numpy()

    def simulate(self, data: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Running full simulation with FLASH model")
        X, _ = self.preprocess_data(data)
        predictions = self.predict(X)
        
        # Create a DataFrame with the predictions
        hru_ids = data.index.get_level_values('hruId').unique()
        times = data.index.get_level_values('time').unique()[self.lookback:]
        
        pred_df = pd.DataFrame(predictions, columns=['streamflow', 'SWE'])
        pred_df['hruId'] = np.repeat(hru_ids, len(times))
        pred_df['time'] = np.tile(times, len(hru_ids))
        pred_df.set_index(['time', 'hruId'], inplace=True)
        
        # Combine with original data
        result = data.join(pred_df, how='left')
        result = result.rename(columns={
            'streamflow': 'predicted_streamflow',
            'SWE': 'predicted_SWE'
        })
        
        self.logger.info("Simulation completed")
        return result

    def save_model(self, path: Path):
        self.logger.info(f"Saving FLASH model to {path}")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'lookback': self.lookback,
        }, path)
        self.logger.info("Model saved successfully")

    def load_model(self, path: Path):
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
            X, y = self.preprocess_data(forcing_df, streamflow_df, snow_df)
            
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
            results = self.simulate(forcing_df)
            
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
        
        forcing_df = forcing_df.set_index(['time', 'hruId']).sort_index()

        # Load streamflow data
        streamflow_path = self.project_dir / 'observations' / 'streamflow' / 'preprocessed' / f"{self.config.get('DOMAIN_NAME')}_streamflow_processed.csv"
        streamflow_df = pd.read_csv(streamflow_path, parse_dates=['datetime'])
        streamflow_df = streamflow_df.set_index('datetime').rename(columns={'discharge_cms': 'streamflow'})

        # Load snow data
        snow_path = self.project_dir / 'observations' / 'snow' / 'preprocessed'
        snow_files = glob.glob(str(snow_path / f"{self.config.get('DOMAIN_NAME')}_filtered_snow_observations.csv"))
        if not snow_files:
            raise FileNotFoundError(f"No snow observation files found in {snow_path}")
        
        snow_df = pd.concat([pd.read_csv(file, parse_dates=['datetime']) for file in snow_files])
        snow_df = snow_df.set_index(['datetime', 'station_id'])
        
        self.logger.info(f"Loaded forcing data with shape: {forcing_df.shape}")
        self.logger.info(f"Loaded streamflow data with shape: {streamflow_df.shape}")
        self.logger.info(f"Loaded snow data with shape: {snow_df.shape}")
        
        return forcing_df, streamflow_df, snow_df

    def _save_results(self, results: pd.DataFrame):
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