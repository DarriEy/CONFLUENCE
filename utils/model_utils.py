import os
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

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

        # Log the run
        self._log_run(mizu_out_path)

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

        # Log the run
        self._log_run(summa_out_path)

    def _get_config_path(self, config_key: str, default_suffix: str) -> Path:
        path = self.config.get(config_key)
        if path == 'default':
            return self.project_dir / default_suffix
        return Path(path)

    def _backup_settings(self, source_path: Path, backup_path: Path):
        backup_path.mkdir(parents=True, exist_ok=True)
        os.system(f"cp -R {source_path}/. {backup_path}")
        self.logger.info(f"Settings backed up to {backup_path}")
