#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import time
from typing import Dict, List, Any, Optional, Tuple, Union

# Import optimizers
from utils.optimization_utils.dds_optimizer import DDSOptimizer
from utils.optimization_utils.sce_ua_optimizer import SCEUAOptimizer

class OptimizerManager:
    """
    Manager class for different optimization algorithms in CONFLUENCE.
    
    This class provides a unified interface to different optimization algorithms,
    selecting the appropriate one based on configuration settings.
    
    Supported algorithms:
    - DDS (Dynamically Dimensioned Search)
    - SCE-UA (Shuffled Complex Evolution - University of Arizona)
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        """
        Initialize the OptimizerManager.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = self.config.get('EXPERIMENT_ID')
        
        # Get optimization algorithm
        self.algorithm = self.config.get('ITERATIVE_OPTIMIZATION_ALGORITHM', 'DDS')
        
        # Validate algorithm
        valid_algorithms = ['DDS', 'SCE-UA']
        if self.algorithm not in valid_algorithms:
            self.logger.warning(f"Invalid optimization algorithm: {self.algorithm}. Using DDS as default.")
            self.algorithm = 'DDS'
        
        self.optimizer = None
        
    def run_optimization(self):
        """
        Run the selected optimization algorithm.
        
        Returns:
            Dict: Dictionary with optimization results
        """
        self.logger.info(f"Starting optimization using {self.algorithm} algorithm")
        
        start_time = time.time()
        
        # Initialize the appropriate optimizer
        if self.algorithm == 'DDS':
            self.optimizer = DDSOptimizer(self.config, self.logger)
            results = self.optimizer.run_dds_optimization()
        elif self.algorithm == 'SCE-UA':
            self.optimizer = SCEUAOptimizer(self.config, self.logger)
            results = self.optimizer.run_sceua_optimization()
        else:
            self.logger.error(f"Unsupported optimization algorithm: {self.algorithm}")
            return None
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        self.logger.info(f"Optimization completed in {elapsed_time:.2f} seconds")
        
        # Save results summary
        self._save_results_summary(results, elapsed_time)
        
        return results
    
    def _save_results_summary(self, results, elapsed_time):
        """
        Save a summary of optimization results.
        
        Args:
            results: Results dictionary from optimizer
            elapsed_time: Time taken for optimization
        """
        output_dir = Path(results['output_dir'])
        
        # Create summary dataframe
        summary = {
            'algorithm': self.algorithm,
            'experiment_id': self.experiment_id,
            'domain_name': self.domain_name,
            'best_score': results['best_score'],
            'elapsed_time': elapsed_time,
            'iterations': len(results['history']),
            'algorithm_settings': str({k: v for k, v in self.config.items() if k.startswith(self.algorithm)})
        }
        
        # Add information about best parameters
        if results['best_parameters']:
            for param_name, values in results['best_parameters'].items():
                if isinstance(values, np.ndarray) and len(values) > 1:
                    summary[f"{param_name}_mean"] = np.mean(values)
                    summary[f"{param_name}_min"] = np.min(values)
                    summary[f"{param_name}_max"] = np.max(values)
                else:
                    value = values[0] if isinstance(values, np.ndarray) else values
                    summary[param_name] = value
        
        # Create dataframe
        summary_df = pd.DataFrame([summary])
        
        # Save to CSV
        summary_file = output_dir / "optimization_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        self.logger.info(f"Saved optimization summary to: {summary_file}")
        
        # Also save to main optimization directory
        opt_dir = self.project_dir / "optimisation"
        experiment_summary_file = opt_dir / f"{self.experiment_id}_summary.csv"
        
        # Check if file exists to append
        if experiment_summary_file.exists():
            try:
                existing_df = pd.read_csv(experiment_summary_file)
                combined_df = pd.concat([existing_df, summary_df], ignore_index=True)
                combined_df.to_csv(experiment_summary_file, index=False)
            except Exception as e:
                self.logger.warning(f"Could not append to existing summary file: {str(e)}")
                summary_df.to_csv(experiment_summary_file, index=False)
        else:
            summary_df.to_csv(experiment_summary_file, index=False)
        
        self.logger.info(f"Saved/updated experiment summary to: {experiment_summary_file}")


def main():
    """
    Main function for running optimization from command line.
    
    This allows running optimizer_manager.py directly with a config file.
    """
    import argparse
    import sys
    import yaml
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run optimization for CONFLUENCE')
    parser.add_argument('--config', type=str, help='Path to configuration file', required=True)
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration: {str(e)}")
        sys.exit(1)
    
    # Setup logger
    log_dir = Path(config['CONFLUENCE_DATA_DIR']) / f"domain_{config['DOMAIN_NAME']}" / "_workLog"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"optimization_{config['EXPERIMENT_ID']}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    logger = logging.getLogger('optimization')
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Run optimization
    try:
        manager = OptimizerManager(config, logger)
        results = manager.run_optimization()
        
        if results:
            logger.info(f"Optimization completed successfully. Best score: {results['best_score']:.4f}")
            sys.exit(0)
        else:
            logger.error("Optimization failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()