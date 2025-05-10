#!/usr/bin/env python3
"""
Script to compare traditional DDS optimization with RF emulation approaches.

This script runs both optimization approaches on the same problem and produces
comparative visualizations and metrics to evaluate their relative performance.
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Import CONFLUENCE utilities
from utils.configHandling_utils.config_utils import ConfigManager
from utils.configHandling_utils.logging_utils import setup_logger, get_function_logger

# Import optimizers
from utils.optimization_utils.traditional_optimization import TraditionalOptimizer
from utils.emulation_utils.random_forest_emulator import RandomForestEmulator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare traditional optimization with RF emulation.')
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to CONFLUENCE configuration file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Custom output directory for comparison results (optional)')
    parser.add_argument('--iterations', type=int, default=None,
                        help='Number of iterations for traditional optimization (overrides config)')
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of samples for RF emulation (overrides config)')
    parser.add_argument('--dds-only', action='store_true',
                        help='Run only traditional DDS optimization')
    parser.add_argument('--rf-only', action='store_true',
                        help='Run only RF emulation optimization')
    parser.add_argument('--save-params', action='store_true',
                        help='Save optimized parameters for both methods to file')
    parser.add_argument('--show-plots', action='store_true',
                        help='Show plots during optimization (not just save them)')
    
    return parser.parse_args()


class OptimizerComparison:
    """
    Class to handle the comparison between traditional and emulation optimizers.
    """
    
    def __init__(self, config_path, output_dir=None, show_plots=False):
        """
        Initialize the comparison.
        
        Args:
            config_path: Path to CONFLUENCE configuration file
            output_dir: Custom output directory (optional)
            show_plots: Whether to show plots during optimization
        """
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        self.config_path = config_path
        
        # Get project directory
        self.data_dir = Path(self.config.get('CONFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.experiment_id = self.config.get('EXPERIMENT_ID')
        
        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.project_dir / "optimisation" / f"{self.experiment_id}_comparison"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.log_file = self.output_dir / f"optimizer_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = setup_logger('optimizer_comparison', self.log_file)
        
        # Show plots flag
        self.show_plots = show_plots
        
        self.logger.info(f"Optimizer comparison initialized for experiment {self.experiment_id}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
    def run_comparison(self, run_dds=True, run_rf=True, dds_iterations=None, rf_samples=None, save_params=False):
        """
        Run the comparison between traditional and emulation optimizers.
        
        Args:
            run_dds: Whether to run DDS optimization
            run_rf: Whether to run RF emulation
            dds_iterations: Number of iterations for traditional optimization (overrides config)
            rf_samples: Number of samples for RF emulation (overrides config)
            save_params: Whether to save optimized parameters to file
            
        Returns:
            Dict: Comparison results
        """
        self.logger.info("Starting optimizer comparison")
        
        # Create results dictionary
        results = {
            'experiment_id': self.experiment_id,
            'config_path': str(self.config_path),
            'output_dir': str(self.output_dir),
            'timestamp': datetime.now().isoformat()
        }
        
        # Override config if iterations or samples provided
        original_config = self.config.copy()
        if dds_iterations is not None:
            self.config['NUMBER_OF_ITERATIONS'] = dds_iterations
            self.logger.info(f"Overriding DDS iterations: {dds_iterations}")
        
        if rf_samples is not None:
            self.config['EMULATION_NUM_SAMPLES'] = rf_samples
            self.logger.info(f"Overriding RF samples: {rf_samples}")
        
        # Run traditional DDS optimization
        if run_dds:
            self.logger.info("Running traditional DDS optimization")
            dds_start_time = time.time()
            
            try:
                # Initialize traditional optimizer
                traditional_optimizer = TraditionalOptimizer(self.config, self.logger)
                
                # Run DDS optimization
                dds_best_params, dds_best_value = traditional_optimizer.optimize_with_dds()
                
                dds_elapsed_time = time.time() - dds_start_time
                
                self.logger.info(f"DDS optimization completed in {dds_elapsed_time:.2f} seconds")
                self.logger.info(f"DDS best {self.config.get('OPTIMIZATION_METRIC')}: {dds_best_value:.4f}")
                
                # Save results
                results['dds'] = {
                    'best_value': dds_best_value,
                    'time': dds_elapsed_time,
                    'params': dds_best_params,
                    'iterations': self.config.get('NUMBER_OF_ITERATIONS')
                }
                
                # Save optimized parameters if requested
                if save_params:
                    dds_params_file = self.output_dir / "dds_optimized_parameters.csv"
                    dds_params_df = pd.DataFrame.from_dict(dds_best_params, orient='index', columns=['Value'])
                    dds_params_df.index.name = 'Parameter'
                    dds_params_df.to_csv(dds_params_file)
                    self.logger.info(f"Saved DDS optimized parameters to {dds_params_file}")
                
            except Exception as e:
                self.logger.error(f"Error during DDS optimization: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                results['dds'] = {
                    'error': str(e),
                    'completed': False
                }
        
        # Run RF emulation optimization
        if run_rf:
            self.logger.info("Running RF emulation optimization")
            rf_start_time = time.time()
            
            try:
                # Initialize RF emulator
                rf_emulator = RandomForestEmulator(self.config, self.logger)
                
                # Run RF emulation workflow
                rf_results = rf_emulator.run_workflow()
                
                rf_elapsed_time = time.time() - rf_start_time
                
                if rf_results:
                    rf_best_params = rf_results['optimized_parameters']
                    rf_best_value = rf_results['predicted_score']
                    
                    self.logger.info(f"RF emulation completed in {rf_elapsed_time:.2f} seconds")
                    self.logger.info(f"RF best {self.config.get('OPTIMIZATION_METRIC')}: {rf_best_value:.4f}")
                    
                    # Save results
                    results['rf'] = {
                        'best_value': rf_best_value,
                        'time': rf_elapsed_time,
                        'params': rf_best_params,
                        'samples': self.config.get('EMULATION_NUM_SAMPLES')
                    }
                    
                    # Save optimized parameters if requested
                    if save_params:
                        rf_params_file = self.output_dir / "rf_optimized_parameters.csv"
                        rf_params_df = pd.DataFrame.from_dict(rf_best_params, orient='index', columns=['Value'])
                        rf_params_df.index.name = 'Parameter'
                        rf_params_df.to_csv(rf_params_file)
                        self.logger.info(f"Saved RF optimized parameters to {rf_params_file}")
                
                else:
                    self.logger.error("RF emulation failed to produce results")
                    results['rf'] = {
                        'error': "RF emulation failed to produce results",
                        'completed': False,
                        'time': rf_elapsed_time
                    }
                    
            except Exception as e:
                self.logger.error(f"Error during RF emulation: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                results['rf'] = {
                    'error': str(e),
                    'completed': False
                }
        
        # Compare results if both methods were run
        if run_dds and run_rf and 'dds' in results and 'rf' in results and 'error' not in results['dds'] and 'error' not in results['rf']:
            self.logger.info("Comparing optimization results")
            try:
                # Extract results
                dds_best_params = results['dds']['params']
                dds_best_value = results['dds']['best_value']
                dds_elapsed_time = results['dds']['time']
                
                rf_best_params = results['rf']['params']
                rf_best_value = results['rf']['best_value']
                rf_elapsed_time = results['rf']['time']
                
                # Create comparison
                comparison_results = self._compare_results(
                    dds_best_params, dds_best_value, dds_elapsed_time,
                    rf_best_params, rf_best_value, rf_elapsed_time
                )
                
                results['comparison'] = comparison_results
                
            except Exception as e:
                self.logger.error(f"Error comparing results: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                results['comparison'] = {
                    'error': str(e),
                    'completed': False
                }
        
        # Restore original config
        self.config = original_config
        
        # Save overall results to file
        self._save_results(results)
        
        return results
    
    def _compare_results(self, dds_params, dds_value, dds_time, rf_params, rf_value, rf_time):
        """
        Compare results from DDS and RF emulation optimization.
        
        Args:
            dds_params: Best parameters from DDS
            dds_value: Best objective value from DDS
            dds_time: Time taken by DDS optimization
            rf_params: Best parameters from RF emulation
            rf_value: Best objective value from RF emulation
            rf_time: Time taken by RF emulation
            
        Returns:
            Dict: Comparison results
        """
        # Create a dataframe with parameter values
        common_params = set(dds_params.keys()) & set(rf_params.keys())
        comparison_df = pd.DataFrame({
            'Parameter': list(common_params),
            'DDS': [dds_params[param] for param in common_params],
            'RF': [rf_params[param] for param in common_params]
        })
        
        # Calculate parameter differences
        comparison_df['Absolute Difference'] = abs(comparison_df['DDS'] - comparison_df['RF'])
        comparison_df['Relative Difference %'] = 100 * comparison_df['Absolute Difference'] / abs(comparison_df['DDS'])
        
        # Save comparison dataframe
        comparison_df.to_csv(self.output_dir / "parameter_comparison.csv", index=False)
        
        # Create performance comparison
        performance_df = pd.DataFrame({
            'Method': ['DDS', 'RF Emulation'],
            self.config.get('OPTIMIZATION_METRIC'): [dds_value, rf_value],
            'Time (s)': [dds_time, rf_time],
            'Speedup': [1.0, dds_time / rf_time]
        })
        
        # Save performance comparison
        performance_df.to_csv(self.output_dir / "performance_comparison.csv", index=False)
        
        # Create parameter comparison plot
        self._plot_parameter_comparison(comparison_df)
        
        # Create performance metrics plot
        self._plot_performance_comparison(performance_df)
        
        # Return comparison metrics
        return {
            'param_diff_max': comparison_df['Absolute Difference'].max(),
            'param_diff_mean': comparison_df['Absolute Difference'].mean(),
            'param_diff_max_pct': comparison_df['Relative Difference %'].max(),
            'param_diff_mean_pct': comparison_df['Relative Difference %'].mean(),
            'value_diff': dds_value - rf_value,
            'value_diff_pct': 100 * (dds_value - rf_value) / abs(dds_value) if dds_value != 0 else np.inf,
            'speedup': dds_time / rf_time,
            'dds_time': dds_time,
            'rf_time': rf_time
        }
        
    def _plot_parameter_comparison(self, comparison_df):
        """
        Create parameter comparison plots.
        
        Args:
            comparison_df: DataFrame with parameter comparisons
        """
        try:
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Number of parameters
            n_params = len(comparison_df)
            
            # Set up bar positions
            ind = np.arange(n_params)
            width = 0.35
            
            # Plot bars for each method
            plt.bar(ind - width/2, comparison_df['DDS'], width, label='DDS')
            plt.bar(ind + width/2, comparison_df['RF'], width, label='RF Emulation')
            
            # Add labels and legend
            plt.xlabel('Parameter')
            plt.ylabel('Parameter Value')
            plt.title('Parameter Comparison - DDS vs RF Emulation')
            plt.xticks(ind, comparison_df['Parameter'], rotation=45, ha='right')
            plt.legend()
            
            # Add grid
            plt.grid(True, axis='y', alpha=0.3)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(self.output_dir / "parameter_comparison.png", dpi=300, bbox_inches='tight')
            if self.show_plots:
                plt.show()
            plt.close()
            
            # Create parameter difference plot
            plt.figure(figsize=(12, 8))
            
            # Plot parameter differences
            plt.bar(ind, comparison_df['Relative Difference %'], width)
            
            # Add labels
            plt.xlabel('Parameter')
            plt.ylabel('Relative Difference (%)')
            plt.title('Parameter Relative Difference (|DDS - RF| / |DDS|) %')
            plt.xticks(ind, comparison_df['Parameter'], rotation=45, ha='right')
            
            # Add reference line at zero
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # Add grid
            plt.grid(True, axis='y', alpha=0.3)
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(self.output_dir / "parameter_difference.png", dpi=300, bbox_inches='tight')
            if self.show_plots:
                plt.show()
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating parameter comparison plot: {str(e)}")
            
    def _plot_performance_comparison(self, performance_df):
        """
        Create performance comparison plots.
        
        Args:
            performance_df: DataFrame with performance comparisons
        """
        try:
            # Create figure with two subplots
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            metric_name = self.config.get('OPTIMIZATION_METRIC')
            
            # Plot objective values
            axes[0].bar(['DDS', 'RF Emulation'], performance_df[metric_name])
            axes[0].set_ylabel(metric_name)
            axes[0].set_title(f'{metric_name} Comparison')
            axes[0].grid(True, axis='y', alpha=0.3)
            
            # Add values on top of bars
            for i, v in enumerate(performance_df[metric_name]):
                axes[0].text(i, v + 0.01, f"{v:.4f}", ha='center')
            
            # Plot computation time
            axes[1].bar(['DDS', 'RF Emulation'], performance_df['Time (s)'])
            axes[1].set_ylabel('Time (s)')
            axes[1].set_title('Computation Time Comparison')
            axes[1].grid(True, axis='y', alpha=0.3)
            
            # Add values on top of bars
            for i, v in enumerate(performance_df['Time (s)']):
                axes[1].text(i, v + 0.01, f"{v:.2f}s", ha='center')
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(self.output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
            if self.show_plots:
                plt.show()
            plt.close()
            
            # Create speedup plot
            plt.figure(figsize=(8, 6))
            
            # Plot speedup
            plt.bar(['DDS', 'RF Emulation'], performance_df['Speedup'])
            plt.ylabel('Relative Speed (1 = baseline)')
            plt.title('Computational Efficiency Comparison')
            plt.grid(True, axis='y', alpha=0.3)
            
            # Add values on top of bars
            for i, v in enumerate(performance_df['Speedup']):
                plt.text(i, v + 0.01, f"{v:.2f}x", ha='center')
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(self.output_dir / "speedup_comparison.png", dpi=300, bbox_inches='tight')
            if self.show_plots:
                plt.show()
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error creating performance comparison plot: {str(e)}")
            
    def _save_results(self, results):
        """
        Save results to file.
        
        Args:
            results: Dictionary with comparison results
        """
        try:
            # Save as JSON
            import json
            
            # Create a JSON-serializable copy (handle non-serializable objects)
            serializable_results = self._make_serializable(results)
            
            with open(self.output_dir / "comparison_results.json", 'w') as f:
                json.dump(serializable_results, f, indent=2)
                
            self.logger.info(f"Saved comparison results to {self.output_dir / 'comparison_results.json'}")
            
            # Create a summary report
            with open(self.output_dir / "comparison_summary.txt", 'w') as f:
                f.write(f"Optimizer Comparison Summary\n")
                f.write(f"===========================\n\n")
                f.write(f"Experiment ID: {self.experiment_id}\n")
                f.write(f"Comparison timestamp: {datetime.now()}\n\n")
                
                if 'dds' in results and 'error' not in results['dds']:
                    f.write(f"DDS Optimization:\n")
                    f.write(f"  Best {self.config.get('OPTIMIZATION_METRIC')}: {results['dds']['best_value']:.6f}\n")
                    f.write(f"  Time: {results['dds']['time']:.2f} seconds\n")
                    f.write(f"  Iterations: {results['dds']['iterations']}\n\n")
                
                if 'rf' in results and 'error' not in results['rf']:
                    f.write(f"RF Emulation:\n")
                    f.write(f"  Best {self.config.get('OPTIMIZATION_METRIC')}: {results['rf']['best_value']:.6f}\n")
                    f.write(f"  Time: {results['rf']['time']:.2f} seconds\n")
                    f.write(f"  Samples: {results['rf'].get('samples', 'unknown')}\n\n")
                
                if 'comparison' in results and 'error' not in results['comparison']:
                    f.write(f"Comparison Metrics:\n")
                    f.write(f"  Value difference (DDS - RF): {results['comparison']['value_diff']:.6f}\n")
                    f.write(f"  Value difference (%): {results['comparison']['value_diff_pct']:.2f}%\n")
                    f.write(f"  Mean parameter difference (%): {results['comparison']['param_diff_mean_pct']:.2f}%\n")
                    f.write(f"  Max parameter difference (%): {results['comparison']['param_diff_max_pct']:.2f}%\n")
                    f.write(f"  Speedup (DDS time / RF time): {results['comparison']['speedup']:.2f}x\n\n")
                
                f.write(f"Results and visualizations are available in: {self.output_dir}\n")
            
            self.logger.info(f"Saved comparison summary to {self.output_dir / 'comparison_summary.txt'}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            
    def _make_serializable(self, obj):
        """
        Make an object JSON-serializable by handling non-serializable types.
        
        Args:
            obj: Object to make serializable
            
        Returns:
            Serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(i) for i in obj]
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, (datetime, Path)):
            return str(obj)
        else:
            return obj


def main():
    """Main function to run the comparison."""
    # Parse arguments
    args = parse_arguments()
    
    # Initialize comparison
    comparison = OptimizerComparison(
        config_path=args.config,
        output_dir=args.output_dir,
        show_plots=args.show_plots
    )
    
    # Determine which optimizers to run
    run_dds = not args.rf_only
    run_rf = not args.dds_only
    
    # Run comparison
    results = comparison.run_comparison(
        run_dds=run_dds,
        run_rf=run_rf,
        dds_iterations=args.iterations,
        rf_samples=args.samples,
        save_params=args.save_params
    )
    
    # Print summary
    print("\nOptimizer Comparison Summary:")
    print("===========================")
    
    if 'dds' in results and 'error' not in results['dds']:
        print(f"\nDDS Optimization:")
        print(f"  Best {comparison.config.get('OPTIMIZATION_METRIC')}: {results['dds']['best_value']:.6f}")
        print(f"  Time: {results['dds']['time']:.2f} seconds")
    
    if 'rf' in results and 'error' not in results['rf']:
        print(f"\nRF Emulation:")
        print(f"  Best {comparison.config.get('OPTIMIZATION_METRIC')}: {results['rf']['best_value']:.6f}")
        print(f"  Time: {results['rf']['time']:.2f} seconds")
    
    if 'comparison' in results and 'error' not in results['comparison']:
        print(f"\nComparison Metrics:")
        print(f"  Value difference (DDS - RF): {results['comparison']['value_diff']:.6f}")
        print(f"  Value difference (%): {results['comparison']['value_diff_pct']:.2f}%")
        print(f"  Speedup (DDS time / RF time): {results['comparison']['speedup']:.2f}x")
    
    print(f"\nDetailed results and visualizations are available in:")
    print(f"  {comparison.output_dir}")


if __name__ == "__main__":
    main()
