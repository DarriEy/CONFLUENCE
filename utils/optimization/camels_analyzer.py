#!/usr/bin/env python3
"""
CAMELS Spatial Run Performance Analyzer

This script analyzes CAMELS spatial model runs to extract performance metrics
from log files and create summary statistics and visualizations.

Enhanced version includes:
- Optimization completion status detection
- Last completed generation tracking for incomplete runs
"""

import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_domain_id(domain_dir_name):
    """Extract domain ID from directory name (e.g., domain_CAN_01BP002_headwater -> CAN_01BP002)"""
    match = re.search(r'domain_([A-Z]{3}_[0-9A-Z]+)_', domain_dir_name)
    return match.group(1) if match else domain_dir_name

def find_latest_log_file(worklog_dir):
    """Find the latest confluence_general log file in the worklog directory"""
    if not os.path.exists(worklog_dir):
        return None
    
    log_pattern = os.path.join(worklog_dir, "confluence_general_*.log")
    log_files = glob.glob(log_pattern)
    
    if not log_files:
        return None
    
    # Sort by modification time to get the latest
    latest_log = max(log_files, key=os.path.getmtime)
    return latest_log

def extract_optimization_metrics(log_file_path):
    """Extract optimization metrics including completion status and generation progress"""
    if not os.path.exists(log_file_path):
        return None, None, False, None, None
    
    initial_score = None
    best_kge = None
    optimization_completed = False
    last_generation = None
    max_generations = None
    
    try:
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # Extract initial population score (handle negative values)
            initial_pattern = r'Initial population evaluated\. Best score: ([-\d\.]+)'
            initial_match = re.search(initial_pattern, content)
            if initial_match:
                initial_score = float(initial_match.group(1))
            
            # Extract all NEW GLOBAL BEST KGE values and get the last one (handle negative values)
            kge_pattern = r'NEW GLOBAL BEST! KGE=([-\d\.]+)'
            kge_matches = re.findall(kge_pattern, content)
            if kge_matches:
                best_kge = float(kge_matches[-1])  # Get the last (most recent) best KGE
            
            # Check for optimization completion
            completion_patterns = [
                r'OPTIMIZATION COMPLETED',
                r'DE OPTIMIZATION COMPLETED'
            ]
            for pattern in completion_patterns:
                if re.search(pattern, content):
                    optimization_completed = True
                    break
            
            # Extract generation progress - find all generation mentions
            # Pattern matches: "Gen XXX/YYY:" or "Gen XXX:" 
            gen_pattern = r'Gen\s+(\d+)(?:/(\d+))?:'
            gen_matches = re.findall(gen_pattern, content)
            
            if gen_matches:
                # Get the last (highest) generation number found
                generations = []
                max_gens = []
                
                for match in gen_matches:
                    gen_num = int(match[0])
                    generations.append(gen_num)
                    
                    # If format is "Gen XXX/YYY:", extract max generations
                    if match[1]:  # match[1] is the YYY part
                        max_gens.append(int(match[1]))
                
                if generations:
                    last_generation = max(generations)
                
                if max_gens:
                    max_generations = max(max_gens)
                
    except Exception as e:
        logger.error(f"Error reading log file {log_file_path}: {e}")
        return None, None, False, None, None
    
    return initial_score, best_kge, optimization_completed, last_generation, max_generations

def debug_log_file(log_file_path):
    """Debug function to examine log file content and regex matches"""
    print(f"\nDebugging log file: {log_file_path}")
    print("="*60)
    
    if not os.path.exists(log_file_path):
        print("Log file does not exist!")
        return
    
    try:
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Search for initial population pattern
        initial_pattern = r'Initial population evaluated\. Best score: ([-\d\.]+)'
        initial_matches = re.findall(initial_pattern, content)
        print(f"Initial population matches: {initial_matches}")
        
        # Search for KGE pattern
        kge_pattern = r'NEW GLOBAL BEST! KGE=([-\d\.]+)'
        kge_matches = re.findall(kge_pattern, content)
        print(f"KGE matches: {kge_matches}")
        
        # Search for completion patterns
        completion_patterns = [
            r'OPTIMIZATION COMPLETED',
            r'DE OPTIMIZATION COMPLETED'
        ]
        for pattern in completion_patterns:
            if re.search(pattern, content):
                print(f"Found completion pattern: {pattern}")
                break
        else:
            print("No completion pattern found")
        
        # Search for generation patterns
        gen_pattern = r'Gen\s+(\d+)(?:/(\d+))?:'
        gen_matches = re.findall(gen_pattern, content)
        print(f"Generation matches: {gen_matches}")
        if gen_matches:
            generations = [int(match[0]) for match in gen_matches]
            print(f"Last generation: {max(generations)}")
        
        # Show some context around matches
        if initial_matches:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'Initial population evaluated' in line:
                    print(f"Initial population context (line {i+1}): {line}")
        
        if kge_matches:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'NEW GLOBAL BEST!' in line:
                    print(f"Best KGE context (line {i+1}): {line}")
        
        # Show generation context
        if gen_matches:
            lines = content.split('\n')
            shown_lines = set()  # To avoid showing duplicate lines
            for i, line in enumerate(lines):
                if re.search(gen_pattern, line) and i not in shown_lines:
                    print(f"Generation context (line {i+1}): {line}")
                    shown_lines.add(i)
                    if len(shown_lines) >= 5:  # Limit output
                        break
                        
    except Exception as e:
        print(f"Error reading log file: {e}")

def analyze_camels_spat_runs(base_path):
    """Analyze all CAMELS spatial runs and extract performance metrics"""
    
    base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Base path does not exist: {base_path}")
    
    results = []
    invalid_basins = []
    
    # Find all domain directories
    domain_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('domain_')]
    
    logger.info(f"Found {len(domain_dirs)} domain directories")
    
    for domain_dir in domain_dirs:
        domain_id = extract_domain_id(domain_dir.name)
        
        # Construct worklog directory path
        worklog_dir_name = f"_workLog_{domain_id.strip('domain_')}_"
        worklog_dir = domain_dir / worklog_dir_name
        
        if not worklog_dir.exists():
            logger.warning(f"Worklog directory not found for {domain_id}: {worklog_dir}")
            invalid_basins.append(domain_id)
            continue
        
        # Find latest log file
        latest_log = find_latest_log_file(str(worklog_dir))
        
        if not latest_log:
            logger.warning(f"No log files found for {domain_id}")
            invalid_basins.append(domain_id)
            continue
        
        # Extract optimization metrics
        initial_score, best_kge, completed, last_gen, max_gen = extract_optimization_metrics(latest_log)
        
        if initial_score is None and best_kge is None:
            logger.warning(f"No valid performance scores found for {domain_id}")
            invalid_basins.append(domain_id)
            continue
        
        # Log missing components for debugging
        if initial_score is None:
            logger.debug(f"Missing initial score for {domain_id}")
        if best_kge is None:
            logger.debug(f"Missing best KGE for {domain_id}")
        
        # Calculate completion percentage
        completion_pct = None
        if last_gen is not None and max_gen is not None:
            completion_pct = (last_gen / max_gen) * 100
        
        results.append({
            'domain_id': domain_id,
            'initial_kge': initial_score,
            'best_kge': best_kge,
            'optimization_completed': completed,
            'last_generation': last_gen,
            'max_generations': max_gen,
            'completion_percentage': completion_pct,
            'log_file': latest_log
        })
        
        status = "COMPLETED" if completed else f"INCOMPLETE ({last_gen}/{max_gen})" if last_gen and max_gen else "UNKNOWN"
        logger.info(f"Processed {domain_id}: Initial={initial_score}, Best={best_kge}, Status={status}")
    
    return results, invalid_basins

def create_performance_table(results):
    """Create a pandas DataFrame with performance metrics"""
    df = pd.DataFrame(results)
    
    # Calculate improvement
    df['improvement'] = df['best_kge'] - df['initial_kge']
    df['improvement_pct'] = (df['improvement'] / df['initial_kge']) * 100
    
    # Sort by best KGE descending
    df = df.sort_values('best_kge', ascending=False)
    
    return df

def plot_kge_cdfs(df, output_dir=None):
    """Create CDF plot for initial and best KGE values (positive values only)"""
    
    # Filter to positive values only and remove NaN values
    initial_kge_valid = df['initial_kge'].dropna()
    initial_kge_positive = initial_kge_valid[initial_kge_valid > 0]
    
    best_kge_valid = df['best_kge'].dropna()
    best_kge_positive = best_kge_valid[best_kge_valid > 0]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: KGE CDFs
    if len(initial_kge_positive) > 0:
        sorted_initial = np.sort(initial_kge_positive)
        p_initial = np.arange(1, len(sorted_initial) + 1) / len(sorted_initial)
        ax1.plot(sorted_initial, p_initial, 'b-', linewidth=2, label=f'Initial KGE')
    
    if len(best_kge_positive) > 0:
        sorted_best = np.sort(best_kge_positive)
        p_best = np.arange(1, len(sorted_best) + 1) / len(sorted_best)
        ax1.plot(sorted_best, p_best, 'r-', linewidth=2, label=f'Best KGE')
    
    ax1.set_xlabel('KGE Value')
    ax1.set_ylabel('Cumulative Probability')
    ax1.set_title('Cumulative Distribution Functions of KGE Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    
    # Right plot: Completion status
    completion_counts = df['optimization_completed'].value_counts()
    colors = ['lightgreen' if idx else 'lightcoral' for idx in completion_counts.index]
    labels = ['Completed' if idx else 'Incomplete/Timeout' for idx in completion_counts.index]
    
    ax2.pie(completion_counts.values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Optimization Completion Status')
    
    plt.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) / 'kge_analysis_enhanced.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analyze CAMELS spatial run performance metrics')
    parser.add_argument('base_path', help='Path to the camels_spat directory containing domain directories')
    parser.add_argument('--output-dir', help='Directory to save outputs (optional)')
    parser.add_argument('--save-csv', action='store_true', help='Save results to CSV file')
    parser.add_argument('--debug-log', help='Debug a specific log file (provide full path)')
    parser.add_argument('--debug-domain', help='Debug a specific domain ID (e.g., CAN_07BE003)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Debug specific log file
    if args.debug_log:
        debug_log_file(args.debug_log)
        return
    
    # Debug specific domain
    if args.debug_domain:
        base_path = Path(args.base_path)
        domain_dir = base_path / f"domain_{args.debug_domain}_"
        worklog_dir = domain_dir / f"_workLog_{args.debug_domain}_"
        
        if worklog_dir.exists():
            latest_log = find_latest_log_file(str(worklog_dir))
            if latest_log:
                debug_log_file(latest_log)
            else:
                print(f"No log files found for domain {args.debug_domain}")
        else:
            print(f"Worklog directory not found for domain {args.debug_domain}: {worklog_dir}")
        return
    
    try:
        # Analyze runs
        logger.info("Starting CAMELS spatial run analysis...")
        results, invalid_basins = analyze_camels_spat_runs(args.base_path)
        
        if not results:
            logger.error("No valid results found!")
            return
        
        # Create performance table
        df = create_performance_table(results)
        
        # Print summary statistics
        print("\n" + "="*80)
        print("CAMELS SPATIAL RUN PERFORMANCE SUMMARY")
        print("="*80)
        print(f"Total domains analyzed: {len(results)}")
        print(f"Invalid basins: {len(invalid_basins)}")
        
        # Completion status analysis
        completed_count = df['optimization_completed'].sum()
        incomplete_count = len(df) - completed_count
        completion_rate = (completed_count / len(df)) * 100
        
        print(f"\nOptimization Completion Status:")
        print(f"Completed optimizations: {completed_count} ({completion_rate:.1f}%)")
        print(f"Incomplete/timeout optimizations: {incomplete_count} ({100-completion_rate:.1f}%)")
        
        # Generation progress analysis for incomplete runs
        incomplete_df = df[~df['optimization_completed']]
        if len(incomplete_df) > 0:
            valid_progress = incomplete_df['completion_percentage'].dropna()
            if len(valid_progress) > 0:
                print(f"\nIncomplete Run Progress:")
                print(f"Average completion: {valid_progress.mean():.1f}%")
                print(f"Min completion: {valid_progress.min():.1f}%")
                print(f"Max completion: {valid_progress.max():.1f}%")
                
                # Show examples of incomplete runs
                incomplete_examples = incomplete_df[incomplete_df['completion_percentage'].notna()].head(5)
                print(f"\nExamples of incomplete runs:")
                for _, row in incomplete_examples.iterrows():
                    print(f"  {row['domain_id']}: {row['last_generation']}/{row['max_generations']} generations ({row['completion_percentage']:.1f}%)")
        
        # Data completeness analysis
        initial_count = df['initial_kge'].notna().sum()
        best_count = df['best_kge'].notna().sum()
        both_count = (df['initial_kge'].notna() & df['best_kge'].notna()).sum()
        
        print(f"\nData Completeness:")
        print(f"Domains with initial KGE: {initial_count}")
        print(f"Domains with best KGE: {best_count}")
        print(f"Domains with both metrics: {both_count}")
        print(f"Domains with only initial KGE: {initial_count - both_count}")
        print(f"Domains with only best KGE: {best_count - both_count}")
        
        if len(invalid_basins) > 0:
            print(f"\nInvalid basin IDs: {', '.join(invalid_basins)}")
        
        print(f"\nKGE Statistics:")
        if initial_count > 0:
            print(f"Initial KGE - Mean: {df['initial_kge'].mean():.4f}, Std: {df['initial_kge'].std():.4f}, Min: {df['initial_kge'].min():.4f}, Max: {df['initial_kge'].max():.4f}")
        if best_count > 0:
            print(f"Best KGE - Mean: {df['best_kge'].mean():.4f}, Std: {df['best_kge'].std():.4f}, Min: {df['best_kge'].min():.4f}, Max: {df['best_kge'].max():.4f}")
        if both_count > 0:
            improvement_valid = df['improvement'].dropna()
            print(f"Improvement - Mean: {improvement_valid.mean():.4f}, Std: {improvement_valid.std():.4f}, Min: {improvement_valid.min():.4f}, Max: {improvement_valid.max():.4f}")
        
        # Display top 10 performers (based on best KGE)
        print(f"\nTop 10 Best Performing Domains:")
        display_cols = ['domain_id', 'initial_kge', 'best_kge', 'improvement', 'optimization_completed', 'completion_percentage']
        top_performers = df.nlargest(10, 'best_kge')[display_cols]
        print(top_performers.to_string(index=False))
        
        # Show completion status breakdown
        print(f"\nCompletion Status Breakdown:")
        
        # Completed runs
        completed_df = df[df['optimization_completed']]
        if len(completed_df) > 0:
            print(f"Completed runs ({len(completed_df)}): Best performers:")
            completed_top = completed_df.nlargest(3, 'best_kge')[['domain_id', 'best_kge']]
            for _, row in completed_top.iterrows():
                print(f"  {row['domain_id']}: KGE = {row['best_kge']:.4f}")
        
        # Incomplete but progressed runs
        incomplete_with_progress = incomplete_df[incomplete_df['completion_percentage'].notna()]
        if len(incomplete_with_progress) > 0:
            print(f"Incomplete runs with progress ({len(incomplete_with_progress)}): Furthest progressed:")
            progress_top = incomplete_with_progress.nlargest(3, 'completion_percentage')[['domain_id', 'completion_percentage', 'best_kge']]
            for _, row in progress_top.iterrows():
                kge_str = f", KGE = {row['best_kge']:.4f}" if pd.notna(row['best_kge']) else ""
                print(f"  {row['domain_id']}: {row['completion_percentage']:.1f}% complete{kge_str}")
        
        # Show negative KGE values if present
        negative_initial = df[df['initial_kge'] < 0]
        negative_best = df[df['best_kge'] < 0]
        
        if len(negative_initial) > 0:
            print(f"\nDomains with negative initial KGE: {len(negative_initial)}")
            print(f"Most negative initial KGE: {negative_initial['initial_kge'].min():.4f} ({negative_initial.loc[negative_initial['initial_kge'].idxmin(), 'domain_id']})")
        
        if len(negative_best) > 0:
            print(f"Domains with negative best KGE: {len(negative_best)}")
            print(f"Most negative best KGE: {negative_best['best_kge'].min():.4f} ({negative_best.loc[negative_best['best_kge'].idxmin(), 'domain_id']})")
        
        # Save results if requested
        if args.save_csv:
            output_dir = Path(args.output_dir) if args.output_dir else Path('.')
            output_dir.mkdir(exist_ok=True)
            
            csv_path = output_dir / 'camels_spat_performance_enhanced.csv'
            df.to_csv(csv_path, index=False)
            logger.info(f"Results saved to {csv_path}")
            
            # Save completion status summary
            completion_summary = pd.DataFrame({
                'status': ['completed', 'incomplete', 'total'],
                'count': [completed_count, incomplete_count, len(df)],
                'percentage': [completion_rate, 100-completion_rate, 100.0]
            })
            completion_path = output_dir / 'completion_summary.csv'
            completion_summary.to_csv(completion_path, index=False)
            logger.info(f"Completion summary saved to {completion_path}")
            
            # Save invalid basins list
            if invalid_basins:
                invalid_path = output_dir / 'invalid_basins.txt'
                with open(invalid_path, 'w') as f:
                    f.write('\n'.join(invalid_basins))
                logger.info(f"Invalid basins saved to {invalid_path}")
        
        # Create plots
        plot_kge_cdfs(df, args.output_dir)
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
