#!/usr/bin/env python3
import sys
import pickle
import os
from pathlib import Path
from mpi4py import MPI
import logging

# Setup logging for MPI debugging - CAPTURE ALL DEBUG MESSAGES
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG
    format='[MPI-%(process)d] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Ensure output goes to stdout
        logging.FileHandler(f'/tmp/mpi_worker_debug_{os.getpid()}.log')  # Also log to file
    ]
)
logger = logging.getLogger(__name__)

# Add CONFLUENCE path
sys.path.append(r"/Users/darrieythorsson/compHydro/code/CONFLUENCE/utils/optimization")

# Import the worker function
from iterative_optimizer import _evaluate_parameters_worker_safe

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    logger.info(f"MPI worker rank {rank}/{size} starting")
    
    tasks_file = Path(sys.argv[1])
    results_file = Path(sys.argv[2])
    
    if rank == 0:
        # Load tasks
        logger.info(f"Rank 0: Loading tasks from {tasks_file}")
        with open(tasks_file, 'rb') as f:
            all_tasks = pickle.load(f)
        
        logger.info(f"Rank 0: Loaded {len(all_tasks)} tasks")
        
        # 🎯 DEBUG: Check first task
        if all_tasks:
            first_task = all_tasks[0]
            logger.info(f"🎯 RANK 0 - First task multiobjective: {first_task.get('multiobjective', 'MISSING')}")
            logger.info(f"🎯 RANK 0 - First task target_metric: {first_task.get('target_metric', 'MISSING')}")
            logger.info(f"🎯 RANK 0 - First task keys: {list(first_task.keys())}")
        
        # Distribute tasks
        tasks_per_rank = len(all_tasks) // size
        extra_tasks = len(all_tasks) % size
        all_results = []
        
        for worker_rank in range(size):
            start_idx = worker_rank * tasks_per_rank + min(worker_rank, extra_tasks)
            end_idx = start_idx + tasks_per_rank + (1 if worker_rank < extra_tasks else 0)
            
            if worker_rank == 0:
                my_tasks = all_tasks[start_idx:end_idx]
                logger.info(f"Rank 0: Processing {len(my_tasks)} tasks locally")
            else:
                worker_tasks = all_tasks[start_idx:end_idx]
                logger.info(f"Rank 0: Sending {len(worker_tasks)} tasks to rank {worker_rank}")
                comm.send(worker_tasks, dest=worker_rank, tag=1)
        
        # Process rank 0 tasks
        for i, task in enumerate(my_tasks):
            logger.info(f"Rank 0: Processing task {i+1}/{len(my_tasks)}")
            logger.info(f"🎯 RANK 0 TASK {i} - multiobjective: {task.get('multiobjective')}")
            logger.info(f"🎯 RANK 0 TASK {i} - target_metric: {task.get('target_metric')}")
            
            try:
                worker_result = _evaluate_parameters_worker_safe(task)
                
                # 🎯 DEBUG: Log what the worker returned
                logger.info(f"🎯 RANK 0 WORKER RESULT {i}:")
                logger.info(f"🎯   score: {worker_result.get('score')}")
                logger.info(f"🎯   objectives: {worker_result.get('objectives')}")
                logger.info(f"🎯   error: {worker_result.get('error')}")
                logger.info(f"🎯   all keys: {list(worker_result.keys())}")
                
                # 🎯 PRESERVE ALL FIELDS - DO NOT FILTER
                all_results.append(worker_result)
                
            except Exception as e:
                logger.error(f"Rank 0: Task {i} failed: {e}")
                error_result = {
                    'individual_id': task.get('individual_id', -1),
                    'params': task.get('params', {}),
                    'score': None,
                    'objectives': None,
                    'error': f'Rank 0 error: {str(e)}'
                }
                all_results.append(error_result)
        
        # Collect from workers
        for worker_rank in range(1, size):
            try:
                logger.info(f"Rank 0: Waiting for results from rank {worker_rank}")
                worker_results = comm.recv(source=worker_rank, tag=2)
                logger.info(f"Rank 0: Received {len(worker_results)} results from rank {worker_rank}")
                
                # 🎯 DEBUG: Log first result from worker
                if worker_results:
                    first_result = worker_results[0]
                    logger.info(f"🎯 RANK 0 RECEIVED FROM {worker_rank}:")
                    logger.info(f"🎯   score: {first_result.get('score')}")
                    logger.info(f"🎯   objectives: {first_result.get('objectives')}")
                    logger.info(f"🎯   error: {first_result.get('error')}")
                    logger.info(f"🎯   all keys: {list(first_result.keys())}")
                
                all_results.extend(worker_results)
                
            except Exception as e:
                logger.error(f"Error receiving from worker {worker_rank}: {e}")
        
        # 🎯 DEBUG: Log final results before saving
        logger.info(f"🎯 RANK 0 FINAL RESULTS: {len(all_results)} total")
        if all_results:
            first_final = all_results[0]
            logger.info(f"🎯 FINAL RESULT 0:")
            logger.info(f"🎯   score: {first_final.get('score')}")
            logger.info(f"🎯   objectives: {first_final.get('objectives')}")
            logger.info(f"🎯   error: {first_final.get('error')}")
        
        # Save results
        logger.info(f"Rank 0: Saving {len(all_results)} results to {results_file}")
        with open(results_file, 'wb') as f:
            pickle.dump(all_results, f)
        logger.info(f"Rank 0: Results saved successfully")
    
    else:
        # Worker process
        logger.info(f"Rank {rank}: Waiting for tasks from rank 0")
        try:
            my_tasks = comm.recv(source=0, tag=1)
            logger.info(f"Rank {rank}: Received {len(my_tasks)} tasks")
            
            # 🎯 DEBUG: Check first received task
            if my_tasks:
                first_task = my_tasks[0]
                logger.info(f"🎯 RANK {rank} RECEIVED TASK - multiobjective: {first_task.get('multiobjective')}")
                logger.info(f"🎯 RANK {rank} RECEIVED TASK - target_metric: {first_task.get('target_metric')}")
            
            my_results = []
            
            for i, task in enumerate(my_tasks):
                logger.info(f"Rank {rank}: Processing task {i+1}/{len(my_tasks)}")
                logger.info(f"🎯 RANK {rank} TASK {i} - multiobjective: {task.get('multiobjective')}")
                
                try:
                    worker_result = _evaluate_parameters_worker_safe(task)
                    
                    # 🎯 DEBUG: Log what the worker returned
                    logger.info(f"🎯 RANK {rank} WORKER RESULT {i}:")
                    logger.info(f"🎯   score: {worker_result.get('score')}")
                    logger.info(f"🎯   objectives: {worker_result.get('objectives')}")
                    logger.info(f"🎯   error: {worker_result.get('error')}")
                    
                    # 🎯 PRESERVE ALL FIELDS - DO NOT FILTER
                    my_results.append(worker_result)
                    
                except Exception as e:
                    logger.error(f"Rank {rank}: Task {i} failed: {e}")
                    error_result = {
                        'individual_id': task.get('individual_id', -1),
                        'params': task.get('params', {}),
                        'score': None,
                        'objectives': None,
                        'error': f'Rank {rank} error: {str(e)}'
                    }
                    my_results.append(error_result)
            
            logger.info(f"Rank {rank}: Sending {len(my_results)} results back to rank 0")
            
            # 🎯 DEBUG: Log what we're sending back
            if my_results:
                first_result = my_results[0]
                logger.info(f"🎯 RANK {rank} SENDING BACK:")
                logger.info(f"🎯   score: {first_result.get('score')}")
                logger.info(f"🎯   objectives: {first_result.get('objectives')}")
                logger.info(f"🎯   error: {first_result.get('error')}")
            
            comm.send(my_results, dest=0, tag=2)
            logger.info(f"Rank {rank}: Results sent successfully")
            
        except Exception as e:
            logger.error(f"Worker {rank} failed: {e}")

if __name__ == "__main__":
    main()
