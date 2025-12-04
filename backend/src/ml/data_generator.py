import scheduler_cpp
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import json
from datetime import datetime


class WorkloadGenerator:
    def __init__(self, random_seed=42):
        random.seed(random_seed)
        np.random.seed(random_seed)

    def generate_cpu_bound_scenario(self, num_processes: int) -> List[Dict]:
        """Generate CPU-intensive workload with high burst times"""
        processes = []

        for i in range(num_processes):
            process = {
                "pid": i + 1,
                "arrival_time": random.randint(0, 10),
                "burst_time": random.randint(50, 200),
            }

            processes.append(process)

        return processes

    def generate_io_bound_scenario(self, num_processes: int) -> List[Dict]:
        """Generate I/O-bound workload with short CPU bursts"""
        processes = []

        for i in range(num_processes):
            process = {
                "pid": i + 1,
                "arrival_time": random.randint(0, 20),
                "burst_time": random.randint(5, 20),
            }
            processes.append(process)

        return processes

    def generate_mixed_scenario(self, num_processes: int) -> List[Dict]:
        """Generate mixed workload combining both patterns (high/low bursts)"""
        processes = []

        for i in range(num_processes):
            if random.random() < 0.5:
                burst_time = random.randint(5, 20)
            else:
                burst_time = random.randint(50, 200)

            process = {
                "pid": i + 1,
                "arrival_time": random.randint(0, 15),
                "burst_time": burst_time,
            }

            processes.append(process)

        return processes

    def run_all_schedulers(self, processes: List[Dict]) -> Dict:
        """Run all schedulers and collect performance metrics"""
        results = {}

        fcfs_results = scheduler_cpp.fcfs_scheduler(processes.copy())
        results["fcfs_results"] = fcfs_results

        sjf_results = scheduler_cpp.sjf_scheduler(processes.copy())
        results["sjf_results"] = sjf_results

        round_robin_results = scheduler_cpp.round_robin_scheduler(
            processes.copy(), time_quantum=4
        )
        results["round_robin_results"] = round_robin_results

        return results

    def calculate_metrics(self, results: Dict[str, List[Dict]]) -> Dict:
        """Calculate performance metrics from scheduler results"""
        metrics = {}

        for algo_name, processes in results.items():
            avg_waiting_time = sum(p["waiting_time"] for p in processes) / len(
                processes
            )
            avg_turnaround_time = sum(p["turn_around_time"] for p in processes) / len(
                processes
            )

            total_completion_time = max(p["finish_time"] for p in processes)

            throughput = (
                len(processes) / total_completion_time
                if total_completion_time > 0
                else 0
            )

            metrics[algo_name] = {
                "avg_waiting_time": avg_waiting_time,
                "avg_turnaround_time": avg_turnaround_time,
                "total_completion_time": total_completion_time,
                "throughput": throughput,
            }

        return metrics

    def generate_training_dataset(
        self, samples_per_scenario: int = 100
    ) -> pd.DataFrame:
        """Generate complete training dataset"""
        dataset = []
        scenarios = ["cpu_bound", "io_bound", "mixed"]

        for i in range(samples_per_scenario):

            for scenario_type in scenarios:
                if scenario_type == "cpu_bound":
                    processes = self.generate_cpu_bound_scenario(random.randint(5, 15))
                elif scenario_type == "io_bound":
                    processes = self.generate_io_bound_scenario(random.randint(5, 15))
                elif scenario_type == "mixed":
                    processes = self.generate_mixed_scenario(random.randint(5, 15))

                results = self.run_all_schedulers(processes)
                metrics = self.calculate_metrics(results)

                total_processes = len(processes)

                sample = {
                    "scenario_type": scenario_type,
                    "num_processes": total_processes,
                    "avg_burst_time": sum(p["burst_time"] for p in processes)
                    / total_processes,
                    "max_burst_time": max(p["burst_time"] for p in processes),
                    "min_burst_time": min(p["burst_time"] for p in processes),
                    "arrival_spread": max(p["arrival_time"] for p in processes)
                    - min(p["arrival_time"] for p in processes),
                }

                for algo_name, algo_metrics in metrics.items():
                    for metric_name, value in algo_metrics.items():
                        sample[f"{algo_name}_{metric_name}"] = value

                dataset.append(sample)

        df = pd.DataFrame(dataset)

        return df
