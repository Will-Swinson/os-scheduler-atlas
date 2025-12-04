import scheduler_cpp  # type: ignore
import random
import numpy as np
import pandas as pd
from typing import List, Dict


class WorkloadGenerator:
    def __init__(self, random_seed=42):
        """
        Initialize the WorkloadGenerator with a fixed random seed to produce deterministic workloads.

        Parameters:
            random_seed (int): Seed value used to initialize both Python's `random` and NumPy's random number generators.
        """
        random.seed(random_seed)
        np.random.seed(random_seed)

    def generate_cpu_bound_scenario(self, num_processes: int) -> List[Dict]:
        """
        Generate a CPU-bound process workload.

        Each process is represented as a dictionary with keys:
        - "pid": process identifier starting at 1
        - "arrival_time": integer arrival time sampled uniformly from 0 to 10 (inclusive)
        - "burst_time": integer CPU burst time sampled uniformly from 50 to 200 (inclusive)

        Parameters:
            num_processes (int): Number of process dictionaries to generate.

        Returns:
            List[Dict]: A list of generated process dictionaries as described above.
        """
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
        """
        Generate a list of I/O-bound process descriptors with short CPU bursts.

        Returns:
            processes (List[Dict]): List of process dictionaries. Each dictionary contains:
                - pid (int): Process identifier starting at 1.
                - arrival_time (int): Arrival time (0–20).
                - burst_time (int): Short CPU burst duration (5–20).
        """
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
        """
        Generate a mixed workload of processes with either short or long CPU bursts.

        Each returned process is a dict containing:
        - "pid" (int): process identifier starting at 1
        - "arrival_time" (int): arrival time in the range 0–15
        - "burst_time" (int): CPU burst length; about half are short (5–20) and half are long (50–200)

        Parameters:
            num_processes (int): Number of processes to generate.

        Returns:
            List[Dict]: List of process dictionaries as described above.
        """
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
        """
        Run FCFS, SJF, and Round Robin schedulers on copies of the provided processes and collect each scheduler's results.

        Parameters:
            processes (List[Dict]): List of process dictionaries. Each process dictionary is expected to include at least
                the keys `pid`, `arrival_time`, and `burst_time`.

        Returns:
            results (Dict): Mapping with keys `"fcfs_results"`, `"sjf_results"`, and `"round_robin_results"` whose values
            are the respective scheduler outputs (lists or structures returned by the scheduler implementations).
        """
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
        """
        Compute aggregate scheduling performance metrics for each algorithm.

        Parameters:
            results (Dict[str, List[Dict]]): Mapping from scheduler name to its list of process records.
                Each process record must contain the numeric fields:
                - "waiting_time": time the process spent waiting,
                - "turn_around_time": total time from arrival to completion,
                - "finish_time": completion time of the process.

        Returns:
            Dict[str, Dict]: Mapping from scheduler name to a metrics dictionary with keys:
                - "avg_waiting_time": average of the processes' waiting_time,
                - "avg_turnaround_time": average of the processes' turn_around_time,
                - "total_completion_time": maximum finish_time across processes,
                - "throughput": number of processes divided by total_completion_time (`0` if total_completion_time <= 0).
        """
        metrics = {}

        for algo_name, processes in results.items():
            if not processes:
                continue
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
        """
        Generate a tabular training dataset of scheduler performance across workload scenarios.

        For each of the scenario types ("cpu_bound", "io_bound", "mixed") this method generates a random-sized process list, runs all configured schedulers on that list, computes aggregate burst/arrival statistics and per-algorithm metrics, and collects the results as one sample row. Repeats this for `samples_per_scenario` iterations.

        Parameters:
            samples_per_scenario (int): Number of independent samples to generate for each scenario type.

        Returns:
            pd.DataFrame: A DataFrame where each row is a single sample and columns include:
                - scenario_type: scenario label ("cpu_bound", "io_bound", or "mixed")
                - num_processes: number of processes in the sample
                - avg_burst_time, max_burst_time, min_burst_time: aggregate burst-time statistics
                - arrival_spread: max arrival_time minus min arrival_time
                - Per-algorithm metrics added as columns named "<algo_name>_<metric_name>" (e.g. "fcfs_avg_waiting_time").
        """
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
