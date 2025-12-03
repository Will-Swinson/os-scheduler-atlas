import pandas as pd


class FeatureEngineer:

    def extract_workload_features(_, raw_dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Create ML-ready workload features from a raw scheduler dataset.
        
        Adds these feature columns to a copy of the input DataFrame:
        - burst_time_variance: max_burst_time - min_burst_time
        - process_density: num_processes / (arrival_spread + 1)
        - burst_time_std: standard deviation of avg_burst_time computed per scenario_type
        - is_cpu_intensive: 1 if avg_burst_time > 50 else 0
        - is_io_intensive: 1 if avg_burst_time < 30 else 0
        - is_mixed_workload: 1 if 30 <= avg_burst_time <= 50 else 0
        - total_cpu_demand: num_processes * avg_burst_time
        - arrival_rate: num_processes / (arrival_spread + 1)
        
        Parameters:
            raw_dataset (pd.DataFrame): Input DataFrame containing scheduler fields required for feature construction (e.g., max_burst_time, min_burst_time, num_processes, arrival_spread, avg_burst_time, scenario_type).
        
        Returns:
            pd.DataFrame: A copy of the input DataFrame augmented with the added workload feature columns.
        """
        df = raw_dataset.copy()

        df["burst_time_variance"] = df["max_burst_time"] - df["min_burst_time"]
        df["process_density"] = df["num_processes"] / (df["arrival_spread"] + 1)
        df["burst_time_std"] = df.groupby("scenario_type")["avg_burst_time"].transform(
            "std"
        )
        df["is_cpu_intensive"] = (df["avg_burst_time"] > 50).astype(int)
        df["is_io_intensive"] = (df["avg_burst_time"] < 30).astype(int)
        df["is_mixed_workload"] = (
            (df["avg_burst_time"] >= 30) & (df["avg_burst_time"] <= 50)
        ).astype(int)
        df["total_cpu_demand"] = df["num_processes"] * df["avg_burst_time"]
        df["arrival_rate"] = df["num_processes"] / (df["arrival_spread"] + 1)

        return df

    def add_performance_ratios(_, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ratio features comparing scheduling algorithm performance.
        
        Parameters:
            df (pd.DataFrame): Input DataFrame that must contain the following columns:
                `sjf_results_avg_waiting_time`, `fcfs_results_avg_waiting_time`,
                `sjf_results_throughput`, `fcfs_results_throughput`,
                `round_robin_results_avg_waiting_time`.
        
        Returns:
            pd.DataFrame: The same DataFrame augmented with the following ratio columns:
                `sjf_vs_fcfs_waiting`: ratio of SJF average waiting time to FCFS average waiting time,
                `sjf_vs_fcfs_throughput`: ratio of SJF throughput to FCFS throughput,
                `round_robin_vs_fcfs_waiting`: ratio of Round Robin average waiting time to FCFS average waiting time,
                `round_robin_vs_sjf_waiting`: ratio of Round Robin average waiting time to SJF average waiting time.
        """
        df["sjf_vs_fcfs_waiting"] = (
            df["sjf_results_avg_waiting_time"] / df["fcfs_results_avg_waiting_time"]
        )

        df["sjf_vs_fcfs_throughput"] = (
            df["sjf_results_throughput"] / df["fcfs_results_throughput"]
        )

        df["round_robin_vs_fcfs_waiting"] = (
            df["round_robin_results_avg_waiting_time"]
            / df["fcfs_results_avg_waiting_time"]
        )

        df["round_robin_vs_sjf_waiting"] = (
            df["round_robin_results_avg_waiting_time"]
            / df["sjf_results_avg_waiting_time"]
        )

        return df

    def add_workload_patterns(_, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add workload-pattern features describing balance, homogeneity, and system stress to the dataset.
        
        Adds the following columns to `df`:
        - `workload_balance`: ratio of `min_burst_time` to `max_burst_time`, indicating how balanced job sizes are (0 = unbalanced, 1 = perfectly balanced).
        - `is_homogeneous`: `1` if `burst_time_variance` is less than 20, `0` otherwise.
        - `system_stress`: estimated CPU demand per arrival interval computed from existing demand and spread.
        - `is_high_stress`: `1` if `system_stress` is greater than 100, `0` otherwise.
        
        Parameters:
            df (pd.DataFrame): Scheduler scenario dataset expected to contain `min_burst_time`, `max_burst_time`, `burst_time_variance`, `total_cpu_demand`, and `arrival_spread`.
        
        Returns:
            pd.DataFrame: The input DataFrame augmented with the workload pattern features.
        """

        # 0 = unbalanced , 1 = balanced
        df["workload_balance"] = df["min_burst_time"] / df["max_burst_time"]

        # Has similar job sizes
        df["is_homogeneous"] = (df["burst_time_variance"] < 20).astype(int)

        df["system_stress"] = df["total_cpu_demand"] / (df["arrival_spread"] + 1)
        df["is_high_stress"] = (df["system_stress"] > 100).astype(int)

        return df

    def prepare_ml_dataset(self, raw_dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Compose an ML-ready feature dataset from raw scheduler data.
        
        Parameters:
            raw_dataset (pd.DataFrame): Raw scheduler input containing the columns required by the feature pipeline.
        
        Returns:
            pd.DataFrame: DataFrame augmented with engineered features suitable for machine-learning models.
        """

        df = self.extract_workload_features(raw_dataset)
        df = self.add_performance_ratios(df)
        df = self.add_workload_patterns(df)

        return df