import pandas as pd


class FeatureEngineer:

    def extract_workload_features(_, raw_dataset: pd.DataFrame) -> pd.DataFrame:
        """Transform raw scheduler data into ML ready features"""
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
        """Calculate performance ratios between algorithms"""
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
        """Extract workload characteristic patterns"""

        # 0 = unbalanced , 1 = balanced
        df["workload_balance"] = df["min_burst_time"] / df["max_burst_time"]

        # Has similar job sizes
        df["is_homogeneous"] = (df["burst_time_variance"] < 20).astype(int)

        df["system_stress"] = df["total_cpu_demand"] / (df["arrival_spread"] + 1)
        df["is_high_stress"] = (df["system_stress"] > 100).astype(int)

        return df

    def prepare_ml_dataset(self, raw_dataset: pd.DataFrame) -> pd.DataFrame:
        """Complete pipeline: raw data -> ML features"""

        df = self.extract_workload_features(raw_dataset)
        df = self.add_performance_ratios(df)
        df = self.add_workload_patterns(df)

        return df
