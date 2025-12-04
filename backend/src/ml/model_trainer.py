import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from typing import Tuple, Dict
from .feature_engineer import FeatureEngineer
from .data_generator import WorkloadGenerator


class ModelTrainer:
    def __init__(self):
        self.model = None
        self.feature_columns = None

    def create_target_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the target label for the model to understand patterns and learn what we are trying to achieve
        """
        turnaround_cols = [
            "fcfs_results_avg_turnaround_time",
            "sjf_results_avg_turnaround_time",
            "round_robin_results_avg_turnaround_time",
        ]

        best_algorithm_cols = df[turnaround_cols].idxmin(axis=1)

        clean_names = best_algorithm_cols.str.replace(
            "_results_avg_turnaround_time", ""
        )

        clean_names = clean_names.str.replace("round_robin", "RR")
        clean_names = clean_names.str.replace("sjf", "SJF")
        clean_names = clean_names.str.replace("fcfs", "FCFS")

        df["best_algorithms"] = clean_names

        return df

    def prepare_training_dataset(self, df: pd.DataFrame) -> Tuple:
        """
        Prepare features (X) and target (y) for ML training
        """

        self._set_feature_columns()

        X = df[self.feature_columns]
        y = df["best_algorithms"]

        return (X, y)

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train ML Model to predict best scheduling algorithm

        Args:
            X: Feature matrix from prepare_training_dataset()
            y: Target labels from prepare_training_dataset()

        Returns:
            Dict with training results and metrics
        """

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42
        )

        model = RandomForestClassifier()

        model.fit(X_train, y_train)
        joblib.dump(model, "trained_model.pkl")
        self.model = model

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        return {"accuracy": accuracy, "model": "RandomForestClassifier"}

    def load_model(self) -> RandomForestClassifier:

        if self.model is None:
            try:
                self.model = joblib.load("trained_model.pkl")
                # Ensure feature_columns are set when loading existing model
                if self.feature_columns is None:
                    self._set_feature_columns()
            except FileNotFoundError:
                self.train_new_model()

        return self.model

    def _set_feature_columns(self):
        """Set the feature columns used by the model"""
        self.feature_columns = [
            "num_processes",
            "avg_burst_time",
            "min_burst_time",
            "max_burst_time",
            "arrival_spread",
            "burst_time_variance",
            "process_density",
            "total_cpu_demand",
            "arrival_rate",
            "is_cpu_intensive",
            "is_io_intensive",
            "is_mixed_workload",
            "workload_balance",
            "is_homogeneous",
            "system_stress",
            "is_high_stress",
        ]

    def train_new_model(self) -> RandomForestClassifier:
        generator = WorkloadGenerator()
        training_data = generator.generate_training_dataset(10000)

        feature_engineer = FeatureEngineer()
        training_features = feature_engineer.prepare_ml_dataset(training_data)

        dataset = self.create_target_labels(training_features)

        (X, y) = self.prepare_training_dataset(dataset)

        self.train_model(X, y)

        return self.model

    def predict_best_algorithm(self, workload_features: pd.DataFrame) -> np.ndarray:
        """
        Predict best algorithm for new workload scenarios

        Args:
            workload_features: DataFrame with workload characteristics

        Returns:
            Array of predicted algorithm names
        """

        if self.model is None:
            raise ValueError(
                "Model must be trained first. Call train_model() before making predictions."
            )

        if self.feature_columns is None:
            self._set_feature_columns()

        X = workload_features[self.feature_columns]

        predictions = self.model.predict(X)

        return predictions
