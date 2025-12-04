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
        """
        Initialize a ModelTrainer instance and set up its persistent state.
        
        Sets:
            model: The trained model object or None if not loaded/trained.
            feature_columns: List of feature column names used for training or None until initialized.
        """
        self.model = None
        self.feature_columns = None

    def create_target_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a target column indicating the best scheduling algorithm per row based on average turnaround times.
        
        Parameters:
            df (pd.DataFrame): Input DataFrame containing the average turnaround time columns
                "fcfs_results_avg_turnaround_time", "sjf_results_avg_turnaround_time", and
                "round_robin_results_avg_turnaround_time".
        
        Returns:
            pd.DataFrame: The same DataFrame with an added "best_algorithms" column whose values
            are `"FCFS"`, `"SJF"`, or `"RR"` corresponding to the algorithm with the smallest
            average turnaround time for each row.
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
        Constructs the feature matrix and target vector from the provided DataFrame using the trainer's predefined feature columns.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing engineered feature columns and a "best_algorithms" column with target labels.
        
        Returns:
            tuple: A pair (X, y) where `X` is a DataFrame with the selected feature columns and `y` is a Series of target algorithm labels from `df["best_algorithms"]`.
        """

        self._set_feature_columns()

        X = df[self.feature_columns]
        y = df["best_algorithms"]

        return (X, y)

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train a RandomForest model on the provided feature matrix and target labels and persist the trained model to disk.
        
        Parameters:
            X (pd.DataFrame): Feature matrix aligned with the trainer's feature_columns.
            y (pd.Series): Target labels indicating the best scheduling algorithm for each sample.
        
        Returns:
            dict: Dictionary containing "accuracy" — the model accuracy on the held-out test set as a float, and "model" — a string identifier for the trained model ("RandomForestClassifier").
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

        """
        Load the persisted RandomForestClassifier into memory, training and persisting a new model if no saved model is available.
        
        Returns:
            RandomForestClassifier: The loaded or newly trained RandomForestClassifier instance.
        """
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
        """
        Generate a new training dataset from a workload generator, prepare features and labels, train a RandomForestClassifier on that data, persist the trained model, and return it.
        
        The method creates synthetic workloads, applies feature engineering and label creation to build a supervised training set, fits a RandomForestClassifier, stores the model on the instance (and on disk), and returns the trained classifier.
        
        Returns:
            RandomForestClassifier: The trained RandomForestClassifier instance stored on the trainer.
        """
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
        Predict the best scheduling algorithm for each provided workload feature set.
        
        The input DataFrame must contain the trainer's required feature columns (in any order); only those columns specified in `self.feature_columns` are used for prediction.
        
        Parameters:
            workload_features (pd.DataFrame): DataFrame of workload feature vectors containing at least the columns listed in `self.feature_columns`.
        
        Returns:
            np.ndarray: Array of predicted algorithm names for each row in `workload_features`.
        
        Raises:
            ValueError: If no model has been trained or loaded into `self.model`.
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