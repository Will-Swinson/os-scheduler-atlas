from fastapi import FastAPI, Request, Depends
from .models import (
    SimulationRequest,
    SimulationResponse,
    PredictionRequest,
    PredictionResponse,
    Process,
)
import scheduler_cpp  # type: ignore
from typing import List, Dict, Annotated
from uuid import uuid4
import pandas as pd
from ..ml.feature_engineer import FeatureEngineer
from ..ml.model_trainer import ModelTrainer
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize and attach a ModelTrainer and its loaded model to the FastAPI app state for the application's lifespan.
    
    Sets app.state.model to a ModelTrainer instance and app.state.loaded_model to the trainer's loaded model, then yields control so the application can continue startup and serve requests.
    """
    model = ModelTrainer()
    app.state.model = model
    app.state.loaded_model = model.load_model()

    yield


app = FastAPI(lifespan=lifespan)


def get_model(request: Request):
    """
    Retrieve the model trainer and the loaded ML model from the application's state for endpoint dependencies.
    
    Returns:
        dict: Mapping with keys "model" (the ModelTrainer instance) and "loaded_model" (the loaded/trained model object).
    """
    return {
        "model": request.app.state.model,
        "loaded_model": request.app.state.loaded_model,
    }


model_deps = Annotated[ModelTrainer, Depends(get_model)]


def run_scheduler(
    processes: List[Process], algorithm_choice: str, time_quantum: int = 2
) -> Dict:
    """
    Run the specified CPU scheduling algorithm on a list of processes.
    
    Parameters:
        processes (List[Process]): Processes to be scheduled.
        algorithm_choice (str): Scheduling algorithm to use; one of "FCFS", "SJF", or "RR".
        time_quantum (int): Time quantum for round-robin scheduling; only used when `algorithm_choice` is "RR".
    
    Returns:
        scheduler_result (Dict): The scheduler's result as a dictionary.
    
    Raises:
        ValueError: If `algorithm_choice` is not one of the supported algorithms.
    """
    process_dicts = [process.model_dump() for process in processes]

    algorithms = {
        "FCFS": scheduler_cpp.fcfs_scheduler,
        "SJF": scheduler_cpp.sjf_scheduler,
        "RR": lambda procs: scheduler_cpp.round_robin_scheduler(procs, time_quantum),
    }

    if algorithm_choice not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm_choice}")

    return algorithms[algorithm_choice](process_dicts)


def analyze_process_workload(processes: List[Process]) -> pd.DataFrame:
    """
    Create a single-row, ML-ready DataFrame of workload features derived from an API list of Process objects.
    
    Parameters:
        processes (List[Process]): List of Process models representing the workload. Must contain at least one process.
    
    Returns:
        pd.DataFrame: A single-row DataFrame containing aggregate workload statistics (e.g., num_processes, avg_burst_time, max_burst_time, min_burst_time, arrival_spread) augmented with engineered workload features and pattern columns produced by FeatureEngineer.
    
    Raises:
        ValueError: If `processes` is None.
    """

    if processes is None:
        raise ValueError("Cannot analyze processes for empty process lists")

    process_dicts = [process.model_dump() for process in processes]

    total_processes = len(process_dicts)
    workload_data = {
        "scenario_type": "api_request",
        "num_processes": total_processes,
        "avg_burst_time": sum(p["burst_time"] for p in process_dicts) / total_processes,
        "max_burst_time": max(p["burst_time"] for p in process_dicts),
        "min_burst_time": min(p["burst_time"] for p in process_dicts),
        "arrival_spread": max(p["arrival_time"] for p in process_dicts)
        - min(p["arrival_time"] for p in process_dicts),
    }

    df = pd.DataFrame([workload_data])
    feature_engineer = FeatureEngineer()

    df_with_features = feature_engineer.extract_workload_features(df)
    workload_features = feature_engineer.add_workload_patterns(df_with_features)
    return workload_features


@app.post("/simulate")
async def simulate(request: SimulationRequest) -> SimulationResponse:
    """
    Run the requested scheduling simulation and return a SimulationResponse with the results.
    
    Parameters:
        request (SimulationRequest): Contains the processes to schedule, the algorithm name, and optional time_quantum for round-robin.
    
    Returns:
        SimulationResponse: Contains a generated `simulation_id`, the `algorithm_used`, and `results` which include:
            - `processes`: scheduler output list,
            - `total_processes`: number of output processes,
            - `simulation_metadata`: dict containing `time_quantum` only when the algorithm is "RR".
    """
    scheduler_output = run_scheduler(
        request.processes, request.algorithm, request.time_quantum
    )
    sim_id = str(uuid4())

    results = {
        "processes": scheduler_output,
        "total_processes": len(scheduler_output),
        "simulation_metadata": {
            "time_quantum": request.time_quantum if request.algorithm == "RR" else None
        },
    }
    return SimulationResponse(
        simulation_id=sim_id, algorithm_used=request.algorithm, results=results
    )


@app.post("/predict")
async def predict(
    model_info: model_deps, request: PredictionRequest
) -> PredictionResponse:
    """
    Predict the best scheduling algorithm for the given processes using the preloaded model.
    
    Parameters:
        request (PredictionRequest): Request containing the list of processes to analyze and predict for.
    
    Returns:
        PredictionResponse: Object with:
            predicted_algorithm (str): The algorithm name selected by the model.
            model_confidence (float): The model's probability for the selected algorithm (0.0–1.0).
            features_used (dict): The workload features derived from the input processes used for prediction.
    """
    workload_features = analyze_process_workload(request.processes)

    model = model_info["model"]
    loaded_model = model_info["loaded_model"]

    predictions = model.predict_best_algorithm(workload_features)

    predicted_algorithm = predictions[0]

    prediction_probs = loaded_model.predict_proba(
        workload_features[model.feature_columns]
    )

    confidence = float(prediction_probs[0].max())

    return PredictionResponse(
        predicted_algorithm=predicted_algorithm,
        model_confidence=confidence,
        features_used=workload_features.to_dict("records")[0],
    )