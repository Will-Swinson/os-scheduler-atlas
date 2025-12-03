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
    model = ModelTrainer()
    app.state.model = model
    app.state.loaded_model = model.load_model()

    yield


app = FastAPI(lifespan=lifespan)


def get_model(request: Request):
    return {
        "model": request.app.state.model,
        "loaded_model": request.app.state.loaded_model,
    }


model_deps = Annotated[ModelTrainer, Depends(get_model)]


def run_scheduler(
    processes: List[Process], algorithm_choice: str, time_quantum: int = 2
) -> Dict:
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
    Convert API process list to ML-ready DataFrame

    Future: Could move to ApiUtils or WorkloadAnalyzer class
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
