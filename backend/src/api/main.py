from fastapi import FastAPI, Depends, status, HTTPException
from .models import (
    SimulationRequest,
    PredictionSimulationRequest,
    SimulationResponse,
    PredictionRequest,
    PredictionResponse,
    Process,
)
from sqlalchemy.orm import Session
from pydantic import Discriminator, Tag
import scheduler_cpp  # type: ignore
from typing import List, Dict, Union, Annotated
import pandas as pd
from ..ml.feature_engineer import FeatureEngineer
from ..ml.model_trainer import ModelTrainer
from ..database.connection import engine, SessionLocal
from ..database.models import Base
from ..database.queries import (
    soft_create_workload,
    soft_create_processes,
    soft_create_simulation,
    get_processes_by_prediction,
    get_prediction_by_id,
    soft_create_prediction,
)
from ..services.simulation_service import calculate_avg_metrics
from .validators import get_simulate_request_type, to_dict

app = FastAPI()

Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


db_deps = Annotated[Session, Depends(get_db)]


def run_scheduler(
    processes: List[Process], algorithm_choice: str, time_quantum: int = 2
) -> Dict:
    process_dicts = [to_dict(process) for process in processes]


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


SimulateRequest = Annotated[
    Union[
        Annotated[SimulationRequest, Tag("normal")],
        Annotated[PredictionSimulationRequest, Tag("prediction")],
    ],
    Discriminator(get_simulate_request_type),
]


@app.post("/simulate", status_code=status.HTTP_201_CREATED)
async def simulate(
    db: db_deps,
    request: SimulateRequest,
) -> SimulationResponse:
    try:
        if isinstance(request, PredictionSimulationRequest):
            prediction = get_prediction_by_id(request.prediction_id, db)
            processes = get_processes_by_prediction(prediction, db)
            algorithm = request.algorithm or prediction.predicted_algorithm

        else:
            processes = request.processes
            algorithm = request.algorithm

        time_quantum = request.time_quantum if request.time_quantum is not None else 2

        scheduler_output = run_scheduler(processes, algorithm, time_quantum)

        print(scheduler_output)
        results = {
            "processes": scheduler_output,
            "total_processes": len(scheduler_output),
            "simulation_metadata": {
                "time_quantum": time_quantum if algorithm == "RR" else None
            },
        }

        avg_waiting_time, avg_turnaround_time = calculate_avg_metrics(scheduler_output)

        if isinstance(request, PredictionSimulationRequest):
            simulation = soft_create_simulation(
                algorithm=algorithm,
                workload_id=prediction.workload_id,
                avg_waiting_time=avg_waiting_time,
                avg_turnaround_time=avg_turnaround_time,
                db=db,
                prediction_id=prediction.id,
            )
        else:
            workload = soft_create_workload(db)

            soft_create_processes(request, workload, db)

            simulation = soft_create_simulation(
                algorithm=algorithm,
                workload_id=workload.id,
                avg_waiting_time=avg_waiting_time,
                avg_turnaround_time=avg_turnaround_time,
                db=db,
            )

        db.commit()

        return SimulationResponse(
            simulation_id=simulation.id, algorithm_used=algorithm, results=results
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation failed: {str(e)}",
        )


@app.post("/predict", status_code=status.HTTP_201_CREATED)
async def predict(db: db_deps, request: PredictionRequest) -> PredictionResponse:
    try:

        workload_features = analyze_process_workload(request.processes)

        model = ModelTrainer()
        loaded_model = model.load_model()

        predictions = model.predict_best_algorithm(workload_features)

        predicted_algorithm = predictions[0]

        prediction_probs = loaded_model.predict_proba(
            workload_features[model.feature_columns]
        )

        confidence = float(prediction_probs[0].max())

        workload = soft_create_workload(db)

        soft_create_processes(request, workload, db)

        prediction = soft_create_prediction(
            predicted_algorithm, confidence, workload, db
        )

        db.commit()

        return PredictionResponse(
            prediction_id=prediction.id,
            predicted_algorithm=predicted_algorithm,
            model_confidence=confidence,
            features_used=workload_features.to_dict("records")[0],
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )
