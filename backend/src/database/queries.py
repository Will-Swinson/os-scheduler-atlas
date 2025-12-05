from .models import Workloads, Processes, Simulations, Predictions
from sqlalchemy.orm import Session


def soft_create_workload(db: Session):
    workload = Workloads()
    db.add(workload)
    db.flush()

    return workload


def soft_create_processes(request, workload, db: Session):
    process_records = [
        Processes(
            arrival_time=p.arrival_time,
            burst_time=p.burst_time,
            workload_id=workload.id,
        )
        for p in request.processes
    ]

    db.add_all(process_records)


def soft_create_simulation(
    algorithm,
    workload_id,
    avg_waiting_time,
    avg_turnaround_time,
    db: Session,
    prediction_id=None,
):
    simulation = Simulations(
        algorithm=algorithm,
        workload_id=workload_id,
        prediction_id=prediction_id,
        avg_waiting_time=avg_waiting_time,
        avg_turnaround_time=avg_turnaround_time,
    )

    db.add(simulation)
    db.flush()

    return simulation


def get_prediction_by_id(prediction_id, db: Session):
    prediction = db.query(Predictions).filter(Predictions.id == prediction_id).first()

    return prediction


def get_processes_by_prediction(prediction, db: Session):

    processes = (
        db.query(Processes)
        .filter(Processes.workload_id == prediction.workload_id)
        .all()
    )

    return processes


def soft_create_prediction(predicted_algorithm, confidence, workload, db: Session):
    prediction = Predictions(
        predicted_algorithm=predicted_algorithm,
        model_confidence=confidence,
        workload_id=workload.id,
    )

    db.add(prediction)
    db.flush()

    return prediction
