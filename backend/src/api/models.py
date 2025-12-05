from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum


class Algorithm(str, Enum):
    FCFS = "FCFS"
    SJF = "SJF"
    RR = "RR"


class Process(BaseModel):
    pid: int = Field(..., ge=1, description="Process ID (Must be >= 1)")
    arrival_time: int = Field(..., ge=0, description="When the process arrives")
    burst_time: int = Field(
        ..., gt=0, description="Processing time needed for the process"
    )


class SimulationRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "example": {
                "processes": [{"pid": 1, "arrival_time": 0, "burst_time": 5}],
                "algorithm": "FCFS",
                "time_quantum": 2,
            }
        }
    }
    processes: List[Process]
    algorithm: Algorithm
    time_quantum: Optional[int] = Field(
        default=None, gt=0, description="Time slice for Round Robin"
    )


class PredictionSimulationRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction_id": 123,
                "algorithm": "FCFS",
                "time_quantum": 2,
            }
        }
    }

    algorithm: Optional[Algorithm] = None
    prediction_id: int
    time_quantum: Optional[int] = Field(
        default=None, gt=0, description="Time slice for Round Robin"
    )


class SimulationResponse(BaseModel):
    simulation_id: int
    algorithm_used: Algorithm
    results: Dict[str, Any]


class PredictionRequest(BaseModel):
    processes: List[Process]


class PredictionResponse(BaseModel):
    prediction_id: int
    predicted_algorithm: Algorithm
    model_confidence: float = Field(..., ge=0, description="Model confidence")
    features_used: Dict[str, Any] = Field(
        description="Feature labels used for the prediction"
    )
