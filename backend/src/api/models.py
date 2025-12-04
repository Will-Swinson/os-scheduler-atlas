from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
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
    processes: List[Process]
    algorithm: Algorithm
    time_quantum: Optional[int] = Field(
        default=2, gt=0, description="Time slice for Round Robin"
    )


class SimulationResponse(BaseModel):
    simulation_id: str
    algorithm_used: Algorithm
    results: Dict[str, Any]


class PredictionRequest(BaseModel):
    processes: List[Process]


class PredictionResponse(BaseModel):
    predicted_algorithm: Algorithm
    model_confidence: float = Field(..., ge=0, description="Model confidence")
    features_used: Dict[str, Any] = Field(
        description="Feature labels used for the prediction"
    )
