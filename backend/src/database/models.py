from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from sqlalchemy import ForeignKey, String, TIMESTAMP, Float, Integer
from datetime import datetime
from typing import Optional


class Base(DeclarativeBase):
    pass


class Workloads(Base):
    __tablename__ = "workloads"
    id: Mapped[int] = mapped_column(autoincrement=True, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=datetime.now)


class Processes(Base):
    __tablename__ = "processes"
    id: Mapped[int] = mapped_column(autoincrement=True, primary_key=True)
    arrival_time: Mapped[int] = mapped_column(Integer)
    burst_time: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=datetime.now)
    workload_id: Mapped[int] = mapped_column(ForeignKey("workloads.id"))


class Simulations(Base):
    __tablename__ = "simulations"
    id: Mapped[int] = mapped_column(autoincrement=True, primary_key=True)
    algorithm: Mapped[str] = mapped_column(String(10))
    avg_waiting_time: Mapped[float] = mapped_column(Float)
    avg_turnaround_time: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=datetime.now)
    workload_id: Mapped[int] = mapped_column(ForeignKey("workloads.id"))
    prediction_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("predictions.id"), nullable=True
    )


class Predictions(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(autoincrement=True, primary_key=True)
    model_confidence: Mapped[float] = mapped_column(Float)
    predicted_algorithm: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=datetime.now)
    workload_id: Mapped[int] = mapped_column(ForeignKey("workloads.id"), unique=True)
