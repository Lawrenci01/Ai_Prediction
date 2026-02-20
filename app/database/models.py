from sqlalchemy import (
    Column, Integer, Float, String, Text, Enum,
    DateTime, DECIMAL, ForeignKey, UniqueConstraint, Index, TIMESTAMP
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.connection import Base


class Barangay(Base):
    __tablename__ = "barangay"

    barangay_id   = Column(Integer, primary_key=True, autoincrement=True)
    barangay_name = Column(String(50), nullable=False)
    latitude      = Column(DECIMAL(9, 6))
    longitude     = Column(DECIMAL(9, 6))
    city          = Column(String(50), default="Naga City")
    created_at    = Column(TIMESTAMP, server_default=func.now())
    updated_at    = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    sensors = relationship("Sensor", back_populates="barangay")


class Establishment(Base):
    __tablename__ = "establishment"

    establishment_id   = Column(Integer, primary_key=True, autoincrement=True)
    establishment_name = Column(String(100), nullable=False)
    establishment_type = Column(String(50))
    barangay_id        = Column(Integer, ForeignKey("barangay.barangay_id"))
    latitude           = Column(DECIMAL(9, 6))
    longitude          = Column(DECIMAL(9, 6))
    created_at         = Column(TIMESTAMP, server_default=func.now())
    updated_at         = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())


class Sensor(Base):
    __tablename__ = "sensor"

    sensor_id        = Column(Integer, primary_key=True, autoincrement=True)
    sensor_name      = Column(String(50), nullable=False)
    barangay_id      = Column(Integer, ForeignKey("barangay.barangay_id"), nullable=False)
    establishment_id = Column(Integer, ForeignKey("establishment.establishment_id"), nullable=True)
    installed_on     = Column(DateTime, server_default=func.now())
    created_at       = Column(TIMESTAMP, server_default=func.now())
    updated_at       = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    barangay = relationship("Barangay", back_populates="sensors")
    readings = relationship("SensorData", back_populates="sensor")


class SensorData(Base):
    __tablename__ = "sensor_data"

    data_id       = Column(Integer, primary_key=True, autoincrement=True)
    sensor_id     = Column(Integer, ForeignKey("sensor.sensor_id"), nullable=False)
    co2_density   = Column(Float)
    temperature_c = Column(Float)
    humidity      = Column(DECIMAL(5, 2))
    heat_index_c  = Column(Float)
    carbon_level  = Column(Enum("LOW", "NORMAL", "HIGH", "VERY HIGH"))
    recorded_at   = Column(DateTime, server_default=func.now())
    created_at    = Column(TIMESTAMP, server_default=func.now())
    updated_at    = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    minute_stamp  = Column(DateTime, nullable=False)

    sensor = relationship("Sensor", back_populates="readings")

    __table_args__ = (
        UniqueConstraint("sensor_id", "minute_stamp", name="unique_sensor_minute"),
        Index("idx_sensordata_recorded_at", "recorded_at"),
        Index("idx_sensordata_sensor_id", "sensor_id"),
    )


class DailyPrediction(Base):
    __tablename__ = "daily_predictions"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    prediction_date = Column(DateTime, nullable=False)
    run_at          = Column(DateTime, server_default=func.now())
    sensor_id       = Column(Integer, ForeignKey("sensor.sensor_id"), nullable=True)
    target          = Column(String(32), nullable=False)
    unit            = Column(String(16), nullable=True)
    mean_value      = Column(Float, nullable=True)
    min_value       = Column(Float, nullable=True)
    max_value       = Column(Float, nullable=True)
    model_used      = Column(String(32), default="LSTM")

    hourly_values = relationship("HourlyPrediction", back_populates="daily", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("prediction_date", "target", "sensor_id", name="uq_daily_pred_date_target_sensor"),
        Index("idx_daily_pred_date", "prediction_date"),
    )


class HourlyPrediction(Base):
    __tablename__ = "hourly_predictions"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    daily_id        = Column(Integer, ForeignKey("daily_predictions.id", ondelete="CASCADE"), nullable=False)
    hour            = Column(Integer, nullable=False)
    timestamp       = Column(DateTime, nullable=False)
    predicted_value = Column(Float, nullable=True)

    daily = relationship("DailyPrediction", back_populates="hourly_values")

    __table_args__ = (
        Index("idx_hourly_daily_id", "daily_id"),
        Index("idx_hourly_timestamp", "timestamp"),
    )


class PredictionInsight(Base):
    __tablename__ = "prediction_insights"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    prediction_date = Column(DateTime, nullable=False)
    run_at          = Column(DateTime, server_default=func.now())
    sensor_id       = Column(Integer, ForeignKey("sensor.sensor_id"), nullable=True)
    barangay        = Column(String(128), nullable=True)
    insight_text    = Column(Text, nullable=True)
    llm_backend     = Column(String(32), nullable=True)

    __table_args__ = (
        Index("idx_insight_date", "prediction_date"),
    )