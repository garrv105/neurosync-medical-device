"""Pydantic models for the NeuroSync medical device monitoring system."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SeverityLevel(str, Enum):
    NONE = "none"
    SLIGHT = "slight"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class AlertPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class DeviceState(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    CALIBRATING = "calibrating"
    ERROR = "error"


class Patient(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    date_of_birth: str
    diagnosis_date: Optional[str] = None
    medications: list[str] = Field(default_factory=list)
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TremorReading(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    device_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sample_rate_hz: float = 100.0
    duration_seconds: float = 10.0
    raw_signal: list[float]
    dominant_frequency_hz: Optional[float] = None
    amplitude: Optional[float] = None
    severity_score: Optional[float] = None
    severity_level: Optional[SeverityLevel] = None


class Alert(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    reading_id: Optional[str] = None
    priority: AlertPriority
    status: AlertStatus = AlertStatus.ACTIVE
    message: str
    details: Optional[dict] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


class DeviceStatus(BaseModel):
    device_id: str
    state: DeviceState = DeviceState.ONLINE
    battery_percent: Optional[float] = None
    firmware_version: Optional[str] = None
    last_calibration: Optional[datetime] = None
    last_reading: Optional[datetime] = None
    drift_detected: bool = False
    error_message: Optional[str] = None


class TremorAnalysisResult(BaseModel):
    dominant_frequency_hz: float
    power_spectral_density_peak: float
    tremor_band_power: float
    total_power: float
    tremor_ratio: float
    amplitude_rms: float
    frequencies: list[float] = Field(default_factory=list)
    psd_values: list[float] = Field(default_factory=list)


class SeverityAssessment(BaseModel):
    score: float = Field(ge=0, le=4)
    level: SeverityLevel
    subscores: dict[str, float] = Field(default_factory=dict)
    clinical_notes: list[str] = Field(default_factory=list)


class ClinicalReport(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    period_start: datetime
    period_end: datetime
    total_readings: int
    average_severity: float
    severity_trend: str
    alerts_summary: dict = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)


class AuditLogEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    action: str
    user_id: Optional[str] = None
    resource_type: str
    resource_id: str
    details: Optional[str] = None
    checksum: Optional[str] = None
