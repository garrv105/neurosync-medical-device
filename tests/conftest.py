"""Shared fixtures for NeuroSync tests."""

from __future__ import annotations

import pytest

from neurosync.core.models import Patient, TremorReading
from neurosync.devices.sensor_interface import MockSensor, SensorConfig
from neurosync.storage.database import Database


@pytest.fixture
def db():
    """In-memory database for testing."""
    database = Database(":memory:")
    _ = database.conn  # Force initialization
    yield database
    database.close()


@pytest.fixture
def sample_patient() -> Patient:
    return Patient(
        id="test-patient-001",
        name="John Doe",
        date_of_birth="1955-03-15",
        diagnosis_date="2020-06-01",
        medications=["Levodopa", "Carbidopa"],
        notes="Stage 2 Parkinson's",
    )


@pytest.fixture
def mock_sensor() -> MockSensor:
    return MockSensor(
        tremor_frequency=5.0,
        tremor_amplitude=1.0,
        noise_level=0.1,
        seed=42,
    )


@pytest.fixture
def sample_signal(mock_sensor) -> list[float]:
    config = SensorConfig(sample_rate_hz=100.0, duration_seconds=10.0)
    return mock_sensor.read(config)


@pytest.fixture
def sample_reading(sample_signal) -> TremorReading:
    return TremorReading(
        id="test-reading-001",
        patient_id="test-patient-001",
        device_id="MOCK-001",
        sample_rate_hz=100.0,
        duration_seconds=10.0,
        raw_signal=sample_signal,
        dominant_frequency_hz=5.0,
        amplitude=1.0,
        severity_score=2.5,
        severity_level="moderate",
    )
