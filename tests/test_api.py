"""Tests for the FastAPI REST API."""

import pytest
from fastapi.testclient import TestClient

from neurosync.api.server import app, init_app
from neurosync.devices.sensor_interface import MockSensor, SensorConfig


@pytest.fixture(autouse=True)
def setup_app():
    init_app(":memory:")
    yield


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def created_patient(client):
    resp = client.post(
        "/patients",
        json={
            "name": "API Patient",
            "date_of_birth": "1960-01-01",
            "medications": ["Levodopa"],
        },
    )
    assert resp.status_code == 201
    return resp.json()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_create_patient(client):
    resp = client.post(
        "/patients",
        json={
            "name": "New Patient",
            "date_of_birth": "1955-03-15",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "New Patient"
    assert "id" in data


def test_list_patients(client, created_patient):
    resp = client.get("/patients")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_get_patient(client, created_patient):
    pid = created_patient["id"]
    resp = client.get(f"/patients/{pid}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "API Patient"


def test_get_patient_not_found(client):
    resp = client.get("/patients/nonexistent")
    assert resp.status_code == 404


def test_update_patient(client, created_patient):
    pid = created_patient["id"]
    resp = client.put(f"/patients/{pid}", json={"notes": "Updated"})
    assert resp.status_code == 200
    assert resp.json()["notes"] == "Updated"


def test_delete_patient(client, created_patient):
    pid = created_patient["id"]
    resp = client.delete(f"/patients/{pid}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True


def test_create_reading(client, created_patient):
    sensor = MockSensor(seed=42)
    config = SensorConfig(sample_rate_hz=100.0, duration_seconds=5.0)
    signal = sensor.read(config)
    pid = created_patient["id"]

    resp = client.post(
        "/readings",
        json={
            "patient_id": pid,
            "device_id": "MOCK-001",
            "sample_rate_hz": 100.0,
            "duration_seconds": 5.0,
            "raw_signal": signal,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "dominant_frequency_hz" in data
    assert "severity_score" in data


def test_get_readings(client, created_patient):
    pid = created_patient["id"]
    resp = client.get(f"/readings/{pid}")
    assert resp.status_code == 200


def test_list_alerts(client):
    resp = client.get("/alerts")
    assert resp.status_code == 200


def test_compliance_endpoint(client):
    resp = client.get("/compliance")
    assert resp.status_code == 200
    assert "compliant" in resp.json()


def test_audit_endpoint(client):
    resp = client.get("/audit")
    assert resp.status_code == 200
