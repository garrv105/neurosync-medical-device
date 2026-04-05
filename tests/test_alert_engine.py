"""Tests for the alert engine."""

import pytest

from neurosync.core.alert_engine import AlertEngine
from neurosync.core.models import AlertPriority, AlertStatus, TremorReading


@pytest.fixture
def alert_engine(db):
    from neurosync.core.models import Patient

    db.create_patient(Patient(id="test-patient-001", name="Test", date_of_birth="1970-01-01"))
    return AlertEngine(db)


def _make_reading(severity: float, frequency: float = 5.0) -> TremorReading:
    return TremorReading(
        patient_id="test-patient-001",
        device_id="MOCK-001",
        raw_signal=[0.0] * 100,
        severity_score=severity,
        severity_level="moderate",
        dominant_frequency_hz=frequency,
    )


def test_no_alert_for_low_severity(alert_engine):
    reading = _make_reading(severity=0.5)
    alerts = alert_engine.evaluate_reading(reading)
    assert len(alerts) == 0


def test_low_priority_alert(alert_engine):
    reading = _make_reading(severity=1.5)
    alerts = alert_engine.evaluate_reading(reading)
    severity_alerts = [
        a for a in alerts if a.details and a.details.get("rule") != "abnormal_frequency"
    ]
    assert len(severity_alerts) == 1
    assert severity_alerts[0].priority == AlertPriority.LOW


def test_high_priority_alert(alert_engine):
    reading = _make_reading(severity=3.2)
    alerts = alert_engine.evaluate_reading(reading)
    severity_alerts = [a for a in alerts if a.details and a.details.get("rule") == "severe_tremor"]
    assert len(severity_alerts) == 1
    assert severity_alerts[0].priority == AlertPriority.HIGH


def test_critical_alert(alert_engine):
    reading = _make_reading(severity=3.8)
    alerts = alert_engine.evaluate_reading(reading)
    severity_alerts = [
        a for a in alerts if a.details and a.details.get("rule") == "critical_tremor"
    ]
    assert len(severity_alerts) == 1
    assert severity_alerts[0].priority == AlertPriority.CRITICAL


def test_abnormal_frequency_alert(alert_engine):
    reading = _make_reading(severity=1.0, frequency=12.0)
    alerts = alert_engine.evaluate_reading(reading)
    freq_alerts = [a for a in alerts if a.details and a.details.get("rule") == "abnormal_frequency"]
    assert len(freq_alerts) == 1


def test_acknowledge_alert(alert_engine, db):
    reading = _make_reading(severity=2.5)
    alerts = alert_engine.evaluate_reading(reading)
    assert len(alerts) > 0
    ack = alert_engine.acknowledge_alert(alerts[0].id)
    assert ack is not None
    assert ack.status == AlertStatus.ACKNOWLEDGED


def test_resolve_alert(alert_engine, db):
    reading = _make_reading(severity=2.5)
    alerts = alert_engine.evaluate_reading(reading)
    resolved = alert_engine.resolve_alert(alerts[0].id)
    assert resolved is not None
    assert resolved.status == AlertStatus.RESOLVED
