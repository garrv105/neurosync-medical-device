"""Tests for the SQLite database layer."""

from neurosync.core.models import Alert, AlertPriority, AlertStatus, Patient


def test_create_and_get_patient(db):
    patient = Patient(name="Test User", date_of_birth="1970-01-01")
    db.create_patient(patient)
    fetched = db.get_patient(patient.id)
    assert fetched is not None
    assert fetched.name == "Test User"


def test_list_patients(db):
    db.create_patient(Patient(name="A", date_of_birth="1970-01-01"))
    db.create_patient(Patient(name="B", date_of_birth="1975-01-01"))
    patients = db.list_patients()
    assert len(patients) == 2


def test_update_patient(db):
    patient = Patient(name="Original", date_of_birth="1970-01-01")
    db.create_patient(patient)
    patient.name = "Updated"
    db.update_patient(patient)
    fetched = db.get_patient(patient.id)
    assert fetched.name == "Updated"


def test_delete_patient(db):
    patient = Patient(name="ToDelete", date_of_birth="1970-01-01")
    db.create_patient(patient)
    assert db.delete_patient(patient.id) is True
    assert db.get_patient(patient.id) is None


def test_delete_nonexistent_patient(db):
    assert db.delete_patient("nonexistent") is False


def test_store_and_get_reading(db, sample_reading):
    patient = Patient(id="test-patient-001", name="P", date_of_birth="1970-01-01")
    db.create_patient(patient)
    db.store_reading(sample_reading)
    readings = db.get_readings("test-patient-001")
    assert len(readings) == 1
    assert readings[0].id == sample_reading.id


def test_get_reading_by_id(db, sample_reading):
    patient = Patient(id="test-patient-001", name="P", date_of_birth="1970-01-01")
    db.create_patient(patient)
    db.store_reading(sample_reading)
    reading = db.get_reading(sample_reading.id)
    assert reading is not None
    assert reading.patient_id == "test-patient-001"


def test_store_and_get_alert(db):
    db.create_patient(Patient(id="p1", name="P", date_of_birth="1970-01-01"))
    alert = Alert(
        patient_id="p1",
        priority=AlertPriority.HIGH,
        message="Test alert",
    )
    db.store_alert(alert)
    alerts = db.get_alerts(patient_id="p1")
    assert len(alerts) == 1
    assert alerts[0].message == "Test alert"


def test_update_alert_status(db):
    db.create_patient(Patient(id="p1", name="P", date_of_birth="1970-01-01"))
    alert = Alert(patient_id="p1", priority=AlertPriority.HIGH, message="Test")
    db.store_alert(alert)
    updated = db.update_alert_status(alert.id, AlertStatus.ACKNOWLEDGED)
    assert updated is not None
    assert updated.status == AlertStatus.ACKNOWLEDGED
    assert updated.acknowledged_at is not None


def test_audit_log_created(db):
    patient = Patient(name="Audited", date_of_birth="1970-01-01")
    db.create_patient(patient)
    log = db.get_audit_log()
    assert len(log) > 0
    assert log[0].action == "CREATE"
    assert log[0].resource_type == "patient"


def test_audit_integrity(db):
    patient = Patient(name="Integrity", date_of_birth="1970-01-01")
    db.create_patient(patient)
    violations = db.verify_audit_integrity()
    assert len(violations) == 0
