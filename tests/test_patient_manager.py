"""Tests for patient management."""

import pytest

from neurosync.core.patient_manager import PatientManager


@pytest.fixture
def pm(db):
    return PatientManager(db)


def test_register_patient(pm):
    patient = pm.register_patient(
        name="Jane Smith",
        date_of_birth="1960-01-01",
        diagnosis_date="2021-03-15",
        medications=["Levodopa"],
    )
    assert patient.name == "Jane Smith"
    assert patient.id is not None
    assert "Levodopa" in patient.medications


def test_get_patient(pm):
    created = pm.register_patient(name="Bob", date_of_birth="1950-05-20")
    fetched = pm.get_patient(created.id)
    assert fetched is not None
    assert fetched.name == "Bob"


def test_list_patients(pm):
    pm.register_patient(name="Alice", date_of_birth="1965-01-01")
    pm.register_patient(name="Bob", date_of_birth="1970-02-02")
    patients = pm.list_patients()
    assert len(patients) == 2


def test_update_patient(pm):
    patient = pm.register_patient(name="Charlie", date_of_birth="1955-08-10")
    patient.notes = "Updated notes"
    updated = pm.update_patient(patient)
    assert updated.notes == "Updated notes"


def test_delete_patient(pm):
    patient = pm.register_patient(name="Delete Me", date_of_birth="1980-01-01")
    assert pm.delete_patient(patient.id) is True
    assert pm.get_patient(patient.id) is None


def test_add_medication(pm):
    patient = pm.register_patient(name="Med Patient", date_of_birth="1960-01-01")
    updated = pm.add_medication(patient.id, "Pramipexole")
    assert updated is not None
    assert "Pramipexole" in updated.medications


def test_remove_medication(pm):
    patient = pm.register_patient(
        name="Med Patient", date_of_birth="1960-01-01", medications=["Levodopa", "Carbidopa"]
    )
    updated = pm.remove_medication(patient.id, "Carbidopa")
    assert updated is not None
    assert "Carbidopa" not in updated.medications
    assert "Levodopa" in updated.medications


def test_patient_summary(pm, db, sample_reading):
    patient = pm.register_patient(
        name="Summary Patient",
        date_of_birth="1960-01-01",
        medications=["Levodopa"],
    )
    # Update the reading to use this patient's ID
    sample_reading.patient_id = patient.id
    db.store_reading(sample_reading)

    summary = pm.get_patient_summary(patient.id)
    assert summary is not None
    assert summary["total_readings"] == 1
    assert summary["medications"] == ["Levodopa"]
