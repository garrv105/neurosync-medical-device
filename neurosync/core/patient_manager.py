"""Patient record management with CRUD operations and treatment tracking."""

from __future__ import annotations

from typing import Optional

from neurosync.core.models import Patient, TremorReading
from neurosync.storage.database import Database


class PatientManager:
    def __init__(self, db: Database):
        self.db = db

    def register_patient(
        self,
        name: str,
        date_of_birth: str,
        diagnosis_date: Optional[str] = None,
        medications: Optional[list[str]] = None,
        notes: Optional[str] = None,
    ) -> Patient:
        patient = Patient(
            name=name,
            date_of_birth=date_of_birth,
            diagnosis_date=diagnosis_date,
            medications=medications or [],
            notes=notes,
        )
        return self.db.create_patient(patient)

    def get_patient(self, patient_id: str) -> Optional[Patient]:
        return self.db.get_patient(patient_id)

    def list_patients(self) -> list[Patient]:
        return self.db.list_patients()

    def update_patient(self, patient: Patient) -> Patient:
        return self.db.update_patient(patient)

    def delete_patient(self, patient_id: str) -> bool:
        return self.db.delete_patient(patient_id)

    def add_medication(self, patient_id: str, medication: str) -> Optional[Patient]:
        patient = self.db.get_patient(patient_id)
        if patient is None:
            return None
        if medication not in patient.medications:
            patient.medications.append(medication)
            return self.db.update_patient(patient)
        return patient

    def remove_medication(self, patient_id: str, medication: str) -> Optional[Patient]:
        patient = self.db.get_patient(patient_id)
        if patient is None:
            return None
        if medication in patient.medications:
            patient.medications.remove(medication)
            return self.db.update_patient(patient)
        return patient

    def get_patient_readings(self, patient_id: str, limit: int = 100) -> list[TremorReading]:
        return self.db.get_readings(patient_id, limit=limit)

    def get_patient_summary(self, patient_id: str) -> Optional[dict]:
        patient = self.db.get_patient(patient_id)
        if patient is None:
            return None
        readings = self.db.get_readings(patient_id, limit=10)
        alerts = self.db.get_alerts(patient_id=patient_id)
        active_alerts = [a for a in alerts if a.status.value == "active"]
        avg_severity = 0.0
        if readings:
            scores = [r.severity_score for r in readings if r.severity_score is not None]
            avg_severity = sum(scores) / len(scores) if scores else 0.0
        return {
            "patient": patient,
            "total_readings": len(readings),
            "recent_avg_severity": round(avg_severity, 2),
            "active_alerts": len(active_alerts),
            "medications": patient.medications,
        }
