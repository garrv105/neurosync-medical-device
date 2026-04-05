"""FastAPI REST API for NeuroSync medical device monitoring."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from neurosync.analysis.severity_scorer import SeverityScorer
from neurosync.core.alert_engine import AlertEngine
from neurosync.core.models import AlertStatus
from neurosync.core.patient_manager import PatientManager
from neurosync.core.tremor_analyzer import analyze_tremor_signal
from neurosync.reporting.clinical_report import ClinicalReportGenerator
from neurosync.reporting.compliance import ComplianceChecker
from neurosync.storage.database import Database

app = FastAPI(
    title="NeuroSync API",
    description="Medical device monitoring API for Parkinson's disease patients",
    version="1.0.0",
)

# Module-level state initialized on startup
_db: Database | None = None
_patient_mgr: PatientManager | None = None
_alert_engine: AlertEngine | None = None
_report_gen: ClinicalReportGenerator | None = None
_compliance: ComplianceChecker | None = None


def get_db() -> Database:
    assert _db is not None, "Database not initialized"
    return _db


def init_app(db_path: str = ":memory:") -> FastAPI:
    global _db, _patient_mgr, _alert_engine, _report_gen, _compliance
    _db = Database(db_path)
    _patient_mgr = PatientManager(_db)
    _alert_engine = AlertEngine(_db)
    _report_gen = ClinicalReportGenerator(_db)
    _compliance = ComplianceChecker(_db)
    return app


# ── Request/Response models ──


class PatientCreate(BaseModel):
    name: str
    date_of_birth: str
    diagnosis_date: Optional[str] = None
    medications: list[str] = []
    notes: Optional[str] = None


class PatientUpdate(BaseModel):
    name: Optional[str] = None
    date_of_birth: Optional[str] = None
    diagnosis_date: Optional[str] = None
    medications: Optional[list[str]] = None
    notes: Optional[str] = None


class ReadingCreate(BaseModel):
    patient_id: str
    device_id: str
    sample_rate_hz: float = 100.0
    duration_seconds: float = 10.0
    raw_signal: list[float]


class AlertStatusUpdate(BaseModel):
    status: str


# ── Endpoints ──


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "neurosync", "timestamp": datetime.utcnow().isoformat()}


@app.post("/patients", status_code=201)
def create_patient(body: PatientCreate):
    assert _patient_mgr is not None
    patient = _patient_mgr.register_patient(
        name=body.name,
        date_of_birth=body.date_of_birth,
        diagnosis_date=body.diagnosis_date,
        medications=body.medications,
        notes=body.notes,
    )
    return patient.model_dump()


@app.get("/patients")
def list_patients():
    assert _patient_mgr is not None
    patients = _patient_mgr.list_patients()
    return [p.model_dump() for p in patients]


@app.get("/patients/{patient_id}")
def get_patient(patient_id: str):
    assert _patient_mgr is not None
    patient = _patient_mgr.get_patient(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient.model_dump()


@app.put("/patients/{patient_id}")
def update_patient(patient_id: str, body: PatientUpdate):
    assert _patient_mgr is not None
    patient = _patient_mgr.get_patient(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    if body.name is not None:
        patient.name = body.name
    if body.date_of_birth is not None:
        patient.date_of_birth = body.date_of_birth
    if body.diagnosis_date is not None:
        patient.diagnosis_date = body.diagnosis_date
    if body.medications is not None:
        patient.medications = body.medications
    if body.notes is not None:
        patient.notes = body.notes
    updated = _patient_mgr.update_patient(patient)
    return updated.model_dump()


@app.delete("/patients/{patient_id}")
def delete_patient(patient_id: str):
    assert _patient_mgr is not None
    if not _patient_mgr.delete_patient(patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"deleted": True}


@app.get("/patients/{patient_id}/summary")
def patient_summary(patient_id: str):
    assert _patient_mgr is not None
    summary = _patient_mgr.get_patient_summary(patient_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    summary["patient"] = summary["patient"].model_dump()
    return summary


@app.post("/readings", status_code=201)
def create_reading(body: ReadingCreate):
    db = get_db()
    assert _alert_engine is not None

    # Verify patient exists
    patient = db.get_patient(body.patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Analyze
    analysis = analyze_tremor_signal(body.raw_signal, body.sample_rate_hz)
    scorer = SeverityScorer()
    assessment = scorer.score(analysis)

    from neurosync.core.models import TremorReading

    reading = TremorReading(
        patient_id=body.patient_id,
        device_id=body.device_id,
        sample_rate_hz=body.sample_rate_hz,
        duration_seconds=body.duration_seconds,
        raw_signal=body.raw_signal,
        dominant_frequency_hz=analysis.dominant_frequency_hz,
        amplitude=analysis.amplitude_rms,
        severity_score=assessment.score,
        severity_level=assessment.level,
    )
    db.store_reading(reading)

    # Evaluate alerts
    alerts = _alert_engine.evaluate_reading(reading)

    result = reading.model_dump()
    result["alerts"] = [a.model_dump() for a in alerts]
    return result


@app.get("/readings/{patient_id}")
def get_readings(patient_id: str, limit: int = 100):
    db = get_db()
    readings = db.get_readings(patient_id, limit=limit)
    return [r.model_dump() for r in readings]


@app.get("/alerts")
def list_alerts(patient_id: Optional[str] = None, status: Optional[str] = None):
    db = get_db()
    alert_status = AlertStatus(status) if status else None
    alerts = db.get_alerts(patient_id=patient_id, status=alert_status)
    return [a.model_dump() for a in alerts]


@app.put("/alerts/{alert_id}")
def update_alert(alert_id: str, body: AlertStatusUpdate):
    assert _alert_engine is not None
    try:
        new_status = AlertStatus(body.status)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid status: {body.status}")
    if new_status == AlertStatus.ACKNOWLEDGED:
        alert = _alert_engine.acknowledge_alert(alert_id)
    elif new_status == AlertStatus.RESOLVED:
        alert = _alert_engine.resolve_alert(alert_id)
    else:
        raise HTTPException(status_code=400, detail="Cannot set status to active")
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert.model_dump()


@app.get("/reports/{patient_id}")
def generate_report(patient_id: str):
    assert _report_gen is not None
    try:
        report = _report_gen.generate_report(patient_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return report.model_dump()


@app.get("/compliance")
def compliance_check():
    assert _compliance is not None
    result = _compliance.run_compliance_check()
    return {
        "compliant": result.compliant,
        "checks_passed": result.checks_passed,
        "checks_failed": result.checks_failed,
        "violations": result.audit_integrity_violations,
        "total_audit_entries": result.total_audit_entries,
        "message": result.message,
    }


@app.get("/audit")
def audit_trail(resource_type: Optional[str] = None, limit: int = 100):
    assert _compliance is not None
    return _compliance.get_audit_trail(resource_type=resource_type, limit=limit)
