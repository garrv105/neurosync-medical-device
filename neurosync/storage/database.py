"""SQLite storage backend for patient data, readings, alerts, and audit log."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from neurosync.core.models import (
    Alert,
    AlertStatus,
    AuditLogEntry,
    Patient,
    TremorReading,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    date_of_birth TEXT NOT NULL,
    diagnosis_date TEXT,
    medications TEXT,
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tremor_readings (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    device_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    sample_rate_hz REAL NOT NULL,
    duration_seconds REAL NOT NULL,
    raw_signal TEXT NOT NULL,
    dominant_frequency_hz REAL,
    amplitude REAL,
    severity_score REAL,
    severity_level TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    reading_id TEXT,
    priority TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    message TEXT NOT NULL,
    details TEXT,
    created_at TEXT NOT NULL,
    acknowledged_at TEXT,
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    action TEXT NOT NULL,
    user_id TEXT,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    details TEXT,
    checksum TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_readings_patient ON tremor_readings(patient_id);
CREATE INDEX IF NOT EXISTS idx_readings_timestamp ON tremor_readings(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_patient ON alerts(patient_id);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
"""


def _compute_checksum(action: str, resource_type: str, resource_id: str, details: str) -> str:
    data = f"{action}|{resource_type}|{resource_id}|{details}"
    return hashlib.sha256(data.encode()).hexdigest()


class Database:
    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _audit(self, action: str, resource_type: str, resource_id: str, details: str = "") -> None:
        entry = AuditLogEntry(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            checksum=_compute_checksum(action, resource_type, resource_id, details),
        )
        self.conn.execute(
            "INSERT INTO audit_log (id, timestamp, action, user_id, resource_type, resource_id, details, checksum) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.id,
                entry.timestamp.isoformat(),
                entry.action,
                entry.user_id,
                entry.resource_type,
                entry.resource_id,
                entry.details,
                entry.checksum,
            ),
        )

    # ── Patients ──

    def create_patient(self, patient: Patient) -> Patient:
        self.conn.execute(
            "INSERT INTO patients (id, name, date_of_birth, diagnosis_date, medications, notes, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                patient.id,
                patient.name,
                patient.date_of_birth,
                patient.diagnosis_date,
                json.dumps(patient.medications),
                patient.notes,
                patient.created_at.isoformat(),
                patient.updated_at.isoformat(),
            ),
        )
        self._audit("CREATE", "patient", patient.id, f"Created patient {patient.name}")
        self.conn.commit()
        return patient

    def get_patient(self, patient_id: str) -> Optional[Patient]:
        row = self.conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
        if row is None:
            return None
        return Patient(
            id=row["id"],
            name=row["name"],
            date_of_birth=row["date_of_birth"],
            diagnosis_date=row["diagnosis_date"],
            medications=json.loads(row["medications"]) if row["medications"] else [],
            notes=row["notes"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def list_patients(self) -> list[Patient]:
        rows = self.conn.execute("SELECT * FROM patients ORDER BY name").fetchall()
        return [
            Patient(
                id=row["id"],
                name=row["name"],
                date_of_birth=row["date_of_birth"],
                diagnosis_date=row["diagnosis_date"],
                medications=json.loads(row["medications"]) if row["medications"] else [],
                notes=row["notes"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            for row in rows
        ]

    def update_patient(self, patient: Patient) -> Patient:
        patient.updated_at = datetime.utcnow()
        self.conn.execute(
            "UPDATE patients SET name=?, date_of_birth=?, diagnosis_date=?, medications=?, notes=?, updated_at=? WHERE id=?",
            (
                patient.name,
                patient.date_of_birth,
                patient.diagnosis_date,
                json.dumps(patient.medications),
                patient.notes,
                patient.updated_at.isoformat(),
                patient.id,
            ),
        )
        self._audit("UPDATE", "patient", patient.id, f"Updated patient {patient.name}")
        self.conn.commit()
        return patient

    def delete_patient(self, patient_id: str) -> bool:
        cursor = self.conn.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
        if cursor.rowcount > 0:
            self._audit("DELETE", "patient", patient_id, "Deleted patient")
            self.conn.commit()
            return True
        return False

    # ── Readings ──

    def store_reading(self, reading: TremorReading) -> TremorReading:
        self.conn.execute(
            "INSERT INTO tremor_readings (id, patient_id, device_id, timestamp, sample_rate_hz, "
            "duration_seconds, raw_signal, dominant_frequency_hz, amplitude, severity_score, severity_level) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                reading.id,
                reading.patient_id,
                reading.device_id,
                reading.timestamp.isoformat(),
                reading.sample_rate_hz,
                reading.duration_seconds,
                json.dumps(reading.raw_signal),
                reading.dominant_frequency_hz,
                reading.amplitude,
                reading.severity_score,
                reading.severity_level.value if reading.severity_level else None,
            ),
        )
        self._audit(
            "CREATE", "reading", reading.id, f"Stored reading for patient {reading.patient_id}"
        )
        self.conn.commit()
        return reading

    def get_readings(self, patient_id: str, limit: int = 100) -> list[TremorReading]:
        rows = self.conn.execute(
            "SELECT * FROM tremor_readings WHERE patient_id = ? ORDER BY timestamp DESC LIMIT ?",
            (patient_id, limit),
        ).fetchall()
        return [self._row_to_reading(row) for row in rows]

    def get_reading(self, reading_id: str) -> Optional[TremorReading]:
        row = self.conn.execute(
            "SELECT * FROM tremor_readings WHERE id = ?", (reading_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_reading(row)

    def _row_to_reading(self, row: sqlite3.Row) -> TremorReading:
        return TremorReading(
            id=row["id"],
            patient_id=row["patient_id"],
            device_id=row["device_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            sample_rate_hz=row["sample_rate_hz"],
            duration_seconds=row["duration_seconds"],
            raw_signal=json.loads(row["raw_signal"]),
            dominant_frequency_hz=row["dominant_frequency_hz"],
            amplitude=row["amplitude"],
            severity_score=row["severity_score"],
            severity_level=row["severity_level"],
        )

    # ── Alerts ──

    def store_alert(self, alert: Alert) -> Alert:
        self.conn.execute(
            "INSERT INTO alerts (id, patient_id, reading_id, priority, status, message, details, "
            "created_at, acknowledged_at, resolved_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                alert.id,
                alert.patient_id,
                alert.reading_id,
                alert.priority.value,
                alert.status.value,
                alert.message,
                json.dumps(alert.details) if alert.details else None,
                alert.created_at.isoformat(),
                alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                alert.resolved_at.isoformat() if alert.resolved_at else None,
            ),
        )
        self._audit("CREATE", "alert", alert.id, f"Alert: {alert.message}")
        self.conn.commit()
        return alert

    def get_alerts(
        self, patient_id: Optional[str] = None, status: Optional[AlertStatus] = None
    ) -> list[Alert]:
        query = "SELECT * FROM alerts WHERE 1=1"
        params: list = []
        if patient_id:
            query += " AND patient_id = ?"
            params.append(patient_id)
        if status:
            query += " AND status = ?"
            params.append(status.value)
        query += " ORDER BY created_at DESC"
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_alert(row) for row in rows]

    def update_alert_status(self, alert_id: str, status: AlertStatus) -> Optional[Alert]:
        now = datetime.utcnow().isoformat()
        if status == AlertStatus.ACKNOWLEDGED:
            self.conn.execute(
                "UPDATE alerts SET status=?, acknowledged_at=? WHERE id=?",
                (status.value, now, alert_id),
            )
        elif status == AlertStatus.RESOLVED:
            self.conn.execute(
                "UPDATE alerts SET status=?, resolved_at=? WHERE id=?",
                (status.value, now, alert_id),
            )
        else:
            self.conn.execute("UPDATE alerts SET status=? WHERE id=?", (status.value, alert_id))
        self._audit("UPDATE", "alert", alert_id, f"Status changed to {status.value}")
        self.conn.commit()
        row = self.conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_alert(row)

    def _row_to_alert(self, row: sqlite3.Row) -> Alert:
        return Alert(
            id=row["id"],
            patient_id=row["patient_id"],
            reading_id=row["reading_id"],
            priority=row["priority"],
            status=row["status"],
            message=row["message"],
            details=json.loads(row["details"]) if row["details"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            acknowledged_at=datetime.fromisoformat(row["acknowledged_at"])
            if row["acknowledged_at"]
            else None,
            resolved_at=datetime.fromisoformat(row["resolved_at"]) if row["resolved_at"] else None,
        )

    # ── Audit Log ──

    def get_audit_log(
        self, resource_type: Optional[str] = None, limit: int = 100
    ) -> list[AuditLogEntry]:
        query = "SELECT * FROM audit_log"
        params: list = []
        if resource_type:
            query += " WHERE resource_type = ?"
            params.append(resource_type)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [
            AuditLogEntry(
                id=row["id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                action=row["action"],
                user_id=row["user_id"],
                resource_type=row["resource_type"],
                resource_id=row["resource_id"],
                details=row["details"],
                checksum=row["checksum"],
            )
            for row in rows
        ]

    def verify_audit_integrity(self) -> list[dict]:
        violations = []
        rows = self.conn.execute("SELECT * FROM audit_log ORDER BY timestamp").fetchall()
        for row in rows:
            expected = _compute_checksum(
                row["action"], row["resource_type"], row["resource_id"], row["details"] or ""
            )
            if row["checksum"] != expected:
                violations.append(
                    {"id": row["id"], "expected": expected, "actual": row["checksum"]}
                )
        return violations
