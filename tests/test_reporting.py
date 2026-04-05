"""Tests for clinical reporting and compliance."""

import pytest

from neurosync.core.models import Patient
from neurosync.reporting.clinical_report import ClinicalReportGenerator
from neurosync.reporting.compliance import ComplianceChecker


@pytest.fixture
def populated_db(db, sample_reading):
    patient = Patient(
        id="test-patient-001",
        name="Test Patient",
        date_of_birth="1960-01-01",
        medications=["Levodopa"],
    )
    db.create_patient(patient)
    db.store_reading(sample_reading)
    return db


def test_generate_report(populated_db):
    gen = ClinicalReportGenerator(populated_db)
    report = gen.generate_report("test-patient-001")
    assert report.patient_id == "test-patient-001"
    assert report.total_readings == 1
    assert report.average_severity >= 0


def test_generate_report_unknown_patient(db):
    gen = ClinicalReportGenerator(db)
    with pytest.raises(ValueError, match="not found"):
        gen.generate_report("nonexistent")


def test_generate_pdf(populated_db):
    gen = ClinicalReportGenerator(populated_db)
    pdf_bytes = gen.generate_pdf("test-patient-001")
    assert len(pdf_bytes) > 0
    assert pdf_bytes[:5] == b"%PDF-"


def test_compliance_check_with_data(populated_db):
    checker = ComplianceChecker(populated_db)
    result = checker.run_compliance_check()
    assert result.total_audit_entries > 0
    assert len(result.checks_passed) > 0


def test_compliance_check_empty_db(db):
    checker = ComplianceChecker(db)
    result = checker.run_compliance_check()
    # Empty DB should have audit trail failure
    assert "empty" in result.checks_failed[0].lower() or not result.compliant


def test_audit_trail_integrity(populated_db):
    checker = ComplianceChecker(populated_db)
    result = checker.run_compliance_check()
    assert len(result.audit_integrity_violations) == 0


def test_get_audit_trail(populated_db):
    checker = ComplianceChecker(populated_db)
    trail = checker.get_audit_trail()
    assert len(trail) > 0
    assert "action" in trail[0]
    assert "checksum" in trail[0]
