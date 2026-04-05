"""FDA 21 CFR Part 11 compliance checks and audit trail management.

21 CFR Part 11 requires:
- Electronic records must be trustworthy, reliable, and equivalent to paper records
- Audit trails must record date/time, operator, and action
- Records must be tamper-evident (checksums)
- System access controls (tracked but not enforced at this layer)
"""

from __future__ import annotations

from dataclasses import dataclass

from neurosync.storage.database import Database


@dataclass
class ComplianceCheckResult:
    compliant: bool
    checks_passed: list[str]
    checks_failed: list[str]
    audit_integrity_violations: list[dict]
    total_audit_entries: int
    message: str


class ComplianceChecker:
    """FDA 21 CFR Part 11 compliance verification."""

    def __init__(self, db: Database):
        self.db = db

    def run_compliance_check(self) -> ComplianceCheckResult:
        passed = []
        failed = []

        # Check 1: Audit trail exists
        audit_log = self.db.get_audit_log(limit=10000)
        if len(audit_log) > 0:
            passed.append("Audit trail exists and contains entries")
        else:
            failed.append("Audit trail is empty — no tracked operations")

        # Check 2: Audit trail integrity (checksums)
        violations = self.db.verify_audit_integrity()
        if not violations:
            passed.append("Audit trail integrity verified (all checksums valid)")
        else:
            failed.append(f"Audit trail integrity compromised: {len(violations)} violations")

        # Check 3: All audit entries have timestamps
        entries_without_timestamp = [e for e in audit_log if e.timestamp is None]
        if not entries_without_timestamp:
            passed.append("All audit entries have timestamps")
        else:
            failed.append(f"{len(entries_without_timestamp)} audit entries missing timestamps")

        # Check 4: All audit entries have action and resource info
        incomplete = [
            e for e in audit_log if not e.action or not e.resource_type or not e.resource_id
        ]
        if not incomplete:
            passed.append("All audit entries have complete action/resource information")
        else:
            failed.append(f"{len(incomplete)} audit entries with incomplete information")

        # Check 5: Patient records have required fields
        patients = self.db.list_patients()
        incomplete_patients = [p for p in patients if not p.name or not p.date_of_birth]
        if not incomplete_patients:
            passed.append("All patient records have required fields")
        else:
            failed.append(f"{len(incomplete_patients)} patients with missing required fields")

        compliant = len(failed) == 0
        message = (
            "All compliance checks passed"
            if compliant
            else (f"Compliance issues found: {len(failed)} check(s) failed")
        )

        return ComplianceCheckResult(
            compliant=compliant,
            checks_passed=passed,
            checks_failed=failed,
            audit_integrity_violations=violations,
            total_audit_entries=len(audit_log),
            message=message,
        )

    def get_audit_trail(
        self,
        resource_type: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        entries = self.db.get_audit_log(resource_type=resource_type, limit=limit)
        return [
            {
                "id": e.id,
                "timestamp": e.timestamp.isoformat(),
                "action": e.action,
                "user_id": e.user_id,
                "resource_type": e.resource_type,
                "resource_id": e.resource_id,
                "details": e.details,
                "checksum": e.checksum,
            }
            for e in entries
        ]
