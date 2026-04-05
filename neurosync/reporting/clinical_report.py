"""Clinical report generation (JSON and PDF).

Generates comprehensive clinical reports summarizing patient tremor data,
severity trends, alert history, and treatment recommendations.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Optional

from neurosync.analysis.severity_scorer import SeverityScorer
from neurosync.analysis.trend_detector import detect_trend
from neurosync.core.models import ClinicalReport, TremorReading
from neurosync.storage.database import Database


class ClinicalReportGenerator:
    def __init__(self, db: Database):
        self.db = db
        self.scorer = SeverityScorer()

    def generate_report(
        self,
        patient_id: str,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> ClinicalReport:
        patient = self.db.get_patient(patient_id)
        if patient is None:
            raise ValueError(f"Patient {patient_id} not found")

        readings = self.db.get_readings(patient_id)

        if period_start:
            readings = [r for r in readings if r.timestamp >= period_start]
        if period_end:
            readings = [r for r in readings if r.timestamp <= period_end]

        now = datetime.utcnow()
        if not period_start:
            period_start = readings[-1].timestamp if readings else now
        if not period_end:
            period_end = readings[0].timestamp if readings else now

        # Compute severity stats
        scores = [r.severity_score for r in readings if r.severity_score is not None]
        avg_severity = sum(scores) / len(scores) if scores else 0.0

        # Trend detection
        trend = detect_trend(readings) if len(readings) >= 2 else None
        severity_trend = trend.direction if trend else "insufficient_data"

        # Alerts summary
        alerts = self.db.get_alerts(patient_id=patient_id)
        alerts_summary = {
            "total": len(alerts),
            "active": sum(1 for a in alerts if a.status.value == "active"),
            "resolved": sum(1 for a in alerts if a.status.value == "resolved"),
            "by_priority": {},
        }
        for a in alerts:
            p = a.priority.value
            alerts_summary["by_priority"][p] = alerts_summary["by_priority"].get(p, 0) + 1

        # Recommendations
        recommendations = self._generate_recommendations(avg_severity, severity_trend, readings)

        return ClinicalReport(
            patient_id=patient_id,
            period_start=period_start,
            period_end=period_end,
            total_readings=len(readings),
            average_severity=round(avg_severity, 2),
            severity_trend=severity_trend,
            alerts_summary=alerts_summary,
            recommendations=recommendations,
        )

    def generate_pdf(
        self,
        patient_id: str,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> bytes:
        """Generate a PDF clinical report."""
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        report = self.generate_report(patient_id, period_start, period_end)
        patient = self.db.get_patient(patient_id)

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("NeuroSync Clinical Report", styles["Title"]))
        story.append(Spacer(1, 12))

        # Patient info
        if patient:
            story.append(Paragraph(f"<b>Patient:</b> {patient.name}", styles["Normal"]))
            story.append(Paragraph(f"<b>DOB:</b> {patient.date_of_birth}", styles["Normal"]))
            if patient.medications:
                story.append(
                    Paragraph(
                        f"<b>Medications:</b> {', '.join(patient.medications)}", styles["Normal"]
                    )
                )
        story.append(Spacer(1, 12))

        # Period
        story.append(
            Paragraph(
                f"<b>Period:</b> {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 12))

        # Summary
        story.append(Paragraph("Summary", styles["Heading2"]))
        story.append(Paragraph(f"Total readings: {report.total_readings}", styles["Normal"]))
        story.append(
            Paragraph(f"Average severity: {report.average_severity:.2f} / 4.0", styles["Normal"])
        )
        story.append(Paragraph(f"Severity trend: {report.severity_trend}", styles["Normal"]))
        story.append(Spacer(1, 12))

        # Alerts
        story.append(Paragraph("Alerts", styles["Heading2"]))
        story.append(
            Paragraph(
                f"Total: {report.alerts_summary.get('total', 0)}, "
                f"Active: {report.alerts_summary.get('active', 0)}",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 12))

        # Recommendations
        story.append(Paragraph("Recommendations", styles["Heading2"]))
        for rec in report.recommendations:
            story.append(Paragraph(f"- {rec}", styles["Normal"]))

        doc.build(story)
        return buffer.getvalue()

    @staticmethod
    def _generate_recommendations(
        avg_severity: float, trend: str, readings: list[TremorReading]
    ) -> list[str]:
        recommendations = []

        if avg_severity >= 3.0:
            recommendations.append(
                "Tremor severity is high (>=3.0). Consider neurologist consultation "
                "for medication adjustment or therapy review."
            )
        elif avg_severity >= 2.0:
            recommendations.append(
                "Moderate tremor severity. Monitor closely and consider medication optimization."
            )

        if trend == "worsening":
            recommendations.append(
                "Severity trend is worsening. Recommend increased monitoring frequency "
                "and potential treatment modification."
            )
        elif trend == "improving":
            recommendations.append(
                "Severity trend is improving. Current treatment appears effective."
            )

        if len(readings) < 5:
            recommendations.append(
                "Limited data available. Recommend more frequent monitoring for reliable trend analysis."
            )

        if not recommendations:
            recommendations.append(
                "Tremor levels are within acceptable range. Continue current monitoring schedule."
            )

        return recommendations
