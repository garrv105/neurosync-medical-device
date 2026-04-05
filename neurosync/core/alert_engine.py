"""Rule-based alerting engine with configurable thresholds and escalation logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from neurosync.core.models import Alert, AlertPriority, TremorReading
from neurosync.storage.database import Database


@dataclass
class AlertRule:
    name: str
    severity_threshold: float
    priority: AlertPriority
    message_template: str
    frequency_min: Optional[float] = None
    frequency_max: Optional[float] = None
    amplitude_threshold: Optional[float] = None


@dataclass
class AlertEngineConfig:
    rules: list[AlertRule] = field(default_factory=list)
    escalation_window_minutes: int = 30
    escalation_count: int = 3

    @staticmethod
    def default() -> AlertEngineConfig:
        return AlertEngineConfig(
            rules=[
                AlertRule(
                    name="mild_tremor",
                    severity_threshold=1.0,
                    priority=AlertPriority.LOW,
                    message_template="Mild tremor detected (severity: {severity:.1f})",
                ),
                AlertRule(
                    name="moderate_tremor",
                    severity_threshold=2.0,
                    priority=AlertPriority.MEDIUM,
                    message_template="Moderate tremor detected (severity: {severity:.1f}, freq: {frequency:.1f} Hz)",
                ),
                AlertRule(
                    name="severe_tremor",
                    severity_threshold=3.0,
                    priority=AlertPriority.HIGH,
                    message_template="Severe tremor detected (severity: {severity:.1f}, freq: {frequency:.1f} Hz)",
                ),
                AlertRule(
                    name="critical_tremor",
                    severity_threshold=3.5,
                    priority=AlertPriority.CRITICAL,
                    message_template="CRITICAL tremor event (severity: {severity:.1f}) — immediate review required",
                ),
                AlertRule(
                    name="abnormal_frequency",
                    severity_threshold=0.0,
                    priority=AlertPriority.MEDIUM,
                    message_template="Abnormal tremor frequency: {frequency:.1f} Hz (outside 3-7 Hz band)",
                    frequency_min=3.0,
                    frequency_max=7.0,
                ),
            ],
            escalation_window_minutes=30,
            escalation_count=3,
        )


class AlertEngine:
    def __init__(self, db: Database, config: Optional[AlertEngineConfig] = None):
        self.db = db
        self.config = config or AlertEngineConfig.default()

    def evaluate_reading(self, reading: TremorReading) -> list[Alert]:
        alerts: list[Alert] = []
        severity = reading.severity_score or 0.0
        frequency = reading.dominant_frequency_hz or 0.0

        # Find highest matching severity rule
        best_severity_rule: Optional[AlertRule] = None
        for rule in self.config.rules:
            if rule.frequency_min is not None or rule.frequency_max is not None:
                continue  # Skip frequency rules in this pass
            if severity >= rule.severity_threshold:
                if (
                    best_severity_rule is None
                    or rule.severity_threshold > best_severity_rule.severity_threshold
                ):
                    best_severity_rule = rule

        if best_severity_rule is not None:
            alert = Alert(
                patient_id=reading.patient_id,
                reading_id=reading.id,
                priority=best_severity_rule.priority,
                message=best_severity_rule.message_template.format(
                    severity=severity, frequency=frequency
                ),
                details={
                    "rule": best_severity_rule.name,
                    "severity_score": severity,
                    "dominant_frequency_hz": frequency,
                },
            )
            self.db.store_alert(alert)
            alerts.append(alert)

        # Check frequency rules
        for rule in self.config.rules:
            if rule.frequency_min is None and rule.frequency_max is None:
                continue
            if frequency > 0:
                outside_band = False
                if rule.frequency_min is not None and frequency < rule.frequency_min:
                    outside_band = True
                if rule.frequency_max is not None and frequency > rule.frequency_max:
                    outside_band = True
                if outside_band:
                    alert = Alert(
                        patient_id=reading.patient_id,
                        reading_id=reading.id,
                        priority=rule.priority,
                        message=rule.message_template.format(
                            severity=severity, frequency=frequency
                        ),
                        details={
                            "rule": rule.name,
                            "frequency_hz": frequency,
                        },
                    )
                    self.db.store_alert(alert)
                    alerts.append(alert)

        # Check escalation: if multiple recent high-priority alerts, escalate
        alerts = self._check_escalation(reading.patient_id, alerts)
        return alerts

    def _check_escalation(self, patient_id: str, new_alerts: list[Alert]) -> list[Alert]:
        recent_alerts = self.db.get_alerts(patient_id=patient_id)
        high_count = sum(
            1
            for a in recent_alerts
            if a.priority in (AlertPriority.HIGH, AlertPriority.CRITICAL)
            and a.status.value == "active"
        )
        if high_count >= self.config.escalation_count:
            escalation = Alert(
                patient_id=patient_id,
                priority=AlertPriority.CRITICAL,
                message=f"ESCALATION: {high_count} active high-priority alerts — clinical review required",
                details={"escalation_trigger": high_count},
            )
            self.db.store_alert(escalation)
            new_alerts.append(escalation)
        return new_alerts

    def acknowledge_alert(self, alert_id: str) -> Optional[Alert]:
        from neurosync.core.models import AlertStatus

        return self.db.update_alert_status(alert_id, AlertStatus.ACKNOWLEDGED)

    def resolve_alert(self, alert_id: str) -> Optional[Alert]:
        from neurosync.core.models import AlertStatus

        return self.db.update_alert_status(alert_id, AlertStatus.RESOLVED)
