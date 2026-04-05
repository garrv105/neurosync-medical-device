"""Longitudinal trend detection and medication response tracking.

Detects trends in tremor severity over time using linear regression
and change-point detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurosync.core.models import TremorReading


@dataclass
class TrendResult:
    direction: str  # "improving", "stable", "worsening"
    slope: float
    r_squared: float
    mean_severity: float
    severity_change: float
    period_days: float
    data_points: int
    message: str


@dataclass
class MedicationResponse:
    medication: str
    pre_severity_mean: float
    post_severity_mean: float
    change: float
    change_percent: float
    effective: bool
    message: str


def detect_trend(readings: list[TremorReading]) -> TrendResult:
    """Analyze severity trend over a series of readings using linear regression."""
    if len(readings) < 2:
        return TrendResult(
            direction="stable",
            slope=0.0,
            r_squared=0.0,
            mean_severity=readings[0].severity_score if readings else 0.0,
            severity_change=0.0,
            period_days=0.0,
            data_points=len(readings),
            message="Insufficient data for trend analysis (need >= 2 readings)",
        )

    # Sort by timestamp
    sorted_readings = sorted(readings, key=lambda r: r.timestamp)

    # Extract timestamps and severity scores
    timestamps = [r.timestamp for r in sorted_readings]
    scores = [r.severity_score or 0.0 for r in sorted_readings]

    # Convert timestamps to days from first reading
    t0 = timestamps[0]
    days = np.array([(t - t0).total_seconds() / 86400 for t in timestamps])
    y = np.array(scores)

    # Linear regression
    if np.std(days) > 0:
        coeffs = np.polyfit(days, y, 1)
        slope = float(coeffs[0])

        # R-squared
        y_pred = np.polyval(coeffs, days)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        slope = 0.0
        r_squared = 0.0

    period_days = float(days[-1]) if len(days) > 0 else 0.0
    severity_change = float(y[-1] - y[0])
    mean_severity = float(np.mean(y))

    # Determine direction
    if abs(slope) < 0.01:
        direction = "stable"
    elif slope < 0:
        direction = "improving"
    else:
        direction = "worsening"

    direction_msg = {
        "improving": f"Tremor severity improving (slope: {slope:.3f}/day)",
        "stable": f"Tremor severity stable (mean: {mean_severity:.2f})",
        "worsening": f"Tremor severity worsening (slope: {slope:.3f}/day)",
    }

    return TrendResult(
        direction=direction,
        slope=slope,
        r_squared=r_squared,
        mean_severity=mean_severity,
        severity_change=severity_change,
        period_days=period_days,
        data_points=len(readings),
        message=direction_msg[direction],
    )


def evaluate_medication_response(
    readings_before: list[TremorReading],
    readings_after: list[TremorReading],
    medication: str,
    improvement_threshold: float = 0.3,
) -> MedicationResponse:
    """Evaluate whether a medication change improved tremor severity."""
    pre_scores = [r.severity_score or 0.0 for r in readings_before]
    post_scores = [r.severity_score or 0.0 for r in readings_after]

    pre_mean = float(np.mean(pre_scores)) if pre_scores else 0.0
    post_mean = float(np.mean(post_scores)) if post_scores else 0.0

    change = post_mean - pre_mean
    change_pct = (change / pre_mean * 100) if pre_mean > 0 else 0.0
    effective = change < -improvement_threshold

    if effective:
        msg = f"{medication}: severity improved by {abs(change):.2f} ({abs(change_pct):.1f}%)"
    elif change > improvement_threshold:
        msg = f"{medication}: severity worsened by {change:.2f} ({change_pct:.1f}%)"
    else:
        msg = f"{medication}: no significant change in severity ({change:+.2f})"

    return MedicationResponse(
        medication=medication,
        pre_severity_mean=round(pre_mean, 2),
        post_severity_mean=round(post_mean, 2),
        change=round(change, 2),
        change_percent=round(change_pct, 1),
        effective=effective,
        message=msg,
    )
