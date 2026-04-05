"""Sensor calibration routines and drift detection.

Provides baseline calibration, noise floor estimation, and
drift detection by comparing current sensor characteristics to baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurosync.devices.sensor_interface import SensorConfig, SensorInterface


@dataclass
class CalibrationResult:
    success: bool
    baseline_mean: float
    baseline_std: float
    noise_floor_rms: float
    dc_offset: float
    message: str


@dataclass
class DriftReport:
    drift_detected: bool
    mean_drift: float
    std_drift: float
    noise_drift: float
    message: str


class CalibrationManager:
    """Manages sensor calibration and drift detection."""

    def __init__(self, sensor: SensorInterface):
        self.sensor = sensor
        self._baseline: CalibrationResult | None = None

    def run_calibration(
        self,
        n_samples: int = 5,
        duration: float = 2.0,
        sample_rate: float = 100.0,
    ) -> CalibrationResult:
        """Run baseline calibration by collecting multiple short readings.

        The sensor should be stationary during calibration.
        """
        config = SensorConfig(
            sample_rate_hz=sample_rate,
            duration_seconds=duration,
        )

        all_means = []
        all_stds = []
        all_rms = []

        for _ in range(n_samples):
            data = np.asarray(self.sensor.read(config))
            all_means.append(float(np.mean(data)))
            all_stds.append(float(np.std(data)))
            all_rms.append(float(np.sqrt(np.mean(data**2))))

        baseline_mean = float(np.mean(all_means))
        baseline_std = float(np.mean(all_stds))
        noise_floor = float(np.mean(all_rms))
        dc_offset = baseline_mean

        self._baseline = CalibrationResult(
            success=True,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            noise_floor_rms=noise_floor,
            dc_offset=dc_offset,
            message=f"Calibration complete. DC offset: {dc_offset:.4f}, noise floor: {noise_floor:.4f}",
        )

        self.sensor.calibrate()
        return self._baseline

    def check_drift(
        self,
        mean_threshold: float = 0.5,
        std_threshold: float = 0.5,
        noise_threshold: float = 0.5,
    ) -> DriftReport:
        """Check for sensor drift by comparing current readings to baseline."""
        if self._baseline is None:
            return DriftReport(
                drift_detected=False,
                mean_drift=0.0,
                std_drift=0.0,
                noise_drift=0.0,
                message="No baseline calibration available. Run calibration first.",
            )

        config = SensorConfig(sample_rate_hz=100.0, duration_seconds=2.0)
        data = np.asarray(self.sensor.read(config))

        current_mean = float(np.mean(data))
        current_std = float(np.std(data))
        current_rms = float(np.sqrt(np.mean(data**2)))

        mean_drift = abs(current_mean - self._baseline.baseline_mean)
        std_drift = abs(current_std - self._baseline.baseline_std)
        noise_drift = abs(current_rms - self._baseline.noise_floor_rms)

        drift_detected = (
            mean_drift > mean_threshold
            or std_drift > std_threshold
            or noise_drift > noise_threshold
        )

        msg = (
            "No significant drift."
            if not drift_detected
            else (
                f"Drift detected! Mean: {mean_drift:.4f}, Std: {std_drift:.4f}, Noise: {noise_drift:.4f}"
            )
        )

        return DriftReport(
            drift_detected=drift_detected,
            mean_drift=mean_drift,
            std_drift=std_drift,
            noise_drift=noise_drift,
            message=msg,
        )

    @property
    def baseline(self) -> CalibrationResult | None:
        return self._baseline
