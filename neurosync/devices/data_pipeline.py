"""Stream processing pipeline: ingest, clean, buffer, store.

Provides a pipeline that takes raw sensor data, applies cleaning,
runs tremor analysis, severity scoring, and stores results.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
from scipy import signal as sp_signal

from neurosync.analysis.severity_scorer import SeverityScorer
from neurosync.core.models import TremorReading
from neurosync.core.tremor_analyzer import analyze_tremor_signal
from neurosync.storage.database import Database


class SignalCleaner:
    """Clean raw sensor signals: remove DC offset, bandpass filter, remove outliers."""

    def __init__(self, sample_rate_hz: float = 100.0):
        self.sample_rate_hz = sample_rate_hz

    def clean(self, raw_signal: list[float]) -> list[float]:
        data = np.asarray(raw_signal, dtype=np.float64)

        # Remove DC offset
        data = data - np.mean(data)

        # Remove outliers (>5 std deviations)
        std = np.std(data)
        if std > 0:
            mask = np.abs(data) > 5 * std
            if np.any(mask):
                data[mask] = np.clip(data[mask], -5 * std, 5 * std)

        # Bandpass filter to relevant physiological range (0.5-30 Hz)
        sos = sp_signal.butter(
            4, [0.5, 30.0], btype="bandpass", fs=self.sample_rate_hz, output="sos"
        )
        data = sp_signal.sosfiltfilt(sos, data)

        return data.tolist()


class DataPipeline:
    """End-to-end pipeline: ingest -> clean -> analyze -> score -> store."""

    def __init__(self, db: Database, sample_rate_hz: float = 100.0):
        self.db = db
        self.sample_rate_hz = sample_rate_hz
        self.cleaner = SignalCleaner(sample_rate_hz)
        self.scorer = SeverityScorer()
        self._buffer: list[list[float]] = []

    def ingest(
        self,
        raw_signal: list[float],
        patient_id: str,
        device_id: str,
        duration_seconds: Optional[float] = None,
    ) -> TremorReading:
        """Ingest a raw signal, process it, and store the result."""
        if duration_seconds is None:
            duration_seconds = len(raw_signal) / self.sample_rate_hz

        # Clean
        cleaned = self.cleaner.clean(raw_signal)

        # Analyze
        analysis = analyze_tremor_signal(cleaned, self.sample_rate_hz)

        # Score
        assessment = self.scorer.score(analysis)

        # Create reading
        reading = TremorReading(
            patient_id=patient_id,
            device_id=device_id,
            timestamp=datetime.utcnow(),
            sample_rate_hz=self.sample_rate_hz,
            duration_seconds=duration_seconds,
            raw_signal=raw_signal,
            dominant_frequency_hz=analysis.dominant_frequency_hz,
            amplitude=analysis.amplitude_rms,
            severity_score=assessment.score,
            severity_level=assessment.level,
        )

        # Store
        self.db.store_reading(reading)

        # Buffer for batch operations
        self._buffer.append(raw_signal)
        if len(self._buffer) > 100:
            self._buffer = self._buffer[-100:]

        return reading

    def get_buffer_size(self) -> int:
        return len(self._buffer)

    def clear_buffer(self) -> None:
        self._buffer.clear()
