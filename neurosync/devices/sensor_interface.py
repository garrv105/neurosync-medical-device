"""Abstract sensor interface and mock sensor for testing.

The mock sensor generates realistic Parkinsonian tremor waveforms:
- Primary resting tremor at 4-6 Hz
- Physiological noise (broadband)
- Optional postural tremor component at 8-12 Hz
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from neurosync.core.models import DeviceState, DeviceStatus


@dataclass
class SensorConfig:
    sample_rate_hz: float = 100.0
    duration_seconds: float = 10.0
    device_id: str = "MOCK-001"


class SensorInterface(abc.ABC):
    @abc.abstractmethod
    def read(self, config: SensorConfig) -> list[float]:
        """Read sensor data for the configured duration."""

    @abc.abstractmethod
    def get_status(self) -> DeviceStatus:
        """Get current device status."""

    @abc.abstractmethod
    def calibrate(self) -> bool:
        """Run calibration routine."""


class MockSensor(SensorInterface):
    """Mock sensor that generates realistic Parkinsonian tremor signals.

    Generates a composite signal:
    1. Primary resting tremor: 4-6 Hz sinusoid with amplitude modulation
    2. Physiological noise: low-amplitude broadband noise
    3. Optional action tremor component: 8-12 Hz
    """

    def __init__(
        self,
        tremor_frequency: float = 5.0,
        tremor_amplitude: float = 1.0,
        noise_level: float = 0.1,
        include_action_tremor: bool = False,
        seed: int | None = None,
    ):
        self.tremor_frequency = tremor_frequency
        self.tremor_amplitude = tremor_amplitude
        self.noise_level = noise_level
        self.include_action_tremor = include_action_tremor
        self.rng = np.random.default_rng(seed)
        self._state = DeviceState.ONLINE
        self._calibrated_at: datetime | None = None
        self._device_id = "MOCK-001"

    def read(self, config: SensorConfig) -> list[float]:
        n_samples = int(config.sample_rate_hz * config.duration_seconds)
        t = np.arange(n_samples) / config.sample_rate_hz

        # Primary resting tremor (4-6 Hz)
        # Add slow amplitude modulation to simulate realistic tremor variability
        am_freq = 0.2  # slow modulation at 0.2 Hz
        amplitude_envelope = self.tremor_amplitude * (1.0 + 0.3 * np.sin(2 * np.pi * am_freq * t))
        tremor = amplitude_envelope * np.sin(2 * np.pi * self.tremor_frequency * t)

        # Add slight frequency jitter via phase modulation
        phase_noise = 0.1 * np.cumsum(self.rng.normal(0, 0.01, n_samples))
        tremor = amplitude_envelope * np.sin(2 * np.pi * self.tremor_frequency * t + phase_noise)

        # Physiological noise (broadband)
        noise = self.noise_level * self.rng.normal(0, 1, n_samples)

        signal = tremor + noise

        # Optional action/postural tremor component (8-12 Hz, lower amplitude)
        if self.include_action_tremor:
            action_freq = self.rng.uniform(8.0, 12.0)
            action_amp = self.tremor_amplitude * 0.3
            signal += action_amp * np.sin(2 * np.pi * action_freq * t)

        self._device_id = config.device_id
        return signal.tolist()

    def get_status(self) -> DeviceStatus:
        return DeviceStatus(
            device_id=self._device_id,
            state=self._state,
            battery_percent=85.0,
            firmware_version="1.0.0-mock",
            last_calibration=self._calibrated_at,
        )

    def calibrate(self) -> bool:
        self._calibrated_at = datetime.utcnow()
        return True
