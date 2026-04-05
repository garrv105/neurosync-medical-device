"""Tests for sensor interface and mock sensor."""

from neurosync.core.tremor_analyzer import analyze_tremor_signal
from neurosync.devices.calibration import CalibrationManager
from neurosync.devices.sensor_interface import MockSensor, SensorConfig


def test_mock_sensor_signal_length():
    sensor = MockSensor(seed=42)
    config = SensorConfig(sample_rate_hz=100.0, duration_seconds=5.0)
    signal = sensor.read(config)
    assert len(signal) == 500


def test_mock_sensor_tremor_frequency():
    """The mock sensor's output should have dominant frequency near the configured value."""
    sensor = MockSensor(tremor_frequency=5.0, tremor_amplitude=2.0, noise_level=0.05, seed=42)
    config = SensorConfig(sample_rate_hz=100.0, duration_seconds=10.0)
    signal = sensor.read(config)
    result = analyze_tremor_signal(signal, 100.0)
    assert 4.0 <= result.dominant_frequency_hz <= 6.0


def test_mock_sensor_with_action_tremor():
    sensor = MockSensor(include_action_tremor=True, seed=42)
    config = SensorConfig(sample_rate_hz=100.0, duration_seconds=10.0)
    signal = sensor.read(config)
    assert len(signal) == 1000
    # Signal should have more spectral complexity
    result = analyze_tremor_signal(signal, 100.0)
    assert result.total_power > 0


def test_mock_sensor_status():
    sensor = MockSensor()
    status = sensor.get_status()
    assert status.state.value == "online"
    assert status.battery_percent == 85.0


def test_mock_sensor_calibrate():
    sensor = MockSensor()
    assert sensor.calibrate() is True
    status = sensor.get_status()
    assert status.last_calibration is not None


def test_calibration_manager():
    sensor = MockSensor(tremor_frequency=5.0, tremor_amplitude=0.01, noise_level=0.01, seed=42)
    cal = CalibrationManager(sensor)
    result = cal.run_calibration(n_samples=3, duration=1.0)
    assert result.success is True
    assert result.baseline_mean is not None


def test_drift_detection_no_baseline():
    sensor = MockSensor(seed=42)
    cal = CalibrationManager(sensor)
    report = cal.check_drift()
    assert report.drift_detected is False
    assert "No baseline" in report.message
