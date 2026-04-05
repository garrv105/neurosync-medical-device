"""Tests for core tremor signal analysis."""

import numpy as np

from neurosync.core.tremor_analyzer import (
    analyze_tremor_signal,
    compute_instantaneous_frequency,
    detect_tremor_peaks,
)


def test_analyze_pure_5hz_signal():
    """A pure 5 Hz sine wave should have dominant frequency near 5 Hz."""
    fs = 100.0
    t = np.arange(0, 10, 1 / fs)
    signal = np.sin(2 * np.pi * 5.0 * t)
    result = analyze_tremor_signal(signal.tolist(), fs)
    assert 4.5 <= result.dominant_frequency_hz <= 5.5
    assert result.tremor_ratio > 0.5
    assert result.amplitude_rms > 0


def test_analyze_signal_outside_tremor_band():
    """A 20 Hz signal should have low tremor ratio."""
    fs = 100.0
    t = np.arange(0, 10, 1 / fs)
    signal = np.sin(2 * np.pi * 20.0 * t)
    result = analyze_tremor_signal(signal.tolist(), fs)
    assert result.tremor_ratio < 0.2


def test_analyze_dc_signal():
    """A constant (DC) signal should show near-zero tremor."""
    signal = [1.0] * 1000
    result = analyze_tremor_signal(signal, 100.0)
    assert result.amplitude_rms < 0.01
    assert result.tremor_band_power < 0.01


def test_analyze_returns_psd_data():
    """Analysis result should include frequency and PSD arrays."""
    fs = 100.0
    t = np.arange(0, 5, 1 / fs)
    signal = np.sin(2 * np.pi * 5.0 * t)
    result = analyze_tremor_signal(signal.tolist(), fs)
    assert len(result.frequencies) > 0
    assert len(result.psd_values) > 0
    assert len(result.frequencies) == len(result.psd_values)


def test_detect_tremor_peaks():
    """Peak detection should find oscillation peaks in a tremor signal."""
    fs = 100.0
    t = np.arange(0, 5, 1 / fs)
    signal = np.sin(2 * np.pi * 5.0 * t)
    peaks = detect_tremor_peaks(signal.tolist(), fs, min_prominence=0.5)
    # 5 Hz for 5 seconds = ~25 peaks
    assert 20 <= len(peaks) <= 30


def test_instantaneous_frequency():
    """Instantaneous frequency of a 5 Hz signal should be near 5 Hz."""
    fs = 100.0
    t = np.arange(0, 10, 1 / fs)
    signal = np.sin(2 * np.pi * 5.0 * t)
    inst_freq = compute_instantaneous_frequency(signal.tolist(), fs)
    # Middle portion should be stable near 5 Hz
    mid = inst_freq[len(inst_freq) // 4 : 3 * len(inst_freq) // 4]
    mean_freq = np.mean(mid)
    assert 4.0 <= mean_freq <= 6.0
