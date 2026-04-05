"""Tests for spectral analysis."""

import numpy as np
import pytest

from neurosync.analysis.frequency_analysis import (
    compute_psd,
    compute_spectral_features,
    find_spectral_peaks,
)


def test_psd_welch():
    fs = 100.0
    t = np.arange(0, 10, 1 / fs)
    signal = np.sin(2 * np.pi * 5.0 * t)
    freqs, psd = compute_psd(signal.tolist(), fs, method="welch")
    assert len(freqs) > 0
    assert len(psd) == len(freqs)
    # Peak should be near 5 Hz
    peak_idx = np.argmax(psd)
    assert 4.0 <= freqs[peak_idx] <= 6.0


def test_psd_periodogram():
    fs = 100.0
    t = np.arange(0, 10, 1 / fs)
    signal = np.sin(2 * np.pi * 5.0 * t)
    freqs, psd = compute_psd(signal.tolist(), fs, method="periodogram")
    peak_idx = np.argmax(psd)
    assert 4.0 <= freqs[peak_idx] <= 6.0


def test_psd_invalid_method():
    with pytest.raises(ValueError, match="Unknown method"):
        compute_psd([1.0] * 100, 100.0, method="invalid")


def test_spectral_features():
    fs = 100.0
    t = np.arange(0, 10, 1 / fs)
    signal = np.sin(2 * np.pi * 5.0 * t)
    features = compute_spectral_features(signal.tolist(), fs)
    assert 4.0 <= features.peak_frequency <= 6.0
    assert features.total_power > 0
    assert features.spectral_entropy > 0
    assert "tremor" in features.band_powers
    assert features.band_powers["tremor"] > features.band_powers.get("high_frequency", 0)


def test_spectral_peaks():
    fs = 100.0
    t = np.arange(0, 10, 1 / fs)
    # Two-frequency signal
    signal = np.sin(2 * np.pi * 5.0 * t) + 0.5 * np.sin(2 * np.pi * 10.0 * t)
    peaks = find_spectral_peaks(signal.tolist(), fs, max_peaks=5)
    assert len(peaks) >= 1
    freqs = [p[0] for p in peaks]
    # Should find a peak near 5 Hz
    assert any(4.0 <= f <= 6.0 for f in freqs)
