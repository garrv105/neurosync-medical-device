"""Spectral analysis of tremor data: FFT, PSD, peak detection.

Provides detailed spectral analysis beyond the core tremor analyzer,
including multi-band decomposition and spectral entropy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sp_signal


@dataclass
class SpectralFeatures:
    peak_frequency: float
    peak_power: float
    mean_frequency: float
    median_frequency: float
    spectral_entropy: float
    band_powers: dict[str, float]
    total_power: float


# Frequency bands relevant to Parkinsonian tremor analysis
FREQUENCY_BANDS = {
    "sub_tremor": (0.5, 3.0),
    "tremor": (3.0, 7.0),
    "action_tremor": (7.0, 12.0),
    "high_frequency": (12.0, 30.0),
}


def compute_psd(
    signal_data: list[float] | np.ndarray,
    sample_rate_hz: float = 100.0,
    method: str = "welch",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density using specified method.

    Returns (frequencies, psd_values).
    """
    data = np.asarray(signal_data, dtype=np.float64)
    data = data - np.mean(data)

    n_samples = len(data)
    nperseg = min(256, n_samples)

    if method == "welch":
        freqs, psd = sp_signal.welch(
            data, fs=sample_rate_hz, nperseg=nperseg, window="hann", detrend="linear"
        )
    elif method == "periodogram":
        freqs, psd = sp_signal.periodogram(data, fs=sample_rate_hz, window="hann", detrend="linear")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'welch' or 'periodogram'.")

    return freqs, psd


def compute_spectral_features(
    signal_data: list[float] | np.ndarray,
    sample_rate_hz: float = 100.0,
) -> SpectralFeatures:
    """Extract comprehensive spectral features from a signal."""
    freqs, psd = compute_psd(signal_data, sample_rate_hz)

    freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    # Peak frequency
    peak_idx = np.argmax(psd)
    peak_frequency = float(freqs[peak_idx])
    peak_power = float(psd[peak_idx])

    # Total power
    total_power = float(np.sum(psd) * freq_resolution)

    # Mean frequency (spectral centroid)
    if total_power > 0:
        mean_frequency = float(np.sum(freqs * psd * freq_resolution) / total_power)
    else:
        mean_frequency = 0.0

    # Median frequency (frequency below which 50% of power lies)
    cumulative_power = np.cumsum(psd * freq_resolution)
    half_power = total_power / 2
    median_idx = np.searchsorted(cumulative_power, half_power)
    median_frequency = float(freqs[min(median_idx, len(freqs) - 1)])

    # Spectral entropy
    psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
    psd_norm = psd_norm[psd_norm > 0]
    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm))) if len(psd_norm) > 0 else 0.0

    # Band powers
    band_powers = {}
    for band_name, (low, high) in FREQUENCY_BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        band_powers[band_name] = float(np.sum(psd[mask]) * freq_resolution)

    return SpectralFeatures(
        peak_frequency=peak_frequency,
        peak_power=peak_power,
        mean_frequency=mean_frequency,
        median_frequency=median_frequency,
        spectral_entropy=spectral_entropy,
        band_powers=band_powers,
        total_power=total_power,
    )


def find_spectral_peaks(
    signal_data: list[float] | np.ndarray,
    sample_rate_hz: float = 100.0,
    min_height: float | None = None,
    max_peaks: int = 5,
) -> list[tuple[float, float]]:
    """Find prominent peaks in the power spectrum.

    Returns list of (frequency, power) tuples sorted by power descending.
    """
    freqs, psd = compute_psd(signal_data, sample_rate_hz)

    if min_height is None:
        min_height = np.mean(psd)

    peaks, properties = sp_signal.find_peaks(psd, height=min_height, distance=2)

    if len(peaks) == 0:
        return []

    peak_powers = psd[peaks]
    sorted_indices = np.argsort(peak_powers)[::-1][:max_peaks]

    return [(float(freqs[peaks[i]]), float(peak_powers[i])) for i in sorted_indices]
