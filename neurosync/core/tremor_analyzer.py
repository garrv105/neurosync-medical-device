"""Signal processing for tremor analysis: FFT, tremor frequency extraction, severity scoring."""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

from neurosync.core.models import TremorAnalysisResult

# Parkinson's tremor frequency band: 3-7 Hz (primary resting tremor 4-6 Hz)
TREMOR_BAND_LOW = 3.0
TREMOR_BAND_HIGH = 7.0


def analyze_tremor_signal(
    raw_signal: list[float] | np.ndarray,
    sample_rate_hz: float = 100.0,
) -> TremorAnalysisResult:
    """Analyze a raw accelerometer signal for Parkinsonian tremor characteristics.

    Uses Welch's method for PSD estimation and peak detection in the tremor band.
    """
    data = np.asarray(raw_signal, dtype=np.float64)

    # Remove DC offset
    data = data - np.mean(data)

    # Apply Hann window to reduce spectral leakage
    n_samples = len(data)
    nperseg = min(256, n_samples)
    noverlap = nperseg // 2

    # Welch's method for power spectral density
    frequencies, psd = sp_signal.welch(
        data,
        fs=sample_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="linear",
    )

    # Find tremor band indices
    tremor_mask = (frequencies >= TREMOR_BAND_LOW) & (frequencies <= TREMOR_BAND_HIGH)
    tremor_freqs = frequencies[tremor_mask]
    tremor_psd = psd[tremor_mask]

    # Dominant frequency: frequency with highest PSD in tremor band
    if len(tremor_psd) > 0 and np.max(tremor_psd) > 0:
        peak_idx = np.argmax(tremor_psd)
        dominant_frequency = float(tremor_freqs[peak_idx])
        psd_peak = float(tremor_psd[peak_idx])
    else:
        dominant_frequency = 0.0
        psd_peak = 0.0

    # Compute band powers
    freq_resolution = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 1.0
    tremor_band_power = float(np.sum(tremor_psd) * freq_resolution)
    total_power = float(np.sum(psd) * freq_resolution)

    # Tremor ratio: proportion of power in tremor band
    tremor_ratio = tremor_band_power / total_power if total_power > 0 else 0.0

    # RMS amplitude
    amplitude_rms = float(np.sqrt(np.mean(data**2)))

    return TremorAnalysisResult(
        dominant_frequency_hz=dominant_frequency,
        power_spectral_density_peak=psd_peak,
        tremor_band_power=tremor_band_power,
        total_power=total_power,
        tremor_ratio=tremor_ratio,
        amplitude_rms=amplitude_rms,
        frequencies=frequencies.tolist(),
        psd_values=psd.tolist(),
    )


def detect_tremor_peaks(
    raw_signal: list[float] | np.ndarray,
    sample_rate_hz: float = 100.0,
    min_prominence: float = 0.1,
) -> list[int]:
    """Detect individual tremor peaks in the time-domain signal."""
    data = np.asarray(raw_signal, dtype=np.float64)
    data = data - np.mean(data)

    # Expected tremor period: ~0.17s to ~0.33s for 3-7 Hz
    min_distance = int(sample_rate_hz / TREMOR_BAND_HIGH)
    peaks, _ = sp_signal.find_peaks(data, distance=min_distance, prominence=min_prominence)
    return peaks.tolist()


def compute_instantaneous_frequency(
    raw_signal: list[float] | np.ndarray,
    sample_rate_hz: float = 100.0,
) -> np.ndarray:
    """Compute instantaneous frequency using the analytic signal (Hilbert transform)."""
    data = np.asarray(raw_signal, dtype=np.float64)
    data = data - np.mean(data)

    # Bandpass filter to tremor band first
    sos = sp_signal.butter(
        4,
        [TREMOR_BAND_LOW, TREMOR_BAND_HIGH],
        btype="bandpass",
        fs=sample_rate_hz,
        output="sos",
    )
    filtered = sp_signal.sosfiltfilt(sos, data)

    # Hilbert transform for analytic signal
    analytic = sp_signal.hilbert(filtered)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) / (2.0 * np.pi) * sample_rate_hz

    return inst_freq
