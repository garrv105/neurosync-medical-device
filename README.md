# NeuroSync Medical Device Monitoring

[![CI](https://github.com/garrv105/neurosync-medical-device/actions/workflows/ci.yml/badge.svg)](https://github.com/garrv105/neurosync-medical-device/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A production-grade Python backend for real-time monitoring and analysis of Parkinsonian tremor data from wearable sensors. Built for clinical use with FDA 21 CFR Part 11 compliance.

## Features

- **Real Signal Processing**: FFT-based tremor analysis using Welch's PSD estimation, peak detection, and Hilbert transform for instantaneous frequency
- **Clinical Severity Scoring**: MDS-UPDRS inspired 0-4 scale scoring algorithm using spectral and time-domain features
- **Alert Engine**: Rule-based alerting with configurable thresholds, frequency band monitoring, and automatic escalation
- **Patient Management**: Full CRUD operations with medication tracking and treatment history
- **Data Pipeline**: Stream processing with signal cleaning, bandpass filtering, outlier removal, and buffered storage
- **Clinical Reports**: JSON and PDF report generation with severity trends and treatment recommendations
- **FDA Compliance**: 21 CFR Part 11 audit trail with SHA-256 checksums and integrity verification
- **REST API**: FastAPI server with endpoints for patients, readings, alerts, reports, and compliance
- **Mock Sensor**: Generates realistic Parkinsonian tremor waveforms (4-6 Hz resting tremor with amplitude modulation and phase jitter)

## Architecture

```
neurosync/
├── core/           # Domain models, tremor analyzer, patient manager, alert engine
├── devices/        # Sensor interface, calibration, data pipeline
├── analysis/       # Frequency analysis, severity scoring, trend detection
├── reporting/      # Clinical reports (JSON/PDF), FDA compliance
├── storage/        # SQLite database with audit trail
├── api/            # FastAPI REST server
└── cli.py          # Click CLI
```

## Quick Start

```bash
# Install
pip install -r requirements.txt
pip install -e .

# Run analysis
neurosync analyze --frequency 5.0 --amplitude 1.0

# Start API server
neurosync serve --port 8000

# Run compliance check
neurosync compliance

# Generate report
neurosync report <patient-id> --pdf
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/patients` | Register patient |
| GET | `/patients` | List patients |
| GET | `/patients/{id}` | Get patient |
| PUT | `/patients/{id}` | Update patient |
| DELETE | `/patients/{id}` | Delete patient |
| POST | `/readings` | Submit tremor reading (auto-analyzed) |
| GET | `/readings/{patient_id}` | Get patient readings |
| GET | `/alerts` | List alerts |
| PUT | `/alerts/{id}` | Update alert status |
| GET | `/reports/{patient_id}` | Generate clinical report |
| GET | `/compliance` | FDA compliance check |
| GET | `/audit` | Audit trail |

## Signal Processing

The tremor analysis pipeline uses real signal processing:

1. **Preprocessing**: DC removal, Hann windowing, linear detrending
2. **PSD Estimation**: Welch's method with 50% overlapping segments
3. **Band Analysis**: Power extraction in tremor (3-7 Hz), action tremor (8-12 Hz), and physiological bands
4. **Peak Detection**: `scipy.signal.find_peaks` with prominence and distance constraints
5. **Instantaneous Frequency**: Hilbert transform on bandpass-filtered signal

## Severity Scoring

Based on MDS-UPDRS Part III (tremor items 3.15-3.18):

| Score | Level | Description |
|-------|-------|-------------|
| 0 | None | No tremor detected |
| 1 | Slight | Minimal tremor activity |
| 2 | Mild | Low-amplitude tremor in PD band |
| 3 | Moderate | Prominent tremor, consider medication review |
| 4 | Severe | High-amplitude sustained tremor |

## Testing

```bash
pytest tests/ -v --cov=neurosync
```

## Docker

```bash
docker build -f docker/Dockerfile -t neurosync:latest .
docker run -p 8000:8000 neurosync:latest
```

## License

MIT
