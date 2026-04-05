"""Tests for MDS-UPDRS inspired severity scoring."""

import pytest

from neurosync.analysis.severity_scorer import SeverityScorer
from neurosync.core.models import SeverityLevel, TremorAnalysisResult


@pytest.fixture
def scorer():
    return SeverityScorer()


def _make_analysis(
    freq: float = 5.0,
    amplitude: float = 1.0,
    tremor_ratio: float = 0.3,
    psd_peak: float = 1.0,
) -> TremorAnalysisResult:
    return TremorAnalysisResult(
        dominant_frequency_hz=freq,
        power_spectral_density_peak=psd_peak,
        tremor_band_power=tremor_ratio * 10,
        total_power=10.0,
        tremor_ratio=tremor_ratio,
        amplitude_rms=amplitude,
    )


def test_no_tremor(scorer):
    analysis = _make_analysis(freq=0.0, amplitude=0.01, tremor_ratio=0.01)
    result = scorer.score(analysis)
    assert result.level == SeverityLevel.NONE
    assert result.score < 0.5


def test_slight_tremor(scorer):
    analysis = _make_analysis(freq=5.0, amplitude=0.1, tremor_ratio=0.1)
    result = scorer.score(analysis)
    assert result.level == SeverityLevel.SLIGHT
    assert 0.5 <= result.score < 1.5


def test_moderate_tremor(scorer):
    analysis = _make_analysis(freq=5.0, amplitude=0.8, tremor_ratio=0.35)
    result = scorer.score(analysis)
    assert result.level in (SeverityLevel.MILD, SeverityLevel.MODERATE)
    assert result.score >= 1.5


def test_severe_tremor(scorer):
    analysis = _make_analysis(freq=5.0, amplitude=2.0, tremor_ratio=0.6)
    result = scorer.score(analysis)
    assert result.level == SeverityLevel.SEVERE
    assert result.score >= 3.5


def test_clinical_notes_generated(scorer):
    analysis = _make_analysis(freq=5.0, amplitude=1.0, tremor_ratio=0.4)
    result = scorer.score(analysis)
    assert len(result.clinical_notes) > 0
    assert any("5.0 Hz" in note for note in result.clinical_notes)


def test_subscores_present(scorer):
    analysis = _make_analysis()
    result = scorer.score(analysis)
    assert "tremor_ratio" in result.subscores
    assert "amplitude" in result.subscores
    assert "frequency" in result.subscores
