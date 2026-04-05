"""MDS-UPDRS inspired tremor severity scoring algorithm.

The Movement Disorder Society Unified Parkinson's Disease Rating Scale (MDS-UPDRS)
Part III Item 3.15-3.18 rates tremor on a 0-4 scale:
  0 = Normal (no tremor)
  1 = Slight (tremor present but < 1 cm amplitude)
  2 = Mild (tremor amplitude 1-3 cm)
  3 = Moderate (tremor amplitude 3-10 cm)
  4 = Severe (tremor amplitude > 10 cm)

This scorer uses signal analysis features as proxies for clinical assessment.
"""

from __future__ import annotations

from neurosync.core.models import SeverityAssessment, SeverityLevel, TremorAnalysisResult

# Thresholds derived from clinical literature correlating accelerometer
# features with MDS-UPDRS ratings
_AMPLITUDE_THRESHOLDS = {
    SeverityLevel.NONE: 0.05,
    SeverityLevel.SLIGHT: 0.2,
    SeverityLevel.MILD: 0.5,
    SeverityLevel.MODERATE: 1.5,
    SeverityLevel.SEVERE: float("inf"),
}

_TREMOR_RATIO_WEIGHT = 2.0
_AMPLITUDE_WEIGHT = 1.5
_FREQUENCY_WEIGHT = 0.5


class SeverityScorer:
    """Score tremor severity based on spectral and time-domain features."""

    def score(self, analysis: TremorAnalysisResult) -> SeverityAssessment:
        # Sub-score 1: Tremor ratio (proportion of power in tremor band)
        ratio_score = self._score_tremor_ratio(analysis.tremor_ratio)

        # Sub-score 2: Amplitude (RMS)
        amplitude_score = self._score_amplitude(analysis.amplitude_rms)

        # Sub-score 3: Frequency quality (how close to typical PD tremor)
        frequency_score = self._score_frequency(analysis.dominant_frequency_hz)

        # Weighted composite score (0-4 scale)
        total_weight = _TREMOR_RATIO_WEIGHT + _AMPLITUDE_WEIGHT + _FREQUENCY_WEIGHT
        composite = (
            ratio_score * _TREMOR_RATIO_WEIGHT
            + amplitude_score * _AMPLITUDE_WEIGHT
            + frequency_score * _FREQUENCY_WEIGHT
        ) / total_weight

        # Clamp to 0-4
        composite = max(0.0, min(4.0, composite))

        level = self._score_to_level(composite)
        notes = self._generate_notes(analysis, composite, level)

        return SeverityAssessment(
            score=round(composite, 2),
            level=level,
            subscores={
                "tremor_ratio": round(ratio_score, 2),
                "amplitude": round(amplitude_score, 2),
                "frequency": round(frequency_score, 2),
            },
            clinical_notes=notes,
        )

    @staticmethod
    def _score_tremor_ratio(ratio: float) -> float:
        """Map tremor ratio (0-1) to 0-4 score."""
        if ratio < 0.05:
            return 0.0
        if ratio < 0.15:
            return 1.0
        if ratio < 0.3:
            return 2.0
        if ratio < 0.5:
            return 3.0
        return 4.0

    @staticmethod
    def _score_amplitude(rms: float) -> float:
        """Map RMS amplitude to 0-4 score."""
        if rms < 0.05:
            return 0.0
        if rms < 0.2:
            return 1.0
        if rms < 0.5:
            return 2.0
        if rms < 1.5:
            return 3.0
        return 4.0

    @staticmethod
    def _score_frequency(freq_hz: float) -> float:
        """Score based on how characteristic the frequency is of PD tremor.

        Classic PD resting tremor is 4-6 Hz. Score higher if in this band.
        """
        if freq_hz <= 0:
            return 0.0
        # Peak score at 5 Hz (classic PD tremor)
        deviation = abs(freq_hz - 5.0)
        if deviation < 1.0:
            return 3.0  # Classic PD range
        if deviation < 2.0:
            return 2.0  # Probable PD
        if deviation < 4.0:
            return 1.0  # Possible tremor
        return 0.5  # Atypical

    @staticmethod
    def _score_to_level(score: float) -> SeverityLevel:
        if score < 0.5:
            return SeverityLevel.NONE
        if score < 1.5:
            return SeverityLevel.SLIGHT
        if score < 2.5:
            return SeverityLevel.MILD
        if score < 3.5:
            return SeverityLevel.MODERATE
        return SeverityLevel.SEVERE

    @staticmethod
    def _generate_notes(
        analysis: TremorAnalysisResult, score: float, level: SeverityLevel
    ) -> list[str]:
        notes = []
        if analysis.dominant_frequency_hz > 0:
            notes.append(f"Dominant tremor frequency: {analysis.dominant_frequency_hz:.1f} Hz")
            if 4.0 <= analysis.dominant_frequency_hz <= 6.0:
                notes.append("Frequency consistent with classic Parkinsonian resting tremor")
            elif 3.0 <= analysis.dominant_frequency_hz <= 7.0:
                notes.append("Frequency within Parkinsonian tremor range")
            else:
                notes.append("Frequency outside typical Parkinsonian tremor band")

        if analysis.tremor_ratio > 0.3:
            notes.append(
                f"High tremor band power ratio ({analysis.tremor_ratio:.1%}) "
                "suggests prominent tremor activity"
            )

        if level in (SeverityLevel.MODERATE, SeverityLevel.SEVERE):
            notes.append("Consider medication review or dosage adjustment")

        return notes
