"""Click CLI for NeuroSync: start server, run analysis, generate reports."""

from __future__ import annotations

import json

import click

from neurosync.analysis.severity_scorer import SeverityScorer
from neurosync.core.tremor_analyzer import analyze_tremor_signal
from neurosync.devices.sensor_interface import MockSensor, SensorConfig
from neurosync.reporting.compliance import ComplianceChecker
from neurosync.storage.database import Database


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """NeuroSync: Medical device monitoring for Parkinson's disease patients."""


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--db", default="neurosync.db", help="Database path")
def serve(host: str, port: int, db: str):
    """Start the NeuroSync API server."""
    import uvicorn

    from neurosync.api.server import init_app

    init_app(db)
    uvicorn.run("neurosync.api.server:app", host=host, port=port, reload=False)


@cli.command()
@click.option("--frequency", default=5.0, help="Tremor frequency in Hz")
@click.option("--amplitude", default=1.0, help="Tremor amplitude")
@click.option("--duration", default=10.0, help="Signal duration in seconds")
@click.option("--sample-rate", default=100.0, help="Sample rate in Hz")
@click.option("--json-output", is_flag=True, help="Output as JSON")
def analyze(
    frequency: float, amplitude: float, duration: float, sample_rate: float, json_output: bool
):
    """Run tremor analysis on a mock sensor signal."""
    sensor = MockSensor(tremor_frequency=frequency, tremor_amplitude=amplitude, seed=42)
    config = SensorConfig(sample_rate_hz=sample_rate, duration_seconds=duration)
    signal = sensor.read(config)

    result = analyze_tremor_signal(signal, sample_rate)
    scorer = SeverityScorer()
    assessment = scorer.score(result)

    if json_output:
        output = {
            "dominant_frequency_hz": result.dominant_frequency_hz,
            "amplitude_rms": result.amplitude_rms,
            "tremor_ratio": result.tremor_ratio,
            "severity_score": assessment.score,
            "severity_level": assessment.level.value,
            "clinical_notes": assessment.clinical_notes,
        }
        click.echo(json.dumps(output, indent=2))
    else:
        click.echo(f"Dominant frequency: {result.dominant_frequency_hz:.2f} Hz")
        click.echo(f"Amplitude (RMS):   {result.amplitude_rms:.4f}")
        click.echo(f"Tremor ratio:      {result.tremor_ratio:.2%}")
        click.echo(f"Severity score:    {assessment.score:.2f} / 4.0")
        click.echo(f"Severity level:    {assessment.level.value}")
        for note in assessment.clinical_notes:
            click.echo(f"  - {note}")


@cli.command()
@click.option("--db", default="neurosync.db", help="Database path")
def compliance(db: str):
    """Run FDA 21 CFR Part 11 compliance check."""
    database = Database(db)
    checker = ComplianceChecker(database)
    result = checker.run_compliance_check()

    click.echo(f"Compliant: {result.compliant}")
    click.echo(f"Audit entries: {result.total_audit_entries}")
    if result.checks_passed:
        click.echo("\nPassed:")
        for c in result.checks_passed:
            click.echo(f"  [PASS] {c}")
    if result.checks_failed:
        click.echo("\nFailed:")
        for c in result.checks_failed:
            click.echo(f"  [FAIL] {c}")
    database.close()


@cli.command()
@click.argument("patient_id")
@click.option("--db", default="neurosync.db", help="Database path")
@click.option("--pdf", is_flag=True, help="Generate PDF report")
@click.option("--output", default=None, help="Output file path")
def report(patient_id: str, db: str, pdf: bool, output: str | None):
    """Generate a clinical report for a patient."""
    from neurosync.reporting.clinical_report import ClinicalReportGenerator

    database = Database(db)
    gen = ClinicalReportGenerator(database)

    if pdf:
        pdf_bytes = gen.generate_pdf(patient_id)
        path = output or f"report_{patient_id[:8]}.pdf"
        with open(path, "wb") as f:
            f.write(pdf_bytes)
        click.echo(f"PDF report saved to {path}")
    else:
        report_data = gen.generate_report(patient_id)
        if output:
            with open(output, "w") as f:
                json.dump(report_data.model_dump(), f, indent=2, default=str)
            click.echo(f"Report saved to {output}")
        else:
            click.echo(json.dumps(report_data.model_dump(), indent=2, default=str))

    database.close()


if __name__ == "__main__":
    cli()
