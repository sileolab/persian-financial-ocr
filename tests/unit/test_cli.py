"""Unit tests for the CLI interface."""

from typer.testing import CliRunner
from hesabyar.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Hesabyar" in result.stdout
    assert "start" in result.stdout
    assert "status" in result.stdout
    assert "reset" in result.stdout


def test_cli_info():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "تنظیم" in result.stdout or "Configuration" in result.stdout
    assert "Hesabyar" in result.stdout


def test_cli_start(tmp_path):
    # Run start on an empty temporary directory
    result = runner.invoke(app, ["start", "--input", str(tmp_path)])
    assert result.exit_code == 0
    assert "input directory" in result.stdout.lower() or "input" in result.stdout.lower()
    assert "infrastructure loaded successfully" in result.stdout.lower() or "database" in result.stdout.lower()


def test_cli_status():
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "document processing overview" in result.stdout.lower() or "overview" in result.stdout.lower()
