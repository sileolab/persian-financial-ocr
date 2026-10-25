"""Unit tests for configuration management."""

import pytest
from pathlib import Path
from hesabyar.core.config import Settings, get_settings


def test_settings_default_values():
    settings = get_settings()
    assert settings.hesabyar_env in ["development", "production", "testing"]
    assert settings.data_dir.is_absolute()
    assert settings.db_path.is_absolute()
    assert settings.max_workers >= 1
    assert settings.batch_size >= 1


def test_ensure_directories(tmp_path: Path):
    custom_settings = Settings(
        data_dir=tmp_path / "custom_data",
        input_dir=tmp_path / "custom_data" / "input",
        output_dir=tmp_path / "custom_data" / "output",
        cache_dir=tmp_path / "custom_data" / "cache",
        ground_truth_dir=tmp_path / "custom_data" / "gt",
        db_path=tmp_path / "custom_data" / "state.db",
        log_file_path=tmp_path / "custom_data" / "app.log",
    )
    custom_settings.ensure_directories()

    assert custom_settings.input_dir.exists()
    assert custom_settings.output_dir.exists()
    assert custom_settings.cache_dir.exists()
    assert custom_settings.ground_truth_dir.exists()
    assert custom_settings.db_path.parent.exists()
    assert custom_settings.log_file_path.parent.exists()
