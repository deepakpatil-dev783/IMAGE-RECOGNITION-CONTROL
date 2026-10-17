"""
tests/test_reporting.py
"""
import json
import pytest
import tempfile
from pathlib import Path
from src.reporting.report_generator import ReportGenerator


MOCK_CONFIG = {
    "paths": {"reports_dir": None},   # set per-test
    "reporting": {
        "company_name": "Test Co.",
        "report_format": "json",
    }
}

MOCK_RESULTS = [
    {
        "inspection_id": "abc123",
        "timestamp": "2024-01-01T10:00:00",
        "image_path": "data/raw/PASS/img_001.jpg",
        "verdict": "PASS",
        "confidence": 0.97,
        "pass_probability": 0.97,
        "defect_probability": 0.03,
        "defect_code": None,
        "heatmap_path": None,
    },
    {
        "inspection_id": "def456",
        "timestamp": "2024-01-01T10:01:00",
        "image_path": "data/raw/DEFECT/img_002.jpg",
        "verdict": "DEFECT",
        "confidence": 0.92,
        "pass_probability": 0.08,
        "defect_probability": 0.92,
        "defect_code": {"code": "D001", "description": "Surface Scratch"},
        "heatmap_path": "data/reports/heatmaps/img_002_heatmap.jpg",
    },
]


def test_json_report(tmp_path):
    config = dict(MOCK_CONFIG)
    config["paths"] = {"reports_dir": str(tmp_path)}
    reporter = ReportGenerator(config)
    path = reporter.save_json(MOCK_RESULTS, "test_batch_001")
    assert Path(path).exists()
    with open(path) as f:
        data = json.load(f)
    assert data["summary"]["total"] == 2
    assert data["summary"]["passed"] == 1
    assert data["summary"]["defects"] == 1
    assert abs(data["summary"]["pass_rate"] - 50.0) < 0.1


def test_summary_calculation():
    summary = ReportGenerator._build_summary(MOCK_RESULTS)
    assert summary["total"] == 2
    assert summary["pass_rate"] == 50.0
    assert "D001" in summary["defect_breakdown"]
