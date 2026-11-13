"""
Unit tests for metric calculation, flag determination, and edge cases.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.evaluator import compute_recall
from src.metrics import determine_flag, compute_p95, calculate_aggregates


def test_compute_recall_perfect():
    retrieved = ["Doc A", "Doc B", "Doc C"]
    gt = ["Doc A", "Doc B"]
    recall = compute_recall(retrieved, gt)
    assert recall == 1.0


def test_compute_recall_partial():
    retrieved = ["Doc A", "Doc C"]
    gt = ["Doc A", "Doc B"]
    recall = compute_recall(retrieved, gt)
    assert recall == 0.5


def test_compute_recall_zero():
    retrieved = ["Doc X", "Doc Y"]
    gt = ["Doc A", "Doc B"]
    recall = compute_recall(retrieved, gt)
    assert recall == 0.0


def test_determine_flag_quality_metrics():
    # Candidate < Baseline -> REGRESS
    assert determine_flag("Recall", 0.85, 0.70) == "REGRESS"
    assert determine_flag("Correctness", 0.90, 0.80) == "REGRESS"
    assert determine_flag("Groundedness", 0.95, 0.65) == "REGRESS"

    # Candidate > Baseline -> IMPROVE
    assert determine_flag("Recall", 0.70, 0.85) == "IMPROVE"
    assert determine_flag("Correctness", 0.80, 0.95) == "IMPROVE"
    assert determine_flag("Groundedness", 0.70, 0.90) == "IMPROVE"

    # Candidate == Baseline -> NEUTRAL
    assert determine_flag("Recall", 0.80, 0.80) == "NEUTRAL"
    assert determine_flag("Correctness", 0.90, 0.90) == "NEUTRAL"


def test_determine_flag_p95_latency():
    # Latency increases by > 10% -> REGRESS
    # 200 * 1.10 = 220
    assert determine_flag("P95_Latency", 200.0, 225.0) == "REGRESS"
    assert determine_flag("P95_Latency", 100.0, 115.0) == "REGRESS"

    # Latency increases within 10% -> NEUTRAL
    assert determine_flag("P95_Latency", 200.0, 210.0) == "NEUTRAL"
    assert determine_flag("P95_Latency", 200.0, 200.0) == "NEUTRAL"

    # Latency decreases -> IMPROVE
    assert determine_flag("P95_Latency", 200.0, 180.0) == "IMPROVE"


def test_compute_p95_values():
    latencies = [10.0 * i for i in range(1, 101)] # 10 to 1000
    p95 = compute_p95(latencies)
    # 95th percentile of 10..1000 with 100 values is approx 950.55
    assert 940.0 <= p95 <= 960.0


def test_calculate_aggregates():
    sample_results = [
        {
            "id": "1",
            "metrics": {"recall": 1.0, "correctness": 1, "groundedness": 1, "latency_ms": 100.0}
        },
        {
            "id": "2",
            "metrics": {"recall": 0.5, "correctness": 0, "groundedness": 0, "latency_ms": 200.0}
        }
    ]
    agg = calculate_aggregates(sample_results)
    assert agg["Recall"] == 0.75
    assert agg["Correctness"] == 0.5
    assert agg["Groundedness"] == 0.5
    assert agg["P95_Latency"] > 150.0
