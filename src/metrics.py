"""
Metrics calculation, aggregation, delta calculation, and CSV regression report generation.
"""
import os
import csv
import numpy as np
from typing import List, Dict, Any


def compute_p95(latencies: List[float]) -> float:
    """Compute the 95th percentile latency."""
    if not latencies:
        return 0.0
    return float(np.percentile(latencies, 95))


def calculate_aggregates(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate mean recall, correctness, groundedness, and p95 latency."""
    recalls = [r["metrics"]["recall"] for r in results]
    correctnesses = [r["metrics"]["correctness"] for r in results]
    groundednesses = [r["metrics"]["groundedness"] for r in results]
    latencies = [r["metrics"]["latency_ms"] for r in results]

    return {
        "Recall": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        "Correctness": round(float(np.mean(correctnesses)), 4) if correctnesses else 0.0,
        "Groundedness": round(float(np.mean(groundednesses)), 4) if groundednesses else 0.0,
        "P95_Latency": round(compute_p95(latencies), 2)
    }


def determine_flag(metric_name: str, baseline: float, candidate: float) -> str:
    """
    Determine regression flag based on contract rules:
    - For Recall, Correctness, Groundedness:
      If candidate < baseline, flag is REGRESS.
      Else if candidate > baseline, flag is IMPROVE.
      Else NEUTRAL.
    - For P95_Latency:
      If candidate > baseline + 10% (candidate > baseline * 1.10), flag is REGRESS.
      Else if candidate < baseline, flag is IMPROVE.
      Else NEUTRAL.
    """
    if metric_name == "P95_Latency":
        if candidate > (baseline * 1.10):
            return "REGRESS"
        elif candidate < baseline:
            return "IMPROVE"
        else:
            return "NEUTRAL"
    else:
        if candidate < baseline:
            return "REGRESS"
        elif candidate > baseline:
            return "IMPROVE"
        else:
            return "NEUTRAL"


def generate_regression_report(
    results_a: List[Dict[str, Any]],
    results_b: List[Dict[str, Any]],
    output_csv_path: str = "results/regression_report.csv"
) -> List[Dict[str, Any]]:
    """
    Generate unified regression report CSV.
    Headers: metric,baseline,candidate,delta,flag
    Metrics: Recall, Correctness, Groundedness, P95_Latency
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    agg_a = calculate_aggregates(results_a)
    agg_b = calculate_aggregates(results_b)

    metrics_order = ["Recall", "Correctness", "Groundedness", "P95_Latency"]
    rows = []

    for metric in metrics_order:
        base_val = agg_a[metric]
        cand_val = agg_b[metric]
        delta_val = round(cand_val - base_val, 4)
        flag = determine_flag(metric, base_val, cand_val)

        rows.append({
            "metric": metric,
            "baseline": base_val,
            "candidate": cand_val,
            "delta": delta_val,
            "flag": flag
        })

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["metric", "baseline", "candidate", "delta", "flag"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return rows
