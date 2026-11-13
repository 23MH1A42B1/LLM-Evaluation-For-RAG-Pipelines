#!/usr/bin/env python3
"""
Main evaluation harness runner:
Executes evaluation for Baseline (Pipeline A) and Candidate (Pipeline B)
across the pinned evaluation dataset, records detailed metrics for both pipelines,
and generates the regression report.
"""
import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import PipelineA, PipelineB
from src.evaluator import LLMJudge, compute_recall
from src.metrics import generate_regression_report


def run_evaluation():
    eval_set_path = "dataset/eval_set.json"
    pins_path = "config/eval_pins.json"
    prompts_path = "prompts/judge_prompts.json"
    corpus_path = "dataset/corpus.json"
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" RAG REGRESSION EVALUATION HARNESS")
    print("=" * 70)

    # Load eval dataset
    with open(eval_set_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    # Load pins
    with open(pins_path, "r", encoding="utf-8") as f:
        pins = json.load(f)

    print(f"Loaded {len(eval_set)} evaluation cases from {eval_set_path}")
    print(f"Pinned Judge Model: {pins.get('llm_judge_model')}")
    print(f"Pipeline A Embedder: {pins.get('pipeline_a_embedder')}")
    print(f"Pipeline B Embedder: {pins.get('pipeline_b_embedder')}")
    print("-" * 70)

    # Initialize pipelines & judge
    print("Initializing Pipeline A (Baseline - Naive Chunking, Top-k=2)...")
    pipe_a = PipelineA(config_path=pins_path, corpus_path=corpus_path)

    print("Initializing Pipeline B (Candidate - Top-k=5 + Reranker)...")
    pipe_b = PipelineB(config_path=pins_path, corpus_path=corpus_path)

    judge = LLMJudge(prompts_path=prompts_path, config_pins_path=pins_path)

    eval_a_details = []
    eval_b_details = []

    print("\nRunning Evaluation on Pipeline A & Pipeline B...")
    for idx, item in enumerate(eval_set, 1):
        q_id = item["id"]
        question = item["question"]
        gt_answer = item["ground_truth_answer"]
        gt_titles = item["ground_truth_context_titles"]

        print(f"[{idx:02d}/{len(eval_set):02d}] Evaluating {q_id}...", end="\r")

        # Evaluate Pipeline A
        res_a = pipe_a.retrieve_and_generate(question, item_id=q_id)
        recall_a = compute_recall(res_a["context_titles"], gt_titles)
        corr_a = judge.evaluate_correctness(question, gt_answer, res_a["answer"])
        ground_a = judge.evaluate_groundedness(res_a["answer"], res_a["retrieved_contexts"])

        eval_a_details.append({
            "id": q_id,
            "generated_answer": res_a["answer"],
            "retrieved_context_titles": res_a["context_titles"],
            "metrics": {
                "recall": float(recall_a),
                "correctness": int(corr_a),
                "groundedness": int(ground_a),
                "latency_ms": float(res_a["latency_ms"])
            }
        })

        # Evaluate Pipeline B
        res_b = pipe_b.retrieve_and_generate(question, item_id=q_id)
        recall_b = compute_recall(res_b["context_titles"], gt_titles)
        corr_b = judge.evaluate_correctness(question, gt_answer, res_b["answer"])
        ground_b = judge.evaluate_groundedness(res_b["answer"], res_b["retrieved_contexts"])

        eval_b_details.append({
            "id": q_id,
            "generated_answer": res_b["answer"],
            "retrieved_context_titles": res_b["context_titles"],
            "metrics": {
                "recall": float(recall_b),
                "correctness": int(corr_b),
                "groundedness": int(ground_b),
                "latency_ms": float(res_b["latency_ms"])
            }
        })

    print(f"\nCompleted evaluation of {len(eval_set)} samples.")

    # Save detailed evaluation files
    path_a = results_dir / "eval_A_details.json"
    with open(path_a, "w", encoding="utf-8") as f:
        json.dump(eval_a_details, f, indent=2)
    print(f"Saved: {path_a} ({len(eval_a_details)} items)")

    path_b = results_dir / "eval_B_details.json"
    with open(path_b, "w", encoding="utf-8") as f:
        json.dump(eval_b_details, f, indent=2)
    print(f"Saved: {path_b} ({len(eval_b_details)} items)")

    # Generate regression report
    report_csv = results_dir / "regression_report.csv"
    report_rows = generate_regression_report(eval_a_details, eval_b_details, str(report_csv))
    print(f"Saved: {report_csv}")

    # Print summary table
    print("\n" + "=" * 70)
    print(" REGRESSION REPORT SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<16} | {'Baseline (A)':<14} | {'Candidate (B)':<14} | {'Delta':<10} | {'Flag':<8}")
    print("-" * 70)
    for row in report_rows:
        delta_str = f"{row['delta']:+.4f}" if isinstance(row['delta'], float) and abs(row['delta']) < 1 else f"{row['delta']:+.2f}"
        print(f"{row['metric']:<16} | {row['baseline']:<14} | {row['candidate']:<14} | {delta_str:<10} | {row['flag']:<8}")
    print("=" * 70)

    # Validate demonstrated regression requirement
    correctness_row = next(r for r in report_rows if r["metric"] == "Correctness")
    groundedness_row = next(r for r in report_rows if r["metric"] == "Groundedness")
    latency_row = next(r for r in report_rows if r["metric"] == "P95_Latency")

    is_correctness_ok = correctness_row["candidate"] >= correctness_row["baseline"]
    is_regression_ok = (groundedness_row["candidate"] < groundedness_row["baseline"]) or (latency_row["candidate"] > latency_row["baseline"])
    has_regress_flag = any(r["flag"] == "REGRESS" for r in report_rows)

    if is_correctness_ok and is_regression_ok and has_regress_flag:
        print("\n[SUCCESS] Regression harness successfully verified trade-off failure mode!")
        print("Candidate maintains/improves Correctness while safely surfacing intentional regressions.")
    else:
        print("\n[WARNING] Intentional regression condition criteria not met!")


if __name__ == "__main__":
    run_evaluation()

