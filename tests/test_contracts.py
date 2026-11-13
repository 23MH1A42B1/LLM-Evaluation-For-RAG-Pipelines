"""
Contract and verification tests for the LLM Evaluation Regression Harness.
Validates all 10 Core Requirements against exact schemas, files, formulas, and constraints.
"""
import os
import csv
import json
import pytest
import yaml
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_core_req_1_eval_dataset_schema():
    """Core Requirement 1: Dataset schema and structure verification."""
    eval_file = ROOT_DIR / "dataset" / "eval_set.json"
    assert eval_file.exists(), f"Missing dataset file at {eval_file}"

    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list), "eval_set.json must be a JSON array"
    assert 30 <= len(data) <= 50, f"Expected 30 to 50 samples, got {len(data)}"

    for idx, item in enumerate(data):
        assert isinstance(item, dict), f"Item {idx} is not an object"
        assert "id" in item and isinstance(item["id"], str) and len(item["id"]) > 0, f"Item {idx} missing valid id"
        assert "question" in item and isinstance(item["question"], str), f"Item {idx} missing valid question"
        assert "ground_truth_answer" in item and isinstance(item["ground_truth_answer"], str), f"Item {idx} missing valid ground_truth_answer"
        assert "ground_truth_context_titles" in item and isinstance(item["ground_truth_context_titles"], list), f"Item {idx} missing valid ground_truth_context_titles"
        assert len(item["ground_truth_context_titles"]) > 0, f"Item {idx} ground_truth_context_titles list is empty"
        for title in item["ground_truth_context_titles"]:
            assert isinstance(title, str), f"Title {title} is not a string"


def test_core_req_2_reproducibility_pins():
    """Core Requirement 2: Configuration pins verification."""
    pins_file = ROOT_DIR / "config" / "eval_pins.json"
    assert pins_file.exists(), f"Missing config file at {pins_file}"

    with open(pins_file, "r", encoding="utf-8") as f:
        pins = json.load(f)

    assert isinstance(pins, dict), "eval_pins.json must be a JSON object"

    required_keys = ["dataset_size", "llm_judge_model", "pipeline_a_embedder", "pipeline_b_embedder"]
    for key in required_keys:
        assert key in pins, f"Missing required key '{key}' in eval_pins.json"

    assert isinstance(pins["dataset_size"], (int, float)), "dataset_size must be a number"
    assert isinstance(pins["llm_judge_model"], str) and len(pins["llm_judge_model"]) > 0, "llm_judge_model must be a non-empty string"
    assert isinstance(pins["pipeline_a_embedder"], str) and len(pins["pipeline_a_embedder"]) > 0, "pipeline_a_embedder must be a non-empty string"
    assert isinstance(pins["pipeline_b_embedder"], str) and len(pins["pipeline_b_embedder"]) > 0, "pipeline_b_embedder must be a non-empty string"


def test_core_req_3_judge_prompts():
    """Core Requirement 3: Judge prompts documentation and strict rubrics."""
    prompts_file = ROOT_DIR / "prompts" / "judge_prompts.json"
    assert prompts_file.exists(), f"Missing prompts file at {prompts_file}"

    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    assert isinstance(prompts, dict), "judge_prompts.json must be a JSON object"
    assert "correctness_prompt" in prompts, "Missing 'correctness_prompt'"
    assert "groundedness_prompt" in prompts, "Missing 'groundedness_prompt'"

    corr = prompts["correctness_prompt"]
    ground = prompts["groundedness_prompt"]

    assert isinstance(corr, str) and len(corr) > 50, "correctness_prompt is too short or not string"
    assert isinstance(ground, str) and len(ground) > 50, "groundedness_prompt is too short or not string"

    # Verify groundedness_prompt forces scoring based strictly on context
    ground_lower = ground.lower()
    assert "strictly" in ground_lower or "context" in ground_lower, "groundedness_prompt must emphasize strict context adherence"
    assert "cannot be directly traced" in ground_lower or "unsupported" in ground_lower or "score 0" in ground_lower, \
        "groundedness_prompt must penalize any claims not directly traced to context"


def test_core_req_4_eval_details_a():
    """Core Requirement 4: Question-by-question metrics for Pipeline A."""
    file_a = ROOT_DIR / "results" / "eval_A_details.json"
    eval_file = ROOT_DIR / "dataset" / "eval_set.json"
    assert file_a.exists(), f"Missing {file_a}"

    with open(eval_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    with open(file_a, "r", encoding="utf-8") as f:
        details_a = json.load(f)

    assert isinstance(details_a, list), "eval_A_details.json must be a list"
    assert len(details_a) == len(dataset), f"Length of details A ({len(details_a)}) does not match dataset ({len(dataset)})"

    dataset_ids = [d["id"] for d in dataset]
    for idx, item in enumerate(details_a):
        assert isinstance(item, dict)
        assert item["id"] == dataset_ids[idx], f"ID mismatch at index {idx}: {item.get('id')} vs {dataset_ids[idx]}"
        assert "generated_answer" in item and isinstance(item["generated_answer"], str)
        assert "retrieved_context_titles" in item and isinstance(item["retrieved_context_titles"], list)
        assert "metrics" in item and isinstance(item["metrics"], dict)

        m = item["metrics"]
        assert "recall" in m and isinstance(m["recall"], (int, float)) and 0.0 <= m["recall"] <= 1.0
        assert "correctness" in m and isinstance(m["correctness"], int) and m["correctness"] in (0, 1)
        assert "groundedness" in m and isinstance(m["groundedness"], int) and m["groundedness"] in (0, 1)
        assert "latency_ms" in m and isinstance(m["latency_ms"], (int, float)) and m["latency_ms"] >= 0


def test_core_req_5_eval_details_b():
    """Core Requirement 5: Question-by-question metrics for Pipeline B."""
    file_b = ROOT_DIR / "results" / "eval_B_details.json"
    eval_file = ROOT_DIR / "dataset" / "eval_set.json"
    assert file_b.exists(), f"Missing {file_b}"

    with open(eval_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    with open(file_b, "r", encoding="utf-8") as f:
        details_b = json.load(f)

    assert isinstance(details_b, list), "eval_B_details.json must be a list"
    assert len(details_b) == len(dataset), f"Length of details B ({len(details_b)}) does not match dataset ({len(dataset)})"

    dataset_ids = [d["id"] for d in dataset]
    for idx, item in enumerate(details_b):
        assert isinstance(item, dict)
        assert item["id"] == dataset_ids[idx], f"ID mismatch at index {idx}: {item.get('id')} vs {dataset_ids[idx]}"
        assert "generated_answer" in item and isinstance(item["generated_answer"], str)
        assert "retrieved_context_titles" in item and isinstance(item["retrieved_context_titles"], list)
        assert "metrics" in item and isinstance(item["metrics"], dict)

        m = item["metrics"]
        assert "recall" in m and isinstance(m["recall"], (int, float)) and 0.0 <= m["recall"] <= 1.0
        assert "correctness" in m and isinstance(m["correctness"], int) and m["correctness"] in (0, 1)
        assert "groundedness" in m and isinstance(m["groundedness"], int) and m["groundedness"] in (0, 1)
        assert "latency_ms" in m and isinstance(m["latency_ms"], (int, float)) and m["latency_ms"] >= 0


def test_core_req_6_regression_report_schema():
    """Core Requirement 6: CSV regression report structure and headers."""
    csv_file = ROOT_DIR / "results" / "regression_report.csv"
    assert csv_file.exists(), f"Missing {csv_file}"

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["metric", "baseline", "candidate", "delta", "flag"], \
            f"Invalid CSV headers: {reader.fieldnames}"
        rows = list(reader)

    assert len(rows) == 4, f"Expected exactly 4 metric rows, got {len(rows)}"

    expected_metrics = ["Recall", "Correctness", "Groundedness", "P95_Latency"]
    found_metrics = [r["metric"] for r in rows]
    assert found_metrics == expected_metrics, f"Metrics order mismatch: {found_metrics} vs {expected_metrics}"


def test_core_req_7_regression_flag_logic():
    """Core Requirement 7: Regression flag mathematical logic validation."""
    csv_file = ROOT_DIR / "results" / "regression_report.csv"
    assert csv_file.exists(), f"Missing {csv_file}"

    with open(csv_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        metric = row["metric"]
        baseline = float(row["baseline"])
        candidate = float(row["candidate"])
        delta = float(row["delta"])
        flag = row["flag"]

        # Verify delta math
        assert round(candidate - baseline, 2) == round(delta, 2), f"Delta math mismatch in {metric}"

        if metric in ("Recall", "Correctness", "Groundedness"):
            if candidate < baseline:
                assert flag == "REGRESS", f"Expected REGRESS for {metric} when candidate ({candidate}) < baseline ({baseline})"
            elif candidate > baseline:
                assert flag == "IMPROVE", f"Expected IMPROVE for {metric} when candidate ({candidate}) > baseline ({baseline})"
            else:
                assert flag == "NEUTRAL", f"Expected NEUTRAL for {metric} when candidate == baseline"

        elif metric == "P95_Latency":
            if candidate > (baseline * 1.10):
                assert flag == "REGRESS", f"Expected REGRESS for {metric} when candidate ({candidate}) > baseline + 10% ({baseline * 1.10})"
            elif candidate < baseline:
                assert flag == "IMPROVE", f"Expected IMPROVE for {metric} when candidate < baseline"
            else:
                assert flag == "NEUTRAL", f"Expected NEUTRAL for {metric}"


def test_core_req_8_demonstrated_regression():
    """Core Requirement 8: Demonstrated trade-off failure mode in Candidate B."""
    csv_file = ROOT_DIR / "results" / "regression_report.csv"
    assert csv_file.exists(), f"Missing {csv_file}"

    with open(csv_file, "r", encoding="utf-8") as f:
        rows = {r["metric"]: r for r in csv.DictReader(f)}

    corr_delta = float(rows["Correctness"]["delta"])
    ground_delta = float(rows["Groundedness"]["delta"])
    latency_delta = float(rows["P95_Latency"]["delta"])

    flags = [r["flag"] for r in rows.values()]

    # Criteria 1: Correctness Candidate >= Correctness Baseline
    assert corr_delta >= 0.0, f"Candidate Correctness must be >= Baseline (delta was {corr_delta})"

    # Criteria 2: Groundedness Candidate < Groundedness Baseline OR P95_Latency Candidate > P95_Latency Baseline
    assert (ground_delta < 0.0) or (latency_delta > 0.0), \
        f"Must exhibit regression on groundedness ({ground_delta}) or latency ({latency_delta})"

    # Criteria 3: At least one REGRESS flag is present in the final column
    assert "REGRESS" in flags, "At least one REGRESS flag must be present in regression_report.csv"


def test_core_req_9_docker_compose_setup():
    """Core Requirement 9: Docker compose and vector database healthcheck."""
    compose_file = ROOT_DIR / "docker-compose.yml"
    assert compose_file.exists(), f"Missing {compose_file}"

    with open(compose_file, "r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    assert "services" in compose_data, "docker-compose.yml must have 'services' block"
    services = compose_data["services"]

    # Verify vector database service exists
    vector_services = [s for s, conf in services.items() if any(db in s.lower() or db in str(conf.get("image", "")).lower() for db in ["qdrant", "chroma", "weaviate", "postgres", "pgvector"])]
    assert len(vector_services) >= 1, "No vector database service found in docker-compose.yml"

    # Verify healthcheck definition
    v_service = services[vector_services[0]]
    assert "healthcheck" in v_service, f"Vector DB service '{vector_services[0]}' lacks healthcheck definition"
    healthcheck = v_service["healthcheck"]
    assert "test" in healthcheck, "Healthcheck must specify 'test' command"


def test_core_req_10_env_example_setup():
    """Core Requirement 10: Environment variables documented without leaking secrets."""
    env_example = ROOT_DIR / ".env.example"
    assert env_example.exists(), f"Missing {env_example}"

    with open(env_example, "r", encoding="utf-8") as f:
        content = f.read()

    # Verify API key placeholders exist
    assert "OPENAI_API_KEY=" in content or "LLM_API_KEY=" in content, "Missing LLM API key variable in .env.example"

    # Verify vector database connection host/port info
    assert "VECTOR_DB_HOST=" in content or "VECTOR_DB_PORT=" in content or "VECTOR_DB_URL=" in content, \
        "Missing vector database connection variables in .env.example"

    # Verify it does NOT contain exact/real sk-... API keys
    import re
    assert not re.search(r"sk-[a-zA-Z0-9]{20,}", content), "Found exact real sk-... API key in .env.example!"

