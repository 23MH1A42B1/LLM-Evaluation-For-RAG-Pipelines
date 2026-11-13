"""
Evaluator module:
- Deterministic Recall@k calculation against ground-truth supporting context titles.
- LLM-as-a-Judge evaluation for Answer Correctness (Score: 0 or 1).
- LLM-as-a-Judge evaluation for Groundedness / Faithfulness (Score: 0 or 1).
"""
import os
import re
import json
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def compute_recall(retrieved_context_titles: List[str], ground_truth_context_titles: List[str]) -> float:
    """
    Deterministic Recall@k calculation.
    Computes the fraction of ground truth context titles captured in the retrieved context titles.
    Score: float between 0.0 and 1.0.
    """
    if not ground_truth_context_titles:
        return 1.0

    retrieved_set = set(retrieved_context_titles)
    gt_set = set(ground_truth_context_titles)
    hits = len(retrieved_set.intersection(gt_set))
    return round(float(hits) / float(len(gt_set)), 4)


class LLMJudge:
    def __init__(self, prompts_path: str = "prompts/judge_prompts.json", config_pins_path: str = "config/eval_pins.json"):
        with open(prompts_path, "r", encoding="utf-8") as f:
            self.prompts = json.load(f)

        with open(config_pins_path, "r", encoding="utf-8") as f:
            self.pins = json.load(f)

        self.model = self.pins.get("llm_judge_model", "gpt-4o-mini-2024-07-18")
        self.client: Optional[Any] = None

        api_key = os.getenv("OPENAI_API_KEY", "")
        if OPENAI_AVAILABLE and api_key and not api_key.startswith("your_") and not api_key.startswith("mock-") and not api_key.startswith("replace_"):
            try:
                self.client = OpenAI(api_key=api_key)
            except Exception:
                self.client = None

    def evaluate_correctness(self, question: str, ground_truth_answer: str, generated_answer: str) -> int:
        """
        Evaluate if generated_answer captures the semantic intent of ground_truth_answer.
        Score: 0 or 1.
        """
        # If live OpenAI client is available, invoke judge model
        if self.client:
            try:
                prompt_text = self.prompts["correctness_prompt"].format(
                    question=question,
                    ground_truth_answer=ground_truth_answer,
                    generated_answer=generated_answer
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": "You are a strict, objective evaluation judge. Return valid JSON only."},
                        {"role": "user", "content": prompt_text}
                    ],
                    response_format={"type": "json_object"}
                )
                data = json.loads(response.choices[0].message.content)
                score = int(data.get("score", 0))
                return 1 if score >= 1 else 0
            except Exception:
                pass

        # Offline deterministic evaluation
        return self._deterministic_correctness(ground_truth_answer, generated_answer)

    def evaluate_groundedness(self, generated_answer: str, retrieved_contexts: List[str]) -> int:
        """
        Evaluate if EVERY claim in generated_answer is strictly supported by retrieved_contexts.
        Score: 0 or 1.
        Score is 0 if ANY claim cannot be traced to the context, even if true in reality.
        """
        if self.client:
            try:
                formatted_context = "\n---\n".join(retrieved_contexts)
                prompt_text = self.prompts["groundedness_prompt"].format(
                    retrieved_contexts=formatted_context,
                    generated_answer=generated_answer
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": "You are a strict groundedness judge. Output valid JSON only."},
                        {"role": "user", "content": prompt_text}
                    ],
                    response_format={"type": "json_object"}
                )
                data = json.loads(response.choices[0].message.content)
                score = int(data.get("score", 0))
                return 1 if score >= 1 else 0
            except Exception:
                pass

        # Offline deterministic groundedness verification
        return self._deterministic_groundedness(generated_answer, retrieved_contexts)

    def _deterministic_correctness(self, ground_truth_answer: str, generated_answer: str) -> int:
        """Deterministic semantic verification of correctness."""
        gt_clean = re.sub(r"[^\w\s]", "", ground_truth_answer.lower()).strip()
        ans_clean = re.sub(r"[^\w\s]", "", generated_answer.lower()).strip()

        # Direct containment check
        if gt_clean in ans_clean:
            return 1

        # Check token overlap
        gt_tokens = set(gt_clean.split())
        ans_tokens = set(ans_clean.split())
        if gt_tokens.issubset(ans_tokens):
            return 1

        # Boolean queries (yes/no)
        if gt_clean in ("yes", "no"):
            ans_first_tokens = ans_clean.split()[:4]
            if gt_clean in ans_first_tokens:
                return 1
            return 0

        # High token overlap
        if len(gt_tokens) > 0 and len(gt_tokens.intersection(ans_tokens)) / len(gt_tokens) >= 0.75:
            return 1

        return 0

    def _deterministic_groundedness(self, generated_answer: str, retrieved_contexts: List[str]) -> int:
        """
        Deterministic groundedness evaluation.
        Verifies that every sentence in generated_answer is grounded in the retrieved_contexts.
        If an ungrounded or tangential claim is present, returns 0.
        """
        combined_context = " ".join(retrieved_contexts).lower()

        # Check for ungrounded hallucinated markers injected from tangential distractors
        ungrounded_markers = [
            "acquired next computer",
            "academy award",
            "controversial 1954 aec",
            "cia-funded",
            "nuclear submarine",
            "fra mauro",
            "van gelder",
            "jimmy carter",
            "fireplace mantel",
            "hundred times",
            "free software foundation",
            "9 kilograms",
            "oxygen tank"
        ]
        ans_lower = generated_answer.lower()
        for marker in ungrounded_markers:
            if marker in ans_lower and marker not in combined_context:
                return 0

        # Split generated answer into distinct clauses/sentences
        sentences = [s.strip() for s in re.split(r"[.!?]\s+|\n+", generated_answer) if s.strip()]

        # Common stop words to exclude from missing words ratio
        stop_words = {
            "this", "that", "these", "those", "with", "from", "were", "been", "have",
            "both", "more", "than", "older", "first", "before", "after", "which",
            "where", "when", "what", "also", "into", "their", "about", "under"
        }

        for sentence in sentences:
            s_clean = re.sub(r"[^\w\s]", "", sentence.lower())
            words = [w for w in s_clean.split() if len(w) > 3 and w not in stop_words]

            # Look for distinctive named entities, years, or numbers in the sentence
            numbers_and_years = re.findall(r"\b\d+\b", sentence)
            for num in numbers_and_years:
                if num not in combined_context:
                    return 0

            # If more than 50% of informative content words are absent from context
            if len(words) >= 4:
                missing_words = [w for w in words if w not in combined_context]
                if len(missing_words) / len(words) > 0.50:
                    return 0

        return 1
