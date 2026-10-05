"""Lightweight RAG evaluation harness — no paid API required.

For each question in the active profile's golden set (profiles/<name>/eval.json),
or a dataset passed on the command line:
  1. Runs the real retrieval + generation pipeline (same code path as production).
  2. Uses the local Ollama LLM itself as a judge, scoring the generated answer
     against a hand-written reference answer on a 1-5 factual-correctness scale.

Usage:
    python evaluate.py                      # the active profile's eval.json
    python evaluate.py path/to/golden.json  # any dataset: [{"question": ..., "reference_answer": ...}]

Output: console summary + eval_results.json with per-question detail.
"""

import json
import re
import sys

from rag import config
from rag.generation import llm
from rag.ingestion import build_index_if_needed
from rag.pipeline import get_response

JUDGE_PROMPT = """You are grading a Q&A system. Compare the MODEL ANSWER to the REFERENCE ANSWER.
Score from 1 to 5 how factually correct and complete the MODEL ANSWER is relative to the REFERENCE ANSWER:
5 = fully correct and complete, 3 = partially correct, 1 = wrong or missing.
Respond with ONLY a single digit (1-5), nothing else.

Question: {question}
Reference Answer: {reference}
Model Answer: {model_answer}
"""


def judge_score(question: str, reference: str, model_answer: str) -> int:
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, model_answer=model_answer)
    raw = llm.invoke(prompt).content.strip()
    match = re.search(r"[1-5]", raw)
    return int(match.group()) if match else 0


def run_evaluation(dataset_path: str) -> None:
    with open(dataset_path, encoding="utf-8") as f:
        eval_set = json.load(f)

    build_index_if_needed()  # make sure the index reflects the current docs folder

    results = []
    for item in eval_set:
        question = item["question"]
        reference = item["reference_answer"]

        answer = get_response(question)
        score = judge_score(question, reference, answer)

        results.append(
            {
                "question": question,
                "reference_answer": reference,
                "model_answer": answer,
                "judge_score": score,
            }
        )
        print(f"[{score}/5] {question}")

    avg_score = sum(r["judge_score"] for r in results) / len(results)
    pass_rate = sum(1 for r in results if r["judge_score"] >= 4) / len(results) * 100

    print(f"\n=== Evaluation Summary ({config.PROFILE_NAME}) ===")
    print(f"Questions evaluated : {len(results)}")
    print(f"Average judge score : {avg_score:.2f} / 5")
    print(f"Pass rate (>=4/5)   : {pass_rate:.1f}%")

    with open("eval_results.json", "w") as f:
        json.dump(
            {"profile": config.PROFILE_NAME, "results": results, "avg_score": avg_score, "pass_rate": pass_rate},
            f,
            indent=2,
        )

    print("\nDetailed results written to eval_results.json")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else config.EVAL_DATASET
    if not path:
        sys.exit(
            f"No evaluation set for profile '{config.PROFILE_NAME}'. Add profiles/{config.PROFILE_NAME}/eval.json "
            "or pass a path: python evaluate.py path/to/golden.json"
        )
    run_evaluation(path)
