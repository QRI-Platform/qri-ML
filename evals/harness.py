
import json
import os
from datetime import datetime
from typing import Any

import langfuse
from dotenv import load_dotenv

from src.core.logger import logger

load_dotenv()


def load_goldens(path):
    mapping = {
        "Retriever_Eval": "goldens/eval_retriever.json",
        "Generator_Eval": "goldens/eval_generator.json",
        "Application_Eval": "goldens/eval_application.json",
    }

    local_path = mapping.get(path)
    if local_path and os.path.exists(local_path):
        logger.info("Loading golden dataset directly from local JSON: %s", local_path)
        return load_goldens_from_local(local_path)

    return None


def load_goldens_from_local(path):
    with open(path, "r") as f:
        return json.load(f)


def summarize_by_metric(result):
    """DeepEval EvaluationResult -> {metric_name: {n, pass_rate, avg/min/max score}}."""
    test_results = getattr(result, "test_results", None)
    if test_results is None:
        test_results = result if isinstance(result, list) else []

    buckets = {}   # metric_name -> {"scores": [...], "passed": int, "total": int}
    for tr in test_results:
        metrics = getattr(tr, "metrics_data", None) or getattr(tr, "metrics", None) or []
        for m in metrics:
            name = getattr(m, "name", "unknown")
            b = buckets.setdefault(name, {"scores": [], "passed": 0, "total": 0})
            b["total"] += 1
            score = getattr(m, "score", None)
            if score is not None:
                b["scores"].append(score)
            if getattr(m, "success", False):
                b["passed"] += 1

    summary = {}
    for name, b in buckets.items():
        scores = b["scores"]
        summary[name] = {
            "n": b["total"],
            "pass_rate": (100 * b["passed"] / b["total"]) if b["total"] else 0.0,
            "avg_score": (sum(scores) / len(scores)) if scores else float("nan"),
            "min_score": min(scores) if scores else float("nan"),
            "max_score": max(scores) if scores else float("nan"),
        }
    return summary


def push_summary_to_langfuse(
    title: str,
    summary: dict[str, Any] | None = None,
    result: Any | None = None,
    run_name: str | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Push per-metric summary to Langfuse as numeric scores under a named run/session."""
    if summary is None:
        if result is None:
            raise ValueError("Either summary or result must be provided.")
        summary = summarize_by_metric(result)

    if not summary:
        logger.warning("No summary data to push for '%s' to Langfuse.", title)
        return None

    try:
        client = langfuse.Langfuse()
    except Exception:
        logger.warning("Langfuse client could not be initialized for summary push: %s", title)
        return None

    session_id = run_name or title or f"eval-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    metric_payload = metadata or {}

    for metric_name, metric_summary in summary.items():
        avg_score = metric_summary.get("avg_score")
        if avg_score is None or avg_score != avg_score:
            continue

        score_metadata = {
            "title": title,
            "metric": metric_name,
            "summary": metric_summary,
            **metric_payload,
        }
        client.create_score(
            name=f"{title}:{metric_name}",
            value=float(avg_score),
            session_id=session_id,
            data_type="NUMERIC",
            metadata=score_metadata,
            comment=json.dumps(
                {
                    "pass_rate": metric_summary.get("pass_rate"),
                    "n": metric_summary.get("n"),
                    "min_score": metric_summary.get("min_score"),
                    "max_score": metric_summary.get("max_score"),
                },
                default=str,
            ),
        )

    client.flush()
    return {"title": title, "run_name": session_id, "summary": summary}


def push_deepeval_result_to_langfuse(
    result: Any | None,
    title: str = "eval_results",
    run_name: str | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Push each DeepEval test result and metric to Langfuse so failed datapoints are visible as red items with reasons."""
    if result is None:
        logger.warning("No DeepEval result provided for Langfuse push: %s", title)
        return None

    try:
        client = langfuse.Langfuse()
    except Exception:
        logger.warning("Langfuse client could not be initialized for deep eval push: %s", title)
        return None

    test_results = getattr(result, "test_results", None)
    if test_results is None:
        test_results = result if isinstance(result, list) else []

    session_id = run_name or title or f"eval-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    base_metadata = metadata or {}

    pushed = []
    for index, tr in enumerate(test_results):
        metrics = getattr(tr, "metrics_data", None) or getattr(tr, "metrics", None) or []
        for metric in metrics:
            metric_name = getattr(metric, "name", "unknown")
            score = getattr(metric, "score", None)
            if score is None:
                continue

            payload = {
                "title": title,
                "metric_name": metric_name,
                "index": index,
                "passed": bool(getattr(metric, "success", False)),
                "reason": getattr(metric, "reason", None) or getattr(metric, "error", None),
                "input": getattr(tr, "input", None),
                "actual_output": getattr(tr, "actual_output", None),
                "expected_output": getattr(tr, "expected_output", None),
                "retrieval_context": getattr(tr, "retrieval_context", None),
                "result_metadata": base_metadata,
            }

            client.create_score(
                name=f"{title}:{metric_name}",
                value=float(score),
                session_id=session_id,
                trace_id=getattr(tr, "trace_id", None),
                data_type="NUMERIC",
                metadata=payload,
                comment=json.dumps(
                    {
                        "passed": payload["passed"],
                        "reason": payload["reason"],
                        "score": score,
                    },
                    default=str,
                ),
            )
            pushed.append(f"{title}:{metric_name}")

    client.flush()
    return {"title": title, "run_name": session_id, "pushed": pushed}


def push_results_to_langfuse(
    result: Any | None = None,
    title: str = "eval_results",
    summary: dict[str, Any] | None = None,
    run_name: str | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Backward-compatible wrapper; accepts either a DeepEval result or precomputed summary."""
    item_payload = push_deepeval_result_to_langfuse(
        result=result,
        title=title,
        run_name=run_name,
        metadata=metadata,
    )
    summary_payload = push_summary_to_langfuse(
        title=title,
        summary=summary,
        result=result,
        run_name=run_name,
        metadata=metadata,
    )
    return {
        "item_level": item_payload,
        "summary": summary_payload,
    }


def print_summary(title, summary):
    """Readable per-metric recap, printed under DeepEval's own report when a file
    is run directly."""
    print("\n" + "=" * 60)
    print(f"{title}  (per-metric summary)")
    print("=" * 60)
    for name, s in summary.items():
        avg = f"{s['avg_score']:.2f}" if s["avg_score"] == s["avg_score"] else "nan"   # nan != nan
        print(f"  {name:<26} pass_rate={s['pass_rate']:5.0f}%  avg={avg}  n={s['n']}")
    print("=" * 60)


