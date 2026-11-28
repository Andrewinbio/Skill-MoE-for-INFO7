"""Standard-library dataset audit and paired-output analysis. See analysis/README.md."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata


def question_id(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def parse_answer(text, finish_reason=None):
    """Require an explicit final answer outside a completed thinking block."""
    if not isinstance(text, str) or not text.strip():
        return {"answer": None, "parse_status": "missing"}
    if finish_reason == "length":
        return {"answer": None, "parse_status": "truncated"}
    if "<think>" in text and "</think>" not in text:
        return {"answer": None, "parse_status": "unfinished_thinking"}
    final = text.rsplit("</think>", 1)[-1]
    matches = re.findall(r"\b(?:the\s+)?answer\s+is\s*\(([A-D])\)", final, re.I)
    if not matches:
        return {"answer": None, "parse_status": "no_explicit_answer"}
    return {"answer": matches[-1].upper(), "parse_status": "valid"}


def vote(answers):
    valid = [a for a in answers if a is not None]
    counts = Counter(valid)
    if not counts:
        return None, False
    maximum = max(counts.values())
    tied = {a for a, n in counts.items() if n == maximum}
    return next(a for a in valid if a in tied), len(tied) > 1


def validate_router(distribution, solvers):
    if distribution is None:
        return None
    if not isinstance(distribution, dict) or not distribution:
        raise ValueError("Router probabilities must be a nonempty model-to-probability mapping")
    if not all(isinstance(name, str) and isinstance(p, (int, float)) and not isinstance(p, bool)
               and math.isfinite(p) and 0 <= p <= 1 for name, p in distribution.items()):
        raise ValueError("Invalid router probabilities")
    if not math.isclose(sum(distribution.values()), 1, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Router probabilities do not sum to one")
    if any(distribution.get(solver, 0) <= 0 for solver in solvers):
        raise ValueError("Selected solver is missing or has zero routing probability")
    return max(distribution.values())


def features(answers, solvers, router_probabilities=None):
    valid = [a for a in answers if a is not None]
    counts = Counter(valid)
    return {
        "router_confidence": validate_router(router_probabilities, solvers),
        "modal_disagreement": 1 - max(counts.values()) / len(valid) if valid else None,
        "normalized_entropy": -sum((n / len(valid)) * math.log(n / len(valid))
                                    for n in counts.values()) / math.log(4) if valid else None,
        "selected_slots": len(solvers), "unique_experts": len(set(solvers)),
        "valid_expert_answers": len(valid), "invalid_expert_answers": len(answers) - len(valid),
    }


def prepare_dataset(test_path, train_path, exposed_csv, output, seed=0):
    test = json.loads(Path(test_path).read_text())
    train = json.loads(Path(train_path).read_text())
    groups = defaultdict(list)
    for index, row in enumerate(test):
        if not isinstance(row.get("question"), str) or row.get("gold_answer") not in ("A", "B", "C", "D"):
            raise ValueError(f"Invalid test row {index}")
        groups[question_id(row["question"])].append(index)
    train_ids = {question_id(row["question"]) for row in train}
    with Path(exposed_csv).open() as handle:
        exposed = {question_id(row["question"]) for row in csv.DictReader(handle)}
    records = []
    for qid, indices in sorted(groups.items()):
        labels = {test[i]["gold_answer"] for i in indices}
        overlap = qid in train_ids
        if len(labels) != 1:
            split = "excluded_label_conflict"
        elif qid in exposed:
            split = "development_smoke"
        elif overlap:
            split = "excluded_train_overlap"
        else:
            bucket = int(hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest(), 16) / 2**256
            split = "fit" if bucket < .6 else "validation" if bucket < .8 else "held_out"
        records.append({"question_id": qid, "source_indices": indices, "split": split,
                        "train_overlap": overlap, "label_conflict": len(labels) != 1,
                        "previously_exposed": qid in exposed})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "split_manifest.json", records)
    audit = {
        "status": "provisional; exact-normalized duplicate grouping only",
        "seed": seed, "split_targets": {"fit": .6, "validation": .2, "held_out": .2},
        "sources": [{"path": str(p), "sha256": digest(p)} for p in (test_path, train_path, exposed_csv)],
        "test_rows": len(test), "train_rows": len(train), "unique_test_questions": len(groups),
        "duplicate_groups": sum(len(v) > 1 for v in groups.values()),
        "duplicate_extra_rows": len(test) - len(groups),
        "train_overlap_groups": len(set(groups) & train_ids),
        "split_group_counts": dict(Counter(r["split"] for r in records)),
        "split_row_counts": dict(Counter(r["split"] for r in records for _ in r["source_indices"])),
        "limitations": ["Local hashes do not establish external benchmark provenance.",
                        "Near-duplicates and model training contamination are not assessed.",
                        "Split approval and primary-run model configuration remain pending."],
    }
    write_json(output / "dataset_audit.json", audit)
    return audit


def paired_records(run_dir, split_path=None):
    run = Path(run_dir)
    with (run / "skills/MedMCQA/round0_seed0.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    files = list((run / "skills/MedMCQA").glob("fixed_Qwen_round1_seed0_*.json"))
    if len(files) != 1:
        raise ValueError("Expected exactly one aggregator artifact")
    aggregated = json.loads(files[0].read_text())
    reference = json.loads((run / "test_data/MedMCQA_test.json").read_text())
    if not len(rows) == len(aggregated) == len(reference):
        raise ValueError("CSV, aggregation and reference counts differ")
    events_path = run / "generation-events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
    if events_path.exists() and not events:
        raise ValueError("Generation telemetry file is empty")
    event_map = {}
    for e in events:
        key = (e["stage"], e["agent"], e["index"])
        if key in event_map:
            raise ValueError(f"Duplicate generation event: {key}")
        event_map[key] = e
    splits = {r["question_id"]: r["split"] for r in json.loads(Path(split_path).read_text())} if split_path else {}
    positions = Counter()
    used_events = set()

    def response(text, stage, agent, index):
        key = (stage, agent, index)
        e = event_map.get(key)
        if events and e is None:
            raise ValueError(f"Missing telemetry: {key}")
        if e and e["raw_output"] != text:
            raise ValueError(f"Telemetry/output mismatch: {key}")
        if e:
            used_events.add(key)
        e = e or {}
        return {"raw_output": text, **parse_answer(text, e.get("finish_reason")),
                **{k: e.get(k) for k in ("model", "prompt_tokens", "output_tokens", "finish_reason",
                                        "stop_reason", "batch_generation_seconds", "max_tokens", "enable_thinking")}}

    records = []
    for index, (row, raw_agg, ref) in enumerate(zip(rows, aggregated, reference)):
        if row["question"] != ref["question"] or row["gold_answer"] != ref["gold_answer"]:
            raise ValueError(f"Reference mismatch at row {index}")
        solvers = ast.literal_eval(row["solvers"])
        if not isinstance(solvers, list) or not solvers or not all(isinstance(s, str) for s in solvers):
            raise ValueError("Invalid solver sequence")
        router_probabilities = json.loads(row["router_probabilities"]) if row.get("router_probabilities") else None
        router_confidence = validate_router(router_probabilities, solvers)
        if row.get("router_confidence") and (
            router_confidence is None or not math.isclose(float(row["router_confidence"]), router_confidence, abs_tol=1e-6)
        ):
            raise ValueError("Recorded router confidence does not match the probability vector")
        # Upstream assigns first repeated draw to answer_N, then counts down.
        remaining = Counter(solvers)
        experts = []
        for slot, solver in enumerate(solvers):
            column = f"{solver}_answer_{remaining[solver]}"
            remaining[solver] -= 1
            entry = response(row.get(column, ""), "expert", solver, positions[solver])
            positions[solver] += 1
            experts.append({"slot": slot, "solver": solver, "source_column": column, **entry})
        agg = response(raw_agg, "aggregator", "Qwen", index)
        answers = [e["answer"] for e in experts]
        majority, tied = vote(answers)
        distinct = {}
        for e in experts:
            distinct.setdefault(e["solver"], e["answer"])
        unique_vote, unique_tie = vote(list(distinct.values()))
        complete = all(e["parse_status"] != "missing" for e in experts + [agg])
        v_correct = majority == row["gold_answer"]
        a_correct = agg["answer"] == row["gold_answer"]
        qid = question_id(row["question"])
        records.append({
            "schema_version": "1.1", "question_id": qid, "source_index": index,
            "split": splits.get(qid, "unassigned"), "run_id": run.name,
            "question": row["question"], "gold_answer": row["gold_answer"],
            "experts": experts, "aggregator": agg,
            "router": {"probabilities": router_probabilities, "confidence": router_confidence,
                       "fallback": row.get("router_fallback") or None},
            "features": features(answers, solvers, router_probabilities),
            "vote": {"answer": majority, "tied": tied, "distinct_expert_answer": unique_vote,
                     "distinct_expert_tied": unique_tie, "tie_rule": "earliest_draw_among_tied_valid_answers"},
            "paired_eligible": complete,
            "labels": {"vote_correct": v_correct, "aggregator_correct": a_correct,
                       "aggregation_failure": not a_correct if complete else None,
                       "helpful": not v_correct and a_correct if complete else None,
                       "harmful": v_correct and not a_correct if complete else None,
                       "neutral": v_correct == a_correct if complete else None,
                       "net_gain": int(a_correct) - int(v_correct) if complete else None},
        })
    if used_events != set(event_map):
        raise ValueError("Unmatched generation events")
    return records


def summarize(records):
    paired = [r for r in records if r["paired_eligible"]]
    n = len(paired)
    stage_metrics = {}
    for stage, outputs in (
        ("expert", [e for r in records for e in r["experts"]]),
        ("aggregator", [r["aggregator"] for r in records]),
    ):
        # Each solver is generated in one batch; duration is repeated per response.
        batches = {}
        for e in outputs:
            if e["batch_generation_seconds"] is not None:
                batches[e.get("solver", "Qwen")] = e["batch_generation_seconds"]
        stage_metrics[stage] = {
            key: sum(e[key] for e in outputs) if outputs and all(e[key] is not None for e in outputs) else None
            for key in ("prompt_tokens", "output_tokens")
        }
        stage_metrics[stage]["generation_seconds"] = sum(batches.values()) if batches else None
        stage_metrics[stage]["finish_reasons"] = dict(Counter(e["finish_reason"] or "unknown" for e in outputs))
    return {
        "stage_metrics": stage_metrics,
        "answer_quality_pass": bool(records) and all(
            e["parse_status"] == "valid" and e["finish_reason"] == "stop"
            for r in records for e in r["experts"] + [r["aggregator"]]
        ),
        "questions": len(records), "paired_questions": n,
        "paired_coverage": n / len(records) if records else None,
        "expert_parse_status": dict(Counter(e["parse_status"] for r in records for e in r["experts"])),
        "aggregator_parse_status": dict(Counter(r["aggregator"]["parse_status"] for r in records)),
        "strict_vote_accuracy": sum(r["labels"]["vote_correct"] for r in paired) / n if n else None,
        "strict_aggregator_accuracy": sum(r["labels"]["aggregator_correct"] for r in paired) / n if n else None,
        "helpful": sum(r["labels"]["helpful"] for r in paired),
        "harmful": sum(r["labels"]["harmful"] for r in paired),
        "neutral": sum(r["labels"]["neutral"] for r in paired),
        "vote_ties": sum(r["vote"]["tied"] for r in records),
        "all_expert_answers_invalid": sum(r["features"]["valid_expert_answers"] == 0 for r in records),
        "note": "Strict parser; invalid outputs score incorrect. Missing outputs excluded from paired metrics. Smoke data is development-only.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dataset = sub.add_parser("dataset")
    dataset.add_argument("--test", required=True)
    dataset.add_argument("--train", required=True)
    dataset.add_argument("--exposed-csv", required=True)
    dataset.add_argument("--output", required=True)
    dataset.add_argument("--seed", type=int, default=0)
    records = sub.add_parser("records")
    records.add_argument("--run-dir", required=True)
    records.add_argument("--splits")
    args = parser.parse_args()
    if args.command == "dataset":
        result = prepare_dataset(args.test, args.train, args.exposed_csv, args.output, args.seed)
    else:
        rows = paired_records(args.run_dir, args.splits)
        result = summarize(rows)
        root = Path(args.run_dir)
        result["analysis_code_sha256"] = digest(__file__)
        result["split_manifest_sha256"] = digest(args.splits) if args.splits else None
        result["artifact_hashes"] = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*"))
                                     if p.is_file() and p.suffix in (".csv", ".py", ".json", ".jsonl", ".sh", ".log")
                                     and p.name not in ("analysis-summary.json", "question-records.jsonl")}
        (root / "question-records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        write_json(root / "analysis-summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
