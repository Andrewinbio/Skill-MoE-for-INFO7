"""Score the frozen routed MedMCQA fit and validation splits only."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from prepare_medmcqa import digest, features, parse_answer, question_id, vote, write_json


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "test_data/MedMCQA_test.json"
MANIFEST = ROOT / "analysis/medmcqa-data/split_manifest.json"
REFERENCE_HASH = "1c7a0851b37f37c60bbb70af289bf95b52158c74a4bb2d81750085c211481807"
MANIFEST_HASH = "082fa4d5e7d6004f7289059a7d50642e7e193b87fc1df839314490fe037df472"
FIT_VALIDATION_COUNTS = {"fit": 2507, "validation": 831}


def checked_answer(output):
    parsed = parse_answer(output["raw_output"], output["finish_reason"])
    if any(output[key] != value for key, value in parsed.items()):
        raise ValueError("Stored answer disagrees with the strict parser")
    return parsed["answer"]


def score_row(row, gold):
    experts = row["experts"]
    router = row["router"]
    solvers = [expert["solver"] for expert in experts]
    if (len(experts) != 3 or [e["slot"] for e in experts] != [0, 1, 2]
            or solvers != router["solvers"]):
        raise ValueError("Expert draws do not match the route")
    answers = [checked_answer(expert) for expert in experts]
    aggregate = checked_answer(row["aggregator"])
    draw_features = features(answers, solvers, router["probabilities"])
    if draw_features != row["features"] or draw_features["router_confidence"] != router["confidence"]:
        raise ValueError("Pre-aggregation features do not match collection")
    first_draw = {}
    for solver, answer in zip(solvers, answers):
        first_draw.setdefault(solver, answer)
    distinct_features = features(list(first_draw.values()), list(first_draw), router["probabilities"])
    majority, tied = vote(answers)
    distinct, distinct_tied = vote(list(first_draw.values()))
    if row["vote"] != {
        "answer": majority, "tied": tied,
        "distinct_expert_answer": distinct, "distinct_expert_tied": distinct_tied,
    }:
        raise ValueError("Vote does not match collection")

    keys = {key: row[key] for key in ("question_id", "source_index", "split")}
    predictors = {
        **keys, **draw_features, "vote_tied": tied, "distinct_vote_tied": distinct_tied,
        **{f"distinct_{key}": distinct_features[key] for key in (
            "modal_disagreement", "normalized_entropy", "valid_expert_answers", "invalid_expert_answers"
        )},
    }
    complete = all(e["parse_status"] != "missing" for e in [*experts, row["aggregator"]])
    a_correct = aggregate == gold
    outcomes = {**keys, "paired_eligible": complete, "aggregator_correct": a_correct if complete else None,
                "aggregation_failure": not a_correct if complete else None}
    for prefix, answer in (("", majority), ("distinct_", distinct)):
        v_correct = answer == gold
        labels = {
            "vote_correct": v_correct, "helpful": not v_correct and a_correct,
            "harmful": v_correct and not a_correct, "neutral": v_correct == a_correct,
            "net_gain": int(a_correct) - int(v_correct),
        }
        outcomes.update({prefix + key: value if complete else None for key, value in labels.items()})
    return predictors, outcomes


def build_tables(records_path):
    for path, expected in ((REFERENCE, REFERENCE_HASH), (MANIFEST, MANIFEST_HASH)):
        if digest(path) != expected:
            raise ValueError(f"Pinned input changed: {path}")
    splits = json.loads(MANIFEST.read_text())
    expected = {
        entry["question_id"]: (index, entry["split"])
        for entry in splits if entry["split"] in FIT_VALIDATION_COUNTS
        for index in entry["source_indices"]
    }
    reference = json.loads(REFERENCE.read_text())
    seen = set()
    predictors, outcomes = [], []
    for row in json.loads(records_path.read_text()):
        if row["split"] not in FIT_VALIDATION_COUNTS:
            continue
        qid, index = row["question_id"], row["source_index"]
        if qid in seen or expected.get(qid) != (index, row["split"]):
            raise ValueError(f"Unexpected, misaligned, or duplicate record: {qid}")
        ref = reference[index]
        if (row["question"] != ref["question"] or question_id(ref["question"]) != qid
                or ref["gold_answer"] not in ("A", "B", "C", "D")):
            raise ValueError(f"Reference mismatch: {qid}")
        try:
            feature_row, outcome_row = score_row(row, ref["gold_answer"])
        except ValueError as error:
            raise ValueError(f"{qid}: {error}") from error
        seen.add(qid)
        predictors.append(feature_row)
        outcomes.append(outcome_row)
    if seen != set(expected) or Counter(r["split"] for r in predictors) != FIT_VALIDATION_COUNTS:
        raise ValueError("Fit and validation coverage differs from the frozen split")
    return predictors, outcomes


def summarize(predictors, outcomes):
    summary = {}
    for split, total in FIT_VALIDATION_COUNTS.items():
        rows = [r for r in predictors if r["split"] == split]
        pairs = [r for r in outcomes if r["split"] == split and r["paired_eligible"]]
        report = {
            "questions": total, "paired_questions": len(pairs),
            "incomplete_questions": total - len(pairs), "paired_coverage": len(pairs) / total,
            "repeated_expert_questions": sum(r["unique_experts"] < r["selected_slots"] for r in rows),
            "unique_expert_counts": dict(Counter(r["unique_experts"] for r in rows)),
        }
        for label, prefix, tied in (("draw_level", "", "vote_tied"),
                                    ("distinct_expert", "distinct_", "distinct_vote_tied")):
            report[label] = {
                "ties": sum(r[tied] for r in rows),
                "all_invalid": sum(r[prefix + "valid_expert_answers"] == 0 for r in rows),
                "disagreement_counts": dict(Counter(str(r[prefix + "modal_disagreement"]) for r in rows)),
                "paired_outcomes": dict(Counter(
                    f'vote_{int(r[prefix + "vote_correct"])}_aggregate_{int(r["aggregator_correct"])}'
                    for r in pairs
                )),
                "vote_accuracy": sum(r[prefix + "vote_correct"] for r in pairs) / len(pairs) if pairs else None,
                "aggregator_accuracy": sum(r["aggregator_correct"] for r in pairs) / len(pairs) if pairs else None,
            }
        summary[split] = report
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    predictors, outcomes = build_tables(args.records)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("features.csv", predictors), ("outcomes.csv", outcomes)):
        with (args.output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(args.output / "summary.json", {
        "collection_sha256": digest(args.records),
        "split_manifest_sha256": MANIFEST_HASH, "reference_sha256": REFERENCE_HASH,
        "held_out_scored": False, "splits": summarize(predictors, outcomes),
    })


if __name__ == "__main__":
    main()
