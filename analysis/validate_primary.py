"""Validate the approved primary-study freeze against local artifacts."""
from collections import Counter
import json
from pathlib import Path

from prepare_medmcqa import digest, question_id


def validate(root):
    config = json.loads((root / "analysis/primary-study.json").read_text())
    if config["status"].startswith("superseded"):
        return {"status": "superseded_configuration", "launch_ready": False,
                "active_protocol": config["superseded_by"],
                "execution_status": "Historical same-model plan is not approval for the restored router study."}
    data = config["dataset"]
    checks = [(data["test_path"], data["test_sha256"]),
              (data["train_path"], data["train_sha256"]),
              (data["split_manifest"], data["split_manifest_sha256"]),
              (config["analysis"]["implementation"], config["analysis"]["implementation_sha256"])]
    generation = config["generation"]
    checks.extend((str(Path(generation["prompt_reference_run"]) / name), checksum)
                  for name, checksum in generation["prompt_source_hashes"].items())
    for name, checksum in checks:
        if digest(root / name) != checksum:
            raise ValueError(f"Frozen artifact changed: {name}")
    groups = json.loads((root / data["split_manifest"]).read_text())
    rows = json.loads((root / data["test_path"]).read_text())
    seen_ids, seen_indices = set(), set()
    counts = Counter()
    for group in groups:
        qid = group["question_id"]
        if qid in seen_ids:
            raise ValueError("Duplicate question ID in split manifest")
        seen_ids.add(qid)
        for index in group["source_indices"]:
            if index in seen_indices or question_id(rows[index]["question"]) != qid:
                raise ValueError("Duplicate or misaligned source index")
            seen_indices.add(index)
            counts[group["split"]] += 1
        if group["split"] in ("fit", "validation", "held_out"):
            if any(group[k] for k in ("previously_exposed", "train_overlap", "label_conflict")):
                raise ValueError("Excluded question leaked into experiment")
    if seen_indices != set(range(len(rows))) or dict(counts) != data["split_counts"]:
        raise ValueError("Frozen split counts or coverage differ")
    if sum(counts[s] for s in ("fit", "validation", "held_out")) != data["primary_questions"]:
        raise ValueError("Primary sample size differs")
    events = [json.loads(line) for line in
              (root / generation["prompt_reference_run"] / "generation-events.jsonl").read_text().splitlines()]
    if len(events) != 20 or not all(
        e["model"] == generation["model"] and e["max_tokens"] == generation["max_new_tokens"]
        and e["enable_thinking"] == generation["enable_thinking"] and e["finish_reason"] == "stop"
        for e in events
    ):
        raise ValueError("Quality-run evidence does not match frozen model settings")
    return {"status": "freeze_validated", "primary_questions": data["primary_questions"],
            "split_counts": dict(counts), "config_sha256": digest(root / "analysis/primary-study.json"),
            "execution_status": "Primary collector, budget ledger and calibration evaluator still need implementation; no primary job submitted."}


if __name__ == "__main__":
    print(json.dumps(validate(Path(__file__).resolve().parents[1]), indent=2))
