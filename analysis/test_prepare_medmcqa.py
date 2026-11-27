import csv
import json
from pathlib import Path
import tempfile
import unittest

from prepare_medmcqa import features, paired_records, parse_answer, prepare_dataset, question_id, summarize, vote


class AnalysisTests(unittest.TestCase):
    def test_strict_parser_rejects_reasoning_and_truncation(self):
        self.assertIsNone(parse_answer("Perhaps A) or B)")["answer"])
        self.assertIsNone(parse_answer("<think>The answer is (A)")["answer"])
        self.assertIsNone(parse_answer("The answer is (A)", "length")["answer"])
        self.assertEqual(parse_answer("<think>The answer is (A)</think>The answer is (B)")["answer"], "B")
        self.assertEqual(parse_answer("")["parse_status"], "missing")

    def test_votes_features_and_all_invalid(self):
        self.assertEqual(vote(["B", "A", "C"]), ("B", True))
        self.assertEqual(vote([None, None]), (None, False))
        f = features(["A", "B", None], ["Qwen"] * 3)
        self.assertEqual(f["modal_disagreement"], .5)
        self.assertEqual(f["normalized_entropy"], .5)
        self.assertEqual(f["unique_experts"], 1)
        self.assertIsNone(features([None], ["Qwen"])["normalized_entropy"])
        self.assertNotIn("gold_answer", f)

    def test_duplicate_grouping_exposure_and_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            rows = [{"question": q, "gold_answer": a} for q, a in
                    [("Same question", "A"), (" same  QUESTION ", "A"),
                     ("Seen in training", "B"), ("Smoke", "C"),
                     ("Conflict", "A"), ("conflict", "B")]]
            (p / "test.json").write_text(json.dumps(rows))
            (p / "train.json").write_text(json.dumps([rows[2]]))
            (p / "smoke.csv").write_text("question\nSmoke\n")
            prepare_dataset(p / "test.json", p / "train.json", p / "smoke.csv", p / "out")
            first = (p / "out/split_manifest.json").read_bytes()
            prepare_dataset(p / "test.json", p / "train.json", p / "smoke.csv", p / "out")
            self.assertEqual(first, (p / "out/split_manifest.json").read_bytes())
            groups = {r["question_id"]: r for r in json.loads(first)}
            self.assertEqual(len(groups[question_id("same question")]["source_indices"]), 2)
            self.assertEqual(groups[question_id("Smoke")]["split"], "development_smoke")
            self.assertEqual(groups[question_id("Seen in training")]["split"], "excluded_train_overlap")
            self.assertEqual(groups[question_id("Conflict")]["split"], "excluded_label_conflict")

    def test_draw_order_and_missing_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "skills/MedMCQA").mkdir(parents=True)
            (root / "test_data").mkdir()
            row = {"question": "Q", "gold_answer": "B", "solvers": "['Qwen', 'Qwen', 'Qwen']",
                   "Qwen_answer_1": "The answer is (A)", "Qwen_answer_2": "The answer is (C)",
                   "Qwen_answer_3": "The answer is (B)"}
            with (root / "skills/MedMCQA/round0_seed0.csv").open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            (root / "test_data/MedMCQA_test.json").write_text(json.dumps([row]))
            output = root / "skills/MedMCQA/fixed_Qwen_round1_seed0_0.json"
            output.write_text(json.dumps(["The answer is (A)"]))
            result = paired_records(root)
            self.assertEqual([e["answer"] for e in result[0]["experts"]], ["B", "C", "A"])
            self.assertEqual(result[0]["vote"]["answer"], "B")
            self.assertEqual(result[0]["vote"]["distinct_expert_answer"], "B")
            self.assertTrue(result[0]["labels"]["harmful"])
            events = []
            for i, text in enumerate([row['Qwen_answer_3'], row['Qwen_answer_2'], row['Qwen_answer_1']]):
                events.append({"stage": "expert", "agent": "Qwen", "index": i,
                               "raw_output": text, "finish_reason": "stop", "output_tokens": 10,
                               "prompt_tokens": 20, "batch_generation_seconds": 3.0})
            events.append({"stage": "aggregator", "agent": "Qwen", "index": 0,
                           "raw_output": "The answer is (A)", "finish_reason": "stop",
                           "output_tokens": 10, "prompt_tokens": 20, "batch_generation_seconds": 2.0})
            event_path = root / "generation-events.jsonl"
            event_path.write_text("".join(json.dumps(e) + "\n" for e in events))
            metrics = summarize(paired_records(root))["stage_metrics"]
            self.assertEqual(metrics["expert"]["output_tokens"], 30)
            self.assertEqual(metrics["expert"]["generation_seconds"], 3.0)
            events[0]["raw_output"] = "Wrong association"
            event_path.write_text("".join(json.dumps(e) + "\n" for e in events))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                paired_records(root)
            event_path.unlink()
            output.write_text(json.dumps([""]))
            missing = paired_records(root)
            self.assertFalse(missing[0]["paired_eligible"])
            self.assertIsNone(summarize(missing)["strict_vote_accuracy"])
            self.assertEqual(summarize(missing)["paired_coverage"], 0)


if __name__ == "__main__":
    unittest.main()
