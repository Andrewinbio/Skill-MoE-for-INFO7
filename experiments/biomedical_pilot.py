"""Pinned, development-only biomedical-pool pilot. No primary evaluation is run."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import threading
import time

MODELS = {
    "qwen": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    "biodecision": ("lighteternal/biodecision-v2-4b", "181a2148dd6e8b815f220b2a090c27b6f69b4b9c"),
    "medgemma": ("unsloth/medgemma-27b-text-it", "b780610baf99c087ba3719a77cf0dacec7261a65"),
}
REPLACEMENT_MODELS = {
    "qwen": MODELS["qwen"],
    "biodecision": MODELS["biodecision"],
    "coder": ("QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ", "c58857a7f41c0920f73d1b56678640f9c02017d7"),
}


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def answer_prompt(question, concise=False):
    if concise:
        return (f"Question: {question}\nGive a brief explanation of at most 150 words. "
                "Choose exactly one option. End with a separate plain-text line in exactly this format: "
                "The answer is (X)\nReplace X with the single capital answer-choice letter. "
                "Use parentheses. Do not use Markdown on that final line or write anything after it.")
    return (f"Question: {question}\nProvide your step-by-step reasoning first, then "
            'print "The answer is (X)", where X is one capital answer-choice letter, '
            "at the end of your response.")


def parse(text, finish):
    if finish == "length" or ("<think>" in text and "</think>" not in text):
        return None
    matches = re.findall(r"\banswer\s+is\s*\(([A-D])\)", text.rsplit("</think>", 1)[-1], re.I)
    return matches[-1].upper() if matches else None


class GPUMonitor:
    """Sample total allocated-device usage (includes vLLM child processes)."""
    def __init__(self, uuid):
        self.uuid = str(uuid) if uuid else None
        if self.uuid and not self.uuid.startswith(("GPU-", "MIG-")):
            self.uuid = "GPU-" + self.uuid
        self.values = []
        self.errors = []
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def poll(self):
        while not self.done.is_set():
            if self.uuid:
                try:
                    result = subprocess.run([
                        "nvidia-smi", "-i", self.uuid, "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ], capture_output=True, text=True, check=True, timeout=5)
                    self.values.append(float(result.stdout.strip()))
                except Exception as exc:
                    if not self.errors:
                        self.errors.append(str(exc))
            self.done.wait(.5)

    def finish(self):
        self.done.set()
        self.thread.join(timeout=6)
        return {"device_uuid": self.uuid, "sampling_interval_seconds": .5,
                "baseline_mib": self.values[0] if self.values else None,
                "peak_device_memory_mib": max(self.values) if self.values else None,
                "samples": len(self.values), "errors": self.errors,
                "measurement": "Sampled total GPU memory on allocated device, including engine subprocesses; not torch main-process allocation."}


def worker(args):
    started = time.monotonic()
    import torch
    from vllm import LLM, SamplingParams

    key = "qwen" if args.stage == "aggregate" else args.stage
    models = REPLACEMENT_MODELS if args.pool == "replacement" else MODELS
    model, revision = models[key]
    model_path = Path(args.cache) / ("models--" + model.replace("/", "--")) / "snapshots" / revision
    props = torch.cuda.get_device_properties(0)
    monitor = GPUMonitor(getattr(props, "uuid", None))
    monitor.thread.start()
    report = {"stage": args.stage, "model": model, "revision": revision,
              "gpu": props.name, "gpu_total_memory_bytes": props.total_memory,
              "versions": {name: importlib.metadata.version(name) for name in ("torch", "vllm", "transformers")},
              "settings": {"seed": 0, "temperature": .7, "max_tokens": 4096,
                           "max_model_len": 8192, "max_num_seqs": args.max_num_seqs,
                           "enforce_eager": not args.compiled, "enable_thinking": False,
                           "concise_prompt": args.concise}, "batches": []}
    output = Path(args.output)
    report_path = output / f"{args.stage}.json"
    try:
        load_start = time.monotonic()
        kwargs = {}
        if key == "biodecision":
            kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        llm = LLM(model=str(model_path), tokenizer=str(model_path),
                  dtype="bfloat16", tensor_parallel_size=1, seed=0,
                  max_model_len=8192, max_num_seqs=args.max_num_seqs, gpu_memory_utilization=.85,
                  enforce_eager=not args.compiled, trust_remote_code=True, **kwargs)
        tokenizer = llm.get_tokenizer()
        report["load_seconds"] = time.monotonic() - load_start
        manifest = json.loads((output / "pilot-inputs.json").read_text())

        def generate(name, prompts, ids, max_tokens=4096, temperature=.7):
            rendered = [tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            ) for text in prompts]
            params = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                    top_p=1, top_k=-1, min_p=0, repetition_penalty=1)
            begin = time.monotonic()
            results = llm.generate(rendered, params)
            elapsed = time.monotonic() - begin
            records = []
            if len(results) != len(ids):
                raise ValueError("Generation count mismatch")
            for identifier, result in zip(ids, results):
                completion = result.outputs[0]
                records.append({"id": identifier, "raw_output": completion.text,
                                "answer": parse(completion.text, completion.finish_reason),
                                "finish_reason": completion.finish_reason,
                                "prompt_tokens": len(result.prompt_token_ids),
                                "output_tokens": len(completion.token_ids)})
            batch = {"name": name, "seconds": elapsed, "count": len(records),
                     "max_tokens": max_tokens, "records": records}
            report["batches"].append(batch)
            save(report_path, report)
            return batch

        if args.stage == "aggregate":
            for kind in ("development", "training_timing"):
                prompts = []
                for i, row in enumerate(manifest[kind]):
                    responses = []
                    for expert in models:
                        path = output / f"{expert}.json"
                        if path.exists():
                            batches = json.loads(path.read_text()).get("batches", [])
                            batch = next((b for b in batches if b["name"] == kind), None)
                            if batch:
                                responses.append(batch["records"][i]["raw_output"].rsplit("</think>", 1)[-1])
                    if len(responses) != 3:
                        raise ValueError("All three experts must complete before pilot aggregation")
                    prompt = ("Critically evaluate the following model responses and synthesize a single accurate answer. "
                              "Do not simply copy an answer; some responses may be incorrect.\n")
                    prompt += "\n".join(f"Model {j+1}: {text}" for j, text in enumerate(responses))
                    prompts.append(prompt + "\n" + answer_prompt(row["question"], args.concise))
                generate("aggregation" if kind == "development" else "aggregation_training_timing",
                         prompts, [r["id"] for r in manifest[kind]])
        else:
            for name in ("development", "training_timing"):
                rows = manifest[name]
                generate(name, [answer_prompt(row["question"], args.concise) for row in rows], [r["id"] for r in rows])
            if key == "qwen":
                rows = manifest["training_timing"]
                prompts, ids = [], []
                for repetition in range(5):
                    for row in rows:
                        prompts.append(f"Question: {row['question']}\nWhat core knowledge or skills are needed? "
                                       "Give ONLY 2-5 comma-separated keywords. Format: Keywords: keyword1, keyword2")
                        ids.append(f"{row['id']}:annotation-{repetition}")
                generate("keyword_timing", prompts, ids, max_tokens=128)
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["memory"] = monitor.finish()
        report["worker_seconds"] = time.monotonic() - started
        save(report_path, report)


def driver(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "pilot-inputs.json").exists():
        raise ValueError("Use a fresh pilot output directory; do not overwrite a prior run")
    root = Path(args.root)
    train_path = root / "test_data/MedMCQA_train.json"
    test_path = root / "test_data/MedMCQA_test.json"
    if sha(train_path) != "dcc23c9d819d0dfe8d92dc7f146e201c43a102afa51e98c3acabd31c5d6b61b0":
        raise ValueError("Training file hash changed")
    if sha(test_path) != "1c7a0851b37f37c60bbb70af289bf95b52158c74a4bb2d81750085c211481807":
        raise ValueError("Test file hash changed")
    train = json.loads(train_path.read_text())
    test = json.loads(test_path.read_text())
    models = REPLACEMENT_MODELS if args.pool == "replacement" else MODELS
    indices = sorted(random.Random(0).sample(range(len(train)), args.training_size))
    # Only the already-exposed five development examples are used from test data.
    # Gold labels are deliberately omitted from all worker inputs.
    manifest = {"seed": 0, "models": models, "pool": args.pool, "train_sha256": sha(train_path),
                "test_sha256": sha(test_path), "script_sha256": sha(Path(__file__)),
                "training_indices": indices,
                "training_timing": [{"id": f"train:{i}", "question": train[i]["question"]} for i in indices],
                "development": [{"id": f"development:{i}", "question": test[i]["question"]} for i in range(5)]}
    save(output / "pilot-inputs.json", manifest)
    states = []
    for stage in [*models, "aggregate"]:
        began = time.monotonic()
        command = [sys.executable, __file__, "worker", "--stage", stage,
                   "--output", str(output), "--cache", args.cache,
                   "--pool", args.pool, "--max-num-seqs", str(args.max_num_seqs)]
        if args.compiled:
            command.append("--compiled")
        if args.concise:
            command.append("--concise")
        with (output / f"{stage}.log").open("w") as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=720)
                code = result.returncode
            except subprocess.TimeoutExpired:
                # Stop the whole pilot on timeout to avoid overlapping engine allocations.
                states.append({"stage": stage, "status": "timeout", "seconds": time.monotonic()-began})
                save(output / "pilot-status.json", states)
                raise
        states.append({"stage": stage, "exit_code": code, "seconds": time.monotonic()-began})
        save(output / "pilot-status.json", states)
        print(json.dumps(states[-1]), flush=True)
    if any(s.get("exit_code") != 0 for s in states):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("driver", "worker"))
    parser.add_argument("--stage", choices=[*MODELS, "coder", "aggregate"])
    parser.add_argument("--pool", choices=("biomedical", "replacement"), default="biomedical")
    parser.add_argument("--training-size", type=int, default=12)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--concise", action="store_true")
    parser.add_argument("--root", default="/workspace")
    parser.add_argument("--cache", default="/opt/model-cache")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    worker(args) if args.mode == "worker" else driver(args)
