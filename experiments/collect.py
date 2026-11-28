"""Resumable, training-only profiling and routed paired collection for the approved adaptation."""
import argparse
import ast
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
from prepare_medmcqa import features, parse_answer, question_id, vote
from biomedical_pilot import REPLACEMENT_MODELS, GPUMonitor, answer_prompt, save, sha


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def freeze(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen input/configuration changed: {path}')
    else:
        save(path, value)


def load_cache(path, repair=False):
    """Only an unterminated final record may be discarded after an interrupted write."""
    if not path.exists():
        return {}
    records = {}
    with path.open('r+b' if repair else 'rb') as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                if repair:
                    handle.seek(offset)
                    handle.truncate()
                else:
                    raise ValueError('Unterminated cache record; resume worker to repair')
                break
            record = json.loads(line)
            if record['id'] in records:
                raise ValueError('Duplicate cached request ID')
            records[record['id']] = record
    return records


def worker(args):
    stage_path = Path(args.stage_file)
    stage = json.loads(stage_path.read_text())
    stage_hash = fingerprint(stage)
    cache_path = stage_path.with_suffix('.jsonl')
    with stage_path.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cached = load_cache(cache_path, repair=True)
        expected = {r['id']: fingerprint(r) for r in stage['requests']}
        if len(expected) != len(stage['requests']):
            raise ValueError('Duplicate requested ID')
        for identifier, record in cached.items():
            if identifier not in expected or record['request_sha256'] != expected[identifier]:
                raise ValueError('Cache does not match frozen requests')
            if record['stage_sha256'] != stage_hash:
                raise ValueError('Cache sampling/model configuration changed')
        pending = [r for r in stage['requests'] if r['id'] not in cached]
        if not pending:
            print('All requests already cached:', stage['name'], flush=True)
            return
        import torch
        from vllm import LLM, SamplingParams
        model, revision = stage['model'], stage['revision']
        model_path = Path(args.cache) / ('models--' + model.replace('/', '--')) / 'snapshots' / revision
        monitor = GPUMonitor(getattr(torch.cuda.get_device_properties(0), 'uuid', None))
        monitor.thread.start()
        start = time.monotonic()
        kwargs = {'limit_mm_per_prompt': {'image': 0, 'video': 0}} if stage['model_key'] == 'biodecision' else {}
        try:
            llm = LLM(model=str(model_path), tokenizer=str(model_path), dtype='bfloat16',
                      seed=0, tensor_parallel_size=1, max_model_len=8192, max_num_seqs=32,
                      gpu_memory_utilization=.85, enforce_eager=False, trust_remote_code=True, **kwargs)
            tokenizer = llm.get_tokenizer()
            load_seconds = time.monotonic() - start
            with cache_path.open('a') as handle:
                for offset in range(0, len(pending), 32):
                    batch = pending[offset:offset+32]
                    prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': r['prompt']}],
                               tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in batch]
                    seeds = [int(hashlib.sha256(('0:' + r['id']).encode()).hexdigest()[:8], 16) % (2**31) for r in batch]
                    sampling = [SamplingParams(temperature=.7, max_tokens=stage['max_tokens'],
                                top_p=1, top_k=-1, min_p=0, repetition_penalty=1, seed=seed) for seed in seeds]
                    began = time.monotonic()
                    outputs = llm.generate(prompts, sampling)
                    duration = time.monotonic() - began
                    if len(outputs) != len(batch):
                        raise ValueError('Output/request count mismatch')
                    for request, output, seed in zip(batch, outputs, seeds):
                        completion = output.outputs[0]
                        record = {k: v for k, v in request.items() if k != 'prompt'}
                        record.update({'request_sha256': expected[request['id']], 'stage_sha256': stage_hash,
                                       'model': model, 'revision': revision, 'seed': seed,
                                       'raw_output': completion.text, **parse_answer(completion.text, completion.finish_reason),
                                       'finish_reason': completion.finish_reason, 'prompt_tokens': len(output.prompt_token_ids),
                                       'output_tokens': len(completion.token_ids), 'batch_seconds': duration,
                                       'batch_id': batch[0]['id'], 'slurm_job_id': os.environ.get('SLURM_JOB_ID')})
                        handle.write(json.dumps(record) + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                    print(json.dumps({'stage': stage['name'], 'newly_cached': min(offset+32, len(pending)),
                                      'pending_at_start': len(pending)}), flush=True)
            save(stage_path.with_suffix('.runtime.json'), {'load_seconds': load_seconds,
                 'elapsed_seconds': time.monotonic()-start, 'memory': monitor.finish()})
        finally:
            monitor.done.set()


def stage(run, name, model_key, requests, cache, max_tokens=4096):
    model, revision = REPLACEMENT_MODELS[model_key]
    payload = {'name': name, 'model_key': model_key, 'model': model, 'revision': revision,
               'max_tokens': max_tokens, 'temperature': .7, 'enable_thinking': False,
               'seed_rule': 'sha256(0:request_id) modulo 2**31', 'requests': requests}
    path = run / 'stages' / (name + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    freeze(path, payload)
    with path.with_suffix('.log').open('a') as log:
        subprocess.run([sys.executable, __file__, 'worker', '--stage-file', str(path), '--cache', cache],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    records = load_cache(path.with_suffix('.jsonl'))
    if set(records) != {r['id'] for r in requests}:
        raise ValueError('Incomplete stage')
    return records


def annotation_requests(rows):
    return [{'id': f"keyword:{row['id']}:{draw}", 'question_id': row['id'],
             'prompt': f"Question: {row['question']}\nWhat core knowledge or skills are needed? "
                       "Give ONLY 2-5 comma-separated keywords. Format: Keywords: keyword1, keyword2"}
            for row in rows for draw in range(5)]


def keywords(rows, outputs):
    result = {}
    for row in rows:
        words = []
        for draw in range(5):
            record = outputs[f"keyword:{row['id']}:{draw}"]
            text = record['raw_output'].rsplit('Keywords:', 1)[-1]
            if record['finish_reason'] != 'length':
                words.extend(w.strip().lower().replace('.', '') for w in text.split(','))
        counts = Counter(w for w in words if w and len(w) <= 20)
        result[row['id']] = [word for word, count in counts.items() if count > 1]
    return result


def source_rows():
    train_path, test_path = ROOT/'test_data/MedMCQA_train.json', ROOT/'test_data/MedMCQA_test.json'
    if sha(train_path) != 'dcc23c9d819d0dfe8d92dc7f146e201c43a102afa51e98c3acabd31c5d6b61b0':
        raise ValueError('Training data changed')
    if sha(test_path) != '1c7a0851b37f37c60bbb70af289bf95b52158c74a4bb2d81750085c211481807':
        raise ValueError('Evaluation data changed')
    return json.loads(train_path.read_text()), json.loads(test_path.read_text())


def initialize(args):
    run = Path(args.output)
    run.mkdir(parents=True, exist_ok=True)
    train, test = source_rows()
    indices = sorted(random.Random(0).sample(range(len(train)), args.profile_rows))
    split_path = ROOT/'analysis/medmcqa-data/split_manifest.json'
    split_hash = sha(split_path)
    if split_hash != '082fa4d5e7d6004f7289059a7d50642e7e193b87fc1df839314490fe037df472':
        raise ValueError('Approved split manifest changed')
    manifest = {'models': {k: list(v) for k, v in REPLACEMENT_MODELS.items()}, 'seed': 0,
                'profile_indices': indices, 'profile_rows': len(indices), 'profile_source': 'MedMCQA_train only',
                'split_manifest_sha256': split_hash, 'concise_prompt': True,
                'sample_seed_rule': 'per request SHA-256 of seed 0 and request ID',
                'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in
                                 (Path(__file__), ROOT/'experiments/biomedical_pilot.py',
                                  ROOT/'analysis/prepare_medmcqa.py', ROOT/'recruit_agents.py')}}
    freeze(run/'run-manifest.json', manifest)
    rows = [{'id': 'train:' + str(i), 'source_index': i, 'question': train[i]['question']} for i in indices]
    freeze(run/'training-inputs.json', rows)
    return run, rows, train, test


def profiles(args):
    run, rows, train, _ = initialize(args)
    annotation = stage(run, 'training-keywords', 'qwen', annotation_requests(rows), args.cache, 128)
    kw = keywords(rows, annotation)
    if not any(kw.values()):
        raise ValueError('No usable training keywords')
    save(run/'training-keywords.json', kw)
    profiles_out, quality = {}, {}
    for model in REPLACEMENT_MODELS:
        requests = [{'id': f"profile:{model}:{r['id']}", 'question_id': r['id'],
                     'prompt': answer_prompt(r['question'], concise=True)} for r in rows]
        outputs = stage(run, 'profile-'+model, model, requests, args.cache)
        scores = {}
        valid = correct = 0
        for row in rows:
            response = outputs[f"profile:{model}:{row['id']}"]
            valid += response['parse_status'] == 'valid'
            # Gold is used only here, after response generation, to score training profiles.
            right = response['answer'] == train[row['source_index']]['gold_answer']
            correct += right
            for word in kw[row['id']]:
                scores[word] = scores.get(word, 0) + (1 if right else -1)
        profiles_out[model] = scores
        quality[model] = {'responses': len(rows), 'valid_answers': valid, 'correct_training_answers': correct}
        save(run/'profile-progress.json', quality)
    save(run/'profiles.json', {'checkpoint_revisions': {k:v[1] for k,v in REPLACEMENT_MODELS.items()},
                              'training_indices': [r['source_index'] for r in rows], 'profiles': profiles_out,
                              'quality': quality, 'scoring': '+1 correct / -1 incorrect or invalid per training keyword'})
    print(json.dumps({'phase': 'profiles_complete', 'quality': quality}), flush=True)


def map_keywords(raw, vocabulary, encoder_path):
    """Match upstream MiniLM mean-pooling + normalized cosine nearest keyword, on CPU."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(8)
    encoder_path = Path(encoder_path)
    manifest = json.loads((encoder_path/'source-manifest.json').read_text())
    for name, checksum in manifest['files_sha256'].items():
        if sha(encoder_path/name) != checksum:
            raise ValueError('Keyword encoder file changed: '+name)
    if manifest['revision'] != '1110a243fdf4706b3f48f1d95db1a4f5529b4d41':
        raise ValueError('Keyword encoder revision differs')
    tokenizer = AutoTokenizer.from_pretrained(encoder_path, local_files_only=True)
    model = AutoModel.from_pretrained(encoder_path, local_files_only=True).eval().cpu()
    def encode(texts):
        vectors = []
        with torch.inference_mode():
            for offset in range(0, len(texts), 64):
                batch = tokenizer(texts[offset:offset+64], padding=True, truncation=True,
                                  max_length=256, return_tensors='pt')
                hidden = model(**batch).last_hidden_state
                mask = batch['attention_mask'].unsqueeze(-1)
                pooled = (hidden*mask).sum(1) / mask.sum(1).clamp(min=1)
                vectors.append(torch.nn.functional.normalize(pooled, dim=1))
        return torch.cat(vectors)
    unique = sorted({w for words in raw.values() for w in words})
    if not unique:
        return {identifier: [] for identifier in raw}
    candidates = encode(vocabulary)
    queries = encode(unique)
    mapping = {}
    for offset in range(0, len(unique), 256):
        nearest = (queries[offset:offset+256] @ candidates.T).argmax(dim=1).tolist()
        mapping.update({word: vocabulary[index] for word, index in zip(unique[offset:offset+256], nearest)})
    return {identifier: [mapping[w] for w in words] for identifier, words in raw.items()}


def route(rows, mapped, profile_data):
    tree = ast.parse((ROOT/'recruit_agents.py').read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('normalize_to_probability', 'recruit_agents')]
    namespace = {'random': random.Random(0)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'upstream-routing', 'exec'), namespace)
    profiles_data = profile_data['profiles']
    names = list(REPLACEMENT_MODELS)
    strengths = namespace['normalize_to_probability']([sum(profiles_data[n].values()) for n in names], .5)
    global_strength = dict(zip(names, strengths))
    result = {}
    for row in rows:
        scores = {}
        for word in mapped[row['id']]:
            for name in names:
                if word in profiles_data[name]:
                    scores[name] = scores.get(name, 0) + profiles_data[name][word]
        if scores:
            ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
            solvers, distribution = namespace['recruit_agents'](ordered, global_strength, k=3, return_probabilities=True)
            fallback = None
        else:
            distribution = {name: 1/len(names) for name in names}
            solvers = namespace['random'].choices(names, k=3)
            fallback = 'empty_keywords_uniform'
        features([None]*3, solvers, distribution)  # validate actual probability vector
        result[row['id']] = {'solvers': solvers, 'probabilities': distribution,
                             'confidence': max(distribution.values()), 'fallback': fallback,
                             'mapped_keywords': mapped[row['id']]}
    return result


def collect(args):
    run, _, _, test = initialize(args)
    profile_data = json.loads((run/'profiles.json').read_text())
    if profile_data['checkpoint_revisions'] != {k:v[1] for k,v in REPLACEMENT_MODELS.items()}:
        raise ValueError('Profiles do not match checkpoints')
    splits = json.loads((ROOT/'analysis/medmcqa-data/split_manifest.json').read_text())
    split_map = {g['question_id']: g['split'] for g in splits}
    if args.mode == 'smoke':
        indices = list(range(5))
    else:
        indices = [i for g in splits if g['split'] in ('fit','validation','held_out') for i in g['source_indices']]
    rows = [{'id': question_id(test[i]['question']), 'source_index': i, 'question': test[i]['question']} for i in indices]
    phase = args.mode
    outputs = stage(run, phase+'-keywords', 'qwen', annotation_requests(rows), args.cache, 128)
    raw = keywords(rows, outputs)
    vocabulary = sorted({w for p in profile_data['profiles'].values() for w in p})
    mapped = map_keywords(raw, vocabulary, args.encoder)
    routing = route(rows, mapped, profile_data)
    freeze(run/(phase+'-routes.json'), routing)
    expert_outputs = {}
    for model in REPLACEMENT_MODELS:
        requests = [{'id': f"{phase}:{model}:{row['id']}:{slot}", 'question_id': row['id'], 'slot': slot,
                     'prompt': answer_prompt(row['question'], concise=True)}
                    for row in rows for slot, name in enumerate(routing[row['id']]['solvers']) if name == model]
        if requests:
            expert_outputs.update(stage(run, phase+'-expert-'+model, model, requests, args.cache))
    aggregator_requests = []
    for row in rows:
        responses = [expert_outputs[f"{phase}:{model}:{row['id']}:{slot}"]
                     for slot, model in enumerate(routing[row['id']]['solvers'])]
        prompt = 'Critically evaluate these model responses and synthesize an accurate answer; some may be wrong.\n'
        prompt += '\n'.join(f"Model {i+1}: {r['raw_output'].rsplit('</think>',1)[-1]}" for i,r in enumerate(responses))
        aggregator_requests.append({'id': f"{phase}:aggregate:{row['id']}", 'question_id': row['id'],
                                    'prompt': prompt+'\n'+answer_prompt(row['question'], concise=True)})
    aggregated = stage(run, phase+'-aggregate', 'qwen', aggregator_requests, args.cache)
    records = []
    for row in rows:
        r = routing[row['id']]
        experts = [{**expert_outputs[f"{phase}:{model}:{row['id']}:{slot}"], 'solver': model}
                   for slot,model in enumerate(r['solvers'])]
        aggregator = aggregated[f"{phase}:aggregate:{row['id']}"]
        majority, tied = vote([e['answer'] for e in experts])
        distinct = {}
        for e in experts:
            distinct.setdefault(e['solver'], e['answer'])
        distinct_vote, distinct_tie = vote(list(distinct.values()))
        # Store reference separately; held-out correctness is not inspected or summarized here.
        records.append({'schema_version': 'routed-1.0', 'question_id': row['id'],
                        'source_index': row['source_index'], 'split': split_map[row['id']],
                        'question': row['question'], 'router': r, 'experts': experts, 'aggregator': aggregator,
                        'features': features([e['answer'] for e in experts], r['solvers'], r['probabilities']),
                        'vote': {'answer': majority, 'tied': tied, 'distinct_expert_answer': distinct_vote,
                                 'distinct_expert_tied': distinct_tie}})
    save(run/(phase+'-records.json'), records)
    save(run/(phase+'-status.json'), {'questions': len(rows), 'complete': True,
         'expert_parse_status': dict(Counter(e['parse_status'] for r in records for e in r['experts'])),
         'aggregator_parse_status': dict(Counter(r['aggregator']['parse_status'] for r in records)),
         'router_confidence_min': min(r['router']['confidence'] for r in records),
         'router_confidence_max': max(r['router']['confidence'] for r in records),
         'outcomes_scored': False})
    print(phase, 'collection complete:', len(rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('worker','profiles','smoke','evaluate'))
    parser.add_argument('--output', default='/workspace/experiments/runs/routed-primary')
    parser.add_argument('--cache', default='/opt/model-cache')
    parser.add_argument('--encoder', default='/workspace/experiments/model-cache/all-MiniLM-L6-v2')
    parser.add_argument('--profile-rows', type=int, default=504)
    parser.add_argument('--stage-file')
    args = parser.parse_args()
    if args.mode == 'worker':
        worker(args)
    else:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        with (out/'pipeline.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            profiles(args) if args.mode == 'profiles' else collect(args)
