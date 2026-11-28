"""Validate profiling artifacts and the recruited development smoke run."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from collect import REPLACEMENT_MODELS, load_cache, source_rows
from prepare_medmcqa import features, parse_answer, question_id, vote


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def check_stage(run, name, count):
    requests = read(run / 'stages' / (name + '.json'))['requests']
    records = load_cache(run / 'stages' / (name + '.jsonl'))
    require(len(requests) == count == len(records), f'{name}: wrong request count')
    require({r['id'] for r in requests} == set(records), f'{name}: request ID mismatch')
    require(all(set(r) <= {'id', 'question_id', 'slot', 'prompt'} for r in requests),
            f'{name}: unexpected inference input fields')
    return records


def validate(run, smoke=False):
    train, test = source_rows()
    models = {key: list(value) for key, value in REPLACEMENT_MODELS.items()}
    manifest = read(run / 'run-manifest.json')
    profiles = read(run / 'profiles.json')
    inputs = read(run / 'training-inputs.json')
    indices = list(range(len(train)))
    require(len(indices) == 504 and [r['source_index'] for r in inputs] == indices,
            'Wrong profiling sample')
    require(inputs == [{'id': f'train:{i}', 'source_index': i, 'question': train[i]['question']}
                       for i in indices], 'Training inputs differ from frozen source')
    require(manifest['models'] == models and profiles['training_indices'] == indices and
            profiles['checkpoint_revisions'] == {k: v[1] for k, v in models.items()},
            'Profile checkpoint or sample mismatch')
    check_stage(run, 'training-keywords', 5 * len(indices))
    keywords = read(run / 'training-keywords.json')
    require(set(keywords) == {r['id'] for r in inputs}, 'Training keyword coverage differs')
    for key, (model, revision) in models.items():
        generated = check_stage(run, 'profile-' + key, len(indices))
        scores, valid, right = Counter(), 0, 0
        for i in indices:
            r = generated[f'profile:{key}:train:{i}']
            require((r['model'], r['revision']) == (model, revision), 'Wrong profile response checkpoint')
            parsed = parse_answer(r['raw_output'], r['finish_reason'])
            require(all(r[k] == v for k, v in parsed.items()), 'Profile parse mismatch')
            correct = parsed['answer'] == train[i]['gold_answer']
            valid += parsed['parse_status'] == 'valid'
            right += correct
            for word in keywords[f'train:{i}']:
                scores[word] += 1 if correct else -1
        require(dict(scores) == profiles['profiles'][key] and scores, 'Training profile scores differ')
        require(profiles['quality'][key] == {'responses': len(indices), 'valid_answers': valid,
                                             'correct_training_answers': right}, 'Profile counts differ')
    report = {'profiles': profiles['quality'], 'training_rows_with_keywords': sum(map(bool, keywords.values()))}
    if not smoke:
        return report

    rows = read(run / 'smoke-records.json')
    routes = read(run / 'smoke-routes.json')
    require(len(rows) == len(routes) == 5, 'Invalid smoke coverage')
    check_stage(run, 'smoke-keywords', 25)
    aggregated = check_stage(run, 'smoke-aggregate', 5)
    counts = Counter(e['solver'] for r in rows for e in r['experts'])
    expert_cache = {}
    for model, count in counts.items():
        expert_cache.update(check_stage(run, 'smoke-expert-' + model, count))
    valid = Counter()
    score = Counter()
    confidence = []
    for i, (row, source) in enumerate(zip(rows, test[:5])):
        qid = question_id(source['question'])
        require(row['question_id'] == qid and row['source_index'] == i and row['split'] == 'development_smoke',
                'Non-development or misaligned smoke question')
        require('gold_answer' not in row, 'Gold in collection record')
        route = routes[qid]
        require(row['router'] == route and set(route['probabilities']) == set(models), 'Invalid routing metadata')
        answers = []
        distinct = {}
        require(len(row['experts']) == 3, 'Missing expert draw')
        for slot, expert in enumerate(row['experts']):
            model = route['solvers'][slot]
            expected = expert_cache[f'smoke:{model}:{qid}:{slot}']
            require(expert == {**expected, 'solver': model} and expert['slot'] == slot and
                    [expert['model'], expert['revision']] == models[model], 'Draw order or checkpoint mismatch')
            answers.append(expert['answer'])
            distinct.setdefault(model, expert['answer'])
            valid['expert'] += expert['parse_status'] == 'valid'
        agg = row['aggregator']
        require(agg == aggregated[f'smoke:aggregate:{qid}'] and [agg['model'], agg['revision']] == models['qwen'],
                'Aggregator cache or checkpoint mismatch')
        valid['aggregator'] += agg['parse_status'] == 'valid'
        predictors = features(answers, route['solvers'], route['probabilities'])
        require(predictors == row['features'] and math.isclose(predictors['router_confidence'], route['confidence']),
                'Router confidence or features differ')
        confidence.append(route['confidence'])
        majority, tied = vote(answers)
        unique_vote, unique_tie = vote(list(distinct.values()))
        require(row['vote'] == {'answer': majority, 'tied': tied, 'distinct_expert_answer': unique_vote,
                               'distinct_expert_tied': unique_tie}, 'Vote mismatch')
        score['vote_correct'] += majority == source['gold_answer']
        score['aggregate_correct'] += agg['answer'] == source['gold_answer']
    require(len(set(confidence)) > 1 and len(counts) > 1, 'Smoke did not exercise informative multi-model routing')
    require(valid['expert'] > 0 and valid['aggregator'] > 0 and all(
        any(e['parse_status'] == 'valid' for e in r['experts']) and r['aggregator']['parse_status'] == 'valid'
        for r in rows), 'Smoke did not produce a parseable paired decision for every question')
    report['smoke'] = {'questions': len(rows), 'draw_counts': dict(counts), 'valid_answers': dict(valid),
                       'router_confidences': confidence, 'development_scores': dict(score)}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    print(json.dumps(validate(args.run, args.smoke), indent=2))
