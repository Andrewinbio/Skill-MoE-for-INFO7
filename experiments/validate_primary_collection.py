"""Check collection coverage, cache identity and pre-aggregation features without scoring."""
import argparse
from collections import Counter
import json
from pathlib import Path

from collect import ROOT, REPLACEMENT_MODELS
from prepare_medmcqa import features, parse_answer, vote
from validate_routed import check_stage, read, require


def validate(run):
    groups = read(ROOT / 'analysis/medmcqa-data/split_manifest.json')
    expected = {g['question_id']: (idx, g['split']) for g in groups
                if g['split'] in ('fit', 'validation', 'held_out') for idx in g['source_indices']}
    rows = read(run / 'evaluate-records.json')
    routes = read(run / 'evaluate-routes.json')
    require(len(expected) == len(rows) == len(routes) == 4178, 'Incorrect question coverage')
    require({r['question_id'] for r in rows} == set(routes) == set(expected), 'Incorrect question IDs')
    split_counts = Counter(r['split'] for r in rows)
    require(split_counts == {'fit': 2507, 'validation': 831, 'held_out': 840}, 'Wrong split counts')
    check_stage(run, 'evaluate-keywords', 5 * len(rows))
    aggregate = check_stage(run, 'evaluate-aggregate', len(rows))
    draws = Counter(e['solver'] for r in rows for e in r['experts'])
    caches = {m: check_stage(run, 'evaluate-expert-' + m, n) for m, n in draws.items()}
    status = {'expert': Counter(), 'aggregator': Counter()}
    finish = {'expert': Counter(), 'aggregator': Counter()}
    confidence, fallback = [], Counter()
    repeats = 0
    for row in rows:
        qid = row['question_id']
        route = routes[qid]
        require((row['source_index'], row['split']) == expected[qid] and row['router'] == route,
                'Misaligned question or routing')
        require('gold_answer' not in row and 'labels' not in row, 'Unexpected scoring fields')
        require(set(route['probabilities']) == set(REPLACEMENT_MODELS), 'Wrong router pool')
        require(len(row['experts']) == len(route['solvers']) == 3, 'Wrong slot count')
        answers, distinct = [], {}
        for slot, expert in enumerate(row['experts']):
            solver = route['solvers'][slot]
            raw = caches[solver][f'evaluate:{solver}:{qid}:{slot}']
            require(expert == {**raw, 'solver': solver} and expert['slot'] == slot, 'Wrong expert slot/cache')
            require((expert['model'], expert['revision']) == REPLACEMENT_MODELS[solver], 'Wrong expert checkpoint')
            answers.append(expert['answer'])
            distinct.setdefault(solver, expert['answer'])
        agg = row['aggregator']
        require(agg == aggregate[f'evaluate:aggregate:{qid}'] and
                (agg['model'], agg['revision']) == REPLACEMENT_MODELS['qwen'], 'Wrong aggregator checkpoint/cache')
        for kind, outputs in [('expert', row['experts']), ('aggregator', [agg])]:
            for output in outputs:
                parsed = parse_answer(output['raw_output'], output['finish_reason'])
                require(all(output[k] == v for k, v in parsed.items()), 'Answer parse mismatch')
                status[kind][parsed['parse_status']] += 1
                finish[kind][output['finish_reason']] += 1
        f = features(answers, route['solvers'], route['probabilities'])
        require(f == row['features'] and f['router_confidence'] == route['confidence'], 'Wrong features')
        majority, tied = vote(answers)
        unique, unique_tied = vote(list(distinct.values()))
        require(row['vote'] == {'answer': majority, 'tied': tied, 'distinct_expert_answer': unique,
                               'distinct_expert_tied': unique_tied}, 'Wrong vote')
        confidence.append(route['confidence'])
        fallback[route['fallback'] or 'none'] += 1
        repeats += len(distinct) < 3
    return {'questions': len(rows), 'splits': dict(split_counts), 'expert_draws': dict(draws),
            'parse_status': status, 'finish_reasons': finish, 'questions_with_repeats': repeats,
            'router_confidence_range': [min(confidence), max(confidence)], 'router_fallback': dict(fallback),
            'outcomes_scored': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.run), indent=2))
