"""Verify published records without downloading or loading a language model."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COUNTS = {'forget': 40, 'retain': 400, 'real_authors': 100, 'world_facts': 117}
UTILITY = [f'{p}_{s}' for p in ('retain', 'ra', 'wf') for s in
           (('Q_A_Prob', 'Q_A_ROUGE', 'Truth_Ratio') if p == 'retain' else
            ('Q_A_Prob_normalised', 'Q_A_ROUGE', 'Truth_Ratio'))]


def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-12), (a, b)


def verify():
    summary = load('reproduction_full/summary.json')
    baseline = load('reproduction_full/runs/baseline/evaluation.json')
    traces = []
    for method in ('baseline', 'GradAscent', 'GradDiff'):
        directory = f'reproduction_full/runs/{method}'
        evaluation = load(f'{directory}/evaluation.json')
        raw = load(f'{directory}/TOFU_EVAL.json')
        assert evaluation['status'] == 'completed' and evaluation['counts'] == COUNTS
        signature = hashlib.sha256(json.dumps(evaluation['provenance'], sort_keys=True).encode()).hexdigest()
        assert signature == evaluation['signature']
        assert evaluation['provenance']['evaluator_sha256'] == sha('evaluate_full.py')
        assert evaluation['provenance']['core_sha256'] == sha('full_core.py')
        mu = len(UTILITY) / sum(1 / raw[k]['agg_value'] for k in UTILITY)
        close(mu, evaluation['metrics']['model_utility'])
        for metric, value in evaluation['metrics'].items():
            close(value, summary['local_results'][method][metric])
            close(value, raw[metric]['agg_value'])
        for domain, count in COUNTS.items():
            rows = evaluation['domains'][domain]
            assert len(rows) == count and sorted(r['source_index'] for r in rows) == list(range(count))
            originals = {r['source_index']: r for r in baseline['domains'][domain]}
            for row in rows:
                assert row['input'] == originals[row['source_index']]['input']
                assert row['ground_truth'] == originals[row['source_index']]['ground_truth']
        if method != 'baseline':
            record = load(f'{directory}/training.json')
            assert record['status'] == 'completed'
            assert (record['steps'], record['epochs'], record['visits']) == (10, 8, 320)
            assert record['trainable_parameters'] == 1235814400
            assert record['checkpoint_sha256'] == evaluation['provenance']['checkpoint_sha256']
            assert record['optimizer_source_sha256'] == sha('streamed_adam.py')
            assert record['execution_adapter']['sha256'] == sha('memory_gd.py')
            traces.append(record['sampling_trace'])
    assert traces[0] == traces[1], 'GA/GD sampled different training pairs'
    for entry in load('artifacts/tofu/manifest.json')['files']:
        assert sha('artifacts/tofu/' + entry['file']) == entry['sha256']
    assert sha('artifacts/tofu/retain99-reference.json') == 'aa6a504c44487fe5174c1828e4c3bfe997de89856c95cba721dc833bbb37aee3'
    suite = load('study/robustness/suite.json')
    assert suite['source_sha256'] == sha('reproduction_full/runs/baseline/evaluation.json')
    assert len(suite['items']) == 320 and len({r['id'] for r in suite['items']}) == 320
    for domain in ('forget', 'retain'):
        subset = [r for r in suite['items'] if r['domain'] == domain]
        assert len(subset) == 160 and len({r['source_index'] for r in subset}) == 40
    assert not summary['tested_claims']['GD_preserves_more_utility_than_GA']['direction_matches']
    print('PASS: complete 657-question evaluations, metric arithmetic, provenance, data hashes, matched sampling, and 320 prepared stress-test prompts. No model inference or new training claimed.')


if __name__ == '__main__':
    verify()
