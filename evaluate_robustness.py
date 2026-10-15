"""Run a separate, resumable prompt stress test on a saved Mac checkpoint."""
import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['baseline', 'GradAscent', 'GradDiff'], required=True)
    parser.add_argument('--limit', type=int, default=0, help='Questions per domain; 0 uses all 40. Nonzero runs are pilots.')
    parser.add_argument('--max-new-tokens', type=int, default=200)
    args = parser.parse_args()
    if not 0 <= args.limit <= 40 or not 1 <= args.max_new_tokens <= 512:
        parser.error('limit must be 0..40; max-new-tokens must be 1..512')
    import torch
    from omegaconf import OmegaConf
    from rouge_score.rouge_scorer import RougeScorer
    from full_core import BASE, official_item, file_hash
    from train_full import load_full
    from data.collators import DataCollatorForSupervisedDataset
    from evals.metrics.utils import eval_text_similarity
    suite_path = ROOT / 'study/robustness/suite.json'
    suite = json.loads(suite_path.read_text())
    rows = [r for r in suite['items'] if not args.limit or r['source_index'] < args.limit]
    model_path = BASE if args.stage == 'baseline' else ROOT / 'reproduction_full/runs' / args.stage / 'model'
    # Models must be restored from a release before running this command.
    signature_data = {'checkpoint_sha256': file_hash(model_path / 'model.safetensors'),
                      'suite_sha256': file_hash(suite_path), 'evaluator_sha256': file_hash(Path(__file__)),
                      'stage': args.stage, 'limit': args.limit, 'max_new_tokens': args.max_new_tokens,
                      'decoding': 'greedy'}
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    destination = ROOT / 'study/robustness/runs' / f'{args.stage}-q{args.limit or 40}-t{args.max_new_tokens}'
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / 'evaluation.json'
    record = {'status': 'running', 'signature': signature, 'provenance': signature_data, 'rows': []}
    if output.exists():
        record = json.loads(output.read_text())
        assert record['signature'] == signature, 'Changed settings; use a distinct output directory'
    finished = {r['id'] for r in record['rows']}
    already_complete = finished == {r['id'] for r in rows}
    scorer = RougeScorer(['rougeL'], use_stemmer=True)
    with (ROOT / 'reproduction_full/gpu.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not already_complete:
            model, tokenizer, template = load_full(model_path)
            model.eval()
            model.generation_config.eos_token_id = json.loads((BASE / 'generation_config.json').read_text())['eos_token_id']
            collator = DataCollatorForSupervisedDataset(tokenizer, padding_side='left')
        for row in rows:
            if row['id'] in finished:
                continue
            item = official_item(tokenizer, template, row['prompt'], '', generation=True)
            start = time.monotonic()
            with torch.inference_mode():
                answer = eval_text_similarity(model, tokenizer, collator([item]), OmegaConf.create({
                    'do_sample': False, 'top_p': None, 'temperature': None,
                    'max_new_tokens': args.max_new_tokens, 'use_cache': True}))[0]['generation']
            record['rows'].append({**row, 'generation': answer,
                                   'rougeL_recall_proxy': scorer.score(row['reference_answer'], answer)['rougeL'].recall,
                                   'seconds': time.monotonic() - start,
                                   'manual_fact_recovered': None})
            record['status'] = 'completed' if len(record['rows']) == len(rows) else 'running'
            temp = output.with_suffix('.tmp')
            temp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
            temp.replace(output)
            print(f"{args.stage} {len(record['rows'])}/{len(rows)} {row['id']}", flush=True)
            torch.mps.empty_cache()
    with (destination / 'manual-review.csv').open('w', newline='') as stream:
        fields = ['id', 'domain', 'family', 'question', 'reference_answer', 'generation',
                  'fact_recovered_yes_no_unclear', 'reviewer', 'notes']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in record['rows']:
            writer.writerow({k: row.get(k, '') for k in fields})
    print('Complete. ROUGE is a screening proxy only; factual recovery requires manual review.')


if __name__ == '__main__':
    main()
