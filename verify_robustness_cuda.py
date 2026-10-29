"""CPU audit of saved prompt stress inference; proxies are not human labels."""
import argparse
from collections import defaultdict
import math
from statistics import mean
from rouge_score.rouge_scorer import RougeScorer
from cuda_common import ROOT,read_json,write_json,digest,file_hash
from robustness_protocol import FREEZE,validate_suite,select_rows,validate_prefix,valid_name,expected_input


def verify(run_id,records_only=False):
    valid_name(run_id)
    directory=ROOT/'study/robustness/cuda-runs'/run_id
    run=read_json(directory/'run.json')
    record=read_json(directory/'evaluation.json')
    provenance=run['provenance']
    assert record['status']=='completed' and record['provenance']==provenance
    assert digest(provenance)==run['signature']==record['signature']
    assert provenance['batch_size']==8 and provenance['max_new_tokens']==200
    assert provenance['decoding']==dict(do_sample=False,top_p=None,temperature=None,use_cache=True)
    assert provenance['suite_sha256']==file_hash(FREEZE/'suite.json')
    suite=validate_suite(read_json(FREEZE/'suite.json'))
    expected=select_rows(suite,provenance['question_limit_per_domain'])
    assert len(record['rows'])==len(expected)
    validate_prefix(record['rows'],expected,8)
    for name,sha in provenance['runtime_sources'].items():
        path=(directory/'code_snapshot'/name).resolve()
        assert path.is_relative_to((directory/'code_snapshot').resolve()) and file_hash(path)==sha,name
    scorer=RougeScorer(['rougeL'],use_stemmer=True)
    groups=defaultdict(list)
    for actual,wanted in zip(record['rows'],expected):
        assert all(actual[key]==value for key,value in wanted.items())
        assert actual['processed_input']==expected_input(wanted)
        assert actual['manual_fact_recovered'] is None
        value=scorer.score(wanted['reference_answer'],actual['generation'])['rougeL'].recall
        assert math.isfinite(value) and math.isclose(value,actual['rougeL_recall_proxy'],rel_tol=1e-12,abs_tol=1e-12)
        groups[(actual['domain'],actual['family'])].append(value)
    count=provenance['question_limit_per_domain'] or 40
    assert len(groups)==8 and all(len(values)==count for values in groups.values())
    model_directory=ROOT/'study/runs'/provenance['model_run']
    model=read_json(model_directory/'run.json')
    assert model['signature']==provenance['model_run_signature']
    assert model['spec']['method']==provenance['method']
    checkpoint=ROOT/'artifacts/model/model.safetensors' if provenance['method']=='baseline' else model_directory/'model/model.safetensors'
    if not records_only:assert file_hash(checkpoint)==provenance['checkpoint_sha256']
    verified=read_json(model_directory/'cpu_verification.json')
    assert verified['status']=='passed' and verified['checkpoint_sha256']==provenance['checkpoint_sha256']
    report=dict(status='passed',run_id=run_id,counts={f'{d}/{f}':len(v) for (d,f),v in groups.items()},
        total_generations=len(expected),rouge_proxy_recomputed=True,processed_inputs_verified=True,
        source_snapshots_verified=True,checkpoint_bytes_verified=not records_only,
        model_run=provenance['model_run'],checkpoint_sha256=provenance['checkpoint_sha256'],
        proxies={f'{d}/{f}':mean(v) for (d,f),v in groups.items()},verifier_sha256=file_hash(ROOT/'verify_robustness_cuda.py'),
        scope='Saved outputs and lexical-overlap proxies audited on CPU. No fresh inference or human factual-recovery claim.')
    write_json(directory/'cpu_verification.json',report)
    print(report,flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run_id')
    parser.add_argument('--records-only',action='store_true',help='Verify saved records without claiming checkpoint-byte verification')
    args=parser.parse_args();verify(args.run_id,args.records_only)
