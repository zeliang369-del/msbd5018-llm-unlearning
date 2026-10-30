"""Separate CUDA prompt stress inference; preserves all formal/frozen results."""
import argparse
import csv
import json
from pathlib import Path
import time
import traceback
import uuid
from cuda_common import ROOT,COUNTS,read_json,write_json,digest,file_hash,source_identity,preflight,gpu_lock
from robustness_protocol import FREEZE,validate_suite,select_rows,validate_prefix,valid_name,expected_input


def snapshot(destination,sources):
    for name,expected in sources.items():
        original=(ROOT/name).resolve()
        target=(destination/'code_snapshot'/name).resolve()
        if not original.is_relative_to(ROOT) or not target.is_relative_to(destination/'code_snapshot'):
            raise ValueError('Invalid snapshot path')
        if target.exists():
            if file_hash(target)!=expected:raise ValueError('Saved snapshot changed: '+name)
        else:
            if file_hash(original)!=expected:raise ValueError('Source changed before snapshot: '+name)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(original.read_bytes())


def review_sheet(destination,record):
    target=destination/'manual-review-template.csv'
    if target.exists():return
    fields=['case_id','model_run','method','domain','source_index','family','question','prompt',
        'reference_answer','generation','fact_recovered_yes_no_unclear','reviewer','reason']
    temporary=target.with_suffix('.csv.tmp')
    with temporary.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in record['rows']:
            writer.writerow(dict(case_id=row['id'],model_run=record['provenance']['model_run'],
                method=record['provenance']['method'],**{k:row[k] for k in ('domain','source_index','family','question','prompt','reference_answer','generation')}))
    temporary.replace(target)


def evaluate(args):
    valid_name(args.run_id);valid_name(args.model_run)
    suite=validate_suite(read_json(FREEZE/'suite.json'))
    rows=select_rows(suite,args.limit)
    model_directory=ROOT/'study/runs'/args.model_run
    model_run=read_json(model_directory/'run.json')
    verified=read_json(model_directory/'cpu_verification.json')
    if model_run['spec']['probe'] or verified['status']!='passed' or verified['counts']!=COUNTS:
        raise ValueError('Use a completely evaluated and verified formal checkpoint')
    checkpoint=ROOT/'artifacts/model' if model_run['spec']['method']=='baseline' else model_directory/'model'
    checkpoint_hash=file_hash(checkpoint/'model.safetensors')
    if checkpoint_hash!=verified['checkpoint_sha256']:raise ValueError('Model checkpoint hash mismatch')
    code=source_identity()
    sources=code['source_sha256']
    for name in ('evaluate_robustness_cuda.py','robustness_protocol.py','build_robustness_suite.py'):
        sources[name]=file_hash(ROOT/name)
    provenance=dict(model_run=args.model_run,method=model_run['spec']['method'],model_seed=model_run['spec']['seed'],
        model_run_signature=model_run['signature'],checkpoint_sha256=checkpoint_hash,
        suite_sha256=file_hash(FREEZE/'suite.json'),runtime_commit=code['commit'],runtime_sources=sources,
        question_limit_per_domain=args.limit,batch_size=8,max_new_tokens=200,seed=0,prompt_date='10 Apr 2025',
        decoding=dict(do_sample=False,top_p=None,temperature=None,use_cache=True),
        scope='pilot' if args.limit else 'full_320_prompt_stress_test')
    signature=digest(provenance)
    destination=ROOT/'study/robustness/cuda-runs'/args.run_id
    output=destination/'evaluation.json'
    manifest=destination/'run.json'
    if manifest.exists():
        if read_json(manifest)['signature']!=signature:raise ValueError('Changed code/settings; use a fresh run ID')
    else:
        if destination.exists() and any(destination.iterdir()):raise ValueError('Do not adopt an unmarked output directory')
        write_json(manifest,dict(signature=signature,provenance=provenance,working_tree=code['working_tree']))
    snapshot(destination,sources)
    record=read_json(output) if output.exists() else dict(status='running',signature=signature,provenance=provenance,rows=[],batch_seconds=[])
    if record['signature']!=signature:raise ValueError('Saved output signature mismatch')
    validate_prefix(record['rows'],rows,8)
    if len(record['rows'])==len(rows):
        if record['status']!='completed':raise ValueError('All rows present but completion marker missing')
        review_sheet(destination,record)
        print('Verified saved stress outputs; no fresh inference claimed',args.run_id,flush=True)
        return
    preflight(destination,training=False)
    import torch
    from omegaconf import OmegaConf
    from rouge_score.rouge_scorer import RougeScorer
    from cuda_runtime import configure,load_cuda
    from full_core import BASE,official_item
    from data.collators import DataCollatorForSupervisedDataset
    from evals.metrics.utils import eval_text_similarity
    configure(0)
    model,tokenizer,template=load_cuda(checkpoint)
    model.eval()
    model.generation_config.eos_token_id=read_json(BASE/'generation_config.json')['eos_token_id']
    collator=DataCollatorForSupervisedDataset(tokenizer,padding_side='left')
    scorer=RougeScorer(['rougeL'],use_stemmer=True)
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    previous=record.get('elapsed_seconds',0.)
    for offset in range(len(record['rows']),len(rows),8):
        chunk=rows[offset:offset+8]
        entries=[official_item(tokenizer,template,row['prompt'],'',True) for row in chunk]
        begin=time.monotonic()
        with torch.inference_mode():
            generations=eval_text_similarity(model,tokenizer,collator(entries),OmegaConf.create(dict(
                do_sample=False,top_p=None,temperature=None,max_new_tokens=200,use_cache=True)))
        for row,result in zip(chunk,generations):
            if result['input']!=expected_input(row):raise ValueError('Processed input differs from the frozen answer-free prompt')
            record['rows'].append(dict(**row,generation=result['generation'],processed_input=result['input'],
                rougeL_recall_proxy=scorer.score(row['reference_answer'],result['generation'])['rougeL'].recall,
                manual_fact_recovered=None))
        record['batch_seconds'].append(time.monotonic()-begin)
        record.update(status='completed' if len(record['rows'])==len(rows) else 'running',
            elapsed_seconds=previous+time.monotonic()-started,
            cuda_peak_allocated_gib=max(record.get('cuda_peak_allocated_gib',0.),torch.cuda.max_memory_allocated()/2**30))
        write_json(output,record)
        print(dict(run_id=args.run_id,completed=len(record['rows']),total=len(rows)),flush=True)
    review_sheet(destination,record)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-run',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    if not 0<=args.limit<=40:parser.error('limit must be 0..40; zero means all 320 prompts')
    with gpu_lock():
        try:evaluate(args)
        except BaseException as error:
            valid_name(args.run_id)
            write_json(ROOT/'study/robustness/cuda-failures'/(args.run_id+'-'+uuid.uuid4().hex[:8]+'.json'),
                dict(status='failed',error=str(error),traceback=traceback.format_exc()))
            raise


if __name__=='__main__':main()
