"""Execute the complete five-weight/three-seed matrix serially and preserve evidence.

Plan is immutable once created. New training never overwrites interrupted runs.
Prompt outputs use a separate, common-runtime series; earlier annotation stays intact.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from cuda_common import ROOT, COUNTS, read_json, write_json, file_hash, source_identity, digest

OUT=ROOT/'study/phase3/grid-20261029'
PLAN=OUT/'plan.json'
STATE=OUT/'execution.json'
ALPHAS=[.25,.5,1.,2.,4.]
SEEDS=[0,1,2]
CACHE=ROOT/'study/phase3/npo-validation-20261029/v2/reference-full-v2.json'
CACHE_SHA='7e31f3a4fe0282b1a5c79339169572c782b8e5eb5872b55856791ac5763c5033'


def now():return datetime.now(timezone.utc).isoformat()


def sources():
    values=source_identity(include_simnpo=True,include_npo=True)['source_sha256']
    for name in ('evaluate_robustness_cuda.py','robustness_protocol.py','build_robustness_suite.py',
                 'verify_cuda_run.py','verify_robustness_cuda.py'):
        values[name]=file_hash(ROOT/name)
    return values


def configuration(method,seed,alpha=1.):
    value=dict(method=method,seed=seed,alpha=alpha,gamma=.125 if method=='SimNPO' else 1.)
    if method=='SimNPO':value.update(beta=4.5,delta=0.)
    if method=='NPO':value['beta']=.1
    return value


def existing(config):
    found=[]
    for path in (ROOT/'study/runs').glob('*/run.json'):
        record=read_json(path);spec=record['spec']
        if spec['probe'] or any(spec.get(key)!=value for key,value in config.items()):continue
        verified=path.parent/'cpu_verification.json'
        if verified.exists() and read_json(verified)['status']=='passed':found.append(spec['run_id'])
    if len(found)>1:raise RuntimeError('Ambiguous verified checkpoint for '+str(config))
    return found[0] if found else None


def make_plan():
    OUT.mkdir(parents=True,exist_ok=True)
    if PLAN.exists():return read_json(PLAN)
    matrix=[]
    configurations=[configuration('baseline',0)]+[configuration('GradAscent',s) for s in SEEDS]
    configurations += [configuration(m,s,a) for m in ('GradDiff','SimNPO','NPO') for a in ALPHAS for s in SEEDS]
    for cfg in configurations:
        reused=existing(cfg)
        prefix={'baseline':'baseline','GradAscent':'ga','GradDiff':'gd','SimNPO':'simnpo','NPO':'npo'}[cfg['method']]
        alpha=str(cfg['alpha']).rstrip('0').rstrip('.').replace('.','p')
        part='-a'+alpha if cfg['method']=='GradDiff' or cfg['alpha']!=1 else ''
        run_id=reused or f"{prefix}{part}-s{cfg['seed']}-cuda-v1"
        matrix.append(dict(**cfg,run_id=run_id,reuse_existing=bool(reused),
            prompt_run_id='grid26-'+run_id.removesuffix('-cuda-v1')+'-v1'))
    assert len(matrix)==49 and sum(r['reuse_existing'] for r in matrix)==11
    assert len({r['run_id'] for r in matrix})==49
    # Complete the newly added default methods' three seeds first.
    matrix.sort(key=lambda r:(0 if r['method'] in ('SimNPO','NPO') and r['alpha']==1 and not r['reuse_existing'] else 1,
        r['seed'],r['method'],r['alpha']))
    assert CACHE.is_file() and file_hash(CACHE)==CACHE_SHA
    data=dict(status='frozen_execution_plan',created_at=now(),alphas=ALPHAS,seeds=SEEDS,matrix=matrix,
        full_evaluation_settings=49,formal_training_settings=48,existing_formal_training_settings=10,
        new_formal_training_settings=38,full_prompt_generations=49*320,reference_cache_sha256=CACHE_SHA,
        source_sha256=sources(),controller_sha256=file_hash(Path(__file__)),
        protocol='All full parameters; ten updates / 320 visits each; 657 evaluation questions; same seed sampling; no LoRA.',
        sweep='Retain alpha only; gamma/beta/delta fixed per method. Full 5 x 3 grid for GD/SimNPO/NPO, GA three seeds, one fixed baseline.',
        prompt_policy='All 49 settings receive 320 fresh common-runtime outputs; earlier 1600-answer series and human files preserved.',
        human_annotation_policy='Real human labels and disagreements remain separate; no automated factual-recovery rate.')
    data['signature']=digest(data)
    write_json(PLAN,data);return data


def check_sources(plan):
    if sources()!=plan['source_sha256'] or file_hash(Path(__file__))!=plan['controller_sha256']:
        raise RuntimeError('Frozen execution code changed; preserve this matrix and inspect before proceeding')
    if file_hash(CACHE)!=plan['reference_cache_sha256']:raise RuntimeError('Fixed reference cache changed')


def resource_gate(state):
    import psutil
    deadline=time.monotonic()+600
    while psutil.virtual_memory().available<3.5*2**30:
        state.update(status='waiting_for_ram',available_ram_gib=psutil.virtual_memory().available/2**30,updated_at=now())
        write_json(STATE,state)
        if time.monotonic()>=deadline:raise RuntimeError('RAM gate did not recover in ten minutes; no processes were closed')
        time.sleep(30)
    if shutil.disk_usage(ROOT).free<28*2**30:raise RuntimeError('Disk reserve below 28 GiB; stop before claiming another run')


def command(args,log_path,state,stage):
    state.update(status='running',current_stage=stage,updated_at=now());write_json(STATE,state)
    print(json.dumps(dict(event='start',run_id=state['current_run'],stage=stage,at=now())),flush=True)
    with log_path.open('a',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,'-X','utf8','-u',*args],cwd=ROOT,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        try:
            for line in process.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            code=process.wait()
        except BaseException:
            if process.poll() is None:process.terminate();process.wait(timeout=30)
            raise
    if code:raise RuntimeError(f'{args[0]} failed ({code}); preserve {log_path}')


def check_formal(row):
    directory=ROOT/'study/runs'/row['run_id']
    spec=read_json(directory/'run.json')['spec'];verified=read_json(directory/'cpu_verification.json')
    assert all(spec.get(k)==row[k] for k in ('method','seed','alpha','gamma'))
    for k in ('beta','delta'):
        if k in row:assert spec[k]==row[k]
    assert verified['status']=='passed' and verified['counts']==COUNTS
    checkpoint=ROOT/'artifacts/model/model.safetensors' if row['method']=='baseline' else directory/'model/model.safetensors'
    assert file_hash(checkpoint)==verified['checkpoint_sha256']
    if row['method']!='baseline':
        training=read_json(directory/'training.json')
        assert training['status']=='completed_training_reload_checked'
        assert (training['steps'],training['visits'],training['trainable_parameters'])==(10,320,1235814400)
        canonical=read_json(ROOT/f"study/runs/ga-s{row['seed']}-cuda-v1/training.json")
        assert training['sampling_trace']==canonical['sampling_trace']
    return verified


def execute(plan):
    if STATE.exists():
        state=read_json(STATE)
        if state['plan_signature']!=plan['signature']:raise RuntimeError('Changed plan')
        if state['status']=='completed_verified':print('Already verified; no new training');return
    else:
        state=dict(status='running',started_at=now(),plan_signature=plan['signature'],runs={},
            current_run=None,current_stage=None,formal_verified=0,prompt_verified=0)
    write_json(STATE,state)
    try:
        for row in plan['matrix']:
            check_sources(plan)
            run_id=row['run_id'];directory=ROOT/'study/runs'/run_id
            entry=state['runs'].setdefault(run_id,dict(status='pending',reuse_existing=row['reuse_existing']))
            state['current_run']=run_id
            if entry['status']=='passed':continue
            if not (directory/'cpu_verification.json').exists():
                config=['--method',row['method'],'--seed',str(row['seed']),'--alpha',str(row['alpha']),
                        '--gamma',str(row['gamma']),'--run-id',run_id]
                for key in ('beta','delta'):
                    if key in row:config.extend(['--'+key,str(row[key])])
                if row['method']=='NPO':config.extend(['--reference-cache',str(CACHE)])
                training=directory/'training.json'
                if not training.exists():
                    if (directory/'training_started.json').exists():raise RuntimeError('Interrupted training must not be silently retried: '+run_id)
                    resource_gate(state)
                    command(['train_cuda.py',*config],OUT/(run_id+'-train.log'),state,'training_and_reload')
                elif read_json(training)['status']!='completed_training_reload_checked':
                    raise RuntimeError('Training/reload incomplete; inspect preserved evidence: '+run_id)
                command(['evaluate_cuda.py',*config],OUT/(run_id+'-eval.log'),state,'full_657_evaluation')
                command(['verify_cuda_run.py',run_id],OUT/(run_id+'-verify.log'),state,'independent_cpu_verification')
            check_formal(row);entry['formal_verified']=True
            state['formal_verified']=sum(v.get('formal_verified',False) for v in state['runs'].values())
            prompt_id=row['prompt_run_id'];prompt=ROOT/'study/robustness/cuda-runs'/prompt_id
            if not (prompt/'cpu_verification.json').exists():
                command(['evaluate_robustness_cuda.py','--model-run',run_id,'--run-id',prompt_id],OUT/(prompt_id+'.log'),state,'full_320_prompt_generation')
                command(['verify_robustness_cuda.py',prompt_id],OUT/(prompt_id+'.log'),state,'prompt_cpu_verification')
            verified=read_json(prompt/'cpu_verification.json')
            assert verified['status']=='passed' and verified['total_generations']==320 and verified['checkpoint_bytes_verified']
            entry.update(status='passed',prompt_verified=True,finished_at=now())
            state['prompt_verified']=sum(v.get('prompt_verified',False) for v in state['runs'].values())
            write_json(STATE,state)
            print(json.dumps(dict(event='setting_verified',run_id=run_id,finished=state['prompt_verified'],total=49)),flush=True)
        assert state['formal_verified']==state['prompt_verified']==49
        state.update(status='completed_verified',current_run=None,current_stage=None,finished_at=now())
        write_json(STATE,state)
        print('COMPLETE_49_SETTING_GRID_VERIFIED',flush=True)
    except BaseException as exc:
        state.update(status='stopped_on_error',error=str(exc),updated_at=now());write_json(STATE,state);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();plan=make_plan()
    print(json.dumps({key:plan[key] for key in ('full_evaluation_settings','new_formal_training_settings','full_prompt_generations','signature')},indent=2),flush=True)
    if args.execute:
        import contextlib
        # A process-lifetime lock prevents two controllers from sharing this queue.
        with (OUT/'controller.lock').open('a+b') as lock:
            lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
            if sys.platform=='win32':
                import msvcrt
                msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            execute(plan)


if __name__=='__main__':main()
