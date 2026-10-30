"""Independent standard-library audit of the complete grid's saved delivery.

The experiment audit checks actual checkpoint bytes. This portable audit checks
the saved evidence and report/package bytes without downloading weights.
"""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import io
import math
from pathlib import Path
from statistics import mean,stdev
import zipfile
import json
from cuda_common import ROOT,COUNTS,read_json,file_hash,digest,write_json

OUT=ROOT/'study/phase3/grid-20261029'
FIGURES=['weight-seed-summary.png','all-settings.png','utility-component-grid.png','prompt-proxy-grid.png']


def csv_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-12),(a,b)


def report_intact():
    path=OUT/'report-manifest.json'
    if not path.exists():return False
    manifest=read_json(path)
    if manifest.get('status')!='verified_full_grid_report':return False
    if (manifest.get('settings'),manifest.get('full_prompt_generations'))!=(49,15680):return False
    if manifest['renderer_sha256']!=file_hash(ROOT/'report_phase3_grid.py'):return False
    for name,expected in manifest['input_sha256'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or file_hash(path)!=expected:return False
    for name,expected in manifest['generated_sha256'].items():
        path=(OUT/name).resolve()
        if not path.is_relative_to(OUT.resolve()) or not path.is_file() or file_hash(path)!=expected:return False
    return True


def visual_intact():
    path=OUT/'visual-review.json'
    if not path.exists():return False
    visual=read_json(path)
    if visual.get('status')!='passed' or visual.get('report_manifest_sha256')!=file_hash(OUT/'report-manifest.json'):return False
    return all(visual.get('files_sha256',{}).get(name)==file_hash(OUT/name) for name in FIGURES)


def audit(require_visual=False):
    state=read_json(OUT/'execution.json')
    if state['status']!='completed_verified':raise RuntimeError('Grid is still running or stopped; no completed delivery is claimed')
    plan=read_json(OUT/'plan.json');signed=dict(plan);signature=signed.pop('signature')
    assert digest(signed)==signature==state['plan_signature']
    expected={('baseline',0,1.),*(('GradAscent',s,1.) for s in (0,1,2)),
              *((m,s,a) for m in ('GradDiff','SimNPO','NPO') for s in (0,1,2) for a in (.25,.5,1.,2.,4.))}
    actual={(r['method'],r['seed'],r['alpha']) for r in plan['matrix']}
    assert actual==expected and len(plan['matrix'])==49
    assert report_intact(),'Report input/output/renderer hashes differ'
    independent=read_json(OUT/'independent-audit.json')
    assert independent['status']=='passed' and independent['actual_checkpoint_bytes_rechecked']
    assert (independent['formal_settings'],independent['prompt_settings'],len(independent['checks']))==(49,49,49)
    by_run={r['run_id']:r for r in independent['checks']}
    results=csv_rows(OUT/'all-49-settings.csv')
    assert len(results)==49
    groups=defaultdict(list);prompt_records={};prompt_metrics={}
    result_lookup={(r['method'],int(r['seed']),float(r['alpha'])):r for r in results}
    assert set(result_lookup)==expected
    for item in plan['matrix']:
        directory=ROOT/'study/runs'/item['run_id'];check=by_run[item['run_id']]
        path=directory/'cpu_verification.json';assert file_hash(path)==check['formal_sha256']
        verification=read_json(path);run=read_json(directory/'run.json')
        assert verification['status']=='passed' and verification['counts']==COUNTS
        assert digest(dict(spec=run['spec'],commit=run['code']['commit'],sources=run['code']['source_sha256']))==run['signature']
        for name,expected_hash in run['code']['source_sha256'].items():
            snapshot=(directory/'code_snapshot'/name).resolve()
            assert snapshot.is_relative_to((directory/'code_snapshot').resolve())
            assert file_hash(snapshot)==expected_hash
        row=result_lookup[item['method'],item['seed'],item['alpha']]
        assert row['run_id']==item['run_id'] and row['checkpoint_sha256']==verification['checkpoint_sha256']
        metrics=verification['metrics_recomputed']
        close(row['fq'],metrics['forget_quality']);close(row['mu'],metrics['model_utility'])
        for key,value in metrics['utility_components'].items():close(row[key],value)
        evaluation=read_json(directory/'evaluation/evaluation.json')
        close(row['forget_rouge'],evaluation['metrics']['forget_Q_A_ROUGE'])
        groups[item['method'],item['alpha']].append(row)
        prompt=ROOT/'study/robustness/cuda-runs'/item['prompt_run_id']
        assert check['prompt_id']==item['prompt_run_id'] and file_hash(prompt/'cpu_verification.json')==check['prompt_verification_sha256']
        pcheck=read_json(prompt/'cpu_verification.json')
        assert pcheck['status']=='passed' and pcheck['total_generations']==320
        record=read_json(prompt/'evaluation.json')
        assert record['status']=='completed' and len(record['rows'])==320
        prompt_records[item['prompt_run_id']]=record
        prompt_metrics[item['method'],item['seed'],item['alpha']]=pcheck['proxies']
    summary=csv_rows(OUT/'three-seed-summary.csv');assert len(summary)==17
    assert {(r['method'],float(r['alpha'])) for r in summary}==set(groups)
    numeric=['fq','mu','forget_rouge']+list(metrics['utility_components'])
    for row in summary:
        values=groups[row['method'],float(row['alpha'])];assert int(row['n'])==len(values)
        for key in numeric:
            scores=[float(r[key]) for r in values];close(row[key+'_mean'],mean(scores))
            if len(scores)==3:close(row[key+'_sample_sd'],stdev(scores))
            else:assert row[key+'_sample_sd']==''
    proxies=csv_rows(OUT/'prompt-proxy-summary.csv');paired=csv_rows(OUT/'paired-prompt-proxies.csv')
    assert len(proxies)==392 and len(paired)==11760
    assert len({(r['method'],r['seed'],r['alpha'],r['domain'],r['family']) for r in proxies})==392
    assert len({(r['method'],r['seed'],r['alpha'],r['domain'],r['source_index'],r['family']) for r in paired})==11760
    for row in paired:close(row['difference'],float(row['changed_prompt_proxy'])-float(row['direct_proxy']))
    for row in proxies:
        close(row['mean_rougeL_recall_proxy'],prompt_metrics[row['method'],int(row['seed']),float(row['alpha'])][row['domain']+'/'+row['family']])
    delivery=read_json(OUT/'delivery-audit.json');package=delivery['review_package']
    archive=(OUT/package['archive']).resolve();assert archive.is_relative_to(OUT.resolve())
    assert file_hash(archive)==package['archive_sha256']
    assert delivery['report_manifest_sha256']==file_hash(OUT/'report-manifest.json')
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        manifest=json.loads(zipped.read('manifest.json'))
        assert manifest['total_responses']==15680 and manifest['human_labels_filled']==0
        for name,expected_hash in manifest['files_sha256'].items():
            assert hashlib.sha256(zipped.read(name)).hexdigest()==expected_hash
        def rows(name):return list(csv.DictReader(io.StringIO(zipped.read(name).decode('utf-8-sig'))))
        all_rows=rows('all49-15680-responses-blinded.csv');core=rows('core13-4160-responses-blinded.csv')
        assert len(all_rows)==15680 and len(core)==4160
        master={r['case_uid']:r for r in all_rows};assert len(master)==15680
        mapping=json.loads(zipped.read('analyst-only/unblinding-key.json'))
        assert len(mapping)==49 and {v['prompt_run'] for v in mapping.values()}==set(prompt_records)
        package_sources={}
        for code,identity in mapping.items():
            record=prompt_records[identity['prompt_run']]
            assert record['provenance']['model_run']==identity['model_run']
            for row in record['rows']:
                uid=hashlib.sha256((identity['prompt_run']+'|'+row['id']).encode()).hexdigest()[:20]
                package_sources[uid]=(code,row)
        assert set(package_sources)==set(master)
        assert sum(r['domain']=='forget' for r in all_rows)==7840
        for row in all_rows:
            assert row['fact_recovered_yes_no_unclear']==row['reviewer']==row['reason']==''
            code,original=package_sources[row['case_uid']]
            assert row['model_code']==code
            assert all(row[key]==str(original[key]) for key in ('domain','source_index','family','question','prompt','reference_answer','generation'))
        for scope,number,case_count in [('all49',3920,7840),('core13',1040,2080)]:
            ids=[]
            for slot in 'ABCD':
                selected=rows(scope+'-reviewer-slot-'+slot+'.csv');assert len(selected)==number
                for row in selected:
                    assert row['domain']=='forget'
                    assert all(row[key]==master[row['case_uid']][key] for key in ('model_code','source_index','family','generation','reference_answer','prompt'))
                    assert row['fact_recovered_yes_no_unclear']==row['reviewer']==row['reason']==''
                    ids.append(row['case_uid'])
            counts=Counter(ids);assert len(counts)==case_count and set(counts.values())=={2}
    visual=visual_intact()
    if require_visual:assert visual,'Real figure inspection or its byte binding is missing'
    result=dict(status='passed',formal_settings=49,prompt_outputs=15680,summary_groups=17,
        paired_prompt_changes=11760,package_rows=15680,human_labels_filled=0,visual_review_passed=visual,
        manifest_sha256=file_hash(OUT/'report-manifest.json'),verifier_sha256=file_hash(Path(__file__)),
        scope='Saved records, source snapshots, report arithmetic, package bytes/rows and independent assignments; actual weights were checked by the independent experiment audit.')
    write_json(OUT/'integrity-audit.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--require-visual',action='store_true')
    args=parser.parse_args();print(json.dumps(audit(args.require_visual),indent=2))


if __name__=='__main__':main()
