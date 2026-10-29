"""Show phase-three acceptance from saved evidence, without model work."""
import json
from pathlib import Path
from pathlib import PureWindowsPath
from cuda_common import ROOT,read_json,write_json,file_hash


def status(path):
    return read_json(path) if path.exists() else None


def formal(run_id):
    directory=ROOT/'study/runs'/run_id
    report=status(directory/'cpu_verification.json')
    return bool(report and report['status']=='passed' and report['counts']==dict(forget=40,retain=400,real_authors=100,world_facts=117))


def prompt(run_id,expected):
    report=status(ROOT/'study/robustness/cuda-runs'/run_id/'cpu_verification.json')
    return bool(report and report['status']=='passed' and report['total_generations']==expected and report['checkpoint_bytes_verified'])


def report_ready():
    directory=ROOT/'study/phase3/phase3-final-20261029'
    report=status(directory/'report-manifest.json')
    if not report or report.get('status')!='verified_phase3_experiments':return False
    if (report.get('formal_settings'),report.get('full_prompt_generations'),report.get('pilot_generations'))!=(5,1600,120):return False
    if file_hash(ROOT/'report_phase3_cuda.py')!=report['renderer_sha256']:return False
    for name,expected in report['input_sha256'].items():
        path=ROOT/name
        if not path.is_file() or file_hash(path)!=expected:return False
    for name,expected in report['generated_sha256'].items():
        path=directory/name
        if not path.is_file() or file_hash(path)!=expected:return False
    return True


def handoff_ready(name):
    receipt=status(ROOT/'study/phase3/robustness-20261029'/name)
    if not receipt:return False
    archive=ROOT/'study/phase3/robustness-20261029'/PureWindowsPath(receipt['archive']).name
    return bool(receipt and receipt['status']=='ready_for_human_annotation_not_annotated'
                and archive.is_file() and file_hash(archive)==receipt['archive_sha256'])


def integration_ready(record):
    if not record or record['status']!='passed':return False
    return all((ROOT/name).is_file() and file_hash(ROOT/name)==expected
               for name,expected in record['source_sha256'].items())


def progress():
    primary=['robust-baseline-s0-v1','robust-ga-s0-v1','robust-gd-a1-s0-v1']
    supplemental=['robust-simnpo-s0-v1','robust-npo-s0-v1']
    npo_probe=status(ROOT/'study/runs/npo-s0-cuda-probe-v1/cpu_verification.json')
    record=dict(date_local='2026-10-29',
        simnpo_formal_completed=formal('simnpo-s0-cuda-v1'),npo_probe_completed=bool(npo_probe and npo_probe['status']=='passed'),
        npo_formal_completed=formal('npo-s0-cuda-v1'),
        primary_prompt_runs={run_id:prompt(run_id,320) for run_id in primary},
        supplementary_prompt_runs={run_id:prompt(run_id,320) for run_id in supplemental},
        primary_handoff=status(ROOT/'study/phase3/robustness-20261029/primary-handoff.json'),
        supplementary_handoff=status(ROOT/'study/phase3/robustness-20261029/supplementary-handoff.json'),
        final_report_ready=report_ready(),
        combined_code_integration=status(ROOT/'study/phase3/code-integration-20261029/verification.json'),
        human_review='Blank independent review templates prepared; submitted human labels and disagreements are tracked separately in the following verification stage.',
        optional_rmu='Not required for phase-three acceptance; not executed.')
    done=record['simnpo_formal_completed'] and record['npo_probe_completed'] and record['npo_formal_completed']
    done=done and all(record['primary_prompt_runs'].values()) and all(record['supplementary_prompt_runs'].values())
    done=done and handoff_ready('primary-handoff.json') and handoff_ready('supplementary-handoff.json') and record['final_report_ready']
    integrated=record['combined_code_integration']
    done=done and integration_ready(integrated)
    record['status']='experimental_execution_completed_verified' if done else 'in_progress'
    write_json(ROOT/'study/phase3/progress.json',record)
    print(json.dumps(record,ensure_ascii=False,indent=2))
    return record


if __name__=='__main__':progress()
