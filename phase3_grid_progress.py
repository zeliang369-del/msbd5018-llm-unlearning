"""Report the full-grid acceptance separately from the preserved first-round record."""
import json
from pathlib import Path
from cuda_common import ROOT, COUNTS, read_json, write_json, file_hash

OUT=ROOT/'study/phase3/grid-20261029'


def read_if(path):return read_json(path) if path.is_file() else None


def progress(quiet=False):
    plan=read_json(OUT/'plan.json');execution=read_if(OUT/'execution.json')
    rows=[]
    for item in plan['matrix']:
        directory=ROOT/'study/runs'/item['run_id']
        verified=read_if(directory/'cpu_verification.json')
        formal=bool(verified and verified['status']=='passed' and verified['counts']==COUNTS)
        prompt=read_if(ROOT/'study/robustness/cuda-runs'/item['prompt_run_id']/'cpu_verification.json')
        prompted=bool(prompt and prompt['status']=='passed' and prompt['total_generations']==320 and prompt['checkpoint_bytes_verified'])
        rows.append(dict(method=item['method'],seed=item['seed'],alpha=item['alpha'],run_id=item['run_id'],
            reused_first_round=item['reuse_existing'],formal_report_passed=formal,prompt_report_passed=prompted))
    report=read_if(OUT/'report-manifest.json');delivery=read_if(OUT/'delivery-audit.json')
    visual=read_if(OUT/'visual-review.json')
    complete=bool(execution and execution['status']=='completed_verified' and all(r['formal_report_passed'] and r['prompt_report_passed'] for r in rows))
    report_ok=visual_ok=integrity_ok=False
    if report:
        from verify_phase3_grid_delivery import report_intact,visual_intact,audit
        report_ok=report_intact();visual_ok=visual_intact() if report_ok else False
        if report_ok and delivery and delivery['status']=='passed':
            integrity=read_if(OUT/'integrity-audit.json')
            report_hash=file_hash(OUT/'report-manifest.json')
            if not integrity or integrity.get('manifest_sha256')!=report_hash or integrity.get('verifier_sha256')!=file_hash(ROOT/'verify_phase3_grid_delivery.py'):
                integrity=audit()
            integrity_ok=integrity['status']=='passed'
    complete=complete and report_ok and bool(delivery and delivery['status']=='passed') and visual_ok and integrity_ok
    record=dict(status='full_grid_experiments_completed_verified' if complete else 'full_grid_in_progress',
        plan_signature=plan['signature'],formal_evaluation_target=49,formal_reports_passed=sum(r['formal_report_passed'] for r in rows),
        new_formal_training_target=38,new_formal_reports_passed=sum(r['formal_report_passed'] and not r['reused_first_round'] for r in rows),
        prompt_settings_target=49,prompt_reports_passed=sum(r['prompt_report_passed'] for r in rows),
        new_prompt_generation_target=15680,current_run=execution.get('current_run') if execution else None,
        current_stage=execution.get('current_stage') if execution else None,controller_status=execution.get('status') if execution else 'not_started',
        final_report_ready=report_ok,
        delivery_audit_passed=bool(delivery and delivery['status']=='passed'),
        visual_review_passed=visual_ok,
        independent_delivery_integrity_passed=integrity_ok,
        human_labels='Await actual independent annotations; no automatic factual labels claimed.',rows=rows)
    write_json(OUT/'progress.json',record)
    if not quiet:print(json.dumps({k:v for k,v in record.items() if k!='rows'},ensure_ascii=False,indent=2))
    return record


if __name__=='__main__':progress()
