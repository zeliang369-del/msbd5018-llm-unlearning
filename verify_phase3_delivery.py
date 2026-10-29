"""Audit packaged handoffs and saved report integrity without loading a model."""
from collections import Counter
import csv
import io
import json
from pathlib import Path, PureWindowsPath
import subprocess
import sys
from contextlib import redirect_stdout
import zipfile
from cuda_common import ROOT, read_json, write_json, file_hash
from phase3_progress import progress


def main():
    with redirect_stdout(io.StringIO()):current=progress()
    assert current['status']=='experimental_execution_completed_verified'
    folder=ROOT/'study/phase3/robustness-20261029'
    packages=[]
    for receipt_name,total,forget_count,per_slot in [('primary-handoff.json',960,480,240),
                                                    ('supplementary-handoff.json',640,320,160)]:
        receipt=read_json(folder/receipt_name)
        archive=folder/PureWindowsPath(receipt['archive']).name
        assert file_hash(archive)==receipt['archive_sha256']
        with zipfile.ZipFile(archive) as zipped:
            assert zipped.testzip() is None
            manifest=json.loads(zipped.read('manifest.json'))
            for name,expected in manifest['files_sha256'].items():
                import hashlib
                assert hashlib.sha256(zipped.read(name)).hexdigest()==expected

            def rows(name):
                return list(csv.DictReader(io.StringIO(zipped.read(name).decode('utf-8-sig'))))

            all_rows=rows(f'all-{total}-responses-blinded.csv')
            assert len(all_rows)==total and len({r['case_uid'] for r in all_rows})==total
            by_id={r['case_uid']:r for r in all_rows}
            assert sum(r['domain']=='forget' for r in all_rows)==forget_count
            assert all(r['fact_recovered_yes_no_unclear']==r['reviewer']==r['reason']=='' for r in all_rows)
            judgments=[]
            for slot in 'ABCD':
                selected=rows('reviewer-slot-'+slot+'.csv')
                assert len(selected)==per_slot
                for r in selected:
                    assert r['domain']=='forget'
                    assert all(r[key]==by_id[r['case_uid']][key] for key in (
                        'model_code','domain','source_index','family','question','prompt','reference_answer','generation'))
                    assert r['fact_recovered_yes_no_unclear']==r['reviewer']==r['reason']==''
                    judgments.append(r['case_uid'])
            counts=Counter(judgments)
            assert len(counts)==forget_count and set(counts.values())=={2}
        packages.append(dict(archive=archive.name,sha256=receipt['archive_sha256'],
            responses=total,forget_responses=forget_count,independent_forget_judgments=len(judgments)))
    frozen_check=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'verify_results.py')],
        cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    assert frozen_check.returncode==0,frozen_check.stderr
    report=ROOT/'study/phase3/phase3-final-20261029'
    result=dict(status='passed',phase3_experiment_acceptance=current['status'],
        packages=packages,total_full_responses=1600,human_labels_filled=0,
        combined_tests=current['combined_code_integration']['counts'],
        frozen_record_check=dict(exit_code=frozen_check.returncode,stdout=frozen_check.stdout.strip()),
        verifier_sha256=file_hash(Path(__file__)),
        report_manifest_sha256=file_hash(report/'report-manifest.json'),
        scope='Saved record integrity, report byte hashes, zip member hashes, actual CSV counts and independent blank review assignments; no model inference or human factual judgments.')
    write_json(report/'delivery-audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
