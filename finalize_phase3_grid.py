"""After serial GPU execution, independently audit and prepare the full-grid handoff.

The process may wait for the active queue. It never starts GPU work, publishes,
invents human labels, or marks final visual review complete.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime,timezone
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile
from cuda_common import ROOT,read_json,write_json,file_hash

OUT=ROOT/'study/phase3/grid-20261029'
FIELDS=['case_uid','model_code','domain','source_index','family','question','prompt','reference_answer',
        'generation','review_pass','fact_recovered_yes_no_unclear','reviewer','reason']


@contextmanager
def prevent_idle_sleep():
    if sys.platform=='win32':
        import ctypes
        if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):raise OSError('Cannot request temporary idle-sleep prevention')
    try:yield
    finally:
        if sys.platform=='win32':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def package(plan):
    folder=OUT/'annotation-grid';archive=OUT/'annotation-grid-20261029.zip'
    if folder.exists() or archive.exists():raise RuntimeError('Preserve existing human work; do not replace the package')
    values=[];mapping={};codes=list(range(1,50));random.Random(20261029).shuffle(codes)
    for item,number in zip(plan['matrix'],codes):
        record=read_json(ROOT/'study/robustness/cuda-runs'/item['prompt_run_id']/'evaluation.json')
        code=f'G{number:02d}'
        mapping[code]=dict(method=item['method'],seed=item['seed'],alpha=item['alpha'],
            model_run=item['run_id'],prompt_run=item['prompt_run_id'],checkpoint_sha256=record['provenance']['checkpoint_sha256'])
        for row in record['rows']:
            assert row['manual_fact_recovered'] is None
            uid=hashlib.sha256((item['prompt_run_id']+'|'+row['id']).encode()).hexdigest()[:20]
            values.append(dict(case_uid=uid,model_code=code,core13=item['alpha']==1,
                **{key:row[key] for key in ('domain','source_index','family','question','prompt','reference_answer','generation')}))
    assert len(values)==15680 and len({r['case_uid'] for r in values})==15680
    assert sum(r['core13'] for r in values)==4160
    folder.mkdir()

    def save(name,rows):
        with (folder/name).open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=FIELDS,extrasaction='ignore');writer.writeheader();writer.writerows(rows)

    save('all49-15680-responses-blinded.csv',values)
    save('core13-4160-responses-blinded.csv',[r for r in values if r['core13']])
    rotation={'A':[(0,10,'first'),(30,40,'independent')],'B':[(0,10,'independent'),(10,20,'first')],
              'C':[(10,20,'independent'),(20,30,'first')],'D':[(20,30,'independent'),(30,40,'first')]}
    for scope,selected,expected in [('all49',values,3920),('core13',[r for r in values if r['core13']],1040)]:
        judgments=[]
        for slot,blocks in rotation.items():
            assigned=[dict(r,review_pass=review_pass) for begin,end,review_pass in blocks for r in selected
                      if r['domain']=='forget' and begin<=r['source_index']<end]
            assert len(assigned)==expected
            random.Random('grid-'+scope+slot).shuffle(assigned)
            save(scope+'-reviewer-slot-'+slot+'.csv',assigned)
            judgments.extend(r['case_uid'] for r in assigned)
        counts=Counter(judgments)
        assert set(counts.values())=={2}
        assert len(counts)==(7840 if scope=='all49' else 2080)
    write_json(folder/'analyst-only/unblinding-key.json',mapping)
    (folder/'README.md').write_text('''# 完整权重／种子矩阵人工复核材料

49 组配置、每组 320 条新生成回答，共 15,680 条；含 7,840 条遗忘回答和 7,840 条保留对照。当前全部人工标签为空，槽位分配仅为建议，不表示成员认领或已经完成。模型代码隐藏方法、权重和种子，身份映射保存在 analyst-only；独立判断前尽量不查看。

all49 文件覆盖全部数据，四份 all49 reviewer 文件各 3,920 次遗忘判断，合计 15,680 次，每条遗忘回答恰有两位槽位。core13 文件覆盖固定起点和四种方法默认 alpha=1 下的三个种子，共 4,160 条回答；四份 core13 reviewer 文件各 1,040 次判断。两套包含重叠案例，不可把 core13 与 all49 的判断直接相加；按 case_uid 管理。

请组内明确人工评审范围和实际负责人，再统一边界、独立填写 yes/no/unclear、真实 reviewer 和 reason。自动 ROUGE 是文本重合代理，不能代替事实判断，也不能据此自动产生事实恢复率。缺失、unclear、排除数量和条件分母须单列，分母为零写 N/A。保留对照抽检须另行固定范围并如实报告，不能外推完整能力准确率。

已有首轮两个标注包保留不变。若要复用其中实际完成的标签，必须核对检查点、提示及实际回答完全一致，保留原判断和审阅者来源；本脚本不自动迁移或覆盖任何标签。TOFU 人物与作品为虚构基准内容。
''',encoding='utf-8',newline='\n')
    write_json(folder/'manifest.json',dict(status='ready_for_human_review_not_annotated',total_responses=15680,
        forget_responses=7840,retain_responses=7840,human_labels_filled=0,core13_responses=4160,
        files_sha256={str(p.relative_to(folder)).replace('\\','/'):file_hash(p) for p in folder.rglob('*') if p.is_file()}))
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(folder.rglob('*')):
            if path.is_file():zipped.write(path,path.relative_to(folder))
    with zipfile.ZipFile(archive) as zipped:assert zipped.testzip() is None
    return dict(archive=archive.name,archive_sha256=file_hash(archive),responses=15680,human_labels_filled=0)


def finalize():
    plan=read_json(OUT/'plan.json');execution=read_json(OUT/'execution.json')
    assert execution['status']=='completed_verified'
    assert execution['formal_verified']==execution['prompt_verified']==49
    receipt=OUT/'finalization.json'
    if receipt.exists() and read_json(receipt)['status'] in ('ready_for_visual_review','completed_verified'):
        print('Finalization already prepared; preserve its human material');return
    state=dict(status='independent_audit_running',started_at=datetime.now(timezone.utc).isoformat())
    write_json(receipt,state)
    checks=[]
    with (OUT/'independent-audit.log').open('w',encoding='utf-8') as log:
        for row in plan['matrix']:
            for script,arg in [('verify_cuda_run.py',row['run_id']),('verify_robustness_cuda.py',row['prompt_run_id'])]:
                process=subprocess.run([sys.executable,'-X','utf8',script,arg],cwd=ROOT,
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
                log.write(process.stdout);log.flush()
                if process.returncode:raise RuntimeError(script+' failed: '+arg)
            checks.append(dict(run_id=row['run_id'],formal_sha256=file_hash(ROOT/'study/runs'/row['run_id']/'cpu_verification.json'),
                prompt_id=row['prompt_run_id'],prompt_verification_sha256=file_hash(ROOT/'study/robustness/cuda-runs'/row['prompt_run_id']/'cpu_verification.json')))
    write_json(OUT/'independent-audit.json',dict(status='passed',formal_settings=49,prompt_settings=49,
        actual_checkpoint_bytes_rechecked=True,checks=checks))
    state['status']='rendering_report';write_json(receipt,state)
    subprocess.run([str(ROOT/'.venv-report/Scripts/python.exe'),'-X','utf8','report_phase3_grid.py'],cwd=ROOT,check=True)
    state['status']='preparing_review_material';write_json(receipt,state)
    pack=package(plan)
    report=read_json(OUT/'report-manifest.json')
    for name,expected in report['input_sha256'].items():assert file_hash(ROOT/name)==expected
    for name,expected in report['generated_sha256'].items():assert file_hash(OUT/name)==expected
    write_json(OUT/'delivery-audit.json',dict(status='passed',formal_settings=49,full_prompt_generations=15680,
        review_package=pack,report_manifest_sha256=file_hash(OUT/'report-manifest.json'),
        human_labels='pending',visual_review='pending',scope='Numeric/source/checkpoint/package checks passed; final visual and editorial acceptance remains explicit.'))
    state.update(status='ready_for_visual_review',finished_at=datetime.now(timezone.utc).isoformat(),package=pack)
    write_json(receipt,state)
    subprocess.run([sys.executable,'-X','utf8','phase3_grid_progress.py'],cwd=ROOT,check=True)
    print('FULL_GRID_NUMERIC_AND_REVIEW_DELIVERY_READY_VISUAL_REVIEW_PENDING',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--wait',action='store_true');args=parser.parse_args()
    with prevent_idle_sleep():
        while True:
            state=read_json(OUT/'execution.json')
            if state['status']=='completed_verified':break
            if state['status']=='stopped_on_error':raise RuntimeError('GPU queue stopped; finalization does not claim success: '+state.get('error',''))
            if not args.wait:raise RuntimeError('Grid is still running')
            time.sleep(30)
        finalize()


if __name__=='__main__':main()
