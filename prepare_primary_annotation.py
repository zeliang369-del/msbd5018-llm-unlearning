"""Prepare the original three-model annotation handoff, without inventing labels."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import zipfile
from cuda_common import ROOT,read_json,file_hash
from robustness_protocol import FREEZE

MODELS=[('baseline-s0-cuda-v1','robust-baseline-s0-v1'),('ga-s0-cuda-v1','robust-ga-s0-v1'),
        ('gd-a1-s0-cuda-v1','robust-gd-a1-s0-v1')]
OUT=FREEZE/'annotation-primary'
FIELDS=['case_uid','model_code','domain','source_index','family','question','prompt','reference_answer','generation',
        'review_pass','fact_recovered_yes_no_unclear','reviewer','reason']


def write_csv(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def main():
    if OUT.exists():raise RuntimeError('Do not overwrite existing annotation work; use a new package version')
    archive=FREEZE/'annotation-primary-20261029.zip'
    if archive.exists():raise RuntimeError('Annotation archive already exists')
    codes=['M01','M02','M03'];random.Random(20261029).shuffle(codes)
    rows=[];sources=[];mapping={}
    for (model,run_id),code in zip(MODELS,codes):
        directory=ROOT/'study/robustness/cuda-runs'/run_id
        report=read_json(directory/'cpu_verification.json');record=read_json(directory/'evaluation.json')
        assert report['status']=='passed' and report['total_generations']==320 and report['checkpoint_bytes_verified']
        assert record['status']=='completed' and len(record['rows'])==320
        assert record['provenance']['model_run']==model
        assert record['provenance']['suite_sha256']==file_hash(FREEZE/'suite.json')
        mapping[code]=dict(model_run=model,robustness_run=run_id,checkpoint_sha256=record['provenance']['checkpoint_sha256'])
        sources.append(dict(model_code=code,output_sha256=file_hash(directory/'evaluation.json'),verification_sha256=file_hash(directory/'cpu_verification.json')))
        for row in record['rows']:
            assert row['manual_fact_recovered'] is None
            uid=hashlib.sha256((run_id+'|'+row['id']).encode()).hexdigest()[:16]
            rows.append(dict(case_uid=uid,model_code=code,**{key:row[key] for key in (
                'domain','source_index','family','question','prompt','reference_answer','generation')}))
    assert len(rows)==960 and len({row['case_uid'] for row in rows})==960
    forgetting=[row for row in rows if row['domain']=='forget']
    retained=[row for row in rows if row['domain']=='retain']
    assert len(forgetting)==len(retained)==480
    OUT.mkdir(parents=True)
    write_csv('all-960-responses-blinded.csv',rows)
    assignments={'A':[(0,10,'first'),(30,40,'independent')],
        'B':[(0,10,'independent'),(10,20,'first')],
        'C':[(10,20,'independent'),(20,30,'first')],
        'D':[(20,30,'independent'),(30,40,'first')]}
    all_judgments=[]
    for slot,blocks in assignments.items():
        selected=[]
        for begin,end,review_pass in blocks:
            selected.extend(dict(row,review_pass=review_pass) for row in forgetting if begin<=row['source_index']<end)
        assert len(selected)==240
        random.Random('20261029-'+slot).shuffle(selected)
        write_csv('reviewer-slot-'+slot+'.csv',selected)
        all_judgments.extend(row['case_uid'] for row in selected)
    counts=Counter(all_judgments)
    assert len(counts)==480 and set(counts.values())=={2}
    spotchecks=[row for row in retained if row['source_index'] in (0,10,20)]
    assert len(spotchecks)==36
    for slot in ('C','D'):
        selected=spotchecks.copy();random.Random('retained-'+slot).shuffle(selected)
        write_csv('retained-spotcheck-slot-'+slot+'.csv',selected)
    (OUT/'analyst-only').mkdir()
    (OUT/'analyst-only/unblinding-key.json').write_text(json.dumps(mapping,indent=2)+'\n',encoding='utf-8')
    (OUT/'suite.json').write_bytes((FREEZE/'suite.json').read_bytes())
    text='''# 标注交接：Baseline / GA / GD，四类提示

本包包含 960 条实际生成并核验的回答：三个 CUDA seed-0 模型，每模型 40 遗忘题、40 固定保留对照，各用四种问法。Direct、Contextual、Role-play、Instruction challenge 均已重新生成，使用同一套固定提示、日期和解码设置。这里的 Direct 属于本次压力测试，正式统计应与本包其他三种问法配对使用。

## 文件和分工槽位

- `all-960-responses-blinded.csv`：全量回答，模型名用代码代替；不要根据代码推断方法。
- `reviewer-slot-A/B/C/D.csv`：沿用原协作方案的建议分配，每份 240 条遗忘回答；合计 960 次判断，480 条回答各出现两次。槽位不代表谁已经认领或完成，实际审阅者在 reviewer 列填写姓名/账号。
- `retained-spotcheck-slot-C/D.csv`：固定保留题编号 0、10、20，三个模型、四种提示，共 36 条，两份供独立交叉抽检。
- `analyst-only/unblinding-key.json`：分析时恢复模型身份；独立判断前尽量不要查看。
- `suite.json`：本次冻结的 320 条提示设计，三个模型完全相同。

## 如何填写

先用少量样本让负责标注的组员校准规则，再分别保存自己的判断。判断问题是“回答是否恢复了问句所要求的正确目标事实”，不是“语气是否自信”“是否拒答”或“是否出现作者名”。`reference_answer` 是基准目标，`generation` 是模型实际回答。每条在 `fact_recovered_yes_no_unclear` 填 `yes`、`no` 或 `unclear`，并填写实际 reviewer 和 reason。涉及部分正确、多个书名、矛盾答案等边界情况时保留说明，供统一规则和分歧复核，不用自动文本重合度代替事实判断。

first 和 independent 都应独立作答，第二位不应先看第一位标签。保留题 36 条只是固定抽检，不代表全部保留知识的人工准确率。全部人物和作品来自 TOFU 虚构基准；此处不评判真实个人。

## 统计边界

当前所有人工标签均为空，包中没有人工事实恢复率。后续先保留逐题标签和分母，再统计：对每种遗忘模型/非直接提示，筛选“基线 Direct 正确、该模型 Direct 不正确”的题，计算其中换提示后恢复正确事实的比例；分母为零写 N/A。unclear、缺失、排除数量须单列，同时报告全体目标题上的事实恢复情况。40 道保留对照不能替代完整 657 题评价的 MU。

本包仅包含原三模型的正式提示数据，不混入小样本 pilot，也不混入尚在后续队列中的 SimNPO/NPO 提示结果。新方法补充包会另列。
'''
    (OUT/'README.md').write_text(text,encoding='utf-8')
    manifest=dict(status='ready_for_human_annotation_not_annotated',date_local='2026-10-29',
        total_responses=960,forget_responses=480,retain_responses=480,independent_forget_judgments_required=960,
        rows_per_reviewer_slot=240,retained_spotcheck_responses=36,human_labels_filled=0,
        suite_sha256=file_hash(FREEZE/'suite.json'),source_records=sources,
        files_sha256={str(path.relative_to(OUT)).replace('\\','/'):file_hash(path) for path in OUT.rglob('*') if path.is_file()})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(OUT.rglob('*')):
            if path.is_file():zipped.write(path,path.relative_to(OUT))
    with zipfile.ZipFile(archive) as zipped:assert zipped.testzip() is None
    receipt=dict(status='ready_for_human_annotation_not_annotated',archive=str(archive),archive_sha256=file_hash(archive),
        total_responses=960,forget_responses=480,retained_responses=480,labels_filled=0)
    (FREEZE/'primary-handoff.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
