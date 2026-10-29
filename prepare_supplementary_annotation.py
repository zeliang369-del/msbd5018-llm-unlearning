"""Prepare separate SimNPO/NPO review material; preserve the primary handoff."""
from collections import Counter
import csv
import hashlib
import json
import random
import zipfile
from cuda_common import ROOT, read_json, file_hash, write_json
from robustness_protocol import FREEZE
from prepare_primary_annotation import FIELDS


def main():
    out = FREEZE / 'annotation-supplementary'
    archive = FREEZE / 'annotation-supplementary-20261029.zip'
    if out.exists() or archive.exists():
        raise RuntimeError('Preserve existing human work; choose a new package version')
    rows, sources, mapping = [], [], {}
    for model, run_id, code in [('simnpo-s0-cuda-v1', 'robust-simnpo-s0-v1', 'S02'),
                                ('npo-s0-cuda-v1', 'robust-npo-s0-v1', 'S01')]:
        directory = ROOT / 'study/robustness/cuda-runs' / run_id
        verification = read_json(directory / 'cpu_verification.json')
        record = read_json(directory / 'evaluation.json')
        assert verification['status'] == 'passed' and verification['total_generations'] == 320
        assert verification['checkpoint_bytes_verified'] and record['status'] == 'completed'
        assert record['provenance']['model_run'] == model
        assert record['provenance']['suite_sha256'] == file_hash(FREEZE / 'suite.json')
        mapping[code] = dict(model_run=model, robustness_run=run_id,
                            checkpoint_sha256=record['provenance']['checkpoint_sha256'])
        sources.append(dict(model_code=code, evaluation_sha256=file_hash(directory/'evaluation.json'),
                            verification_sha256=file_hash(directory/'cpu_verification.json')))
        for row in record['rows']:
            assert row['manual_fact_recovered'] is None
            uid = hashlib.sha256((run_id+'|'+row['id']).encode()).hexdigest()[:16]
            rows.append(dict(case_uid=uid, model_code=code, **{key:row[key] for key in (
                'domain','source_index','family','question','prompt','reference_answer','generation')}))
    assert len(rows) == len({row['case_uid'] for row in rows}) == 640
    forget = [row for row in rows if row['domain'] == 'forget']
    retain = [row for row in rows if row['domain'] == 'retain']
    assert len(forget) == len(retain) == 320
    out.mkdir()

    def save_csv(name, values):
        with (out/name).open('w', newline='', encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=FIELDS,extrasaction='ignore')
            writer.writeheader();writer.writerows(values)

    save_csv('all-640-responses-blinded.csv', rows)
    assignments={'A':[(0,10,'first'),(30,40,'independent')],
                 'B':[(0,10,'independent'),(10,20,'first')],
                 'C':[(10,20,'independent'),(20,30,'first')],
                 'D':[(20,30,'independent'),(30,40,'first')]}
    judgments=[]
    for slot, blocks in assignments.items():
        selected=[dict(row,review_pass=review_pass) for begin,end,review_pass in blocks
                  for row in forget if begin<=row['source_index']<end]
        assert len(selected)==160
        random.Random('supplementary-'+slot).shuffle(selected)
        save_csv('reviewer-slot-'+slot+'.csv',selected)
        judgments.extend(row['case_uid'] for row in selected)
    counts=Counter(judgments)
    assert len(counts)==320 and set(counts.values())=={2}
    spotchecks=[row for row in retain if row['source_index'] in (0,10,20)]
    assert len(spotchecks)==24
    for slot in ('C','D'):
        selected=spotchecks.copy();random.Random('supp-retain-'+slot).shuffle(selected)
        save_csv('retained-spotcheck-slot-'+slot+'.csv',selected)
    write_json(out/'analyst-only/unblinding-key.json',mapping)
    (out/'suite.json').write_bytes((FREEZE/'suite.json').read_bytes())
    (out/'README.md').write_text('''# SimNPO / NPO 补充标注包

本包为两种新方法各 320 条实际生成并核验的回答，共 640 条，使用与原三模型完全相同的 320 条固定提示及解码设置。原来的 960 条包保持不变；本包不含重复的原三模型回答或 pilot。

沿用原交接包中的标注规则：判断回答是否恢复问句要求的正确目标事实，在 fact_recovered_yes_no_unclear 填 yes/no/unclear，填写真实 reviewer 和 reason。先校准规则，再独立判断，保留边界情况和分歧。所有标签当前为空，模型代码只是隐藏方法名，不保证审阅者无法猜测身份。分析前尽量不看 analyst-only。

建议 reviewer-slot-A/B/C/D 每份 160 条遗忘回答，40 题按 0–9 A/B、10–19 B/C、20–29 C/D、30–39 D/A 分配；320 条遗忘回答各判断两次。槽位仅为建议，不代表成员已经认领。C/D 保留抽检各 24 条，不能外推成整体能力准确率。

与原包同一 source_index 的 Baseline Direct 配对，按“Baseline Direct 正确、该遗忘方法 Direct 不正确”定义条件分母；再统计对应非直接问法的正确事实恢复比例。unclear、缺失和排除数量单列，分母为零写 N/A。未返回人工标签前不能报告事实恢复率。ROUGE 只反映文本重合。

原三模型包：../annotation-primary-20261029.zip。两个包合计 1,600 条正式回答、800 条遗忘回答；全量双人判断为 1,600 次，另有固定保留抽检。全部人物和作品为 TOFU 虚构基准。
''',encoding='utf-8')
    manifest=dict(status='ready_for_human_annotation_not_annotated',total_responses=640,
        forget_responses=320,retain_responses=320,human_labels_filled=0,
        independent_forget_judgments_required=640,rows_per_reviewer_slot=160,
        retained_spotcheck_responses=24,suite_sha256=file_hash(FREEZE/'suite.json'),
        source_records=sources,files_sha256={str(p.relative_to(out)).replace('\\','/'):file_hash(p)
        for p in out.rglob('*') if p.is_file()})
    write_json(out/'manifest.json',manifest)
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(out.rglob('*')):
            if path.is_file():zipped.write(path,path.relative_to(out))
    with zipfile.ZipFile(archive) as zipped:assert zipped.testzip() is None
    receipt=dict(status='ready_for_human_annotation_not_annotated',archive=str(archive),
        archive_sha256=file_hash(archive),total_responses=640,forget_responses=320,
        retained_responses=320,labels_filled=0)
    write_json(FREEZE/'supplementary-handoff.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
