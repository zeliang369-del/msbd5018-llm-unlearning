"""Freeze and validate the existing answer-free prompt-format stress suite."""
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parent
FREEZE=ROOT/'study/phase3/robustness-20261029'
BASELINE=ROOT/'reproduction_full/runs/baseline/evaluation.json'
from build_robustness_suite import TEMPLATES


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_suite(suite, baseline_path=BASELINE):
    baseline_path=Path(baseline_path)
    baseline=read(baseline_path)
    if suite.get('source_sha256')!=sha(baseline_path):
        raise ValueError('Suite baseline source hash mismatch')
    items=suite['items']
    if len(items)!=320 or len({row['id'] for row in items})!=320:
        raise ValueError('Expected 320 unique prompt IDs')
    expected=[]
    for domain in ('forget','retain'):
        for source in baseline['domains'][domain][:40]:
            for family,template in TEMPLATES.items():
                expected.append(dict(id=f"{domain}-{source['source_index']:03d}-{family}",domain=domain,
                    source_index=source['source_index'],family=family,question=source['question'],
                    prompt=template.format(question=source['question']),reference_answer=source['reference_answer']))
    if items!=expected:
        raise ValueError('Prompt content, order, target, family, or IDs differ from the frozen design')
    return suite


def select_rows(suite, limit):
    if not 0<=limit<=40:raise ValueError('limit must be 0..40')
    return [row for row in suite['items'] if not limit or row['source_index']<limit]


@lru_cache(maxsize=4)
def baseline_record(path):
    return read(path)


def expected_input(row, baseline_path=BASELINE):
    source=baseline_record(baseline_path)['domains'][row['domain']][row['source_index']]
    text=source['input']
    if text.count(row['question'])!=1:raise ValueError('Question cannot be uniquely located in processed input')
    return text.replace(row['question'],row['prompt'],1)


def validate_prefix(saved, expected, batch_size):
    ids=[row['id'] for row in saved]
    if ids!=[row['id'] for row in expected[:len(ids)]] or len(ids)>len(expected):
        raise ValueError('Saved output is not an exact prefix of the frozen prompt order')
    if len(ids)!=len(expected) and len(ids)%batch_size:
        raise ValueError('Partial output must end at a complete batch boundary')


def valid_name(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',name):raise ValueError('Invalid run identifier')
    return name


def freeze():
    source=ROOT/'study/robustness/suite.json'
    suite=validate_suite(read(source))
    FREEZE.mkdir(parents=True,exist_ok=True)
    destination=FREEZE/'suite.json'
    if destination.exists() and destination.read_bytes()!=source.read_bytes():
        raise ValueError('Do not replace an existing frozen suite')
    if not destination.exists():destination.write_bytes(source.read_bytes())
    counts=Counter((r['domain'],r['family']) for r in suite['items'])
    report=dict(status='frozen_validated',date_local='2026-10-29',suite_sha256=sha(destination),
        baseline_source_sha256=sha(BASELINE),prompts=320,questions_per_domain=40,
        families=list(TEMPLATES),counts={f'{d}/{f}':n for (d,f),n in counts.items()},
        leakage_check='Each prompt equals the approved wrapper applied only to the source question. Reference answers are not included.',
        scope='Prompt-format stress test, not a standardized jailbreak benchmark. Forty retained questions are a control subset, not full MU.',
        owner_confirmation='Zeliang confirmed use of the existing repository suite on 2026-10-29')
    (FREEZE/'suite-manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':print(json.dumps(freeze(),indent=2))
