"""Guard method-specific CLI defaults and provenance after both PRs are combined."""
import argparse
import json
from pathlib import Path
import pytest
import cuda_common


def parse(method,*extra):
    parser=argparse.ArgumentParser()
    cuda_common.add_run_arguments(parser)
    return parser.parse_args(['--method',method,'--run-id','combined-unit',*extra])


def test_method_defaults_reach_both_spec_and_runtime_arguments(tmp_path):
    sim=parse('SimNPO','--gamma','0.125')
    _,spec=cuda_common.run_spec(sim)
    assert (spec['beta'],sim.beta,spec['delta'],spec['gamma'])==(4.5,4.5,0.,.125)
    cache=tmp_path/'cache.json'
    cache.write_text(json.dumps({'test':'hash binding only'}))
    npo=parse('NPO','--reference-cache',str(cache))
    _,spec=cuda_common.run_spec(npo)
    assert (spec['beta'],npo.beta)==(.1,.1)
    assert spec['reference_cache_sha256']==cuda_common.file_hash(cache)
    assert 'delta' not in spec
    explicit=parse('SimNPO','--beta','2.5','--delta','0.2')
    assert cuda_common.run_spec(explicit)[1]['beta']==2.5


def test_invalid_new_method_controls_fail_before_claiming_run():
    for method in ('SimNPO','NPO'):
        with pytest.raises(ValueError):cuda_common.run_spec(parse(method,'--beta','0'))
    with pytest.raises(ValueError):cuda_common.run_spec(parse('NPO'))
    with pytest.raises(ValueError):cuda_common.run_spec(parse('GradAscent','--gamma','0.125'))


def test_new_sources_are_required_and_bound_only_for_used_method():
    ordinary=cuda_common.source_identity()['source_sha256']
    sim=cuda_common.source_identity(include_simnpo=True)['source_sha256']
    npo=cuda_common.source_identity(include_npo=True)['source_sha256']
    assert 'simnpo_cuda.py' in sim and 'simnpo_cuda.py' not in ordinary
    assert 'npo_cuda.py' not in sim and 'simnpo_cuda.py' not in npo
    assert {'npo_cuda.py','npo_data.py','npo_reference.py','prepare_npo_reference.py'}<=set(npo)
    for name in ('simnpo_cuda.py','npo_cuda.py','npo_data.py','npo_reference.py','prepare_npo_reference.py'):
        recorded=(sim if name.startswith('simnpo') else npo)[name]
        assert recorded==cuda_common.file_hash(cuda_common.ROOT/name)
