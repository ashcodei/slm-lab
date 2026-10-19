"""Prelaunch rejection tests; these fixtures never certify actual GPU capacity."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from slm_lab.corpus import file_hash
import slm_lab.execution_production_v6 as production
from test_code_residual_lifecycle_v4 import plan_fixture as original_fixture


def fixture(tmp_path, *, capacity=False):
    plan=original_fixture(tmp_path)
    plan.update(format=production.CAPACITY_FORMAT if capacity else production.FORMAT,
        execution_feedback=str(tmp_path/'execution-feedback'),execution_manifest_sha256='b'*64,
        execution_admission_sha256='c'*64,
        execution_options=dict(positive_batch_size=1,positive_accumulation=1,pair_batch_size=1,
            pair_accumulation=1,positive_coefficient=.3,DPO_alpha=.1,DPO_beta=.1))
    plan['source_files_sha256'].update({str(production.PACKAGE/n):'a'*64 for n in production.RUNTIME_FILES})
    if capacity:
        plan.update(candidate_export=False,storage=dict(maximum_report_bytes=8*1024**2,minimum_free_bytes=26*1024**3))
    else:
        plan.update(execution_reference=str(tmp_path/'reference.json'),execution_reference_sha256='d'*64,
            execution_capacity_proof=str(tmp_path/'execution-proof.json'),execution_capacity_proof_sha256='e'*64,
            execution_auxiliary_fingerprint='f'*64)
    return plan


@pytest.mark.parametrize('capacity',[False,True])
def test_original_source_layout_and_new_reference_runtime_options_remain_mandatory(tmp_path,capacity):
    plan=fixture(tmp_path,capacity=capacity)
    production.validate_plan(plan,capacity=capacity)
    changes=[('world_size',1),('residual_width',4096),('execution_manifest_sha256',None),
        ('execution_feedback','relative'),('execution_options',{**plan['execution_options'],'DPO_alpha':0})]
    if not capacity:changes.append(('training_ready',False))
    for key,value in changes:
        with pytest.raises(ValueError):production.validate_plan({**plan,key:value},capacity=capacity)
    bad=deepcopy(plan);bad['source_files_sha256'].pop(str(production.PACKAGE/'execution_collective_v2.py'))
    with pytest.raises(ValueError):production.validate_plan(bad,capacity=capacity)


@pytest.mark.parametrize('capacity',[False,True])
def test_unfinalized_actual_sources_reject_before_GPU_queries_or_output_creation(tmp_path,monkeypatch,capacity):
    plan=fixture(tmp_path,capacity=capacity)
    source=Path(plan['source_bundle']);source.mkdir()
    (source/'manifest.json').write_text(json.dumps(dict(format='own-complete-code-final-source-bundle-v4',
        all30_actual_prepared_families_opened=False,full_original_paired_admission_completed=False)))
    plan['source_bundle_manifest_sha256']=file_hash(source/'manifest.json')
    path=tmp_path/'run-plan.json';path.write_text(json.dumps(plan))
    def forbidden(*args,**kwargs):raise AssertionError('Cannot query GPUs, launch workers or create output before actual final sources')
    monkeypatch.setattr(production.subprocess,'run',forbidden)
    with pytest.raises(ValueError,match='actual30-family|complete final-source'):
        production.controller(path,capacity=capacity)
    assert not Path(plan['output']).exists()


def test_old_capacity_or_cpu_replica_cannot_approve_new_execution_training(tmp_path):
    plan=fixture(tmp_path)
    reference=dict(feedback=dict(pairs=7),policy=dict(device_type='cpu',precision='fp32',model_counts=dict(total_unique=8688)))
    for proof in [dict(format='own-complete-code-residual-capacity-proof-v4',passed=True),
            dict(format=production.PROOF_FORMAT,passed=True,actual_world_size=4,model_counts=dict(total_unique=8688))]:
        with pytest.raises(ValueError,match='actual exact|Actual exact'):
            production.validate_capacity_proof(proof,plan,reference,
                auxiliary_maximum_contexts=dict(positive=4066,chosen=4066,rejected=4067))


def test_shortened_auxiliary_probe_is_not_capacity_for_complete_admitted_programs(tmp_path):
    from slm_lab.code_residual_collective_checkpoint_v2 import fingerprint
    from slm_lab.evaluation_provenance import capture
    from slm_lab.execution_training_v5 import auxiliary_fingerprint
    plan=fixture(tmp_path)
    maxima=dict(positive=4066,chosen=4066,rejected=4067)
    reference=dict(feedback=dict(pairs=210),policy=dict(device_type='cuda',precision='bf16',
        model_counts=production.original.MODEL_COUNTS,base_checkpoint_sha256=plan['base_checkpoint_sha256'],
        tokenizer_sha256=plan['tokenizer_sha256']))
    # Pure schema/gate fixture. No model, CUDA process, reference scoring,
    # training or real capacity proof is produced by this unit test.
    rank_paths={};hashes={}
    for rank in range(4):
        path=tmp_path/f'rank-{rank}.json'
        item=dict(scope='STRUCTURAL_UNIT_FIXTURE_NOT_RUNTIME_CAPACITY',rank=rank,device_name='NVIDIA L40',
            model_counts=production.original.MODEL_COUNTS,actual_full_kernel_optimizer_updates=1,
            actual_maximum_context_backward=4096,actual_image_batch_sizes=[[448,448]],
            all_declared_source_families_exercised=list(production.REQUIRED_FAMILIES),
            all_auxiliary_maximum_contexts_exercised=True,auxiliary_maximum_contexts=maxima,
            original_backbone_unchanged=True,peak_allocated_bytes=1024,
            execution_extreme_backwards=[dict(global_execution_pairs=4,finite_all_and_only_residual_gradients=True,
                **{name+'_context_tokens':value for name,value in maxima.items()})])
        path.write_text(json.dumps(item));rank_paths[str(rank)]=str(path);hashes[str(path)]=file_hash(path)
    proof=dict(format=production.PROOF_FORMAT,passed=True,
        **{k:plan[k] for k in ['base_checkpoint_sha256','tokenizer_sha256','replay_contract_fingerprint',
            'execution_manifest_sha256','execution_admission_sha256','execution_reference_sha256']},
        training_options_fingerprint=fingerprint(plan['training_options']),
        execution_options_fingerprint=fingerprint(plan['execution_options']),
        execution_auxiliary_fingerprint=auxiliary_fingerprint(reference,plan['execution_options']),
        model_counts=production.original.MODEL_COUNTS,actual_world_size=4,actual_reference_pairs_scored=210,
        runtime_provenance=capture([production.PACKAGE/n for n in sorted(production.RUNTIME_FILES)]),
        candidate_weights_saved=False,rank_reports=rank_paths,files_sha256=hashes)
    production.validate_capacity_proof(proof,plan,reference,auxiliary_maximum_contexts=maxima)
    path=Path(rank_paths['3']);item=json.loads(path.read_text())
    item['execution_extreme_backwards'][0]['positive_context_tokens']=2048
    path.write_text(json.dumps(item));proof['files_sha256'][str(path)]=file_hash(path)
    with pytest.raises(ValueError,match='Every actual rank'):
        production.validate_capacity_proof(proof,plan,reference,auxiliary_maximum_contexts=maxima)
