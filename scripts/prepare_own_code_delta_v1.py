"""CPU-only compression of an own learned coding delta; no neural training."""
import os
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
    raise ValueError('Disable GPU visibility before this CPU-only preparation')

from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import time

import torch

from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.own_code_delta_v1 import OwnCodeDelta,factorize,matrix_names,norm_names
from slm_lab.training import load_checkpoint

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'checkpoints/extended-curriculum-v1/joint/model.pt'
TARGET=ROOT/'checkpoints/apps-full-language-v1/model.pt'
TOKEN=BASE.parent/'tokenizer.json'
REPORTS=ROOT/'reports/own-code-delta-v1'
OUTPUT=ROOT/'checkpoints/own-code-delta-v1'
BASE_SHA='2dacd490e4a6ed20527a8393629d560e7230c1b6fdec54c1a49dc2eedf7f4e49'
TARGET_SHA='86b8c12ebb0d7fdbf14e222d3899494cd7510b9eceede4c59b34ae22e061a6f4'


def main():
    if REPORTS.exists() or OUTPUT.exists():
        raise ValueError('Preserve earlier compression or partial evidence; no implicit restart')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    tests=ROOT/'reports/own-code-delta-v1-tests.log'
    if '2 passed' not in tests.read_text():
        raise ValueError('Actual own-small-model routing/freezing/cache/code-gradient tests required')
    paths=[Path(__file__).resolve(),ROOT/'src/slm_lab/own_code_delta_v1.py',ROOT/'src/slm_lab/model.py',
        ROOT/'src/slm_lab/training.py',ROOT/'src/slm_lab/codenet_checkpoint_v1.py',
        ROOT/'tests/test_own_code_delta_v1.py',tests,TOKEN,
        ROOT/'reports/apps-full-language-endpoint-v1/complete-neural-and-source-readback.json',
        ROOT/'reports/apps-full-language-endpoint-v1/matched-native-capability-comparison.json']
    files={str(p):file_hash(p) for p in paths};files.update({str(BASE):BASE_SHA,str(TARGET):TARGET_SHA})
    guard=FileBindings(files)
    before=torch.load(BASE,map_location='cpu',weights_only=True,mmap=True)
    after=torch.load(TARGET,map_location='cpu',weights_only=True,mmap=True)
    if (before['config']!=after['config'] or before['tokenizer_sha256']!=after['tokenizer_sha256']
            or before['model'].keys()!=after['model'].keys() or after['steps']!=3203
            or not torch.equal(after['model']['embedding.weight'],after['model']['lm_head.weight'])
            or after['training_kind']!='complete_APPS_human_solution_language_v1'):
        raise ValueError('Exact same own native architecture/tokenizer and complete own code policy required')
    REPORTS.mkdir();OUTPUT.mkdir();rank=352;seed=783211
    base_count=sum(t.numel() for n,t in before['model'].items() if n!='lm_head.weight')
    plan=dict(created_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        general_base_sha256=BASE_SHA,own_code_target_sha256=TARGET_SHA,tokenizer_sha256=file_hash(TOKEN),
        source_models_are_own_scratch_trained=True,external_teacher_or_weights_used=False,
        general_base_parameters=base_count,matrix_rank=rank,randomized_SVD_oversampling=16,power_iterations=2,seed=seed,
        full_own_code_embedding_and_norms=True,reserved_future_own_point_head_parameters=786432,
        model_parameter_limit=10**9,CPU_threads=2,held_data_used=False,
        purpose='Unproven own task-delta initialization; no instruction training, external distillation or measured code gain.',
        qualifications=['Plain matrix-space SVD is not activation-whitened or generation-aware compression.',
            'Preserving delta Frobenius energy does not prove output, syntax, finite-case or algorithmic correctness.',
            'The source code target itself passes only3/175supported public-Python problems; general and multimodal capability remain weak.',
            'Automatic intent routing and production serving are not established.'],goal_achieved=False)
    atomic_json(REPORTS/'plan.json',plan)
    components=dict(matrices={},embedding=after['model']['embedding.weight'].clone(),
        norms={n:after['model'][n].clone() for n in norm_names(before['config']['num_layers'])})
    observations=[];started=time.monotonic()
    for index,name in enumerate(matrix_names(before['config']['num_layers'])):
        guard.check()
        original=before['model'][name];target=after['model'][name]
        if original.shape!=target.shape or original.dtype!=torch.float32 or not bool(torch.isfinite(target).all()):
            raise ValueError('Finite exact original own FP32 matrix shapes required')
        delta=target-original
        parts=factorize(delta,rank,seed=seed+index,iterations=2)
        approximation=parts['up']@parts['down']
        energy=float(delta.double().square().sum());error=float((delta-approximation).double().square().sum())
        components['matrices'][name]=parts
        observations.append(dict(matrix=name,shape=list(delta.shape),rank=parts['down'].shape[0],
            actual_factor_parameters=sum(t.numel() for t in parts.values()),
            original_delta_squared_Frobenius=energy,residual_delta_squared_Frobenius=error,
            retained_delta_energy=1-error/energy if energy else 1.))
        atomic_json(REPORTS/'progress.json',dict(observed_at=datetime.now(timezone.utc).isoformat(),
            completed_matrices=index+1,planned_matrices=168,latest=observations[-1],elapsed_seconds=time.monotonic()-started,
            GPU_initialized=torch.cuda.is_initialized(),neural_training_updates=0,goal_achieved=False))
        print('own_learned_delta_matrix_factorized',index+1,'/168',name,observations[-1]['retained_delta_energy'],flush=True)
        del delta,approximation
    atomic_json(REPORTS/'matrix-observations.json',observations)
    added=sum(t.numel() for value in components['matrices'].values() for t in value.values())+components['embedding'].numel()+sum(t.numel() for t in components['norms'].values())
    if base_count!=706012672 or base_count+added!=985664512 or base_count+added+786432>=10**9:
        raise ValueError('Actual unique general/code/future-point parameter budget differs')
    payload=dict(format='own-conditional-code-delta-v1',components=components,config=before['config'],
        base_checkpoint_sha256=BASE_SHA,own_code_source_checkpoint_sha256=TARGET_SHA,tokenizer_sha256=file_hash(TOKEN),
        matrix_rank=352,base_parameters=base_count,additional_code_parameters=added,total_unique_parameters=base_count+added,
        reserved_point_head_parameters=786432,total_with_reserved_point_head=base_count+added+786432,
        source_plan_sha256=file_hash(REPORTS/'plan.json'),
        measured_code_capability_preserved=False,candidate_promoted=False,goal_achieved=False)
    guard.check(full=True)
    atomic_checkpoint(OUTPUT/'model.pt',payload,2*1024**3,26*1024**3)
    base,_=load_checkpoint(BASE,'cpu',training_state=False)
    model=OwnCodeDelta(base,components,reserved_parameters=786432);model.eval()
    counts=model.counts()
    if counts['total_unique']!=985664512 or counts['total_with_reserved_head']!=986450944:
        raise ValueError('Actual instantiated combined unique model inventory must fit under1B')
    current=base.state_dict()
    if current.keys()!=before['model'].keys() or any(not torch.equal(t,current[n]) for n,t in before['model'].items()):
        raise ValueError('Every original general/vision/critic byte must be unchanged')
    model.assert_partition()
    if torch.cuda.is_initialized():
        raise ValueError('CPU compression must never initialize a CUDA context')
    guard.check(full=True)
    energy=sum(r['original_delta_squared_Frobenius'] for r in observations)
    error=sum(r['residual_delta_squared_Frobenius'] for r in observations)
    result=dict(completed_at=datetime.now(timezone.utc).isoformat(),actual_combined_inventory=counts,
        actual_decoder_matrices_factorized=len(observations),retained_matrix_delta_energy=1-error/energy,
        all_original_general_vision_critic_tensors_byte_identical=True,
        actual_code_route_includes_every_unique_general_and_code_parameter=True,
        original_general_and_code_views_share_one_backbone_and_one_vision_encoder=True,
        original_code_target_vision_tensors_differing_from_general=sum(not torch.equal(t,after['model'][n]) for n,t in before['model'].items() if n.startswith('vision.')),
        actual_factors_checkpoint_sha256=file_hash(OUTPUT/'model.pt'),actual_checkpoint_bytes=(OUTPUT/'model.pt').stat().st_size,
        source_plan_sha256=file_hash(REPORTS/'plan.json'),elapsed_seconds=time.monotonic()-started,
        CUDA_initialized=False,neural_training_updates=0,
        held_data_used=False,measured_code_generation_or_correctness_preserved=False,
        actual_full_model_GPU_capacity=False,automatic_intent_routing_established=False,
        candidate_promoted=False,high_end_or_allround_capability_established=False,goal_achieved=False)
    atomic_json(REPORTS/'complete.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
