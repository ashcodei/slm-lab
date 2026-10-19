"""Four CPU workers exercise real374-pair frozen references and exact recovery.

Original code/replay providers and16px images are artificial test fixtures.
The complete195/374 execution views and own32768 BPE are actual admitted data.
This is not a full857M GPU run or a model capability experiment.
"""
from contextlib import nullcontext
from dataclasses import asdict,replace
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image
import pytest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from tokenizers import Tokenizer

from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.code_residual_collective_checkpoint_v2 import fingerprint
from slm_lab.code_residual_source_contract_v4 import build_required_contract,COMPONENT_NAMES
from slm_lab.code_residual_stage_v3 import ExpandedReplayEncoder
from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.codenet_token_pack_v1 import packed_encoder
from slm_lab.corpus import file_hash
from slm_lab.execution_collective_v2 import prepare_reference_bundle
from slm_lab.execution_production_v6 import open_feedback
from slm_lab.execution_training_v5 import ExecutionResidualKernel,auxiliary_fingerprint,reference_from_checkpoint
from slm_lab.model import TinyMultimodalLM
from slm_lab.recurrent_ablation import state_digest

from test_code_residual_stage_v2 import Guard
from test_code_residual_source_contract_v4 import Code,Replay
from test_codenet_runner_v1 import small_config,options

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/execution-feedback-union-v1'
TOKEN=ROOT/'checkpoints/extended-curriculum-v1/joint/tokenizer.json'


def worker(rank,rendezvous,directory):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=4,timeout=timedelta(minutes=10))
    try:
        folder=Path(directory);config=replace(small_config(),vocab_size=32768,max_text_length=4096)
        token=Tokenizer.from_file(str(TOKEN));feedback=open_feedback(dict(execution_feedback=str(DATA),tokenizer=str(TOKEN),
            execution_manifest_sha256=file_hash(DATA/'manifest.json'),execution_admission_sha256=file_hash(DATA/'admission.json')))
        code=Code(folder);replay=Replay(folder)
        def fresh():
            torch.manual_seed(50189)
            return RoutedCodeResidual(TinyMultimodalLM(config),width=12)
        source=folder/'base.pt';plan=folder/'plan.json'
        if rank==0:
            model=fresh();torch.save(dict(config=asdict(config),model=model.base.state_dict(),tokenizer_sha256=file_hash(TOKEN)),source)
        dist.barrier()
        model=fresh();bundle=prepare_reference_bundle(model,feedback,base_checkpoint_sha256=file_hash(source),
            tokenizer_sha256=file_hash(TOKEN),device='cpu',amp=nullcontext,precision='fp32')
        assert len(bundle['cache']['records'])==374
        opts={**options(),'gradient_accumulation':1,'replay_accumulation':1,'image_size':16,'log_every':1}
        aux=dict(positive_batch_size=1,positive_accumulation=1,pair_batch_size=1,pair_accumulation=2,
            positive_coefficient=.3,DPO_alpha=.1,DPO_beta=.1)
        auxfp=auxiliary_fingerprint(bundle,aux)
        if rank==0:
            plan.write_text(json.dumps(dict(scope='Own small CPU replica; artificial original code/all30 replay providers; actual195/374admitted execution views.',
                production_stage_ready=False,execution_auxiliary_fingerprint=auxfp)))
            (folder/'external-reference.json').write_text(json.dumps(bundle))
        dist.barrier()
        sources=dict(base_checkpoint_sha256=file_hash(source),tokenizer_sha256=file_hash(TOKEN),run_plan_sha256=file_hash(plan))
        contract=build_required_contract(replay,file_hash(TOKEN),{n:hashlib.sha256(('fixture:'+n).encode()).hexdigest() for n in COMPONENT_NAMES})
        def construct(reference):
            m=fresh();o=torch.optim.AdamW(m.adapter_parameters(),lr=opts['learning_rate'],weight_decay=opts['weight_decay'],foreach=False)
            k=ExecutionResidualKernel(m,o,code,packed_encoder(code,token,config),replay,
                ExpandedReplayEncoder(token,config,folder,image_size=16),opts,sources,'cpu',nullcontext,
                replay_contract=contract,expected_contract_fingerprint=fingerprint(contract),guard=Guard(),
                feedback=feedback,reference_bundle=reference,execution_options=aux,expected_auxiliary_fingerprint=auxfp)
            return m,o,k
        model,optimizer,kernel=construct(bundle);base_before=state_digest(model.base.state_dict())
        initial_residuals=state_digest(model.residuals.state_dict())
        random.seed(1500+rank);np.random.seed(1600+rank);torch.manual_seed(1700+rank)
        def state():return fingerprint(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=capture_rng('cpu'),
            metrics=kernel.metrics,sampler=kernel.sampler_state(kernel.completed)))
        def update():
            draws=[random.random(),float(np.random.random()),float(torch.rand(()))]
            return draws,kernel.update()
        first=update();partial=kernel.snapshot(folder/'resume.pt',maximum_bytes=3_000_000,minimum_free_bytes=0)
        second=update();expected=state()
        if rank==0:(folder/'external-reference.json').unlink()
        dist.barrier()
        recovered=reference_from_checkpoint(folder/'resume.pt',partial['checkpoint_sha256'])
        assert recovered==bundle
        model,optimizer,kernel=construct(recovered);kernel.restore(folder/'resume.pt',partial['checkpoint_sha256'])
        assert update()==second and state()==expected
        assert state_digest(model.base.state_dict())==base_before
        assert state_digest(model.residuals.state_dict())!=initial_residuals
        assert all(m['gradient_norm_before_clipping']>0 for m in kernel.metrics)
        progress=kernel.sampler_state(2)
        assert progress['execution_source_cursors']['positive']['examples']==8
        assert progress['execution_source_cursors']['pair']['examples']==16
        assert all(m['global_execution_pairs']==8 for m in kernel.metrics)
        assert all(m['execution_reference_fingerprint']==fingerprint(bundle) for m in kernel.metrics)
        kernel.snapshot(folder/'step2.pt',maximum_bytes=3_000_000,minimum_free_bytes=0)
        report=dict(rank=rank,actual_four_CPU_workers=True,model_counts=model.parameter_counts(),
            actual_original_positive_records=195,actual_original_pair_reference_scores=374,
            actual_own_BPE_vocabulary=32768,actual_model_text_context=4096,
            actual_reference_updates=2,actual_replayed_update=2,planned_steps=kernel.steps,production_stage_complete=False,
            all_reference_scores_embedded_in_complete_checkpoint=True,external_reference_deleted_before_resume=True,
            exact_weights_Adam_per_rank_RNG_metrics_and_all_source_cursors_resume=True,
            original_frozen_random_backbone_identical=True,all_only_residual_parameters_updated=True,
            scope='Actual195/374 admitted complete source views/BPE; own small CPU replica and artificial original code/all30 replay fixtures with16px images. Not actual857M GPU capacity, all-source production or model capability.')
        (folder/f'rank-{rank}.json').write_text(json.dumps(report))
    finally:dist.destroy_process_group()


def test_actual_union_complete_reference_training_and_exact_four_worker_recovery(tmp_path):
    Image.new('RGB',(8,4),(21,55,90)).save(tmp_path/'image.png')
    Image.new('RGB',(8,4),(220,85,10)).save(tmp_path/'other.png')
    mp.spawn(worker,args=(str(tmp_path/'rendezvous'),str(tmp_path)),nprocs=4,join=True)
    for rank in range(4):
        report=json.loads((tmp_path/f'rank-{rank}.json').read_text())
        assert report['actual_original_pair_reference_scores']==374
        assert report['exact_weights_Adam_per_rank_RNG_metrics_and_all_source_cursors_resume']
