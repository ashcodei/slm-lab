"""Bounded actual857M CUDA Adam save/restore/next-update equivalence check."""
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
from tokenizers import Tokenizer
import torch

from prepare_codenet_native_contract_pack_v2 import occupied_writer_reservation
from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.code_residual_checkpoint_v1 import make_training_state, restore_training_state
from slm_lab.codenet_checkpoint_v1 import FileBindings, atomic_checkpoint, capture_rng
from slm_lab.codenet_replay_v1 import FullReplayEncoder
from slm_lab.codenet_token_pack_v1 import CompleteCodePack, packed_encoder
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig
from slm_lab.recurrent_ablation import state_digest
from slm_lab.training import load_checkpoint, supervised_gradients, TrainingOutputs


ROOT = Path(__file__).resolve().parents[1]


def probe_update(model, optimizer, encoder, selected, device, guard, step):
    """Exact bounded update reused by the separate-process resume check."""
    guard.check(); model.train(); optimizer.zero_grad(set_to_none=True)
    draws = [random.randrange(1024), int(np.random.randint(1024)), int(torch.randint(0,1024,())),
             int(torch.randint(0,1024,(),device=device))]
    ordering = selected if sum(draws)%2 else list(reversed(selected))
    groups = [[{'packed_code_index':s['local_position']}] for s in ordering]
    def amp():return torch.autocast('cuda',dtype=torch.bfloat16)
    loss, targets = supervised_gradients(TrainingOutputs(model.code,'sft'),encoder,groups,'response',device,amp)
    if targets != 382 or encoder.truncated or any(p.grad is not None for p in model.base.parameters()):
        raise ValueError('Whole original programme targets and frozen base required')
    parameters = model.adapter_parameters()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise ValueError('Finite gradients for all residual parameters required')
    norm = torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
    optimizer.step(); model.eval()
    row = dict(step=step,stochastic_draws=draws,programme_order=[s['id'] for s in ordering],
               token_mean_nll=float(loss),response_targets=targets,gradient_norm_before_clipping=float(norm))
    print('actual_cuda_residual_update',row,flush=True)
    return row


def tree_digest(tree):
    """Compare every tensor byte and scalar, including Adam moments and all RNGs."""
    digest = hashlib.sha256()
    def visit(value):
        digest.update(type(value).__name__.encode()+b'\0')
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode()); digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=lambda k:(type(k).__name__,repr(k))):visit(key); visit(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(str(len(value)).encode())
            for child in value:visit(child)
        elif value is None or isinstance(value, (bool, int, float, str)):
            digest.update(repr(value).encode())
        else:
            raise ValueError('Unknown training state comparison type: '+type(value).__name__)
    visit(tree)
    return digest.hexdigest()


def main():
    output = ROOT/'checkpoints/code-residual-cuda-resume-v1'
    reports = ROOT/'reports/code-residual-cuda-resume-v1'
    if output.exists() or reports.exists():raise ValueError('Preserve every earlier CUDA resume artifact')
    occupied = occupied_writer_reservation()
    reserve = 26*1024**3-occupied['credited_bytes']+256*1024**2
    maximum_checkpoint = 2*1024**3
    if shutil.disk_usage(ROOT).free < reserve+maximum_checkpoint+64*1024**2:
        raise ValueError('Whole residual/Adam/RNG checkpoint above remaining original writer reserve required')
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info(device)[0] < 12*1024**3:
        raise ValueError('At least12GiB initially free on the shared BF16 GPU required')
    torch.cuda.set_per_process_memory_fraction(.20, device); torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source = ROOT/'checkpoints/extended-curriculum-v1/language-pretraining/model.pt'
    expected_source = '4c05d99b0c3fc775bf89e70c6cbac9c63017eb36f67e17c738b5be268fb191ca'
    if file_hash(source) != expected_source:raise ValueError('Exact own completed-language source required')
    prior = json.loads((ROOT/'reports/code-residual-readiness-v1/plan.json').read_text())
    selected = prior['selected_complete_original_training_rows']
    if len(selected) != 2 or len({s['group_id'] for s in selected}) != 2:
        raise ValueError('Same two distinct original training programme controls required')
    pack = CompleteCodePack(ROOT/'data/codenet-durable-pack-v2', 'train', purpose='training')
    for s in selected:
        ids, labels = pack.sequence(s['local_position'])
        if (int(pack.positions[s['local_position']]) != s['global_position']
                or hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest() != s['full_input_ids_sha256']
                or len(ids) != s['context_tokens'] or sum(v != -100 for v in labels) != s['target_tokens']):
            raise ValueError('Whole admitted original train inputs/targets changed')
    joint = ROOT/'data/extended-curriculum-v1/joint.train.jsonl'
    controls_by_id = {r['id']:r['kind'] for r in prior['original_training_control_ids']}
    controls = []
    for line in joint.open():
        row = json.loads(line)
        if row['id'] in controls_by_id:controls.append((controls_by_id[row['id']],row))
        if len(controls) == 2:break
    if {kind for kind,row in controls} != {'text','image'}:
        raise ValueError('Same original text and448px image invariance controls required')
    names = ['scripts/check_code_residual_cuda_resume_v1.py', 'src/slm_lab/code_residual_checkpoint_v1.py',
             'src/slm_lab/code_residual_branch_v1.py', 'src/slm_lab/model.py', 'src/slm_lab/training.py',
             'src/slm_lab/codenet_checkpoint_v1.py', 'src/slm_lab/codenet_token_pack_v1.py',
             'src/slm_lab/codenet_replay_v1.py', 'src/slm_lab/recurrent_ablation.py',
             'scripts/prepare_codenet_native_contract_pack_v2.py', 'configs/model.json',
             'tests/test_code_residual_checkpoint_v1.py', 'reports/code-residual-checkpoint-v1-focused-tests.log',
             'reports/code-residual-readiness-v1/plan.json']
    bound = {str(ROOT/name):file_hash(ROOT/name) for name in names}
    for p in [source,joint,pack.folder/'manifest.json',pack.folder/'admission.json']:
        bound[str(p)] = file_hash(p)
    bound.update({str(pack.folder/name):h for name,h in pack.manifest['files_sha256'].items()})
    for kind,row in controls:
        if row.get('image'):
            p = Path(row['image']); p = (p if p.is_absolute() else joint.parent/p).resolve()
            bound[str(p)] = file_hash(p)
            if bound[str(p)] != row['stored_image_sha256']:raise ValueError('Original photo bytes changed')
    guard = FileBindings(bound); output.mkdir(); reports.mkdir()
    plan = dict(format='own-code-residual-CUDA-resume-mechanics-v1',recorded_at=datetime.now(timezone.utc).isoformat(),
                source_files_sha256=bound,base_checkpoint_sha256=expected_source,
                selected_complete_original_training_rows=selected,planned_steps=3,checkpoint_after=2,
                actual_Adam_step_executions=4,residual_width=2048,world_size=1,
                GPU='cuda:0',maximum_process_memory_fraction=.20,maximum_checkpoint_bytes=maximum_checkpoint,
                remaining_original_writer_reserve_bytes=reserve,occupied_original_writer=occupied,
                optimizer=dict(name='AdamW',learning_rate=3e-5,weight_decay=.01,foreach=False),
                dtype='FP32-parameters-BF16-autocast',deterministic_algorithms=True,
                no_external_model_weights=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(reports/'plan.json',plan);plan_hash=file_hash(reports/'plan.json')
    def status(message,**fields):
        atomic_json(reports/'status.json',dict(status=message,updated_at=datetime.now(timezone.utc).isoformat(),
                    candidate_promoted=False,goal_achieved=False,**fields))
    status('loading exact own model and fresh residuals')
    random.seed(44171);np.random.seed(44172);torch.manual_seed(44173);torch.cuda.manual_seed_all(44174)
    base,metadata = load_checkpoint(source,'cpu',training_state=False)
    if metadata['tokenizer_sha256'] != file_hash(pack.folder/'tokenizer.json'):
        raise ValueError('Exact own source tokenizer required')
    original_base = state_digest(base.state_dict())
    model = RoutedCodeResidual(base,width=2048).to(device);model.code.gradient_checkpointing=True;model.eval()
    if model.parameter_counts() != dict(total_unique=857007616,frozen_base=706012672,trainable_code=150994944):
        raise ValueError('Actual full own model parameter budget changed')
    token = Tokenizer.from_file(str(pack.folder/'tokenizer.json'));config=ModelConfig.load(ROOT/'configs/model.json')
    if config != model.config:raise ValueError('Actual own source model configuration differs')
    encoder=packed_encoder(pack,token,config);replay_encoder=FullReplayEncoder(token,config,joint.parent,448)
    batches=[replay_encoder.batch([row],'response',device) for kind,row in controls]
    for (kind,row),batch in zip(controls,batches):
        if kind=='image':
            rgb=np.rint((batch['pixel_values'][0].cpu().permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
            if hashlib.sha256(rgb.tobytes()).hexdigest()!=row['model_pixel_sha256_448']:
                raise ValueError('Actual original448px photo control pixels differ')
    def amp():return torch.autocast('cuda',dtype=torch.bfloat16)
    def general_logits(batch):
        with torch.no_grad(),amp():
            return model(batch['input_ids'],pixel_values=batch['pixel_values'],attention_mask=batch['attention_mask'])['logits'].cpu()
    before_logits=[general_logits(batch) for batch in batches]
    cache_ids=torch.tensor([pack.sequence(selected[0]['local_position'])[0][:17]],device=device)
    general_cache=model.base.prefill(cache_ids[:,:16])['cache']
    def new_optimizer():return torch.optim.AdamW(model.adapter_parameters(),lr=3e-5,weight_decay=.01,foreach=False)
    optimizer=new_optimizer()
    def update(step,optimizer):
        return probe_update(model,optimizer,encoder,selected,device,guard,step)
    bindings=dict(base_checkpoint_sha256=expected_source,tokenizer_sha256=file_hash(pack.folder/'tokenizer.json'),
                  run_plan_sha256=plan_hash,planned_steps=3,device=device)
    torch.cuda.reset_peak_memory_stats(device)
    history=[update(1,optimizer),update(2,optimizer)]
    bundle=make_training_state(model,optimizer,**bindings,completed_steps=2,metrics=history)
    if bundle['frozen_backbone_state_sha256']!=original_base:raise ValueError('Original backbone changed before snapshot')
    snapshot=output/'resume.pt';atomic_checkpoint(snapshot,bundle,maximum_checkpoint,reserve)
    snapshot_hash=file_hash(snapshot);del bundle;gc.collect()
    status('complete residual Adam/RNG checkpoint saved; comparing actual next update',completed_steps=2)
    uninterrupted_row=update(3,optimizer)
    expected=make_training_state(model,optimizer,**bindings,completed_steps=3,metrics=history+[uninterrupted_row])
    expected_digest=tree_digest(expected)
    # Restore a CPU-mapped durable snapshot into a newly constructed optimizer.
    optimizer.zero_grad(set_to_none=True);del optimizer;gc.collect();torch.cuda.empty_cache()
    random.seed(987);np.random.seed(654);torch.manual_seed(321);torch.cuda.manual_seed_all(111)
    optimizer=new_optimizer();restored=torch.load(snapshot,map_location='cpu',weights_only=True,mmap=True)
    stale_code_cache=model.code.prefill(cache_ids[:,:16])['cache']
    start,resumed_history=restore_training_state(model,optimizer,restored,**bindings)
    if start!=2 or resumed_history!=history:raise ValueError('Exact original completed budget/history required')
    try:model.code.decode(cache_ids[:,16:],stale_code_cache)
    except ValueError as error:
        if 'parameters changed' not in str(error):raise
    else:raise ValueError('Restored residuals must invalidate older code-route caches')
    resumed_row=update(3,optimizer)
    actual=make_training_state(model,optimizer,**bindings,completed_steps=3,metrics=resumed_history+[resumed_row])
    actual_digest=tree_digest(actual)
    if actual_digest!=expected_digest or resumed_row!=uninterrupted_row:
        raise ValueError('Resumed actual CUDA weights/Adam/RNG/history differ from uninterrupted next update')
    if actual['frozen_backbone_state_sha256']!=original_base:raise ValueError('Original backbone changed during resume')
    for batch,before in zip(batches,before_logits):
        torch.testing.assert_close(general_logits(batch),before,rtol=0,atol=0)
    with torch.no_grad():
        cached=model.base.decode(cache_ids[:,16:],general_cache)['logits']
        full=model.base(cache_ids)['logits'][:,-1:]
    torch.testing.assert_close(cached,full,rtol=1e-4,atol=1e-4)
    guard.check(full=True)
    if file_hash(snapshot)!=snapshot_hash:raise ValueError('Durable resumable checkpoint was modified')
    result=dict(completed_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,run_plan_sha256=plan_hash,
                model_counts=model.parameter_counts(),completed_step=3,actual_GPU_Adam_step_executions=4,
                saved_at_step=2,resumed_from_step=2,uninterrupted_step3=uninterrupted_row,resumed_step3=resumed_row,
                every_updated_residual_tensor_and_Adam_moment_bitwise_equal=True,
                all_Python_NumPy_CPU_CUDA_rng_states_bitwise_equal=True,
                complete_uninterrupted_state_sha256=expected_digest,complete_resumed_state_sha256=actual_digest,
                original_backbone_state_sha256=original_base,all_original_base_tensors_unchanged=True,
                sampled_general_text_and448px_image_logits_bitwise_unchanged=True,
                restored_code_cache_invalidated=True,general_cache_valid_after_resume=True,
                original_response_targets_per_update=382,encoder_truncated_sequences=encoder.truncated,
                checkpoint=str(snapshot),checkpoint_sha256=snapshot_hash,checkpoint_bytes=snapshot.stat().st_size,
                complete_optimizer_and_RNG_serialized=True,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device),
                same_process_new_optimizer_CPU_mapped_durable_restore=True,fresh_process_restart_verified=False,
                capability_or_sustained_training_gain_assessed=False,candidate_promoted=False,goal_achieved=False,
                scope='Actual own857M CUDA/BF16 graph, full adapter/Adam/RNG durable state and repeated stochastic next update; same-process fresh-optimizer restore only. Whole-source sustained training, fresh-process restart, automatic routing, final-base integration and held capability remain unverified.')
    atomic_json(reports/'results.json',result);status('actual full own CUDA optimizer/Adam/RNG next-update resume passed',completed_step=3)
    print('actual_857M_CUDA_full_optimizer_resume_passed',expected_digest,flush=True)


if __name__=='__main__':main()
