"""Fresh CUDA process must reproduce the sealed uninterrupted next update."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from tokenizers import Tokenizer
import torch

from check_code_residual_cuda_resume_v1 import probe_update, tree_digest
from prepare_codenet_native_contract_pack_v2 import occupied_writer_reservation
from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.code_residual_checkpoint_v1 import make_training_state, restore_training_state
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_replay_v1 import FullReplayEncoder
from slm_lab.codenet_token_pack_v1 import CompleteCodePack, packed_encoder
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.training import load_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def main():
    reports=ROOT/'reports/code-residual-fresh-process-resume-v1'
    if reports.exists():raise ValueError('Preserve every earlier fresh-process report')
    original_launch=json.loads((ROOT/'reports/code-residual-cuda-resume-v1-launch.json').read_text())
    if os.getpid()==original_launch['pid']:raise ValueError('A distinct fresh Python/CUDA process is required')
    process=Path('/proc')/str(original_launch['pid'])
    if process.exists():
        stat=(process/'stat').read_text();parts=stat[stat.rfind(')')+2:].split()
        if parts[19]==original_launch['start_ticks'] and parts[0]!='Z':
            raise ValueError('Original GPU optimizer-check process must be terminal before fresh restart')
    parent_results=ROOT/'reports/code-residual-cuda-resume-v1/results.json'
    previous=json.loads(parent_results.read_text())
    if (previous['every_updated_residual_tensor_and_Adam_moment_bitwise_equal'] is not True
            or previous['all_Python_NumPy_CPU_CUDA_rng_states_bitwise_equal'] is not True):
        raise ValueError('Completed actual uninterrupted-reference/resume evidence required')
    plan_path=ROOT/'reports/code-residual-cuda-resume-v1/plan.json'
    if file_hash(plan_path)!=previous['run_plan_sha256']:raise ValueError('Original exact update plan changed')
    plan=json.loads(plan_path.read_text());snapshot=Path(previous['checkpoint'])
    if file_hash(snapshot)!=previous['checkpoint_sha256']:raise ValueError('Original durable full checkpoint changed')
    occupied=occupied_writer_reservation();reserve=26*1024**3-occupied['credited_bytes']+256*1024**2
    if shutil.disk_usage(ROOT).free<reserve+64*1024**2:raise ValueError('Original writer reserve and report budget required')
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info(device)[0]<12*1024**3:
        raise ValueError('Initially12GiB free on shared BF16 GPU required')
    torch.cuda.set_per_process_memory_fraction(.20,device);torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    bound={**plan['source_files_sha256'],str(plan_path):previous['run_plan_sha256'],
           str(parent_results):file_hash(parent_results),str(snapshot):previous['checkpoint_sha256'],
           str(Path(__file__)):file_hash(Path(__file__))}
    guard=FileBindings(bound);reports.mkdir()
    def status(message,**fields):
        atomic_json(reports/'status.json',dict(status=message,updated_at=datetime.now(timezone.utc).isoformat(),
                    candidate_promoted=False,goal_achieved=False,**fields))
    status('fresh process loading exact own base and full adapter/Adam/RNG checkpoint')
    source=ROOT/'checkpoints/extended-curriculum-v1/language-pretraining/model.pt'
    base,metadata=load_checkpoint(source,'cpu',training_state=False)
    model=RoutedCodeResidual(base,width=2048).to(device);model.code.gradient_checkpointing=True;model.eval()
    pack=CompleteCodePack(ROOT/'data/codenet-durable-pack-v2','train',purpose='training')
    token=Tokenizer.from_file(str(pack.folder/'tokenizer.json'))
    if metadata['tokenizer_sha256']!=file_hash(pack.folder/'tokenizer.json'):raise ValueError('Exact own tokenizer required')
    encoder=packed_encoder(pack,token,model.config)
    # Retain actual general text/image controls before and after the fresh update.
    prior=json.loads((ROOT/'reports/code-residual-readiness-v1/plan.json').read_text())
    ids={r['id'] for r in prior['original_training_control_ids']};controls=[]
    joint=ROOT/'data/extended-curriculum-v1/joint.train.jsonl'
    for line in joint.open():
        row=json.loads(line)
        if row['id'] in ids:controls.append(row)
        if len(controls)==len(ids):break
    if len(controls)!=2 or sum(bool(r.get('image')) for r in controls)!=1:
        raise ValueError('Both original whole text and image control rows required')
    replay_encoder=FullReplayEncoder(token,model.config,joint.parent,448)
    batches=[replay_encoder.batch([row],'response',device) for row in controls]
    def general_logits(batch):
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            return model(batch['input_ids'],pixel_values=batch['pixel_values'],attention_mask=batch['attention_mask'])['logits'].cpu()
    before=[general_logits(batch) for batch in batches]
    optimizer=torch.optim.AdamW(model.adapter_parameters(),lr=3e-5,weight_decay=.01,foreach=False)
    saved=torch.load(snapshot,map_location='cpu',weights_only=True,mmap=True)
    bindings=dict(base_checkpoint_sha256=plan['base_checkpoint_sha256'],
                  tokenizer_sha256=file_hash(pack.folder/'tokenizer.json'),
                  run_plan_sha256=previous['run_plan_sha256'],planned_steps=3,device=device)
    torch.cuda.reset_peak_memory_stats(device)
    start,history=restore_training_state(model,optimizer,saved,**bindings)
    if start!=2:raise ValueError('Original exact completed step2 state required')
    status('full optimizer/RNG state restored in fresh process',resumed_from_step=start)
    row=probe_update(model,optimizer,encoder,plan['selected_complete_original_training_rows'],device,guard,3)
    actual=make_training_state(model,optimizer,**bindings,completed_steps=3,metrics=history+[row])
    digest=tree_digest(actual)
    if digest!=previous['complete_uninterrupted_state_sha256'] or row!=previous['uninterrupted_step3']:
        raise ValueError('Fresh-process actual residual/Adam/RNG next update differs from sealed uninterrupted reference')
    if actual['frozen_backbone_state_sha256']!=previous['original_backbone_state_sha256']:
        raise ValueError('Original own frozen backbone changed in fresh process')
    for batch,original in zip(batches,before):torch.testing.assert_close(general_logits(batch),original,rtol=0,atol=0)
    guard.check(full=True)
    result=dict(completed_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,
                original_reference_process_pid=original_launch['pid'],original_reference_process_terminal=True,
                fresh_process_pid=os.getpid(),fresh_process_PID_distinct=True,world_size=1,model_counts=model.parameter_counts(),
                resumed_from_step=2,actual_GPU_Adam_step_executions=1,completed_step=3,
                complete_fresh_state_sha256=digest,sealed_uninterrupted_state_sha256=previous['complete_uninterrupted_state_sha256'],
                every_residual_weight_Adam_moment_and_step_bitwise_equal=True,
                all_Python_NumPy_CPU_CUDA_rng_states_bitwise_equal=True,stochastic_training_record=row,
                all_original_backbone_tensors_unchanged=True,sampled_general_text_and448px_image_logits_bitwise_unchanged=True,
                original_checkpoint=str(snapshot),original_checkpoint_sha256=previous['checkpoint_sha256'],
                durable_complete_checkpoint_bytes=snapshot.stat().st_size,
                peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device),
                fresh_process_optimizer_checkpoint_restart_verified=True,
                sustained_training_or_functional_capability_gain_assessed=False,candidate_promoted=False,goal_achieved=False,
                scope='Actual own857M GPU/BF16 step3 in a separate new Python/CUDA process from the durable step2 residual/Adam/RNG checkpoint; exact match to all sealed uninterrupted weights/moments/RNG/history. Two original train programmes establish restart mechanics only, not sustained whole-source learning or held/general coding quality.')
    atomic_json(reports/'results.json',result);status('actual own857M fresh-process full Adam/RNG CUDA restart passed',completed_step=3)
    print('actual_857M_fresh_process_CUDA_resume_passed',digest,flush=True)


if __name__=='__main__':main()
