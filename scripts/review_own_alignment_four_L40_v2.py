"""CPU readback of all actual four-device extension/Adam/RNG receipts."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import torch

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.code_residual_checkpoint_v1 import _validate_adam, _validate_rng
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.own_aligned_assistant_v2 import FORMAT
from slm_lab.whole_human_data_v2 import WholeHumanData

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/own-alignment-four-L40-v2'


def nested_digest(value):
    digest=hashlib.sha256()
    def visit(item):
        if isinstance(item,torch.Tensor):
            tensor=item.detach().cpu().contiguous()
            digest.update(json.dumps(['tensor',str(tensor.dtype),list(tensor.shape)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item,dict):
            digest.update(b'dict')
            for key in sorted(item,key=lambda k:(str(type(k)),str(k))):
                visit(key);visit(item[key])
        elif isinstance(item,(tuple,list)):
            digest.update(str(type(item)).encode())
            for part in item:visit(part)
        else:
            digest.update(json.dumps(item,allow_nan=False).encode())
    visit(value)
    return digest.hexdigest()


def main():
    destination=REPORTS/'independent-complete-readback.json'
    if destination.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():
        raise ValueError('Fresh GPU-hidden CPU readback required')
    torch.set_num_threads(2)
    plan=json.loads((REPORTS/'plan.json').read_text());complete=json.loads((REPORTS/'complete.json').read_text())
    files={**plan['source_files_sha256'],str(REPORTS/'plan.json'):file_hash(REPORTS/'plan.json'),
        str(REPORTS/'complete.json'):file_hash(REPORTS/'complete.json')}
    guard=FileBindings(files)
    data=WholeHumanData(ROOT/'data/whole-human-alignment-v2','dialogue','train','audit')
    proof_by_device={p['physical_GPU']:p for p in complete['proofs']}
    if set(proof_by_device)!=set(range(4)):
        raise ValueError('Every actual original physical L40 proof required')
    states=[];reviews=[]
    for shard in range(4):
        proof=proof_by_device[shard];path=REPORTS/f'capacity-{shard}-resume.pt'
        if file_hash(path)!=proof['actual_resume_sha256'] or path.stat().st_size!=proof['actual_resume_bytes']:
            raise ValueError('Every whole actual saved CUDA state must match its retained bytes')
        state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        payload=state['model'];parts=payload['extensions'];names=state['optimized_names']
        if (payload['format']!=FORMAT or payload['settings']['word_frame_expansion']!=5
                or payload['actual_counts']['total_unique']!=999860418
                or sum(t.numel() for t in parts.values())!=13409474 or len(parts)!=600
                or len(names)!=600 or len(set(names))!=600 or set(names)!=set(parts)
                or state['steps']!=1 or state['planned_steps']!=2 or state['world_size']!=1
                or state['metrics']!=[proof['initial_source_extreme_update']]
                or state['sampler_state']['original_source_extremes']!=plan['source_extremes']
                or payload['sources']['run_plan_sha256']!=file_hash(REPORTS/'plan.json')
                or payload['sources']['source_admission_sha256']!=file_hash(ROOT/'data/whole-human-alignment-v2/admission.json')
                or any(any(k in name for k in ('original','embedding','lm_head','base.')) for name in parts)
                or any(not bool(torch.isfinite(t).all()) for t in parts.values())):
            raise ValueError('Exact complete source-bound new-only whole-model inventory and optimizer order required')
        _validate_adam(state['optimizer'],[parts[n] for n in names],1)
        # Workers validated actual private CUDA generator state before mutation.
        # Here inspect its complete saved bytes and independently validate the
        # Python/NumPy/CPU states without initializing a GPU in this reviewer.
        cuda=state['rng']['cuda']
        if not isinstance(cuda,torch.Tensor) or cuda.dtype!=torch.uint8 or cuda.ndim!=1 or not cuda.numel():
            raise ValueError('Actual private-device CUDA RNG bytes must be retained')
        _validate_rng({**state['rng'],'cuda':None},'cpu')
        for objective in proof['initial_source_extreme_update']['objectives']:
            if objective['kind']=='whole_human_SFT':
                native=data.rows[objective['source_position']]['native']['response']
                if (native['length']!=objective['context'] or native['length']-native['target_start']!=objective['targets']):
                    raise ValueError('Every actual longest whole training source context/target count must match its source')
        if (proof['complete_Adam_states']!=600 or proof['actual_counts']!=payload['actual_counts']
                or not proof['actual_CUDA_next_update_all_metrics_new_tensors_Adam_and_RNG_exact']
                or not proof['all_original_bytes_versions_native_probes_and_code_cache_exact']
                or proof['initial_source_extreme_update']['actual_CTC_frames']!=140):
            raise ValueError('Every actual full-graph CUDA/recovery/original-retention check must pass')
        states.append(nested_digest(dict(parts=parts,Adam=state['optimizer'],RNG=state['rng'])))
        reviews.append(dict(physical_GPU=shard,new_tensors=600,actual_Adam_states=600,
            actual_full_unique_parameters=payload['actual_counts']['total_unique'],
            actual_peak_GPU_allocated_bytes=proof['peak_GPU_allocated_bytes'],
            parts_Adam_RNG_digest=states[-1],actual_saved_state_sha256=file_hash(path)))
    guard.check(full=True)
    report=dict(completed_at=datetime.now(timezone.utc).isoformat(),source_reader_sha256=file_hash(Path(__file__).resolve()),
        source_plan_sha256=file_hash(REPORTS/'plan.json'),actual_completed_GPU_check_sha256=file_hash(REPORTS/'complete.json'),
        all_original_sources_and_executed_kernels_and_4_saved_states_rehashed=True,actual_saved_state_reviews=reviews,
        all4saved_new_weights_Adam_and_RNG_bytes_identical=len(set(states))==1,
        actual_4_GPU_next_update_and_original_route_retention_proofs_checked=True,
        every_source_extreme_whole_context_and_response_target_count_reconciled=True,
        CUDA_private_generator_validation_performed_by_actual_workers=True,CPU_reader_did_not_initialize_CUDA=True,
        new_model_or_neural_forward_loaded_by_this_reader=False,CTC_reduction_on_CPU_qualification_retained=True,
        disposable_stochastic_critic_target_not_RLHF_return=True,production_updates=0,
        actual_new_held_quality_gain_not_established=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(destination,report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
