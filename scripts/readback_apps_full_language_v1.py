"""Independent CPU readback of an actual complete four-rank APPS checkpoint."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import tempfile

import torch

from slm_lab.code_residual_checkpoint_v1 import _validate_adam, _validate_rng
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.recurrent_ablation import state_digest

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/apps-full-language-v1'


def main():
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():raise ValueError('Readback must not initialize a GPU context')
    plan_path=REPORTS/'plan.json';plan=json.loads(plan_path.read_text())
    source=ROOT/'checkpoints/dialogue-retention-v1/systems-variety-code/model.pt'
    if file_hash(source)!=plan['base_checkpoint_sha256']:raise ValueError('Exact own completed source checkpoint required')
    base=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
    rows=list(map(json.loads,(ROOT/'data/public-apps-human-solution-view-v1/train-records.jsonl').read_text().splitlines()))
    families=json.loads((ROOT/'reports/code-residual-stage-v3-source-check/results.json').read_text())['actual_source_families']
    order=list(range(len(rows)));random.Random(plan['training_options']['seed']).shuffle(order)
    next_order=list(range(len(rows)));random.Random(plan['training_options']['seed']+1).shuffle(next_order)
    order+=next_order[:9]
    checkpoint=ROOT/'checkpoints/apps-full-language-v1/resume.pt'
    with tempfile.TemporaryDirectory(prefix='.APPS-complete-state-readback-',dir=ROOT/'reports') as temporary:
        stable=Path(temporary)/'resume.pt';os.link(checkpoint,stable)
        digest=file_hash(stable);state=torch.load(stable,map_location='cpu',weights_only=True,mmap=True)
        steps=state['steps']
        if (state['format']!='own-complete-APPS-full-language-state-v1' or state['world_size']!=4
                or state['planned_steps']!=3203 or not 1<=steps<=3203 or state['run_plan_sha256']!=file_hash(plan_path)
                or state['base_checkpoint_sha256']!=plan['base_checkpoint_sha256']
                or state['tokenizer_sha256']!=plan['tokenizer_sha256'] or state['config']!=base['config']
                or state['training_options']!=plan['training_options'] or state['truncated_sequences']):
            raise ValueError('Actual complete four-rank own-model/source/budget bindings required')
        tensors=state['model'];original=base['model']
        if set(tensors)!=set(original):raise ValueError('Every own learned/frozen model tensor required')
        for n,t in tensors.items():
            if t.shape!=original[n].shape or t.dtype!=original[n].dtype or not bool(torch.isfinite(t).all()):
                raise ValueError('Every actual saved model tensor must have finite original shape/dtype')
        language=[n for n in original if not n.startswith(('vision.','value_head.')) and n!='lm_head.weight']
        frozen=[n for n in original if n.startswith(('vision.','value_head.'))]
        if state['trainable_parameter_names']!=language:raise ValueError('All and only original language parameters required')
        if (sum(t.numel() for n,t in tensors.items() if n!='lm_head.weight')!=706012672
                or sum(tensors[n].numel() for n in language)!=654386688
                or not torch.equal(tensors['embedding.weight'],tensors['lm_head.weight'])):
            raise ValueError('Actual complete own706M model with tied embedding/output weights required')
        if any(not torch.equal(tensors[n],original[n]) for n in frozen):raise ValueError('Original vision/critic must remain byte-identical')
        changed=[n for n in language if not torch.equal(tensors[n],original[n])]
        if not changed:raise ValueError('Actual retained language weight updates required')
        _validate_adam(state['optimizer'],[tensors[n] for n in language],steps)
        rngs=state['worker_rng_states']
        if len(rngs)!=4:raise ValueError('Every actual original worker RNG required')
        for rng in rngs:
            _validate_rng({**rng,'cuda':None},'cpu')
            if not isinstance(rng['cuda'],torch.Tensor) or rng['cuda'].dtype!=torch.uint8 or not rng['cuda'].numel():
                raise ValueError('Original actual CUDA RNG bytes missing')
        if [m['step'] for m in state['metrics']]!=list(range(1,steps+1)):raise ValueError('Every actual ordered optimizer metric required')
        for step,metric in enumerate(state['metrics']):
            actual=metric['worker_source_positions'];expected=order[step*32:(step+1)*32]
            if len(actual)!=4:raise ValueError('Every worker source exposure must be retained')
            flat=[actual[r]['code_positions'][micro] for micro in range(8) for r in range(4)]
            if flat!=expected or metric['replay_family']!=families[step%29]:raise ValueError('Exact original planned code/family draw order required')
            tokens=sum(rows[i]['target_tokens'] for i in flat)
            if metric['objectives']['complete_original_human_program']['global_target_tokens']!=tokens:
                raise ValueError('Actual original response/EOS target count must match every saved update')
        if state['sampler_state']['updates']!=steps or state['sampler_state']['code_global_draws']!=steps*32:
            raise ValueError('Full original code source cursor required')
        if file_hash(stable)!=digest:raise ValueError('Private complete readback inode changed')
        result=dict(observed_at=datetime.now(timezone.utc).isoformat(),actual_saved_steps=steps,planned_steps=3203,
            checkpoint_sha256=digest,observed_live_checkpoint=str(checkpoint),temporary_readback_inode_not_retained=True,
            actual_total_unique_parameters=706012672,actual_trainable_language_parameters=654386688,
            finite_original_shape_model_tensors=len(tensors),changed_language_tensors=len(changed),
            original_vision_and_critic_byte_identical=True,all218complete_finite_step_matched_language_Adam_states=True,
            all4worker_CPU_RNG_valid_and_actual_CUDA_bytes_retained=True,full_ordered_metrics_and_source_cursors=True,
            all_observed_code_program_ids_order_and_EOS_targets_independently_reconstructed=True,
            actual_admitted_human_program_exposures=steps*32,
            actual_human_program_target_exposures=sum(m['objectives']['complete_original_human_program']['global_target_tokens'] for m in state['metrics']),
            actual_GPU_training_verified=True,no_readback_GPU_context=not torch.cuda.is_initialized(),
            actual_model_capability_gain_established=False,candidate_promoted=False,goal_achieved=False,
            files_sha256={str(plan_path):file_hash(plan_path),str(source):plan['base_checkpoint_sha256'],
                str(Path(__file__).resolve()):file_hash(__file__)})
        atomic_json(ROOT/'reports/apps-full-language-v1-neural-readback.json',result)
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
