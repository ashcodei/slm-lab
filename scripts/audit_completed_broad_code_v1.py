"""Read actual completed706M code weights and all936 original training metrics."""
from datetime import datetime,timezone
import json
import math
from pathlib import Path

import torch

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(2)
    folder=ROOT/'checkpoints/dialogue-retention-v1/broad-code'
    model_path=folder/'model.pt';base_path=ROOT/'checkpoints/extended-curriculum-v1/joint/model.pt'
    report_folder=ROOT/'reports/dialogue-retention-v1'
    paths=[model_path,base_path,folder/'training.json',folder/'progress.json',folder/'tokenizer.json',
        report_folder/'broad-code-plan.json',report_folder/'broad-code-actual-target-receipt.json',
        report_folder/'broad-code-vision-preservation.json',report_folder/'broad-code-completed-state-cleanup.json',
        ROOT/'data/broad-code-continuation-v1/packed/corpus.json',Path(__file__).resolve()]
    hashes={str(p):file_hash(p) for p in paths}
    state=torch.load(model_path,map_location='cpu',weights_only=True,mmap=True)
    base=torch.load(base_path,map_location='cpu',weights_only=True,mmap=True)
    train=json.loads((folder/'training.json').read_text());progress=json.loads((folder/'progress.json').read_text())
    plan=json.loads((report_folder/'broad-code-plan.json').read_text())
    receipt=json.loads((report_folder/'broad-code-actual-target-receipt.json').read_text())
    preserved=json.loads((report_folder/'broad-code-vision-preservation.json').read_text())
    retired=json.loads((report_folder/'broad-code-completed-state-cleanup.json').read_text())
    if (state['steps']!=936 or state['planned_steps']!=936 or state['world_size']!=4 or state['stage']!='pretrain'
            or state['trainable']!='language' or state['truncated_sequences']!=0 or train['metrics']!=state['metrics']
            or state['config']!=base['config'] or state['tokenizer_sha256']!=base['tokenizer_sha256']
            or state['tokenizer_sha256']!=file_hash(folder/'tokenizer.json')
            or state['data_sha256']!=plan['data_sha256'] or state['data_sha256']!=hashes[str(paths[-2])]
            or hashes[str(base_path)]!=plan['base_checkpoint_sha256']):
        raise ValueError('Actual complete original936-update/four-worker code stage required')
    metrics=state['metrics']
    if len(metrics)!=936:raise ValueError('Every actual training update must remain available')
    for i,metric in enumerate(metrics):
        if (metric['step']!=i+1 or metric['global_target_tokens']!=262080
                or any(not math.isfinite(metric[k]) for k in ['loss','learning_rate','step_seconds','global_target_tokens_per_second'])):
            raise ValueError('Complete finite ordered actual metrics required')
    targets=sum(m['global_target_tokens'] for m in metrics)
    if targets!=245306880 or progress['processed_target_tokens']!=targets or receipt['actual_target_exposures']!=targets:
        raise ValueError('Actual full936-update target exposure counts differ')
    if set(state['model'])!=set(base['model']):raise ValueError('Every original own tensor must remain present')
    changed=[]
    for name,tensor in state['model'].items():
        original=base['model'][name]
        if tensor.shape!=original.shape or tensor.dtype!=original.dtype or not bool(torch.isfinite(tensor).all()):
            raise ValueError('Finite unchanged tensor layouts required')
        if not torch.equal(tensor,original):changed.append(name)
    if not changed or any(n.startswith('vision.') or n=='value_head.weight' for n in changed):
        raise ValueError('Language learning must preserve all original vision/critic tensors')
    unique=sum(t.numel() for n,t in state['model'].items() if n!='lm_head.weight')
    if unique!=706012672 or not torch.equal(state['model']['embedding.weight'],state['model']['lm_head.weight']):
        raise ValueError('Actual706M count and tied token/output weights required')
    if (preserved['continued_checkpoint_sha256']!=hashes[str(model_path)]
            or retired['retained_model_sha256']!=hashes[str(model_path)]
            or retired['all_learned_tensors_verified_equal'] is not True or retired['removed'] is not True):
        raise ValueError('Exact original retained-weight/retired-optimizer receipts required')
    torch.Generator().set_state(state['torch_rng_state'])
    for path,digest in hashes.items():
        if file_hash(path)!=digest:raise ValueError('Completed evidence changed during independent readback')
    result=dict(observed_at=datetime.now(timezone.utc).isoformat(),passed=True,
        completed_updates=936,world_size=4,actual_target_exposures=targets,truncated_sequences=0,
        unique_model_parameters=unique,all_learned_tensors_finite=len(state['model']),
        changed_language_tensor_names=changed,every_original_vision_and_critic_tensor_identical=True,
        tied_embedding_output_exact=True,all936_metrics_match_checkpoint_and_report=True,
        actual_retained_checkpoint_sha256=hashes[str(model_path)],original_base_sha256=hashes[str(base_path)],
        new_library_tokens=30711863,prior_code_rehearsal_tokens=214411271,
        completed_optimizer_retired_by_original_controller=True,complete_independent_four_rank_resume_certificate=False,
        new_execution_feedback_used=False,capability_gain_claim=False,goal_achieved=False,files_sha256=hashes,
        scope='Actual completed own706M original broad/library-code training; weights, source identities and target counts. This is not improved correctness, full curriculum completion or a resumable completed Adam state.')
    target=ROOT/'reports/completed-broad-code-state-readback-v1.json';atomic_json(target,result)
    print(json.dumps({k:result[k] for k in ['passed','completed_updates','actual_target_exposures','unique_model_parameters',
        'all_learned_tensors_finite','actual_retained_checkpoint_sha256','goal_achieved']},indent=2))


if __name__=='__main__':main()
