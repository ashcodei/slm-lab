"""Actual full-size text/word joint CPU learning before four-GPU capacity.

Two complete original single-example updates are disposable. The complete
production source/bucket plan is reconstructed separately; CPU width1does not
stand in for the configured multi-worker/microbatch GPU allocation checks.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/fast-joint-full-CPU-v2'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('All GPUs must be hidden for this CPU control')
    if REPORTS.exists():raise ValueError('Preserve previous actual CPU proof attempts')
    import numpy as np
    import torch
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint,FileBindings
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.dense_foundation_inference_v1 import fresh
    from slm_lab.fast_joint_data_v1 import FastJointSources,CompleteBucketSchedule
    from slm_lab.fast_joint_stage_v1 import make_optimizer,parameter_groups,update
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if '5 passed' not in (ROOT/'reports/fast-joint-v1-tests.log').read_text():raise ValueError('Bucket, distributed partition, complete masks and independent CE/CTC gradients must pass')
    settings=json.loads((ROOT/'reports/dense-foundation-full-CPU-v1/plan.json').read_text())['initial_dense_settings']
    files={}
    for name in ['dense-foundation-v1','shared-words-v1','code-initialized-alignment-v1']:
        files.update(json.loads((ROOT/f'reports/{name}-queue.json').read_text())['source_files_sha256'])
    for directory in ['documented-doctest-verified-v1','codenet-replay-v1','codenet-image-replay-v1']:
        for p in (ROOT/'data'/directory).iterdir():
            if p.is_file():files[str(p)]=file_hash(p)
    for name in ['src/slm_lab/fast_joint_data_v1.py','src/slm_lab/fast_joint_stage_v1.py','src/slm_lab/fast_joint_inference_v1.py',
        'src/slm_lab/codenet_image_replay_v1.py','src/slm_lab/codenet_replay_v1.py','src/slm_lab/codenet_runner_v1.py',
        'scripts/probe_fast_joint_full_CPU_v1.py','tests/test_fast_joint_v1.py','reports/fast-joint-v1-tests.log',
        'reports/documented-doctest-verified-v1/independent-complete-source-and-runtime-readback-v1.json']:
        files[str(ROOT/name)]=file_hash(ROOT/name)
    guard=FileBindings(files);guard.check(full=True);REPORTS.mkdir();started=time.monotonic()
    model,_=fresh(settings,'cpu');data=FastJointSources(ROOT,model.config)
    widths={name:(32 if data.kinds[name]=='image' else 16) for name in data.populations}
    schedule=CompleteBucketSchedule(data.lengths,widths,956431)
    coverage={name:[] for name in data.populations};padded=0;context=0
    for step in range(len(schedule)):
        name,positions=schedule.at(step);coverage[name]+=positions
        for rank in range(4):
            local=positions[rank::4]
            if local:padded+=len(local)*max(data.lengths[name][p] for p in local)
        context+=sum(data.lengths[name][p] for p in positions)
    if any(sorted(coverage[n])!=list(range(data.populations[n])) for n in coverage):raise ValueError('Every complete original source exactly once required')
    selected={'program':0,'words':0}
    # Choose the first original position in the actual independently seeded
    # family permutation, before any new model or held score is observed.
    for name in selected:selected[name]=next(positions[0] for family,positions in schedule.entries if family==name)
    options=dict(learning_rates=dict(language=5e-5,vision=1e-4,word=3e-4),betas=[.9,.95],weight_decay=.01,
        gradient_clip=1.,warmup_fraction=.02,cosine_floor=.1,maximum_microbatch=8,maximum_microbatch_tokens=8192,
        ctc_weight=1.,word_native_CE_weight=1.,gradient_bucket_values=16*1024**2,image_size=448)
    atomic_json(REPORTS/'plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        initial_dense_settings=settings,actual_counts=model.counts(),complete_source_budget=data.budget(),
        global_family_batch_widths=widths,production_seed=956431,production_steps=len(schedule),
        complete_train_context_tokens=context,rank_group_padded_text_positions=padded,
        selected_original_CPU_positions=selected,training_options=options,CPU_threads=2,CPU_global_family_width=1,
        all_updates_disposable=True,production_four_GPU_capacity_not_yet_proved=True,goal_achieved=False))
    local=SimpleNamespace(role='train',purpose='optimization',kinds={n:data.kinds[n] for n in selected},
        lengths={n:[data.lengths[n][p]] for n,p in selected.items()},targets={n:[data.targets[n][p]] for n,p in selected.items()})
    local.batch=lambda name,positions,device:data.batch(name,[selected[name] for _ in positions],device)
    small=CompleteBucketSchedule(local.lengths,{n:1 for n in selected},956431)
    optimizer=make_optimizer(model,options)
    groups=parameter_groups(model);chosen={id(p) for g in groups.values() for _,p in g}
    fixed_before=tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in chosen)
    before={n:tensor_digest(g) for n,g in groups.items()}
    random.seed(956431);np.random.seed(956431);torch.manual_seed(956431);records=[]
    for step in range(len(small)):
        records.append(update(model,optimizer,local,small,step,options))
        print(json.dumps(records[-1]),flush=True)
    after={n:tensor_digest(g) for n,g in groups.items()}
    if any(before[n]==after[n] for n in before) or fixed_before!=tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in chosen):
        raise ValueError('Actual full language, vision and word groups must learn while all other heads remain exact')
    source=dict(joint_CPU_plan_sha256=file_hash(REPORTS/'plan.json'));model.eval()
    snapshot=model.export(source)
    ids=torch.tensor([[1,4,5,2]]);photo=data.words.batch([selected['words']],'cpu')['pixel_values']
    with torch.no_grad():expected_text=model(ids)['logits'];expected_image=model(ids,photo)['logits'];expected_word=model.word_logits(photo)
    cold,_=fresh(settings,'cpu');cold.import_export(snapshot,expected_sources=source)
    with torch.no_grad():
        if not (torch.equal(expected_text,cold(ids)['logits']) and torch.equal(expected_image,cold(ids,photo)['logits'])
                and torch.equal(expected_word,cold.word_logits(photo))):raise ValueError('Actual trained native text/image/word cold load must be exact')
    atomic_checkpoint(REPORTS/'disposable-trained-model.pt',snapshot,5*1024**3,26*1024**3)
    guard.check(full=True)
    atomic_json(REPORTS/'complete.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),actual_counts=model.counts(),
        actual_selected_parameter_values={n:sum(p.numel() for _,p in g) for n,g in groups.items()},
        all_language_vision_word_groups_changed=True,all_other_heads_exact=True,actual_trained_native_text_image_word_cold_reload_exact=True,
        original_complete_source_coverage_checked=True,complete_source_budget=data.budget(),production_steps=len(schedule),
        complete_train_context_tokens=context,rank_group_padded_text_positions=padded,actual_CPU_records=records,
        complete_Adam_states=len(optimizer.state),all_updates_disposable=True,CPU_width1_not_four_GPU_capacity=True,
        production_updates=0,checkpoint_sha256=file_hash(REPORTS/'disposable-trained-model.pt'),seconds=time.monotonic()-started,goal_achieved=False))


if __name__=='__main__':main()
