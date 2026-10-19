"""Read all four actual warm-up states and reexecute all96native code outputs."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import gzip
import json
import math
import os
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/verified-views-warmup88-readback-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Readback must hide all GPUs')
    if REPORTS.exists():raise ValueError('Preserve previous complete/partial readback')
    import numpy as np
    import torch
    from slm_lab.codenet_token_pack_v1 import CompleteCodePack
    from slm_lab.code_quality import execute
    from slm_lab.code_tasks import TASKS,request
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1);run=ROOT/'reports/verified-views-execution-v1';checkpoints=ROOT/'checkpoints/verified-views-execution-v1'
    plan=json.loads((run/'plan.json').read_text());done=json.loads((run/'warmup-training-complete.json').read_text())
    receipt=json.loads((run/'warmup-warmup-88-distributed-state.json').read_text());journal=list(map(json.loads,(run/'warmup-metrics.jsonl').read_text().splitlines()))
    if done['actual_updates']!=88 or len(journal)!=88 or len(receipt['all_four_private_states'])!=4:raise ValueError('Complete actual88-update four-worker warm-up required')
    initial=torch.load(checkpoints/'initial-actor.pt',map_location='cpu',weights_only=True,mmap=True)
    trained=torch.load(checkpoints/'warmup-actor.pt',map_location='cpu',weights_only=True,mmap=True)
    if (initial['actual_counts']!=trained['actual_counts'] or trained['actual_counts']['actual_unique']!=999706442
            or initial['frozen_dense_sha256']!=trained['frozen_dense_sha256'] or initial['sources']!=trained['sources']):
        raise ValueError('Complete counted own base/source/actor graph must match')
    pack=CompleteCodePack(ROOT/'data/verified-code-views-v1','train','training')
    with gzip.open(pack.folder/'original-rows.jsonl.gz','rt') as f:rows=list(map(json.loads,f))
    mapping={}
    for i,row in enumerate(rows):mapping.setdefault(row['original_source_position'],{})[row['view']]=i
    source_exposures=Counter();view_exposures=Counter();epochs=[];part_digests=[];Adam_digests=[];bindings={};private=[]
    for epoch in range(4):
        order=list(range(173));random.Random(957713+epoch).shuffle(order);CE=KL=0.;targets=context=0
        for j in range(22):
            r=journal[epoch*22+j];functions=order[j*8:(j+1)*8];views=[]
            for p in functions:
                modes=mapping[p];views.extend([modes['original_instruction'],modes['prefix_completion'],modes.get('verified_statement_repair',modes['prefix_completion'])])
            pairs=[pack.sequence(p) for p in views];n=sum(sum(y!=-100 for y in labels) for ids,labels in pairs);c=sum(len(ids) for ids,labels in pairs)
            if (r['step']!=epoch*22+j+1 or r['original_function_positions']!=functions or r['complete_view_positions']!=views
                    or r['response_EOS_targets']!=n or r['complete_context_tokens']!=c or r['complete_presentations']!=len(views)
                    or not math.isclose(r['token_mean_CE'],r['CE_sum']/n,rel_tol=1e-12)
                    or not math.isclose(r['token_mean_reference_KL'],r['KL_sum']/n,rel_tol=1e-12)):
                raise ValueError('Every actual complete native target, source draw and loss denominator must reconstruct')
            source_exposures.update(functions);view_exposures.update(rows[p]['view'] for p in views)
            CE+=r['CE_sum'];KL+=r['KL_sum'];targets+=n;context+=c
        epochs.append(dict(epoch=epoch+1,complete_context_tokens=context,response_EOS_targets=targets,
            online_train_token_mean_CE=CE/targets,online_train_target_reference_KL=KL/targets,
            online_training_statistics_not_independent_quality=True))
    if len(source_exposures)!=173 or set(source_exposures.values())!={4}:raise ValueError('All173functions must appear in each of four epochs')
    for rank,item in enumerate(sorted(receipt['all_four_private_states'],key=lambda a:a['rank'])):
        path=Path(item['path'])
        if rank!=item['rank'] or file_hash(path)!=item['sha256']:raise ValueError('Every exact private rank checkpoint required')
        state=torch.load(path,map_location='cpu',weights_only=True,mmap=True);core=state['core']
        if (core['completed']!=88 or core['planned']!=88 or core['metrics']!=journal
                or state['settings']['replicas']!=4 or state['settings']['replica_rank']!=rank
                or state['private_state_is_not_whole_distributed_checkpoint'] is not True
                or core['model']['frozen_dense_sha256']!=trained['frozen_dense_sha256']):raise ValueError('Actual whole-source private profile and metrics must match')
        parts=core['model']['parts'];part_digests.append(tensor_digest(parts.items()))
        if parts.keys()!=trained['parts'].keys() or any(not torch.equal(t,trained['parts'][k]) or not bool(torch.isfinite(t).all()) for k,t in parts.items()):
            raise ValueError('All native and resumable policy/critic values must agree and be finite')
        names=core['optimized_names'];group=core['optimizer']['param_groups'][0]
        if len(names)!=561 or len(core['optimizer']['state'])!=561 or len(group['params'])!=561:raise ValueError('Every actual selected actor Adam state required')
        for key,name in zip(group['params'],names):
            adam=core['optimizer']['state'][key]
            if (set(adam)!={'step','exp_avg','exp_avg_sq'} or float(adam['step'])!=88 or adam['exp_avg'].shape!=parts[name].shape
                    or adam['exp_avg_sq'].shape!=parts[name].shape or any(not bool(torch.isfinite(t).all()) for t in adam.values())
                    or bool((adam['exp_avg_sq']<0).any())):raise ValueError('Every finite complete Adam moment and exact88clock must match')
        Adam_digests.append(tensor_digest((str(i)+'.'+k,s[k]) for i,s in core['optimizer']['state'].items() for k in ['step','exp_avg','exp_avg_sq']))
        torch.Generator(device='cpu').set_state(core['rng']['torch']);random.Random().setstate(core['rng']['python'])
        private.append(dict(rank=rank,CPU_RNG_valid=True,private_CUDA_RNG_bytes=int(core['rng']['cuda'].numel()),
            actual_CUDA_semantics_previously_proved_by_full_GPU_capacity=True))
        bindings[str(path)]=file_hash(path)
    if len(set(part_digests))!=1 or len(set(Adam_digests))!=1:raise ValueError('Every replica must have identical final weights/Adam')
    changed=sum(not torch.equal(t,initial['parts'][name]) for name,t in trained['parts'].items() if name!='critic.weight')
    if changed==0 or not torch.equal(trained['parts']['critic.weight'],initial['parts']['critic.weight']):raise ValueError('Actual actor must learn while separate critic stays exact')
    task_map={t[0]:t for t in TASKS};jobs=[];scores={}
    for arm in ['initial','warmup']:
        scores[arm]={}
        for frame in ['completion','instruction']:
            path=run/arm/f'code-{frame}.json';report=json.loads(path.read_text())
            if report['generation_budget']!=512 or report['task_count']!=24:raise ValueError('Full unchanged code512protocol required')
            scores[arm][frame]={k:report[k] for k in ['syntax_pass_rate','functional_pass_at_1','efficiency_stress_pass_rate']}
            jobs.extend((arm,frame,row) for row in report['results']);bindings[str(path)]=file_hash(path)
    def replay(item):
        arm,frame,row=item;actual=execute(request(task_map[row['task']],row['language'],row['code']))
        if any(actual.get(k,False)!=row['candidate'].get(k,False) for k in ['syntax_valid','functional_passed']):
            raise ValueError('Actual syntax/functional verdict changed on independent candidate replay')
        return dict(arm=arm,frame=frame,language=row['language'],task=row['task'],actual_execution=actual)
    with ThreadPoolExecutor(max_workers=4) as pool:reexecutions=list(pool.map(replay,jobs))
    for path in [Path(__file__),run/'plan.json',run/'warmup-training-complete.json',run/'warmup-warmup-88-distributed-state.json',checkpoints/'initial-actor.pt',checkpoints/'warmup-actor.pt']:
        bindings[str(path)]=file_hash(path)
    REPORTS.mkdir();atomic_json(REPORTS/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),actual_unique_parameters=999706442,
        actual_logical_updates=88,actual_functions=173,original_function_exposures=692,complete_presentations=2076,view_exposures=dict(view_exposures),
        actual_targets=sum(r['response_EOS_targets'] for r in epochs),actual_epoch_training_statistics=epochs,
        changed_actor_tensors=changed,all_four_native_actor_Adam_states_equal=True,all561Adam_clocks_are88=True,
        private_RNG_readback=private,all96code_outputs_independently_reexecuted=True,actual_code_reexecutions=reexecutions,
        matched_full512_code_scores=scores,training_loss_not_a_capability_gain=True,neural_backwards_not_replayed=True,
        native_coding_gain_proven=False,candidate_promoted=False,source_files_sha256=bindings,goal_achieved=False))
    print(json.dumps(dict(updates=88,changed_actor_tensors=changed,epochs=epochs,code_scores=scores)),flush=True)


if __name__=='__main__':main()
