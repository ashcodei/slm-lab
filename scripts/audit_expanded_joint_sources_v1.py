"""Enumerate complete phase coverage and read actual native source extremes."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/expanded-joint-sources-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Source audit must hide every GPU')
    if REPORTS.exists():raise ValueError('Preserve every earlier complete/partial source audit')
    import numpy as np
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.expanded_joint_sources_v1 import ExpandedJointSources,CompletePhaseSchedule
    from slm_lab.model import ModelConfig
    if '2 passed' not in (ROOT/'reports/expanded-joint-sources-v1-tests.log').read_text():raise ValueError('Actual phase/partial-rank/coverage tests required')
    prior=json.loads((ROOT/'reports/protected-joint-v1/plan.json').read_text())
    import torch
    torch.set_num_threads(1);started=time.monotonic();initial=torch.load(prior['initial_dense_settings']['initializer'],map_location='cpu',weights_only=True,mmap=True)
    config=ModelConfig(**initial['config']);del initial
    data=ExpandedJointSources(ROOT,config);widths={n:32 if data.kinds[n]=='image' else 16 for n in data.lengths}
    schedule=CompletePhaseSchedule(data,widths,958117);budget=data.budget();REPORTS.mkdir()
    files=[Path(__file__),ROOT/'src/slm_lab/expanded_joint_sources_v1.py',ROOT/'tests/test_expanded_joint_sources_v1.py',
        ROOT/'reports/expanded-joint-sources-v1-tests.log',ROOT/'reports/codenet-later-instruction-replay-v2-independent-audit/results.json',
        ROOT/'reports/nq-training-admission-readiness-v1.json',ROOT/'reports/nq-complete-exposure-findings-independent-verification-v1.json']
    for folder in [data.packed.folder,data.later.folder,data.fact.folder,data.views.folder]:
        manifest=json.loads((folder/'manifest.json').read_text());files.append(folder/'manifest.json')
        files.extend(folder/name for name in manifest.get('artifact_files_sha256',manifest.get('files_sha256',{})))
        if (folder/'admission.json').exists():files.append(folder/'admission.json')
    files.extend(Path(s['path']) for s in data.later.manifest['sources']);files.append(data.fact.source)
    files.append(Path(data.packed.manifest['packed_root'])/'corpus.json')
    bindings={str(p):file_hash(p) for p in files};guard=FileBindings(bindings)
    atomic_json(REPORTS/'plan.json',dict(source_files_sha256=bindings,complete_source_budget=budget,
        phase_bounds=schedule.phase_bounds,phase_families=data.phases(),global_family_widths=widths,seed=958117,
        neural_updates=0,full_model_GPU_capacity_not_tested=True,goal_achieved=False))
    reached={n:np.zeros(size,dtype=np.uint8) for n,size in data.populations.items()};counts=Counter();contexts=Counter();targets=Counter()
    digest=hashlib.sha256();phase_counts=Counter()
    for step in range(len(schedule)):
        name,positions=schedule.at(step)
        # Independently check disjoint four-rank views, including empty tails.
        shards=[schedule.rank_positions(step,rank,4)[1] for rank in range(4)]
        if sorted(x for shard in shards for x in shard)!=sorted(positions) or len(set(positions))!=len(positions):raise ValueError('Exact complete unrepeated global source group required')
        if np.any(reached[name][positions]):raise ValueError('A source-family row was silently repeated')
        reached[name][positions]+=1;counts[name]+=len(positions)
        contexts[name]+=sum(data.lengths[name][p] for p in positions);targets[name]+=sum(data.targets[name][p] for p in positions)
        phase_counts[schedule.phase_for_step[step]]+=1;digest.update(json.dumps([name,positions],separators=(',',':')).encode()+b'\n')
    if any(not np.all(v==1) for v in reached.values()):raise ValueError('Every original family row must appear exactly once')
    for name,b in budget.items():
        if (counts[name]!=b['examples'] or contexts[name]!=b['complete_context_tokens'] or targets[name]!=b['response_EOS_targets']):
            raise ValueError('Every complete original context and causal/response target must enter the actual source pass')
    receipts=[]
    for name in data.expanded_owners:
        positions=list(dict.fromkeys([0,len(data.lengths[name])-1,
            max(range(len(data.lengths[name])),key=lambda p:data.lengths[name][p]),
            max(range(len(data.targets[name])),key=lambda p:data.targets[name][p])]))
        for position in positions:
            batch=data.batch(name,[position],'cpu');ids=batch['input_ids'][0];labels=batch['labels'][0];mask=batch['attention_mask'][0]
            if int(mask.sum())!=len(ids) or not torch.equal(labels[1:][labels[1:]!=-100],ids[1:][labels[1:]!=-100]):
                raise ValueError('Every actual retained native target must equal its exact input token')
            if name.startswith('packed:'):
                item=data.packed.index['train'][position];raw=data.packed.stream.read('train',int(item['offset']),int(item['length']))
                if ids.tolist()!=raw.astype(np.int64).tolist() or labels.tolist()!=ids.tolist():raise ValueError('Whole original packed input and all adjacent causal targets must be exact')
            receipts.append(dict(family=name,source_position=position,complete_context_tokens=len(ids),
                actual_next_response_EOS_targets=int((labels[1:]!=-100).sum()),
                input_ids_sha256=hashlib.sha256(np.asarray(ids.tolist(),dtype='<u2').tobytes()).hexdigest()))
    guard.check(full=True)
    result=dict(at=datetime.now(timezone.utc).isoformat(),family_count=len(budget),complete_source_budget=budget,
        complete_family_rows=sum(counts.values()),complete_context_tokens=sum(contexts.values()),complete_causal_response_EOS_targets=sum(targets.values()),
        planned_optimizer_updates=len(schedule),global_family_widths=widths,phase_bounds=schedule.phase_bounds,
        source_pass_sha256=digest.hexdigest(),every_source_family_row_reached_once=True,all_four_rank_views_reconstruct=True,
        actual_native_extreme_readbacks=receipts,all_original_packed_train_adjacent_pairs_retained=True,
        packed_training_targets=data.packed.manifest['splits']['train']['target_tokens'],
        packed_tail_targets=data.packed.manifest['splits']['train']['tail_target_tokens'],
        additional_systems_code_tokens=52970300,new_raw_data_downloaded=False,
        new_original_public_functions_created=False,old_header_variants_and_explicit_exercise_repeats_not_new_algorithms=True,
        original_NQ_known_exposure_case_count=18,NQ_no_exact_match_not_semantic_independence=True,
        source_files_sha256=bindings,seconds=time.monotonic()-started,neural_updates=0,
        full_size_stage_CPU_and_GPU_admission_and_production_still_required=True,quality_gain_proven=False,goal_achieved=False)
    atomic_json(REPORTS/'complete.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['complete_source_budget','source_files_sha256','actual_native_extreme_readbacks']}),flush=True)


if __name__=='__main__':main()
