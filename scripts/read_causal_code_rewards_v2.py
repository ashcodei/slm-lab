"""Independent output-derived checks and fresh confined no-op replays."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import copy
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('All GPUs must be hidden')
    from slm_lab.causal_code_reward_v2 import pass_ablation
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.documented_doctest_v1 import contract,compare,execute
    from slm_lab.public_example_code_reward_v1 import PublicExampleCodeReward
    folder=ROOT/'reports/causal-code-rewards-v2';report=folder/'independent-reader.json'
    if report.exists() or (folder/'admission.json').exists():raise ValueError('Preserve earlier reward admission/readback')
    prepared=json.loads((folder/'prepared.json').read_text());catalog=PublicExampleCodeReward(ROOT)
    observations=list(map(json.loads,(folder/'observations.jsonl').open()))
    if file_hash(folder/'observations.jsonl')!=prepared['observations_sha256']:raise ValueError('Raw control bytes changed')
    if [r['source_id'] for r in observations]!=sorted(catalog.candidates):raise ValueError('All admitted original source identities required')
    bindings=dict(prepared['source_files_sha256'])
    for p in [Path(__file__),folder/'prepared.json',folder/'observations.jsonl']:bindings[str(p)]=file_hash(p)
    guard=FileBindings(bindings);started=time.monotonic();supported=0;post_checks=0;replays=[]
    for row in observations:
        key=row['source_id'];c=catalog.candidates[key];control=row['ablation'];examples=contract(c)
        if control['original_program_sha256']!=c['program_sha256'] or control['source_id']!=key:raise ValueError('Original source identity required')
        if not control['supported']:continue
        actual=control['counterfactual']['raw_machine_mutant_execution'];flags=[compare(e,r) for e,r in zip(examples,actual['raw_results'])]
        indices=[i for i,x in enumerate(flags) if not x]
        if (len(flags)!=len(examples) or flags!=actual['example_passes'] or indices!=control['dependent_example_indices']
                or not indices or not actual['all_original_examples_observed']):raise ValueError('Actual whole stream causal checks must reconstruct')
        noop=pass_ablation(c)
        if hashlib.sha256(noop.encode()).hexdigest()!=control['ablation_program_sha256']:raise ValueError('Exact full-body counterfactual required')
        for name,want in [('original_publisher_program_control',1.),('deterministic_no_op_control',0.)]:
            observation=row[name];raw=observation['raw_candidate_execution'];passes=[compare(e,r) for e,r in zip(examples,raw['raw_results'])]
            if (len(passes)!=len(examples) or not raw['all_original_examples_observed'] or passes!=raw['example_passes']):raise ValueError('Every actual reward-control example must reconstruct')
            calls=sum(r['entry_body_calls'] for r in raw['raw_results']);full=all(passes)
            reward=(1. if full else .9*sum(passes[i] for i in indices)/len(indices)) if calls else 0.
            if reward!=want or observation['reward']!=reward or observation['full_original_stream_passed']!=full:
                raise ValueError('Actual original/no-op outcomes and numerical rewards must reconstruct')
        supported+=1;post_checks+=len(control['post_call_dependent_check_indices']);replays.append((key,noop,indices))
    def replay(item):
        key,noop,indices=item;c=copy.deepcopy(catalog.candidates[key]);formatted='```python\n'+noop+'\n```'
        digest=lambda x:hashlib.sha256(x.encode()).hexdigest()
        c['source']=dict(c['source'],id='independent-counterfactual:'+key,response=formatted,annotation_origin='Independent machine no-op control; not policy/human feedback')
        c.update(program=noop,program_sha256=digest(noop),existing_formatted_source_target_sha256=digest(formatted))
        result=execute(c,seconds=5);examples=contract(c);flags=[compare(e,r) for e,r in zip(examples,result.get('raw_results',[]))]
        if (not result.get('all_original_examples_observed') or len(flags)!=len(examples)
                or [i for i,x in enumerate(flags) if not x]!=indices):raise ValueError('Fresh complete counterfactual replay must reproduce causal checks')
        return dict(source_id=key,raw_counterfactual_execution=result)
    replay_path=folder/'independent-no-op-replays.jsonl'
    with replay_path.open('x') as f,ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(replay,replays):f.write(json.dumps(result,ensure_ascii=False)+'\n')
    if supported!=prepared['reward_informative_functions'] or post_checks!=prepared['post_call_value_or_side_effect_checks']:raise ValueError('Whole population/control counts differ')
    guard.check(full=True)
    result=dict(at=datetime.now(timezone.utc).isoformat(),all_source_identities_and_raw_control_rewards_reconstructed=True,
        complete_original_functions=len(observations),reward_informative_functions=supported,
        independently_reexecuted_complete_no_ops=supported,post_call_dependent_checks=post_checks,
        raw_replays_sha256=file_hash(replay_path),source_files_sha256=bindings,seconds=time.monotonic()-started,
        production_RL_updates=0,human_preferences=0,quality_gain_proven=False,goal_achieved=False)
    atomic_json(report,result)
    atomic_json(folder/'admission.json',dict(reward_ledger_sha256=file_hash(folder/'observations.jsonl'),
        independent_reader_sha256=file_hash(report),reward_scoring_admitted=True,source_files_sha256=prepared['source_files_sha256'],
        production_RL_started=False,goal_achieved=False))
    print(json.dumps({k:v for k,v in result.items() if k!='source_files_sha256'}),flush=True)


if __name__=='__main__':main()
