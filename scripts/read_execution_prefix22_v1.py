"""Seal and replay actual own responses from a matched22-update prefix."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/execution-prefix22-readback-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU reward readback must hide all GPUs')
    if REPORTS.exists():raise ValueError('Preserve earlier complete/partial replay')
    from slm_lab.causal_code_reward_catalog_v1 import CausalPublicCodeReward
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.execution_group_controlled_v2 import advantages
    from tokenizers import Tokenizer
    run=ROOT/'reports/verified-views-execution-v1';sealed={};jobs=[];catalog=CausalPublicCodeReward(ROOT)
    token=Tokenizer.from_file(str(ROOT/'data/documented-doctest-verified-v1/tokenizer.json'))
    for arm in ['verified_solution','execution_hybrid']:
        raw=(run/(arm+'-metrics.jsonl')).read_text();rows=list(map(json.loads,raw[:raw.rfind('\n')+1].splitlines()))[:22]
        if len(rows)!=22:raise ValueError('Both actual arms must complete the same22updates')
        sealed[arm]=rows
        for i,r in enumerate(rows):
            trajectories=r['trajectories'];rewards=[t['actual_execution_reward']['reward'] for t in trajectories]
            if (r['step']!=i+1 or len(trajectories)!=4 or r['global_rewards']!=rewards
                    or r['global_advantages']!=advantages(rewards) or r['policy_scale']!=int(arm=='execution_hybrid')
                    or r['complete_public_target_used']!=(arm=='verified_solution' or max(rewards)==0)):
                raise ValueError('Actual complete global groups/objective/fallback decisions must reconstruct')
            for j,t in enumerate(trajectories):
                if (not 1<=len(t['response_ids'])<=512 or token.decode(t['response_ids'])!=t['own_response']
                        or not t['full_vocabulary_sampling'] or t['temperature']!=1.):raise ValueError('Actual bounded own-policy outputs required')
                jobs.append((arm,i+1,j,r['source_id'],t))
    if any(a['source_id']!=b['source_id'] for a,b in zip(sealed['verified_solution'],sealed['execution_hybrid'])):
        raise ValueError('Actual original source order must match between arms')
    REPORTS.mkdir()
    for arm,rows in sealed.items():
        with (REPORTS/(arm+'-sealed-prefix.jsonl')).open('x') as f:
            for r in rows:f.write(json.dumps(r)+'\n')
    def replay(item):
        arm,step,index,key,t=item;actual=catalog.score(key,t['own_response']);previous=t['actual_execution_reward']
        if actual['reward']!=previous['reward']:raise ValueError('Actual replay changed an applied training reward')
        if previous['reward']>0 and (actual.get('full_original_stream_passed')!=previous.get('full_original_stream_passed')
                or actual.get('scored_examples_passed')!=previous.get('scored_examples_passed')
                or actual.get('scored_examples')!=previous.get('scored_examples')):raise ValueError('Positive reward must reproduce the complete published checks')
        return dict(arm=arm,step=step,global_sample=index,original_source_id=key,actual_reexecution=actual)
    with ThreadPoolExecutor(max_workers=2) as pool:observations=list(pool.map(replay,jobs))
    summary={}
    for arm,rows in sealed.items():
        rewards=[x for r in rows for x in r['global_rewards']]
        summary[arm]=dict(actual_updates=22,own_samples=len(rewards),positive_partial_or_full=sum(x>0 for x in rewards),
            full_published_stream_passes=sum(x==1 for x in rewards),mean_published_execution_reward=sum(rewards)/len(rewards),
            complete_public_target_updates=sum(r['complete_public_target_used'] for r in rows),
            nonzero_applied_policy_surrogate_updates=sum(r['policy_scale']!=0 and r['policy_surrogate']!=0 for r in rows))
    files=[Path(__file__),run/'plan.json',ROOT/'reports/causal-code-rewards-v2/admission.json',
        ROOT/'src/slm_lab/causal_code_reward_v2.py',ROOT/'src/slm_lab/causal_code_reward_catalog_v1.py',
        *[REPORTS/(a+'-sealed-prefix.jsonl') for a in sealed]]
    atomic_json(REPORTS/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),matched_updates_per_arm=22,
        actual_own_response_rewards_independently_reexecuted=len(observations),actual_reexecutions=observations,
        actual_matched_training_prefix_summary=summary,all_applied_rewards_reproduced=True,
        finite_training_reward_not_native_benchmark_correctness=True,no_full_correct_training_program_observed=all(s['full_published_stream_passes']==0 for s in summary.values()),
        complete173_function_comparison_and_held_quality_still_pending=True,genuine_human_preference_RLHF=False,
        source_files_sha256={str(p):file_hash(p) for p in files},candidate_promoted=False,goal_achieved=False))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
