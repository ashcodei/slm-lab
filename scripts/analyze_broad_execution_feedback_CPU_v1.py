"""Read the closed512-update CE-control execution-feedback distribution.

Checks recorded trajectories/rewards/advantages. It neither reexecutes the
programs nor treats recorded reference agreement as independent correctness.
The CE control logs rewards but does not optimize an execution-policy loss.
"""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/tmp/slm-broad-code-step512-snapshot-CPU-v1')
OUT=ROOT/'reports/broad-code-execution-feedback-diagnostic-v1'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def summarize(rows,*,expected_groups,expected_world):
    total=Counter();families={};statuses=Counter();histogram=Counter();group_rows=[]
    for index,row in enumerate(rows):
        if (row['step']!=index+1 or row['method']!='anchor_CE' or row['world']!=expected_world
                or len(row['trajectories'])!=expected_world or len(row['global_rewards'])!=expected_world
                or len(row['global_advantages'])!=expected_world or row['genuine_human_RLHF'] is not False):
            raise ValueError('The complete ordered original CE-control groups and actual machine feedback required')
        scores=row['global_rewards']
        if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in scores):
            raise ValueError('Every observed finite bounded actual reward must remain exact')
        mean=math.fsum(scores)/len(scores)
        std=math.sqrt(math.fsum((v-mean)**2 for v in scores)/len(scores))
        expected=[(v-mean)/(std+1e-6) for v in scores]
        if any(not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10) for a,b in zip(expected,row['global_advantages'])):
            raise ValueError('Every group-relative advantage must reconstruct from actual recorded rewards')
        family=row['source_id'].split(':',1)[0];counts=families.setdefault(family,Counter())
        informative=max(scores)>min(scores)
        for counter in [total,counts]:
            counter['groups']+=1;counter['groups_with_any_positive_reward']+=any(v>0 for v in scores)
            counter['groups_with_reward_variation']+=informative;counter['all_zero_reward_groups']+=all(v==0 for v in scores)
            counter['all_equal_reward_groups']+=not informative
        actions=0;EOS=0
        for score,trajectory in zip(scores,row['trajectories']):
            ids=trajectory['response_ids'];observation=trajectory['actual_execution_reward']
            if (not ids or len(ids)>row['action_horizon'] or 2 in ids[:-1] or observation['reward']!=score
                    or observation.get('complete_delivery_stopped_on_EOS') is not (ids[-1]==2)
                    or len(trajectory['actual_cached_behavior_logps'])!=len(ids)
                    or any(not math.isfinite(v) or v>0 for v in trajectory['actual_cached_behavior_logps'])):
                raise ValueError('Every original sampled action, genuine stop, behavior score and reward must reconstruct')
            actions+=len(ids);EOS+=ids[-1]==2
            for counter in [total,counts]:
                counter['trajectories']+=1;counter['EOS_stops']+=ids[-1]==2
                counter['positive_reward_trajectories']+=score>0;counter['zero_reward_trajectories']+=score==0
                counter['full_reward_trajectories']+=score==1
            statuses[family+':'+observation.get('status','missing_status')]+=1
            histogram[format(float(score),'.17g')]+=1
        if actions!=row['global_sampled_action_tokens']:raise ValueError('Every complete original sampled action must remain counted')
        group_rows.append(dict(step=row['step'],position=row['position'],source_id=row['source_id'],family=family,
            original_rewards=scores,independently_reconstructed_advantages=expected,has_relative_reward_signal=informative,
            actual_action_horizon=row['action_horizon'],actual_sampled_actions=actions,EOS_stops=EOS))
    if total['groups']!=expected_groups:raise ValueError('The entire declared committed group prefix is required')
    return dict(totals=dict(total),by_family={k:dict(v) for k,v in families.items()},statuses=dict(statuses),
        reward_histogram=dict(histogram),every_original_group=group_rows,
        reward_varying_group_fraction=total['groups_with_reward_variation']/total['groups'],
        sampled_EOS_fraction=total['EOS_stops']/total['trajectories'],
        CE_control_rewards_observed_not_execution_loss_optimized=True,unrun_execution_arm_behavior_not_predicted=True,
        recorded_runtime_qualifications_not_certified_algorithmic_failures=True,
        weak_generated_reference_probes_not_independent_semantic_or_human_supervision=True)


def main():
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Fresh CPU-only closed-prefix diagnostic required')
    receipt=json.loads((SOURCE/'complete.json').read_text());journal=SOURCE/'committed-journal-prefix.jsonl'
    if (receipt.get('complete') is not True or receipt.get('completed')!=512 or receipt.get('arm')!='anchor_CE'
            or receipt['source_files_sha256'][str(journal)]!=sha(journal)):
        raise ValueError('Exact immutable original512-update CE-control journal required')
    files={str(path):sha(path) for path in [SOURCE/'complete.json',journal,Path(__file__).resolve()]}
    with journal.open() as stream:result=summarize((json.loads(line) for line in stream),expected_groups=512,expected_world=4)
    if any(sha(path)!=wanted for path,wanted in files.items()):raise ValueError('Every closed source byte must remain unchanged')
    OUT.mkdir();result.update(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        new_models=0,new_optimizer_updates=0,new_program_executions=0,original_workers_unchanged=True,
        independent_correctness_or_new_policy_improvement_proved=False,goal_achieved=False)
    (OUT/'complete.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:result[key] for key in ['totals','by_family','reward_varying_group_fraction','sampled_EOS_fraction']}))


if __name__=='__main__':main()
