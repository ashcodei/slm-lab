"""Follow the exact pilot owner; independently recompute final development scores."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from assess_joint_java_comparison_v1 import identity, same
from slm_lab.code_prompt_results_v1 import source_metrics, code_metrics, knowledge_metrics
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/code-prompt-matching-pilot-v1'


def main():
    plan_path=REPORTS/'plan.json'
    plan=json.loads(plan_path.read_text())
    initial_guard=FileBindings({str(plan_path):file_hash(plan_path),str(Path(__file__).resolve()):file_hash(__file__),
        str(ROOT/'src/slm_lab/code_prompt_results_v1.py'):file_hash(ROOT/'src/slm_lab/code_prompt_results_v1.py')})
    if not same(plan['owner']):
        raise ValueError('Reader must initially verify its exact original pilot owner live')
    atomic_json(REPORTS/'independent-reader.json',dict(actor=identity(__import__('os').getpid()),
        exact_owner=plan['owner'],registered_while_owner_live=True,goal_achieved=False))
    while same(plan['owner']):
        initial_guard.check()
        time.sleep(2)
    initial_guard.check(full=True)
    if (REPORTS/'failure.json').exists():
        raise ValueError('Pilot owner stopped with a retained failure; do not relabel it as a completed experiment')
    restored=json.loads((REPORTS/'restored.json').read_text())
    if restored.get('same_original_waiter_resumed') is not True:
        raise ValueError('Original exact queued trainer restoration receipt required')
    guard=FileBindings({**plan['source_files_sha256'],str(plan_path):file_hash(plan_path)})
    selected=json.loads((ROOT/'reports/code-residual-lr-pilot-v1/selected-original-rows.json').read_text())
    questions=list(map(json.loads,(ROOT/'data/knowledge-eval-v1/questions.jsonl').read_text().splitlines()))
    baselines={mode:json.loads((ROOT/'reports/extended-curriculum-v1'/file).read_text())
        for mode,file in [('completion','code-quality.json'),('instruction','code-quality-instruction.json'),('knowledge','knowledge.json')]}
    result={};inputs={};initial_policies=[];histories=[];initial_reports=[]
    def read(path):
        inputs[str(path)]=file_hash(path)
        return json.loads(path.read_text())
    for profile in plan['profiles']:
        folder=REPORTS/profile['name']
        state=read(folder/'status.json')
        if state['status']!='bounded source-matching training and capability assessment completed' or state['completed_steps']!=128:
            raise ValueError('Both actual complete fixed128-update training/evaluation runs required')
        frozen=read(folder/'frozen-final-policy.json');binding=read(folder/'evaluation-binding.json')
        checkpoint=ROOT/'checkpoints/code-prompt-matching-pilot-v1'/profile['name']/'latest.pt'
        if (file_hash(checkpoint)!=frozen['checkpoint_sha256'] or binding['checkpoint'] != {k:v for k,v in frozen.items()
                if k not in {'sealed_before_benchmark_evaluation','training_seconds','candidate_promoted','goal_achieved'}}
                or binding['loader_calls']!=3 or not binding['unchanged_prompt_generation_and_scoring']):
            raise ValueError('Complete frozen saved policy and unchanged three evaluation loader calls required')
        initial=read(folder/'initial-policy.json');initial_policies.append(initial)
        history=read(folder/'training-metrics.json');histories.append(history)
        if (len(history)!=128 or [r['step'] for r in history]!=list(range(1,129))
                or [i for r in history for i in r['train_source_ids']] != [r['id'] for r in selected['train']]
                or any(r['source_matching']['alpha']!=profile['alpha'] for r in history)):
            raise ValueError('Every exact original training example and declared objective/update required')
        before=read(folder/'initial-source-scores.json');initial_reports.append(before)
        after=read(folder/'final-source-scores.json')
        result[profile['name']]=dict(checkpoint_sha256=frozen['checkpoint_sha256'],
            initial_source=source_metrics(before,selected),final_source=source_metrics(after,selected),
            training_seconds=frozen['training_seconds'],
            code={mode:code_metrics(read(folder/('code-'+mode+'.json')),baselines[mode],frozen['checkpoint_sha256'])
                for mode in ['completion','instruction']},
            knowledge=knowledge_metrics(read(folder/'knowledge.json'),baselines['knowledge'],questions,frozen['checkpoint_sha256']))
    if initial_policies[0] != initial_policies[1] or initial_reports[0]!=initial_reports[1]:
        raise ValueError('Identical initial policy tensors and all starting source scores required')
    for a,b in zip(*histories):
        fields=['step','replay_family','replay_position','replay_occurrence','learning_rate','train_source_ids','mismatched_source_ids']
        if any(a[k]!=b[k] for k in fields):raise ValueError('Exactly matched original source/rehearsal schedule required')
    control=result['CE_control'];experimental=result['CE_plus_source_matching']
    gains={}
    for mode in ['completion','instruction']:
        before={tuple(v) for v in control['code'][mode]['functional_correct_tasks']}
        after={tuple(v) for v in experimental['code'][mode]['functional_correct_tasks']}
        gains[mode]=dict(gained=[list(v) for v in sorted(after-before)],lost=[list(v) for v in sorted(before-after)])
    guard.check(full=True)
    atomic_json(REPORTS/'independent-comparison.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
        original_plan_sha256=file_hash(plan_path),profiles=result,exact_initial_policy_and_all_source_scores_match=True,
        exact_code_and29_family_rehearsal_schedule_match=True,code_functional_changes_versus_CE=gains,
        input_reports_sha256=inputs,original_source_files_sha256=plan['source_files_sha256'],
        reader_source_files_sha256={str(Path(__file__).resolve()):file_hash(__file__),
            str(ROOT/'src/slm_lab/code_prompt_results_v1.py'):file_hash(ROOT/'src/slm_lab/code_prompt_results_v1.py')},
        model_selection_or_promotion=False,goal_achieved=False,independent_final_benchmarks_certified=False,
        scope='Bounded source-association objective test on actual857M policies. Source NLL/separation cannot establish coding correctness. Local24-task and reused559-question public development assessments only; no universal, independent-final or frontier capability claim.'))
    print(json.dumps({k:{'code':v['code'],'knowledge':{n:v['knowledge'][n] for n in ['questions','generated','likelihood']}}
        for k,v in result.items()},indent=2))


if __name__=='__main__':main()
