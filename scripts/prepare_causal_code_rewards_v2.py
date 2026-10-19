"""Execute and retain no-op counterfactuals on every admitted public train function."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/causal-code-rewards-v2'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('No GPU needed for actual confined public examples')
    from slm_lab.causal_code_reward_v2 import ablate,evaluate,pass_ablation
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.public_example_code_reward_v1 import PublicExampleCodeReward
    if REPORTS.exists():raise ValueError('Preserve all previous complete/partial ledgers')
    if '3 passed' not in (ROOT/'reports/causal-code-reward-v2-tests.log').read_text():raise ValueError('Actual assignment/side-effect/role tests required')
    catalog=PublicExampleCodeReward(ROOT)
    files=[Path(__file__),ROOT/'src/slm_lab/causal_code_reward_v2.py',ROOT/'src/slm_lab/code_training_views_v1.py',
        ROOT/'src/slm_lab/documented_doctest_v1.py',ROOT/'src/slm_lab/public_example_code_reward_v1.py',ROOT/'reports/causal-code-reward-v2-tests.log',
        ROOT/'data/documented-doctest-verified-v1/manifest.json',ROOT/'data/documented-doctest-verified-v1/admission.json',
        ROOT/'data/documented-doctest-stage-v1/candidates.jsonl',ROOT/'reports/documented-doctest-verified-v1/observations.jsonl']
    bindings={str(p):file_hash(p) for p in files};guard=FileBindings(bindings);started=time.monotonic();REPORTS.mkdir()
    atomic_json(REPORTS/'plan.json',dict(source_files_sha256=bindings,complete_original_train_functions=len(catalog.candidates),
        no_op_causal_checks=True,full_reward_requires_every_original_example=True,workers=2,production_RL_updates=0,goal_achieved=False))
    def one(key):
        c=catalog.candidates[key];reference=catalog.references[key];control=ablate(c,reference)
        result=dict(source_id=key,ablation=control)
        if control['supported']:
            correct=evaluate(c,reference,control,c['program'],origin='test_fixture');noop=evaluate(c,reference,control,pass_ablation(c),origin='test_fixture')
            if correct['reward']!=1. or noop['reward']!=0. or correct['full_original_stream_passed'] is not True:
                raise ValueError('Actual full source must get full reward; actual no-op must get zero')
            result.update(original_publisher_program_control=correct,deterministic_no_op_control=noop)
        return result
    results=[]
    with (REPORTS/'observations.jsonl').open('x') as f,ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(one,sorted(catalog.candidates)):
            results.append(result);f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush()
    guard.check(full=True);supported=[r for r in results if r['ablation']['supported']]
    result=dict(at=datetime.now(timezone.utc).isoformat(),complete_original_functions=len(results),reward_informative_functions=len(supported),
        source_reward_uninformative_functions=len(results)-len(supported),
        post_call_value_or_side_effect_checks=sum(len(r['ablation'].get('post_call_dependent_check_indices',[])) for r in results),
        actual_original_program_reward_one_controls=len(supported),actual_no_op_reward_zero_controls=len(supported),
        observations_sha256=file_hash(REPORTS/'observations.jsonl'),source_files_sha256=bindings,
        independent_reader_pending=True,production_RL_admitted=False,production_RL_updates=0,human_preferences=0,
        seconds=time.monotonic()-started,goal_achieved=False)
    atomic_json(REPORTS/'prepared.json',result);print(json.dumps({k:v for k,v in result.items() if k!='source_files_sha256'}),flush=True)


if __name__=='__main__':main()
