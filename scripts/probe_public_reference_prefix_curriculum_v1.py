"""Prepare partial-program training views and execute balanced train-only controls.

No student samples or optimizer steps occur. Original reference reconstruction
and empty-suffix controls test whether proposed conditional episodes are well
formed and nontrivial. Reward-density or unassisted capability gains require a
later, separately admitted full-size experiment.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/public-reference-prefix-curriculum-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only train-source curriculum probe required')
    import torch
    from slm_lab.active_execution_input_bindings_v1 import input_bindings,merge_bindings
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.public_reference_prefix_curriculum_v1 import QUARTERS,view
    torch.set_num_threads(1);OUT.mkdir();started=time.monotonic()
    pid=os.getpid();fields=(Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()
    owner=dict(pid=pid,ticks=fields[19],CUDA_VISIBLE_DEVICES='')
    atomic_json(OUT/'owner.json',dict(owner=owner,model_generations=0,model_updates=0,automatic_retry=False))
    try:
        corpus=ExpandedExecutionCorpus(ROOT);strata=defaultdict(list)
        for position,row in enumerate(corpus.rows):
            if row['kind']=='apps' and not corpus.apps_controls[row['source_id']]['informative']:continue
            stratum='documented' if row['kind']=='documented' else 'APPS:'+row['original']['difficulty']
            strata[stratum].append(position)
        selected=[]
        for stratum in ['documented','APPS:introductory','APPS:interview','APPS:competition']:
            population=strata[stratum]
            if len(population)<4:raise ValueError('Predeclared balanced original train-source control population missing')
            selected.extend(population[i*len(population)//4] for i in range(4))
        bindings=input_bindings(ROOT,corpus)
        seeds=[Path(__file__),ROOT/'tests/test_public_reference_prefix_curriculum_v1.py']
        merge_bindings(bindings,{str(p):file_hash(p) for p in python_sources(ROOT,seeds)})
        log=ROOT/'reports/public-reference-prefix-curriculum-v1-tests.log'
        merge_bindings(bindings,{str(log):file_hash(log)})
        plan=dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,
            original_train_sources=3721,retained_reference_quarters=list(QUARTERS),
            original_native_context_and_all_anchor_target_bytes_must_reconstruct=True,
            minimum_non_EOS_missing_reference_tokens_when_hint_present=16,
            controls_selected_before_execution=[corpus.row(p)['id'] for p in selected],
            controls_are_train_only_original_reference_and_counterfactual_empty_suffix_not_student_samples=True,
            actual_student_reward_density_and_unassisted_capability_gain_not_measured=True,
            source_files_sha256=bindings,model_generations=0,model_updates=0,admitted_to_RL=False,
            existing_live_and_queued_training_recipes_unchanged=True,automatic_retry=False,goal_achieved=False)
        atomic_json(OUT/'plan.json',plan)
        views_path=OUT/'native-views.jsonl';totals={q:Counter() for q in QUARTERS};last=time.monotonic()
        with views_path.open('x') as output:
            for position in range(len(corpus)):
                source=corpus.row(position);ids,labels=corpus.sequence(position)
                for quarters in QUARTERS:
                    value=view(corpus.tokenizer,source['original'],ids,labels,quarters)
                    record=dict(id=source['id'],original_id=value['id'],retained_reference_quarters=quarters,
                        original_native_input_sha256=value['original_native_input_sha256'],
                        complete_context_positions=value['original_native_context_positions'],
                        complete_original_anchor_targets=value['original_response_EOS_targets'],
                        retained_reference_tokens=value['retained_reference_tokens'],
                        original_native_prompt_positions=len(value['original_native_prompt_ids']),
                        conditional_rollout_prefix_positions=len(value['native_rollout_prefix_ids']),
                        missing_original_response_EOS_targets=value['missing_original_response_EOS_targets'],
                        conditional_rollout_action_budget=value['complete_missing_suffix_training_budget'],
                        training_only=True,prefix_only_execution_control_required=True,admitted_to_RL=False)
                    output.write(json.dumps(record,separators=(',',':'))+'\n')
                    totals[quarters].update(views=1,complete_original_anchor_targets=record['complete_original_anchor_targets'],
                        complete_original_context_positions=len(ids),retained_reference_tokens=value['retained_reference_tokens'],
                        missing_original_response_EOS_targets=value['missing_original_response_EOS_targets'],
                        views_with_actual_hint=int(value['retained_reference_tokens']>0))
                if time.monotonic()-last>15:
                    atomic_json(OUT/'progress.json',dict(owner=owner,prepared_original_sources=position+1,total_sources=3721,
                        elapsed_seconds=time.monotonic()-started,model_updates=0));last=time.monotonic()
        if any(t['views']!=3721 or t['complete_original_anchor_targets']!=584218
                or t['complete_original_context_positions']!=2287522 for t in totals.values()):
            raise ValueError('Every fixed curriculum level must preserve the complete original source population and CE budget')
        observations=[];ledger=OUT/'execution-controls.jsonl'
        with ledger.open('x') as output:
            for position in selected:
                source=corpus.row(position);ids,labels=corpus.sequence(position);original=source['original']['response']
                reference=corpus.score(position,original,stopped_on_eos=True)
                for quarters in [3,2]:
                    value=view(corpus.tokenizer,source['original'],ids,labels,quarters)
                    prefix=value['public_reference_prefix_text']
                    actual=corpus.score(position,prefix,stopped_on_eos=True)
                    reference_ok=reference['reward']==1.
                    empty_suffix_wins=actual['reward']==1.
                    eligible=reference_ok and value['retained_reference_tokens']>0 and not empty_suffix_wins
                    record=dict(id=source['id'],position=position,kind=source['kind'],
                        difficulty=source['original'].get('difficulty'),retained_reference_quarters=quarters,
                        original_native_input_sha256=value['original_native_input_sha256'],
                        supplied_public_reference_prefix=prefix,retained_reference_tokens=value['retained_reference_tokens'],
                        missing_original_response_EOS_targets=value['missing_original_response_EOS_targets'],
                        full_original_reference_execution=reference,counterfactual_prefix_then_EOS_execution=actual,
                        full_reference_passed=reference_ok,empty_suffix_would_earn_full_credit=empty_suffix_wins,
                        proposed_conditional_episode_nontrivial=eligible,
                        counterfactual_execution_controls_not_neural_samples=True,
                        admitted_to_RL=False,model_generations=0,model_updates=0)
                    output.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n');output.flush();os.fsync(output.fileno())
                    observations.append(record)
                    print(json.dumps(dict(source=source['id'],quarters=quarters,reference_passed=reference_ok,
                        prefix_only_reward=actual['reward'],nontrivial_conditional_episode=eligible)),flush=True)
        result=dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,complete=True,
            original_train_sources=3721,prepared_native_views=4*3721,quarter_level_counts={str(k):dict(v) for k,v in totals.items()},
            selected_original_train_sources=16,actual_conditional_execution_controls=len(observations),
            full_reference_source_passes=len({r['id'] for r in observations if r['full_reference_passed']}),
            nontrivial_hinted_episodes=sum(r['proposed_conditional_episode_nontrivial'] for r in observations),
            trivial_empty_suffix_full_credit_episodes_rejected=sum(r['empty_suffix_would_earn_full_credit'] for r in observations),
            unhinted_or_failed_reference_controls_not_admitted=sum(not r['proposed_conditional_episode_nontrivial'] for r in observations),
            plan_sha256=file_hash(OUT/'plan.json'),native_views_sha256=file_hash(views_path),execution_ledger_sha256=file_hash(ledger),
            source_files_sha256={str(p):file_hash(p) for p in [Path(__file__),OUT/'plan.json',views_path,ledger,
                ROOT/'src/slm_lab/public_reference_prefix_curriculum_v1.py',ROOT/'tests/test_public_reference_prefix_curriculum_v1.py',log]},
            actual_student_reward_density_or_capability_gain_measured=False,
            all_actual_execution_observations_are_public_train_reference_controls_not_student_generations=True,
            full_size_GPU_capacity_and_actual_sampled_conditional_RL_experiment_still_required=True,
            curriculum_not_applied_to_current_main_human_or_code_queues=True,
            model_generations=0,model_updates=0,admitted_to_RL=False,candidate_promoted=False,goal_achieved=False)
        atomic_json(OUT/'complete.json',result)
        print(json.dumps({k:result[k] for k in ['prepared_native_views','actual_conditional_execution_controls',
            'full_reference_source_passes','nontrivial_hinted_episodes','trivial_empty_suffix_full_credit_episodes_rejected']}),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(owner=owner,type=type(error).__name__,message=str(error),
            traceback=traceback.format_exc(),automatic_retry=False,model_updates=0,goal_achieved=False));raise


if __name__=='__main__':main()
