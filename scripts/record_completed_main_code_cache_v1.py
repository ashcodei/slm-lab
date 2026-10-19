"""Record complete matched code outcomes and scoped real-model cache results."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/completed-main-code-cache-v1-review'


def alive(value,waiting=False):
    proc=Path('/proc')/str(value['pid']);fields=(proc/'stat').read_text().split(') ',1)[1].split()
    if fields[19]!=value['ticks'] or fields[0] in {'Z','X'}:raise ValueError('Exact live owner/worker birth required')
    children=(proc/'task'/str(value['pid'])/'children').read_text().strip()
    visible=next((s.split('=',1)[1] for s in (proc/'environ').read_bytes().decode().split('\0') if s.startswith('CUDA_VISIBLE_DEVICES=')),None)
    if waiting and (children or visible!=''):raise ValueError('Waiting queue must remain hidden and childless')
    return dict(pid=value['pid'],ticks=fields[19],state=fields[0],children=children,CUDA_VISIBLE_DEVICES=visible)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only scope review required')
    master_path=ROOT/'reports/completion-audit.json';master=json.loads(master_path.read_text())
    previous_path=ROOT/master['latest_full_scope_review'];requirements=json.loads(previous_path.read_text())['requirements']
    code_path=ROOT/'reports/main-matched-code-complete-v1-readback/complete.json';code=json.loads(code_path.read_text())
    cache_path=ROOT/'reports/main-native-cache-full-CPU-v1/complete.json';cache=json.loads(cache_path.read_text())
    FileBindings(code['source_files_sha256']).check(full=True);FileBindings(cache['source_files_sha256']).check(full=True)
    totals=code['actual_matched_endpoint_totals']
    if (totals['foundation12853']['syntax']!=20 or totals['instruction14336']['syntax']!=36
            or any(r['functional']!=0 or r['tasks']!=48 for r in totals.values())):
        raise ValueError('Every complete fixed native coding outcome must reconcile')
    cache_rows=[]
    for endpoint,result in cache['actual_fixed_endpoint_results'].items():
        if (not result['actual_greedy_tokens_all_three_paths_equal']
                or not result['all_observed_logit_paths_within_predeclared_bound']
                or not result['actual_parameters_and_all_private_CPU_RNG_unchanged']):
            raise ValueError('This review expects successful actual cache/full-forward diagnostics')
        for entry in result['actual_cache_full_forward_comparisons']:
            for row in entry['actual_paths']:cache_rows.append(dict(endpoint=endpoint,input=entry['input'],**row))
    maximum=max(r['maximum_absolute_logit_error'] for r in cache_rows)
    if any(r['actual_greedy_argmax_disagreements'] for r in cache_rows):raise ValueError('Any observed argmax change requires explicit diagnosis')
    main_folder=ROOT/'reports/phase-joint-v3';plan=json.loads((main_folder/'plan.json').read_text());progress=json.loads((main_folder/'progress.json').read_text())
    workers=[alive(r) for r in json.loads((main_folder/'children.json').read_text()) if r['subject'].startswith('train-rank-')]
    if len(workers)!=4:raise ValueError('Every original main GPU worker required')
    protected=dict(plan['source_files_sha256']);queues={}
    for name in ['phase-human-preference-v4','broad-anchored-code-v3','full-held-QA-pipeline-v1','full-AI2D-pipeline-v1',
        'focused-code-continuation-v1','full-original-MBPP-pipeline-v1']:
        path=ROOT/'reports'/(name+'-queue.json');q=json.loads(path.read_text());queues[name]=alive(q['owner'],True);protected.update(q['source_files_sha256'])
    source_jobs={}
    for name in ['CodeContests-full-Python-references-v1','CodeContests-public-execution-traces-candidate-v1','CodeContests-reference-outputs-v1-readback',
        'public-execution-targets-v1-readback-admission','AI2D-fixed-exposure-strata-v1']:
        path=ROOT/'reports'/name/'plan.json';value=json.loads(path.read_text())
        if path.with_name('failure.json').exists():raise ValueError('Original source job failed: '+name)
        source_jobs[name]=alive(value['owner']);protected.update(value['source_files_sha256'])
    at=datetime.now(timezone.utc).isoformat()
    runtime=dict(at=at,main_owner=alive(plan['owner']),original_four_main_workers=workers,actual_main_updates=progress['actual_updates'],
        planned_main_updates=35284,phase=progress['phase'],actual_waiting_queues=queues,actual_source_owners=source_jobs,
        actual_GPU_observation=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True))
    requirements['any_code_correctness_and_efficiency']['evidence'] += (
        ' The fixed main foundation12853versus instruction14336CPU observer is now completely finished. The static '
        'reader reconstructs all96original input/action/code deliveries, every fresh native case denominator/verdict '
        'and all48secondary ordinary-Python programs from the completed evidence, without a new NN generation or '
        'unnecessary program rerun. Syntax passes improve20/48to36/48; functional and empirical-efficiency passes '
        'remain0/48at both endpoints. Both ordinary-Python totals remain0. This supersedes the partial-frame status; '
        'it is a full matched developmental result, not an independent final score or a functional gain. Actual '
        'outputs include wrong operations, nonexistent methods and undefined variables. A new diagnostic tests '
        'whether cached decoding explains this, rather than assuming failure is a scorer/cache problem.')
    requirements['own_architecture']['evidence'] += (
        ' New actual full-size984399690-value CPUFP32 diagnostics at BOTH fixed main endpoints compare original '
        'full-forward logits against original and accelerated rotary-table caches. They cover a real original '
        'native code prefix and16actual old actions, a complete4096publisher-code context, single-token decoding '
        'near the4096limit and a whole real448human training photo with196visual prefix tokens. Two whole4096 '
        'chunk paths compare every32768-vocabulary value at every text position. There are '
        f'{sum(r["actual_vocabulary_values_compared"] for r in cache_rows)}actual vocabulary comparisons across all '
        f'paths; maximum absolute logit error is{maximum},with0next-token argmax disagreements. Actual16-budget '
        'greedy actions are identical for full recomputation, original cache and accelerated cache at both '
        'endpoints. Every parameter/input and actual private Python/NumPy/TorchCPU RNG stays unchanged. Two real '
        'small-model tests cover text/image/chunk/single alignment and deliberately damaged decoder detection. '
        'Full logit tensors are not archived; per-position maximum errors/argmax flags and real greedy actions '
        'are retained. This finds no cache fault on those CPU inputs. It does not prove all contexts, GPU/BF16 '
        'parity, algorithmic correctness, a new architecture or model-quality gain.')
    requirements['gpu_runtime']['evidence'] += (
        ' The two real984MCPU diagnostic endpoints completed terminally, as did the full matched observer and '
        'its static reader. They initialized no CUDA devices and never changed existing512-budget assessments '
        'or model weights. The original main run continues on allfour original L40s, with allsix original/new '
        'GPU successors still hidden/childless and all original public-source/reference owners live. No frozen '
        'future queue was retired, changed or restarted in this turn.')
    requirements['serious_full_model_training']['evidence'] += (
        f' The original actual full-weight main run remains at{progress["actual_updates"]}/35284updates. Because '
        'complete code correctness remains0despite improved syntax and healthy sampled CPU cache computation, '
        'the next priority is to construct and verify a versioned code-first training pipeline: the complete '
        'new public code/output/replay epoch should precede sparse execution-reward learning and heavy final '
        'assessments. A design record preserves all current source populations, human/code epochs, under1B '
        'limits and full QA/image/MBPP benchmark coverage. This is not yet implemented or queued, no original '
        'controller is cancelled, and no code-first training gain is claimed.')
    if len(requirements)!=10 or any(r['status']!='incomplete_for_requested_final_model' for r in requirements.values()):raise ValueError('All final original requirements remain incomplete')
    OUT.mkdir()
    design=dict(at=at,status='design_recorded_not_implemented_not_queued',
        reason='Complete native coding remains0/48 after instruction training; sampled actual CPU caches match full forward. Put full code supervision before execution RL and postpone heavy final scoring until after learning.',
        intended_training_order=['Complete current phase-joint-v3 unchanged','Complete new code-first CE/output/12-family-replay epoch from full main checkpoint',
            'Complete chosen_CE and human_DPO original85786pairs each','Complete anchor_CE and anchor_execution original3721sources each',
            'Export final own actor as a single counted dense standalone student','Run full fixed native code, QA, AI2D and original MBPP assessments of all declared own endpoints'],
        new_source_population=58488,all_admitted_public_execution_examples_required=True,all12broad_raw_image_word_replay_families_preserved=True,
        full_QA_questions_each_endpoint=16961,original_AI2D_questions_each_endpoint=3088,AI2D_conditions=['proper_image','gray_image','wrong_image'],
        original_MBPP_questions_each_endpoint=600,publisher_MBPP_primary_test_problems=500,
        old_main_training_and_public_source_jobs_must_not_be_preempted=True,
        only_hidden_childless_unstarted_future_controllers_may_be_superseded_after_new_pipeline_and_tests_exist=True,
        retain_every_old_plan_artifact_and_source_control_no_silent_mutation=True,
        new_dense_standalone_factory_needed_for_plain_dense_initializer_without_fictitious_actor_fold=True,
        code_first_initializer_must_be_complete_main35284_not_old2048_or_partial12853_or14336=True,
        updated_human_and_code_policy_loaders_needed_for_complete_code_first_source_lineage=True,
        no_external_pretrained_model_or_teacher=True,all_ten_final_requirements_incomplete=True,goal_achieved=False)
    atomic_json(OUT/'next-code-first-design.json',design)
    evidence=[Path(__file__),previous_path,code_path,cache_path,ROOT/'reports/native-cache-comparison-v1-tests.log',OUT/'next-code-first-design.json']
    atomic_json(OUT/'review.json',dict(at=at,requirements=requirements,previous_full_scope_review=str(previous_path.relative_to(ROOT)),
        actual_full_matched_code_outcomes=code,actual_native_cache_comparison=cache,actual_runtime=runtime,
        source_files_sha256={str(p):file_hash(p) for p in evidence},previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True,no_original_scope_removed=True,candidate_promoted=False,goal_achieved=False))
    docs=[ROOT/'README.md',ROOT/'RESEARCH.md',master_path]
    if any(str(p) in protected for p in docs):
        print(json.dumps(dict(review=str(OUT/'review.json'),bound_root_documents_preserved=True)),flush=True);return
    (ROOT/'reports/completion-audit-before-completed-main-code-cache-v1.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={k:r['evidence'] for k,r in requirements.items()},latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),
        latest_actual_main_runtime=runtime,latest_complete_main_matched_code_readback=str(code_path.relative_to(ROOT)),
        latest_real_full_size_native_cache_diagnostic=str(cache_path.relative_to(ROOT)),next_code_first_design=str((OUT/'next-code-first-design.json').relative_to(ROOT)),
        current_goal_turn_classification='progress',all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph=(f'Completed main coding and cache checks, **{at}**: across both unchanged original frames, syntax '
        'passes improve **20/48 to 36/48** between the completed foundation and instruction step 14,336. '
        '**Functional and empirical-efficiency passes remain 0/48 at both checkpoints.** Static readback '
        'reconstructs all 96 native deliveries and 48 secondary Python deliveries from retained actual actions, '
        'program bytes and fresh execution outcomes. This is a matched developmental result.\n\n'
        'Both actual 984,399,690-parameter checkpoints also complete CPU FP32 cache/full-forward checks on a '
        '4,096-token public-code example, an original native code prefix and a real 448px photo. Original and '
        f'accelerated caches have maximum logit error **{maximum:.8f}**, no observed next-token choice changes '
        'and identical 16-budget greedy deliveries. Parameters and private CPU RNG remain unchanged. This '
        'finds no cache fault on those inputs; GPU/BF16 and arbitrary-context parity are not proved. '
        'The next priority is a versioned code-first training pipeline that supplies broad code supervision '
        'before execution RL and defers heavy final scoring until learning is complete. **That reordering is '
        'a design only: current training and all existing queues remain unchanged. All ten final requirements '
        'remain incomplete.** [Review](reports/completed-main-code-cache-v1-review/review.json), '
        '[matched coding](reports/main-matched-code-complete-v1-readback/complete.json), '
        '[actual cache diagnostic](reports/main-native-cache-full-CPU-v1/complete.json).\n\n')
    for path in docs[:2]:
        first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),main_updates=progress['actual_updates'],syntax_before=20,syntax_after=36,
        coding_denominator=48,functional_before=0,functional_after=0,actual_cache_max_logit_error=maximum,
        cache_compared_vocab_values=sum(r['actual_vocabulary_values_compared'] for r in cache_rows),code_first_reordering_implemented=False,goal_achieved=False)),flush=True)


if __name__=='__main__':main()
