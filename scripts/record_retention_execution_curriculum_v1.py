"""Record measured instruction-phase drift and new pending retention work."""
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import subprocess

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/retention-execution-curriculum-v1-review'


def live(identity,*,waiting=False):
    proc=Path('/proc')/str(identity['pid']);fields=(proc/'stat').read_text().split(') ',1)[1].split()
    if fields[19]!=identity['ticks'] or fields[0] in {'Z','X'}:raise ValueError('Exact original process birth required')
    children=(proc/'task'/str(identity['pid'])/'children').read_text().strip()
    visible=next((s.split('=',1)[1] for s in (proc/'environ').read_bytes().decode().split('\0') if s.startswith('CUDA_VISIBLE_DEVICES=')),None)
    if waiting and (children or visible!=''):raise ValueError('Original waiting GPU successor must remain hidden and childless')
    return dict(pid=identity['pid'],ticks=fields[19],state=fields[0],children=children,CUDA_VISIBLE_DEVICES=visible)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-hidden review required')
    master_path=ROOT/'reports/completion-audit.json';master=json.loads(master_path.read_text())
    previous_path=ROOT/master['latest_full_scope_review'];previous=json.loads(previous_path.read_text())
    requirements=previous['requirements']
    prep_path=ROOT/'reports/focused-code-continuation-sources-v2-preparation/complete.json'
    prep=json.loads(prep_path.read_text())
    if (not prep['all12actual_replay_family_extremes_rebuilt']
            or not prep['all58488_CODE_example_batches_identical_to_preceding_full_actual_rebuild']
            or not prep['execution_source_missing_so_optimization_forbidden']):
        raise ValueError('Actual whole-source preparation must remain separate from optimization')
    for family in ['packed:original_packed_code','education']:
        if prep['actual_selected_source_budgets'][family]['response_EOS_targets']!=19918080:
            raise ValueError('Exact whole raw retention budgets required')
    phase=ROOT/'reports/phase-joint-v3';main=json.loads((phase/'plan.json').read_text())
    progress=json.loads((phase/'progress.json').read_text())
    selected=['initial','12853','14336','16384']
    held={name:json.loads((phase/('held-packed-'+name+'.json')).read_text()) for name in selected}
    protocol=['original_split','selected_original_positions','original_valid_population','evaluated_windows',
        'complete_causal_targets','maximum_original_context','original_manifest_sha256']
    for row in held.values():
        if any(row[k]!=held['initial'][k] for k in protocol):raise ValueError('Every token-loss observation must use identical fixed held inputs')
    foundation=held['12853']['token_mean_NLL'];latest=held['16384']['token_mean_NLL'];initial=held['initial']['token_mean_NLL']
    drift=dict(initial_NLL=initial,completed_foundation_12853_NLL=foundation,
        instruction_14336_NLL=held['14336']['token_mean_NLL'],instruction_16384_NLL=latest,
        relative_increase_from_completed_foundation=(latest-foundation)/foundation,
        relative_reduction_from_initial=(initial-latest)/initial,latest_perplexity=math.exp(latest),
        unchanged_fixed_held_windows=128,unchanged_original_causal_targets=524160,
        producer_neural_observations_protocol_and_arithmetic_checked=True,
        fresh_neural_loss_or_weight_Adam_RNG_recomputation_not_performed=True,
        phase_association_not_proof_of_cause=True,token_prediction_not_generated_program_correctness=True)
    jobs={};protected=set(main['source_files_sha256'])
    for name in ['CodeContests-full-Python-references-v1','CodeContests-public-execution-traces-candidate-v1',
        'CodeContests-reference-outputs-v1-readback','public-execution-targets-v1-readback-admission',
        'main-foundation-instruction14336-code-CPU-v2','AI2D-fixed-exposure-strata-v1']:
        path=ROOT/'reports'/name/'plan.json';plan=json.loads(path.read_text())
        if path.with_name('failure.json').exists():raise ValueError('Live source/assessment job failed: '+name)
        jobs[name]=dict(identity=live(plan['owner']),plan=str(path.relative_to(ROOT)))
        for snapshot in ['progress.json','waiting.json']:
            if path.with_name(snapshot).exists():jobs[name][snapshot]=json.loads(path.with_name(snapshot).read_text())
        protected.update(plan['source_files_sha256'])
    queues={}
    for name in ['phase-human-preference-v4','broad-anchored-code-v3','full-held-QA-pipeline-v1','full-AI2D-pipeline-v1']:
        path=ROOT/'reports'/(name+'-queue.json');queue=json.loads(path.read_text())
        queues[name]=live(queue['owner'],waiting=True);protected.update(queue['source_files_sha256'])
    workers=[live(row) for row in json.loads((phase/'children.json').read_text()) if row['subject'].startswith('train-rank-')]
    if len(workers)!=4:raise ValueError('Every original four-L40 main worker required')
    at=datetime.now(timezone.utc).isoformat()
    runtime=dict(at=at,original_main_owner=live(main['owner']),original_four_main_workers=workers,
        actual_main_updates=progress['actual_updates'],planned_main_updates=progress['planned_updates'],phase=progress['phase'],
        original_hidden_childless_GPU_queues=queues,CPU_jobs=jobs,
        GPU_observation=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True))
    requirements['public_data']['evidence'] += (
        ' The new v2 retention preparation preserves every58488complete screened new-code batch exactly, verifies '
        'its unchanged v1 full-native rebuild and all source bindings, and actually reconstructs all12replay-family '
        'extremes. It adds4864whole raw-code and4864education-text presentations,19922944context positions and19918080 '
        'causal targets EACH. All original10replay families remain present; their individual selected replay batches '
        'are reduced because the fixed total replay budget now cycles12families. All original source pools remain '
        'available, and the preparation-only plan cannot optimize or silently omit the unadmitted execution task. '
        'Two new schedule/admission/tail tests pass. An independent live reader now reconstructs every original '
        'reference source/case contract and checks retained base64stdout tokens directly, without calling the producer '
        'comparator. Three actual execution tests verify full counts, damaged stdout/source rejection, last-case '
        'failure and syntax qualifications. The new public-target reader/admitter waits for all3959reference '
        'dispositions, original candidate completion and full independent output readback. It independently rebuilds '
        'all native public-only targets, scans all32953held stems, excludes flagged candidate components and cold-reads '
        'every separately written admitted sequence. Its three controls prove actual stdout rather than gold text, '
        'private/generated poison exclusion, whole-context first-fit selection and source/partial/output rejection. '
        'That reader is currently waiting; no auxiliary admission, execution-reward admission or new training is '
        'claimed. No external pretrained model or teacher was added.')
    requirements['serious_full_model_training']['evidence'] += (
        f' The original four-GPU main run remains active at{progress["actual_updates"]}/35284updates. On the identical '
        f'128held packed-code windows/524160causal targets, producer NLL goes{initial}initial to{foundation}at completed '
        f'foundation12853, then{held["14336"]["token_mean_NLL"]}at14336and{latest}at16384. The latest value is '
        f'{100*drift["relative_increase_from_completed_foundation"]}%higher than completed foundation, while '
        f'{100*drift["relative_reduction_from_initial"]}%below initial. This is a measured retention concern associated '
        'with instruction training, not proof of its cause or generated-code accuracy. The new replay curriculum '
        'is a preparation-only hypothesis; it has not trained a full source epoch or demonstrated recovery.')
    requirements['any_code_correctness_and_efficiency']['evidence'] += (
        ' The fixed main-foundation12853versus instruction14336observer continues unchanged on CPU, capturing actual '
        'native actions for all48tasks per endpoint and later original/ordinary fresh execution. No complete matched '
        'quality result is yet available, no cherry-picked partial score is treated as the outcome, and loss drift '
        'cannot substitute for program correctness or efficiency. The previous completed old machine-contrast '
        'trial remains0/48functional correctness. No new functional improvement is established.')
    requirements['gpu_runtime']['evidence'] += (
        ' Exact process-birth checks confirm allfour original main workers, all original hidden/childless human, '
        'code and full text/visual assessment successors, both original reference/data jobs, the main interim '
        'observer, fixed visual reader and both new CPU-only execution readers. None was preempted. The new retention '
        'curriculum has no admitted four-GPU production driver or capacity gate yet; that remains required before '
        'claiming it is training. Original training/queued scientific sources and held populations are unchanged.')
    if len(requirements)!=10 or any(r['status']!='incomplete_for_requested_final_model' for r in requirements.values()):
        raise ValueError('All original final requirements remain incomplete')
    OUT.mkdir()
    evidence=[Path(__file__),previous_path,prep_path,*[phase/('held-packed-'+v+'.json') for v in selected],
        ROOT/'reports/focused-code-continuation-data-v2-tests.log',ROOT/'reports/code-contests-execution-readback-v1-tests.log',
        ROOT/'reports/public-execution-target-readback-v1-tests.log',
        *[ROOT/v['plan'] for v in jobs.values()]]
    atomic_json(OUT/'review.json',dict(at=at,requirements=requirements,previous_full_scope_review=str(previous_path.relative_to(ROOT)),
        measured_same_protocol_retention=drift,actual_v2_retention_source_preparation=prep,actual_runtime=runtime,
        source_files_sha256={str(p):file_hash(p) for p in evidence},
        current_goal_turn_classification='progress',all_ten_final_requirements_incomplete=True,
        new_curriculum_training_GPU_admission_and_quality_results_pending=True,
        no_original_scope_removed=True,candidate_promoted=False,goal_achieved=False))
    docs=[ROOT/'README.md',ROOT/'RESEARCH.md',master_path]
    if any(str(p) in protected for p in docs):
        print(json.dumps(dict(review=str(OUT/'review.json'),bound_root_docs_preserved=True)),flush=True);return
    (ROOT/'reports/completion-audit-before-retention-execution-curriculum-v1.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={k:r['evidence'] for k,r in requirements.items()},
        latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),latest_actual_main_runtime=runtime,
        latest_same_protocol_retention_diagnostic=drift,
        latest_prepared_complete_continuation_data=str(prep_path.relative_to(ROOT)),
        current_goal_turn_classification='progress',all_ten_final_requirements_incomplete=True,
        candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph=(f'Retention and execution curriculum, **{at}**: the fixed packed-code token loss rises from '
        f'**{foundation:.4f} at completed foundation12853 to{latest:.4f}at step16384** '
        f'({100*drift["relative_increase_from_completed_foundation"]:.2f}%worse; still '
        f'{100*drift["relative_reduction_from_initial"]:.2f}%below initial). This is a retention concern, '
        'not generated-code correctness or proof of the cause. The new preparation adds **19,918,080 raw-code '
        'targets and19,918,080education-text targets**, keeps every58,488new-code batch unchanged and reconstructs '
        'all12replay-family extremes. The total replay-batch budget stays fixed, so individual earlier replay '
        'families receive fewer selected batches. **The new curriculum has not trained or proved recovery.**\n\n'
        'The original main run remains active on allfour L40s. Its fixed matched coding comparison continues on '
        'CPU. New independent readers check every retained reference stdout against original cases and will '
        'rebuild/screen/cold-read every public-only execution target before admitting a separate auxiliary CE '
        'pack. They wait for complete original jobs; no partial candidate is admitted, no execution-reward '
        'admission is granted and no pretrained teacher is added. All original assessment populations and '
        'queues remain intact. **No new functional coding, knowledge or visual gain is established; all ten '
        'final requirements remain incomplete.** '
        '[Review](reports/retention-execution-curriculum-v1-review/review.json), '
        '[new source preparation](reports/focused-code-continuation-sources-v2-preparation/complete.json).\n\n')
    for path in docs[:2]:
        first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),main_updates=progress['actual_updates'],
        latest_held_NLL=latest,relative_drift_from_foundation=drift['relative_increase_from_completed_foundation'],
        actual_added_raw_targets_each=19918080,new_training_started=False,goal_achieved=False)),flush=True)


if __name__=='__main__':main()
