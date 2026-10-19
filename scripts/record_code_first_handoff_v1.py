"""Record the applied code-first pipeline without claiming a learned gain."""
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import observe,terminal
from slm_lab.code_first_endpoints_v1 import ENDPOINTS,descriptors

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/code-first-handoff-v1-review'
NAMES=('code-first-continuation-v1','phase-human-preference-v5','broad-anchored-code-v4',
    'code-first-final-standalone-CPU-v1','full-held-QA-pipeline-v2','full-AI2D-pipeline-v2',
    'full-original-MBPP-pipeline-v2','AI2D-fixed-exposure-strata-v2')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-hidden full-scope review required')
    handoff_path=ROOT/'reports/code-first-v1-handoff/complete.json';handoff=json.loads(handoff_path.read_text())
    if (handoff.get('old_unstarted_controllers_all_terminal') is not True
            or handoff.get('explicit_code_first_handoff_complete') is not True
            or len(handoff['retired_exact_processes'])!=7
            or not all(terminal(p) for p in handoff['retired_exact_processes'])):
        raise ValueError('Actual complete terminal handoff required, not an implementation plan')
    master_path=ROOT/'reports/completion-audit.json';master=json.loads(master_path.read_text())
    previous_path=ROOT/master['latest_full_scope_review'];prior=json.loads(previous_path.read_text());requirements=deepcopy(prior['requirements'])
    tests={'code-first-driver-and-standalone-v1-tests.log':4,'phase-human-policy-v2-final-tests.log':2,
        'broad-anchored-policy-v2-tests.log':2,'phase-human-preference-v5-driver-tests.log':3,
        'broad-anchored-code-v4-driver-tests.log':4,'own-dense-finalizer-v1-tests.log':1,
        'code-first-finalizer-driver-v1-tests.log':2,'code-first-full-assessment-drivers-v1-tests.log':11,
        'code-first-visual-reader-v2-tests.log':2,'exact-waiting-process-retirement-v1-final-tests.log':1}
    for name,count in tests.items():
        if f'{count} passed' not in (ROOT/'reports'/name).read_text():raise ValueError('All actual focused controls must pass')
    plan_path=ROOT/'reports/phase-joint-v3/plan.json';plan=json.loads(plan_path.read_text())
    if plan['owner']!=handoff['original_main_owner']:raise ValueError('Exact main owner retained through handoff required')
    main_proc=Path('/proc')/str(plan['owner']['pid']);fields=(main_proc/'stat').read_text().split(') ',1)[1].split()
    if fields[19]!=plan['owner']['ticks'] or fields[0] in {'Z','X'}:raise ValueError('Current original main must remain live for this snapshot')
    progress=json.loads((plan_path.parent/'progress.json').read_text());queues={};protected=dict(plan['source_files_sha256'])
    for name in NAMES:
        path=ROOT/'reports'/(name+'-queue.json');q=json.loads(path.read_text());queues[name]=dict(owner=observe(q['owner']),queue=str(path),queue_sha256=file_hash(path))
        protected.update(q['source_files_sha256'])
    workers=[p for p in json.loads((plan_path.parent/'children.json').read_text()) if p['subject'].startswith('train-rank-')]
    for identity in workers:
        f=(Path('/proc')/str(identity['pid'])/'stat').read_text().split(') ',1)[1].split()
        if f[19]!=identity['ticks'] or f[0] in {'Z','X'}:raise ValueError('All four original main workers remain required')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True)
    source_jobs={}
    for name in handoff['original_public_source_owners']:
        folder=ROOT/'reports'/name;p=json.loads((folder/'plan.json').read_text());identity=p['owner']
        if (folder/'failure.json').exists():raise ValueError('Original public source job failed')
        source_jobs[name]=dict(owner=identity,terminal=terminal(identity),complete=(folder/'complete.json').is_file())
        if source_jobs[name]['terminal'] and not source_jobs[name]['complete']:raise ValueError('Original public source epoch may not be dropped')
        protected.update(p['source_files_sha256'])
    at=datetime.now(timezone.utc).isoformat()
    changes={
        'own_architecture':' A new complete-dense standalone format copies an already dense main without inventing actor-fold math. Later the completed own code actor is folded into a single dense student, with actual full-context text/photo/real-word probes, every-parameter cold reload and RNG/source preservation required. Small actual trained networks pass these controls; the new full-size gates/export have not run. This is a custom implementation and curriculum change, not a globally novel architecture or capability claim.',
        'public_data':' The applied code-first queue requires all58488screened whole CodeContests code examples and every separately admitted public execution-prediction example exactly once. No candidate output is admitted early. One original public input and actual stdout form each auxiliary shared-vocabulary target; no private/generated case value enters this training text. All12text/raw-code/education/photo/visual/word replay families remain, with explicitly selected repeats. This is not all public data or a new source-independent benchmark certificate.',
        'under_1b_parameters':' The revised fixed lineage retains the actual984399690-value full student and999706442-value human/code actor policies. The final single dense student is planned at984399690including vision, word, auxiliary and critic values and one tied vocabulary. Loader/gate checks enforce these counts; the completed final export still does not exist.',
        'text_and_images':' Ordinary photographs, original visual reasoning, image reading and real-word replay remain in the code-first curriculum. All3088original visual questions with proper/gray/fixed-wrong native448images are queued at eight fixed endpoints. Pre-score source groups are reused byte-for-byte from the already completed original preparation. No fresh visual gain is inferred.',
        'human_feedback':' Updated human-policy loading requires the complete code-first source and its complete-main ancestry. Matched chosenCE and humanDPO still each consume all85786original human-ranked public training pairs (5362updates each), and held4642pairs remain. These are public rankings of other-model responses, not newly collected human judgments on our model. Fresh four-L40 capacity/recovery remains mandatory before actual post-training.',
        'serious_full_model_training':f' The original main continues unchanged at{progress["actual_updates"]}/35284updates in this snapshot. A concrete tested code-first pipeline is now queued immediately after its complete source epoch/quick assessments and public-output admission, before human training and executionRL. Heavy full QA/image/MBPP scoring moves after learning. Only seven exact CUDA-hidden childless never-started old controllers were retired through verified pidfds after all replacements existed. All old files/plans/preparation are retained; no running training or public-source job was stopped. New code-first production updates remain0.',
        'high_end_llm_capability':' The code-first reordering addresses the observed semantic coding failures but provides no measured new capability. Complete matched main coding remains0/48functional and empirical-efficiency passes at both fixed endpoints, with syntax20/48to36/48. There is no evidence of high-end-LLM parity, arbitrary-code mastery or broad final-model excellence.',
        'broad_independent_benchmarks':' Eight fixed endpoints now cover main initial/main complete/code-first complete/chosenCE/humanDPO/codeCE/codeExecution/final standalone. Each retains all16961text questions, all3088visual questions times3conditions, and all600original MBPP problems/1816finite cases, with primary500publisher test problems separate from90validation/10prompting. Planned totals are135688text,74112visual-condition and4800MBPP problem deliveries. Native48code, held64APPS and4642human-pair comparisons remain in training controllers. Source exposure and custom protocols stay disclosed. Full outcomes and independent final output readback remain incomplete.',
        'gpu_runtime':' The same four original L40 workers remain live and exclusively own the GPUs. All eight replacement controllers are hidden and childless, awaiting exact predecessors and restoration. A tested native-kernel pidfd path handles the older conda Python/libc lacking wrappers and prevents signalling a reused numeric PID. The controlled test actually terminates one CPU child while preserving another. Fresh full-size capacity and actual recovery, not small tests or older weight calibration, still gate every new training stage.',
        'any_code_correctness_and_efficiency':' The new code-first objective supplies full original-code CE/EOS, actual public-output prediction and12-family retention replay before sparse execution rewards. Both later code arms retain all3721original sources, full CE anchors and actual case-based feedback; neither source references nor benchmark cases are silently filtered. Full original MBPP uses ordinary Python raw-value observations and every original case; finite tests do not prove arbitrary-code or asymptotic efficiency. No new functional score is available.'}
    for key,text in changes.items():requirements[key]['evidence']+=text
    if len(requirements)!=10 or any(v['status']!='incomplete_for_requested_final_model' for v in requirements.values()):raise ValueError('Every original final requirement remains incomplete')
    runtime=dict(at=at,actual_main_updates=progress['actual_updates'],planned_main_updates=35284,
        original_main_owner=plan['owner'],original_four_main_workers=workers,actual_GPU_observation=gpu,
        replacement_queues=queues,original_public_source_jobs=source_jobs,new_code_first_updates=0)
    OUT.mkdir()
    pipeline=dict(at=at,status='implemented_tested_handed_off_and_queued_not_yet_training',training_order=[
        'complete current original main35284','complete all58488code and admitted public outputs with12family replay',
        'complete85786human pairs each chosenCE/DPO','complete3721code sources each anchorCE/executionRL',
        'CPU fold and standalone cold-load check','eight-endpoint full QA, AI2D and original MBPP assessments'],
        endpoints=descriptors(ROOT),focused_tests_passed=sum(tests.values()),actual_code_first_production_updates=0,
        full_size_new_capacity_and_recovery_still_required=True,measured_new_quality_gain=False,all_ten_final_requirements_incomplete=True,goal_achieved=False)
    atomic_json(OUT/'pipeline.json',pipeline)
    evidence=[Path(__file__),handoff_path,previous_path,OUT/'pipeline.json',*[ROOT/'reports'/name for name in tests]]
    atomic_json(OUT/'review.json',dict(at=at,requirements=requirements,previous_full_scope_review=str(previous_path.relative_to(ROOT)),
        actual_runtime=runtime,applied_code_first_pipeline=pipeline,source_files_sha256={str(p):file_hash(p) for p in evidence},
        previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True,no_original_scope_removed=True,candidate_promoted=False,goal_achieved=False))
    docs=[ROOT/'README.md',ROOT/'RESEARCH.md',master_path]
    if any(str(p) in protected for p in docs):raise ValueError('Review complete; bound root documents must stay unchanged')
    (ROOT/'reports/completion-audit-before-code-first-handoff-v1.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={k:v['evidence'] for k,v in requirements.items()},
        latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),latest_actual_main_runtime=runtime,
        latest_code_first_handoff=str(handoff_path.relative_to(ROOT)),latest_code_first_pipeline=str((OUT/'pipeline.json').relative_to(ROOT)),
        latest_queued_new_code_continuation='reports/code-first-continuation-v1-queue.json',
        latest_full_original_MBPP_queue='reports/full-original-MBPP-pipeline-v2-queue.json',
        next_code_first_design_status='implemented_tested_handed_off_and_queued',current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph=(f'Code-first pipeline applied, **{at}**: the original four-L40 main continues at **{progress["actual_updates"]}/35,284 updates**. '
        'The tested replacement order is complete main → **58,488 public code examples plus all admitted public execution-output targets**, '
        'with12-family broad/code/photo/word replay → human chosen-CE/DPO → code CE/execution RL → standalone export and full final assessment. '
        '**32 focused checks pass. New code-first updates are still0**, and fresh full-size GPU capacity/recovery remains required. '
        'Seven exact hidden childless unstarted old controllers were explicitly retired after all eight replacements existed. '
        'The active training, four GPU workers, public-source jobs and every old scientific artifact are preserved.\n\n'
        'All **16,961 text questions**, **3,088 visual questions ×3image conditions**, and **600original MBPP problems** remain queued at eight fixed lineage endpoints. '
        'The latest complete matched coding result remains **0/48 functional passes** at both main checkpoints, despite syntax improving20/48to36/48. '
        '**No new capability gain or frontier parity is established; all ten final requirements remain incomplete.** '
        '[Applied pipeline](reports/code-first-handoff-v1-review/pipeline.json), [handoff](reports/code-first-v1-handoff/complete.json), '
        '[full review](reports/code-first-handoff-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for path in docs[:2]:
        first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),main_updates=progress['actual_updates'],new_code_first_updates=0,
        focused_tests_passed=sum(tests.values()),new_pipeline_queued=True,all_ten_final_requirements_incomplete=True,goal_achieved=False)))


if __name__=='__main__':main()
