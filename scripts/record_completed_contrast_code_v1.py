"""Record the completed matched code trial and broader Python interpretation."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/completed-own-contrast-code-v1-review'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-only completed trial review required')
    master_path=ROOT/'reports/completion-audit.json';master=json.loads(master_path.read_text())
    previous_path=ROOT/master['latest_full_scope_review'];previous=json.loads(previous_path.read_text())
    code_path=ROOT/'reports/own-execution-contrast-full-CPU-v2-outcome-readback/complete.json'
    ordinary_path=ROOT/'reports/own-execution-contrast-ordinary-Python-v1/complete.json'
    code=json.loads(code_path.read_text());ordinary=json.loads(ordinary_path.read_text())
    FileBindings(code['source_files_sha256']).check(full=True);FileBindings(ordinary['source_files_sha256']).check(full=True)
    if (not code['all144_native_development_code_and12_actual_train_source_programs_freshly_executed']
            or not ordinary['all72_actual_Python_programs_and12_original_reference_main_guard_controls_executed']):
        raise ValueError('All original code endpoint, source and broader runtime assessments required')
    comparison={}
    for endpoint in ['initial','anchor_CE','anchor_contrast']:
        result=code['independent_outcomes'][endpoint];frames=result['native_code']
        if any(frames[f]['tasks'] != 24 or frames[f]['empirical_timing_disagreements'] for f in ['completion','instruction']):
            raise ValueError('Every original native case and remeasured timing outcome required')
        comparison[endpoint]=dict(local_native_deliveries=48,
            syntax_valid=sum(frames[f]['syntax_passes'] for f in frames),
            functionally_correct=sum(frames[f]['correct'] for f in frames),
            empirical_efficiency_stress_passes=sum(frames[f]['empirical_efficiency_stress_passes'] for f in frames),
            four_original_training_problems_passed=result['public_train_finite_case_passes'],frames=frames)
    if (comparison['initial']['syntax_valid'],comparison['anchor_CE']['syntax_valid'],comparison['anchor_contrast']['syntax_valid']) != (22,23,31):
        raise ValueError('This review requires the actual complete22/23/31 syntax observations')
    if any(r['functionally_correct'] for r in comparison.values()) or any(r['ordinary_Python_correct'] for r in ordinary['actual_outcomes'].values()):
        raise ValueError('This review requires the actual zero-functional-gain outcomes')
    requirements=previous['requirements']
    requirements['any_code_correctness_and_efficiency']['evidence'] += (
        ' The original own-negative40-update controlled trial is now fully assessed and independently reexecuted. '
        'All144local native code deliveries and12actual training-source programs were freshly executed without repair. '
        'Across the same24completion plus24instruction cases, syntax-valid programs increase from22/48initial to23/48CE '
        'and31/48own-negative contrast; functional and empirical efficiency remain0/48at every endpoint. Both trained '
        'arms pass3/4original training problems, versus0/4initial. The contrast arm emits syntactically valid Python for '
        'all12functions in each frame, but all remain functionally incorrect. Every one of72actual Python programs '
        'across all3endpoints/two frames was also executed under the broader ordinary stdlib/module runtime:0correct, '
        'no runtime-only gained/lost task, with all12canonical main-guard references passing. Thus the custom objective '
        'improves local syntactic validity, with no generated algorithmic correctness/efficiency gain or advantage in '
        'training-source finite-case passes. The two own984M-base/rank16actor experimental students have999706442values '
        'and are separate from the original active full-weight main run. No student is promoted. The48cases are '
        'local development deliveries, not an independent final benchmark or evidence of frontier-level capability.')
    if len(requirements)!=10 or any(r['status']!='incomplete_for_requested_final_model' for r in requirements.values()):
        raise ValueError('All ten requested final requirements remain incomplete')
    at=datetime.now(timezone.utc).isoformat();progress=json.loads((ROOT/'reports/phase-joint-v3/progress.json').read_text())
    OUT.mkdir();report=dict(previous,at=at,requirements=requirements,previous_full_scope_review=str(previous_path.relative_to(ROOT)),
        completed_matched_own_contrast_trial=comparison,completed_all72_ordinary_Python=ordinary['actual_outcomes'],
        main_published_updates_observed=progress['actual_updates'],all_ten_final_requirements_incomplete=True,
        experimental_syntactic_validity_gain=True,experimental_functional_or_efficiency_gain=False,
        source_files_sha256={str(p):file_hash(p) for p in [Path(__file__),previous_path,code_path,ordinary_path]},
        current_goal_turn_classification='progress',candidate_promoted=False,goal_achieved=False)
    atomic_json(OUT/'review.json',report)
    plans=[ROOT/'reports/phase-joint-v3/plan.json',ROOT/'reports/code-contests-complete-curriculum-v1-readback/plan.json',
        ROOT/'reports/AI2D-fixed-exposure-strata-v1/plan.json']
    plans += [ROOT/'reports'/(n+'-queue.json') for n in ['phase-human-preference-v4','broad-anchored-code-v3','full-held-QA-pipeline-v1','full-AI2D-pipeline-v1']]
    protected=set()
    for p in plans:protected.update(json.loads(p.read_text())['source_files_sha256'])
    docs=[ROOT/'README.md',ROOT/'RESEARCH.md',master_path]
    if any(str(p) in protected for p in docs):
        print(json.dumps(dict(review=str(OUT/'review.json'),bound_root_documents_preserved=True)),flush=True);return
    (ROOT/'reports/completion-audit-before-completed-own-contrast-code-v1.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={k:r['evidence'] for k,r in requirements.items()},
        latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),
        latest_completed_own_contrast_code=comparison,latest_completed_all72_ordinary_Python=str(ordinary_path.relative_to(ROOT)),
        current_goal_turn_classification='progress',all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph=(f'Completed own-negative code trial, **{at}**: independently executed all **144 local native deliveries** '
        'and **12 training-source programs**. On the same 48 local code presentations, syntax validity is '
        '**22/48 initial, 23/48 CE, 31/48 contrastive**; **functional correctness and empirical efficiency remain 0/48** '
        'at every endpoint. Both trained arms fit 3/4 original training problems. All **72 actual Python programs** '
        'also remain incorrect under the broader ordinary Python runtime, with 12 canonical main-guard controls passing. '
        'The custom objective improves syntax in this small experiment; it has no demonstrated general functional gain '
        'and is not promoted into the separate active main model. The new 65,395-example public-code candidate remains '
        'unadmitted while its complete target readback and further source/execution review proceed. '
        '**All ten requested final requirements remain incomplete.** '
        '[Completed comparison](reports/completed-own-contrast-code-v1-review/review.json).\n\n')
    for path in docs[:2]:
        first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),syntax=[22,23,31],functional=[0,0,0],goal_achieved=False)),flush=True)


if __name__=='__main__':main()
