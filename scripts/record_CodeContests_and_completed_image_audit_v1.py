"""Record complete new public code preparation without claiming model gains."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/CodeContests-and-completed-image-audit-v1-review'


def current(owner, waiting=False):
    proc = Path('/proc') / str(owner['pid'])
    fields = (proc / 'stat').read_text().split(') ', 1)[1].split()
    if fields[19] != owner['ticks'] or fields[0] in {'Z','X'}: raise ValueError('Exact live original birth identity required')
    children = (proc / 'task' / str(owner['pid']) / 'children').read_text().strip()
    visible = next((s.split('=',1)[1] for s in (proc / 'environ').read_bytes().decode().split('\0') if s.startswith('CUDA_VISIBLE_DEVICES=')), None)
    if waiting and (children or visible != ''): raise ValueError('Waiting original owner must remain CPU-hidden and childless')
    return dict(pid=owner['pid'], ticks=fields[19], state=fields[0], children=children, CUDA_VISIBLE_DEVICES=visible)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-hidden evidence review required')
    master_path = ROOT / 'reports/completion-audit.json'; master = json.loads(master_path.read_text())
    prior_path = ROOT / master['latest_full_scope_review']; prior = json.loads(prior_path.read_text())
    requirements = prior['requirements']
    original_path = ROOT / 'reports/code-contests-all-original-records-v1-readback/complete.json'
    prepared_path = ROOT / 'reports/code-contests-complete-curriculum-candidate-v1/complete.json'
    image_path = ROOT / 'reports/AI2D-current-native-image-exposure-v1/complete.json'
    groups_path = ROOT / 'reports/AI2D-fixed-exposure-strata-v1/preparation-complete.json'
    ce_path = ROOT / 'reports/own-execution-contrast-full-CPU-v2-outcome-readback/anchor_CE/complete.json'
    reports = {k:json.loads(p.read_text()) for k,p in [('original',original_path),('prepared',prepared_path),
        ('image',image_path),('groups',groups_path),('ce',ce_path)]}
    for report in reports.values(): FileBindings(report['source_files_sha256']).check(full=True)
    original, prepared, image, groups, ce = [reports[k] for k in ['original','prepared','image','groups','ce']]
    if (not original['every_original_record_and_extracted_byte_compared'] or original['complete_original_train_shards'] != 128
            or original['actual_counts']['problem_records'] != 13328 or original['training_admitted'] is not False
            or prepared['dataset_admitted_to_training'] is not False or prepared['model_updates'] != 0):
        raise ValueError('Complete original source readback and unadmitted prepared candidates required')
    if (image['actual_source_positions'] != 302627 or image['exact_original_diagram_file_matched_questions'] != 0
            or image['exact_actual_proper_native448_input_matched_questions'] != 0
            or groups['original_questions'] != 3088 or not groups['source_groups_fixed_before_original_GPU_visual_assessment']):
        raise ValueError('Every current image-source position and fixed original question group required')
    code = ce['independent_outcomes']['native_code']
    if ce['actual_raw_code_deliveries_reexecuted'] != 48 or any(code[f]['tasks'] != 24 or code[f]['correct'] != 0 for f in ['completion','instruction']):
        raise ValueError('Complete independently reexecuted CE48 local code endpoint required')
    ordinary = {}; ordinary_paths = []
    for endpoint in ['initial','anchor_CE','anchor_contrast']:
        for frame in ['completion','instruction']:
            path = ROOT / 'reports/own-execution-contrast-ordinary-Python-v1' / f'{endpoint}-{frame}-complete.json'
            if not path.exists(): continue
            row = json.loads(path.read_text()); FileBindings(row['source_files_sha256']).check(full=True)
            ordinary[endpoint+':'+frame] = row['actual_outcomes']; ordinary_paths.append(path)
    main_plan_path = ROOT / 'reports/phase-joint-v3/plan.json'; main_plan = json.loads(main_plan_path.read_text())
    progress = json.loads((ROOT / 'reports/phase-joint-v3/progress.json').read_text())
    workers = [current(r) for r in json.loads((ROOT / 'reports/phase-joint-v3/children.json').read_text()) if r['subject'].startswith('train-rank-')]
    if len(workers) != 4: raise ValueError('Every original main GPU worker must remain live')
    queue_names = ['phase-human-preference-v4','broad-anchored-code-v3','full-held-QA-pipeline-v1','full-AI2D-pipeline-v1']
    queues = {name:json.loads((ROOT / 'reports' / (name+'-queue.json')).read_text()) for name in queue_names}
    at = datetime.now(timezone.utc).isoformat()
    runtime = dict(at=at, original_main_owner=current(main_plan['owner']), original_main_workers=workers,
        main_actual_updates=progress['actual_updates'], main_planned_updates=35284, main_phase=progress['phase'],
        original_successors={name:current(q['owner'],True) for name,q in queues.items()},
        GPUs=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True))
    native = prepared['actual_native_pack_counts']; counts = original['actual_counts']
    requirements['public_data']['evidence'] += (
        f' Direct official CodeContests public GCS collection retains all128 original train shards and all13,328 original '
        f'problem protobuf records, including{counts["publisher_correct_solutions"]:,}publisher-correct and '
        f'{counts["publisher_incorrect_solutions"]:,}publisher-incorrect solutions kept separate. All19,000,649,096 original '
        'protobuf bytes were independently read back against their original publisher shards; every record roundtrips '
        'byte-identically under the original official protobuf schema. The bounded structural Riegeli reader uses publisher '
        'whole-file SHA256 rather than computing internal HighwayHash. Original platform/language/test inventories were '
        'rederived, with Python2 and Python3 distinct and all fields preserved. No validation/test shards were collected. '
        f'A full-source candidate pass visits all{counts["publisher_correct_solutions"]:,}positives and selects at most3whole '
        f'publisher-order programs per problem/language fitting4096 native tokens, producing{native["train"]["examples"]:,}train '
        f'and{native["validation"]["examples"]:,}internal-validation presentations after known-source quarantine. Exact original '
        'questions/code and EOS target masks are retained; private/generated tests never enter model prompts, incorrect code '
        'is not a positive CE target, and every excluded/unselected original remains in the raw inventory. Known APPS test/'
        'internal-validation and CodeNet held identities plus named full-question/MBPP program matches are quarantined across '
        'linked original problems. This is an unadmitted candidate pack: complete target readback, fresh source-program '
        'execution and further unresolved cross-platform/semantic overlap review remain pending. It adds no model updates '
        'or proof of capability; publisher correctness is not new local execution.')
    requirements['text_and_images']['evidence'] += (
        ' The complete current-image audit now visits all302,627original image/word source positions across9families, '
        'hashes195,121actual unique original files and reconstructs246,706actual source-protocol native inputs. None of the '
        '3,088held AI2D question images matches an original current-pool file or actual proper native448tensor. Word448 and '
        'CTC64x448 protocols remain separate. This establishes only exact absence in the inspected current pool; earlier '
        'unrecorded image ancestry, perceptual/crop variants and semantic independence are not certified. All3,088 remain '
        'primary in the queued6-endpoint full visual evaluation. Eight exposure strata were fixed before any GPU visual '
        'assessment:16generic-question matches,3,072without one,zero stronger exact text/image matches. No visual model '
        'score or quality gain follows from a data-source audit.')
    requirements['any_code_correctness_and_efficiency']['evidence'] += (
        ' The complete40-update CE-only experimental endpoint now has all48original native local deliveries independently '
        'reexecuted:11/24completion syntax and12/24instruction syntax,0/24functional and0/24empirical efficiency for each '
        'frame. Its3/4training-source fit does not generalize to this development suite. Every available ordinary-Python '
        'program frame is separately retained, with no runtime-only functional gain observed. The own-negative contrastive '
        'endpoint assessment continues; no functional advantage, final-model promotion, ANY-code correctness or efficient '
        'algorithm capability is established. New source corpus preparation is not a model improvement.')
    requirements['serious_full_model_training']['evidence'] += (
        f' The original own984399690-value full-weight main run remains active at{progress["actual_updates"]}/35284updates '
        'on its original four L40workers. New code-source preparation/readback uses CPU, leaves frozen training sources '
        'unchanged, and does not reset or replace the current epoch. The prior same-input10240held NLL0.7699061050079763 '
        '(11.49947084829329%below initial0.8699451996362768) remains token-prediction evidence, not generated quality.')
    requirements['gpu_runtime']['evidence'] += (
        ' Allfour original main worker birth identities and allfour human/code/text/visual successor owners were freshly '
        'checked; waiting successors remain CPU-hidden/childless. Allfour L40 utilization/memory readings are retained in '
        'the new immutable review. No additional GPU was attached, no lease was preempted, and no live experiment was edited.')
    if len(requirements) != 10 or any(r['status'] != 'incomplete_for_requested_final_model' for r in requirements.values()):
        raise ValueError('Every original final requirement must remain incomplete')
    OUT.mkdir()
    evidence = [Path(__file__),prior_path,original_path,prepared_path,image_path,groups_path,ce_path,main_plan_path,*ordinary_paths]
    reader_path = ROOT / 'reports/code-contests-complete-curriculum-v1-readback/complete.json'
    reader_plan_path = reader_path.with_name('plan.json')
    reader_state = dict(complete=reader_path.exists(), failure=reader_path.with_name('failure.json').exists())
    if reader_path.exists(): evidence.append(reader_path)
    elif reader_plan_path.exists():
        p=json.loads(reader_plan_path.read_text()); reader_state['original_live_owner']=current(p['owner']); evidence.append(reader_plan_path)
    report = dict(at=at,requirements=requirements,actual_runtime=runtime,
        complete_original_CodeContests_counts=counts, complete_candidate_native_counts=native,
        complete_candidate_original_problem_dispositions=prepared['actual_original_problem_dispositions'],
        prepared_candidate_not_admitted_to_training=True, new_native_target_readback_state=reader_state,
        actual_independently_reexecuted_CE_native_code=code, actual_ordinary_Python_program_outcomes=ordinary,
        complete_current_image_source_inventory={k:image[k] for k in ['actual_source_positions','actual_unique_original_image_files',
            'actual_unique_reconstructed_source_protocol_inputs','exact_original_diagram_file_matched_questions','exact_actual_proper_native448_input_matched_questions']},
        fixed_all3088_visual_source_groups=groups['source_group_sizes'],
        source_files_sha256={str(p):file_hash(p) for p in evidence},
        previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        no_remaining_original_scope_removed=True,all_ten_final_requirements_incomplete=True,
        new_generated_functional_gain_or_high_end_equivalence_established=False,candidate_promoted=False,goal_achieved=False)
    atomic_json(OUT / 'review.json',report)
    protected=set(main_plan['source_files_sha256'])
    for q in queues.values(): protected.update(q['source_files_sha256'])
    plans = ['reports/own-execution-contrast-full-CPU-v1/plan.json','reports/own-execution-contrast-full-CPU-v2-outcome-readback/plan.json',
        'reports/own-execution-contrast-ordinary-Python-v1/plan.json','reports/AI2D-fixed-exposure-strata-v1/plan.json',
        'reports/code-contests-complete-curriculum-candidate-v1/plan.json','reports/code-contests-complete-curriculum-v1-readback/plan.json']
    for name in plans:
        path=ROOT/name
        if path.exists(): protected.update(json.loads(path.read_text())['source_files_sha256'])
    docs=[ROOT/'README.md',ROOT/'RESEARCH.md',master_path]
    if any(str(p) in protected for p in docs):
        print(json.dumps(dict(review=str(OUT/'review.json'),bound_root_documents_preserved=True)),flush=True);return
    (ROOT/'reports/completion-audit-before-CodeContests-image-audit-v1.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={k:r['evidence'] for k,r in requirements.items()},
        latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),latest_actual_main_runtime=runtime,
        latest_CodeContests_prepared_candidate=str(prepared_path.relative_to(ROOT)),
        latest_complete_current_AI2D_image_audit=str(image_path.relative_to(ROOT)),
        latest_full_CE48_program_execution_readback=str(ce_path.relative_to(ROOT)),
        current_goal_turn_classification='progress',all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph=(f'Complete public CodeContests source preparation and image audit, **{at}**: '
        '**13,328 original training problems**, **4,432,447 publisher-correct** and **8,653,752 publisher-incorrect solutions** '
        'were collected directly from official GCS and retained separately. All 128 shards and 19.0 GB of original protobuf bytes '
        'were read back and their original schema, language and test counts rederived. Python 2 and 3 remain distinct. '
        f'The whole-context candidate curriculum contains **{native["train"]["examples"]:,} train** and '
        f'**{native["validation"]["examples"]:,} internal-validation presentations** after known held-source quarantine; '
        'all original data remain retained. It is **not training-admitted**: native target readback, local reference '
        'execution and further cross-platform/semantic source review remain pending. No new model update is attributed '
        'to this data preparation.\n\n'
        'The complete image audit visits **302,627 source positions**, 195,121 actual image files and 246,706 actual reconstructed '
        'source-protocol inputs. There are **zero exact original-file or native448 matches** for all 3,088 AI2D question images '
        'in that current pool; this is not a perceptual/all-ancestry independence certificate or visual quality result. '
        'All 3,088 questions and eight predeclared source groups remain in queued native visual evaluation. The full '
        '**CE40 local coding endpoint remains 0/48 correct** after fresh execution, despite 3/4 training-source fit. '
        f'The original main run remains active at **{progress["actual_updates"]:,}/35,284 updates on four L40s**; human '
        'preference, anchored code RL, full held text QA and full visual QA retain their original queues. '
        '**All ten final requirements remain incomplete.** '
        '[Review](reports/CodeContests-and-completed-image-audit-v1-review/review.json), '
        '[new corpus](data/code-contests-complete-curriculum-candidate-v1/manifest.json), '
        '[complete original-record readback](reports/code-contests-all-original-records-v1-readback/complete.json).\n\n')
    for path in docs[:2]:
        first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),new_source_problems=13328,native_candidate_counts=native,
        main_updates=progress['actual_updates'],goal_achieved=False)),flush=True)


if __name__ == '__main__': main()
