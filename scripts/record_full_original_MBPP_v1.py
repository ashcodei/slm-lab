"""Record complete original source controls and the actual waiting code judge."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/full-original-MBPP-v1-review'


def live(value,waiting=False):
    proc=Path('/proc')/str(value['pid']);fields=(proc/'stat').read_text().split(') ',1)[1].split()
    if fields[19]!=value['ticks'] or fields[0] in {'Z','X'}:raise ValueError('Exact live birth identity required')
    children=(proc/'task'/str(value['pid'])/'children').read_text().strip()
    visible=next((s.split('=',1)[1] for s in (proc/'environ').read_bytes().decode().split('\0') if s.startswith('CUDA_VISIBLE_DEVICES=')),None)
    if waiting and (visible!='' or children):raise ValueError('Waiting CPU owner must remain hidden and childless')
    return dict(pid=value['pid'],ticks=fields[19],state=fields[0],children=children,CUDA_VISIBLE_DEVICES=visible)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-hidden full-original review required')
    master_path=ROOT/'reports/completion-audit.json';master=json.loads(master_path.read_text())
    previous_path=ROOT/master['latest_full_scope_review'];requirements=json.loads(previous_path.read_text())['requirements']
    prep_path=ROOT/'reports/MBPP-all-reserved-original-v2-preparation/complete.json';prep=json.loads(prep_path.read_text())
    if (prep['actual_counts']['reference_fully_qualified_problems']!=600 or prep['actual_counts']['actual_passed_cases']!=1816
            or not prep['all600actual_original_reference_programs_executed'] or not prep['all600native_prompts_and_full512budgets_fit']):
        raise ValueError('Every original reference, input and original test/challenge case must actually pass')
    queue_path=ROOT/'reports/full-original-MBPP-pipeline-v1-queue.json';queue=json.loads(queue_path.read_text())
    FileBindings(queue['source_files_sha256']).check(full=True)
    if len(queue['endpoints'])!=8 or (ROOT/'reports/full-original-MBPP-pipeline-v1').exists():raise ValueError('Fixed eight-endpoint queue remains waiting')
    queued=live(queue['owner'],True)
    tests=[ROOT/'reports/mbpp-original-function-cases-v2-tests.log',ROOT/'reports/MBPP-all-native-evaluation-v1-final-tests.log',
        ROOT/'reports/full-original-MBPP-pipeline-v1-tests.log']
    for path,count in zip(tests,[4,2,2]):
        if str(count)+' passed' not in path.read_text():raise ValueError('Actual runtime, neural delivery, verdict and owner controls required')
    phase=ROOT/'reports/phase-joint-v3';plan=json.loads((phase/'plan.json').read_text());progress=json.loads((phase/'progress.json').read_text())
    workers=[live(r) for r in json.loads((phase/'children.json').read_text()) if r['subject'].startswith('train-rank-')]
    if len(workers)!=4:raise ValueError('Original four main GPU workers must remain live')
    protected={**plan['source_files_sha256'],**queue['source_files_sha256']};successors={}
    for name in ['phase-human-preference-v4','broad-anchored-code-v3','full-held-QA-pipeline-v1','full-AI2D-pipeline-v1','focused-code-continuation-v1']:
        path=ROOT/'reports'/(name+'-queue.json');q=json.loads(path.read_text());successors[name]=live(q['owner'],True);protected.update(q['source_files_sha256'])
    jobs={}
    for name in ['CodeContests-full-Python-references-v1','CodeContests-public-execution-traces-candidate-v1','CodeContests-reference-outputs-v1-readback',
        'public-execution-targets-v1-readback-admission','main-foundation-instruction14336-code-CPU-v2','AI2D-fixed-exposure-strata-v1']:
        path=ROOT/'reports'/name/'plan.json';value=json.loads(path.read_text())
        if path.with_name('failure.json').exists():raise ValueError('Live source or assessment failed: '+name)
        jobs[name]=live(value['owner']);protected.update(value['source_files_sha256'])
    foundation_path=ROOT/'reports/main-foundation-instruction14336-code-CPU-v2/foundation12853/complete.json'
    foundation=json.loads(foundation_path.read_text())
    if foundation['actual_results']['instruction']['native_functionally_correct']!=0 or foundation['actual_results']['completion']['native_functionally_correct']!=0:
        raise ValueError('This review expects the actual older completed-foundation observation')
    at=datetime.now(timezone.utc).isoformat()
    runtime=dict(at=at,main_owner=live(plan['owner']),original_four_main_workers=workers,actual_main_updates=progress['actual_updates'],
        planned_main_updates=35284,original_and_continuation_queues=successors,new_MBPP_queue_owner=queued,original_CPU_jobs=jobs,
        GPU_observation=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True))
    requirements['broad_independent_benchmarks']['evidence'] += (
        ' All600original reserved publisher MBPP records1through600have now been rebuilt byte/field-exact from the '
        'pinned original source, without source repair or supported-subset filtering. All1816original test/challenge '
        'equalities are retained. The confined child evaluates unchanged source, complete input setup and original '
        'left expressions only; expected right-hand values remain in the trusted parent and are never written to '
        'candidate filesystem/process. Data-only transport preserves tuple/list/set/dictionary/complex equality and '
        'explicit Counter values. The first completed source pass qualified598/600because Counter outputs of '
        'problems40and88were unsupported by the old transport. Those closed source/control artifacts remain intact '
        'with a diagnosis; they are not relabelled algorithm errors. A separately versioned corrected complete '
        'source pass actually executes all600again and passes every1816original case. All600native prompts and '
        'whole512greedy budgets fit. Prompts contain only descriptions/public interface headers and required input '
        'attribute names, no source bodies or original test/gold values. Publisher test500, validation90and prompting10 '
        'remain separate; source qualification never removes a primary model question. No official few-shot, '
        'semantic/all-ancestry independence or general/asymptotic-efficiency claim is made. A newly live frozen '
        'four-L40 queue fixes all eight original/focused model endpoints before their new MBPP outputs. It waits '
        'for the exact complete continuation, its source epochs, original QA/visual assessments, native cold-load '
        'receipts and worker restoration, then will retain all4800native program deliveries and all original '
        'execution/raw-value evidence. That NN assessment has not started.')
    requirements['any_code_correctness_and_efficiency']['evidence'] += (
        ' Four actual stdlib/runtime tests cover globals/classes/input setup, exact tuple/list/set/dictionary/complex '
        'data equality, original challenge coverage, no expected values in worker settings, infinite execution '
        'bounds, strict parent-literal evaluation and actual/nested Counter transport. Two evaluator tests include '
        'a real small own model delivering all512actual greedy actions on CPU, preserving unchanged parameters '
        'and retaining an empty-code failure; the other executes actual original cases and rejects changed raw '
        'values/action receipts. Two ownership tests fix eight endpoints and reject live/partial predecessors or '
        'incomplete native cold-load receipts. All eight tests pass; none is a full-size useful-model score. The '
        'older main foundation12853observer has now completed BOTH original frames:20/48syntax-valid and0/48native '
        'functional/empirical-efficiency passes; all24Python deliveries also remain0under the secondary ordinary '
        'runtime. Its matched instruction14336assessment is still live. The completed600/600reference controls '
        'prove evaluator/source coverage, not learned model capability. No new coding accuracy gain is established.')
    requirements['gpu_runtime']['evidence'] += (
        ' Exact process-birth checks confirm the new MBPP evaluation successor is CPU-hidden and childless, all '
        'original five GPU training/text/visual/continuation queues remain live and hidden/childless, and all four '
        f'original main GPU ranks continue at{progress["actual_updates"]}/35284updates. Both full source control passes '
        'completed terminally on CPU; no GPU or model training was used for their600original programs. All original '
        'source/reference/output jobs and the fixed matched code observer remain live. No job was restarted for '
        'age/observation timeout, no original work was preempted and no new GPU capacity or learned gain is claimed.')
    if len(requirements)!=10 or any(v['status']!='incomplete_for_requested_final_model' for v in requirements.values()):
        raise ValueError('All original final requirements remain incomplete')
    OUT.mkdir();evidence=[Path(__file__),previous_path,prep_path,queue_path,foundation_path,*tests,
        ROOT/'reports/MBPP-v1-Counter-qualification-diagnosis.json',ROOT/'reports/MBPP-all-native-evaluation-v1-tests.log']
    atomic_json(OUT/'review.json',dict(at=at,requirements=requirements,previous_full_scope_review=str(previous_path.relative_to(ROOT)),
        actual_full_original_MBPP_controls=prep,actual_runtime=runtime,older_main_foundation_complete_code_observation=foundation,
        source_files_sha256={str(p):file_hash(p) for p in evidence},previous_goal_turn_classification='progress',
        current_goal_turn_classification='progress',all_ten_final_requirements_incomplete=True,
        no_original_scope_removed=True,new_full_MBPP_model_assessment_pending=True,candidate_promoted=False,goal_achieved=False))
    docs=[ROOT/'README.md',ROOT/'RESEARCH.md',master_path]
    if any(str(p) in protected for p in docs):
        print(json.dumps(dict(review=str(OUT/'review.json'),bound_root_documents_preserved=True)),flush=True);return
    (ROOT/'reports/completion-audit-before-full-original-MBPP-v1.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={k:v['evidence'] for k,v in requirements.items()},
        latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),latest_actual_main_runtime=runtime,
        latest_full_original_MBPP_preparation=str(prep_path.relative_to(ROOT)),latest_full_original_MBPP_queue=str(queue_path.relative_to(ROOT)),
        current_goal_turn_classification='progress',all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph=(f'Full original coding assessment, **{at}**: all **600 reserved MBPP problems and 1,816 original '
        'test/challenge cases** are retained. The corrected runtime executes every original reference program and '
        'passes **600/600 programs and 1,816/1,816 cases**, including classes, globals, standard-library imports and '
        'Counter values. Expected answers stay in the trusted parent; the confined program receives only input '
        'setup and expressions to evaluate. The earlier transport limitation for two Counter-returning references '
        'is preserved and diagnosed. These are **reference controls, not model accuracy**.\n\n'
        'An actual frozen queue will assess all **eight fixed model endpoints**, retaining all **4,800 native '
        'program deliveries** with 512 greedy actions and complete original execution outcomes. The publisher’s '
        '**500 test problems** remain primary; its 90 validation and 10 prompting examples are separate. Nothing '
        'is removed for reference eligibility. This custom native protocol does not establish an official '
        'few-shot score or asymptotic efficiency. Eight runtime/neural-delivery/ownership tests pass. The queue '
        'is waiting. The older completed-foundation checkpoint now has **0/48 functional passes and 20/48 syntax '
        'passes** across both original local frames; the matched later checkpoint is still being assessed. '
        '**No new coding gain is established; all ten final requirements remain incomplete.** '
        '[Review](reports/full-original-MBPP-v1-review/review.json), '
        '[complete source controls](reports/MBPP-all-reserved-original-v2-preparation/complete.json), '
        '[fixed model queue](reports/full-original-MBPP-pipeline-v1-queue.json).\n\n')
    for path in docs[:2]:
        first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),reference_programs_passed=600,original_cases_passed=1816,
        actual_new_waiting_MBPP_owner=queued,main_updates=progress['actual_updates'],new_model_MBPP_deliveries=0,goal_achieved=False)),flush=True)


if __name__=='__main__':main()
