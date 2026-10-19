"""Record actual trace-label research while preserving every final requirement."""
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import gzip
import json
import os
from pathlib import Path
import subprocess

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import observe,terminal

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/public-state-trace-research-v1-review'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only complete-scope research review required')
    pilot_path=ROOT/'reports/public-state-trace-pilot-v1/complete.json';reader_path=ROOT/'reports/public-state-trace-pilot-v1-readback/complete.json'
    pilot=json.loads(pilot_path.read_text());reader=json.loads(reader_path.read_text())
    for value in [pilot,reader]:FileBindings(value['source_files_sha256']).check(full=True)
    if (pilot['original_predeclared_source_population']!=64 or pilot['actual_counts']['complete_native_candidates']!=29
            or reader['actual_counts']['all29_complete_native_targets_rebuilt']!=29 or reader['all29_whole_state_traces_reproduced'] is not True
            or reader['actual_counts']['actual_original_reference_case_outputs_checked']!=6659
            or reader['actual_counts']['actual_state_trace_response_EOS_targets']!=30076
            or reader['actual_counts']['stdout_only_response_EOS_targets_same_cases']!=123
            or reader['dataset_admitted_to_training'] is not False):raise ValueError('Every exact actual source/native/repeated-state pilot outcome required')
    tests={'public-python-state-trace-v1-final-tests.log':4,'public-state-trace-target-v1-tests.log':2,'public-state-trace-readback-v1-tests.log':1}
    for name,count in tests.items():
        if f'{count} passed' not in (ROOT/'reports'/name).read_text():raise ValueError('Actual bounded source-state and independent target controls required')
    labels=Counter()
    with gzip.open(ROOT/'data/public-state-trace-pilot-candidate-v1/original-rows.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line)
            for event in json.loads(row['response'])['trace']:
                kind=event[0];labels['event:'+kind]+=1;changes=event[4] if kind=='call' else event[3]
                labels['changed_bindings']+=len(changes)
                for value in changes.values():labels['binding_type:'+value[0]]+=1
                if kind=='return':labels['return_value_type:'+event[5][0]]+=1
    if labels['changed_bindings']!=702:raise ValueError('Every actual local-state label denominator must match')
    master_path=ROOT/'reports/completion-audit.json';master=json.loads(master_path.read_text());previous_path=ROOT/master['latest_full_scope_review']
    previous=json.loads(previous_path.read_text());requirements=deepcopy(previous['requirements'])
    main_folder=ROOT/'reports/phase-joint-v3';main_plan=json.loads((main_folder/'plan.json').read_text());progress=json.loads((main_folder/'progress.json').read_text())
    if terminal(main_plan['owner']):raise ValueError('This snapshot requires the unchanged original main still running')
    workers=[r for r in json.loads((main_folder/'children.json').read_text()) if r['subject'].startswith('train-rank-')]
    if len(workers)!=4 or any(terminal(r) for r in workers):raise ValueError('All four original main GPU workers required')
    protected=dict(main_plan['source_files_sha256']);queues={}
    handoff=json.loads((ROOT/'reports/code-first-v1-handoff/complete.json').read_text())
    for name in handoff['new_waiting_pipeline']:
        path=ROOT/'reports'/(name+'-queue.json');q=json.loads(path.read_text());queues[name]=observe(q['owner']);protected.update(q['source_files_sha256'])
    source_jobs={}
    for name in ['CodeContests-full-Python-references-v1','CodeContests-public-execution-traces-candidate-v1',
            'CodeContests-reference-outputs-v1-readback','public-execution-targets-v1-readback-admission','public-state-traces-full-v1']:
        folder=ROOT/'reports'/name;plan=json.loads((folder/'plan.json').read_text());identity=plan['owner'];done=(folder/'complete.json').is_file()
        if (folder/'failure.json').exists() or (terminal(identity) and not done):raise ValueError('Every original/new public-source owner must remain live or complete')
        source_jobs[name]=dict(owner=identity,live=not terminal(identity),complete=done)
        if (folder/'progress.json').exists():source_jobs[name]['actual_progress']=json.loads((folder/'progress.json').read_text()).get('actual_counts',{})
        protected.update(plan['source_files_sha256'])
    at=datetime.now(timezone.utc).isoformat()
    changes={
        'own_architecture':' New auxiliary code-state targets use the same own vocabulary and language head with0new neural values. A lossless delta transport preserves observed CPython source-frame call/line/return/exception events, non-dunder local changes/deletions and tagged return data. Non-data objects are explicit opaque types; standard-library/external internals are not observed. This is research inspired by execution-pretraining literature, not a globally novel architecture claim or a completed model improvement.',
        'public_data':' A fixed-before-observation pilot covers64first fully case-qualified original public training references, always original public input0. There are62instrumented runs,36whole bounded traces whose stdout byte-matches the retained original run,29complete4096-fitting candidates,7context qualifications,25bounded/incomplete observations,1unframed-output transport qualification and2sources without public inputs. No failed trace is clipped, repaired or silently promoted. A separate full-source preparation now declares all3959original references, accounts for each, and requires two identical fresh-interpreter state observations before a native candidate can be written. All new state candidates remain unadmitted; original stdout-only admission and training pipelines are unchanged.',
        'under_1b_parameters':' Trace preparation/readback loads no neural network or external pretrained teacher. It adds0parameters and changes no984399690-value dense model or999706442-value actor. Exact final under1B exported quality remains incomplete.',
        'text_and_images':' The active original main and queued12-family text/code/education/photo/visual/word replay remain unchanged. This new code-data mechanism provides no new image capability evidence or substitute for full visual assessment.',
        'human_feedback':' The human chosenCE/DPO85786-pair epochs and held4642human comparisons remain queued unchanged. Program-generated state observations are machine feedback, not new human judgments or completion of genuine own-output human RLHF.',
        'serious_full_model_training':f' The original four-L40 main has advanced to{progress["actual_updates"]}/35284actual updates. All eight code-first successors remain hidden and childless behind their exact predecessors. This turn adds a completed source-state data experiment, independent raw-target reconstruction and real repeated-state stability checks; it performs0new model updates and does not alter any frozen training schedule or old scientific artifact.',
        'high_end_llm_capability':' A denser learning signal is not measured model improvement. Pilot targets contain30076response/EOS tokens versus123for stdout-only on the same29cases (244.52032520325204times), including event/JSON framing. Actual observed content includes702local binding changes:437integer,164list,33string,21dictionary,2boolean and45opaque-type bindings. These are event occurrences, not distinct knowledge facts or evidence of frontier parity. No new functional coding or broader final-model score is available.',
        'broad_independent_benchmarks':' Every original code/QA/image/MBPP/human/APPS evaluation population, endpoint and native budget remains queued. The independent pilot reader reconstructs64source dispositions and6659retained original public/private/generated case-output verdicts, then every one of29native candidate prompts/targets/BPE masks/EOS spans without the producer formatter. It repeats all29candidate instrumentations in fresh interpreters and all observed states/stdout match. Repetition uses the same trace worker, not an independent trace engine, and the original6659case executions are not redundantly rerun. Full source completion, held-overlap screening and a separate admission are still required for new state training data. This is training-source data evidence, not a held benchmark.',
        'gpu_runtime':' The same original four GPU workers continue uninterrupted. New state work is CPU-only; all eight queued model/evaluation controllers remain waiting. The full-state producer is live with an exact declared birth and will finalize only after every original source disposition and the exact original reference owner completes. No observation deadline restarts or stops an existing job.',
        'any_code_correctness_and_efficiency':' Execution-state supervision is being prepared to address semantic failures despite improved code syntax. Seven actual tests cover nested function/loop/buffer output, real before-line states, containers/deletions/caught exceptions, infinite/cyclic/bounded traces, worker input separation, full own native tokens/EOS masks/context bounds, and damaged source/input/raw-state rejection. All pass. The latest complete matched main result remains0/48functional/empirical-efficiency passes at both fixed checkpoints, with syntax20/48to36/48. These new source labels do not change that measured outcome.'}
    for key,text in changes.items():requirements[key]['evidence']+=text
    if len(requirements)!=10 or any(v['status']!='incomplete_for_requested_final_model' for v in requirements.values()):raise ValueError('Every original final requirement stays incomplete')
    runtime=dict(at=at,main_owner=main_plan['owner'],actual_main_updates=progress['actual_updates'],planned_main_updates=35284,
        original_four_main_workers=workers,waiting_code_first_pipeline=queues,public_source_jobs=source_jobs,
        actual_GPU_observation=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
        new_state_model_updates=0,new_code_first_model_updates=0)
    OUT.mkdir();atomic_json(OUT/'actual-label-composition.json',dict(actual_label_occurrences=dict(labels),
        raw_token_density_includes_formatting_and_is_not_accuracy=True,actual_source_events=1110,original_native_candidates=29,model_updates=0))
    evidence=[Path(__file__),previous_path,pilot_path,reader_path,OUT/'actual-label-composition.json',*[ROOT/'reports'/n for n in tests]]
    atomic_json(OUT/'review.json',dict(at=at,requirements=requirements,previous_full_scope_review=str(previous_path.relative_to(ROOT)),
        actual_runtime=runtime,actual_pilot=pilot,actual_independent_readback_and_repetitions=reader,
        source_files_sha256={str(p):file_hash(p) for p in evidence},related_primary_research=[
            dict(title='Code Execution with Pre-trained Language Models',url='https://arxiv.org/abs/2305.05383',
                relevance='Motivates testing actual code-state labels; its model/results are not our own checkpoint or evidence of our gain.'),
            dict(title='NExT: Teaching Large Language Models to Reason about Code Execution',url='https://arxiv.org/abs/2404.14662',
                relevance='Motivates runtime-state supervision; we use no external-model rationales or teacher weights.'),
            dict(title='Curriculum Learning for Small Code Language Models',url='https://aclanthology.org/2024.acl-srw.44/',
                relevance='Reports stronger execution than completion gains in its own small-model setting; does not establish code generation gains here.')],
        previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True,no_original_scope_removed=True,candidate_promoted=False,goal_achieved=False))
    docs=[ROOT/'README.md',ROOT/'RESEARCH.md',master_path]
    if any(str(p) in protected for p in docs):raise ValueError('Review saved; source-bound root documents must remain unchanged')
    (ROOT/'reports/completion-audit-before-public-state-trace-research-v1.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={k:v['evidence'] for k,v in requirements.items()},latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),
        latest_actual_main_runtime=runtime,latest_public_state_pilot=str(pilot_path.relative_to(ROOT)),
        latest_public_state_pilot_readback=str(reader_path.relative_to(ROOT)),
        latest_full_public_state_preparation='reports/public-state-traces-full-v1/plan.json',
        current_goal_turn_classification='progress',all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph=(f'Public program-state research, **{at}**: the unchanged four-L40 main is at **{progress["actual_updates"]}/35,284updates**. '
        'A predeclared64-source CPU pilot yields **29complete source-state candidates** in the unchanged4096-token context. '
        'All29native inputs/targets/EOS masks rebuild independently, and all29state traces/stdout reproduce in a second fresh interpreter. '
        'The reader also checks all64original source dispositions and6659retained case-output verdicts. Seven focused tests pass. '
        'The candidates contain **30,076response/EOS target tokens versus123for stdout-only** on the same cases, including format overhead, '
        'and **702actual local-binding changes**. This is denser source supervision, **not a measured model gain**.\n\n'
        'A new CPU job now prepares state candidates across **all3,959original source references**, retaining every source disposition and requiring repeatable whole traces. '
        'The data remains **unadmitted** pending full completion, independent readback and held-overlap checks. '
        'Current training and all eight waiting successors remain unchanged. Latest complete matched code correctness remains **0/48** at both fixed main checkpoints; '
        '**all ten final requirements remain incomplete**. [Pilot](reports/public-state-trace-pilot-v1/complete.json), '
        '[readback and repeats](reports/public-state-trace-pilot-v1-readback/complete.json), '
        '[full-source job](reports/public-state-traces-full-v1/plan.json), [full review](reports/public-state-trace-research-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for path in docs[:2]:
        first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),main_updates=progress['actual_updates'],pilot_candidates=29,
        actual_state_targets=30076,actual_binding_changes=702,all29_whole_state_traces_reproduced=True,
        new_state_training_admitted=False,actual_model_gain_not_established=True,goal_achieved=False)))


if __name__=='__main__':main()
