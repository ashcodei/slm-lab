"""Full-scope review of qualified public code and contract-sensitive preferences."""
from collections import Counter
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.function_repair_GPU_contract_v1 import QUEUE,ADMISSION,handoff_admitted

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/public-contract-qualification-v2-review'


def read(path):return json.loads(Path(path).read_text())


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CUDA-hidden full-scope review required')
    source=ROOT/'reports/public-contract-census-CPU-v2';admitted=ROOT/'reports/public-contract-census-v3-readback-admission/complete.json'
    intent=ROOT/'reports/same-name-contract-preferences-v1-readback-admission/complete.json'
    census=read(source/'complete.json');admission=read(admitted);preferences=read(intent);plan=read(source/'plan.json')
    if (not census.get('complete') or not admission.get('complete') or not preferences.get('complete')
            or admission['actual_admitted']['direct']['examples']!=57 or admission['actual_admitted']['contrast']['examples']!=5
            or preferences['actual_admitted_machine_source_preferences']!=6):raise ValueError('All exact actual complete admissions required')
    files={**admission['source_files_sha256'],**preferences['source_files_sha256']}
    previous=ROOT/'reports/function-semantic-source-quality-v1-review/review.json'
    for path in [Path(__file__),source/'complete.json',admitted,intent,previous]:files[str(path)]=file_hash(path)
    guard=FileBindings(files);queue=read(ROOT/QUEUE)
    if not handoff_admitted(ROOT,queue['owner']):raise ValueError('All original scientific downstream stages must remain actually live')
    main=read(ROOT/'reports/public-pretraining-recovery-v1/plan.json');cpu_plan=read(ROOT/'reports/full-function-repair-comparison-CPU-v1/plan.json')
    if terminal(main['owner']) or terminal(cpu_plan['owner']):raise ValueError('Actual original neural runs must remain live')
    raw=(ROOT/'reports/public-pretraining-recovery-v1/training-metrics.jsonl').read_bytes();prefix=raw[:raw.rfind(b'\n')+1];rows=list(map(json.loads,prefix.splitlines()))
    if [v['step'] for v in rows]!=list(range(1,len(rows)+1)):raise ValueError('Complete contiguous actual public journal prefix required')
    current=dict(owner=main['owner'],complete_actual_updates=len(rows),planned_updates=4447,
        response_EOS_targets=sum(int(v['response_EOS_targets']) for v in rows),complete_prefix_sha256=hashlib.sha256(prefix).hexdigest())
    cpu=read(ROOT/'reports/full-function-repair-comparison-CPU-v1/progress.json');at=datetime.now(timezone.utc).isoformat();OUT.mkdir()
    observations=[json.loads(v) for v in (source/'all-selected-original-public-and-contract-runs.jsonl').read_text().splitlines()]
    fixed={v['source_id']:v for v in read(source/'all-fixed-contract-cases.json')};bad=[]
    lines=['# Public code contract qualification','',
        'This qualifies additional training data. No new model generation, training, benchmark score or general correctness claim is made.','',
        f"All 841 original source records are inventoried: 826 train and 15 validation. The 62 exact reviewed implementations cover {len(plan['actual_contract_families'])} contracts. All {plan['actual_literal_public_expectations_checked']} in-domain literal published expectations agree with their assigned contracts before any wrong-code label is allowed.",
        '', '| Data ready for future training | Examples | Complete positive response/EOS targets |', '|---|---:|---:|',
        f"| Whole original functions passing finite contracts and all original public examples | 57 | {admission['actual_admitted']['direct']['response_EOS_targets']} |",
        f"| Public correction programs, with whole original faulty programs as machine negatives | 5 | {admission['actual_admitted']['contrast']['response_EOS_targets']} |",
        f"| Same-name programs with different documented task contracts | 6 | {preferences['actual_positive_response_EOS_targets']} |", '',
        '| Original public reference | Counterexample arguments | Expected return | Actual return |','|---|---|---|---|']
    for row in observations:
        if row['disposition']!='verified_contract_counterexample':continue
        observed=row['actual_semantic_runs'][0];position=next(i for i,v in enumerate(observed['case_passes']) if not v)
        case=fixed[row['source_id']]['cases'][position];actual=observed['raw_results'][position]
        item=dict(source=row['source_function_identity'],args=case['args'],expected=case['expected'],actual_raw=actual)
        bad.append(item);lines.append('| '+row['source_function_identity']+' | `'+repr(case['args'])+'` | `'+repr(case['expected'])+'` | `'+actual['stdout'].strip()+'` |')
    if len(bad)!=5:raise ValueError('Every and only verified source-contract counterexample must be reported')
    lines += ['',
        'The initial v1 registry incorrectly conflated recursive digital root with a single-pass digit sum. It was not admitted. Original public checks rejected both attempted incompatible donor programs. The revised registry separates those tasks and checks literal public expectations first. A later direct-row metadata readback also failed closed because original pack ordinals were inherited; the v3 reader verifies the new ordinal, full native SHA and target count explicitly. All failed attempts remain preserved.',
        '', 'The five original wrong-code targets now have complete alternative public implementations from the same pinned MIT repository, with only a static forwarding alias when entrypoint names differ. Both original and donor public examples and all target finite cases are checked. Every positive/negative response and genuine EOS is preserved. The 57 direct targets retain their original input tokens, prompt masks and whole source targets exactly.',
        '', 'The six task-specific preference pairs cover digit sum versus digital root, index versus boolean search, and list versus set return contracts. Both programs pass their own contracts; only the wrong task context is rejected. All wrong-context programs execute the correct entrypoint. One set/list contract violation is not caught by the original public examples, so that limitation stays explicit.',
        '', f"The census admission performs {admission['actual_fresh_confined_program_runs']} fresh confined program runs, and the separate same-name preference admission performs {preferences['actual_fresh_confined_program_runs']}. They reuse the bound numerical contracts and execution engine in fresh processes; they are not separate oracle implementations.",
        '', 'Unsupported sources are unqualified, not called incorrect. None of the original 826 train functions, 15 validation functions, 17 live pretraining families, human-preference sources, execution-RL sources or full benchmark populations is removed or relabelled. The new packs are additional data and have not been optimized.',
        '', 'Last completed broader coding correctness remains 0/48. The earlier source-train CE and DPO diagnostic remains 7/10 under its extra checks with 0/4 unoptimized controls; this corpus qualification is not a new neural quality result. All ten final requirements remain incomplete.','']
    (OUT/'metrics.md').write_text('\n'.join(lines))
    proof=dict(complete=True,at=at,source_files_sha256=files,all_original_sources_inventoried=841,
        exact_reviewed_implementations=62,explicit_contract_families=len(plan['actual_contract_families']),literal_published_expectations_checked=plan['actual_literal_public_expectations_checked'],
        actual_admitted=admission['actual_admitted'],actual_admitted_task_contract_preferences=6,
        actual_admitted_preference_positive_targets=preferences['actual_positive_response_EOS_targets'],all_five_source_counterexamples=bad,
        census_readback_actual_confined_program_runs=admission['actual_fresh_confined_program_runs'],
        task_preference_readback_actual_confined_program_runs=preferences['actual_fresh_confined_program_runs'],
        original_main=current,original_CPU_comparison=cpu,GPU_repair_study_owner=queue['owner'],
        actual_GPU_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True),
        previous_goal_turn_classification='progress: actual fourteen-source fresh-input diagnosis and three admitted public corrections',
        current_goal_turn_classification='progress: full source inventory, revised literal-checked contract census and independently admitted code/contrast/intent data',
        original_full_training_and_benchmark_scope_preserved=True,actual_new_neural_forwards=0,actual_new_neural_updates=0,
        no_new_generated_quality_score=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(OUT/'complete.json',proof);review=copy.deepcopy(read(previous));review['at']=at
    message=(f" The complete841record source inventory now preserves826train and15held roles.62implementations across36explicit finite contracts "
        f"pass212literal-public-expectation preflight checks. Fresh source qualification admits57unchanged whole-function positives (20891targets),"
        "5public alternative-program corrections (2840targets) and6same-name/different-contract machine preference pairs (2416positive targets). "
        "Readbacks perform134and36fresh confined program runs. The first conflated digital-root registry and failed inherited-pack-metadata readback "
        "were preserved without admission; corrected versions pass full source/native/runtime readback. Both programs in the same-name preferences "
        "are valid for their own qualified task; no program is globally labelled wrong from a different intended context. New optimization remains0; "
        "no broad corpus defect rate, independent model benchmark or generated-code improvement is inferred.")
    for key in ['public_data','human_feedback','high_end_llm_capability','any_code_correctness_and_efficiency']:
        review['requirements'][key]['evidence']+=message
    review['requirements']['serious_full_model_training']['evidence']+=(f" Latest actual complete public prefix is{len(rows)}/4447updates; "
        f"unchanged CPU comparison is{cpu['actual_complete_updates']}/306in its current{cpu['profile']}arm. The GPU comparison still waits for its full public predecessor. "
        "Source qualification uses no neural model or optimizer and does not substitute for those full experiments.")
    if any(v['status']!='incomplete_for_requested_final_model' for v in review['requirements'].values()):raise ValueError('No final requirement is complete from data qualification')
    review.update(previous_full_scope_review=str(previous),current_goal_turn_classification=proof['current_goal_turn_classification'],
        source_files_sha256={**files,str(OUT/'complete.json'):file_hash(OUT/'complete.json')},current_actual_runtime=dict(main=current,CPU=cpu,GPU_queue_owner=queue['owner']),goal_achieved=False)
    atomic_json(OUT/'review.json',review);guard.check(full=True)
    live=read(ROOT/ADMISSION)['new_controllers'];maps=[read(ROOT/v['queue'])['source_files_sha256'] for v in live]
    maps += [queue['source_files_sha256'],main['source_files_sha256'],cpu_plan['source_files_sha256']]
    for name in ['reports/completion-audit.json','README.md','RESEARCH.md']:
        if any(str(ROOT/name) in mapping for mapping in maps):raise ValueError('Do not modify any live source-bound review document')
    audit=read(ROOT/'reports/completion-audit.json');audit.update(at=at,latest_full_scope_review=str(OUT/'review.json'),
        requirements=review['requirements'],latest_actual_main_runtime=current,latest_public_contract_qualification=str(OUT/'complete.json'),goal_achieved=False)
    atomic_json(ROOT/'reports/completion-audit.json',audit)
    paragraph=(f"Expanded public contract qualification, **{at}**: all841original records retain826train/15validation roles. "
        "62reviewed implementations cover36contracts with212literal-public preflight checks. Independent readback admits**57whole-function positives, "
        "5public-source corrections and6task-specific machine preferences**. Both same-name implementations remain valid for their own contracts. "
        "The revised corpus separates digital root from digit sum; failed earlier qualification/readback attempts are preserved and not adopted. "
        "**No new model training or accuracy gain is established.** The original main is at"+str(len(rows))+"/4,447updates; CPU comparison at"+
        str(cpu['actual_complete_updates'])+"/306in its current arm. All original human/RL/full-benchmark scopes remain. **All ten final requirements "
        "remain incomplete.** [Qualified data and counterexamples](reports/public-contract-qualification-v2-review/metrics.md), "
        "[full review](reports/public-contract-qualification-v2-review/review.json). Earlier entries are historical snapshots.\n\n")
    for name in ['README.md','RESEARCH.md']:
        path=ROOT/name;text=path.read_text()
        if text.startswith('#'):
            first,rest=text.split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
        else:path.write_text(paragraph+text)
    print(json.dumps(dict(review=str(OUT/'review.json'),qualified_direct=57,corrected_public_examples=5,task_preferences=6,neural_updates=0,goal_achieved=False)),flush=True)


if __name__=='__main__':main()
