"""Record the actual interrupted/recovered lineage and unchanged final scope."""
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/instance-recovery-and-matched-results-v1-review'

def read(path):return json.loads(Path(path).read_text())

def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh GPU-hidden full-scope observation required')
    master_path=ROOT/'reports/completion-audit.json';master=read(master_path);previous=ROOT/master['latest_full_scope_review']
    requirements=deepcopy(read(previous)['requirements'])
    runtime=ROOT/'reports/public-pretraining-recovery-v1';lineage=ROOT/'reports/instance-lineage-recovery-v2'
    plan=read(runtime/'plan.json');handoff=read(lineage/'complete.json')
    if terminal(plan['owner']) or len(handoff['actual_ten_new_controllers'])!=10:
        raise ValueError('Actual complete ten-controller recovery and live public GPU owner required')
    if any(terminal(r['actual_owner']) for r in handoff['actual_ten_new_controllers']):
        raise ValueError('All ten restored unstarted controllers must remain live')
    raw=read(ROOT/'reports/public-billion-causal-candidate-v2-full-readback-v2/complete.json')
    matched=read(ROOT/'reports/completed-code-epoch-matched-metrics-v1/complete.json')
    interruption=read(ROOT/'reports/instance-interruption-20261018-v1/complete.json')
    original=read(ROOT/'reports/public-pretraining-v1/plan.json')
    if (raw['actual_train_targets_admitted']!=1000085544 or not raw['data_admitted_to_training']
            or original['planned_steps']!=4447 or not matched['complete']):raise ValueError('Full public source and completed matched evidence required')
    metrics=[]
    journal=runtime/'training-metrics.jsonl'
    if journal.exists():
        for line in journal.read_bytes().splitlines(keepends=True):
            if not line.endswith(b'\n'):break
            metrics.append(json.loads(line))
    children=read(runtime/'children.json');active=[r for r in children if not terminal(r)]
    capacity=read(runtime/'capacity.json') if (runtime/'capacity.json').exists() else None
    cpu=ROOT/'reports/own-function-trajectory-preferences-full-CPU-v4'
    research={name:read(cpu/name) for name in ['plan.json','preflight-progress.json','progress.json','complete.json','failure.json'] if (cpu/name).exists()}
    at=datetime.now(timezone.utc).isoformat();OUT.mkdir()
    observation=dict(at=at,actual_pretraining_owner=plan['owner'],active_registered_children=active,
        fresh_capacity_complete=bool(capacity and capacity.get('passed')),recovered_completed_neural_updates=len(metrics),
        planned_neural_updates=4447,original_source_schedule_sha256=original['source_schedule_sha256'],
        latest_completed_update=metrics[-1] if metrics else None,interruption=interruption,
        CPU_research=research,persistent_free_bytes=shutil.disk_usage(ROOT).free,
        actual_GPU_observation=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
        extra48GiB_resource_question_unanswered_not_a_current_pretraining_blocker=True)
    atomic_json(OUT/'runtime.json',observation)
    explanations={
        'own_architecture':'Own scratch-origin dense text/image model, tokenizer and word heads remain. The completed model has984399690unique parameters; bounded actor variants999706442. The own-failure experiment compares supervised CE with full actual sampled-trajectory machine DPO; no external weights or teacher and no global architectural novelty or useful gain is claimed.',
        'public_data':'The independent full original-source/native/mask/rights readback admits1000085544training targets and10148261validation targets from public code in8languages, math and web sources.826whole source-train functions are separately admitted. All17families and1105735574total selected response/EOS targets remain in the4447update schedule. Broad all-domain source coverage is not thereby established.',
        'under_1b_parameters':'The own complete dense student has984399690unique values. Policy variants include every adapter, norm and separate critic for999706442unique values. Final useful text/image/word and alignment capability under the limit is incomplete.',
        'text_and_images':'The full protected retention schedule includes ordinary photo captions, visual reasoning, image reading and real/synthetic word crops. Completed code-epoch local OCR remains0/20exact; objects3/15andphoto retrieval1/32in both directions. Actual image grounding and high-quality all-round multimodal competence remain unproven.',
        'human_feedback':'All85786public human-ranked pairs remain scheduled for chosen-CE/human-DPO matched arms, followed by all3721coding sources per CE/execution-RL arm and1620machine preference pairs per matched arm. All ten interrupted unstarted controllers now have actual fresh queues and process births. Own-failure trajectory preferences are machine feedback, not new human judgments.',
        'serious_full_model_training':f'The original complete code/output/state/replay run finished7558updates. The separately admitted public pretraining stage was interrupted after27observed RAM-only updates before its first512interval resume; those updates are unrecoverable. Recovery uses the intact initial model and unchanged original schedule/horizon/options. Current recovered complete journal prefix is{len(metrics)}/4447. First-update and64update immutable CPU snapshots now write atomically in the background; future restore validates full model/Adam/source/metrics and all4private RNGs.',
        'high_end_llm_capability':'NOT ACHIEVED. Matched final code-epoch correctness remains0/48; syntax29/48to20/48; ordinaryPython0/24; generated knowledge2/56; math0/16; OCR0/20. Held packed-code NLL improves0.868549to0.819275(5.67%)on the same128windows/524160targets, using producer observations. This has not produced a meaningful generated-code or all-round gain and gives no basis for a GPT-level comparison.',
        'broad_independent_benchmarks':'The nine predeclared endpoints and complete16961QA,3088AI2Dunder3imageconditions,600MBPPwith1816cases,4642heldhumanpairs and3429heldwordsunder3imageconditions remain intact. Recovery scripts preserve every original scientific function byte-for-byte and change only fresh queue literals and unstarted strata-reader and held-cohort copy destinations. These complete model assessments remain pending. Local matched evidence is small development evidence; only decoded code bytes were freshly reexecuted, not original unavailable action/EOS journals.',
        'gpu_runtime':f'Every22inventoried original birth is terminal after instance process/tmpfs loss; no physical host reboot is inferred and no old process was signalled. A distinct actual runtime owns the kernel GPU lease while all neural mathematics retain the immutable original plan SHA. Fresh all4L40capacity/recovery/cold-native proof complete={bool(capacity and capacity.get("passed"))}; actual active workers are recorded in runtime.json. Async durability tests5pass and complete10controller route/function preservation1pass. No GPU utilization claim is inferred from queue existence.',
        'any_code_correctness_and_efficiency':'NOT ACHIEVED. Final local functional code remains0/48. A completed own-function diagnostic records0/14public-example passes,1/14syntax and12budget exhaustions on source-train functions, with original answers visible. Independent readback admits10optimization/4unoptimized source-train controls and actual censored negative actions without invented EOS. A full-size CPU CE versus trajectory-DPO recovery regenerates the same-actor baseline and preserves exact original parameter/source identity. Its earlier alternate-graph discrepancy is retained and unresolved; this experiment is separate from production and independent held evaluation; no improvement, arbitrary-code correctness, efficiency or generalization is claimed before results.'}
    for name,evidence in explanations.items():requirements[name]=dict(status='incomplete_for_requested_final_model',evidence=evidence)
    if len(requirements)!=10 or any(v['status']!='incomplete_for_requested_final_model' for v in requirements.values()):raise ValueError('All ten final requirements must remain incomplete')
    # Root summaries are mutable only when no experiment freezes them.
    protected=set()
    for name in subprocess.check_output(['rg','--files','reports','-g','plan.json','-g','*-queue.json'],cwd=ROOT,text=True).splitlines():
        value=read(ROOT/name)
        if isinstance(value,dict):protected.update(value.get('source_files_sha256',{}))
    for path in [master_path,ROOT/'README.md',ROOT/'RESEARCH.md']:
        if str(path) in protected:raise ValueError('A source-bound summary must not change')
    paths=[Path(__file__),previous,OUT/'runtime.json',runtime/'plan.json',lineage/'complete.json',
        ROOT/'reports/completed-code-epoch-matched-metrics-v1/complete.json',
        ROOT/'reports/public-billion-causal-candidate-v2-full-readback-v2/complete.json',
        ROOT/'reports/public-pretraining-sources-CPU-v4/complete.json',ROOT/'reports/async-durable-pretraining-recovery-v1-final-tests.log',
        ROOT/'reports/instance-lineage-recovery-v2-tests.log']
    atomic_json(OUT/'review.json',dict(at=at,requirements=requirements,previous_full_scope_review=str(previous.relative_to(ROOT)),
        source_files_sha256={str(p):file_hash(p) for p in paths},previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True,original_full_scope_preserved=True,candidate_promoted=False,goal_achieved=False))
    (OUT/'previous-completion-audit.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={n:r['evidence'] for n,r in requirements.items()},latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),
        latest_actual_main_runtime=observation,latest_complete_code_epoch_matched_metrics='reports/completed-code-epoch-matched-metrics-v1/complete.json',
        all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False,current_goal_turn_classification='progress')
    atomic_json(master_path,master)
    paragraph=(f'Instance recovery and completed matched results, **{at}**: the independent reader now admits **1,000,085,544 public training targets**. '
        'The new run lost **27 RAM-only updates** when the instance processes/tmpfs disappeared; its initial model and all completed prior work survived. '
        f'The recovered unchanged full curriculum is at **{len(metrics)}/4,447 updates**, with a separate actual runtime and first-update / every-64-update background durable snapshots. '
        'All ten full human-feedback, execution-RL, export and evaluation controllers have fresh actual queues; every original scientific function and benchmark population remains.\n\n'
        '**Final code-epoch correctness remains 0/48**, syntax29/48to20/48, ordinaryPython0/24, generated knowledge2/56, math0/16, OCR0/20, objects3/15andphoto retrieval1/32. '
        'Same-window packed-code NLL falls5.67%, which has not established generated correctness. A separate full-size own-failure CE/trajectory-DPO experiment uses10optimization functions and4unoptimized source-train controls; it is not a held benchmark or human RLHF. '
        '**All ten final requirements remain incomplete.** [Matched metrics](reports/completed-code-epoch-matched-metrics-v1/metrics.md), '
        '[recovery review](reports/instance-recovery-and-matched-results-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for path in [ROOT/'README.md',ROOT/'RESEARCH.md']:
        first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),recovered_updates=len(metrics),all_ten_final_requirements_incomplete=True)),flush=True)

if __name__=='__main__':main()
