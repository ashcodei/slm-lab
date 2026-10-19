"""Publish the current full-scope audit, preserving superseded evidence."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Audit is CPU-only')
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    import assess_joint_java_comparison_v1 as resources
    def read(path):return json.loads((ROOT/path).read_text())
    plan=read('reports/fast-joint-v3/plan.json');progress=read('reports/fast-joint-v3/progress.json')
    proof=read('reports/fast-joint-v3-progress-readback-v1.json')
    if not resources.same(plan['owner']) or progress['actual_step']<proof['production_update_records_read']:
        raise ValueError('Exact current learning owner and actual record readback required')
    destination=ROOT/'reports/fast-joint-v3-and-execution-hybrid-v1-review.json'
    if destination.exists():raise ValueError('Preserve prior full-scope review')
    cpu_path=ROOT/'reports/execution-group-RL-full-CPU-v1/complete.json'
    if not cpu_path.exists():raise ValueError('Finish the started full-size CPU hybrid experiment before final review')
    cpu=json.loads(cpu_path.read_text())
    if not (cpu['exact_actual_next_actions_execution_rewards_records_weights_Adam_CPU_RNG_recovered']
            and cpu['actual_trained_text_image_word_cold_reload_exact']):raise ValueError('Full-size actual hybrid evidence required')
    requirements={
      'own_architecture':'The scratch-origin 35-layer dense student folds our own previously trained code factors and has 11 initially neutral extra residual blocks. No external pretrained model, tokenizer, vision encoder or model teacher is loaded. New full-weight optimization is live. Final useful architecture/capability remains unverified.',
      'public_data':'The selected joint epoch has 458,450 source-family records, 254,042,336 complete context positions and 156,677,748 next/response/EOS targets across 15 families. It covers all selected 112,609 CodeNet programs in 54 languages, 12,287 dialogues, 3,548 APPS programs, 173 verified complete functions, 91,585 word crops, education/math and real/generic image sources. Original train roles, notices and exact hashes are preserved. Source-family records are not certified semantically distinct; some raw data was previously seen. This is not all public data or all open-source code. The separately admitted 52,970,300 new OS/network/security code tokens remain a required future exposure, alongside continued source diversity.',
      'under_1b_parameters':'The actual live joint model counts 984,399,690 total unique parameters including all auxiliary heads and critic. Selected full-weight language/vision/word parameters total 983,276,610. The separate disposable RL actor prototype counts 999,706,442 including its 15,305,216 actor values and separate 1,536-value critic. Its same counted frozen base supplies reference probabilities; no second teacher or reward encoder is loaded for this execution experiment. A final trained, quality-qualified unified model remains incomplete.',
      'text_and_images':'Full dense language, vision and shared word parameters are now actually updated in one four-GPU run. The selected sources include generic photos/captions, visual questions, whole-photo text questions and cropped words; cropped reading is not whole-photo detection. Development pulses are isolated and bounded; full native OCR/object/retrieval checks remain required. Last completed comparison still has 0/170 exact OCR and poor object/retrieval controls. No new held reading or generic-image gain is established.',
      'human_feedback':'Genuine public human-ranking data and prior reward/DPO/PPO mechanics are retained. The four unstarted old word/foundation/preference/PPO controllers were intentionally retired for the authorized faster joint run, with all source versions/evidence preserved. Preference and genuine-human PPO work is deferred, not removed. The new code hybrid uses only machine execution of published train examples and complete original public targets; it adds zero genuine human judgments and is not RLHF. Useful own-output human alignment and the complete human-feedback stage remain unfinished.',
      'serious_full_model_training':f"Actual production joint learning is live, with {progress['actual_step']} updates recorded in the current progress file; a separate reader reconciles the first {proof['production_update_records_read']} complete original source batches, target counts and loss arithmetic. The complete source epoch requires 22,065 updates. Actual full configured four-L40 word/book/image capacity and exact next weights/Adam/Python/NumPy/CPU/private-CUDA RNG recovery passed before production. This replaces recent limited-extension studies; the original backbone had already received billions of logged target exposures. The running epoch, later broad-code exposures and final policy training remain incomplete.",
      'high_end_llm_capability':'NOT ACHIEVED. The prior code-initialized CE/KL studies improved held teacher likelihood but regressed generated knowledge from 8/559 to 2/559; all local functional/efficiency scores stayed zero. New production update/loss evidence is not a held capability improvement. No sub1B high-end parity or inevitable path to it is claimed.',
      'broad_independent_benchmarks':'Full unchanged native initial/final knowledge, both 24-task Python/JS prompt frames, 12 C++, 175 public code problems, 1,319 math problems, 170 OCR, 164 object controls and 128 retrieval pairs remain required. Development code/math use separately declared 128/256 action budgets and subsets, while full endpoint budgets remain 512; these are different protocols. Old remaining math/trained-public evaluations were deliberately ended as incomplete after both training arms completed. Current source/schedule arithmetic is separately reconstructed but not a complete independent neural/Adam replay. Benchmark reuse and semantic contamination limitations remain explicit.',
      'gpu_runtime':f"All four actual local L40s are owned by exact controller birth identity {plan['owner']['pid']}/{plan['owner']['ticks']}. The stalled v2 evaluation was retired with zero production updates and all exact workers terminal; original waiting work was restored before new ownership. NCCL handles learning, CPU Gloo handles outer control barriers, and optional development evaluation uses isolated exact children with a 180-second wall budget. The source/schedule reader measured mean GPU busy percentages {proof['training_GPU_busy_percent_mean']} across five one-second samples during actual learning. Busy percentage is not useful-FLOPs utilization or throughput efficiency. Endpoint evaluation and successful final restoration remain pending.",
      'any_code_correctness_and_efficiency':'NOT ACHIEVED. The new machine-code reward scores only published original train examples that actually call the generated entry body, with expected outputs outside the worker; setup statements earn no credit. Clipped group-relative execution RL uses actual sampled behavior log-probabilities and full-vocabulary same-base reference KL. All-zero execution groups explicitly add complete verified public-target CE. Focused mathematical/worker tests and a full-size disposable two-logical-update CPU experiment with exact next-sample/reward/Adam/RNG recovery and trained native cold reload pass. The response horizon is only four actions per sample, so this does not prove full-length GPU capacity, useful policy learning, algorithmic generalization, efficiency or genuine RLHF. Prior local correctness/efficiency remains 0/24, C++ remains 0/12.'
    }
    evidence=['reports/fast-joint-v3/plan.json','reports/fast-joint-v3/capacity.json',
      'reports/fast-joint-v2/capacity.json','reports/fast-joint-v2/restored.json',
      'reports/fast-joint-v3-handover.json','reports/fast-joint-v1-handover/retired-unstarted-controllers.json',
      'reports/fast-joint-v1-handover/code-evaluation-retirement.json','reports/fast-joint-full-CPU-v2/complete.json',
      'reports/fast-joint-v3-progress-readback-v1.json','reports/fast-joint-driver-v3-tests.log',
      'reports/parallel-file-bindings-v1-tests.log','reports/inference-rotary-cache-v1-tests.log',
      'reports/execution-group-RL-v1-tests-v2.log','reports/public-example-code-reward-v1-tests.log',
      'reports/execution-group-RL-small-graph-v2.json','reports/execution-group-RL-full-CPU-v1/complete.json',
      'reports/code-initialized-native-interim-v2.json','reports/systems-variety-training-admission-readiness-recovery-v2.json']
    for target in ['README.md','RESEARCH.md','reports/completion-audit.json']:
        if str(ROOT/target) in plan['source_files_sha256']:raise ValueError('Never mutate a live bound document')
    review=dict(observed_at=datetime.now(timezone.utc).isoformat(),objective_preserved=True,
      previous_goal_turn_classification='progress',previous_goal_turn_evidence='173 complete original publisher-train functions / 715 examples independently reexecuted; native regressions reconciled.',
      current_goal_turn_classification='progress',current_goal_turn_progress=['Full-size language/vision/word learning and cold reload passed on CPU.',
        'Actual complete-profile learning and exact recovery passed on all four L40s after replacing unsupported deterministic bicubic backward.',
        'Four unstarted controllers and completed regressing-arm evaluations were intentionally retired with exact birth/worker checks and preserved evidence.',
        'Initial distributed evaluation stalled before production; a new isolated evaluation/Gloo wrapper inherited only the byte-identical proved learning graph.',
        'Actual four-GPU full-weight production updates now run and the first observed complete source batches/targets/loss arithmetic are reconstructed separately.',
        'Machine execution group RL plus a verified-public-target fallback has focused tests and an actual full-size disposable recovery/cold-load proof.'],
      actual_progress_snapshot=progress,source_schedule_readback=proof,
      requirements={k:dict(status='incomplete_for_requested_final_model',current_evidence_and_limits=v,prior_evidence_retained=True) for k,v in requirements.items()},
      source_files_sha256={str(ROOT/p):file_hash(ROOT/p) for p in evidence},new_genuine_human_judgments=0,
      new_held_capability_gain_proven=False,candidate_promoted=False,all_ten_final_requirements_incomplete=True,
      next_required_work=['Finish and durably retain the live joint epoch with timed development checks and complete native endpoints.',
        'Independently reconcile production model/Adam/source/RNG evidence and all generated functional/efficiency outcomes.',
        'Train the admitted additional OS/network/security code and continue broad source diversity without relabeling held data.',
        'Run full configured four-GPU action-horizon capacity/recovery before any controlled execution-RL experiment; compare against matched public-target CE.',
        'Complete genuine human-feedback policy training with measured reward fidelity and matched broad/code/image retention controls.',
        'Only promote a counted under1B unified student after actual useful broad/code/vision/human-alignment gains; high-end target remains open.'],
      no_remaining_scope_removed=True,goal_achieved=False)
    atomic_json(destination,review)
    audit_path=ROOT/'reports/completion-audit.json';snapshot=ROOT/'reports/completion-audit-before-fast-joint-v3.json'
    if snapshot.exists():raise ValueError('Preserve original audit snapshot')
    shutil.copyfile(audit_path,snapshot);audit=json.loads(audit_path.read_text())
    audit['requirements']=requirements;audit['updated_at']=review['observed_at'];audit['goal_achieved']=False
    audit['latest_fast_joint_and_execution_hybrid_full_scope_review']=str(destination.relative_to(ROOT))
    audit['live_joint_training']=dict(plan='reports/fast-joint-v3/plan.json',progress='reports/fast-joint-v3/progress.json',
        production_updates_observed=progress['actual_step'],observed_at=review['observed_at'],all_four_GPU_training_live=True,
        held_gain_measured=False,goal_achieved=False)
    for key in ['dense_foundation_live_queue','dense_human_preference_live_queue','dense_PPO_live_queue']:
        audit['historical_'+key]=audit.get(key);audit[key]=None
    audit['previous_unstarted_queues_retired']='reports/fast-joint-v1-handover/retired-unstarted-controllers.json'
    audit['previous_bound_code_evaluation_retired_incomplete']='reports/fast-joint-v1-handover/code-evaluation-retirement.json'
    atomic_json(audit_path,audit)
    message=f'''Current joint learning, **2026-10-17**: the own **984,399,690-parameter** student is now training on **all four local L40s**, with **{progress['actual_step']} actual production updates** observed. It updates **931,244,544 language**, **51,624,448 vision** and **407,618 word-head** values. The selected full epoch contains **458,450 source-family records / 156,677,748 target tokens** across code, education, dialogue, math, generic photos and text-bearing images; **22,065 updates** are planned. These are selected source-family exposures, not certified distinct new raw examples or exhaustive public-code coverage.\n\nActual configured four-GPU full-batch training and exact weight/Adam/CPU/private-CUDA RNG recovery passed. A source/schedule reader reconciles the first **{proof['production_update_records_read']} production updates**, covering **1,632 complete source-family records / 586,124 targets**, and measured mean GPU busy percentages **{proof['training_GPU_busy_percent_mean']}** over five one-second samples. This is activity/source evidence, not a new held quality score or useful-FLOPs utilization. The earlier zero-update wrapper stalled during initial evaluation; its exact workers were stopped and restored before the new controller acquired the GPUs. Development checks now run in isolated workers with a 180-second budget, after learning begins. Full native endpoint evaluations retain their original 512-token code/math protocols.\n\nThe custom execution-RL/public-target fallback also passes an actual **999,706,442-parameter CPU mechanics proof**: two disposable logical updates, exact next sampled actions/rewards/weights/Adam/RNG recovery, and trained native text/image/word cold reload. It uses only **four actions per sample**; this is not full-horizon GPU capacity, production RL, genuine human feedback or a measured coding gain. When every sample fails its published train examples, the objective explicitly adds the complete verified public solution. All ten final-model requirements remain incomplete; the latest local functional/efficiency score is still **0/24**. The full human-feedback stage and the separately admitted **52,970,300 new OS/network/security code tokens** remain required.\n\nThe old unstarted word/foundation/preference/PPO controllers were intentionally retired for this faster joint run; the dated queue statements below are historical. Sources, failed/partial reports and completed older training remain preserved. See [live plan](reports/fast-joint-v3/plan.json), [live progress](reports/fast-joint-v3/progress.json), [source/schedule readback](reports/fast-joint-v3-progress-readback-v1.json), [full-size hybrid mechanics](reports/execution-group-RL-full-CPU-v1/complete.json), and [current full-scope review](reports/fast-joint-v3-and-execution-hybrid-v1-review.json). **No candidate is promoted and no high-end parity is claimed.**\n\n'''
    for name in ['README.md','RESEARCH.md']:
        path=ROOT/name;text=path.read_text();first,separator,rest=text.partition('\n\n')
        path.write_text(first+'\n\n'+message+rest)
    print(json.dumps(dict(review=str(destination),production_updates_observed=progress['actual_step'],all_ten_final_requirements_incomplete=True)))


if __name__=='__main__':main()
