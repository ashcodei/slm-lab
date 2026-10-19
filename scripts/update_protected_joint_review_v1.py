"""Current full-scope audit after measured regression and protected GPU admission."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Audit must hide every GPU')
    import assess_joint_java_comparison_v1 as resources
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    def read(path):return json.loads((ROOT/path).read_text())
    run=ROOT/'reports/protected-joint-v1';plan=read('reports/protected-joint-v1/plan.json')
    progress=read('reports/protected-joint-v1/progress.json');capacity=read('reports/protected-joint-v1/capacity.json')
    if not resources.same(plan['owner']) or progress['actual_step']<1 or not capacity['passed']:
        raise ValueError('Actual live production updates and every passed new GPU capacity proof required')
    actors=read('reports/protected-joint-v1/children.json');live=[a for a in actors if resources.same(a)]
    if sum(a['subject'].startswith('train-rank-') for a in live)!=4:raise ValueError('Actual four live production workers required')
    old=read('reports/fast-joint-v3/plan.json')
    if resources.same(old['owner']) or any(resources.same(a) for a in read('reports/fast-joint-v3/children.json')):
        raise ValueError('Prior exact GPU workers must be terminal')
    if not (ROOT/'reports/fast-joint-v3/restored.json').exists():raise ValueError('Prior original-work restoration required')
    outcome=read('reports/fast-joint-development2048-readback-v1.json');state=read('reports/fast-joint-step2048-readback-v1/complete.json')
    cpu=read('reports/protected-joint-full-CPU-v2/complete.json');positive=read('reports/protected-positive-KL-full-CPU-v1.json')
    requirements={
      'own_architecture':'The own scratch-origin 35-layer dense text/image student retains its own tokenizer and all auxiliary heads. No external pretrained policy, tokenizer, image encoder or model teacher is loaded. The new protected objective is implemented, mechanically checked at full size and now running. Its frozen reference is a second training-only copy of our own initial model. The requested useful final architecture remains unverified.',
      'public_data':'The full selected curriculum remains 458,450 source-family records, 254,042,336 complete context positions and 156,677,748 next/response/EOS targets across 15 families, including 112,609 CodeNet programs in 54 languages, 12,287 dialogues, 3,548 APPS programs, 173 verified complete functions, 91,585 word crops, educational books, math, generic photos and text-bearing images. Previous partial exposures are retained history, not new raw sources. Source-family records are not certified distinct semantic examples. Full original selected-source exposure is still incomplete. The separately admitted additional 52,970,300 OS/network/security code tokens and further source variety remain required; this is not all public data or all open-source code.',
      'under_1b_parameters':'The actual running/exported policy contains 984,399,690 unique values, including every frozen and trainable head and critic. The additional reference also contains 984,399,690 training-only values: resident policy plus reference totals 1,968,799,380 per worker, and the reference is excluded from the exported policy. The separate experimental own actor/critic graph remains 999,706,442 values, sharing its frozen base for reference scores. Final trained, quality-qualified unified export remains incomplete.',
      'text_and_images':'The preceding mixed full-weight run regressed: matched GPU/BF16 OCR remains0/20exact with CER0.95417initial,2.01458at1024,0.97917at2048. Corrected CPU/FP32 objects change6/30to7/30proper matches, with blank/wrong7/30both, so this does not establish useful image dependence. Matched CPU caption retrieval is I2T1/32to1/32, T2I3/32to1/32; caption NLL worsens1.83179to2.08251. The new objective trains image/word batches through a fixed language decoder and anchors text updates to the own initial model. Generic useful vision, cropped/whole-photo reading quality and final multimodal alignment remain incomplete.',
      'human_feedback':'Public genuine human-ranking data, trained reward-head proxy and DPO/PPO implementations/proofs are retained. The previously unstarted human-preference/PPO controllers were deliberately retired and their source versions remain preserved. The new execution-group code learner and verified-solution fallback use machine execution of public training examples, adding zero human judgments. Production genuine-human preference/PPO training and useful own-output reward fidelity remain unfinished; execution feedback must not be called RLHF.',
      'serious_full_model_training':f"The preceding 2048-update durable model is real: a separate CPU reader matches every native/resumable weight, all445Adam states and every logged complete source batch, totaling14,286,535targets. All317language,112vision and16word tensors changed; leaf clocks are2048language,1144vision,532word. It was retired as an explicitly incomplete epoch after persistent coding regression. New protected production is live with {progress['actual_step']} updates observed, targeting all22065updates of the unchanged selected curriculum. It starts from the same own initial model, uses language LR5e-6, own full-context KL weight0.1, and blocks language updates on image/word batches. Useful full-source training, later additional code exposures and final policy training remain incomplete.",
      'high_end_llm_capability':'NOT ACHIEVED. Under the matched128-token development code protocol, initial syntax20/24and correctness1/24 become9/24and0/24at1024, then12/24and0/24at2048. Efficiency remains0/24. Knowledge generated answers0/56become0/56then1/56; math1/16becomes0/16at both checkpoints. One knowledge answer does not demonstrate broad improvement. This protocol differs from the previous full512-token code suite. The new protected method has zero measured held quality gains so far, and no high-end parity is claimed.',
      'broad_independent_benchmarks':'All72original code outputs across three development endpoints were independently reexecuted, and every raw knowledge/math/OCR response score and arithmetic was reconstructed. The state reader does not replay2048neural backwards, and this score reader does not replay neural generation. Small development subsets and reused benchmarks do not establish independent final generalization. The original leading-row object subset was invalid, and the old view lacked its native vision property. Failed/partial reports are preserved. New side assessments use balanced independent-photo object controls and the exact own vision module; the new driver adopts the complete view and a validated object subset. Full unchanged559knowledge,24code in both frames,12C++,175public code,1319math,170OCR,164object controls and128retrieval-pair comparisons plus held word controls remain required.',
      'gpu_runtime':f"The new exact owner is {plan['owner']['pid']}/{plan['owner']['ticks']}, with all four original local L40 training workers confirmed live. Fresh actual configured four-GPU book/word/image capacity and next-update/weights/all445Adam/Python/NumPy/CPU/private-CUDA RNG recovery passed. All four prove positive full-context KL, exact language bytes during image updates, and unchanged reference bytes. Peak allocated is about27.11GiB and reserved about30.06GiB per worker. This old capacity was not reused. New production updates are running, while final completion, independent result reconciliation and original-work restoration remain pending. GPU busy percentage is not useful-FLOPs utilization.",
      'any_code_correctness_and_efficiency':'NOT ACHIEVED. The custom controlled execution learner uses globally normalized variable-length own-response groups, complete reference KL, clipped relative rewards and original verified public-target CE when every sample fails. Two analytic/control tests pass. A real four-CPU/Gloo small-network test reproduces next samples/rewards/Adam/RNG and shows paired CE/hybrid weights and Adam exactly equal when all rewards are zero. A synthetic fixed-horizon gradient-only test reaches the actual kernel without optimizer/RNG/source mutation. Earlier full-size short own-code proof remains valid, but full-size configured response-horizon four-GPU capacity, controlled production RL and held functional/efficiency gains remain pending. The new pretraining objective has its own full-size CPU recovery/cold-load proof and genuine four-L40 capacity admission, not a coding-correctness result.'
    }
    evidence=['reports/protected-joint-v1/plan.json','reports/protected-joint-v1/capacity.json',
      'reports/protected-joint-full-CPU-v2/plan.json','reports/protected-joint-full-CPU-v2/complete.json',
      'reports/protected-positive-KL-full-CPU-v1.json','reports/protected-joint-v1-handover.json','reports/fast-joint-v3/restored.json',
      'reports/fast-joint-regression-monitor-v1/complete.json','reports/fast-joint-development2048-readback-v1.json',
      'reports/fast-joint-step2048-readback-v1/complete.json','reports/fast-joint-step1024-cache-CPU-v1.json',
      'reports/fast-joint-corrected-visual-CPU-v1/initial/objects.json','reports/fast-joint-corrected-visual-CPU-v1/step-1024/objects.json',
      'reports/fast-joint-caption-CPU-v2/initial.json','reports/fast-joint-caption-CPU-v2/step-1024.json',
      'reports/execution-group-controlled-v2-tests.log','reports/execution-group-controlled-four-CPU-v1/complete.json',
      'reports/execution-group-capacity-v1-tests.log','reports/protected-joint-stage-v1-tests.log',
      'reports/protected-joint-update-v1-tests.log','reports/protected-joint-driver-v1-tests.log']
    destination=ROOT/'reports/protected-joint-production-v1-review.json'
    if destination.exists():raise ValueError('Preserve previous full-scope review')
    review=dict(observed_at=datetime.now(timezone.utc).isoformat(),objective_preserved=True,
      previous_goal_turn_classification='progress',previous_goal_turn_evidence='Actual425production four-GPU full-weight updates and full-size short execution-hybrid mechanics/recovery.',
      current_goal_turn_classification='progress',
      current_goal_turn_progress=['Actual durable2048-update weights/all445Adam/source/RNG state read back; all72generated code outputs reexecuted.',
        'Measured coding and broad regressions changed the next research action; no regressing candidate promoted.',
        'Original trained-model CPU cache and optimized RoPE controls agree; two evaluator integration/subset faults isolated and fixed in new versions.',
        'New globally normalized controlled execution-group kernel, four-CPU replica recovery and gradient-only full-horizon mechanism tested; production RL still pending.',
        'Full-size protected language/vision/word CPU updates/recovery/cold outputs and positive own-reference KL proved.',
        'Exact old GPU owner/workers retired with evaluated2048-state retained; original work restored before new ownership.',
        'Fresh four-L40 protected full-profile capacity and exact recovery passed; actual new production updates now run.'],
      requirements={k:dict(status='incomplete_for_requested_final_model',current_evidence_and_limits=v,prior_evidence_retained=True) for k,v in requirements.items()},
      old_matched_development_outcomes=outcome['matched_development_scores'],old_complete_state_readback=state,
      actual_live_new_production_snapshot=progress,new_actual_capacity_proofs=capacity,
      actual_training_only_reference_values=984399690,exported_student_values=984399690,
      source_files_sha256={str(ROOT/p):file_hash(ROOT/p) for p in evidence},
      new_genuine_human_judgments=0,new_protected_held_quality_gain_proven=False,candidate_promoted=False,
      all_ten_final_requirements_incomplete=True,no_remaining_scope_removed=True,goal_achieved=False,
      next_required_work=['Measure protected versus its matched initial model at development checkpoints; retain clear syntax/function/efficiency and broad/image regressions.',
        'Complete the unchanged source epoch and full native endpoints only if the candidate remains useful; preserve and explicitly account for any pruned partial experiment.',
        'Independently reconcile trained model/Adam/source/RNG and every actual full benchmark result; test all held word controls.',
        'Train additional admitted OS/network/security code and continue broader public-source variety.',
        'Prove full configured four-GPU response-horizon capacity/recovery before controlled code-RL production; compare with matched verified-target training.',
        'Complete genuine-human feedback policy learning and verify reward fidelity and broad/code/vision retention.',
        'Require actual useful high-end capability evidence and a qualified counted under1B final model before completion.'])
    for name in ['README.md','RESEARCH.md','reports/completion-audit.json']:
        if str(ROOT/name) in plan['source_files_sha256']:raise ValueError('Never edit live bound documentation')
    atomic_json(destination,review)
    audit_path=ROOT/'reports/completion-audit.json';snapshot=ROOT/'reports/completion-audit-before-protected-joint-v1.json'
    if snapshot.exists():raise ValueError('Preserve original completion audit snapshot')
    shutil.copyfile(audit_path,snapshot);audit=json.loads(audit_path.read_text())
    audit.update(requirements=requirements,updated_at=review['observed_at'],goal_achieved=False,
      latest_protected_joint_full_scope_review=str(destination.relative_to(ROOT)),
      live_joint_training=dict(plan='reports/protected-joint-v1/plan.json',progress='reports/protected-joint-v1/progress.json',
        actual_observed_updates=progress['actual_step'],all_four_exact_GPU_workers_live=True,held_quality_gain_proven=False,goal_achieved=False),
      prior_joint_retired_for_coding_regression=dict(handover='reports/protected-joint-v1-handover.json',
        retained_complete_updates=2048,source_epoch_and_full_endpoints_incomplete=True),
      latest_native_development_readback='reports/fast-joint-development2048-readback-v1.json')
    atomic_json(audit_path,audit)
    message=f'''Protected joint training, **2026-10-17**: the own **984,399,690-parameter** student is now running a new four-L40 objective, with **{progress['actual_step']} actual production updates** observed. Image/word batches update vision and reading through a fixed language decoder. Text batches use a **5e-6 peak language learning rate** and **0.1 full-context KL** to a frozen copy of our own initial model. The additional **984,399,690 reference values are training-only**, so resident policy plus reference is **1,968,799,380**; the reference is excluded from the exported under1B student.\n\nThe preceding full-weight recipe was retired for persistent coding regression. These are matched **development** results, with code budget128 and math256, separate from full512-token endpoints:\n\n| Measure | Initial | 1,024 updates | 2,048 updates |\n| --- | ---: | ---: | ---: |\n| Code syntax | 20/24 | 9/24 | 12/24 |\n| Code functional correctness | 1/24 | 0/24 | 0/24 |\n| Code efficiency | 0/24 | 0/24 | 0/24 |\n| Generated knowledge | 0/56 | 0/56 | 1/56 |\n| Math exact answers | 1/16 | 0/16 | 0/16 |\n| OCR exact | 0/20 | 0/20 | 0/20 |\n| OCR character error | 0.954 | 2.015 | 0.979 |\n\nAll **72 code programs were independently reexecuted** and raw knowledge/math/OCR outcomes reconciled. The saved2,048-update state contains **445 actual Adam states / 14,286,535 targets**, with every selected language/vision/word tensor changed. The original model and evaluated checkpoints remain preserved; later uncheckpointed updates are not claimed as retained weights. The automatic all-front nonimprovement rule correctly took no action after one extra knowledge answer; the separate manual handover explicitly prioritizes the still-regressing coding milestone. This is an incomplete source epoch and incomplete full endpoint comparison, not a promoted result.\n\nThe protected replacement passed fresh configured **all-four-L40 full-length book/word/image capacity and exact weight/Adam/all-worker RNG recovery**. Each worker retains **445 Adam states**, proves nonzero own-reference KL, keeps language bytes exact during image updates, and keeps the reference unchanged. Peak allocated is about **27.11 GiB**, reserved about **30.06 GiB** per worker. Its short full-size CPU recovery/cold-load test and actual positive-KL backward also pass. **No protected-model held gain is established yet.**\n\nThe broken leading-row object subset and missing native vision-view property are preserved as failures and corrected in new evaluation versions. Matched CPU object scores6/30to7/30remain no better than blank/wrong7/30; CPU caption retrieval I2T1/32to1/32 and T2I3/32to1/32 remain weak. The new driver uses validated object controls and the exact own vision module. Its complete source curriculum is still **458,450 source-family records / 156,677,748 targets /22,065planned updates**. Additional admitted OS/network/security code, complete independent endpoints and held word checks, controlled production execution RL and genuine-human policy learning remain required. All ten final-model requirements remain incomplete.\n\nSee [live protected progress](reports/protected-joint-v1/progress.json), [fresh four-GPU proof](reports/protected-joint-v1/capacity.json), [actual earlier state readback](reports/fast-joint-step2048-readback-v1/complete.json), [raw development outcome readback](reports/fast-joint-development2048-readback-v1.json) and [current full-scope review](reports/protected-joint-production-v1-review.json). The dated entries below are historical. **No high-end parity or final candidate promotion is claimed.**\n\n'''
    for name in ['README.md','RESEARCH.md']:
        path=ROOT/name;first,_,rest=path.read_text().partition('\n\n');path.write_text(first+'\n\n'+message+rest)
    print(json.dumps(dict(review=str(destination),actual_new_production_updates=progress['actual_step'],all_ten_final_requirements_incomplete=True)))


if __name__=='__main__':main()
