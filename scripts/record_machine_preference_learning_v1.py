"""Report actual matched machine-preference learning without promoting a probe."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/full-machine-preference-learning-v1-review'


def read(path):return json.loads(Path(path).read_text())


def live(identity):
    if terminal(identity):raise ValueError('The same original live training/controller birth is required')
    f=(Path('/proc')/str(identity['pid'])/'stat').read_text().split(') ',1)[1].split()
    return dict(pid=identity['pid'],ticks=f[19],state=f[0])


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('A fresh CPU-only full-scope learning review is required')
    master_path=ROOT/'reports/completion-audit.json';master=read(master_path);previous_path=ROOT/master['latest_full_scope_review']
    requirements=deepcopy(read(previous_path)['requirements'])
    census_path=ROOT/'reports/execution-preference-edit-census-CPU-v1/complete.json';census=read(census_path)
    probe_path=ROOT/'reports/full-machine-preference-CPU-v1/complete.json';probe=read(probe_path)
    reader_path=ROOT/'reports/full-machine-preference-CPU-v2-readback/complete.json';reader=read(reader_path)
    probe_plan=read(probe_path.parent/'plan.json');tests=ROOT/'reports/machine-code-preference-stage-v1-tests.log'
    if (not terminal(probe_plan['owner']) or not probe['complete'] or not reader['complete']
            or census['actual_original_pairs']!=1620 or '4 passed' not in tests.read_text()
            or reader['actual_unique_student_values']!=999706442 or reader['actual_durable_Adam_leaves']!=561
            or not probe['actual_durable_next_source_actor_Adam_private_CPU_RNG_replay_exact']
            or not probe['every_actor_export_value_and_whole_short_source_output_cold_import_exact']):
        raise ValueError('The actual complete full-size matched learning/recovery and separate static readback must finish')
    summaries=reader['actual_source_summaries']
    if any(row['correct_program_higher_total_log_probability']!=0 for row in summaries.values()):
        raise ValueError('This recorded experiment has no successful source-ranking flip; do not invent one')
    stage=ROOT/'reports/code-first-continuation-v3';plan=read(stage/'plan.json');progress=read(stage/'progress.json')
    parent=live(plan['owner']);ranks=[p for p in read(stage/'children.json') if p['subject'].startswith('train-rank-')]
    if len(ranks)!=4:raise ValueError('All four actual production rank identities are required')
    workers=[live(p) for p in ranks]
    if (stage/'failure.json').exists():raise ValueError('Current original GPU training must remain successful')
    protected={**plan['source_files_sha256'],**probe_plan['source_files_sha256']}
    at=datetime.now(timezone.utc).isoformat()
    runtime=dict(at=at,code_first_owner=parent,actual_four_live_training_workers=workers,
        actual_code_first_updates=progress['actual_updates'],planned_code_first_updates=7558,
        actual_GPU_observation=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
        actual_GPU_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True),
        new_machine_preference_production_updates=0)
    notes={
        'own_architecture':' A new source-specific training implementation now supports full-program machine DPO and matched chosen-code CE above the own shared dense actor, without new neural parameters or an external teacher. It uses a frozen snapshot of the actual nonzero starting actor instead of substituting the bare base for that reference. Four meaningful tests compare independent joint-objective gradients, global SUM normalization, real durable recovery and corrupt-source/RNG/reference rejection. This is a custom implementation; no globally novel architecture or state-of-the-art capability is claimed.',
        'public_data':' Every one of the1620admitted original execution preferences now has a source-only similarity/length census. Of these,1471share under25percent of their complete original lines, with median exact-line similarity0.05229and chosen/rejected target-length ratio median0.99119(range0.03122to340). This does not measure semantic equivalence or model quality, but it shows why a changed-line-only objective would often cover almost the entire program. The current implementation retains full responses/EOS and all admitted pairs for its full epoch; the original pack is unchanged. The isolated CPU experiment uses only predeclared shortest whole source positions873(warm chosen code),396and27(matched training), not a completed1620pair pass.',
        'under_1b_parameters':' The actual CPU experiment counts999706442unique exported student values:984399690frozen own dense/vision/word/auxiliary values plus15306752actor/critic values. The separate static reader derives all562actor/critic tensor shapes from the actual complete main architecture, verifies561durable Adam leaves and561changed actor tensors in each arm, and checks the original base fingerprint. The training-only own reference adapter copy is excluded from the exported student and explicitly counted by the reference API. No external pretrained backbone or tokenizer is used.',
        'text_and_images':' Every original base/vision/word/auxiliary value and unused separate critic remains exact in the isolated CPU experiment, according to the producer value digests. Its two whole short-program outputs also cold-import exactly after deliberate actor perturbation. The static reader checks stored actor values and original source fingerprints; it does not repeat those neural probes. Fixed visual-encoder bytes do not prove unchanged image-conditioned answers, since language adapters change. Full visual/word/generic-photo capability remains required.',
        'human_feedback':' The new code objective uses bounded original program-execution preferences and is explicitly labelled machine feedback throughout data, metrics, recovery and exports. It is separate from the queued human-v7chosenCE/DPOexperiment on all85786human-ranked training pairs and4642held pairs. The chosen-code control makes no rejected/reference training forwards; actual full-size hooks record2policy/0referencecalls, versus6policy/6referencecalls for two DPOupdates plus one recovery replay. There are no new human judgments or claimed genuine human feedback on these code outputs.',
        'serious_full_model_training':f' The original four-L40 code/state/replay continuation remains live at{progress["actual_updates"]}/7558updates. Its completed fresh four-GPU capacity, source/RNG rejection, next-update replay and full4096/word cold checks remain intact. In parallel, an isolated complete-main CPU copy runs one chosen-code warm update to create a nonzero own starting actor, then two actual DPOupdates and a matched two-update chosen-code control from the same warm values. A real durable after1snapshot replays the second source, every actor/Adam value and private CPU RNG exactly; the retained file bytes stay unchanged. These are2pair CPU mechanics/source-learning arms, not the running production policy or a completed full preference epoch.',
        'high_end_llm_capability':f' On the two fixed training pairs, mean DPOloss changes from{summaries["baseline"]["mean_DPO_loss"]}to{summaries["machine_DPO"]["mean_DPO_loss"]}, versus{summaries["chosen_CE"]["mean_DPO_loss"]}for the matched chosen-code control. Mean own-reference-relative preference margin improves to{summaries["machine_DPO"]["mean_own_reference_margin"]}versus{summaries["chosen_CE"]["mean_own_reference_margin"]}. Both methods still rank the original incorrect program higher by total log probability on0of2successful flips; no new generated or held coding success is established. The latest completed main native coding result remains0/48functional passes. These weak source effects do not establish frontier parity.',
        'broad_independent_benchmarks':' The new CPU probe consults only admitted train sources and does not generate or score any held benchmark answers. It is not substituted for the original full16961QA,3088AI2Dtimes3conditions and600MBPP/1816case comparisons at eight fixed endpoints. All original code/human/broad and vision assessments remain queued unchanged. A full production machine-preference comparison and its own complete held/visual retention assessment are still required before any capability claim or promotion.',
        'gpu_runtime':' The same four production rank births and parent remain live with the actual GPU observations captured here. New machine-preference work uses a separate2thread CPU process, never initializesCUDA and never edits the bound current training or pending pipeline. The CPU producer and independent static reader are terminal complete; no job is restarted or stopped because of elapsed observation time. No new model candidate is promoted.',
        'any_code_correctness_and_efficiency':' The new full-program objective can use all1620execution-admitted original pairs without mislabelling rejected programs as positiveCEtargets. The actual full-size controlled result shows a small improvement in source-relative preference, and chosenCE improves chosen-program NLL slightly more, so neither method dominates all measured quantities. All original finite case, whole-input andEOS source masks remain; no generated-program correctness, arbitrary-code mastery or efficiency improvement is measured here. A first static reader assumed Linear-style factor names; its shape check correctly failed, and version2uses the actual direct down/upParameter names without modifying the experiment or weights.'}
    for key,note in notes.items():requirements[key]['evidence']+=note
    if len(requirements)!=10 or any(row['status']!='incomplete_for_requested_final_model' for row in requirements.values()):
        raise ValueError('Every original final requirement must remain explicitly incomplete')
    OUT.mkdir();atomic_json(OUT/'runtime.json',runtime)
    next_experiment=dict(status='implementation_and_GPU_capacity_still_required_not_queued',
        starting_policy='complete own broad-anchored-code-v6 anchor_execution actor; retain all complete main/code-first/human/code ancestry',
        matched_arms=['chosen_CE','machine_DPO'],all_original_admitted_pairs_per_arm=1620,
        proposed_global_batch=16,complete_unrepeated_updates_per_arm=102,
        fixed_beta=.1,fixed_chosen_anchor_weight=.1,proposed_learning_rate=1e-6,
        own_nonzero_starting_actor_reference_shares_single_counted_base=True,
        fresh_four_L40_capacity_all_private_states_and_real_cold_load_required=True,
        keep_all_existing_eight_endpoint_populations_and_source_exposure_disclosures=True,
        full_code_knowledge_math_visual_and_human_retention_comparisons_required=True,
        current_GPU_training_and_queued_models_not_changed=True,automatic_winner_promotion=False,goal_achieved=False)
    atomic_json(OUT/'next-full-source-experiment.json',next_experiment)
    evidence=[Path(__file__),previous_path,census_path,probe_path,reader_path,tests,
              ROOT/'src/slm_lab/machine_code_preference_stage_v1.py',OUT/'runtime.json',OUT/'next-full-source-experiment.json']
    atomic_json(OUT/'review.json',dict(at=at,requirements=requirements,previous_full_scope_review=str(previous_path.relative_to(ROOT)),
        actual_runtime=runtime,actual_matched_source_summaries=summaries,
        source_files_sha256={str(p):file_hash(p) for p in evidence},
        previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True,no_original_scope_removed=True,candidate_promoted=False,goal_achieved=False))
    docs=[ROOT/'README.md',ROOT/'RESEARCH.md',master_path]
    if any(str(p) in protected for p in docs):raise ValueError('Review complete; an actual fixed scientific source document must stay unchanged')
    (OUT/'previous-completion-audit.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={k:v['evidence'] for k,v in requirements.items()},
        latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),latest_actual_main_runtime=runtime,
        latest_machine_preference_edit_census=str(census_path.relative_to(ROOT)),
        latest_actual_full_machine_preference_learning=str(probe_path.relative_to(ROOT)),
        latest_full_machine_preference_static_readback=str(reader_path.relative_to(ROOT)),
        next_full_machine_preference_experiment=str((OUT/'next-full-source-experiment.json').relative_to(ROOT)),
        next_required_work=['Finish the live7558update full code/output/state/replay epoch and its complete native assessments.',
            'Retain the full85786pair human chosenCE/DPO and3721source code CE/executionRL comparisons with all planned source and recovery checks.',
            'Implement and verify a complete1620pair machine-preference comparison before claiming that its source-learning signal improves code generation; preserve all original full assessment endpoints.',
            'Complete standalone export, full code/knowledge/math/image/word/human assessments and independent outcome readback; preserve the original high-end capability objective.'],
        current_goal_turn_classification='progress',all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph=(f'Matched machine-preference learning, **{at}**: the original four-L40 code/state run continues at '
        f'**{progress["actual_updates"]}/7,558 updates**. A new machine-DPO implementation passes four independent '
        'gradient/control/recovery tests. An isolated **999,706,442-value** own model then runs matched two-pair '
        'DPO and chosen-code controls from the same nonzero starting actor, with actual durable next-update replay '
        'and cold actor import. Both arms change all 561 actor tensors; the original base/head/critic values stay exact.\n\n'
        'On these two training pairs, mean DPO loss is **0.693147 → 0.688114**, compared with **0.690569** '
        'for chosen-code CE. **Neither pair flips to preferring the correct program overall**, and no new '
        'generated-code or held capability gain is established. The full 1,620-pair production comparison remains '
        'to be implemented and validated; the running pipeline is unchanged. **All ten final requirements remain '
        'incomplete.** [Actual experiment](reports/full-machine-preference-CPU-v1/complete.json), '
        '[independent stored-state/source readback](reports/full-machine-preference-CPU-v2-readback/complete.json), '
        '[full review](reports/full-machine-preference-learning-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for path in docs[:2]:
        first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),live_code_first_updates=progress['actual_updates'],
        actual_full_size_matched_learning_complete=True,new_held_gain=False,all_ten_final_requirements_incomplete=True,goal_achieved=False)),flush=True)


if __name__=='__main__':main()
