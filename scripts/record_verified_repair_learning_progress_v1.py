"""Publish measured source-learning limits and complete new repair preparation."""
from copy import deepcopy
from collections import Counter
from datetime import datetime,timezone
import json,os,shutil,subprocess
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
import own_training_gpu_lease_v2 as owner

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/verified-repair-curriculum-and-matched-learning-v1-review'

def read(path):return json.loads(Path(path).read_text())

def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only full-scope progress audit required')
    master_path=ROOT/'reports/completion-audit.json';master=read(master_path);previous=ROOT/master['latest_full_scope_review']
    requirements=deepcopy(read(previous)['requirements']);runtime=ROOT/'reports/public-pretraining-recovery-v1'
    plan=read(runtime/'plan.json');owner.REPORTS=runtime;owner.resources.REPORTS=runtime;owner.owned(plan)
    active=[row for row in read(runtime/'children.json') if row['subject'].startswith('train-rank-') and not terminal(row)]
    if len(active)!=4:raise ValueError('All four actual owned original training ranks required')
    handoff=read(ROOT/'reports/instance-lineage-recovery-v2/complete.json')
    if any(terminal(row['actual_owner']) for row in handoff['actual_ten_new_controllers']):raise ValueError('All ten unchanged full downstream queues must remain live')
    rows=[]
    for line in (runtime/'training-metrics.jsonl').read_bytes().splitlines(keepends=True):
        if not line.endswith(b'\n'):break
        rows.append(json.loads(line))
    if [r['step'] for r in rows]!=list(range(1,len(rows)+1)):raise ValueError('Actual complete recovered source prefix required')
    matched_path=ROOT/'reports/own-function-training-results-CPU-v1-readback/complete.json';matched=read(matched_path)
    repairs_path=ROOT/'reports/public-function-semantic-repair-v1-readback-admission/complete.json';repairs=read(repairs_path)
    prepared_path=ROOT/'reports/full-function-repair-comparison-sources-CPU-v1/complete.json';prepared=read(prepared_path)
    validation_path=ROOT/'reports/source-function-validation-runtime-CPU-v1/complete.json';validation=read(validation_path)
    precision_path=ROOT/'reports/full-own-actor-CPU-precision-v1/complete.json';precision=read(precision_path)
    cache_path=ROOT/'reports/full-own-actor-CPU-cast-cache-v1/complete.json';cache=read(cache_path)
    regions_path=ROOT/'reports/full-function-code-target-regions-CPU-v1/complete.json';regions=read(regions_path)
    for result in [matched,repairs,prepared,validation,precision,cache,regions]:
        if not result.get('complete'):raise ValueError('Every claimed new source, actual experiment, readback and resource proof must finish')
    if (repairs['complete_semantic_repair_views']!=791 or prepared['planned_updates_per_arm']!=306
            or validation['original_source_pack_validation_functions']!=15 or matched['actual_fresh_original_and_generated_program_runs']!=112):
        raise ValueError('Full actual original source and outcome denominators required')
    at=datetime.now(timezone.utc).isoformat();OUT.mkdir();experiments=matched['actual_matched_source_train_results']
    observation=dict(at=at,actual_pretraining_owner=plan['owner'],actual_four_owned_training_ranks=active,
        actual_complete_pretraining_updates=len(rows),planned_pretraining_updates=4447,
        actual_new_response_EOS_targets=int(sum(r['response_EOS_targets'] for r in rows)),
        actual_source_family_update_counts=dict(Counter(r['family'] for r in rows)),
        actual_GPU_observation=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
        actual_matched_source_train_results=experiments,actual_fresh_program_result_reexecution_runs=112,
        repair_data_admitted_but_not_used_for_new_model_training=True,full_repair_source_preparation=prepared['actual_matched_source_budgets'],
        body_weighted_objective_implemented_and_mechanics_tested_not_full_size_trained=True,
        CPU_cached_profile_has_identical_actual_metrics_actor_and_Adam=cache['both_profiles_produce_identical_actual_metrics_actor_weights_and_Adam'],
        CPU_cast_cache_resident_inventory=cache['actual_profiles']['CPU_BF16_cached']['cast_cache_inventory'],
        persistent_free_bytes=shutil.disk_usage(ROOT).free,
        previous_goal_turn_classification='progress',current_goal_turn_classification='progress')
    atomic_json(OUT/'runtime.json',observation)
    evidence={
        'own_architecture':'Own984399690parameter scratch-origin text/image/word model remains the live dense foundation. Actual999706442parameter research actors keep the original base and train15305216actor parameters with a separate unused1536value critic. A new CPU frozen-cast cache changes no learned parameter or exported tensor and separately reports931135488derived bfloat16cast values/1862270976additional training-only bytes. A new code-region weighted CE primitive preserves every original target and true EOS; no globally novel architecture or demonstrated general capability gain is claimed.',
        'public_data':'The independently admitted1000085544target public code/math/web corpus and original17family full source schedule remain intact. Separately all826admitted original train functions are inspected for deterministic single-body AST bugs.791complete repair views pass separate source/AST/native/mask/rights readback and3504fresh program runs. Every reference and accepted negative is freshly reexecuted; every observed mutant is reconstructed/reexecuted, with0excluded first-outcome differences. Positive targets remain complete original MITsource;15original validation functions never enter the new optimization data. These are machine-derived public-example repair views, not human labels, sampled model trajectories or general correctness proofs.',
        'under_1b_parameters':'The dense student remains984399690unique registered learned values; actual research policies include the entire frozen dense backbone,15305216actor values and1536separate critic values for999706442. Additional cache tensors are deterministic bfloat16representations of the same own frozen matrices, separately counted as1862270976training-only bytes and excluded from exported learned parameter state. No hidden external teacher or expert is used. Final useful broad multimodal capability under1Bremains incomplete.',
        'text_and_images':'The live dense curriculum still retains the original selected ordinary-photo, visual-reasoning, scene/word-reading and generic-text exposures. The new language-only research keeps every dense/vision/word/unused-critic value fixed; that is not proof that adapted generated visual answers are preserved. Last completed whole-code-epoch local OCR remains0/20exact, objects3/15andphoto retrieval1/32. No new all-round image or text-in-image accuracy gain is established.',
        'human_feedback':'All85786human-ranked pairs,3721coding sources per CE/execution-RL arm and1620machine preference pairs per matched arm remain in the ten actual unchanged downstream queues. The newly completed25update own-trajectory DPO/control experiment is machine feedback on10source-train functions plus4unoptimized train-pool controls, not new human RLHF.112fresh independent program reruns reproduce every42actual generated delivery and stopping boundary. Larger public human alignment/execution RL and full final assessment remain pending.',
        'serious_full_model_training':f'All four exact original L40training ranks remain live under the recovered kernel lease. The unchanged4447update public/raw/function/retention epoch now has{len(rows)}complete actual updates and{observation["actual_new_response_EOS_targets"]}response/EOS targets in this captured prefix. First-update/every64update background durable checkpointing is active. Separate actual full-size own actors complete25logical/26physical updates per CE/trajectory-DPO arm with exact next-update weights/Adam/private-RNG recovery and cold actor export. The larger matched repair study has all306actual batches per arm prepared,4851complete positive presentations and2287110original response/EOS targets each; its full-size source capacity, training and generated evaluations have not run.',
        'high_end_llm_capability':'NOT ACHIEVED. Latest completed whole-code-epoch functional score remains0/48, ordinaryPython0/24, generated knowledge2/56, math0/16, OCR0/20, objects3/15andphoto retrieval1/32. In the separate actual source-train experiment, both CEandtrajectoryDPOmove trained public-function correctness0/10to9/10but unoptimized controls stay0/4. Syntax on all14moves1to13to14andEOSstops2to13to14(initial,CE,DPO). Every unoptimized DPOprogram has the correct entrypoint and executes its body; errors are semantic. This establishes source learning and formatting gains, not correctness transfer, high-end capability or a new result for the live public-pretraining checkpoint.',
        'broad_independent_benchmarks':'Every original nine endpoint and all16961QA,3088AI2Dunder3imageconditions,600MBPP/1816cases,4642heldhumanpairs and3429heldwordsunder3conditions remain scheduled in actual live queues. New14function research is explicitly source-train, including4unoptimized controls; it is not independent held accuracy. All15original source-pack validation functions,83public examples and1024action budgets are now fixed and the original references pass30fresh runs. Those roles remain validation/valid and are never relabelled train; all-history independence is not certified. No neural model has yet been evaluated on this new15function protocol.',
        'gpu_runtime':'All four actual L40ranks remain owned/live with the full-profile capacity, actual next-source recovery and cold native proof passed. The completed CPU2thread full-size precision probe measures25.73sFP32versus12.40sbfloat16per update. A separate within-bfloat16two-update comparison measures9.23sstandardversus7.51scached, with every actual metric/learned actor weight/Adam value identical and both recovery checks passed. Different probe timings are not multiplied into a general throughput claim. New cache/source preparation is CPU-only and never pauses or modifies the original GPU or benchmark jobs.',
        'any_code_correctness_and_efficiency':'NOT ACHIEVED. All42original own-function native action/prompt/EOS deliveries reconstruct and112fresh public runs reproduce outcomes. Both methods learn9/10optimized functions, but0/4unoptimized controls transfer, so DPOis not promoted as a correctness improvement. New791repair views provide logical counterexamples across the complete826source pool. All306matched control/repair batches retain identical source responses/EOS;15validation functions remain out of optimization. Analysis of every389458original response/EOS target finds22.745%executable token-overlap mass and53.852%docstring mass. Candidate code4weighting raises code token-weight share to54.080%while retaining all targets/EOS; Unicode,CRLFandshifted gradient tests pass. This does not measure per-region gradient contribution, establish the cause of failures, or demonstrate full-model weighted-training, generated correctness, arbitrary-code coverage or asymptotic efficiency.'}
    if set(evidence)!=set(requirements):raise ValueError('Every original final requirement must remain in scope')
    for key,value in evidence.items():requirements[key]=dict(status='incomplete_for_requested_final_model',evidence=value)
    protected=set()
    for name in subprocess.check_output(['rg','--files','reports','-g','plan.json','-g','*-queue.json'],cwd=ROOT,text=True).splitlines():
        value=read(ROOT/name)
        if isinstance(value,dict):protected.update(value.get('source_files_sha256',{}))
    for path in [master_path,ROOT/'README.md',ROOT/'RESEARCH.md']:
        if str(path) in protected:raise ValueError('Source-bound summaries cannot be edited')
    paths=[Path(__file__),previous,OUT/'runtime.json',matched_path,repairs_path,prepared_path,validation_path,precision_path,cache_path,regions_path,
        runtime/'plan.json',ROOT/'reports/instance-lineage-recovery-v2/complete.json']
    atomic_json(OUT/'review.json',dict(at=at,requirements=requirements,source_files_sha256={str(p):file_hash(p) for p in paths},
        previous_full_scope_review=str(previous.relative_to(ROOT)),previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True,original_full_scope_preserved=True,candidate_promoted=False,goal_achieved=False))
    (OUT/'previous-completion-audit.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={n:v['evidence'] for n,v in requirements.items()},latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),
        latest_actual_main_runtime=observation,all_ten_final_requirements_incomplete=True,current_goal_turn_classification='progress',candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph=(f'Verified coding research and repair curriculum, **{at}**: the unchanged four-L40public pretraining run is at **{len(rows)}/4,447updates** '
        f'(**{observation["actual_new_response_EOS_targets"]:,} response/EOS targets** in this captured prefix). All ten original full human-feedback/RL/export/evaluation queues remain live.\n\n'
        '**No correctness transfer is established:** both matched methods improve10optimized public functions from0/10to9/10, while4unoptimized source-train controls stay0/4. '
        'CEthenDPOimprove total syntax13/14to14/14andEOSstops13/14to14/14.112fresh program runs independently reproduce all42native deliveries. '
        '**The last whole-code-epoch result remains0/48functional.** These are different cohorts and models.\n\n'
        'New independent readback admits **791complete public-source repair views** after3504fresh program runs. '
        'The full826function/791repaircomparison has **306actual matched batches per arm**,4851positive presentations and2287110identical response/EOS targets; '
        'its model training is not yet run. All15source-pack validation functions/83public examples are fixed and pass30reference runs. '
        'CPU bfloat16and cast-cache resource probes pass actual full-size next-update recovery; the cache preserves learned results exactly. '
        'An optional executable-token weighting primitive is implemented/tested, with no full-model quality claim. '
        '**All ten final requirements remain incomplete.** '
        '[Matched source outcomes](reports/own-function-training-results-CPU-v1-readback/complete.json), '
        '[repair admission](reports/public-function-semantic-repair-v1-readback-admission/complete.json), '
        '[full review](reports/verified-repair-curriculum-and-matched-learning-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for path in [ROOT/'README.md',ROOT/'RESEARCH.md']:
        first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),actual_public_pretraining_updates=len(rows),all_ten_final_requirements_incomplete=True)),flush=True)

if __name__=='__main__':main()
