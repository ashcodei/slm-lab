"""Record the complete epoch, evaluation recovery and current full goal scope."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/completed-public-epoch-and-visual-integration-v1-review'
PREVIOUS = ROOT / 'reports/whole-word-microbatch-stage-v2-review/review.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def process(value):
    try:
        parts = Path('/proc', str(value['pid']), 'stat').read_text().split(') ', 1)[1].split()
    except FileNotFoundError:
        return dict(expected=value, actual_live=False)
    return dict(expected=value, actual_state=parts[0], actual_ticks=parts[19],
        actual_live=parts[19] == value['ticks'] and parts[0] not in ['Z', 'X'])


def main():
    if OUT.exists():
        raise ValueError('Preserve every prior full-scope review')
    at = datetime.now(timezone.utc).isoformat()
    epoch_path = ROOT / 'reports/public-pretraining-complete-epoch-value-readback-CPU-v2/complete.json'
    epoch = read(epoch_path)
    if not epoch['passed'] or epoch['actual_complete_updates'] != 4447:
        raise ValueError('Actual independent complete source and all-value readback required')
    evaluation = ROOT / 'reports/public-pretraining-assessment-recovery-v1'
    fresh = ROOT / 'reports/public-pretraining-native-program-action-readback-CPU-v2'
    lineage = ROOT / 'reports/assessment-recovery-lineage-v1'
    cpu = ROOT / 'reports/public-CPU-queues-assessment-recovery-v1'
    evaluation_complete = (evaluation / 'complete.json').exists()
    if not evaluation_complete:
        raise ValueError('Both full original matched native populations must complete before the final comparison review')
    recovery_state = 'completed' if evaluation_complete else 'active'
    phases = {}
    for name, folder in [('native_assessment_recovery', evaluation), ('canonical_lineage_recovery', lineage),
            ('fresh_native_program_reader', fresh), ('CPU_queue_recovery', cpu)]:
        phases[name] = {n: read(folder / (n + '.json')) for n in ['progress', 'complete', 'failure'] if (folder / (n + '.json')).exists()}
    jobs = [dict(pid=18545, ticks='50128787'), dict(pid=25240, ticks='51635952'),
        dict(pid=81363, ticks='53700533'), dict(pid=51248, ticks='53113593'),
        dict(pid=100343, ticks='54121978'), dict(pid=101001, ticks='54136176'),
        dict(pid=101000, ticks='54136176'), dict(pid=101066, ticks='54137054')]
    jobs += [row['owner'] for row in phases['canonical_lineage_recovery'].get('progress', {}).get('new_controllers', [])]
    jobs += [row['owner'] for row in phases['CPU_queue_recovery'].get('complete', {}).get('queues', [])]
    snapshot = dict(at=at, processes=[process(x) for x in jobs], operational_phases=phases,
        original_public_parent_terminal=not process(dict(pid=1241, ticks='49351203'))['actual_live'],
        actual_GPU_rows=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'], text=True), disk_free_bytes=__import__('shutil').disk_usage(ROOT).free,
        no_agent_issued_signal_or_priority_or_affinity_change_to_existing_neural_workers=True,
        changing_action_journals_and_child_registries_and_human_queue_inodes_not_bound=True)
    prior = read(PREVIOUS)
    requirements = copy.deepcopy(prior['requirements'])
    nll_dir = ROOT / 'reports/public-pretraining-recovery-v1'
    nll_paths = [nll_dir / n for n in ['held-packed-recovered-0.json', 'held-packed-4447.json']]
    baseline, trained = [read(p) for p in nll_paths]
    common = ['original_split', 'selected_original_positions', 'original_valid_population', 'evaluated_windows',
        'complete_causal_targets', 'maximum_original_context', 'original_manifest_sha256']
    if any(baseline[k] != trained[k] for k in common):
        raise ValueError('Both actual held-token observations must use the exact same whole windows and targets')
    loss = dict(initial=baseline['token_mean_NLL'], trained=trained['token_mean_NLL'],
        relative_reduction=(baseline['token_mean_NLL'] - trained['token_mean_NLL']) / baseline['token_mean_NLL'],
        original_whole_valid_windows=128, complete_causal_targets=524160,
        source_role='development validation', independent_code_correctness_not_proved=True)
    code = {}
    closed_code = []
    for arm in ['initial', 'trained']:
        count = syntax = functional = efficiency = 0
        successes = []
        for path in sorted((evaluation / arm).glob('code-*-shard-*.raw.json')):
            report = read(path)
            count += len(report['results'])
            syntax += sum(bool(r['candidate'].get('syntax_valid')) for r in report['results'])
            functional += sum(bool(r['candidate'].get('functional_passed')) for r in report['results'])
            efficiency += sum(bool(r['efficiency_stress_passed']) for r in report['results'])
            successes += [dict(language=r['language'], task=r['task'], native_finite_cases=r['candidate']['total_cases'],
                efficiency_stress_passed=r['efficiency_stress_passed']) for r in report['results'] if r['candidate'].get('functional_passed')]
            closed_code.append(path)
        code[arm] = dict(deliveries=count, full_original48case_population_complete=count == 48,
            syntax=syntax, functional=functional, efficiency=efficiency, functional_cases=successes,
            fresh_all96program_execution_complete=(fresh / 'complete.json').exists())
    if any(code[a]['deliveries'] != 48 for a in ['initial', 'trained']):
        raise ValueError('This review requires both full original native code populations')
    actual_fresh = read(fresh / 'complete.json') if (fresh / 'complete.json').exists() else None
    broad = {}
    broad_paths = []
    if evaluation_complete:
        for arm in ['initial', 'trained']:
            reports = {name: read(evaluation / arm / (name + '.json'))
                for name in ['knowledge', 'math', 'ocr', 'objects', 'vision']}
            broad_paths.extend(evaluation / arm / (name + '.json') for name in reports)
            broad[arm] = dict(knowledge_correct=sum(bool(r['generated_correct']) for r in reports['knowledge']['predictions']),
                knowledge_questions=56, math_correct=reports['math']['correct'], math_questions=16,
                OCR_exact=reports['ocr']['overall']['exact_match_accuracy'], OCR_examples=20,
                OCR_character_error_rate=reports['ocr']['overall']['character_error_rate'],
                objects_exact=reports['objects']['overall']['correct_image_exact_accuracy'], object_questions=15,
                photo_image_to_text=reports['vision']['caption_retrieval']['image_to_text_accuracy'],
                photo_text_to_image=reports['vision']['caption_retrieval']['text_to_image_accuracy'], photo_examples=32,
                photo_proper_NLL=reports['vision']['proper_nll'])
    coding_evidence = (
        'Both complete native code populations are present under matched unchanged prompts,512actions,greedyBF16. '
        f"Producer scores: functional{code['initial']['functional']}/48 to{code['trained']['functional']}/48; "
        f"syntax{code['initial']['syntax']}/48 to{code['trained']['syntax']}/48; "
        f"efficiency{code['initial']['efficiency']}/48 to{code['trained']['efficiency']}/48. "
        'The JavaScript instruction lower_bound passes268finite cases using a linear scan; it fails the logarithmic-efficiency gate. '
        'This is a small local observation, not broad coding proficiency or frontier parity. '
        + ('The independent whole96program reader passes with zero verdict differences and all24fresh reference controls; all48Python deliveries are also executed under the ordinary protocol, with0/24functional at each endpoint.' if actual_fresh else
           'The queued independent reader must still verify all96actual action-bound programs and48ordinaryPython deliveries; no favorable case is substituted for that full scope.'))
    evidence = {
        'own_architecture': 'Own scratch-origin text/image/word model remains the sole foundation. No external pretrained weights or live teacher API were loaded. New visual math kernels and a three-group optimizer integration add no learned parameters; full-size visual training admission remains pending.',
        'public_data': 'The full4447update17family raw/function/retention epoch is independently reconstructed. All244348raw windows and826Git functions occur exactly once, with every original retained exposure. The100000-row public precomputed synthetic coding shard has24271weak candidates; its full fresh reader remains unadmitted. A separate preserved finite-value diagnosis finds one order-sensitive synthetic test contradicting its explicitly unordered public contract. It does not rewrite tests or certify semantics. Those synthetic assertions remain forbidden as independent RL truth.',
        'under_1b_parameters': 'Actual dense student984399690learned values; full research actor plus critic999706442. No new learned values. Parameter limit is respected for these artifacts; the requested final capability remains incomplete.',
        'text_and_images': 'The full epoch changed all317intended language,112vision and16word tensors; all16unoptimized auxiliary/critic tensors remain bitexact. All984399690unique values are finite. Actual native object exact answers regress3/15to0/15, OCR stays0/20with higher character error, and photo caption NLL worsens1.94024to2.34348. Thus image weights changing has not established image improvement. New three-objective visual optimizer integration passes5small-own-neural checks, including four actual CPU replicas, sparse group clocks and exact next-update Adam/RNG recovery. The full302627view nine-family curriculum retains all91585words; physical pixel validation, fresh pixel readback, production private checkpoint, full-size GPU capacity/recovery and learned quality remain pending. Historical red word/prefix numerical tests stay preserved and unwaived.',
        'human_feedback': 'All85786human preference pairs, matched chosenCE/humanDPO arms and full held populations remain in the recovered canonical scopes. The ten old waiting owners ended after the terminal evaluator failure; a versioned tested handoff is restoring every exact scope. No newly completed full human-feedback milestone is claimed.',
        'serious_full_model_training': f'The actual complete own model finished4447updates and1105735574response/EOS targets. The independent reader reconstructs every original batch/order/budget and reads all984399690unique values. Held-token NLL improves{loss["initial"]:.6f} to{loss["trained"]:.6f} ({100*loss["relative_reduction"]:.2f}% lower) on the same128whole4096windows/524160targets. This is token prediction evidence. Original full-size CPU repair and interim assessment continue unchanged. New observer and source checks are not training accuracy.',
        'high_end_llm_capability': 'NOT ACHIEVED. No evidence supports GPT6-class capability or excellent all-round coding from this sub1B model. One local finite-case coding pass and token-loss reduction cannot establish that target.',
        'broad_independent_benchmarks': 'All nine fixed complete final endpoints and all16961QA,3088AI2Dquestions under three image conditions,600MBPPproblems and all original code/human/word populations remain. The original broad evaluator failed on a transparent logger positional-image API mismatch after training completed. The separate recovery completes every unchanged matched native population; the original failed runtime is never relabelled successful. Knowledge remains2/56and math0/16; object answers regress3/15to0/15. The full48case baseline has byte-identical programs and native actions under both observers on the actual984Mmodel. These are small development suites; final broad independent capability remains incomplete.',
        'gpu_runtime': f'Original training ranks and parent1241are terminal after the full epoch and a later evaluator TypeError; the old runtime restoration verifies every registered child terminal. The new exclusive evaluation-only owner100343has{recovery_state} its original fourL40 assessment scope with tested image-call transparency; actual ownership is in the runtime snapshot. No existing neural worker was signalled, reprioritized or restarted by this work. The optional three-profile full repair study and all ten canonical successors remain gated on the separately complete native assessment recovery and tested actual lineage admission.',
        'any_code_correctness_and_efficiency': coding_evidence,
    }
    if set(evidence) != set(requirements) or len(requirements) != 10:
        raise ValueError('Every original final requirement must be reviewed')
    for key, value in evidence.items():
        requirements[key] = dict(status='incomplete_for_requested_final_model', evidence=value, previous_evidence=str(PREVIOUS))
    classification = 'progress: complete4447update public epoch and all984Mvalues independently verified; visual optimizer integration5checks pass; terminal evaluation API failure preserved and recovered; full96program action-bound replay verifies coding0to1of48but syntax and visual scores regress; full human/RL/benchmark routing preserved'
    names = [
        'reports/joint-visual-objective-update-v1-tests.log', 'src/slm_lab/joint_visual_objective_update_v1.py',
        'tests/test_joint_visual_objective_update_v1.py',
        'reports/public-pretraining-complete-epoch-value-readback-CPU-v2/closed-epoch-observations.json',
        'reports/public-pretraining-complete-epoch-value-readback-CPU-v1-preflight-failure/failure-record.json',
        'reports/public-pretraining-recovery-v1/failure.json', 'reports/public-pretraining-recovery-v1/restored.json',
        'reports/public-pretraining-recovery-v1/assess-initial-broad-2.log',
        'reports/native-code-action-readback-v1-tests.log', 'src/slm_lab/native_code_action_readback_v1.py',
        'tests/test_native_code_action_readback_v1.py',
        'reports/public-pretraining-native-program-action-readback-CPU-v1/failure.json',
        'reports/native-greedy-observer-v2-tests.log', 'src/slm_lab/native_greedy_observer_v2.py',
        'tests/test_native_greedy_observer_v2.py', 'reports/public-pretraining-assessment-recovery-v1-tests.log',
        'tests/test_public_pretraining_assessment_recovery_v1.py', 'src/slm_lab/public_pretraining_assessment_recovery_v1.py',
        'scripts/recover_public_pretraining_assessments_v1.py',
        'reports/public-pretraining-assessment-recovery-v1/plan.json', 'reports/public-pretraining-assessment-recovery-v1-launch.json',
        'scripts/read_public_pretraining_native_programs_CPU_v2.py',
        'reports/public-pretraining-native-program-action-readback-CPU-v2-launch.json',
        'reports/assessment-recovery-lineage-v1-file-map.json', 'reports/assessment-recovery-lineage-v1-tests.log',
        'tests/test_assessment_recovery_lineage_v1.py', 'scripts/handoff_assessment_recovery_lineage_v1.py',
        'src/slm_lab/function_repair_GPU_contract_v2.py', 'scripts/train_full_function_repair_comparison_GPU_v2.py',
        'reports/assessment-recovery-lineage-v1/plan.json',
        'reports/public-CPU-queues-assessment-recovery-v1-file-map.json',
        'reports/public-reasoning-warmup-gate-v2-tests.log', 'tests/test_public_reasoning_warmup_gate_v2.py',
        'scripts/train_public_edge_preferences_CPU_v2.py', 'scripts/train_public_reasoning_warmup_CPU_v2.py',
        'src/slm_lab/public_reasoning_warmup_gate_v2.py', 'reports/public-CPU-queues-assessment-recovery-v1/complete.json',
        'reports/public-code-unordered-contract-diagnosis-CPU-v1/complete.json',
        'reports/native-observer-full-size-equivalence-CPU-v1/complete.json']
    paths = [PREVIOUS, epoch_path, Path(__file__), *nll_paths, *closed_code, *broad_paths, *[ROOT / n for n in names]]
    if evaluation_complete:
        paths.extend([evaluation / 'complete.json', evaluation / 'restored.json'])
    if (lineage / 'complete.json').exists():
        paths.append(lineage / 'complete.json')
    if actual_fresh is not None:
        paths.append(fresh / 'complete.json')
    files = {str(p): sha(p) for p in paths}
    OUT.mkdir()
    atomic(OUT / 'runtime-snapshot.json', snapshot)
    atomic(OUT / 'complete.json', dict(complete=True, at=at, source_files_sha256=files,
        actual_complete_epoch_updates=4447, actual_response_EOS_targets=1105735574,
        actual_unique_dense_values_verified=984399690, held_token_loss=loss,
        complete_code_producer_scores=code, independent_all96program_readback=actual_fresh,
        full_original_matched_native_broad_metrics=broad, image_regression_not_hidden=True,
        full_size_baseline_observer_transparency_cases=48, visual_optimizer_integration_small_checks=5,
        every_original_final_requirement_still_incomplete=True, goal_achieved=False))
    files[str(OUT / 'complete.json')] = sha(OUT / 'complete.json')
    files[str(OUT / 'runtime-snapshot.json')] = sha(OUT / 'runtime-snapshot.json')
    review = dict(at=at, previous_full_scope_review=str(PREVIOUS),
        previous_goal_turn_classification=prior.get('current_goal_turn_classification'),
        current_goal_turn_classification=classification, requirements=requirements,
        original_full_population_training_human_RL_and_nine_final_endpoints_preserved=True,
        source_files_sha256=files, goal_achieved=False)
    atomic(OUT / 'review.json', review)
    text = (
        f'The full public pretraining epoch completed **4,447 updates and 1,105,735,574 response/EOS targets**. '
        'The independent reader reconstructs all original source order/budgets and verifies every one of 984,399,690 unique checkpoint values. '
        f'Held code-token NLL fell **{loss["initial"]:.4f} to {loss["trained"]:.4f} ({100*loss["relative_reduction"]:.2f}%)**. '
        'This does not prove correct generated programs.\n\n'
        + coding_evidence + '\n\n'
        'The full native broad comparison is mixed: knowledge stays **2/56**, math **0/16**, and OCR **0/20**. '
        'Object answers regress **3/15 to 0/15**. Photo caption NLL worsens **1.9402 to 2.3435**; '
        'image-to-text retrieval stays **1/32**, while text-to-image rises **1/32 to 2/32**. '
        'These small development suites do not show all-round improvement. The new checkpoint is not promoted as the best all-round model.\n\n'
        'The original evaluator stopped after training because its observer rejected positional image inputs. '
        'The failure, logs and restoration remain intact. A separate exclusive runtime uses a corrected observer that passes seven own-neural transparency controls; '
        'all 48 actual full-size baseline programs and native action sequences are also unchanged across the observers. '
        'Both complete original broad populations are preserved; the independent all-96-program reader remains mandatory.\n\n'
        'A new three-objective visual optimizer integration passes five small-model checks, including four replicas and exact next-update Adam/RNG recovery. '
        'Full-size visual source admission, private checkpoint, GPU capacity and quality are still pending. '
        'The public synthetic code audit also identifies an unordered-output contract conflicting with ordered synthetic assertions; '
        'neither test verdicts nor reward protocols are rewritten.\n\n'
        'The CPU research queues have been restored with identical source and assessment scopes. '
        'The versioned ten-controller canonical handoff preserves all 85,786 human pairs, 3,721 code sources per arm, '
        '1,620 machine pairs per arm and all nine complete final endpoints. Its current actual state is in the runtime snapshot. '
        'No existing neural worker was signalled, reprioritized or restarted by this work. '
        '**The requested high-end all-round capability is not achieved; all ten final requirements remain incomplete.**\n')
    (OUT / 'metrics.md').write_text(text)
    audit_path = ROOT / 'reports/completion-audit.json'
    audit = read(audit_path)
    audit.update(at=at, updated_at=at, requirements=requirements, current_goal_turn_classification=classification,
        latest_full_scope_review=str(OUT / 'review.json'),
        latest_actual_completed_public_epoch_and_evaluation_recovery=str(OUT / 'complete.json'), goal_achieved=False)
    atomic(audit_path, audit)
    paragraph = (
        f'Complete public epoch and evaluation recovery, **{at}**: **4,447 updates / 1,105,735,574 response-EOS targets** completed. '
        'An independent source/value reader checks the full epoch and all 984M unique values. '
        f'Matched held code-token NLL improves **0.8193 to 0.7897 ({100*loss["relative_reduction"]:.2f}%)**. '
        f'Complete native producer coding scores are **{code["initial"]["functional"]}/48 to {code["trained"]["functional"]}/48 functional**, '
        f'**{code["initial"]["syntax"]}/48 to {code["trained"]["syntax"]}/48 syntax**, with **0/48 efficiency**; '
        'the independent full action-bound program replay remains required. '
        f'The terminal image-observer API failure is preserved and the separate tested four-GPU evaluation recovery is **{recovery_state}**. '
        'Knowledge/math/OCR stay2/56,0/16,0/20; object answers regress3/15to0/15. This is not all-round improvement. '
        'All full human/RL/final benchmark scopes remain; no high-end capability is established. '
        '[Evidence and limits](reports/completed-public-epoch-and-visual-integration-v1-review/metrics.md), '
        '[all ten requirements](reports/completed-public-epoch-and-visual-integration-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md', 'RESEARCH.md']:
        path = ROOT / name
        first, rest = path.read_text().split('\n', 1)
        path.write_text(first + '\n\n' + paragraph + rest.lstrip('\n'))
    if any(sha(p) != digest for p, digest in files.items()):
        raise ValueError('Every bound immutable evidence byte must remain unchanged')
    print(json.dumps(dict(review=str(OUT / 'review.json'), complete_epoch=4447,
        code_producer_scores=code, fresh_all96complete=actual_fresh is not None, goal_achieved=False)))


if __name__ == '__main__':
    main()
