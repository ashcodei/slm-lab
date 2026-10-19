"""Record full weak public coding data admission, not a model-quality gain."""
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess

from record_visual_stage_and_first_repair_arm_v1 import atomic, read, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/public-weak-code-admission-v1-review'
PREVIOUS = ROOT / 'reports/native-suffix-indexed-data-readiness-v1-review/review.json'


def observe(owner):
    try: fields = Path('/proc', str(owner['pid']), 'stat').read_text().split(') ', 1)[1].split()
    except FileNotFoundError: return dict(owner=owner, live=False)
    live = fields[19] == owner['ticks'] and fields[0] not in ['Z', 'X']
    return dict(owner=owner, live=live, actual_ticks=fields[19], state=fields[0],
        nice=os.getpriority(os.PRIO_PROCESS, owner['pid']) if live else None,
        scheduler=os.sched_getscheduler(owner['pid']) if live else None)


def main():
    if OUT.exists(): raise ValueError('Preserve all previous reviews')
    source = ROOT / 'reports/public-weak-code-pack-CPU-v1'
    reader = ROOT / 'reports/public-weak-code-pack-readback-admission-CPU-v1'
    probe = ROOT / 'reports/admitted-public-weak-code-interface-CPU-v1'
    pack = ROOT / 'data/public-weak-code-SFT-v1'
    for folder in [source, reader, probe]:
        if (folder / 'failure.json').exists() or not (folder / 'complete.json').is_file():
            raise ValueError('All exact complete original pack/readback/interface owners must finish successfully')
        if observe(read(folder / 'plan.json')['owner'])['live']:
            raise ValueError('Every exact source data owner must be authoritatively terminal before this closed review')
    tests = ROOT / 'reports/public-weak-code-pack-rows-v2-tests.log'
    if '8 passed' not in tests.read_text(): raise ValueError('All actual weak-only source/native/role checks must pass')
    produced, checked, interface = [read(folder / 'complete.json') for folder in [source, reader, probe]]
    admitted = read(pack / 'admission.json')
    if (not checked['data_only_weak_SFT_admitted'] or checked['admission_sha256'] != sha(pack / 'admission.json')
            or admitted['positive_train_rows'] != 21739 or admitted['separately_reserved_rows'] != 2509
            or admitted['complete_original_source_decisions'] != 100000 or admitted['complete_packed_rows_read_back'] != 24248
            or admitted['admitted_use'] != 'weak_supervised_coding_only' or admitted['human_preferences'] is not False
            or admitted['own_model_on_policy'] is not False or admitted['use_as_RL_reward_oracle_forbidden'] is not True
            or not interface['reserved_optimization_rejected'] or not interface['synthetic_test_reward_oracle_rejected']
            or not interface['all_actual_native_source_arrays_masks_true_EOS_and_padding_preserved']
            or interface['actual_counts']['source_presentations'] != 21739
            or interface['actual_counts']['response_EOS_targets'] != 6502280
            or interface['actual_counts']['original_context_tokens'] != 11557813
            or interface['actual_counts']['actual_complete_batches'] != 1359):
        raise ValueError('Every actual complete weak train/reserved role, source byte/native target and use restriction must reconcile')
    at = datetime.now(timezone.utc).isoformat()
    human = read(ROOT / 'reports/phase-human-preference-v8/chosen_CE-progress.json')
    owners = [observe(dict(pid=pid, ticks=birth)) for pid, birth in [(151931, '55253729'), (151932, '55253730'),
        (151933, '55253731'), (151934, '55253731')]]
    if not all(o['live'] for o in owners): raise ValueError('Inspect the exact original neural-worker transition')
    snapshot = dict(at=at, actual_original_human_progress=human, actual_original_GPUowners=owners,
        actual_new_source_owners=[observe(read(folder / 'plan.json')['owner']) for folder in [source, reader, probe]],
        actual_GPU_rows=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'], text=True), actual_disk_rows=subprocess.check_output(['df', '-B1', str(ROOT)], text=True),
        no_existing_neural_worker_signalled_reprioritized_restarted_or_changed=True,
        no_new_neural_model_or_program_execution_or_GPUjob_or_lease=True,
        mutable_queues_progress_journals_and_child_registries_observed_not_bound=True,
        shared_cache_cleanup_actual_user_reply_pending=not (ROOT / 'reports/shared-pretrained-cache-reclamation-v1/user-approval.json').exists())
    requirements = deepcopy(read(PREVIOUS)['requirements'])
    requirements['own_architecture']['evidence'] += (
        ' The own scratch-origin weights remain the sole foundation. A newly admitted weak public coding pack '
        'contains released Qwen2.5-Coder-32B-Instruct generated text and tests, explicitly labelled synthetic SFT '
        'data; no Qwen weights or live teacher API are loaded. This is data use, not an outside pretrained backbone. '
        'No new neural model or optimizer update occurs during source admission.')
    requirements['public_data']['evidence'] += (
        ' The complete pinned100000row OpenCodeInstruct shard now yields a separately admitted weak SFTpack with '
        '21739train/2509reserved complete programs,11557813train context tokens and6502280train response/EOS targets. '
        'The entire original100000decision inventory and23fresh failed/abstaining candidates remain. Original roles '
        'were fixed before fresh outcomes; no failed train rows are relabelled reserved. Full separate-owner pack '
        'readback reconstructs every source decision/all24248stored rows and independently retokenizes all original '
        'question/response/native masks/EOS. An actual terminal source-interface probe then constructs every1359 '
        'native train batch/all21739rows and verifies complete IDs/labels/padding/trueEOS, exact token totals and '
        'held optimization/reward-oracle rejection. Source attribution/CC-BY4/pinned revision and actual teacher '
        'response/test provenance remain. Data is weak synthetic SFTonly: generated-test agreement does not certify '
        'independent semantic correctness, human preferences, own on-policy feedback or historical independence. '
        'This one Python shard does not provide all programming languages or the whole5Mdataset.')
    requirements['human_feedback']['evidence'] = (
        f'The unchanged original four-L40chosen-response arm is{human["actual_updates"]}/5362updates, retaining all85786 '
        'public human-ranked pairs per chosen_CE/human_DPOarm. Those other-model texts ranked by humans are distinct '
        'from the new public synthetic Qwen-generated SFTdata, which has no human preferences or own on-policy '
        'samples. Full original human completion, fresh own-output judgments and held quality remain pending.')
    requirements['serious_full_model_training']['evidence'] += (
        f' Latest original human first-arm observation:{human["actual_updates"]}/5362. The new21739row weak coding '
        'source is fully admitted for data use and every train batch is exercised as tensors only. It adds0neural '
        'forwards/optimizer steps/GPUcontexts. Actual full-size mixed-source weak SFTcapacity/training/export/full '
        'matched quality remain uncompleted. All original17families/human/code/reasoning/visual/nine-endpoint scopes '
        'remain; no model promotion or new accuracy gain is inferred from this larger source admission.')
    requirements['gpu_runtime']['evidence'] += (
        ' The new source API explicitly rejects generated-test reward scoring and reserved optimization. Actual '
        'data-only batching uses1359groups/epoch and4077over3epochs, preserving65217source presentations and '
        '19506840response/EOS targets. These are actual source schedule/native transport counts, not production '
        'model updates. New source jobs use CUDA-hidden nice19single-thread idle CPUexecution and never acquire '
        'the existing exclusive GPUlease. Full actual production runtime/capacity remains required.')
    for value in requirements.values():
        if value['status'] != 'incomplete_for_requested_final_model': raise ValueError('All ten final requirements remain incomplete')
        value['previous_evidence'] = str(PREVIOUS)
    if len(requirements) != 10: raise ValueError('Retain all ten final requirements')
    names = ['src/slm_lab/public_weak_code_pack_rows_v1.py', 'src/slm_lab/public_weak_code_pack_rows_v2.py',
        'tests/test_public_weak_code_pack_rows_v1.py', 'tests/test_public_weak_code_pack_rows_v2.py',
        'scripts/run_public_weak_code_pack_rows_v1_tests.py', 'scripts/run_public_weak_code_pack_rows_v2_tests.py',
        'scripts/prepare_public_weak_code_pack_CPU_v1.py', 'scripts/read_admit_public_weak_code_pack_CPU_v1.py',
        'scripts/probe_admitted_public_weak_code_CPU_v1.py', 'src/slm_lab/public_weak_code_train_data_v1.py',
        'reports/public-weak-code-pack-rows-v1-tests.log', 'reports/public-weak-code-pack-CPU-v1-launch.json',
        'reports/public-weak-code-pack-readback-admission-CPU-v1-launch.json', 'reports/admitted-public-weak-code-interface-CPU-v1-launch.json']
    closed = [Path(__file__), PREVIOUS, tests, *[ROOT / n for n in names],
        *[folder / n for folder in [source, reader, probe] for n in ['plan.json', 'complete.json']],
        source / 'all100000-pack-dispositions.jsonl', reader / 'all100000-pack-readback-observations.jsonl',
        reader / 'independent-complete-source-readback.json', pack / 'admission.json', pack / 'manifest.json',
        *[pack / n for n in read(pack / 'manifest.json')['files_sha256']]]
    files = {str(p): sha(p) for p in closed}
    classification = ('progress:complete21739train2509reserved weak public coding source admitted after full100000decision/'
        '24248native-row independent readback; every1359actual train batch exercised without a model;8source '
        'controls pass and raw overlap-representation failure preserved;0new model training or accuracy gain')
    OUT.mkdir()
    atomic(OUT / 'runtime-snapshot.json', snapshot)
    atomic(OUT / 'complete.json', dict(complete=True, at=at, source_files_sha256=files,
        actual_whole_weak_source_controls=8, original_failed_strict_overlap_representation_test_preserved=True,
        actual_original_source_decisions=100000, actual_complete_native_rows_read_back=24248,
        actual_weak_positive_train_rows=21739, actual_separately_reserved_rows=2509,
        actual_complete_train_context_tokens=11557813, actual_train_response_EOS_targets=6502280,
        actual_all_train_native_batches_exercised=1359, actual_three_epoch_source_presentations=65217,
        actual_three_epoch_response_EOS_targets=19506840,
        weak_SFT_data_admitted=True, human_preferences=False, own_model_on_policy=False,
        generated_test_semantic_correctness_not_certified=True, use_as_RL_reward_oracle_forbidden=True,
        known_semantic_veto_and_all_fresh_failed_roles_preserved=True,
        actual_original_human_updates=human['actual_updates'], new_neural_models=0, new_neural_forwards=0,
        new_program_executions=0, new_optimizer_updates=0, new_model_quality_gain=False,
        all_ten_final_requirements_incomplete=True, goal_achieved=False))
    metrics = (
        '**The larger public coding source is now admitted for weak SFT data use:** 21,739 train examples and '
        '2,509 separately reserved examples, with 6,502,280 train response/EOS targets and 11,557,813 train context '
        'tokens. The independent owner reconstructs all 100,000 original source decisions and every 24,248 stored '
        'native row. The actual train interface then constructs all 1,359 batches and checks every ID, complete '
        'label mask, true EOS and non-target padding against the source. No neural model is involved.\n\n'
        '**Provenance stays explicit.** These are public precomputed Qwen2.5-Coder-32B-Instruct responses and generated '
        'tests from NVIDIA/OpenCodeInstruct, pinned revision 8f3ba5bafe4d6e8db46082cf7ae6741bc370604d, CC-BY-4.0, '
        'Wasi Uddin Ahmad et al. (2025). No outside weights, live teacher API or new program executions are used. '
        'Both earlier actual local execution observations remain. Agreement with generated tests is weak evidence, '
        'not independent semantic correctness, human feedback or own-policy RL. The source API rejects reward '
        'scoring and optimization of reserved rows. This is one Python shard, not all languages or all five million '
        'publisher rows.\n\n'
        '**All exclusions and roles remain.** The full 100,000-row decision ledger retains the known semantically '
        'wrong source, protected-input overlap, unsupported examples, first-pass failures and all 23 fresh '
        'mismatches/abstentions. Roles were assigned to whole duplicate components before fresh outcomes; failed '
        'train rows are not relabelled as reserved. Input/AST separation is the declared structural criterion; '
        'historical exposure and arbitrary semantic independence remain uncertified.\n\n'
        '**Eight actual source controls pass.** Complete train/reserved examples retain both closed observations '
        'and native targets; failures stay excluded, a changed returned literal cannot keep an all-pass label, '
        'reserved roles cannot switch to training, and generated examples cannot enter the human-code RLcurriculum. '
        'The first test version incorrectly accepted only null as no-overlap, while the original audit also records '
        'an empty list. That failed source/log remains; v2 accepts exactly those two original no-hit forms without '
        'changing any source role or decision.\n\n'
        'The actual complete three-epoch source schedule has 4,077 groups, 65,217 presentations and 19,506,840 '
        'response/EOS targets. These are data counts, **not completed model updates**. Full model capacity, mixed-source '
        'weak SFT training, export and complete matched code/general/visual/preference evaluation remain required. '
        'All original 17 families, human/RL sources and nine final endpoints stay intact.\n\n'
        f'The original four-L40 human first arm is **{human["actual_updates"]}/5,362 updates**. Existing neural workers '
        'remain unchanged. New source jobs are CUDA-hidden idle CPUworkers; they open no GPUcontext or lease. '
        '**No new model-quality gain is established. Latest completed coding remains 1/48 at every original study '
        'endpoint, and all ten final requirements remain incomplete.** Shared-cache cleanup still awaits the earlier '
        'actual user reply; no shared pretrained weights were used or deleted.\n')
    (OUT / 'metrics.md').write_text(metrics)
    for p in [OUT / 'complete.json', OUT / 'runtime-snapshot.json', OUT / 'metrics.md']: files[str(p)] = sha(p)
    atomic(OUT / 'review.json', dict(at=at, requirements=requirements, source_files_sha256=files,
        previous_full_scope_review=str(PREVIOUS), current_goal_turn_classification=classification, goal_achieved=False))
    path = ROOT / 'reports/completion-audit.json'
    audit = read(path)
    audit.update(at=at, updated_at=at, requirements=requirements, latest_full_scope_review=str(OUT / 'review.json'),
        latest_public_weak_coding_SFT_data_admission=str(OUT / 'complete.json'),
        admitted_public_weak_coding_SFT_data=str(pack), current_goal_turn_classification=classification, goal_achieved=False)
    atomic(path, audit)
    entry = (f'Larger public weak coding SFT source admitted, **{at}**: 21,739 train/2,509 reserved rows and 6,502,280 '
        'train response/EOS targets survive full 100,000-decision/24,248-native-row independent readback and every '
        'actual train-batch check. Public Qwen-generated text/tests remain explicit weak SFT only; no outside '
        'weights/API or RLoracle. All failed rows and original roles remain. '
        f'Original human first arm: {human["actual_updates"]}/5,362 updates. **No new model training or quality gain; '
        'all ten final requirements remain incomplete.** [Evidence](reports/public-weak-code-admission-v1-review/metrics.md), '
        '[full review](reports/public-weak-code-admission-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md', 'RESEARCH.md']:
        p = ROOT / name
        p.write_text(entry + p.read_text())
    print(str(OUT / 'complete.json'))


if __name__ == '__main__': main()
