"""Record full-source capacity preparation, actual CPUcontrols and disk relief."""
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess

from record_visual_stage_and_first_repair_arm_v1 import read, sha, atomic
from slm_lab.exact_waiting_process_retirement_v1 import terminal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/batched-reasoning-capacity-controller-v1-review'
PREVIOUS = ROOT / 'reports/parallel-batched-reasoning-stage-v2-review/review.json'
TESTS = ROOT / 'reports/batched-reasoning-capacity-controller-v2-pytest'


def main():
    if OUT.exists(): raise ValueError('Preserve every earlier completed or failed review')
    log = ROOT / 'reports/batched-reasoning-capacity-controller-v2-tests.log'
    if '5 passed' not in log.read_text(): raise ValueError('All5complete source/controller/rejection checks must pass')
    failed = ROOT / 'reports/batched-reasoning-capacity-controller-v1-tests.log'
    if '1 failed, 4 passed' not in failed.read_text() or 'Actual bounded patch-aligned complete RGB image required' not in failed.read_text():
        raise ValueError('Preserve the actual rejected out-of-range small-fixture word image and complete failure log')
    cases_path = ROOT / 'reports/batched-reasoning-actual-capacity-cases-CPU-v1/complete.json'
    cases = read(cases_path)
    expected = dict(updates=1926, source_presentations=7704, complete_context_tokens=735570,
        response_EOS_targets=57240, full_native_generation_horizon_sum=726336)
    if (not cases['complete'] or cases['complete_training_budget'] != expected
            or cases['actual_intact_original_selected_batches'] != 27 or cases['actual_complete_capacity_entries'] != 109
            or cases['actual_capacity_budget'] != dict(updates=109, source_presentations=433,
                complete_context_tokens=43899, response_EOS_targets=3688, full_native_generation_horizon_sum=38656)
            or cases['optimizer_updates'] != 0 or cases['full_size_GPUcapacity_passed'] is not False):
        raise ValueError('Every complete original training and intact actual capacity budget must remain unchanged')
    for p, digest in cases['source_files_sha256'].items():
        if sha(p) != digest: raise ValueError('Every closed original source/selection proof must retain its bytes')
    proofs = sorted(TESTS.rglob('anchor_*-rank-*.json'))
    if len(proofs) != 8: raise ValueError('Both matched profiles require all4actual CPUreplica reports')
    actual = [read(p) for p in proofs]
    for profile in ['anchor_CE', 'anchor_RL']:
        rows = sorted((r for r in actual if r['profile'] == profile), key=lambda r: r['rank'])
        if ([r['rank'] for r in rows] != list(range(4)) or len({r['final_actual_shared_checkpoint_sha256'] for r in rows}) != 1
                or any(r['actual_research_actor_parameters'] != 267718 or r['actual_native_parameters'] != 267430
                    or r['actual_complete_capacity_updates'] != 3 or r['actual_physical_optimizer_calls'] != 4
                    or r['actual_source_budget']['source_presentations'] != 9 or r['actual_Adam_states'] != 17
                    or r['actual_committed_prefix_trajectory_count'] != 36 or r['production_updates'] != 0
                    or r['full_size_GPU_capacity_passed'] is not False
                    or not all(r[k] for k in ['all_actual_next_actions_verifiers_unclipped_gradients_Adam_private_RNG_exact',
                        'invalid_peer_CPU_RNG_rejected_before_mutation', 'peer_only_valid_foreign_RNG_rejected_before_mutation',
                        'actual_peer_only_hot_model_alias_rejected_without_live_mutation',
                        'actual_peer_only_source_mode_error_coordinated_before_any_update',
                        'actual_missing_mechanics_opt_in_rejected_before_any_update'])
                    or r['independent_complete_actor_cold']['actual_text_positions'] != 32
                    or r['independent_complete_actor_cold']['actual_photo_side'] != 32
                    or r['independent_complete_actor_cold']['actual_word_shape'] != [64, 64]
                    or not r['independent_complete_actor_cold']['complete_text_photo_word_outputs_bit_exact'] for r in rows)):
            raise ValueError('Actual neural counts, recovery/cold outputs and coordinated pre-mutation rejection must reconcile')
    cache_path = ROOT / 'reports/public-PyPI-download-cache-v5/complete.json'
    cache = read(cache_path)
    if (not cache['complete'] or cache['removed_public_downloads'] != 57 or cache['removed_allocated_bytes'] != 906043392
            or not cache['installed_package_metadata_and_all_original_owners_unchanged']):
        raise ValueError('Actual safe public download-cache removal must be complete and preserve original live work')
    for p, digest in cache['source_files_sha256'].items():
        if sha(p) != digest: raise ValueError('The exact removal plan and full actual deletion journal must remain')
    human = ROOT / 'reports/phase-human-preference-v8'
    plan = read(human / 'plan.json')
    progress = read(human / 'chosen_CE-progress.json')
    live = [p for p in read(human / 'children.json') if not terminal(p)]
    if (terminal(plan['owner']) or (human / 'failure.json').exists() or not read(human / 'capacity.json')['passed']
            or len([p for p in live if p['subject'].startswith('train-chosen_CE-rank-')]) != 4):
        raise ValueError('All original first-arm human training ranks must remain live and unchanged')
    suffix = read(ROOT / 'reports/full-public-token-suffix-qualification-CPU-v1/progress.json')
    visual = read(ROOT / 'reports/full-multimodal-pixel-readback-CPU-v1/progress.json')
    at = datetime.now(timezone.utc).isoformat()
    shared = ROOT / 'reports/shared-pretrained-cache-reclamation-v1'
    shared_plan = read(shared / 'plan.json')
    shared_removed = (shared / 'complete.json').exists()
    snapshot = dict(at=at, actual_human_progress=progress, original_human_owner=plan['owner'], actual_live_human_children=live,
        actual_GPU_rows=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'], text=True), actual_free_bytes=os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize,
        full_public_code_suffix_progress=suffix, full_public_code_suffix_producer_live=not terminal(dict(pid=146084, ticks='55111340')),
        full_public_code_suffix_fresh_reader_live=not terminal(dict(pid=146633, ticks='55128888')),
        full_visual_pixel_progress=visual, full_visual_pixel_reader_live=not terminal(dict(pid=106808, ticks='54228332')),
        full_visual_batch_reader_live=not terminal(dict(pid=108345, ticks='54257301')),
        full_visual_source_admission_live=not terminal(dict(pid=109189, ticks='54271186')),
        full_visual_source_only_preparation_live=not terminal(dict(pid=136092, ticks='54738417')),
        actual_public_download_cache_bytes_reclaimed=cache['removed_allocated_bytes'],
        exact_shared_pretrained_model_cache_plan=str(shared / 'plan.json'), shared_cache_removal_applied=shared_removed,
        shared_cache_user_approval_pending=not (shared / 'user-approval.json').exists(),
        exact_shared_pretrained_cache_candidate_bytes=shared_plan['allocated_regular_file_bytes'],
        no_existing_neural_worker_signalled_reprioritized_restarted_or_patched=True,
        no_competing_GPU_job_context_or_lease_added=True, mutable_live_progress_and_registries_observed_not_bound=True)
    requirements = deepcopy(read(PREVIOUS)['requirements'])
    requirements['gpu_runtime']['evidence'] += (
        ' A connected actual-capacity controller now passes5source/runtime controls. Source selection independently enumerates '
        'every2568native input/label array and all12task/whole-batch maxima in1and3complete epochs. The full3epoch proposal keeps '
        '7704sources/57240answer-EOS targets/1926updates. Capacity retains27intact original batches, all4whole-batch rank rotations '
        'and a separate one-source zero-gradient control:109updates/433presentations/3688targets/43899context tokens. '
        'The original production epoch is divisible by4and has no empty training tail; the extra control is not called one. '
        'Both actual four-CPU267718parameter matched fixtures complete3logical/4physical calls per replica, all17Adam clocks, '
        'exact next actions/verifiers/logps/unclipped gradients/weights/Adam/private RNG, committed readback and independent '
        'cold32text/32RGB/64x64word outputs. Peer-only source-mode and hot-model alias errors reject without live mutation. '
        'The first invalid16pixel-base fixture attempted64pixel words and was correctly rejected; its failure remains, and '
        'v2uses a32pixel own fixture with a64pixel limit. No production input or runtime guard was relaxed. Actual4GPU999M '
        'capacity/FP32timing/memory/NCCL/4096text448photo64x448word cold and full source-state admission remain pending.')
    requirements['serious_full_model_training']['evidence'] += (
        f" Latest original chosen_CE observation:{progress['actual_updates']}/5362updates, all4original ranks live. "
        'The new capacity/source preparation performs0production RLupdates; all original human, warmup, visual and9endpoint '
        'scopes remain. Proposed109capacity updates are not a completed1926update reasoning run or the old single-source protocol.')
    requirements['text_and_images']['evidence'] += (
        ' New reasoning-controller cold checks preserve all own base/head/actor/critic values in a separately loaded own '
        'model, with exact FP32text/photo/whole-word outputs on explicit small CPUshapes. Loading still uses the own base '
        'plus committed actor parts; this is not the final complete standalone package or full-size multimodal-quality evidence.')
    requirements['public_data']['evidence'] += (
        f" Current full human-code source controls cover{suffix['actual_counts']['complete_original_sources']}/3721originals; "
        'the separate full fresh reader remains.864.07MiBof unused positively identified public PyPIHTTPwheel downloads '
        'were reclaimed with exact-byte/version/open-file/source-binding and live-owner checks, allowing source validation '
        'to make progress again. No installed files, own model artifacts, datasets or bound inputs were removed. '
        'A distinct32.1GiBshared pretrained-cache removal plan awaits actual user approval; no external weights enter training.')
    for row in requirements.values(): row['previous_evidence'] = str(PREVIOUS)
    if len(requirements) != 10 or any(r['status'] != 'incomplete_for_requested_final_model' for r in requirements.values()):
        raise ValueError('All ten final requirements remain incomplete')
    names = ['src/slm_lab/batched_reasoning_capacity_cases_v1.py', 'src/slm_lab/batched_reasoning_capacity_controller_v1.py',
        'tests/test_batched_reasoning_capacity_controller_v1.py', 'tests/test_batched_reasoning_capacity_controller_v2.py',
        'scripts/run_batched_reasoning_capacity_controller_v1_tests.py', 'scripts/run_batched_reasoning_capacity_controller_v2_tests.py',
        'scripts/prepare_batched_reasoning_actual_capacity_CPU_v1.py', 'scripts/reclaim_public_installed_download_cache_v3.py',
        'scripts/reclaim_public_installed_download_cache_v4.py', 'scripts/reclaim_public_PyPI_download_cache_v5.py',
        'scripts/reclaim_shared_pretrained_cache_v1.py', 'reports/public-installed-download-cache-v3-prepare.log',
        'reports/public-installed-download-cache-v4-prepare.log', 'reports/public-installed-download-cache-v4/plan.json',
        'reports/public-installed-download-cache-v4/not-applied.json', 'reports/public-PyPI-download-cache-v5-prepare.log',
        'reports/public-PyPI-download-cache-v5-apply.log', 'reports/batched-reasoning-actual-capacity-cases-CPU-v1.log',
        'reports/shared-pretrained-cache-reclamation-v1-prepare.log']
    closed = [Path(__file__), PREVIOUS, log, failed, cases_path, cache_path, shared / 'plan.json',
        *[ROOT / n for n in names], *[Path(p) for p in cases['source_files_sha256']],
        *[Path(p) for p in cache['source_files_sha256']], *sorted(TESTS.rglob('*.json')), *sorted(TESTS.rglob('*.pt'))]
    files = {str(p): sha(p) for p in closed}
    classification = ('progress:full source capacity109steps/433presentations selected from intact3epoch reasoning scope; '
        '5source/controller checks pass with actual4CPUreplica private recovery and text/photo/word cold; '
        '864MiBpublic cache reclaimed and full source validation resumes; shared32.1GiBmodel-cache removal awaits approval; '
        '0new production RLupdates or accuracy gain')
    OUT.mkdir()
    atomic(OUT / 'runtime-snapshot.json', snapshot)
    atomic(OUT / 'complete.json', dict(complete=True, at=at, source_files_sha256=files, actual_new_controls=5,
        actual_own_small_research_parameters=267718, actual_own_small_native_parameters=267430,
        actual_matched_profiles=['anchor_CE', 'anchor_RL'], actual_CPUreplicas_per_profile=4,
        actual_logical_updates_per_profile=3, actual_physical_Adam_calls_per_replica_per_profile=4,
        actual_independent_text_photo_word_cold_outputs_bit_exact=True, actual_fixture_word_size_failure_preserved=True,
        full_original_proposed_training_budget=expected, full_original_capacity_budget=cases['actual_capacity_budget'],
        actual_intact_source_batches=27, actual_capacity_entries=109, all_selected_worst_sources_reach_every_replica=True,
        actual_public_download_bytes_reclaimed=cache['removed_allocated_bytes'],
        exact_shared_cache_candidate_bytes=shared_plan['allocated_regular_file_bytes'], shared_cache_removal_applied=shared_removed,
        actual_human_first_arm_updates=progress['actual_updates'], actual_human_first_arm_planned_updates=5362,
        full_size_GPUcapacity_owned_dispatch_standalone_and_full_assessment_pending=True,
        new_production_RL_updates=0, new_model_quality_gain=False, all_ten_final_requirements_incomplete=True, goal_achieved=False))
    metrics = (
        f"The original four-L40human first arm is **{progress['actual_updates']}/5362updates** at this observation. "
        'All85786human pairs per arm and original full assessments remain.\n\n'
        '**Five new source/controller checks pass.** Complete native input/label enumeration verifies all2568train sources, '
        '12task families and whole-batch/source maxima. The full three-epoch proposal retains7704source presentations, '
        '57240whole answer/EOS targets and1926global updates. Capacity selects27intact original batches, rotates every whole '
        'batch across all4physical ranks and adds one separate zero-gradient control: **109updates/433presentations/3688targets**. '
        'All64/256action budgets remain. The original complete epoch has no empty training tail; the control is separate.\n\n'
        'Both deliberately initialized267718parameter CPUfixtures run4actual replicas,3logical/4physical Adam calls,9sources '
        'and17Adam states. Committed next private draws, actions, verifiers, probabilities, unclipped-gradient digest, weights, '
        'Adam and private RNG replay exactly. Separate own-base-plus-actor cold loads preserve every parameter and complete '
        '32text/32RGB/64x64whole-word FP32outputs. A peer-only hot-model alias and source-mode error reject without live mutation. '
        'The first smaller fixture correctly rejected a64pixel word outside its32pixel image limit; its failed log and states '
        'remain. V2uses a valid32pixel base/64pixel limit. **No full-model guard or source horizon was shortened.**\n\n'
        '**These are small mechanics and source preparation.** Actual999Mfour-GPU capacity, NCCL, FP32memory/timing, '
        '4096text/448photo/64x448word cold, complete source-state readback, owned dispatch, production RL, final standalone '
        'packaging and full matched assessments remain pending. The capacity report validator alone grants no production admission. '
        'Current coding correctness remains1/48at all four completed coding-study endpoints; no winner or new quality gain is claimed.\n\n'
        '**864.07MiBof unused public package download copies were reclaimed.** All57responses were identified as public '
        'PyPIwheels, with matching cache/package bytes and guarded source bindings/open files/live neural owners. Installed '
        'packages, own models and datasets were preserved. Source validation is again progressing. Two shared pretrained '
        'cache folders total32.1GiB; their exact file inventory/removal plan is prepared and awaits actual user approval because '
        'the folders are shared outside this project. They are not used by SLMtraining. No timed or default approval is inferred.\n\n'
        '**All ten final requirements remain incomplete.** The complete original human, reasoning warmup, visual, code and '
        'nine final endpoint scopes remain unchanged. No competing GPUcontext/job/lease or existing neural worker change occurred.\n')
    (OUT / 'metrics.md').write_text(metrics)
    for p in [OUT / 'runtime-snapshot.json', OUT / 'complete.json', OUT / 'metrics.md']: files[str(p)] = sha(p)
    atomic(OUT / 'review.json', dict(at=at, requirements=requirements, source_files_sha256=files,
        previous_full_scope_review=str(PREVIOUS), previous_goal_turn_classification=read(PREVIOUS)['current_goal_turn_classification'],
        current_goal_turn_classification=classification, goal_achieved=False))
    path = ROOT / 'reports/completion-audit.json'
    audit = read(path)
    audit.update(at=at, updated_at=at, requirements=requirements, latest_full_scope_review=str(OUT / 'review.json'),
        latest_batched_reasoning_capacity_controller=str(OUT / 'complete.json'),
        next_batched_reasoning_capacity_controller=str(ROOT / 'src/slm_lab/batched_reasoning_capacity_controller_v1.py'),
        next_batched_reasoning_actual_capacity_plan=str(ROOT / 'reports/batched-reasoning-actual-capacity-cases-CPU-v1/actual-complete-source-capacity-plan.json'),
        current_goal_turn_classification=classification, goal_achieved=False)
    atomic(path, audit)
    entry = (f'Full-source batched reasoning capacity, **{at}**: five source/controller checks pass with all2568native sources/12families '
        'and actual four-CPUprivate recovery/cold. Full capacity preserves27original batches×4rank rotations plus a separate '
        'zero-gradient control:109updates/433presentations. The full proposed7704source/57240target training scope remains. '
        f"Original human first arm:{progress['actual_updates']}/5362updates.864MiBpublic download cache reclaimed; shared32.1GiBcache "
        'removal awaits approval. **No new production RLupdates/accuracy gain; all ten final requirements remain incomplete.** '
        '[Evidence](reports/batched-reasoning-capacity-controller-v1-review/metrics.md), '
        '[full review](reports/batched-reasoning-capacity-controller-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md', 'RESEARCH.md']:
        p = ROOT / name
        p.write_text(entry + p.read_text())
    print(str(OUT / 'complete.json'))


if __name__ == '__main__': main()
