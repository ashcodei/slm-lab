"""Record measured modality exposure and full source readiness without shrinking the goal."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from slm_lab.exact_waiting_process_retirement_v1 import terminal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/visual-modality-clock-and-full-source-chain-v1-review'
PREVIOUS = ROOT / 'reports/completed-public-epoch-and-visual-integration-v1-review/review.json'


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


def actual(birth):
    result = dict(expected=birth, actual_live=not terminal(birth))
    p = Path('/proc', str(birth['pid']), 'stat')
    if p.exists():
        parts = p.read_text().split(') ', 1)[1].split()
        result.update(actual_state=parts[0], actual_ticks=parts[19])
    return result


def main():
    if OUT.exists():
        raise ValueError('Preserve all prior full-scope reviews')
    at = datetime.now(timezone.utc).isoformat()
    prior = read(PREVIOUS)
    clocks_path = ROOT / 'reports/public-pretraining-modality-clock-readback-CPU-v1/complete.json'
    clocks = read(clocks_path)
    if (not clocks['passed'] or clocks['actual_Adam_leaves_checked'] != 445
            or {k: v['actual_clock'] for k, v in clocks['actual_groups'].items()} != dict(language=4426, vision=21, word=5)):
        raise ValueError('Actual complete private Adam clocks and all445leaf observations required')
    for name, expected in [('independent-multimodal-pixels-v1-tests.log', '6 passed'),
            ('full-visual-source-admission-v1-tests.log', '9 passed')]:
        if expected not in (ROOT / 'reports' / name).read_text():
            raise ValueError('Actual canonical/producer input parity and full-scope admission controls required')
    stage = ROOT / 'reports/full-function-repair-comparison-GPU-v2'
    children = read(stage / 'children.json')
    capacity = {}
    capacity_paths = []
    for profile in ['plain_repeat', 'semantic_repair', 'weighted_semantic_repair']:
        folder = stage / ('capacity-' + profile)
        paths = [folder / ('rank-' + str(rank) + '.json') for rank in range(4)]
        workers = [c for c in children if c.get('subject', '').startswith('capacity-' + profile)]
        if not all(p.exists() for p in paths) or not workers or any(not terminal(c) for c in workers):
            continue
        rows = [read(p) for p in paths]
        for rank, p in enumerate(rows):
            if (not p['complete'] or p['physical_GPU'] != rank or p['actual_Adam_states'] != 561
                    or p['logical_updates'] != 3 or p['physical_optimizer_calls'] != 4
                    or not p['recovery']['exact_next_draws_global_update_actor_Adam_and_private_RNG']
                    or not p['recovery']['wrong_valid_peer_RNG_rejected_before_mutation']
                    or not p['cold_export_import']['actual4096text448photo_real_word_outputs_bit_exact']
                    or p['production_updates'] != 0):
                raise ValueError('Every full actual fourL40capacity/private-recovery/cold-output control must match')
        capacity[profile] = dict(actual_L40s=4, actual_Adam_leaves_each=561, actual_logical_updates_each=3,
            actual_physical_optimizer_calls_each=4, next_source_update_Adam_and_private_RNG_exact=True,
            actual4096text448photo_real_word_outputs_bitexact=True,
            maximum_peak_allocated_bytes=max(p['peak_allocated_bytes'] for p in rows), production_updates=0)
        capacity_paths.extend(paths)
    chain = []
    for stem in ['full-multimodal-pixel-readback-CPU-v1', 'full-corrected-multimodal-batch-readback-CPU-v1',
            'full-corrected-multimodal-source-admission-CPU-v1']:
        receipt = read(ROOT / 'reports' / (stem + '-launch.json'))
        folder = ROOT / 'reports' / stem
        if (folder / 'failure.json').exists():
            raise ValueError('A new complete source-chain worker failed; preserve and resolve it before reporting readiness')
        chain.append(dict(stage=stem, birth=receipt['owner'], actual_process=actual(receipt['owner']),
            current_progress=read(folder / 'progress.json') if (folder / 'progress.json').exists() else None,
            data_or_training_complete=(folder / 'complete.json').exists()))
    snapshot = dict(at=at, full_source_chain=chain, complete_repair_GPU_capacity_profiles=capacity,
        active_GPU_controller=actual(read(ROOT / 'reports/full-function-repair-comparison-GPU-v2-queue.json')['owner']),
        registered_GPU_children_observed_without_binding_mutable_registry=[actual(c) for c in children],
        original_CPU_training=actual(dict(pid=18545, ticks='50128787')),
        original_interim_assessment=actual(dict(pid=25240, ticks='51635952')),
        original_code_reader=actual(dict(pid=81363, ticks='53700533')),
        original_pixel_producer=actual(dict(pid=51248, ticks='53113593')),
        original_pixel_progress=read(ROOT / 'reports/full-multimodal-pixel-validation-CPU-v1/progress.json'),
        actual_GPU_rows=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'], text=True), disk_free_bytes=__import__('shutil').disk_usage(ROOT).free,
        no_existing_neural_worker_signalled_reprioritized_or_restarted=True)
    requirements = deepcopy(prior['requirements'])
    changes = {
        'public_data': 'The complete nine-family visual data path now has exact live source-only readers queued in order: full195121physical redecodes covering302627views/91585words, then all4733globalsteps/18932actualrankbatch deliveries, then a separate data-only admission. The independently implemented decoder passes6canonical/producer tensor-geometry controls. Nine admission-scope tests reject partial populations, changed targets/word bytes, lost empty ranks/negatives and held roles. No new source training admission or neural job is inferred before every actual full reader passes. Existing public code readback and all original17family/human/RL populations remain.',
        'text_and_images': 'The actual final private Adam snapshot and all4447journal records independently confirm4426language updates,21vision updates and5word-head updates, across all445optimized leaves. Vision consumes19488source presentations; word consumes4864. This is evidence of sparse visual updates during the completed epoch, not a proof of the regression cause or a lifetime exposure count. The prepared full visual epoch retains4733vision updates and1432word updates with the text model fixed, but has not trained. Full source-chain checks are queued; production private checkpoint, actual full-size GPU visual capacity/recovery and quality remain pending. Historical failed word/prefix numeric controls stay unwaived.',
        'serious_full_model_training': 'The complete4447update/1105735574target public epoch and its measured mixed quality remain. New full-size repair capacity has completed the profiles listed in the runtime snapshot on all four actualL40s,561Adam leaves each,3logical/4physicalupdates, exact next-source Adam/private-RNG recovery and cold4096text448photo/word outputs. Every remaining profile must pass before all306productionupdates per each of the three original arms. Independent CPU repair and interim assessment continue unchanged. New source checks and private-clock observations are not model accuracy gains.',
        'gpu_runtime': 'The exact recovered GPU controller101066continues the complete three-profile fourL40study. Closed actual capacity profiles are verified separately from the changing child registry. New source-only workers are CUDA-hidden, nice19, one thread and SCHED_IDLE; they never obtain a GPU lease or construct a model. Existing neural workers are not signalled, reprioritized or restarted. Current utilization belongs to the exact snapshot; capacity success is not an efficiency or capability score.',
    }
    for key, value in changes.items():
        requirements[key] = dict(status='incomplete_for_requested_final_model', evidence=value, previous_evidence=str(PREVIOUS))
    for key in requirements:
        requirements[key]['previous_evidence'] = str(PREVIOUS)
    if len(requirements) != 10 or any(x['status'] != 'incomplete_for_requested_final_model' for x in requirements.values()):
        raise ValueError('All ten original final-model requirements remain unachieved')
    names = ['scripts/read_public_pretraining_modality_clocks_CPU_v1.py',
        'reports/public-pretraining-modality-clock-readback-CPU-v1/all445-actual-saved-Adam-clocks.jsonl',
        'src/slm_lab/independent_multimodal_pixels_v1.py', 'tests/test_independent_multimodal_pixels_v1.py',
        'reports/independent-multimodal-pixels-v1-tests.log', 'scripts/read_full_multimodal_pixels_CPU_v1.py',
        'scripts/audit_full_corrected_multimodal_batches_CPU_v1.py', 'src/slm_lab/full_visual_source_admission_v1.py',
        'tests/test_full_visual_source_admission_v1.py', 'reports/full-visual-source-admission-v1-tests.log',
        'scripts/admit_full_corrected_multimodal_source_CPU_v1.py',
        'reports/full-function-repair-comparison-GPU-v2/plan.json',
        'reports/assessment-recovery-lineage-v1/complete.json',
        'reports/full-multimodal-corrected-source-index-CPU-v1/complete.json',
        'reports/full-multimodal-corrected-source-index-v1-readback/complete.json']
    for item in chain:
        names += ['reports/' + item['stage'] + '-launch.json', 'reports/' + item['stage'] + '/plan.json']
    paths = [PREVIOUS, clocks_path, Path(__file__), *capacity_paths, *[ROOT / name for name in names]]
    files = {str(p): sha(p) for p in paths}
    OUT.mkdir()
    atomic(OUT / 'runtime-snapshot.json', snapshot)
    proof = dict(complete=True, at=at, source_files_sha256=files, actual_complete_public_updates=4447,
        actual_group_Adam_clocks={k: v['actual_clock'] for k, v in clocks['actual_groups'].items()},
        actual_Adam_leaves_read=445, new_independent_pixel_protocol_tests_passed=6,
        new_complete_source_admission_scope_tests_passed=9, new_actual_source_chain=chain,
        actual_full_size_four_GPU_capacity_profiles_complete=capacity,
        new_visual_source_training_admitted=False, new_visual_training_updates=0,
        new_model_quality_gain_not_established=True, all_original_human_RL_and_nine_final_endpoints_preserved=True,
        previous_goal_turn_classification='progress: completed full epoch evaluations, independently reexecuted all96programs and restored all ten canonical queues',
        current_goal_turn_classification='progress: all445private Adam clocks measured; full source decoder parity6and admission scope9checks pass; exact complete pixel/rank-batch/data-admission chain running; full actual fourL40repair capacity advances',
        goal_achieved=False)
    atomic(OUT / 'complete.json', proof)
    files[str(OUT / 'complete.json')] = sha(OUT / 'complete.json')
    files[str(OUT / 'runtime-snapshot.json')] = sha(OUT / 'runtime-snapshot.json')
    atomic(OUT / 'review.json', dict(at=at, previous_full_scope_review=str(PREVIOUS), requirements=requirements,
        current_goal_turn_classification=proof['current_goal_turn_classification'],
        source_files_sha256=files, all_original_scopes_preserved=True, goal_achieved=False))
    paragraph = (
        f'Visual exposure diagnosis and full source readiness, **{at}**: the actual final private optimizer confirms '
        '**4,426 language / 21 vision / 5 word-head updates** in the completed 4,447-update epoch, across all445Adam leaves. '
        'Sparse visual exposure while the language model changes is a plausible contributor to the image regressions; the counts do not prove causation. '
        'The larger visual epoch remains untrained. Its exact **195,121-image / 302,627-view / 91,585-word** pixel, '
        '**4,733-step / 18,932-rank-call** batch and data-admission chain is queued, with **6input-parity and9scope checks passing**. '
        f'Actual four-L40 repair capacity profiles completed so far: **{", ".join(capacity) or "none"}**; these are runtime/recovery controls, not quality scores. '
        '**No new all-round capability is established; all ten final requirements remain incomplete.** '
        '[Measured clocks and pending work](reports/visual-modality-clock-and-full-source-chain-v1-review/metrics.md), '
        '[full scope review](reports/visual-modality-clock-and-full-source-chain-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    table = ('The private snapshot confirms the completed epoch exposure, not lifetime training or a causal explanation. '
        'The planned visual epoch uses smaller batches and different objectives, so update ratios are not performance gains.\n\n'
        '| Optimized group | Completed public epoch | Prepared visual epoch, untrained |\n'
        '|---|---:|---:|\n| Language | 4,426 | 0 (fixed) |\n| Vision | 21 | 4,733 |\n| Word head | 5 | 1,432 |\n\n'
        'All445saved optimizer clocks match every actual closed journal record. The new independent image decoder passes six actual input-protocol controls '
        'against the producer and the established training arrays. The queued full reader must still decode all195121physical sources and reconstruct all302627views. '
        'The following batch audit must compare every native tensor, word byte, matching caption, corrected positive and retained original negative across all4733steps and four rank positions. '
        'A separate data-only admission worker waits for every exact full source stage; nine scope tests prevent small representative results or altered budgets/roles from opening it. '
        'GPU visual training additionally requires a complete private checkpoint/controller and actual full-size capacity/recovery; these remain pending.\n\n'
        f'Closed full-size fourL40repair capacity profiles: {", ".join(capacity) or "none"}. Each completed profile includes all561Adam leaves, '
        'actual next-source private-RNG recovery and cold4096text/448photo/word outputs. Production and quality remain separate. '
        'The complete three-profile source budgets and all original human/RL/nine-endpoint evaluations remain.\n\n'
        'Latest verified coding remains1/48functional and0/48efficiency, with0/24ordinaryPython; image/object scores did not improve. '
        'No model-quality gain is claimed by these data/runtime checks. [Previous matched metrics](../completed-public-epoch-and-visual-integration-v1-review/matched-metrics.md).\n')
    (OUT / 'metrics.md').write_text(table)
    audit = read(ROOT / 'reports/completion-audit.json')
    audit.update(at=at, updated_at=at, requirements=requirements, latest_full_scope_review=str(OUT / 'review.json'),
        current_goal_turn_classification=proof['current_goal_turn_classification'],
        latest_actual_visual_modality_clock_and_full_source_chain=str(OUT / 'complete.json'), goal_achieved=False)
    atomic(ROOT / 'reports/completion-audit.json', audit)
    for name in ['README.md', 'RESEARCH.md']:
        path = ROOT / name
        value = path.read_text()
        if value.startswith('# '):
            first, rest = value.split('\n', 1)
            path.write_text(first + '\n\n' + paragraph + rest.lstrip('\n'))
        else:
            path.write_text(paragraph + value)
    if any(sha(p) != digest for p, digest in files.items()):
        raise ValueError('All bound closed evidence bytes must remain exact')
    print(json.dumps(dict(review=str(OUT / 'review.json'), Adam_leaves=445, new_tests=15,
        actual_four_GPU_capacity_profiles=list(capacity), new_visual_updates=0, goal_achieved=False)))


if __name__ == '__main__':
    main()
