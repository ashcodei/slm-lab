"""Check every real rank batch in the complete corrected visual source epoch.

Wait for the exact independent full pixel reader, then deliver all302627
original/current views in the original4733steps across all four real shard
positions. Compare every complete tensor to established encoders, including
all word bytes, caption-only pairs, true EOS and the three empty tail ranks.
No neural graph or source-training admission is created here.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import torch

from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
from slm_lab.codenet_replay_v1 import FullReplayEncoder
from slm_lab.corrected_multimodal_batches_v1 import CorrectedMultimodalBatches
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.model import ModelConfig
from slm_lab.shared_word_data_v1 import SharedWordData

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/full-corrected-multimodal-batch-readback-CPU-v1'
PIXELS = ROOT / 'reports/full-multimodal-pixel-readback-CPU-v1'
PIXEL_OWNER = dict(pid=106808, ticks='54228332')
REPRESENTATIVE = ROOT / 'reports/corrected-multimodal-batches-CPU-v1/complete.json'
CONFIG = ROOT / 'data/public-reasoning-curriculum-v1/model-config.json'
RESERVE = 40 * 1024**3


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def state(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def atomic(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def tensor_sha(value):
    if value.device.type != 'cpu' or value.requires_grad:
        raise ValueError('Source-only batch tensors must remain on CPU without neural gradients')
    return hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh one-thread CUDA-hidden nice19 complete batch source readback required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    original = read(REPRESENTATIVE)
    pixel_plan = read(PIXELS / 'plan.json')
    if (not original['passed'] or original['actual_source_families'] != 9
            or {k: pixel_plan['owner'][k] for k in ['pid', 'ticks']} != PIXEL_OWNER):
        raise ValueError('Actual representative tensor-boundary controls and exact full pixel-reader birth required')
    files = {str(p): sha(p) for p in [REPRESENTATIVE, CONFIG, PIXELS / 'plan.json',
        *python_sources(ROOT, [Path(__file__)])]}
    states = {p: state(p) for p in files}
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    OUT.mkdir()
    atomic(OUT / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=owner, source_files_sha256=files,
        exact_independent_pixel_reader=PIXEL_OWNER, all_original_native_views=302627, all_original_full_steps=4733,
        world_size=4, original_full_rank_calls=18932, whole_word_views=91585,
        every_actual_rank_native_tensor_compared_bitexact_to_established_encoder=True,
        every_caption_only_matching_input_and_word_UTF8_target_and_true_EOS_required=True,
        all4original_complete_correction_negatives_and_three_empty_singleton_ranks_required=True,
        actual_declared_maximum_native_context=2952, original_source_order_and_budgets_preserved=True,
        scheduler='SCHED_IDLE', CPU_threads=1, CUDA_hidden=True, minimum_free_bytes=RESERVE,
        CPU_working_allowance=8 * 1024**3, other_worker_headroom=64 * 1024**3,
        parent_address_space_limit=24 * 1024**3, maximum_output_bytes=1024**3,
        no_existing_neural_job_changed=True, models_constructed=0, neural_forwards=0,
        source_role='train', training_admitted=False, goal_achieved=False))
    waits = Counter()

    def guard():
        if os.sched_getscheduler(0) != os.SCHED_IDLE or any(state(p) != expected for p, expected in states.items()):
            raise ValueError('Every immutable source and source-only idle scheduling must remain exact')

    def checked():
        result = wait_for_reserve(ROOT, minimum_bytes=RESERVE, guard=guard)
        waits['seconds'] += result['waited_seconds']
        waits['failed_space_observations'] += result['failed_space_observations']
        stat = {k: int(v) for k, v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
        rss = next(int(line.split()[1]) * 1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
        if (rss > 8 * 1024**3 or int(Path('/sys/fs/cgroup/memory.max').read_text())
                - stat['anon'] - stat['shmem'] - stat['kernel'] < 64 * 1024**3):
            raise ValueError('Declared complete source-audit working memory and other-worker headroom required')

    try:
        while not terminal(PIXEL_OWNER):
            guard()
            atomic(OUT / 'progress.json', dict(owner=owner, phase='waiting for exact independent full pixel readback',
                exact_predecessor_live=True, neural_forwards=0, training_admitted=False))
            time.sleep(30)
        if (PIXELS / 'failure.json').exists() or not (PIXELS / 'complete.json').exists():
            raise ValueError('The exact full physical readback ended without a complete result')
        pixels = read(PIXELS / 'complete.json')
        if (not pixels['passed'] or pixels['actual_original_physical_paths'] != 195121
                or pixels['actual_original_native_views'] != 302627 or pixels['actual_original_whole_word_views'] != 91585):
            raise ValueError('Every original physical pixel source and all302627native views must pass before batch delivery')
        files[str(PIXELS / 'complete.json')] = sha(PIXELS / 'complete.json')
        resource.setrlimit(resource.RLIMIT_AS, (24 * 1024**3, 24 * 1024**3))
        checked()
        config = ModelConfig(**read(CONFIG))
        source = CorrectedMultimodalBatches(ROOT, config, purpose='audit')
        if any(p in files and files[p] != digest for p, digest in source.files.items()):
            raise ValueError('The complete native adapter and original source evidence have conflicting bindings')
        files.update(source.files)
        schedule_path = source.folder / 'complete-corrected-single-epoch-schedule.json'
        schedule = read(schedule_path)
        files[str(schedule_path)] = sha(schedule_path)
        if (schedule['global_width'] != 64 or schedule['world_size'] != 4 or len(schedule['entries']) != 4733
                or schedule['all_original_source_occurrences_once'] is not True or source.training_admitted):
            raise ValueError('Exact unadmitted complete original64source/fourrank schedule required')
        if any(sha(p) != digest for p, digest in files.items()):
            raise ValueError('Every corrected/original source byte and actual encoder must match before full delivery')
        states.update({p: state(p) for p in files})
        atomic(OUT / 'closed-source-bindings.json', dict(owner=owner, source_files_sha256=files,
            full_independent_pixel_reader_terminal_and_passed=True, no_live_source_outputs_bound=True))
        golden = FullReplayEncoder(source.tokenizer, config, ROOT, 448)
        words = SharedWordData(ROOT / 'data/shared-byte-words-v1', 'train', 'audit')
        seen = {name: Counter() for name in source.names}
        budgets = {name: dict(original_views=0, complete_context_tokens=0, response_EOS_targets=0) for name in source.names}
        totals = Counter()
        corrected = set()
        observations = OUT / 'all4733-complete-four-rank-source-batch-observations.jsonl'
        with observations.open('x') as stream:
            for step, entry in enumerate(schedule['entries'], 1):
                checked()
                name, positions = entry['family'], entry['original_local_positions']
                if (entry['step'] != step or name not in seen or len(positions) != len(set(positions))
                        or not 1 <= len(positions) <= 64
                        or entry['proposed_rank_source_counts'] != [len(positions[rank::4]) for rank in range(4)]):
                    raise ValueError('Every whole original global batch and actual rank assignment must remain exact')
                ranks = []
                context = target = 0
                for rank in range(4):
                    local = positions[rank::4]
                    actual = source.batch(name, local, device='cpu')
                    totals['actual_rank_calls'] += 1
                    if not local:
                        if actual is not None or name != 'words' or positions != [21055]:
                            raise ValueError('Only the three real empty singleton word-tail ranks may omit input delivery')
                        totals['actual_empty_ranks'] += 1
                        ranks.append(dict(rank=rank, positions=[], empty_without_fabricated_source=True))
                        continue
                    if actual['training_admitted'] or actual['source_role'] != 'train' or actual['purpose'] != 'audit':
                        raise ValueError('The full source readback must never infer source training admission')
                    if name == 'words':
                        expected = words.batch(local, 'cpu')
                        canonical = {k: expected[k] for k in ['input_ids', 'attention_mask', 'labels', 'pixel_values']}
                        if (actual['word']['texts'] != expected['texts']
                                or not torch.equal(actual['word']['word_pixels'], expected['word_pixels'])
                                or actual['word']['native_byte_targets'] != [list(text.encode('utf8')) for text in expected['texts']]):
                            raise ValueError('Every whole word crop/transcript/UTF8byte must match the original established encoder')
                        totals['whole_word_views'] += len(local)
                        totals['whole_word_UTF8_bytes'] += actual['word']['whole_UTF8_bytes']
                    else:
                        rows = [source.row(name, p)['current_row'] for p in local]
                        canonical = golden.batch(rows, 'response', 'cpu')
                    if set(actual['native']) != set(canonical) or any(
                            not torch.equal(actual['native'][key], canonical[key]) for key in canonical):
                        raise ValueError('Every actual complete native token/mask/label/pixel tensor must be bitexact to the established encoder')
                    if name == 'images:additional_photo_captions':
                        matching = actual['matching']
                        if matching is None or len(matching['captions']) != len(local):
                            raise ValueError('Every original photo must retain its paired caption-only matching inputs')
                        for i, text in enumerate(matching['captions']):
                            ids = [1] + source.tokenizer.encode(text, add_special_tokens=False).ids + [2]
                            if (matching['text_ids'][i][matching['text_mask'][i]].tolist() != ids
                                    or matching['pixel_keys'][i] != actual['source_observations'][i]['native_RGB448_sha256']):
                                raise ValueError('Every whole original positive image/caption association must reproduce')
                        totals['actual_whole_photo_matching_pairs'] += len(local)
                    elif actual['matching'] is not None:
                        raise ValueError('No non-photo source may fabricate matching positives')
                    for item in actual['source_observations']:
                        totals['maximum_native_context'] = max(totals['maximum_native_context'], item['complete_context_tokens'])
                        if item['correction_index'] >= 0:
                            key = (name, item['original_local_position'])
                            if key in corrected:
                                raise ValueError('No corrected source may be repeated or substituted for another original occurrence')
                            corrected.add(key)
                            original_row = source.row(*key)['original_row']
                            negative = source.negative(*key)
                            ids, labels = golden.sequence(original_row, 'response')
                            if (negative is None or negative['native']['input_ids'] != ids
                                    or negative['native']['labels'] != labels):
                                raise ValueError('Every full original corrective negative must remain byte-exact and separate from positive CE')
                    ranks.append(dict(rank=rank, positions=local,
                        complete_context_tokens=actual['complete_context_tokens'], response_EOS_targets=actual['response_EOS_targets'],
                        actual_native_shapes={k: list(v.shape) for k, v in actual['native'].items()},
                        actual_complete_native_tensor_sha256={k: tensor_sha(v) for k, v in actual['native'].items()},
                        actual_word_pixel_sha256=None if actual['word'] is None else tensor_sha(actual['word']['word_pixels']),
                        actual_whole_word_texts=None if actual['word'] is None else actual['word']['texts'],
                        actual_matching_ids_sha256=None if actual['matching'] is None else tensor_sha(actual['matching']['text_ids']),
                        every_actual_tensor_bitexact_to_established_encoder=True))
                    seen[name].update(local)
                    budgets[name]['original_views'] += len(local)
                    budgets[name]['complete_context_tokens'] += actual['complete_context_tokens']
                    budgets[name]['response_EOS_targets'] += actual['response_EOS_targets']
                    context += actual['complete_context_tokens']
                    target += actual['response_EOS_targets']
                    totals['actual_native_rank_batches'] += 1
                    totals['original_source_views'] += len(local)
                    del actual, canonical
                    if name == 'words':
                        del expected
                if context != entry['complete_context_tokens'] or target != entry['response_EOS_targets']:
                    raise ValueError('Every global batch must reconstruct its whole current context and response/EOS budget')
                row = dict(step=step, family=name, original_local_positions=positions,
                    complete_context_tokens=context, response_EOS_targets=target, rank_deliveries=ranks,
                    source_role='train', training_admitted=False)
                encoded = json.dumps(row, ensure_ascii=True) + '\n'
                if stream.tell() + len(encoded.encode()) > 1024**3:
                    raise ValueError('Complete rank-delivery observations must fit the original output reserve')
                stream.write(encoded)
                if step % 8 == 0:
                    stream.flush()
                    atomic(OUT / 'progress.json', dict(owner=owner, phase='all complete original source rank batches',
                        complete_global_steps=step, actual_totals=dict(totals), family_budgets=budgets,
                        bounded_disk_waits=dict(waits), models_constructed=0, training_admitted=False))
            os.fsync(stream.fileno())
        if (totals['original_source_views'] != 302627 or totals['whole_word_views'] != 91585
                or totals['whole_word_UTF8_bytes'] != 644063 or totals['actual_rank_calls'] != 18932
                or totals['actual_empty_ranks'] != 3 or len(corrected) != 4 or totals['maximum_native_context'] != 2952
                or any(seen[name] != Counter(range(len(source.positions[name]))) for name in source.names)):
            raise ValueError('Every original source, whole word byte, rank call, correction and true singleton tail must be covered exactly')
        for name, value in budgets.items():
            expected = source.proof['actual_family_budgets'][name]
            if any(expected[key] != count for key, count in value.items()):
                raise ValueError('Every full original corrected family budget must match its independent index readback')
        if (torch.cuda.is_initialized() or sum(v['response_EOS_targets'] for v in budgets.values()) != 4669243
                or sum(v['complete_context_tokens'] for v in budgets.values()) != 17681556
                or any(sha(p) != digest for p, digest in files.items())):
            raise ValueError('All current source bytes, full target budgets and CPU-only isolation must remain exact')
        atomic(OUT / 'complete.json', dict(complete=True, passed=True, at=datetime.now(timezone.utc).isoformat(), owner=owner,
            source_files_sha256=files, actual_complete_global_steps=4733, actual_full_source_views=302627,
            actual_totals=dict(totals), actual_family_budgets=budgets,
            all302627whole_original_current_views_once_and_all4rank_deliveries_rechecked=True,
            all_native_tensors_bitexact_to_established_encoders=True,
            all4complete_original_preference_negatives_preserved=True,
            all_actual_rank_observations_sha256=sha(observations), bounded_disk_waits=dict(waits),
            source_role='train', training_admitted=False, semantic_label_quality_not_certified=True,
            source_training_admission_and_full_size_GPU_capacity_recovery_and_learned_quality_pending=True,
            models_constructed=0, neural_forwards=0, new_production_optimizer_updates=0,
            existing_training_human_RL_and_all_nine_final_endpoints_preserved=True, goal_achieved=False))
        print(json.dumps(dict(report=str(OUT / 'complete.json'), all_source_views=302627, full_steps=4733, training_admitted=False)))
    except BaseException as error:
        atomic(OUT / 'failure.json', dict(owner=owner, error=repr(error), automatic_retry=False,
            source_role='train', training_admitted=False, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
