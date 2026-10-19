"""Audit native-CE microbatch coverage of the entire original visual epoch.

No pixels or neural parameters are loaded. This metadata audit preserves all
nine source families, but only seven families use the new CE-only kernel;
photo matching and whole-word CTC retain their separate outstanding gates.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource

import numpy as np

from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
from slm_lab.global_native_visual_stage_kernel_v1 import native_microbatches
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.multimodal_source_index_format_v1 import DTYPE, FORMAT

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'reports/full-multimodal-corrected-source-index-CPU-v1'
READER = ROOT/'reports/full-multimodal-corrected-source-index-v1-readback/complete.json'
OUT = ROOT/'reports/full-visual-native-microbatch-audit-CPU-v1'
TESTS = ROOT/'reports/global-native-visual-stage-kernel-v1-tests.log'
RESERVE = 40*1024**3
SPECIAL = {'words': 'whole-word-CTC-and-native-CE', 'images:additional_photo_captions': 'global-paired-matching-and-native-CE'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, value):
    pending = path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value, indent=2)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
            or '6 passed' not in TESTS.read_text()):
        raise ValueError('Fresh tested CUDA-hidden one-thread nice19 metadata-only audit required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    wait_for_reserve(ROOT, minimum_bytes=RESERVE)
    resource.setrlimit(resource.RLIMIT_AS, (24*1024**3, 24*1024**3))
    proof, reader = read(SOURCE/'complete.json'), read(READER)
    index = SOURCE/'complete-native-source-index.npy'
    schedule_path = SOURCE/'complete-corrected-single-epoch-schedule.json'
    if (not proof['complete'] or not reader['complete'] or proof['format'] != FORMAT
            or proof['training_admitted'] or reader['training_admitted']
            or proof['all_original_train_views'] != 302627 or reader['actual_all_original_entries'] != 302627
            or proof['index_sha256'] != sha(index) or proof['schedule_sha256'] != sha(schedule_path)):
        raise ValueError('Complete corrected index/native source readback and exact whole schedule required')
    files = {str(p):sha(p) for p in [SOURCE/'complete.json', READER, index, schedule_path, TESTS,
        *python_sources(ROOT, [Path(__file__), ROOT/'tests/test_global_native_visual_stage_kernel_v1.py'])]}
    states = {p:(Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in files}

    def guard():
        if os.sched_getscheduler(0) != os.SCHED_IDLE or any(
                (Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) != value for p,value in states.items()):
            raise ValueError('All fixed source/runtime evidence and idle scheduling must remain exact')

    def checked():
        wait_for_reserve(ROOT, minimum_bytes=RESERVE, guard=guard)

    me = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    OUT.mkdir()
    atomic(OUT/'plan.json', dict(owner=me, source_files_sha256=files, exact_source_occurrences=302627,
        exact_global_batches=4733, world=4, maximum_microbatch_examples=4, maximum_padded_decoder_tokens=4096,
        native_image_size=448, native_visual_prefix_tokens=196, every_rank_and_partial_batch_preserved=True,
        photo_matching_and_word_CTC_additional_kernels_not_certified_by_native_plan=True,
        all_image_pixels_and_caption_tokens_not_redecoded_by_this_metadata_audit=True,
        neural_forwards=0, training_admitted=False, new_production_optimizer_updates=0, goal_achieved=False))
    try:
        values = np.load(index, mmap_mode='r', allow_pickle=False)
        if values.dtype != DTYPE or values.shape != (302627,):
            raise ValueError('Exact complete compact source index schema required')
        names = proof['family_names']
        families = {name:values[values['family'] == number] for number,name in enumerate(names)}
        seen = {name:set() for name in names}
        counts = Counter()
        family_counts = {name:Counter() for name in names}
        longest = dict(complete_context_tokens=0)
        tails = []
        schedule = read(schedule_path)
        if len(schedule['entries']) != 4733 or schedule['global_width'] != 64 or schedule['world_size'] != 4:
            raise ValueError('All original4733 global source steps required')
        with (OUT/'all4733-complete-native-rank-microbatch-plans.jsonl').open('x') as stream:
            for step, entry in enumerate(schedule['entries'], 1):
                if step % 128 == 1:
                    checked()
                name, positions = entry['family'], entry['original_local_positions']
                if entry['step'] != step or not 1 <= len(positions) <= 64 or len(set(positions)) != len(positions):
                    raise ValueError('Complete ordered global steps and original occurrence identities required')
                array = families[name]
                if any(type(p) is not int or not 0 <= p < len(array) or p in seen[name] for p in positions):
                    raise ValueError('No source omission, duplication or out-of-scope position is allowed')
                lengths = [int(array[p]['context_tokens']) for p in positions]
                targets = [int(array[p]['response_EOS_targets']) for p in positions]
                if sum(lengths) != entry['complete_context_tokens'] or sum(targets) != entry['response_EOS_targets']:
                    raise ValueError('Every original context and full response/EOS count must reconstruct')
                ranks = []
                for rank in range(4):
                    local = positions[rank::4]
                    local_lengths = lengths[rank::4]
                    chunks = native_microbatches(local_lengths, prefix_tokens=196)
                    # Independent exhaustive prefix search checks the greedy
                    # grouping without calling the kernel's planner again.
                    remaining = list(range(len(local)))
                    independent = []
                    while remaining:
                        width = max(n for n in range(1, min(4,len(remaining))+1)
                            if n * (196 + max(local_lengths[i] for i in remaining[:n])) <= 4096)
                        independent.append(remaining[:width])
                        remaining = remaining[width:]
                    if chunks != independent or [i for chunk in chunks for i in chunk] != list(range(len(local))):
                        raise ValueError('Every complete original source must occur in exactly one legal microbatch')
                    observations = []
                    for chunk in chunks:
                        contexts = [local_lengths[i] for i in chunk]
                        sources = [local[i] for i in chunk]
                        token_count = len(chunk)*(max(contexts)+196)
                        observations.append(dict(original_positions=sources, complete_context_lengths=contexts,
                            response_EOS_targets=sum(int(array[p]['response_EOS_targets']) for p in sources),
                            padded_decoder_tokens=token_count))
                        counts['native_microbatches'] += 1
                        counts['padded_decoder_tokens'] += token_count
                        counts['maximum_microbatch_padded_tokens'] = max(counts['maximum_microbatch_padded_tokens'], token_count)
                        if max(contexts) > longest['complete_context_tokens']:
                            longest = dict(family=name, step=step, rank=rank, complete_context_tokens=max(contexts),
                                original_positions=sources, microbatch_examples=len(sources), padded_decoder_tokens=token_count)
                    ranks.append(dict(rank=rank, original_positions=local, complete_context_tokens=sum(local_lengths), microbatches=observations))
                    if not local:
                        counts['empty_rank_slots'] += 1
                if [len(positions[i::4]) for i in range(4)] != entry['proposed_rank_source_counts']:
                    raise ValueError('Every originally declared rank source count must remain unchanged')
                if len(positions) < 4:
                    tails.append(dict(step=step,family=name,original_positions=positions,rank_counts=[len(v['original_positions']) for v in ranks]))
                if name == 'images:additional_photo_captions' and entry['global_matching_has_observed_negatives'] is not True:
                    raise ValueError('Every original photo matching batch must retain its negatives')
                seen[name].update(positions)
                budget = dict(original_views=len(positions), complete_context_tokens=sum(lengths), response_EOS_targets=sum(targets), global_batches=1)
                family_counts[name].update(budget)
                for key in ['original_views','complete_context_tokens','response_EOS_targets','global_batches']:
                    counts[key] += budget[key]
                encoded = json.dumps(dict(step=step, family=name, intended_kernel=SPECIAL.get(name,'native-CE-only'),
                    ranks=ranks, all_original_occurrences_preserved=True, training_admitted=False))+'\n'
                if stream.tell()+len(encoded.encode()) > 64*1024**2:
                    raise ValueError('The declared full metadata output allowance was exceeded')
                stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        for name in names:
            wanted = proof['actual_family_budgets'][name]
            if (seen[name] != set(range(len(families[name]))) or any(family_counts[name][key] != wanted[key]
                    for key in ['original_views','complete_context_tokens','response_EOS_targets'])):
                raise ValueError('Every full original family budget and occurrence must remain intact')
        if (counts['original_views'] != 302627 or counts['complete_context_tokens'] != 17681556
                or counts['response_EOS_targets'] != 4669243 or counts['global_batches'] != 4733
                or tails != [dict(step=3205, family='words', original_positions=[21055], rank_counts=[1,0,0,0])]
                or longest['complete_context_tokens'] != 2952):
            raise ValueError('Exact full corrected source epoch, true EOS budget, longest context and singleton word tail required')
        checked()
        if any(sha(p) != digest for p,digest in files.items()):
            raise ValueError('Every fixed metadata/runtime byte must remain unchanged')
        generic = Counter()
        for name,budget in family_counts.items():
            if name not in SPECIAL:
                generic.update(budget)
        atomic(OUT/'complete.json', dict(complete=True,passed=True,at=datetime.now(timezone.utc).isoformat(),owner=me,
            source_files_sha256=files,actual_counts=dict(counts),actual_family_counts={k:dict(v) for k,v in family_counts.items()},
            actual_seven_native_CE_only_family_counts=dict(generic),actual_longest_complete_native_context=longest,
            actual_partial_tail_ranks=tails,all_original302627_source_occurrences_once=True,
            all4733_steps_and_full4669243_EOS_target_budget_preserved=True,
            full_rank_microbatch_plans_sha256=sha(OUT/'all4733-complete-native-rank-microbatch-plans.jsonl'),
            physical_pixel_and_distributed_full_size_CUDA_capacity_recovery_training_quality_still_pending=True,
            photo_matching_and_word_CTC_additional_microbatch_capacity_still_pending=True,
            metadata_token_budget_is_not_a_GPU_memory_or_throughput_guarantee=True,
            all_current_neural_workers_and_original_human_RL_and_nine_endpoints_unchanged=True,
            neural_forwards=0,training_admitted=False,new_production_optimizer_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=True,passed=True,counts=dict(counts),native_CE_only=dict(generic),longest=longest)),flush=True)
    except BaseException as error:
        atomic(OUT/'failure.json',dict(owner=me,error=repr(error),automatic_retry=False,training_admitted=False));raise


if __name__ == '__main__':
    main()
