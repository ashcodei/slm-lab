"""Freeze matched exposure/documentation probes before any model predictions."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import heapq
import json
from pathlib import Path
import random
import shutil

import numpy as np
from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig
from slm_lab.training import Encoder
from prepare_codenet_native_contract_pack_v2 import occupied_writer_reservation

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ['Python', 'JavaScript', 'Java', 'Go', 'PHP', 'Ruby', 'general_human_instructions']


def main():
    folder = ROOT / 'reports/instruction-fidelity-probe-v1'
    if folder.exists():
        raise ValueError('Preserve all earlier selections and partial outputs')
    occupied=occupied_writer_reservation()
    reserve=26*1024**3-occupied['credited_bytes']
    if shutil.disk_usage(ROOT).free<reserve+64*1024**2:
        raise ValueError('Preserve remaining writer commitments plus64MiB probe budget')
    source = ROOT / 'data/extended-curriculum-v1/code-instruction.train.jsonl'
    source_audit = ROOT / 'reports/code-instruction-input-audit-v1.json'
    audit = json.loads(source_audit.read_text())
    n, exposures, seed = audit['totals']['rows'], 4000*32, 42
    if n != 906952 or audit['data_sha256'] != file_hash(source):
        raise ValueError('Exact independently audited active SFT population required')
    order = list(range(n)); random.Random(seed).shuffle(order)
    draw_rank = np.empty(n, dtype=np.int64); draw_rank[order] = np.arange(n)
    paths = [Path(__file__), source, source_audit, ROOT / 'src/slm_lab/training.py',
        ROOT / 'src/slm_lab/model.py', ROOT / 'src/slm_lab/codenet_checkpoint_v1.py',
        ROOT / 'configs/model.json', ROOT / 'checkpoints/code-instruction-4000-assessment-v1/tokenizer.json']
    bound = {str(p): file_hash(p) for p in paths}; guard = FileBindings(bound)
    cells, counts, images = defaultdict(list), Counter(), 0
    digest = hashlib.sha256(); byte_offset = 0
    with source.open('rb') as f:
        for index, raw in enumerate(f):
            digest.update(raw); row = json.loads(raw)
            language = row.get('language', 'general_human_instructions')
            if language not in LANGUAGES:
                raise ValueError('Unknown source stratum; do not silently drop it')
            rank = int(draw_rank[index]); phase = 'exposed' if rank < exposures else 'not_yet_exposed'
            counts[language, phase] += 1; images += bool(row.get('image'))
            quota = 16 if language == 'general_human_instructions' else 8
            heap = cells[language, phase]
            candidate = (-rank, index, byte_offset, len(raw), hashlib.sha256(raw).hexdigest(), row)
            if len(heap) < quota:
                heapq.heappush(heap, candidate)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, candidate)
            byte_offset += len(raw)
    if index+1 != n or digest.hexdigest() != audit['data_sha256'] or images:
        raise ValueError('Exact text-only population required for original random-shuffle replay')
    encoder = Encoder(Tokenizer.from_file(str(paths[-1])), ModelConfig.load(ROOT / 'configs/model.json'), ROOT)
    selected = []
    for language in LANGUAGES:
        for phase in ['exposed', 'not_yet_exposed']:
            entries = sorted(cells[language, phase], key=lambda x: -x[0])
            expected = 16 if language == 'general_human_instructions' else 8
            if len(entries) != expected:
                raise ValueError('Every predeclared source/exposure stratum required')
            groups = [r[-1].get('group_id', r[-1]['id']) for r in entries]
            shifts = [s for s in range(1,len(entries)) if all(groups[i] != groups[(i+s)%len(entries)] for i in range(len(entries)))]
            if not shifts:
                raise ValueError('Distinct source-group control assignment unavailable; preserve selection')
            shift = shifts[0]
            for i, (negative_rank,index,offset,length,raw_sha,row) in enumerate(entries):
                ids, labels = encoder.sequence(row, 'response')
                start = next(k for k,v in enumerate(labels) if v != -100)
                selected.append(dict(probe_id=f'{language}:{phase}:{i}', language=language, exposure_phase=phase,
                    original_row_index=index, original_row_byte_offset=offset, original_row_bytes=length,
                    original_raw_row_sha256=raw_sha, actual_stage_draw_rank=-negative_rank,
                    first_4000_stage_updates_exposure_count=int(-negative_rank < exposures),
                    original_row=row, actual_training_ids=ids, actual_training_target_start=start,
                    actual_training_ids_sha256=hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest(),
                    actual_training_targets=len(ids)-start,
                    original_response_tokens=len(encoder.tokens(row['response']))+1,
                    shuffled_prompt_control_id=f'{language}:{phase}:{(i+shift)%len(entries)}',
                    generate_free_response=i < expected//2, audit_padding_and_cache=i == 0))
    if len(selected)!=128 or sum(r['generate_free_response'] for r in selected)!=64:
        raise ValueError('Full128 teacher-forcing and64 free-response probes required')
    guard.check(full=True); folder.mkdir()
    path=folder/'selection.jsonl'
    with path.open('x') as f:
        for r in selected:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    atomic_json(folder/'plan.json',dict(fixed_at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=bound, selection_sha256=file_hash(path),
        source_rows=n, first_4000_stage_example_exposures=exposures, source_seed=seed,
        exact_order_rule='The original text-only train() seeds Python random with42 and shuffles once; absolute draws wrap only after906952 rows.',
        teacher_forced_rows=128, free_generation_rows=64, free_generation_budget=128,
        padding_cache_rows_per_checkpoint=14,
        population_counts={language:{phase:counts[language,phase] for phase in ['exposed','not_yet_exposed']} for language in LANGUAGES},
        selected_contexts_clipped_by_original_reader=encoder.truncated,
        storage=dict(occupied_verified_writer=occupied,remaining_writer_reserve_bytes=reserve,maximum_report_bytes=64*1024**2),
        checkpoints=[dict(label='completed_language_before_SFT',path=str(ROOT/'checkpoints/extended-curriculum-v1/language-pretraining/model.pt'),
            sha256='4c05d99b0c3fc775bf89e70c6cbac9c63017eb36f67e17c738b5be268fb191ca'),
            dict(label='own_code_instruction_4000',path=str(ROOT/'checkpoints/code-instruction-4000-assessment-v1/model.pt'),
            sha256='ceb71bbb459950667400632e4a9033d718df85ae7941f8f6398f4bcc8c8306a3')],
        control='Same gold response under a different source-group prompt in the same language/exposure stratum; at most128 gold targets, whole prompt retained.',
        interpretation='Training-interface diagnostic. Not-yet-exposed means only absent from these first4000 SFT updates, not unseen in pretraining or an independent capability benchmark.',
        optimizer_updates=0, candidate_promoted=False, goal_achieved=False))
    print('frozen_teacher_rows',len(selected),'free_generation_rows',sum(r['generate_free_response'] for r in selected),
          'reader_truncated_rows',encoder.truncated,flush=True)


if __name__=='__main__':main()
