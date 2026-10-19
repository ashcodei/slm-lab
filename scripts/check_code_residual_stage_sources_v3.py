"""Check real29-family encoders and every prospective deterministic replay draw."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
from tokenizers import Tokenizer

from slm_lab.code_residual_stage_v3 import (EXPANDED_REPLAY_FAMILIES, ExpandedReplayEncoder,
    open_expanded_replay, replay_field, replay_objective)
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_fact_replay_v1 import CombinedReplay
from slm_lab.codenet_packed_replay_v1 import FAMILY as PACKED_FAMILY
from slm_lab.codenet_replay_schedule_v1 import coverage_replay_schedule
from slm_lab.codenet_stage_v1 import EpochOrder, exposure_budget
from slm_lab.codenet_token_pack_v1 import CompleteCodePack, packed_encoder
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.model import ModelConfig


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/code-residual-stage-v3-source-check'


def status(message, **fields):
    atomic_json(REPORTS / 'status.json', dict(status=message,
        updated_at=datetime.now(timezone.utc).isoformat(), goal_achieved=False, **fields))


def leaves(replay):
    if isinstance(replay, CombinedReplay):
        return [leaf for component in replay.components for leaf in leaves(component)]
    return [replay]


def owner(replay, name):
    while isinstance(replay, CombinedReplay):
        replay = replay.owners[name]
    return replay


def counts_and_contexts(component, name):
    if name == PACKED_FAMILY:
        context = component.index['train']['length'].astype(np.int64)
        return context - 1, context
    positions = component.positions[name] if hasattr(component, 'positions') else np.arange(len(component.index))
    items = component.index[positions]
    repetitions, remainder = divmod(component.size(name), len(items))
    if remainder or repetitions < 1:
        raise ValueError('Exact source repetitions/populations required')
    return np.tile(items['target_tokens'].astype(np.int64), repetitions), np.tile(items['context_tokens'].astype(np.int64), repetitions)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only real-source interface/schedule check required')
    if REPORTS.exists():
        raise ValueError('Retain earlier actor/source evidence; no implicit restart')
    if shutil.disk_usage(ROOT).free < 26 * 1024**3 + 64 * 1024**2:
        raise ValueError('Fund64MiB reports above unchanged26GiB reserve')
    files = [Path(__file__), ROOT / 'tests/test_code_residual_stage_v3.py',
        ROOT / 'reports/code-residual-stage-v3-focused-tests.log',
        ROOT / 'reports/code-residual-stage-v3/CPU-kernel-readback.json']
    files += [ROOT / 'src/slm_lab' / (name + '.py') for name in ['code_residual_stage_v3',
        'code_residual_stage_v2', 'code_residual_collective_checkpoint_v2', 'code_residual_checkpoint_v1',
        'code_residual_branch_v1', 'codenet_checkpoint_v1', 'codenet_runner_v1', 'codenet_stage_v1',
        'codenet_token_pack_v1', 'codenet_replay_v1', 'codenet_fact_replay_v1', 'codenet_dialogue_replay_v1',
        'codenet_image_replay_v1', 'codenet_later_instruction_replay_v2', 'codenet_packed_replay_v1',
        'codenet_replay_schedule_v1']]
    folders = {name: ROOT / 'data' / folder for name, folder in [
        ('joint', 'codenet-replay-v1'), ('facts', 'codenet-fact-replay-v1'),
        ('dialogue', 'codenet-dialogue-replay-v1'), ('images', 'codenet-image-replay-v1'),
        ('later', 'codenet-later-instruction-replay-v2'), ('packed', 'codenet-packed-pretraining-replay-v1')]}
    files += [folder / 'manifest.json' for folder in folders.values()]
    code_folder = ROOT / 'data/codenet-durable-pack-v2'
    files += [code_folder / 'manifest.json', code_folder / 'admission.json']
    bindings = {str(path): file_hash(path) for path in files}
    bindings.update({str(ROOT / path): digest for path, digest in capture()['files_sha256'].items()})
    initial = FileBindings(bindings)
    REPORTS.mkdir()
    status('opening every actual prepared source and all29 original objective interfaces')
    try:
        code = CompleteCodePack(code_folder, 'train', purpose='training')
        tokenizer_path = code.folder / 'tokenizer.json'
        replay = open_expanded_replay(folders, file_hash(tokenizer_path))
        if len(code) != 112609 or sum(replay.size(name) for name in replay.families) != 471300:
            raise ValueError('All prepared actual code and29-family replay populations required')
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        config = ModelConfig.load(code.folder / 'model-config.json')
        encoder = ExpandedReplayEncoder(tokenizer, config, owner(replay, 'human_instruction').source.parent, 448)
        code_encoder = packed_encoder(code, tokenizer, config)
        bindings.update({str(code.folder / name): digest for name, digest in code.manifest['files_sha256'].items()})
        for component in leaves(replay):
            bindings.update({str(component.folder / name): digest for name, digest in component.manifest['artifact_files_sha256'].items()})
            if hasattr(component, 'source'):
                bindings[str(component.source)] = component.manifest['original_training_source_sha256']
            for source in getattr(component, 'sources', {}).values():
                bindings[source['path']] = source['sha256']
                for path_key, hash_key in [('manifest', 'manifest_sha256'),
                        ('original_manifest', 'original_manifest_sha256'), ('original_file_path', 'original_file_sha256')]:
                    if path_key in source:
                        bindings[source[path_key]] = source[hash_key]
            if hasattr(component, 'stream'):
                bindings[str(component.stream.path)] = component.manifest['packed_manifest_sha256']
                for split in ('train', 'valid'):
                    for item in component.stream.manifest['splits'][split]['shards']:
                        bindings[str(component.stream.folder / item['path'])] = item['sha256']

        target_counts, family_checks = {}, []
        for name in replay.families:
            targets, contexts = counts_and_contexts(owner(replay, name), name)
            target_counts[name] = targets
            selected = sorted({0, len(targets) - 1, int(np.argmax(targets)), int(np.argmax(contexts))})
            checks = []
            for position in selected:
                row, field = replay.row(name, position), replay_field(name)
                ids, labels = encoder.sequence(row, field)
                batch = encoder.batch([row], field, 'cpu')
                if (batch['input_ids'][0].tolist() != ids or batch['labels'][0].tolist() != labels
                        or len(ids) != int(contexts[position])
                        or int((batch['labels'][:, 1:] != -100).sum()) != int(targets[position])
                        or hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).hexdigest() != row['_replay_input_sha256']):
                    raise ValueError('Actual original full input/causal-or-response mask differs')
                if row.get('image') and tuple(batch['pixel_values'].shape) != (1, 3, 448, 448):
                    raise ValueError('Original448px golden-verified image batch required')
                checks.append(dict(position=position, id=row.get('id'), context_tokens=len(ids),
                    target_tokens=int(targets[position]), input_sha256=row['_replay_input_sha256'],
                    response_field=field, image=bool(row.get('image'))))
            family_checks.append(dict(family=name, size=replay.size(name), objective=replay_objective(name), checks=checks))
            status('actual original29-family full encoder interfaces being checked', completed_families=len(family_checks))

        selected_code = sorted({0, int(np.argmax(code.index[code.positions]['length'])),
            int(np.argmax(code.index[code.positions]['length'] - code.index[code.positions]['target_start']))}
            | {group[0] for rank in range(4) for group in EpochOrder(len(code), 71).rank_groups(0, rank, 4, 1, 16)})
        code_checks = []
        for position in selected_code:
            ids, labels = code.sequence(position)
            batch = code_encoder.batch([{'packed_code_index': position}], 'response', 'cpu')
            if batch['input_ids'][0].tolist() != ids or batch['labels'][0].tolist() != labels:
                raise ValueError('Actual complete original code batch differs')
            code_checks.append(dict(position=position, context_tokens=len(ids),
                target_tokens=sum(label != -100 for label in labels[1:]),
                input_sha256=hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).hexdigest()))

        budget = exposure_budget(len(code), 3, 4, 1, 16)
        populations = tuple((name, replay.size(name)) for name in replay.families)
        rejected = []
        for accumulation in range(1, 65):
            try:
                schedule = coverage_replay_schedule(populations, budget['optimizer_steps'], 4 * accumulation)
            except ValueError as error:
                if 'cannot reach' not in str(error):
                    raise
                rejected.append(accumulation)
                continue
            break
        else:
            raise ValueError('No declared prospective layout covers all original source rows')
        width = 4 * accumulation
        offsets = np.cumsum([0] + [size for name, size in populations])
        exposed = np.zeros(int(offsets[-1]), dtype=np.uint32)
        draws = np.empty((budget['optimizer_steps'] * width, 2), dtype='<u4')
        orders = {name: EpochOrder(size, 71 + 1000 + i) for i, (name, size) in enumerate(populations)}
        for step in range(budget['optimizer_steps']):
            name, occurrence = schedule.at(step)
            number = replay.families.index(name)
            positions = [orders[name].at(occurrence * width + j) for j in range(width)]
            if occurrence == 0:
                rank_positions = [value for rank in range(4) for group in orders[name].rank_groups(
                    occurrence, rank, 4, 1, accumulation) for value in group]
                if sorted(rank_positions) != sorted(positions):
                    raise ValueError('Actual distributed rank-group sampler differs from prospective global draws')
            begin = step * width
            draws[begin:begin + width, 0], draws[begin:begin + width, 1] = number, positions
            np.add.at(exposed, offsets[number] + np.asarray(positions), 1)
        if np.any(exposed == 0) or int(exposed.sum()) != len(draws):
            raise ValueError('Every original prepared replay record must be reached')
        np.save(REPORTS / 'prospective-draw-positions-u32.npy', draws, allow_pickle=False)
        np.save(REPORTS / 'prospective-exposure-counts-u32.npy', exposed, allow_pickle=False)
        family_exposures = {}
        for i, (name, size) in enumerate(populations):
            counts = exposed[offsets[i]:offsets[i + 1]]
            family_exposures[name] = dict(source_records=size, actual_prospective_draws=int(counts.sum()),
                minimum_exposures=int(counts.min()), maximum_exposures=int(counts.max()),
                actual_target_exposures=int(np.dot(counts.astype(np.int64), target_counts[name])),
                objective=replay_objective(name), response_field=replay_field(name))
        if code_encoder.truncated or encoder.truncated:
            raise ValueError('No clipped original code/replay sequences permitted')
        initial.check(full=True)
        FileBindings(bindings).check(full=True)
        atomic_json(REPORTS / 'results.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=bindings, actual_source_families=list(EXPANDED_REPLAY_FAMILIES),
            prepared_code_train_rows=len(code), prepared_replay_logical_records=471300,
            selected_actual_code_batches=code_checks, selected_actual_replay_batches=family_checks,
            all29_actual_reader_encoder_interfaces_verified=True, explicit_causal_and_response_fields_verified=True,
            prospective_code_budget=budget, prospective_seed=71, minimum_replay_accumulation=accumulation,
            rejected_smaller_accumulations=rejected, prospective_global_replay_width=width,
            prospective_draws=len(draws), every_prepared_logical_record_reached=True,
            prospective_family_exposures=family_exposures,
            prospective_response_target_exposures=sum(value['actual_target_exposures'] for value in family_exposures.values()
                if value['response_field'] == 'response'),
            prospective_causal_target_exposures=family_exposures[PACKED_FAMILY]['actual_target_exposures'],
            artifact_files_sha256={name: file_hash(REPORTS / name) for name in [
                'prospective-draw-positions-u32.npy', 'prospective-exposure-counts-u32.npy']},
            original_running_curriculum_unchanged=True, original_source_bytes_copied=0,
            model_weights_allocated=False, optimizer_updates=0, no_GPU_context_created=True,
            full_source_training_started=False, final_base_match=False, all_later_queued_sources_covered=False,
            finalized_paired_photo_replay_ready=False, production_launcher_implemented=False,
            goal_achieved=False, candidate_promoted=False,
            scope='Every actual prepared reader opens its admitted sources; selected first/last/extreme code, '
                'instruction, image and packed batches use the real complete encoders. All deterministic '
                'prospective logical draws/target exposures are enumerated from fully audited immutable indices. '
                'Repeated instruction variants/windows are not certified distinct programs. This is not a '
                'full857M kernel/GPU run, final-base/baseline binding, finalized photo-pair integration or capability gain.'))
        status('all29 real interfaces and complete prospective replay coverage verified; production launch remains pending',
            prepared_logical_records=471300, minimum_replay_accumulation=accumulation,
            prospective_draws=len(draws), optimizer_updates=0)
        print(json.dumps(dict(families=29, logical_replay_records=471300,
            minimum_replay_accumulation=accumulation, prospective_draws=len(draws), optimizer_updates=0)), flush=True)
    except Exception as error:
        status('expanded source/schedule verification stopped; preserve evidence and inspect', error=str(error))
        raise


if __name__ == '__main__':
    main()
