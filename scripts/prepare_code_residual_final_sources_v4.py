"""After paired-source preparation, bind all actual inputs and full replay draws."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from tokenizers import Tokenizer

from slm_lab.code_residual_collective_checkpoint_v2 import fingerprint
from slm_lab.code_residual_source_contract_v4 import (REQUIRED_FAMILIES, FIELDS,
    open_final_source_replay)
from slm_lab.code_residual_stage_v3 import ExpandedReplayEncoder
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_replay_schedule_v1 import coverage_replay_schedule
from slm_lab.codenet_stage_v1 import EpochOrder, exposure_budget
from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.model import ModelConfig

from check_code_residual_stage_sources_v3 import leaves, owner, counts_and_contexts


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'data/codenet-final-source-contract-v4'
REPORTS = ROOT / 'reports/codenet-final-source-contract-v4'
PRECEDING_RESULT = ROOT / 'reports/codenet-paired-photo-replay-v1/results.json'


def status(message, **fields):
    atomic_json(REPORTS / 'status.json', dict(status=message,
        updated_at=datetime.now(timezone.utc).isoformat(), goal_achieved=False, **fields))


def live_preparer(pid, ticks):
    proc = Path('/proc') / str(pid)
    try:
        fields = (proc / 'stat').read_text().split(') ', 1)[1].split()
        argv = (proc / 'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return False
    return (fields[0] != 'Z' and fields[19] == ticks and argv[:3] == [str(ROOT / '.venv/bin/python').encode(),
        b'-u', str(ROOT / 'scripts/prepare_codenet_paired_photo_replay_v1.py').encode()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired-preparer-pid', type=int, required=True)
    parser.add_argument('--paired-preparer-start-ticks', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUTPUT.exists() or REPORTS.exists():
        raise ValueError('CPU-only new final source preparation required; preserve earlier artifacts')
    if not PRECEDING_RESULT.exists() and not live_preparer(args.paired_preparer_pid, args.paired_preparer_start_ticks):
        raise ValueError('Exact live paired preparer or its complete results required')
    files = [Path(__file__), ROOT / 'src/slm_lab/code_residual_source_contract_v4.py',
        ROOT / 'src/slm_lab/codenet_paired_photo_replay_v1.py',
        ROOT / 'scripts/check_code_residual_stage_sources_v3.py',
        ROOT / 'scripts/prepare_codenet_paired_photo_replay_v1.py',
        ROOT / 'tests/test_code_residual_source_contract_v4.py',
        ROOT / 'reports/code-residual-source-contract-v4-focused-tests.log',
        ROOT / 'reports/code-residual-source-contract-v4/CPU-kernel-readback.json']
    bindings = {str(path): file_hash(path) for path in files}
    bindings.update({str(ROOT / name): digest for name, digest in capture()['files_sha256'].items()})
    for relative in ['code_residual_stage_v2', 'code_residual_stage_v3', 'code_residual_branch_v1',
            'code_residual_checkpoint_v1', 'code_residual_collective_checkpoint_v2', 'codenet_checkpoint_v1',
            'codenet_token_pack_v1', 'codenet_replay_v1', 'codenet_fact_replay_v1', 'codenet_dialogue_replay_v1',
            'codenet_image_replay_v1', 'codenet_later_instruction_replay_v2', 'codenet_packed_replay_v1',
            'codenet_runner_v1', 'codenet_stage_v1', 'codenet_replay_schedule_v1']:
        path = ROOT / 'src/slm_lab' / (relative + '.py')
        bindings[str(path)] = file_hash(path)
    guard = FileBindings(bindings)
    REPORTS.mkdir()
    try:
        while not PRECEDING_RESULT.exists():
            if not live_preparer(args.paired_preparer_pid, args.paired_preparer_start_ticks):
                raise ValueError('Exact original paired preparer ended without completion; inspect before restarting')
            guard.check()
            status('waiting for exact live original paired-photo preparer; no incomplete source bundle created',
                paired_preparer_pid=args.paired_preparer_pid, paired_preparer_start_ticks=args.paired_preparer_start_ticks,
                original_preparer_verified_live=True, output_data_created=False)
            time.sleep(15)
        if shutil.disk_usage(ROOT).free < 26 * 1024**3 + 128 * 1024**2:
            raise ValueError('Fund64MiB artifacts/reports each above unchanged26GiB reserve')
        guard.check(full=True)
        preceding = json.loads(PRECEDING_RESULT.read_text())
        if not (preceding['all_original_source_admissions_reconstructed']
                and preceding['all_actual_replay_rows_inputs_masks_and_metadata_read_back']
                and preceding['original_validation_not_returned_for_training']):
            raise ValueError('Complete actual paired question/source/mask readback required')
        FileBindings(preceding['source_files_sha256']).check(full=True)
        code_folder = ROOT / 'data/codenet-durable-pack-v2'
        folders = {name: ROOT / 'data' / folder for name, folder in [
            ('joint', 'codenet-replay-v1'), ('facts', 'codenet-fact-replay-v1'),
            ('dialogue', 'codenet-dialogue-replay-v1'), ('images', 'codenet-image-replay-v1'),
            ('later', 'codenet-later-instruction-replay-v2'), ('packed', 'codenet-packed-pretraining-replay-v1'),
            ('paired', 'codenet-paired-photo-replay-v1')]}
        if file_hash(folders['paired'] / 'manifest.json') != preceding['manifest_sha256']:
            raise ValueError('Exact actual completed paired replay required')
        status('opening all actual admitted sources and binding the complete30-family objective contract')
        code = CompleteCodePack(code_folder, 'train', purpose='training')
        token = code_folder / 'tokenizer.json'
        replay, contract = open_final_source_replay(folders, file_hash(token))
        if tuple(replay.families) != REQUIRED_FAMILIES or len(code) != 112609:
            raise ValueError('All actual required source families and complete code programs required')
        encoder = ExpandedReplayEncoder(Tokenizer.from_file(str(token)), ModelConfig.load(code_folder / 'model-config.json'),
            owner(replay, 'human_instruction').source.parent, 448)
        for path in [PRECEDING_RESULT, code_folder / 'manifest.json', code_folder / 'admission.json']:
            bindings[str(path)] = file_hash(path)
        bindings.update({str(code_folder / name): digest for name, digest in code.manifest['files_sha256'].items()})
        family_checks, target_counts = [], {}
        for component in leaves(replay):
            bindings[str(component.folder / 'manifest.json')] = file_hash(component.folder / 'manifest.json')
            bindings.update({str(component.folder / name): digest for name, digest in component.manifest['artifact_files_sha256'].items()})
            if hasattr(component, 'source') and 'original_training_source_sha256' in component.manifest:
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
                    for shard in component.stream.manifest['splits'][split]['shards']:
                        bindings[str(component.stream.folder / shard['path'])] = shard['sha256']
        bindings.update(preceding['source_files_sha256'])
        for item in contract['families']:
            name, field = item['name'], FIELDS[item['objective']]
            targets, contexts = counts_and_contexts(owner(replay, name), name)
            target_counts[name] = targets
            checked = []
            for position in sorted({0, len(targets) - 1, int(np.argmax(targets)), int(np.argmax(contexts))}):
                row = replay.row(name, position)
                batch = encoder.batch([row], field, 'cpu')
                if (batch['input_ids'].shape[1] != int(contexts[position])
                        or int((batch['labels'][:, 1:] != -100).sum()) != int(targets[position])):
                    raise ValueError('Actual full original input/causal-or-response mask differs')
                checked.append(dict(position=position, context_tokens=int(contexts[position]),
                    target_tokens=int(targets[position]), input_sha256=row['_replay_input_sha256'],
                    actual448px_golden_pixels_checked=bool(row.get('image'))))
            family_checks.append(dict(name=name, objective=item['objective'], source_records=len(targets), checks=checked))
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
            raise ValueError('Complete original replay needs a larger explicitly funded layout')
        width = accumulation * 4
        offsets = np.cumsum([0] + [size for name, size in populations])
        exposed = np.zeros(int(offsets[-1]), dtype='<u4')
        draws = np.empty((budget['optimizer_steps'] * width, 2), dtype='<u4')
        orders = {name: EpochOrder(size, 71 + 1000 + i) for i, (name, size) in enumerate(populations)}
        for step in range(budget['optimizer_steps']):
            name, occurrence = schedule.at(step)
            number = replay.families.index(name)
            positions = [orders[name].at(occurrence * width + j) for j in range(width)]
            draws[step * width:(step + 1) * width, 0] = number
            draws[step * width:(step + 1) * width, 1] = positions
            np.add.at(exposed, offsets[number] + np.asarray(positions), 1)
        if np.any(exposed == 0) or encoder.truncated:
            raise ValueError('Every actual admitted full source must be covered without clipping')
        OUTPUT.mkdir()
        atomic_json(OUTPUT / 'replay-contract.json', contract)
        np.save(OUTPUT / 'prospective-draw-positions-u32.npy', draws, allow_pickle=False)
        np.save(OUTPUT / 'prospective-exposure-counts-u32.npy', exposed, allow_pickle=False)
        family_exposures = {}
        for i, (name, size) in enumerate(populations):
            counts = exposed[offsets[i]:offsets[i + 1]]
            family_exposures[name] = dict(source_records=size, draws=int(counts.sum()), minimum_exposures=int(counts.min()),
                maximum_exposures=int(counts.max()), target_exposures=int(np.dot(counts.astype(np.int64), target_counts[name])))
        guard.check(full=True)
        FileBindings(bindings).check(full=True)
        if sum(path.stat().st_size for path in OUTPUT.iterdir()) > 64 * 1024**2:
            raise ValueError('Bounded source-contract/draw artifacts exceed committed storage')
        atomic_json(OUTPUT / 'manifest.json', dict(format='own-complete-code-final-source-bundle-v4',
            created_at=datetime.now(timezone.utc).isoformat(), tokenizer_sha256=file_hash(token),
            code_manifest_sha256=file_hash(code_folder / 'manifest.json'), code_admission_sha256=file_hash(code_folder / 'admission.json'),
            replay_folders={name: str(folder) for name, folder in folders.items()},
            replay_contract_fingerprint=fingerprint(contract), source_files_sha256=bindings,
            artifact_files_sha256={path.name: file_hash(path) for path in OUTPUT.iterdir()},
            all30_actual_prepared_families_opened=True, full_original_paired_admission_completed=True,
            prospective_code_budget=budget, prospective_seed=71, minimum_replay_accumulation=accumulation,
            rejected_smaller_accumulations=rejected, prospective_global_replay_width=width,
            prospective_draws=len(draws), every_prepared_logical_record_reached=True,
            prospective_family_exposures=family_exposures, selected_actual_full_encoder_checks=family_checks,
            final_checkpoint_replay_match_verified=False, all_later_queued_sources_covered=False,
            exact_final_base_and_capability_baselines_bound=False, full857M_kernel_GPU_proof=False,
            production_training_ready=False, actual_neural_updates=0, goal_achieved=False))
        atomic_json(REPORTS / 'results.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
            manifest_sha256=file_hash(OUTPUT / 'manifest.json'), replay_contract_fingerprint=fingerprint(contract),
            all30_actual_prepared_source_interfaces_verified=True, minimum_replay_accumulation=accumulation,
            prospective_draws=len(draws), final_base_match=False, production_training_ready=False, goal_achieved=False))
        status('all actual30-family sources and prospective coverage bound; final-base/baseline/production work remains',
            source_records=int(offsets[-1]), prospective_draws=len(draws), minimum_replay_accumulation=accumulation)
    except Exception as error:
        status('final original source-bundle preparation stopped; retain evidence and inspect', error=str(error))
        raise


if __name__ == '__main__':
    main()
