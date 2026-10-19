"""Exercise the actual complete-code and all19 replay reader/encoder interfaces."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
from tokenizers import Tokenizer

from slm_lab.code_residual_stage_v2 import FULL_REPLAY_FAMILIES, open_full_replay
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_replay_schedule_v1 import coverage_replay_schedule
from slm_lab.codenet_runner_v1 import VerifiedReplayEncoder
from slm_lab.codenet_stage_v1 import EpochOrder, exposure_budget
from slm_lab.codenet_token_pack_v1 import CompleteCodePack, packed_encoder
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/code-residual-stage-v2-source-check'


def status(message, **fields):
    atomic_json(REPORTS / 'status.json', dict(status=message,
        updated_at=datetime.now(timezone.utc).isoformat(), goal_achieved=False, **fields))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only source/interface verification required')
    if REPORTS.exists():
        raise ValueError('Preserve earlier source verification; inspect exact existing actor')
    if shutil.disk_usage(ROOT).free < 16 * 1024**3 + 32 * 1024**2:
        raise ValueError('Retain16GiB persistent space plus32MiB report commitment')
    REPORTS.mkdir()
    status('opening actual complete original code and all nineteen prepared replay families')
    files = [Path(__file__), ROOT / 'src/slm_lab/code_residual_stage_v2.py',
        ROOT / 'src/slm_lab/code_residual_collective_checkpoint_v2.py',
        ROOT / 'src/slm_lab/codenet_runner_v1.py', ROOT / 'src/slm_lab/codenet_stage_v1.py',
        ROOT / 'src/slm_lab/codenet_token_pack_v1.py', ROOT / 'src/slm_lab/codenet_replay_v1.py',
        ROOT / 'src/slm_lab/codenet_fact_replay_v1.py', ROOT / 'src/slm_lab/codenet_dialogue_replay_v1.py',
        ROOT / 'src/slm_lab/codenet_image_replay_v1.py', ROOT / 'src/slm_lab/codenet_replay_schedule_v1.py',
        ROOT / 'src/slm_lab/training.py', ROOT / 'src/slm_lab/model.py',
        ROOT / 'tests/test_code_residual_stage_v2.py', ROOT / 'reports/code-residual-stage-v2-focused-tests.log']
    folders = {name: ROOT / 'data' / folder for name, folder in [
        ('joint', 'codenet-replay-v1'), ('facts', 'codenet-fact-replay-v1'),
        ('dialogue', 'codenet-dialogue-replay-v1'), ('images', 'codenet-image-replay-v1')]}
    files += [folder / 'manifest.json' for folder in folders.values()]
    code_folder = ROOT / 'data/codenet-durable-pack-v2'
    files += [code_folder / 'manifest.json', code_folder / 'admission.json']
    initial = FileBindings({str(path): file_hash(path) for path in files})
    code = CompleteCodePack(code_folder, 'train', purpose='training')
    assert len(code) == 112609
    token_path = code.folder / 'tokenizer.json'
    token_sha = file_hash(token_path)
    replay = open_full_replay(folders, token_sha)
    assert tuple(replay.families) == FULL_REPLAY_FAMILIES
    assert sum(replay.size(name) for name in replay.families) == 348133
    tokenizer = Tokenizer.from_file(str(token_path))
    config = ModelConfig.load(code.folder / 'model-config.json')
    code_encoder = packed_encoder(code, tokenizer, config)
    encoder = VerifiedReplayEncoder(tokenizer, config, replay.components[0].source.parent, 448)
    steps = exposure_budget(len(code), 3, 4, 1, 16)['optimizer_steps']
    assert steps == 5279
    schedule = coverage_replay_schedule(tuple((name, replay.size(name)) for name in replay.families), steps, 68)
    family_checks = []
    bindings = {str(path): file_hash(path) for path in files}
    bindings.update({str(code.folder / name): digest for name, digest in code.manifest['files_sha256'].items()})
    for component in replay.components:
        bindings.update({str(component.folder / name): digest for name, digest in component.manifest['artifact_files_sha256'].items()})
        if hasattr(component, 'source'):
            bindings[str(component.source)] = component.manifest['original_training_source_sha256']
    for name in replay.families:
        owner = replay.owners[name]
        positions = owner.positions[name] if hasattr(owner, 'positions') else np.arange(len(owner.index))
        items = owner.index[positions]
        selected = sorted({0, int(np.argmax(items['context_tokens'])), int(np.argmax(items['target_tokens']))})
        checks = []
        for position in selected:
            row = replay.row(name, position)
            ids, labels = encoder.sequence(row, 'response')
            batch = encoder.batch([row], 'response', 'cpu')
            assert batch['input_ids'][0].tolist() == ids and batch['labels'][0].tolist() == labels
            assert len(ids) == row['_replay_context_tokens']
            assert sum(label != -100 for label in labels) == row['_replay_target_tokens']
            if row.get('image'):
                pixels = batch['pixel_values'][0]
                assert tuple(pixels.shape) == (3, 448, 448)
                rgb = np.rint((pixels.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
                assert hashlib.sha256(rgb.tobytes()).hexdigest() == row['model_pixel_sha256_448']
            checks.append(dict(id=row['id'], local_position=position, context_tokens=len(ids),
                targets=row['_replay_target_tokens'], image=bool(row.get('image')),
                complete_input_sha256=row['_replay_input_sha256'], original_aliases_preserved=row.get('reference_answers')))
        family_checks.append(dict(family=name, complete_fit_population=replay.size(name), checks=checks))
        status('actual original replay reader and full encoder interfaces being verified', completed_families=len(family_checks), families=19)
    selected_code = sorted({0, int(np.argmax(code.index[code.positions]['length']))}
        | {int(np.argmax(code.index[code.positions]['length'] - code.index[code.positions]['target_start']))}
        | {group[0] for rank in range(4) for group in EpochOrder(len(code), 71).rank_groups(0, rank, 4, 1, 16)})
    code_checks = []
    for position in selected_code:
        ids, labels = code.sequence(position)
        batch = code_encoder.batch([dict(packed_code_index=position)], 'response', 'cpu')
        assert batch['input_ids'][0].tolist() == ids and batch['labels'][0].tolist() == labels
        code_checks.append(dict(local_position=position, global_position=int(code.positions[position]),
            context_tokens=len(ids), target_tokens=sum(v != -100 for v in labels),
            complete_input_sha256=hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).hexdigest()))
    assert code_encoder.truncated == encoder.truncated == 0
    initial.check(full=True)
    FileBindings(bindings).check(full=True)
    atomic_json(REPORTS / 'results.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=bindings, all19_actual_reader_encoder_interfaces_verified=True,
        prepared_complete_fit_replay_rows=348133, admitted_complete_code_train_rows=112609,
        selected_actual_code_batches=code_checks, selected_actual_replay_batches=family_checks,
        encoder_truncated_sequences=0, production_config_loaded_without_allocating_model_weights=True,
        prospective_optimizer_steps=steps, prospective_global_replay_width=68,
        prospective_replay_minimum_required_updates=schedule.minimum_required_updates,
        prospective_family_update_quotas=dict(zip(replay.families, schedule.quotas)),
        no_GPU_context_created=True, optimizer_updates=0, full_source_training_started=False,
        final_checkpoint_match=False, all_later_queued_sources_covered=False, production_stage_ready=False,
        goal_achieved=False, candidate_promoted=False,
        scope='Actual full audited readers open all prepared data; selected first/extreme whole examples from every '
            'family and the first64 code draws use the real full-mask448px batch encoders. '
            'This does not rerun every earlier pixel audit, train a model, establish final-base/all-later coverage '
            'or replace required capability/evaluation/resource/lifecycle gates.'))
    status('actual112609-code and all19 prepared replay reader/encoder interfaces verified; production training remains gated')
    print(json.dumps(dict(code_train_rows=len(code), replay_rows=348133, families=19,
        selected_code_batches=len(code_checks), optimizer_updates=0)), flush=True)


if __name__ == '__main__':
    main()
