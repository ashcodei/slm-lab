"""Revise six source-declared runtime prompts without touching the live v1 pack."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from tokenizers import Tokenizer
import torch

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_complete_context_v1 import CompleteCodeEncoder
from slm_lab.codenet_native_contracts_v1 import native_inventory, with_native_contract
from slm_lab.codenet_token_pack_v1 import CodePackWriter, CompleteCodePack, ROLES
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig

ROOT = Path(__file__).resolve().parents[1]


def occupied_writer_reservation():
    """Credit only verified bytes already occupying the original writer budget."""
    process = Path('/proc/1598')
    stat = (process / 'stat').read_text()
    if (stat[stat.rfind(')') + 2:].split()[19] != '18631959'
            or (process / 'cmdline').read_bytes().split(b'\0')[:4]
            != [b'.venv/bin/python', b'-u', b'scripts/train_extended_curriculum.py', b'--resume-frozen-plan']):
        raise ValueError('Verified original live writer identity required for reservation credit')
    status = json.loads((ROOT / 'reports/extended-curriculum-v1/status.json').read_text())
    if status['stage'] != 'multilingual-code-instruction':
        raise ValueError('Recalculate writer commitments for a changed active stage')
    directory = ROOT / 'checkpoints/extended-curriculum-v1/multilingual-code-instruction'
    snapshots = sorted(directory.glob('step-*.pt'))
    if not snapshots:
        raise ValueError('No occupied active-stage snapshot to credit')
    p = snapshots[-1]
    state = torch.load(p, map_location='cpu', weights_only=True, mmap=True)
    if (state.get('stage') != 'sft' or state.get('world_size') != 4
            or state.get('planned_steps') != 28343 or not state.get('optimizer')
            or state.get('data_sha256') != 'e0101ece31a2cbe7b47e41d19741e34602e12e4a5565baea84a33a45f454ca24'):
        raise ValueError('Only the exact occupied active-writer snapshot can be credited')
    size = p.stat().st_size
    return {'path': str(p), 'bytes': size, 'credited_bytes': min(size, 9 * 1024**3),
            'stage': state['stage'], 'steps': state['steps'],
            'planned_steps': state['planned_steps'], 'data_sha256': state['data_sha256']}


def main():
    output, reports = ROOT / 'data/codenet-durable-pack-v2', ROOT / 'reports/codenet-native-contract-pack-v2'
    if output.exists() or reports.exists():
        raise ValueError('Preserve every earlier preparation and partial result')
    occupied = occupied_writer_reservation()
    reserve = 26 * 1024**3 - occupied['credited_bytes']
    maximum_pack, maximum_reports = 512 * 1024**2, 32 * 1024**2
    if shutil.disk_usage(ROOT).free < reserve + maximum_pack + maximum_reports:
        raise ValueError('Preserve remaining writer reserve plus full output budgets')
    output.mkdir(); reports.mkdir()
    writer = None

    def status(message, **fields):
        atomic_json(reports / 'status.json', dict(status=message,
            updated_at=datetime.now(timezone.utc).isoformat(),
            dataset_admitted_to_training=False, new_model_training_started=False,
            goal_achieved=False, **fields))

    try:
        old = ROOT / 'data/codenet-durable-pack-v1'
        old_audit = ROOT / 'reports/codenet-durable-pack-v1/reconstruction-check.json'
        pack = CompleteCodePack(old, 'train', purpose='audit')
        if (json.loads(old_audit.read_text())['source_files_sha256'][str(old / 'manifest.json')]
                != file_hash(old / 'manifest.json')):
            raise ValueError('Independently reconstructed exact original pack required')
        qpath = ROOT / 'reports/codenet-numba-original-references-v1/source-declared-aot-qualification-v1.json'
        qualification = json.loads(qpath.read_text())
        qualified = {q['original_archive_member']: q for q in qualification['source_declared_native_preparation_qualifications']}
        if len(qualified) != 6 or any(q['split'] != 'train' for q in qualified.values()):
            raise ValueError('Exactly the six original training-source qualifications required')
        sources = [Path(__file__), ROOT / 'src/slm_lab/codenet_native_contracts_v1.py',
            ROOT / 'src/slm_lab/codenet_complete_context_v1.py', ROOT / 'src/slm_lab/codenet_token_pack_v1.py',
            ROOT / 'src/slm_lab/codenet_checkpoint_v1.py', ROOT / 'src/slm_lab/training.py',
            ROOT / 'src/slm_lab/model.py', old / 'manifest.json', old_audit, qpath]
        sources += [old / n for n in pack.manifest['files_sha256']]
        bound = {str(p): file_hash(p) for p in sources}
        guard = FileBindings(bound)
        plan = dict(recorded_at=datetime.now(timezone.utc).isoformat(),
            input_files_sha256=bound, original_manifest_sha256=file_hash(old / 'manifest.json'),
            source_qualification_report_sha256=file_hash(qpath),
            qualified_members=sorted(qualified),
            change='Add explicit source-declared two-stage runtime contract to exactly six training prompts; original programmes, roles and target tokens preserved.',
            complete_source_population=pack.manifest['source_examples'],
            storage=dict(original_writer_reserve_bytes=26*1024**3,
                occupied_verified_writer=occupied, remaining_writer_reserve_bytes=reserve,
                maximum_pack_bytes=maximum_pack, maximum_reports_bytes=maximum_reports,
                existing_files_deleted_or_modified=False),
            actual_native_preparation_executed=False, dataset_admitted_to_training=False,
            new_model_training_started=False, goal_achieved=False)
        atomic_json(reports / 'plan.json', plan)
        for name in ['tokenizer.json', 'model-config.json', 'DATASET-LICENSE.txt', 'publisher-README.md']:
            shutil.copyfile(old / name, output / name)
        encoder = CompleteCodeEncoder(Tokenizer.from_file(str(output / 'tokenizer.json')),
                                      ModelConfig.load(output / 'model-config.json'), ROOT)
        writer = CodePackWriter(output, maximum_bytes=maximum_pack, minimum_free_gib=reserve / 1024**3)
        inventory, changed, seen = [], [], set()
        reverse_roles = {v: k for k, v in ROLES.items()}
        with gzip.open(old / 'original-rows.jsonl.gz', 'rt', encoding='utf-8') as f:
            for position, line in enumerate(f):
                row = json.loads(line); member = row['original_archive_member']; item = pack.index[position]
                if member in seen or row['token_pack_index'] != position or row['split'] != reverse_roles[int(item['role'])]:
                    raise ValueError('Exact original population, order and roles required')
                seen.add(member)
                start, length, target = int(item['offset']), int(item['length']), int(item['target_start'])
                original_ids = pack.ids[start:start+length].astype(np.int64).tolist()
                if (hashlib.sha256(np.asarray(original_ids, dtype='<u2').tobytes()).hexdigest()
                        != row['complete_input_ids_sha256']):
                    raise ValueError('Original whole token identity differs')
                facts = native_inventory(row)
                if facts is not None:
                    inventory.append({**facts, 'required_contract_revision': member in qualified})
                if member in qualified:
                    revised = with_native_contract(row, qualified[member], file_hash(qpath))
                    ids, labels = encoder.sequence(revised, 'response')
                    if ([x for x in labels if x != -100] != original_ids[target:]
                            or revised['response'] != row['response']):
                        raise ValueError('Original complete answer/EOS token sequence changed')
                    changed.append(dict(original_archive_member=member, token_pack_index=position,
                        split=row['split'], old_context_tokens=length, new_context_tokens=len(ids),
                        target_tokens=len(original_ids)-target,
                        original_source_body_sha256=row['original_source_body_sha256']))
                    row = revised
                else:
                    ids, labels = original_ids, [-100] * target + original_ids[target:]
                writer.add(row, ids, labels)
                if (position + 1) % 2000 == 0:
                    guard.check(); status('revising complete-code runtime contracts; no model training',
                        completed_examples=position+1, changed_prompts=len(changed))
                    print('complete_original_examples', position+1, flush=True)
        if (len(seen) != len(pack.index) or {r['original_archive_member'] for r in changed} != set(qualified)
                or len(inventory) != 8 or encoder.truncated):
            raise ValueError('Full original population, six revisions and eight static imports required')
        catalog_path = reports / 'native-contract-inventory.json'
        atomic_json(catalog_path, dict(records=inventory, changed=changed,
            required_native_contracts=6, other_import_sources_unchanged=2,
            actual_native_preparation_executed=False, original_targets_or_roles_modified=False))
        guard.check(full=True)
        manifest = writer.finish(dict(preparation_plan=str(reports / 'plan.json'),
            preparation_plan_sha256=file_hash(reports / 'plan.json'), data_revision='source-declared-runtime-contracts-v2',
            previous_dataset_manifest=str(old / 'manifest.json'),
            previous_dataset_manifest_sha256=file_hash(old / 'manifest.json'),
            native_contract_inventory=str(catalog_path), native_contract_inventory_sha256=file_hash(catalog_path),
            original_source_partitions=pack.manifest['provenance']['source_partitions'],
            original_source_partitions_sha256=pack.manifest['provenance']['source_partitions_sha256'],
            qualification='Historical Accepted human programmes; six source-declared native contracts are explicit. Actual AOT preparation, all-runtime correctness and earlier-corpus contamination remain unverified.'))
        for role in ROLES:
            if any(manifest['counts'][role][key] != pack.manifest['counts'][role][key]
                   for key in ['examples', 'target_tokens']):
                raise ValueError('Original role populations or answer-target exposures changed')
        guard.check(full=True)
        status('complete runtime-contract revision prepared; independent full readback and admission pending',
            completed_examples=len(seen), changed_prompts=len(changed),
            stored_bytes=manifest['stored_bytes'], manifest_sha256=file_hash(output / 'manifest.json'))
        print('revision_complete', len(seen), 'changed_prompts', len(changed), flush=True)
    except Exception as error:
        status('runtime-contract revision stopped; inspect preserved partial files', error=str(error))
        raise
    finally:
        if writer is not None:
            writer.close()


if __name__ == '__main__':
    main()
