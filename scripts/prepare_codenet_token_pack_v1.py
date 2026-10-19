"""Export durable original programmes with complete tokens and sealed roles."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from tokenizers import Tokenizer

from slm_lab.codenet_complete_context_v1 import CompleteCodeEncoder, problem_program_row
from slm_lab.codenet_token_pack_v1 import CodePackWriter
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports, output = ROOT / 'reports/codenet-durable-pack-v1', ROOT / 'data/codenet-durable-pack-v1'
    if reports.exists() or output.exists():
        raise ValueError('Preserve earlier durable code preparation')
    if shutil.disk_usage(ROOT).free < 26 * 1024**3 + 768 * 1024**2:
        raise ValueError('Require full pack budget above the 26GiB persistent reserve')
    reports.mkdir()
    output.mkdir()
    writer = None
    def status(message, **fields):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
            'dataset_admitted_to_training': False, 'new_model_training_started': False, 'goal_achieved': False, **fields})
    try:
        body_path = ROOT / 'reports/codenet-candidate-bodies-v1/bodies.json'
        body = json.loads(body_path.read_text())
        cache = Path(body['candidate_cache'])
        source_path, index_path = cache / 'source-bodies.bin', cache / 'source-index.jsonl'
        metadata_path = ROOT / 'data/codenet-candidate-selection-v1/selected-metadata.jsonl'
        roles_manifest_path = ROOT / 'data/codenet-internal-splits-v1/manifest.json'
        roles_path = roles_manifest_path.with_name('members.jsonl')
        roles_manifest = json.loads(roles_manifest_path.read_text())
        desc_summary_path = ROOT / 'reports/codenet-description-format-v1/summary.json'
        desc_path = desc_summary_path.with_name('records.jsonl')
        desc_summary = json.loads(desc_summary_path.read_text())
        tokenizer_path = ROOT / 'checkpoints/instruction-generalization-probe-v3/candidate/tokenizer.json'
        config_path = ROOT / 'configs/model.json'
        expected = {source_path: body['source_body_store_sha256'], index_path: body['source_index_sha256'],
                    metadata_path: body['selected_metadata_sha256'], roles_path: roles_manifest['members_sha256'],
                    desc_path: desc_summary['records_sha256'], tokenizer_path: desc_summary['tokenizer_sha256']}
        for path, digest in expected.items():
            if file_hash(path) != digest:
                raise ValueError('Sealed original source bytes differ')
        other = [Path(__file__), ROOT / 'src/slm_lab/codenet_token_pack_v1.py',
                 ROOT / 'src/slm_lab/codenet_complete_context_v1.py', ROOT / 'src/slm_lab/training.py',
                 config_path, body_path, roles_manifest_path, desc_summary_path,
                 ROOT / 'data/codenet-original-source-v1/DATASET-LICENSE.txt',
                 ROOT / 'data/codenet-original-source-v1/publisher-README.md']
        bound = {**expected, **{path: file_hash(path) for path in other}}
        states = {p: (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in bound}
        def guard(full=False):
            for path, digest in bound.items():
                s = path.stat()
                if (s.st_ino, s.st_size, s.st_mtime_ns) != states[path] or (full and file_hash(path) != digest):
                    raise ValueError('Bound complete-code source/runtime changed')
        atomic_json(reports / 'plan.json', {'recorded_at': datetime.now(timezone.utc).isoformat(),
            'input_files_sha256': {str(p): digest for p, digest in bound.items()},
            'minimum_persistent_free_gib': 26, 'maximum_artifact_bytes': 768 * 1024**2,
            'fixed_role_counts': roles_manifest['counts'],
            'original_programme_and_full_mask_preserved': True,
            'dataset_admitted_to_training': False, 'new_model_training_started': False, 'goal_achieved': False})
        for original, name in [(tokenizer_path, 'tokenizer.json'), (config_path, 'model-config.json'),
            (ROOT / 'data/codenet-original-source-v1/DATASET-LICENSE.txt', 'DATASET-LICENSE.txt'),
            (ROOT / 'data/codenet-original-source-v1/publisher-README.md', 'publisher-README.md')]:
            shutil.copyfile(original, output / name)
        descriptions = {row['problem_id']: row for row in map(json.loads, desc_path.open())}
        metadata = {row['archive_member']: row['metadata'] for row in map(json.loads, metadata_path.open())}
        indices = {row['archive_member']: row for row in map(json.loads, index_path.open())}
        config = ModelConfig.load(config_path)
        if config.vocab_size != 32768 or config.max_text_length != 4096:
            raise ValueError('Canonical original pack vocabulary/window differs')
        encoder = CompleteCodeEncoder(Tokenizer.from_file(str(tokenizer_path)), config, ROOT)
        writer = CodePackWriter(output)
        count = 0
        with source_path.open('rb') as sources:
            for selected in map(json.loads, roles_path.open()):
                if selected['context_qualification'] != 'complete_original_pair_fits' or selected['duplicate_complete_example']:
                    continue
                member = selected['archive_member']
                source = indices[member]
                sources.seek(source['body_offset'])
                raw = sources.read(source['body_bytes'])
                if len(raw) != source['body_bytes'] or hashlib.sha256(raw).hexdigest() != source['body_sha256']:
                    raise ValueError('Original complete-program bytes differ')
                m, problem = metadata[member], descriptions[selected['problem_id']]
                row = problem_program_row(problem, m, raw.decode())
                row.update(group_id=selected['group_id'], split=selected['split'],
                    original_archive_member=member, original_source_body_sha256=source['body_sha256'],
                    original_html_sha256=problem['original_html_sha256'], original_submission_metadata=m,
                    publisher_size_outlier=selected['publisher_size_outlier'], dataset_admitted_to_training=False)
                ids, labels = encoder.sequence(row, 'response')
                if (hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).hexdigest() != selected['complete_input_ids_sha256']
                    or len(ids) != selected['context_tokens'] or sum(value != -100 for value in labels) != selected['target_tokens']):
                    raise ValueError('Complete packed context/target differs from sealed population')
                writer.add(row, ids, labels)
                count += 1
                if count % 1000 == 0:
                    guard()
                    status('packing complete original source pairs; no model training', completed_examples=count)
                    print('complete_examples', count, flush=True)
        if count != sum(c['unique_complete_examples'] for c in roles_manifest['counts'].values()) or encoder.truncated:
            raise ValueError('Complete unique source population differs or any target clipped')
        guard(full=True)
        manifest = writer.finish({'preparation_plan': str(reports / 'plan.json'),
            'preparation_plan_sha256': file_hash(reports / 'plan.json'),
            'original_archive_sha256': '7e96620307f874269a62cc0225f2a5a13d2de2b3c368cf61319c87ec276620fb',
            'source_partitions': str(roles_manifest_path), 'source_partitions_sha256': file_hash(roles_manifest_path),
            'qualification': 'Historical original accepted programmes with complete text fit. All language/runtime semantics, broader execution and earlier-corpus contamination are not certified.'})
        for role, expected_counts in roles_manifest['counts'].items():
            actual = manifest['counts'][role]
            if actual != {'examples': expected_counts['unique_complete_examples'],
                          'context_tokens': expected_counts['unique_context_tokens'],
                          'target_tokens': expected_counts['unique_target_tokens']}:
                raise ValueError('Durable source role counts/targets differ')
        status('durable complete-code candidate pack finished; independent reconstruction and admission pending',
            completed_examples=count, stored_bytes=manifest['stored_bytes'], manifest_sha256=file_hash(output / 'manifest.json'))
    except Exception as error:
        status('durable code preparation stopped; inspect preserved partial output', error=str(error))
        raise
    finally:
        if writer is not None:
            writer.close()


if __name__ == '__main__':
    main()
