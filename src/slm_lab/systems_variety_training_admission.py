"""Verify actual own-tokenizer packed streams and complete older-code rehearsal."""
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from .corpus import file_hash, stable_id
from .diverse_images import jsonl
from .evaluation_provenance import capture
from .systems_variety_admission import audit_collection


def packed(root, tokenizer_sha):
    root = Path(root); manifest = json.loads((root / 'corpus.json').read_text())
    if manifest['tokenizer_sha256'] != tokenizer_sha or manifest['dtype'] != '<u2':
        raise ValueError('Systems own-tokenizer/packed dtype differs')
    for split in manifest['splits'].values():
        if sum(item['tokens'] for item in split['shards']) != split['tokens']: raise ValueError('Systems shard counts differ')
        for item in split['shards']:
            path = root / item['path']
            if path.stat().st_size != item['tokens'] * 2 or file_hash(path) != item['sha256']:
                raise ValueError('Systems packed shard bytes/hash differ')
    return manifest


def verify_stream(rows, tokenizer, arrays):
    eos = tokenizer.token_to_id('<eos>')
    if eos is None: raise ValueError('Own systems tokenizer lacks EOS')
    cursors = {split: [0, 0] for split in arrays}; counts = dict.fromkeys(arrays, 0); groups = {split: set() for split in arrays}
    for row in rows:
        split = 'valid' if int(stable_id(str(row['group_id']))[:8], 16) % 10 == 0 else 'train'
        tokens = np.asarray(tokenizer.encode(row['text']).ids + [eos], dtype=np.uint16)
        position = 0; groups[split].add(row['group_id'])
        while position < len(tokens):
            shard, offset = cursors[split]
            while shard < len(arrays[split]) and offset == len(arrays[split][shard]): shard += 1; offset = 0
            if shard >= len(arrays[split]): raise ValueError('Complete original systems token stream is missing')
            amount = min(len(tokens) - position, len(arrays[split][shard]) - offset)
            if not np.array_equal(tokens[position:position + amount], arrays[split][shard][offset:offset + amount]):
                raise ValueError('Actual systems packed token order/content/partition differs from original Encoder inputs')
            position += amount; cursors[split] = [shard, offset + amount]; counts[split] += amount
    for split in arrays:
        if counts[split] != sum(len(array) for array in arrays[split]): raise ValueError('Extra non-original systems packed tokens')
    if groups['train'] & groups['valid']: raise ValueError('Original systems file groups overlap splits')
    return {'actual_complete_tokens': counts, 'source_file_groups': {split: len(value) for split, value in groups.items()},
            'source_file_group_overlap': 0, 'every_complete_original_text_and_EOS_reencoded_and_matched': True}


def admit(root):
    root = Path(root).resolve(); data = (root / 'data/systems-variety-code-continuation-v1').resolve()
    path = data / 'manifest.json'; digest = file_hash(path); manifest = json.loads(path.read_text())
    old_runtime = manifest['preparation_runtime']
    original_files = [root / name for name in old_runtime['files_sha256']]
    if capture(original_files) != old_runtime: raise ValueError('Original systems token preparation implementation/runtime changed')
    files = original_files + [Path(__file__), root / 'src/slm_lab/systems_variety_admission.py',
                             root / 'src/slm_lab/systems_variety.py', root / 'src/slm_lab/wikipedia_corpus.py']
    runtime = capture(files)
    rows, source_receipt = audit_collection(root)
    if source_receipt != manifest['source_receipt'] or source_receipt['source_manifest_sha256'] != manifest['source_manifest_sha256']:
        raise ValueError('Prepared systems source reconstruction differs')
    records = data / 'new-systems.jsonl'
    if file_hash(records) != manifest['prepared_jsonl_sha256'] or list(jsonl(records)) != rows or len(rows) != manifest['original_source_records']:
        raise ValueError('Prepared complete systems rows differ from original sources')
    tokenizer_path = root / 'data/tokenizer-corpus/tokenizer.json'; tokenizer_sha = file_hash(tokenizer_path)
    if tokenizer_sha != manifest['tokenizer_sha256']: raise ValueError('Prepared systems own tokenizer changed')
    new_root = data / 'new-packed'; old_root = root / 'data/broad-code-continuation-v1/packed'; combined_root = data / 'packed'
    for folder, key in [(new_root, 'new_packed_manifest_sha256'), (old_root, 'previous_packed_manifest_sha256'),
                        (combined_root, 'packed_manifest_sha256')]:
        if file_hash(folder / 'corpus.json') != manifest[key]: raise ValueError('Systems source/component/mixture manifest changed')
    new = packed(new_root, tokenizer_sha); old = packed(old_root, tokenizer_sha); combined = packed(combined_root, tokenizer_sha)
    arrays = {split: [np.memmap(new_root / item['path'], dtype='<u2', mode='r') for item in new['splits'][split]['shards']]
              for split in ('train', 'valid')}
    actual = verify_stream(rows, Tokenizer.from_file(str(tokenizer_path)), arrays)
    if (new['documents'] != len(rows) or new['duplicates_removed']
            or new['source']['sha256'] != manifest['prepared_jsonl_sha256']
            or new['splits']['train']['tokens'] != manifest['new_train_tokens']
            or new['splits']['valid']['tokens'] != manifest['new_valid_tokens']
            or old['splits']['train']['tokens'] != manifest['prior_code_train_tokens_rehearsed']):
        raise ValueError('Complete systems token/file/rehearsal counts differ')
    if [item['name'] for item in combined['components']] != ['prior-broad-code', 'new-os-network-security']:
        raise ValueError('Original systems mixture components differ')
    for split in ('train', 'valid'):
        expected = [(name, row['tokens'], row['sha256']) for name, component in [('prior-broad-code', old), ('new-os-network-security', new)]
                    for row in component['splits'][split]['shards']]
        present = [(row['component'], row['tokens'], row['sha256']) for row in combined['splits'][split]['shards']]
        if expected != present or combined['splits'][split]['tokens'] != sum(row[1] for row in expected):
            raise ValueError('Systems mixture omits or alters earlier/new packed code')
    if (file_hash(path) != digest or file_hash(records) != manifest['prepared_jsonl_sha256']
            or capture(files) != runtime or file_hash(tokenizer_path) != tokenizer_sha):
        raise ValueError('Systems source/token evidence changed during training admission')
    return combined_root, {'passed': True, 'manifest_sha256': digest, 'source_receipt': source_receipt, **actual,
        'new_train_tokens': manifest['new_train_tokens'], 'new_valid_tokens': manifest['new_valid_tokens'],
        'all_prior_code_train_tokens_rehearsed': manifest['prior_code_train_tokens_rehearsed'],
        'combined_train_tokens': combined['splits']['train']['tokens'], 'packed_manifest_sha256': manifest['packed_manifest_sha256'],
        'tokenizer_sha256': tokenizer_sha, 'admission_runtime': runtime, 'training_admitted': True, 'trained': False,
        'goal_achieved': False, 'scope': 'Original file/archive/license reconstruction and all actual packed tokens verified; subsequent model capability remains unmeasured.'}
