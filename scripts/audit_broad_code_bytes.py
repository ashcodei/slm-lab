"""Independently reconstruct library chunks and both own-tokenizer streams."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
from tokenizers import Tokenizer
from slm_lab.corpus import file_hash
from slm_lab.data import stable_id
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.public_sources import chunks, fetch

ROOT = Path(__file__).resolve().parents[1]


class TokenReader:
    def __init__(self, root, manifest, split):
        self.root = root; self.shards = manifest['splits'][split]['shards']; self.index = 0; self.offset = 0; self.values = None
        self.position = 0; self.total = manifest['splits'][split]['tokens']

    def compare(self, expected):
        expected = np.asarray(expected, dtype='<u2'); position = 0
        while position < len(expected):
            if self.values is None:
                if self.index >= len(self.shards): raise ValueError('Packed stream ended before original documents')
                self.values = np.memmap(self.root / self.shards[self.index]['path'], dtype='<u2', mode='r')
            count = min(len(expected) - position, len(self.values) - self.offset)
            if not np.array_equal(expected[position:position + count], self.values[self.offset:self.offset + count]):
                raise ValueError('Packed tokens differ from independent original JSONL tokenization')
            position += count; self.offset += count; self.position += count
            if self.offset == len(self.values): self.values = None; self.index += 1; self.offset = 0


def main():
    source = ROOT / 'data/reviewed-systems-code-v1'
    prepared = ROOT / 'data/broad-code-continuation-v1'
    source_manifest = json.loads((source / 'manifest.json').read_text())
    preparation = json.loads((prepared / 'manifest.json').read_text())
    if file_hash(source / 'all.jsonl') != source_manifest['all_sha256'] or file_hash(prepared / 'new-libraries.jsonl') != preparation['prepared_jsonl_sha256']:
        raise ValueError('Frozen library JSONL changed before independent audit')
    archives = {}; files = 0; audited_chunks = 0
    for item in source_manifest['sources']:
        archive_path = Path(item.get('archive_path', source / 'archives' / (item['name'] + '.zip')))
        if not archive_path.exists():
            raw = fetch('https://codeload.github.com/' + item['repo'] + '/zip/' + item['revision'])
            if hashlib.sha256(raw).hexdigest() != item['archive_sha256']:
                raise ValueError('Original pinned archive byte equivalence not verified: ' + item['repo'])
            archive_path.write_bytes(raw)
        if file_hash(archive_path) != item['archive_sha256']: raise ValueError('Cached original archive bytes differ')
        archive = zipfile.ZipFile(archive_path); prefix = archive.namelist()[0].split('/')[0] + '/'
        groups = defaultdict(list)
        for row in jsonl(source / (item['name'] + '.jsonl')): groups[row['source_url']].append(row)
        try:
            for url, rows in groups.items():
                relative = url.split('/blob/' + item['revision'] + '/', 1)[1]
                if PurePosixPath(relative).is_absolute() or '..' in PurePosixPath(relative).parts: raise ValueError('Unsafe publisher path')
                raw = archive.read(prefix + relative); text = raw.decode('utf-8'); files += 1
                framed = 'File: ' + relative + '\nLanguage: ' + PurePosixPath(relative).suffix + '\n\n' + text
                original_chunks = list(chunks(framed))
                for row in rows:
                    if original_chunks[row['chunk']] != row['text'] or stable_id(row['text']) != row['id']:
                        raise ValueError('Stored source chunk differs from original pinned publisher bytes')
                    if row.get('source_file_sha256') and row['source_file_sha256'] != hashlib.sha256(raw).hexdigest():
                        raise ValueError('Per-file original publisher digest differs')
                    audited_chunks += 1
        finally: archive.close()
        archives[item['repo']] = {'revision': item['revision'], 'archive_sha256': item['archive_sha256'], 'path': str(archive_path)}
        print('Original library byte audit', item['repo'], len(groups), 'files', flush=True)
    packed = json.loads((prepared / 'new-packed/corpus.json').read_text())
    tokenizer_path = ROOT / 'data/tokenizer-corpus/tokenizer.json'
    if file_hash(tokenizer_path) != packed['tokenizer_sha256']: raise ValueError('Own tokenizer changed')
    tokenizer = Tokenizer.from_file(str(tokenizer_path)); eos = tokenizer.token_to_id('<eos>')
    readers = {split: TokenReader(prepared / 'new-packed', packed, split) for split in ['train', 'valid']}
    originals = {row['id']: row for row in jsonl(source / 'all.jsonl')}
    assigned = {}; count = 0
    for row in jsonl(prepared / 'new-libraries.jsonl'):
        original = originals[row['source_chunk_id']]
        if row['text'] != 'Programming language: ' + row['declared_language'] + '\n' + original['text']:
            raise ValueError('Prepared library text differs from its declared metadata plus original chunk')
        for key in ['group_id', 'source_url', 'revision', 'license', 'license_url', 'language_extension']:
            if row[key] != original[key]: raise ValueError('Prepared source provenance differs: ' + key)
        group_key = stable_id(str(row['group_id'])); split = 'valid' if int(group_key[:8], 16) % 10 == 0 else 'train'
        if row['group_id'] in assigned and assigned[row['group_id']] != split: raise ValueError('Source-file group split drift')
        assigned[row['group_id']] = split
        readers[split].compare(tokenizer.encode(row['text']).ids + [eos]); count += 1
    if count != packed['documents'] or any(reader.position != reader.total for reader in readers.values()):
        raise ValueError('Retokenized document/stream totals differ')
    report = {'completed_at': datetime.now(timezone.utc).isoformat(), 'sources': archives, 'original_files_checked': files,
              'original_source_chunks_checked': audited_chunks, 'prepared_chunks_retokenized': count,
              'train_tokens_exactly_matched': readers['train'].position, 'valid_tokens_exactly_matched': readers['valid'].position,
              'file_groups': len(assigned), 'source_group_train_validation_overlap': 0,
              'prepared_text_equals_metadata_plus_original_source': True,
              'source_manifest_sha256': file_hash(source / 'manifest.json'),
              'preparation_manifest_sha256': file_hash(prepared / 'manifest.json'),
              'packed_manifest_sha256': file_hash(prepared / 'new-packed/corpus.json'),
              'audit_source_sha256': file_hash(__file__), 'goal_achieved': False,
              'scope': 'Original bytes/chunk reconstruction and independent exact token-stream/group audit; no trained-model capability claim.'}
    atomic_json(ROOT / 'reports/broad-code-original-bytes-token-audit-v1.json', report)
    print('Independent code-source/token audit complete', count, 'chunks', flush=True)


if __name__ == '__main__': main()
