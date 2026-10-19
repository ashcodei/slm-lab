"""Freeze only committed, verified Wikipedia parts while collection continues."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from .corpus import file_hash
from .diverse_images import atomic_json


def verify(root, manifest):
    root = Path(root)
    for part in manifest['parts']:
        if file_hash(Path(part['path'])/'corpus.json') != part['manifest_sha256']:
            raise ValueError('Committed source part changed after freezing')
    for values in manifest['splits'].values():
        for shard in values['shards']:
            path = root/shard['path']
            if path.stat().st_size != shard['tokens']*2 or file_hash(path) != shard['sha256']:
                raise ValueError('Frozen token shard changed')


def freeze(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    configuration = json.loads((source/'collection-config.json').read_text())
    if configuration['extraction_version'] != 'wiki-clean-v2':
        raise ValueError('Require the audited Wikipedia extraction version')
    output.mkdir(parents=True, exist_ok=True); manifest_path = output/'corpus.json'
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing['source_working_root'] != str(source) or existing['source_configuration_sha256'] != file_hash(source/'collection-config.json'):
            raise ValueError('Frozen corpus source configuration changed')
        verify(output, existing); return existing
    if any(output.iterdir()): raise ValueError('Incomplete snapshot needs inspection; use a new directory')
    # The collector writes this manifest last, after shard flush and provenance
    # commit. An in-progress directory without corpus.json is never admitted.
    parts = sorted((source/'parts').glob('*/corpus.json'))
    if not parts: raise ValueError('No committed source parts are ready')
    result = {'source': {'dataset': 'Frozen committed Wikimedia article parts', 'date': configuration['date']},
              'source_working_root': str(source), 'source_configuration_sha256': file_hash(source/'collection-config.json'),
              'created_at': datetime.now(timezone.utc).isoformat(), 'dtype': '<u2',
              'tokenizer_sha256': configuration['tokenizer_sha256'],
              'protected_questions_sha256': configuration['questions_sha256'],
              'splits': {name: {'tokens': 0, 'shards': []} for name in ['train', 'valid']},
              'parts': [], 'documents': 0, 'domain_tokens': {},
              'split_rule': 'Preserve original canonical page-ID train/validation assignments.',
              'limitations': ['Frozen subset of completed parts, not completion of the ongoing full collection.',
                              'These tokens recur in the eventual full corpus; later passes are additional exposures, not new unique data.',
                              'Exact protected question filtering does not certify semantic/near-duplicate independence.']}
    domains = defaultdict(Counter)
    for index, path in enumerate(parts):
        part = json.loads(path.read_text())
        if part['dtype'] != '<u2' or part['tokenizer_sha256'] != result['tokenizer_sha256'] or part['extraction_version'] != 'wiki-clean-v2':
            raise ValueError('Committed part dtype/tokenizer/extraction differs')
        if part['source']['snapshot']['date'] != configuration['date'] or not part['source'].get('official_sha1'):
            raise ValueError('Committed part lacks matching publisher verification')
        if file_hash(path.parent/'documents.jsonl') != part['document_provenance_sha256']:
            raise ValueError('Committed article provenance changed')
        result['parts'].append({'path': str(path.parent), 'manifest_sha256': file_hash(path)})
        result['documents'] += part['documents']
        for domain, values in part['domain_tokens'].items(): domains[domain].update(values)
        for split, values in part['splits'].items():
            if sum(shard['tokens'] for shard in values['shards']) != values['tokens']:
                raise ValueError('Committed shard token count mismatch')
            for number, shard in enumerate(values['shards']):
                origin = (path.parent/shard['path']).resolve()
                if not origin.is_relative_to(path.parent.resolve()): raise ValueError('Shard path escapes committed source part')
                if origin.stat().st_size != shard['tokens']*2 or file_hash(origin) != shard['sha256']:
                    raise ValueError('Committed shard bytes/hash mismatch')
                target = output/f'part-{index:03d}-{split}-{number:05d}.bin'; target.symlink_to(origin)
                result['splits'][split]['shards'].append({**shard, 'path': target.name})
            result['splits'][split]['tokens'] += values['tokens']
    if any(values['tokens'] < 4096 for values in result['splits'].values()):
        raise ValueError('Insufficient committed train/validation tokens')
    result['domain_tokens'] = {name: dict(values) for name, values in domains.items()}
    result['total_tokens'] = sum(values['tokens'] for values in result['splits'].values())
    atomic_json(manifest_path, result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); manifest = freeze(args.source, args.output)
    print(json.dumps({'parts': len(manifest['parts']), 'tokens': manifest['total_tokens'],
                      'train_tokens': manifest['splits']['train']['tokens'], 'output': str(Path(args.output).resolve())}, indent=2))


if __name__ == '__main__': main()
