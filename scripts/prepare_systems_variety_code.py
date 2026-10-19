"""Pack new public systems/security code while preserving the entire older code mixture."""
from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
import json
from pathlib import Path

from slm_lab.corpus import file_hash, pack
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.mix_corpora import mix
from slm_lab.systems_variety_admission import audit_collection
from prepare_broad_code_continuation import verify_packed

ROOT = Path(__file__).resolve().parents[1]


def main(report_path=None):
    report = Path(report_path).resolve() if report_path else ROOT / 'reports/systems-variety-code-preparation-v1.json'
    if report_path and report.exists():
        raise ValueError('Preserve the existing named systems preparation report')
    collection = (ROOT / 'data/public-systems-variety-v1').resolve()
    output = collection.parent / 'prepared-code'
    alias = ROOT / 'data/systems-variety-code-continuation-v1'
    if output.exists() or alias.exists() or alias.is_symlink(): raise ValueError('Preserve previous systems token preparation')
    files = [Path(__file__), ROOT / 'src/slm_lab/systems_variety.py', ROOT / 'src/slm_lab/systems_variety_admission.py',
        ROOT / 'src/slm_lab/corpus.py', ROOT / 'src/slm_lab/mix_corpora.py', ROOT / 'configs/systems-variety-sources-v1.json',
        ROOT / 'scripts/prepare_broad_code_continuation.py']
    runtime = capture(files)
    rows, receipt = audit_collection(ROOT); output.mkdir(); alias.symlink_to(output, target_is_directory=True)
    records = output / 'new-systems.jsonl'
    records.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    tokenizer = ROOT / 'data/tokenizer-corpus/tokenizer.json'; tokenizer_sha = file_hash(tokenizer)
    pack(Namespace(jsonl=str(records), output=str(output / 'new-packed'), tokenizer=str(tokenizer),
        max_tokens=200_000_000, shard_tokens=8_000_000, dataset=None))
    new = verify_packed(output / 'new-packed', tokenizer_sha)
    if new['documents'] != len(rows) or new['duplicates_removed']:
        raise ValueError('Complete systems source packing omitted or duplicated original files')
    old = ROOT / 'data/broad-code-continuation-v1/packed'; prior = verify_packed(old, tokenizer_sha)
    combined = mix([('prior-broad-code', old, None), ('new-os-network-security', output / 'new-packed', None)], output / 'packed')
    verify_packed(output / 'packed', tokenizer_sha)
    if capture(files) != runtime: raise ValueError('Systems preparation source/runtime changed')
    manifest = {'prepared_at': datetime.now(timezone.utc).isoformat(), 'source_collection': str(collection),
        'source_manifest_sha256': receipt['source_manifest_sha256'], 'source_receipt': receipt,
        'original_source_records': len(rows), 'prepared_jsonl_sha256': file_hash(records), 'tokenizer_sha256': tokenizer_sha,
        'new_packed_manifest_sha256': file_hash(output / 'new-packed/corpus.json'),
        'previous_packed_manifest_sha256': file_hash(old / 'corpus.json'),
        'packed_manifest_sha256': file_hash(output / 'packed/corpus.json'),
        'prior_code_train_tokens_rehearsed': prior['splits']['train']['tokens'],
        'new_train_tokens': new['splits']['train']['tokens'], 'new_valid_tokens': new['splits']['valid']['tokens'],
        'combined_train_tokens': combined['splits']['train']['tokens'], 'combined_valid_tokens': combined['splits']['valid']['tokens'],
        'preparation_runtime': runtime, 'training_admitted': False, 'trained': False, 'goal_achieved': False,
        'scope': 'All 16 original public OS/network/security projects with complete original files and prior code rehearsal; own tokenizer, no external model teacher.',
        'limitations': ['Source-file split hashing and exact benchmark filters do not establish semantic independence.',
            'Recorded licenses are publisher/file declarations; original projects were not executed.',
            'Prior code tokens are rehearsal, not newly collected code.', 'Temporary source/token paths must survive the later source-bound training.']}
    atomic_json(output / 'manifest.json', manifest)
    atomic_json(report, manifest)
    print('New systems variety tokens ready', manifest['new_train_tokens'], 'new training tokens', flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--report', help='Write a new preparation receipt without replacing historical evidence')
    main(parser.parse_args().report)
