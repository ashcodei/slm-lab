"""Validate original human QA inputs independently, then require exposure evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unicodedata

from tokenizers import Tokenizer

from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .evaluation_provenance import capture
from .model import ModelConfig
from .training import Encoder


def validate_row(row, original, split, line, encoder, revision):
    question = original['question']; answers = original['answer']
    normalized = ' '.join(unicodedata.normalize('NFKC', question).casefold().split())
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    expected = {'id': 'nq-open:' + split + ':' + str(line) + ':' + digest,
                'group_id': 'nq-open-question:' + digest, 'source_line': line, 'source_split': split,
                'split': 'train' if split == 'train' else 'valid', 'task': 'nq_open_question',
                'language': 'English', 'domain': 'open_domain_knowledge',
                'prompt': question + '\nAnswer briefly.', 'response': answers[0],
                'reference_answers': answers, 'source_original_question': original,
                'feedback_source': 'human', 'source_revision': revision, 'license': 'CC BY-SA 3.0',
                'license_url': 'https://creativecommons.org/licenses/by-sa/3.0/',
                'source_url': 'https://github.com/google-research-datasets/natural-questions/blob/' + revision + '/nq_open/NQ-open.' + split + '.jsonl#L' + str(line)}
    if any(row.get(key) != value for key, value in expected.items()) or any(key in row for key in ['tokens', 'text', 'image', 'chosen', 'rejected']):
        raise ValueError('Prepared NQ input/answer/aliases/identity differs from original publisher source')
    prompt_ids = encoder.tokenizer.encode('User: ' + question + '\nAnswer briefly.\nAssistant:').ids
    answer_ids = encoder.tokenizer.encode(answers[0]).ids + [encoder.eos]
    ids, labels = encoder.sequence(row, 'response')
    if (encoder.truncated or ids != [encoder.bos] + prompt_ids + answer_ids
            or labels != [-100] * (len(prompt_ids) + 1) + answer_ids
            or row['target_tokens'] != len(answer_ids)):
        raise ValueError('Actual NQ encoder input/masked target differs from complete original question/answer')
    return len(answer_ids)


def audit(root):
    root = Path(root).resolve(); data = root / 'data/nq-human-questions-v1'; source = root / 'data/nq-open-original-source-v1'
    manifest_path = data / 'manifest.json'; manifest = json.loads(manifest_path.read_text())
    original_manifest_path = source / 'manifest.json'; original_manifest = json.loads(original_manifest_path.read_text())
    prep_files = [root / name for name in manifest['source_provenance']['files_sha256']]
    if capture(prep_files) != manifest['source_provenance'] or file_hash(original_manifest_path) != manifest['source_manifest_sha256']:
        raise ValueError('NQ preparation source/runtime/original publisher inventory changed')
    if manifest['source_revision'] != original_manifest['revision'] or original_manifest['data_license'] != 'CC BY-SA 3.0':
        raise ValueError('NQ original revision or explicit data license differs')
    paths = [manifest_path, original_manifest_path, root / 'configs/model.json', root / 'data/tokenizer-corpus/tokenizer.json']
    paths += [data / name for name in manifest['files_sha256']]
    paths += [source / name for name in original_manifest['sources']]
    for name, item in original_manifest['sources'].items():
        raw = (source / name).read_bytes()
        if (hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() != item['publisher_git_blob_sha1']
                or hashlib.sha256(raw).hexdigest() != item['sha256']): raise ValueError('Original publisher NQ bytes/blob changed')
    hashes = {str(path): file_hash(path) for path in paths}
    for name, digest in manifest['files_sha256'].items():
        if hashes[str(data / name)] != digest: raise ValueError('Prepared NQ split bytes changed')
    tokenizer_path = root / 'data/tokenizer-corpus/tokenizer.json'; config_path = root / 'configs/model.json'
    if hashes[str(tokenizer_path)] != manifest['tokenizer_sha256'] or hashes[str(config_path)] != manifest['model_config_sha256']:
        raise ValueError('NQ own tokenizer/config changed')
    files = list(dict.fromkeys([*prep_files, Path(__file__)])); runtime = capture(files)
    encoder = Encoder(Tokenizer.from_file(str(tokenizer_path)), ModelConfig.load(config_path), data, 448)
    groups = {}; counts = {}
    for split, filename in [('train', 'train.jsonl'), ('dev', 'valid.jsonl')]:
        seen = set(); target_tokens = 0; count = 0
        for count, (row, original) in enumerate(zip(jsonl(data / filename), jsonl(source / ('nq_open/NQ-open.' + split + '.jsonl')), strict=True), 1):
            target_tokens += validate_row(row, original, split, count, encoder, manifest['source_revision'])
            if row['group_id'] in seen: raise ValueError('Repeated original NQ question group')
            seen.add(row['group_id'])
        if count != manifest['splits'][split]['examples'] or target_tokens != manifest['splits'][split]['target_token_exposures_per_pass']:
            raise ValueError('NQ original population or target exposure differs')
        groups[split] = seen; counts[split] = {'examples': count, 'target_tokens': target_tokens}
    if groups['train'] & groups['dev']: raise ValueError('NQ original question groups cross train/development split')
    if capture(files) != runtime or any(file_hash(path) != digest for path, digest in hashes.items()):
        raise ValueError('NQ actual inputs/encoder/runtime changed during independent audit')
    return {'completed_at': datetime.now(timezone.utc).isoformat(), 'passed': True,
            'manifest_sha256': hashes[str(manifest_path)], 'files_sha256': hashes, 'splits': counts,
            'implementation_runtime': runtime, 'original_question_answer_alias_rows_verified': sum(x['examples'] for x in counts.values()),
            'complete_actual_input_ids_and_label_masks_verified': True, 'training_validation_group_overlap': 0,
            'goal_achieved': False, 'scope': 'Original source field identity, complete own-encoder inputs/targets and reference aliases independently verified for every prepared row; no capability claim.'}


def admit(root):
    root = Path(root).resolve(); data = root / 'data/nq-human-questions-v1'
    checked = json.loads((root / 'reports/nq-original-input-audit-v1.json').read_text())
    exposure_path = root / 'reports/nq-development-exposure-v1/audit.json'
    exposure = json.loads(exposure_path.read_text())
    if (checked['passed'] is not True or checked['manifest_sha256'] != file_hash(data / 'manifest.json')
            or checked['complete_actual_input_ids_and_label_masks_verified'] is not True
            or checked['splits']['train']['examples'] != 87925 or checked['splits']['dev']['examples'] != 3610
            or exposure.get('completed') is not True or exposure['benchmark_questions'] != 3610
            or exposure['benchmark_data_sha256'] != file_hash(data / 'valid.jsonl')):
        raise ValueError('Completed NQ source/input/exposure audit required before training admission')
    for report in [checked, exposure]:
        files = [root / name for name in report['implementation_runtime']['files_sha256']]
        if capture(files) != report['implementation_runtime']: raise ValueError('NQ completed auditor/runtime changed')
    for path, digest in {**checked['files_sha256'], **exposure['source_files_sha256']}.items():
        if file_hash(path) != digest: raise ValueError('Audited NQ original/processed/training-source bytes changed')
    plan = json.loads((exposure_path.parent / 'plan.json').read_text())
    for name, path in plan['packed_corpora'].items():
        path = Path(path); manifest = json.loads((path / 'corpus.json').read_text())
        if file_hash(path / 'corpus.json') != exposure['packed_findings'][name]['corpus_manifest_sha256']:
            raise ValueError('NQ exposure-audited packed source inventory changed')
        for shard in manifest['splits']['train']['shards']:
            if file_hash(path / shard['path']) != shard['sha256']: raise ValueError('NQ exposure-audited packed training bytes changed')
    rows = list(jsonl(data / 'train.jsonl'))
    if len(rows) != 87925 or any(row['split'] != 'train' or row.get('image') for row in rows): raise ValueError('Original NQ training-only population changed')
    return rows, {'passed': True, 'input_audit_sha256': file_hash(root / 'reports/nq-original-input-audit-v1.json'),
                  'exposure_audit_sha256': file_hash(exposure_path), 'manifest_sha256': file_hash(data / 'manifest.json'),
                  'original_training_questions': len(rows), 'original_development_questions': 3610,
                  'development_cases_with_exact_potential_training_exposure': exposure['exposed_case_ids'],
                  'validation_rows_not_returned_for_training': True, 'goal_achieved': False,
                  'scope': 'Full original human input audit and completed potential-exposure inventory required; training rows only admitted, exposed dev cases remain explicitly identified.'}


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    result = audit(root); atomic_json(root / 'reports/nq-original-input-audit-v1.json', result)
    print(json.dumps({key: result[key] for key in ['passed', 'splits', 'original_question_answer_alias_rows_verified']}), flush=True)
