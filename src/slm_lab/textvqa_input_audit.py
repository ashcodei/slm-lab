"""Recheck original answers, full-photo binding, actual pixels and training labels."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from PIL import Image
from tokenizers import Tokenizer

from .corpus import file_hash
from .diverse_images import atomic_json, jsonl, pixel_hash
from .evaluation_provenance import capture
from .model import ModelConfig
from .textvqa_data import SOURCE_SHA, same_photo
from .training import Encoder
from .vqa_pairs import known_photo_aliases

ROOT = Path(__file__).resolve().parents[2]
PHOTO_FIELDS = ['group_id', 'source', 'source_url', 'source_image_id', 'original_url', 'original_md5',
                'author', 'author_profile_url', 'download_sha256', 'download_url', 'license', 'license_url',
                'stored_image_sha256', 'model_pixel_sha256', 'model_pixel_sha256_448']


def validate_question(row, original, original_split):
    answers = original['answers']
    if len(answers) != 10 or any(not isinstance(answer, str) or not answer.strip() for answer in answers):
        raise ValueError('Original question does not have ten nonempty human answers')
    votes = Counter(answers)
    if (row['original_textvqa_question'] != original or original['set_name'] != original_split
            or row['id'] != 'textvqa:' + original_split + ':' + str(original['question_id'])
            or row['group_id'] != 'openimages:' + original['image_id']
            or row['prompt'] != original['question'] + '\nAnswer briefly.'
            or votes[row['response']] < 6 or votes[row['response']] != max(votes.values())
            or row['identical_original_answer_votes'] != votes[row['response']]
            or row['task'] != 'text_reading_question' or not same_photo(original, row)):
        raise ValueError('Question/target/photo binding differs from original human source')


def validate_photo_binding(row, source, source_file):
    if source.get('task') not in {'caption', 'region_ocr'}:
        raise ValueError('Reading input is a crop rather than an admitted full photograph')
    expected = (Path(source_file).parent / source['image']).resolve()
    if Path(row['image']).resolve() != expected or any(row.get(key) != source.get(key) for key in PHOTO_FIELDS):
        raise ValueError('Photo path/rights/identity/pixel metadata differs from actual reused source record')
    if row['license'] != 'CC-BY-2.0' or not row['author'] or not row['author_profile_url']:
        raise ValueError('Original published photo credit/license is incomplete')


def audit(output=None):
    data = ROOT / 'data/textvqa-human-reading-v1'
    output = ROOT / 'reports/textvqa-original-input-audit-v1.json' if output is None else Path(output)
    if output.exists():
        raise ValueError('Preserve prior original-input audit; inspect before rerunning')
    files = [Path(__file__), ROOT / 'src/slm_lab/textvqa_data.py', ROOT / 'src/slm_lab/vqa_pairs.py',
             ROOT / 'src/slm_lab/diverse_images.py', ROOT / 'src/slm_lab/training.py', ROOT / 'src/slm_lab/model.py']
    runtime = capture(files)
    manifest_path = data / 'manifest.json'; manifest_sha = file_hash(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if file_hash(ROOT / 'src/slm_lab/textvqa_data.py') != manifest['implementation_sha256']:
        raise ValueError('Real-reading preparation implementation changed')
    originals = {}; source_hashes = {}
    for split, expected in SOURCE_SHA.items():
        path = ROOT / 'data/textvqa-source-cache-v1' / ('TextVQA_0.5.1_' + split + '.json')
        if file_hash(path) != expected or manifest['original_sources'][split]['sha256'] != expected:
            raise ValueError('Actual original publisher question bytes changed')
        source_hashes[str(path)] = expected
        obj = json.loads(path.read_text())
        originals[split] = {row['question_id']: row for row in obj['data']}
        if len(originals[split]) != len(obj['data']): raise ValueError('Repeated original source question IDs')
    license_receipt = manifest['publisher_license_declaration']
    license_path = ROOT / 'data/textvqa-source-cache-v1/publisher-readme.txt'
    if file_hash(license_path) != license_receipt['sha256'] or 'license:\n- cc-by-4.0' not in license_path.read_text():
        raise ValueError('Pinned original annotation license changed')
    source_hashes[str(license_path)] = license_receipt['sha256']
    all_rows = {split: list(jsonl(data / (split + '.jsonl'))) for split in ['train', 'valid']}
    wanted = {row['group_id'] for rows in all_rows.values() for row in rows}
    source_records = {}; protected = {'train': {'groups': set(), 'aliases': set(), 'pixels': set()},
                                     'valid': {'groups': set(), 'aliases': set(), 'pixels': set()}}
    for name, digest in manifest['reused_input_files_sha256'].items():
        path = Path(name)
        if not path.resolve().is_relative_to(ROOT) or file_hash(path) != digest:
            raise ValueError('Protected original image inventory changed')
        source_hashes[name] = digest
        split = 'valid' if path.name.endswith('valid.jsonl') else 'train'
        candidates = {}; guard = protected[split]
        for row in jsonl(path):
            if not row.get('image'): continue
            guard['groups'].add(row['group_id']); guard['aliases'].update(known_photo_aliases(row))
            guard['pixels'].update(value for value in [row.get('model_pixel_sha256'), row.get('model_pixel_sha256_448')] if value)
            if row['group_id'] in wanted and row.get('task') in {'caption', 'region_ocr'}:
                candidates.setdefault(row['group_id'], []).append(row)
        source_records[str(path.relative_to(ROOT))] = candidates
    tokenizer = ROOT / 'data/tokenizer-corpus/tokenizer.json'
    if file_hash(tokenizer) != manifest['tokenizer_sha256']:
        raise ValueError('Own tokenizer changed')
    encoder = Encoder(Tokenizer.from_file(str(tokenizer)), ModelConfig.load(ROOT / 'configs/model.json'), data, 448)
    verified = {}; statistics = {}; groups = {}
    for split, original_split in [('train', 'train'), ('valid', 'val')]:
        rows = all_rows[split]; path = data / (split + '.jsonl')
        if file_hash(path) != manifest['files_sha256'][path.name] or len({row['id'] for row in rows}) != len(rows):
            raise ValueError('Prepared real-reading record hashes/identities differ')
        target_tokens = 0; opposing = protected['valid' if split == 'train' else 'train']
        for row in rows:
            original = originals[original_split][row['original_textvqa_question']['question_id']]
            validate_question(row, original, original_split)
            source_name = row['reused_record_source']
            options = source_records[source_name][row['group_id']]
            source_file = ROOT / source_name
            matches = [source for source in options if (source_file.parent / source['image']).resolve() == Path(row['image']).resolve()]
            if len(matches) != 1: raise ValueError('No unique actual source full-photo record for reading input')
            validate_photo_binding(row, matches[0], source_file)
            if (row['group_id'] in opposing['groups'] or known_photo_aliases(row) & opposing['aliases']
                    or row['model_pixel_sha256'] in opposing['pixels'] or row['model_pixel_sha256_448'] in opposing['pixels']):
                raise ValueError('Prepared reading image overlaps an opposing stored training/validation input')
            image = Path(row['image']).resolve()
            if not image.is_relative_to(ROOT): raise ValueError('Reading photo path escapes the project')
            if image not in verified:
                actual_sha = file_hash(image)
                with Image.open(image) as pixels:
                    actual_pixels = {size: pixel_hash(pixels, size) for size in [224, 448]}
                verified[image] = (actual_sha, actual_pixels)
            actual_sha, actual_pixels = verified[image]
            if (actual_sha != row['stored_image_sha256'] or actual_pixels[224] != row['model_pixel_sha256']
                    or actual_pixels[448] != row['model_pixel_sha256_448']):
                raise ValueError('Actual full-photo bytes or model input pixels differ from recorded image')
            ids, labels = encoder.sequence(row, 'response')
            targets = sum(value != -100 for value in labels[1:])
            if encoder.truncated or len(ids) != row['context_tokens'] or targets != row['target_tokens']:
                raise ValueError('Complete original question/answer own-tokenizer labels differ')
            if encoder.prompt(row) != 'User: ' + original['question'] + '\nAnswer briefly.\nAssistant:':
                raise ValueError('Actual model prompt contains more than the original question/task request')
            target_tokens += targets
        groups[split] = {row['group_id'] for row in rows}
        statistics[split] = {'examples': len(rows), 'original_image_groups': len(groups[split]), 'target_tokens': target_tokens}
        for key, value in statistics[split].items():
            if manifest['splits'][split][key] != value: raise ValueError('Audited source totals differ from manifest')
        print('Original human reading inputs verified', split, statistics[split], flush=True)
    if groups['train'] & groups['valid']: raise ValueError('Real-reading photo groups overlap')
    for path, digest in source_hashes.items():
        if file_hash(path) != digest: raise ValueError('Original source/inventory changed during input audit')
    if file_hash(manifest_path) != manifest_sha: raise ValueError('Prepared manifest changed during audit')
    if capture(files) != runtime or file_hash(tokenizer) != manifest['tokenizer_sha256']:
        raise ValueError('Loaded audit implementation/runtime/tokenizer changed during verification')
    for name, digest in manifest['files_sha256'].items():
        if file_hash(data / name) != digest: raise ValueError('Audited question records changed during verification')
    report = {'audited_at': datetime.now(timezone.utc).isoformat(), 'passed': True,
              'manifest_sha256': manifest_sha, 'audit_source_sha256': file_hash(__file__),
              'preparation_source_sha256': manifest['implementation_sha256'], 'splits': statistics,
              'files_sha256': manifest['files_sha256'], 'original_source_and_inventory_sha256': source_hashes,
              'actual_full_photos_rehashed': len(verified), 'actual_model_pixel_sizes_rechecked': [224, 448],
              'implementation_runtime': runtime, 'tokenizer_sha256': manifest['tokenizer_sha256'],
              'validation_group_overlap': 0, 'goal_achieved': False,
              'scope': 'Every original question/ten answers/consensus target, actual reused full-photo binding/credits/file/pixels, and native prompt/labels; shared encoder/pixel/alias helpers, not independent reimplementations.'}
    atomic_json(output, report)
    return report


if __name__ == '__main__': audit()
