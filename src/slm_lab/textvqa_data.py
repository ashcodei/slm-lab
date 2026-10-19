"""Original human text-reading questions on existing credited full photographs."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from tokenizers import Tokenizer

from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .model import ModelConfig
from .training import Encoder
from .vqa_pairs import flickr_photo_id, known_photo_aliases

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = {
    'train': '95f5c407db56cba56a177799dcd685a7cc0ec7c0d851b59910acf7786d31b68a',
    'val': '4ceb5aadc1a41719d0a3e4dfdf06838bcfee1db569a9a65ee67d31c99893081d',
}
INPUTS = {
    'train': ['alignment-curriculum-v1/images.train.jsonl', 'extended-curriculum-v1/images.train.jsonl',
              'visual-pretraining-v1/train.jsonl', 'expanded-everyday-photos-v1/train.jsonl',
              'expanded-scene-words-v1/train.jsonl'],
    'valid': ['alignment-curriculum-v1/images.valid.jsonl', 'visual-pretraining-v1/valid.jsonl'],
}


def agreed_target(example):
    answers = example['answers']
    if len(answers) != 10 or any(not isinstance(answer, str) or not answer.strip() for answer in answers):
        raise ValueError('Original reading question must contain ten nonempty human answer strings')
    counts = Counter(answers)
    answer, votes = counts.most_common(1)[0]
    if votes < 6 or answer.strip().casefold() in {'unanswerable', 'not a question', 'no text in image'}:
        return None
    return answer, votes


def same_photo(example, stored):
    if stored['group_id'] != 'openimages:' + example['image_id']:
        return False
    expected = {flickr_photo_id(example.get(key)) for key in ['flickr_original_url', 'flickr_300k_url']}
    expected.discard(None)
    actual = known_photo_aliases(stored)
    # Multiple/conflicting identities and missing identities are not evidence of equivalence.
    return len(expected) == len(actual) == 1 and expected == actual


def prepare(output):
    output = Path(output)
    if output.exists():
        raise ValueError('Preserve existing real-reading preparation; inspect before resuming')
    cache = ROOT / 'data/textvqa-source-cache-v1'
    sources = {}
    originals = {}
    for split, expected in SOURCE_SHA.items():
        path = cache / ('TextVQA_0.5.1_' + split + '.json')
        receipt = json.loads(path.with_suffix('.json.source.json').read_text())
        if file_hash(path) != expected or receipt['sha256'] != expected:
            raise ValueError('Original publisher question bytes differ from recorded release')
        if receipt['url'] != 'https://dl.fbaipublicfiles.com/textvqa/data/' + path.name:
            raise ValueError('Question file is not from its original publisher endpoint')
        obj = json.loads(path.read_text())
        if obj['dataset_name'] != 'textvqa' or obj['dataset_type'] != split or obj['dataset_version'] != '0.5.1':
            raise ValueError('Original reading-question dataset/split/version differs')
        if len(obj['data']) != {'train': 34602, 'val': 5000}[split]:
            raise ValueError('Original question inventory is incomplete')
        if len({row['question_id'] for row in obj['data']}) != len(obj['data']):
            raise ValueError('Repeated original question identity')
        if any(row['set_name'] != split for row in obj['data']):
            raise ValueError('Original question/image split mismatch')
        sources[split] = receipt
        originals[split] = obj['data']
    if {row['question_id'] for row in originals['train']} & {row['question_id'] for row in originals['val']}:
        raise ValueError('Original question identities overlap across splits')
    license_path = cache / 'publisher-readme.txt'
    license_receipt = json.loads(license_path.with_suffix('.txt.source.json').read_text())
    license_text = license_path.read_text()
    if (file_hash(license_path) != license_receipt['sha256'] or 'license:\n- cc-by-4.0' not in license_text
            or 'https://huggingface.co/datasets/facebook/textvqa/raw/30a47cfa557c996f64903f01250799d453be6215/' not in license_receipt['url']):
        raise ValueError('Pinned publisher annotation license declaration differs')
    wanted = {row['image_id'] for examples in originals.values() for row in examples}
    protected = {split: {'groups': set(), 'aliases': set(), 'pixels224': set(), 'pixels448': set()}
                 for split in INPUTS}
    available = {split: {} for split in INPUTS}
    input_hashes = {}
    for split, names in INPUTS.items():
        for name in names:
            path = ROOT / 'data' / name
            input_hashes[str(path)] = file_hash(path)
            manifest = json.loads((path.parent / 'manifest.json').read_text()) if (path.parent / 'manifest.json').exists() else None
            if manifest and manifest.get('files_sha256', {}).get(path.name) != input_hashes[str(path)]:
                raise ValueError('Stored image inventory differs from its frozen manifest')
            for row in jsonl(path):
                if not row.get('image'):
                    continue
                guard = protected[split]
                guard['groups'].add(row['group_id']); guard['aliases'].update(known_photo_aliases(row))
                for size in [224, 448]:
                    key = 'model_pixel_sha256' + ('_448' if size == 448 else '')
                    if row.get(key): guard['pixels' + str(size)].add(row[key])
                image_id = row.get('source_image_id', row['group_id'].removeprefix('openimages:'))
                if image_id not in wanted or row.get('task') not in {'caption', 'region_ocr'}:
                    continue
                image = (path.parent / row['image']).resolve()
                # Full photos only; word/object crops are never used as whole-image inputs.
                candidate = {**row, 'image': str(image), 'reused_record_source': str(path.relative_to(ROOT))}
                available[split].setdefault(image_id, candidate)
    tokenizer = ROOT / 'data/tokenizer-corpus/tokenizer.json'
    encoder = Encoder(Tokenizer.from_file(str(tokenizer)), ModelConfig.load(ROOT / 'configs/model.json'), output, 448)
    verified_images = {}
    rows_by_split = {}; statistics = {}; files = {}; groups = {}
    for split, original_split in [('train', 'train'), ('valid', 'val')]:
        opposing = protected['valid' if split == 'train' else 'train']
        exclusions = Counter(); rows = []
        for original in originals[original_split]:
            target = agreed_target(original)
            if target is None:
                exclusions['less than six identical original votes or non-reading target'] += 1; continue
            stored = available[split].get(original['image_id'])
            if stored is None:
                exclusions['no existing admitted full photo in matching split'] += 1; continue
            if not same_photo(original, stored):
                exclusions['missing or conflicting original photo identity'] += 1; continue
            if (stored['group_id'] in opposing['groups'] or known_photo_aliases(stored) & opposing['aliases']
                    or stored.get('model_pixel_sha256') in opposing['pixels224']
                    or stored.get('model_pixel_sha256_448') in opposing['pixels448']):
                exclusions['opposing prepared image/group/alias/pixel overlap'] += 1; continue
            if (stored.get('license') != 'CC-BY-2.0' or not stored.get('author')
                    or not stored.get('author_profile_url') or not stored.get('source_url')):
                raise ValueError('Reused photo lacks original published license/author attribution')
            image = Path(stored['image'])
            if image not in verified_images:
                verified_images[image] = file_hash(image)
            if verified_images[image] != stored['stored_image_sha256']:
                raise ValueError('Actual stored full-photo bytes differ from original admission')
            response, votes = target
            row = {**stored, 'id': 'textvqa:' + original_split + ':' + str(original['question_id']),
                   'prompt': original['question'] + '\nAnswer briefly.', 'response': response,
                   'domain': 'scene_text', 'task': 'text_reading_question', 'split': split,
                   'question_source_url': sources[original_split]['url'], 'question_source_sha256': SOURCE_SHA[original_split],
                   'annotation_license': 'CC-BY-4.0', 'annotation_license_url': 'https://creativecommons.org/licenses/by/4.0/',
                   'annotation_origin': 'Original TextVQA question and ten crowdsourced answer strings; no OCR system output.',
                   'original_textvqa_question': original, 'identical_original_answer_votes': votes,
                   'external_model_teacher_used': False}
            ids, labels = encoder.sequence(row, 'response')
            if encoder.truncated:
                raise ValueError('Complete real-photo question/answer exceeds own context')
            row['context_tokens'] = len(ids); row['target_tokens'] = sum(value != -100 for value in labels[1:])
            rows.append(row)
        if not rows:
            raise ValueError('No original human reading questions survived split-safe stored-photo admission')
        groups[split] = {row['group_id'] for row in rows}
        rows_by_split[split] = rows
        statistics[split] = {'examples': len(rows), 'original_image_groups': len(groups[split]),
                             'target_tokens': sum(row['target_tokens'] for row in rows), 'excluded': dict(exclusions)}
    if groups['train'] & groups['valid']:
        raise ValueError('Prepared real-reading image groups overlap')
    for path, digest in input_hashes.items():
        if file_hash(path) != digest:
            raise ValueError('Stored protected image inventory changed during preparation')
    output.mkdir(parents=True)
    for split, rows in rows_by_split.items():
        path = output / (split + '.jsonl')
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        files[path.name] = file_hash(path)
    manifest = {'prepared_at': datetime.now(timezone.utc).isoformat(), 'implementation_sha256': file_hash(__file__),
                'original_sources': sources, 'publisher_license_declaration': license_receipt,
                'reused_input_files_sha256': input_hashes, 'files_sha256': files, 'splits': statistics,
                'tokenizer_sha256': file_hash(tokenizer), 'validation_group_overlap': 0,
                'new_images_downloaded': 0, 'external_model_weights_used': False, 'goal_achieved': False,
                'trained_in_main_curriculum': False,
                'limitations': ['Reuses existing development validation photos; not independent final benchmark or full TextVQA.',
                                'Only already stored credited full photos and six-of-ten exact answer consensus are retained.',
                                'Photo mirror equivalence to original Flickr bytes remains unverified as in existing image admissions.',
                                'Identity/alias/exact-pixel exclusions do not certify semantic/perceptual near-duplicate absence.',
                                'Original OCR-system tokens and image-class labels are never supplied as model inputs.']}
    atomic_json(output / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    print(json.dumps(prepare(ROOT / 'data/textvqa-human-reading-v1'), indent=2))
