"""Check finalized CLEVR rows against original metadata, archive entries and pixels."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import time
import zipfile
import zlib

from PIL import Image, ImageOps
from tokenizers import Tokenizer
from slm_lab.clevr_programs import answer, array_items
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.masked_visual_pretraining import process_identity
from slm_lab.model import ModelConfig
from slm_lab.remote_zip import RemoteZipFile
from slm_lab.training import Encoder

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = 'd4435dceab30022b64f7d34b3ec217162bfc1c2950c79999784750ced22674c4'
IMAGE_URL = 'https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip'
SUFFIX = '\nAnswer with only the number, attribute name, or yes/no.'


def source_identity():
    paths = [Path(__file__), ROOT / 'configs/model.json']
    paths.extend(ROOT / 'src/slm_lab' / (name + '.py') for name in
                 ['clevr_data', 'clevr_programs', 'remote_zip', 'training', 'model', 'diverse_images', 'corpus'])
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def original_fields(question, scene):
    split = question['split']; filename = question['image_filename']
    if (split not in {'train', 'val'} or scene['split'] != split or scene['image_filename'] != filename
            or scene['image_index'] != question['image_index']
            or filename != f'CLEVR_{split}_{scene["image_index"]:06d}.png'):
        raise ValueError('Original question and scene identities differ')
    if answer(scene, question['program']) != question['answer']:
        raise ValueError('Original symbolic answer differs')
    return {'id': f'clevr:{split}:question:{question["question_index"]}', 'group_id': f'clevr:{split}:{filename}',
            'image': 'images/' + filename, 'prompt': question['question'] + SUFFIX, 'response': question['answer'],
            'question_program': question['program'], 'question_family_index': question['question_family_index'],
            'reasoning_type': question['program'][-1]['function'], 'source_subset': split,
            'original_zip_member': f'CLEVR_v1.0/images/{split}/{filename}',
            'scene_sha256': hashlib.sha256(json.dumps(scene, sort_keys=True).encode()).hexdigest(),
            'task': 'visual_reasoning', 'domain': 'visual_reasoning', 'license': 'CC-BY-4.0',
            'symbolic_answer_verified': True, 'synthetic_rendered_scene': True, 'external_model_teacher_used': False}


def validate_question(row, question, scene, encoder):
    expected = original_fields(question, scene)
    for key, value in expected.items():
        if row.get(key) != value or type(row.get(key)) is not type(value):
            raise ValueError('Prepared question differs from original source: ' + key)
    prompt = 'User: ' + expected['prompt'] + '\nAssistant:'
    if encoder.prompt(row) != prompt:
        raise ValueError('Hidden metadata changed the actual model prompt')
    prompt_ids = encoder.tokenizer.encode(prompt).ids
    response_ids = encoder.tokenizer.encode(question['answer']).ids + [encoder.eos]
    wanted = [encoder.bos] + prompt_ids + response_ids
    if len(wanted) > encoder.config.max_text_length:
        raise ValueError('Original full visual question exceeds context')
    ids, labels = encoder.sequence(row, 'response')
    if (ids != wanted or labels != [-100] * (len(prompt_ids) + 1) + response_ids or encoder.truncated
            or row['context_tokens'] != len(wanted) or row['target_tokens'] != len(response_ids)):
        raise ValueError('Prepared tokens/targets differ from complete original question and answer')
    return expected


def validate_image(root, row, entry):
    relative = PurePosixPath(row['image'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Image path escapes the finalized dataset')
    path = (root / row['image']).resolve()
    if not path.is_relative_to(root.resolve()): raise ValueError('Image symlink escapes dataset')
    raw = path.read_bytes()
    if (len(raw) != entry.file_size or len(raw) != row['original_member_bytes']
            or zlib.crc32(raw) != entry.CRC or entry.CRC != row['original_member_crc32']
            or hashlib.sha256(raw).hexdigest() != row['stored_image_sha256']):
        raise ValueError('Stored image bytes differ from original archive entry/record')
    result = {}
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != 'PNG' or image.size != (480, 320): raise ValueError('Original CLEVR image format/dimensions differ')
        for size in [224, 448]:
            pixel_bytes = ImageOps.pad(image.convert('RGB'), (size, size), method=Image.Resampling.BICUBIC,
                                       color=(128, 128, 128)).tobytes()
            digest = hashlib.sha256(pixel_bytes).hexdigest()
            key = 'model_pixel_sha256' if size == 224 else 'model_pixel_sha256_448'
            if digest != row[key]: raise ValueError('Stored image input pixels differ: ' + str(size))
            result[size] = digest
    return result, len(raw)


def audit(data, output, expected_sources=None):
    sources = source_identity()
    if expected_sources is not None and sources != expected_sources: raise ValueError('Loaded audit/preparation sources changed while waiting')
    data = data.resolve(); manifest_path = data / 'manifest.json'
    manifest_digest = file_hash(manifest_path); manifest = json.loads(manifest_path.read_text())
    for key, name in [('implementation_sha256', 'clevr_data'), ('program_verifier_sha256', 'clevr_programs'),
                      ('range_reader_sha256', 'remote_zip')]:
        if manifest[key] != sources['src/slm_lab/' + name + '.py']: raise ValueError('Finalized preparation implementation differs: ' + name)
    cache = ROOT / 'data/clevr-source-cache-v1'; source_path = cache / 'CLEVR_v1.0_no_images.zip'
    receipt = json.loads((cache / 'CLEVR_v1.0_no_images.zip.source.json').read_text())
    if (receipt['sha256'] != SOURCE_SHA256 or file_hash(source_path) != SOURCE_SHA256
            or manifest['source_metadata_archive'] != receipt):
        raise ValueError('Original metadata archive/source receipt changed')
    for name, digest in manifest['files_sha256'].items():
        path = (data / name).resolve()
        if not path.is_relative_to(data) or file_hash(path) != digest: raise ValueError('Finalized dataset file changed')
    tokenizer_path = ROOT / 'data/tokenizer-corpus/tokenizer.json'
    if file_hash(tokenizer_path) != manifest['tokenizer_sha256']: raise ValueError('Own tokenizer changed')
    encoder = Encoder(Tokenizer.from_file(str(tokenizer_path)), ModelConfig.load(ROOT / 'configs/model.json'), data, 448)
    rows = {split: list(jsonl(data / (split + '.jsonl'))) for split in ['train', 'valid']}
    pixels = {split: {224: set(), 448: set()} for split in rows}; checked = {}; stats = {}; storage = 0
    scenes = {}; original_questions = {}; selected_counts = {'train': 4000, 'val': 512}
    with zipfile.ZipFile(source_path) as source:
        for name in ['LICENSE.txt', 'COPYRIGHT.txt']:
            raw = source.read('CLEVR_v1.0/' + name)
            if (data / name).read_bytes() != raw: raise ValueError('Original license/copyright bytes differ')
            key = 'license_sha256' if name == 'LICENSE.txt' else 'copyright_sha256'
            if hashlib.sha256(raw).hexdigest() != manifest[key]: raise ValueError('Manifest license/copyright digest differs')
        for split, limit in selected_counts.items():
            with source.open(f'CLEVR_v1.0/scenes/CLEVR_{split}_scenes.json') as stream:
                scenes[split] = {scene['image_filename']: scene for scene in array_items(stream, 'scenes') if scene['image_index'] < limit}
            if len(scenes[split]) != limit: raise ValueError('Original selected scene count differs')
            name = 'train' if split == 'train' else 'valid'
            snapshot = list(jsonl(data / 'metadata' / (name + '-scenes.jsonl')))
            if snapshot != list(scenes[split].values()): raise ValueError('Prepared scene metadata differs from original ZIP')
            questions = {}
            with source.open(f'CLEVR_v1.0/questions/CLEVR_{split}_questions.json') as stream:
                for question in array_items(stream, 'questions'):
                    if question['image_filename'] not in scenes[split]: continue
                    identity = original_fields(question, scenes[split][question['image_filename']])['id']
                    if identity in questions: raise ValueError('Repeated original question identity')
                    questions[identity] = question
            original_questions[name] = questions
            print('Original metadata independently compared', name, len(questions), 'questions', flush=True)
    excluded = {item['image'] for item in manifest['excluded']}
    if len(excluded) != len(manifest['excluded']): raise ValueError('Repeated excluded image')
    all_ids = set()
    all_image_pixels = {'train': {}, 'val': {}}
    with RemoteZipFile(IMAGE_URL) as remote:
        if remote.identity != manifest['source_image_archive_identity']: raise ValueError('Original image archive HTTP identity changed')
        with zipfile.ZipFile(remote) as archive:
            for split, values in rows.items():
                part = 'train' if split == 'train' else 'val'; ids = set(); groups = set(); tasks = Counter(); targets = 0
                for index, row in enumerate(values):
                    if row['id'] in all_ids: raise ValueError('Repeated prepared question identity')
                    all_ids.add(row['id']); ids.add(row['id']); groups.add(row['group_id']); tasks[row['reasoning_type']] += 1
                    if row['id'] not in original_questions[split]: raise ValueError('Prepared question absent from original selected source')
                    original = original_questions[split][row['id']]; scene = scenes[part][original['image_filename']]
                    validate_question(row, original, scene, encoder); targets += row['target_tokens']
                    entry = archive.getinfo(row['original_zip_member'])
                    if row['image'] not in checked:
                        image_pixels, count = validate_image(data, row, entry); checked[row['image']] = (image_pixels, count); storage += count
                    else:
                        image_pixels, count = checked[row['image']]
                        if (row['stored_image_sha256'] != file_hash(data / row['image']) or count != row['original_member_bytes']
                                or row['original_member_crc32'] != entry.CRC
                                or any(image_pixels[s] != row['model_pixel_sha256' if s == 224 else 'model_pixel_sha256_448'] for s in image_pixels)):
                            raise ValueError('Repeated image has inconsistent records')
                    for size, digest in image_pixels.items(): pixels[split][size].add(digest)
                    if (index + 1) % 5000 == 0: print('Original questions and image records checked', split, index + 1, flush=True)
                expected_ids = {identity for identity, q in original_questions[split].items() if split == 'valid' or q['image_filename'] not in excluded}
                if ids != expected_ids: raise ValueError('Prepared questions omit/add original admitted cases')
                stats[split] = {'examples': len(values), 'images': len(groups), 'reasoning_types': dict(tasks), 'target_tokens': targets}
                if stats[split] != manifest['splits'][split]: raise ValueError('Prepared split aggregates differ from original cases')
            # Excluded training images are still stored by preparation. Inspect
            # them too, then prove that each omission really protects a held-out input.
            for part, selected in scenes.items():
                for filename in selected:
                    relative = 'images/' + filename
                    if relative not in checked:
                        entry = archive.getinfo(f'CLEVR_v1.0/images/{part}/{filename}')
                        path = data / relative
                        with Image.open(path) as image:
                            digests = {size: hashlib.sha256(ImageOps.pad(image.convert('RGB'), (size, size),
                                       method=Image.Resampling.BICUBIC, color=(128, 128, 128)).tobytes()).hexdigest() for size in [224, 448]}
                        record = {'image': relative, 'stored_image_sha256': file_hash(path), 'original_member_bytes': entry.file_size,
                                  'original_member_crc32': entry.CRC, 'model_pixel_sha256': digests[224], 'model_pixel_sha256_448': digests[448]}
                        image_pixels, count = validate_image(data, record, entry); checked[relative] = (image_pixels, count); storage += count
                    all_image_pixels[part][filename] = checked[relative][0]
            if {p.name for p in (data / 'images').iterdir()} != {filename for selected in scenes.values() for filename in selected}:
                raise ValueError('Stored image inventory differs from original selected scenes')
        range_bytes = remote.fetched_bytes
    protected = {224: set(), 448: set()}
    for row in jsonl(ROOT / 'data/alignment-curriculum-v1/images.valid.jsonl'):
        protected[224].add(row['model_pixel_sha256']); protected[448].add(row['model_pixel_sha256_448'])
    for values in all_image_pixels['val'].values():
        for size, digest in values.items(): protected[size].add(digest)
    actually_excluded = {name for name, values in all_image_pixels['train'].items()
                         if any(digest in protected[size] for size, digest in values.items())}
    if excluded != actually_excluded or any(item['reason'] != 'Protected held-out input pixels' for item in manifest['excluded']):
        raise ValueError('Omitted training images are not exactly the original protected-pixel exclusions')
    for size in pixels['train']:
        if pixels['train'][size] & (pixels['valid'][size] | protected[size]): raise ValueError('Training pixels overlap known held-out inputs')
    if storage != manifest['stored_image_bytes']: raise ValueError('Finalized image byte total differs')
    if file_hash(manifest_path) != manifest_digest: raise ValueError('Manifest changed during audit')
    if source_identity() != sources: raise ValueError('Audit/preparation sources changed during audit')
    report = {'completed_at': datetime.now(timezone.utc).isoformat(), 'passed': True, 'goal_achieved': False,
              'manifest_sha256': manifest_digest, 'source_metadata_archive_sha256': SOURCE_SHA256,
              'tokenizer_sha256': manifest['tokenizer_sha256'], 'splits': stats, 'stored_images_checked': len(checked),
              'stored_image_bytes_checked': storage, 'files_sha256': manifest['files_sha256'],
              'input_pixel_overlap_with_known_validation': {'224': 0, '448': 0},
              'original_image_archive_identity': manifest['source_image_archive_identity'], 'central_directory_range_bytes': range_bytes,
              'audit_source_sha256': file_hash(__file__),
              'source_files_sha256': sources,
              'scope': 'Original source questions/scenes, stored PNG bytes/archive CRC, both pixel resolutions and own full tokens; not model capability.',
              'limitations': ['Shares symbolic interpreter and JSON stream reader with preparation; not an independent reimplementation of those algorithms.',
                              'Original image member CRC/size and HTTP identity rechecked; whole19GB archive SHA not independently available.',
                              'Exact protected pixels only; perceptual/semantic independence and visual capability are not established.']}
    atomic_json(output, report); print('Original CLEVR input audit passed', stats, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/clevr-reasoning-v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/clevr-original-input-audit-v1.json')
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args(); status_path = args.output.with_suffix('.status.json')
    loaded_sources = source_identity()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.output.exists(): raise ValueError('Preserve existing audit; use another report path for a fresh run')
        try:
            if not (args.data / 'manifest.json').exists():
                if args.wait_pid is None: raise ValueError('Finalized CLEVR manifest absent')
                identity = process_identity(args.wait_pid)
                if identity is None or b'slm_lab.clevr_data' not in Path(f'/proc/{args.wait_pid}/cmdline').read_bytes().split(b'\0'):
                    raise ValueError('Expected original-image collector is not verified live')
                atomic_json(status_path, {'status': 'waiting for verified live original-image collector', 'collector_pid': args.wait_pid,
                                         'collector_start_ticks': identity, 'goal_achieved': False})
                while not (args.data / 'manifest.json').exists():
                    if process_identity(args.wait_pid) != identity: raise ValueError('Collector ended without finalized manifest')
                    time.sleep(15)
            atomic_json(status_path, {'status': 'auditing original metadata, image entries and prepared inputs', 'goal_achieved': False})
            audit(args.data, args.output, expected_sources=loaded_sources)
            atomic_json(status_path, {'status': 'original CLEVR inputs audited; model capability unverified', 'goal_achieved': False})
        except Exception as error:
            atomic_json(status_path, {'status': 'input audit failed; inspect evidence before admission', 'error': str(error), 'goal_achieved': False})
            raise


if __name__ == '__main__': main()
