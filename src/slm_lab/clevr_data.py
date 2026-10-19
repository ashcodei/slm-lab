"""Original CC-BY CLEVR subset with verified scene-program answers and images."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

from PIL import Image
from tokenizers import Tokenizer
from .clevr_programs import answer, array_items
from .corpus import file_hash
from .diverse_images import atomic_json, jsonl, pixel_hash
from .model import ModelConfig
from .remote_zip import RemoteZipFile
from .training import Encoder

ROOT = Path(__file__).resolve().parents[2]
PUBLISHER = 'https://cs.stanford.edu/people/jcjohns/clevr/'
IMAGE_URL = 'https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip'


def collect(output, train_images=4000, valid_images=512):
    output = Path(output).resolve(); cache = ROOT / 'data/clevr-source-cache-v1'
    if output.exists(): raise ValueError('Existing CLEVR preparation requires inspection, not automatic overwrite')
    if shutil.disk_usage(ROOT).free < 35 * 1024**3: raise ValueError('Need35 GiB free before bounded CLEVR collection')
    output.mkdir(); (output / 'images').mkdir(); (output / 'metadata').mkdir()
    source_path = cache / 'CLEVR_v1.0_no_images.zip'
    source_receipt = json.loads((cache / 'CLEVR_v1.0_no_images.zip.source.json').read_text())
    if file_hash(source_path) != source_receipt['sha256']: raise ValueError('Original scene/question archive changed')
    source = zipfile.ZipFile(source_path)
    license_bytes = source.read('CLEVR_v1.0/LICENSE.txt'); copyright_bytes = source.read('CLEVR_v1.0/COPYRIGHT.txt')
    if not license_bytes.startswith(b'Attribution 4.0 International') or b'Creative Commons Attribution 4.0' not in copyright_bytes:
        raise ValueError('Original dataset license declaration differs')
    (output / 'LICENSE.txt').write_bytes(license_bytes); (output / 'COPYRIGHT.txt').write_bytes(copyright_bytes)
    limits = {'train': train_images, 'val': valid_images}; scenes = {}; questions = {}; metadata_checks = {}
    for split, limit in limits.items():
        scene_path = f'CLEVR_v1.0/scenes/CLEVR_{split}_scenes.json'
        question_path = f'CLEVR_v1.0/questions/CLEVR_{split}_questions.json'
        selected = {}
        with source.open(scene_path) as stream:
            for scene in array_items(stream, 'scenes'):
                if scene['image_index'] < limit:
                    if scene['split'] != split: raise ValueError('Original scene split mismatch')
                    selected[scene['image_filename']] = scene
        if len(selected) != limit: raise ValueError('Original selected scene count differs')
        scenes[split] = selected; questions[split] = []
        with source.open(question_path) as stream:
            for question in array_items(stream, 'questions'):
                if question['image_filename'] not in selected: continue
                if question['split'] != split or answer(selected[question['image_filename']], question['program']) != question['answer']:
                    raise ValueError('Original question answer differs from its symbolic scene program')
                questions[split].append(question)
        for path in [scene_path, question_path]:
            item = source.getinfo(path); metadata_checks[path] = {'crc32': item.CRC, 'uncompressed_bytes': item.file_size}
        print('Original CLEVR programs verified', split, len(selected), 'images', len(questions[split]), 'questions', flush=True)
    protected = {224: set(), 448: set()}
    for row in jsonl(ROOT / 'data/alignment-curriculum-v1/images.valid.jsonl'):
        protected[224].add(row['model_pixel_sha256']); protected[448].add(row['model_pixel_sha256_448'])
    tokenizer_path = ROOT / 'data/tokenizer-corpus/tokenizer.json'
    encoder = Encoder(Tokenizer.from_file(str(tokenizer_path)), ModelConfig.load(ROOT / 'configs/model.json'), output, 448)
    remote = RemoteZipFile(IMAGE_URL); image_records = {}; storage = 0
    try:
        with zipfile.ZipFile(remote) as archive:
            for path, expected in metadata_checks.items():
                item = archive.getinfo(path)
                if item.CRC != expected['crc32'] or item.file_size != expected['uncompressed_bytes']:
                    raise ValueError('Original image/annotation archive metadata CRC/size differs')
            ordered = []
            for split, selected in scenes.items():
                for filename in selected:
                    path = f'CLEVR_v1.0/images/{split}/{filename}'; info = archive.getinfo(path)
                    if info.file_size > 4 * 1024**2: raise ValueError('Original image exceeds bounded image size')
                    ordered.append((info.header_offset, split, filename, path, info))
            # Byte-offset ordering keeps HTTP page caching efficient and original selection fixed.
            for index, (_, split, filename, path, info) in enumerate(sorted(ordered)):
                raw = archive.read(info)
                with Image.open(io.BytesIO(raw)) as image:
                    pixels = {size: pixel_hash(image, size) for size in [224, 448]}
                image_path = output / 'images' / filename; image_path.write_bytes(raw); storage += len(raw)
                if storage > 1024**3: raise ValueError('Bounded CLEVR stored-image budget exceeded')
                image_records[(split, filename)] = {'image': str(image_path.relative_to(output)),
                    'stored_image_sha256': hashlib.sha256(raw).hexdigest(), 'model_pixel_sha256': pixels[224],
                    'model_pixel_sha256_448': pixels[448], 'original_zip_member': path, 'original_member_crc32': info.CRC,
                    'original_member_bytes': info.file_size, 'scene_sha256': hashlib.sha256(json.dumps(scenes[split][filename], sort_keys=True).encode()).hexdigest()}
                if (index + 1) % 100 == 0: print('Original CLEVR images', index + 1, 'stored MiB', storage // 1024**2, flush=True)
    finally: source.close()
    for (split, _), record in image_records.items():
        if split == 'val':
            protected[224].add(record['model_pixel_sha256']); protected[448].add(record['model_pixel_sha256_448'])
    excluded = []; seen_train = {224: set(), 448: set()}; splits = {}; files = {}; target_totals = Counter()
    for split in ['train', 'val']:
        admitted = []; selected_images = set(); domain = 'visual_reasoning'; name = 'train' if split == 'train' else 'valid'
        blocked = set()
        for (part, filename), record in image_records.items():
            if part != split or split != 'train': continue
            if record['model_pixel_sha256'] in protected[224] or record['model_pixel_sha256_448'] in protected[448]:
                blocked.add(filename); excluded.append({'image': filename, 'reason': 'Protected held-out input pixels'})
        scene_file = output / 'metadata' / (name + '-scenes.jsonl')
        scene_file.write_text(''.join(json.dumps(value, sort_keys=True) + '\n' for value in scenes[split].values()))
        for question in questions[split]:
            filename = question['image_filename']
            if filename in blocked: continue
            image = image_records[(split, filename)]
            row = {**image, 'id': f'clevr:{split}:question:{question["question_index"]}',
                   'group_id': f'clevr:{split}:{filename}', 'source': 'CLEVR original v1.0', 'source_url': PUBLISHER,
                   'license': 'CC-BY-4.0', 'license_url': 'https://creativecommons.org/licenses/by/4.0/',
                   'attribution': 'CLEVR (c)2017 Facebook, Inc.; Johnson et al., CVPR2017.',
                   'domain': domain, 'task': 'visual_reasoning', 'source_subset': split,
                   'prompt': question['question'] + '\nAnswer with only the number, attribute name, or yes/no.',
                   'response': question['answer'], 'question_program': question['program'],
                   'reasoning_type': question['program'][-1]['function'], 'question_family_index': question['question_family_index'],
                   'symbolic_answer_verified': True, 'synthetic_rendered_scene': True, 'external_model_teacher_used': False}
            ids, labels = encoder.sequence(row, 'response')
            if encoder.truncated: raise ValueError('Original visual question/answer does not fit own context')
            row['context_tokens'] = len(ids); row['target_tokens'] = sum(value != -100 for value in labels[1:])
            admitted.append(row); selected_images.add(filename); target_totals[name] += row['target_tokens']
        path = output / (name + '.jsonl'); path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in admitted))
        files[path.name] = file_hash(path); files[str(scene_file.relative_to(output))] = file_hash(scene_file)
        splits[name] = {'examples': len(admitted), 'images': len(selected_images),
                        'reasoning_types': dict(Counter(row['reasoning_type'] for row in admitted)), 'target_tokens': target_totals[name]}
    result = {'created_at': datetime.now(timezone.utc).isoformat(), 'source_metadata_archive': source_receipt,
              'source_image_archive_identity': remote.identity, 'source_range_bytes_fetched': remote.fetched_bytes,
              'source_range_requests': remote.requests, 'stored_image_bytes': storage, 'license': 'CC-BY-4.0',
              'license_sha256': file_hash(output / 'LICENSE.txt'), 'copyright_sha256': file_hash(output / 'COPYRIGHT.txt'),
              'tokenizer_sha256': file_hash(tokenizer_path), 'splits': splits, 'files_sha256': files, 'excluded': excluded,
              'original_metadata_crc_size_matches_image_archive': metadata_checks,
              'implementation_sha256': file_hash(__file__), 'program_verifier_sha256': file_hash(ROOT / 'src/slm_lab/clevr_programs.py'),
              'range_reader_sha256': file_hash(ROOT / 'src/slm_lab/remote_zip.py'),
              'trained': False, 'goal_achieved': False, 'external_model_weights_used': False,
              'limitations': ['Original synthetic scenes and templated questions supplement real photos; they do not establish generic-image performance.',
                              'First4000 train/512 validation image indices, fixed original splits; not full CLEVR/CoGenT or independent final broad assessment.',
                              'Image ranges have pinned HTTP identity/member CRC/SHA; the full19GB archive was not downloaded or whole-file hashed.',
                              'Metadata archive equivalence uses CRC/size, not cryptographic equality of every large image-archive JSON member.',
                              'Exact protected input pixels are screened; perceptual near-duplicate independence is not proved.']}
    atomic_json(output / 'manifest.json', result); print('Original CLEVR subset ready', splits, flush=True); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', required=True)
    args = parser.parse_args(); collect(args.output)


if __name__ == '__main__': main()
