"""Reconstruct all retained rendered/real word pixels and whole native labels."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import csv
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from tokenizers import Tokenizer

from slm_lab.byte_word_reader import load_rows
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.rendered_text import render, text_group
from slm_lab.shared_word_data_v1 import FORMAT, array_digest, image_arrays, native_bytes, original_crop
from slm_lab.whole_human_alignment_v2 import WholeResponseEncoder

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'data/shared-byte-words-v1'
CACHE = ROOT/'data/image-source-cache-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('Word preparation requires explicitly hidden GPUs')
    if OUTPUT.exists(): raise ValueError('Preserve every prior source reconstruction attempt')
    if '3 passed' not in (ROOT/'reports/shared-word-data-v1-tests.log').read_text():
        raise ValueError('Whole-byte, original crop, native pixel/label and held-role checks required')
    rows, old = load_rows(ROOT/'data/byte-word-reader-v1')
    scene = json.loads((ROOT/'data/scene-text-images-v1/manifest.json').read_text())
    rendered = ROOT/'data/rendered-text-v1'; render_config = json.loads((rendered/'configuration.json').read_text())
    paths = ['scripts/prepare_shared_byte_words_v1.py', 'src/slm_lab/shared_word_data_v1.py',
        'src/slm_lab/byte_word_reader.py', 'src/slm_lab/rendered_text.py', 'src/slm_lab/whole_human_alignment_v2.py',
        'tests/test_shared_word_data_v1.py', 'reports/shared-word-data-v1-tests.log', 'configs/model.json',
        'data/byte-word-reader-v1/manifest.json', 'data/scene-text-images-v1/manifest.json',
        'data/rendered-text-v1/configuration.json', 'checkpoints/whole-human-SFT-v1/initial/tokenizer.json']
    files = {str(ROOT/name):file_hash(ROOT/name) for name in paths}; files.update(old['source_files_sha256'])
    annotations = {}; original_receipts = {}
    for role, ending in (('train', 'train'), ('valid', 'val')):
        path = CACHE/f'TextOCR_0.1_{ending}.json'
        receipt = next(s for s in scene['sources'] if s['url'].endswith('/'+path.name))
        if file_hash(path) != receipt['sha256']: raise ValueError('Pinned original-format human annotation bytes changed')
        annotations[role] = json.loads(path.read_text()); original_receipts[role] = receipt
        files[str(path)] = receipt['sha256']
    fonts = {}
    for name, digest in render_config['fonts'].items():
        matches = [p for p in Path('/usr/share/fonts').rglob(name) if file_hash(p) == digest]
        if len(matches) != 1: raise ValueError('Exact original rendering font required: '+name)
        fonts[name] = matches[0]; files[str(matches[0])] = digest
    notice = rendered/'font-license.txt'
    if file_hash(notice) != render_config['font_license_sha256']: raise ValueError('Original font notice changed')
    files[str(notice)] = render_config['font_license_sha256']
    protected = {r['group_id'] for r in rows['valid']}
    for name in ['alignment-curriculum-v1/images.valid.jsonl', 'object-qa-v1/valid.jsonl',
                 'alignment-curriculum-v1/contrastive.valid.jsonl', 'visual-pretraining-v1/valid.jsonl']:
        path = ROOT/'data'/name; files[str(path)] = file_hash(path)
        protected.update(json.loads(line)['group_id'] for line in path.open())
    if any(r['group_id'] in protected for r in rows['train']):
        raise ValueError('Existing original held-photo/text group entered shared-word training')
    parent_ids = {r['source_image_id'] for role in rows for r in rows[role] if r['byte_family'] != 'rendered'}
    metadata = {}
    for subset in ('train', 'validation', 'test'):
        path = CACHE/f'openimages-{subset}.csv'
        receipt = next(s for s in scene['sources'] if s['url'].endswith('/'+('train-images-boxable-with-rotation.csv' if subset == 'train' else subset+'-images-with-rotation.csv')))
        if file_hash(path) != receipt['sha256']: raise ValueError('Original publisher photograph metadata changed')
        files[str(path)] = receipt['sha256']
        with path.open(newline='') as source:
            for row in csv.DictReader(source):
                if row['ImageID'] in parent_ids:
                    if row['ImageID'] in metadata: raise ValueError('An original photograph occurs in multiple publisher subsets')
                    metadata[row['ImageID']] = {**row, 'Subset':subset}
    if set(metadata) != parent_ids: raise ValueError('Every original real-word parent needs publisher creator/license metadata')
    guard = FileBindings(files); guard.check(full=True)
    tokenizer = Tokenizer.from_file(str(ROOT/'checkpoints/whole-human-SFT-v1/initial/tokenizer.json'))
    config = SimpleNamespace(**json.loads((ROOT/'configs/model.json').read_text()))
    encoder = WholeResponseEncoder(tokenizer, config)
    OUTPUT.mkdir()
    atomic_json(OUTPUT/'plan.json', dict(created_at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        original_selected_counts={role:len(values) for role,values in rows.items()}, workers=8,
        input_geometry=dict(word=[3,64,448], native=[3,448,448]), ctc_native_bytes='0..255; blank256',
        ctc_frames=140, no_source_transcript_or_native_context_truncation=True,
        original_annotation_receipts=original_receipts, model_updates=0, goal_achieved=False))

    @lru_cache(maxsize=16)
    def parent_image(key, subset, digest):
        path = CACHE/'source-images'/subset/(key+'.jpg')
        if file_hash(path) != digest: raise ValueError('Original photograph bytes changed: '+key)
        with Image.open(path) as image: return image.convert('RGB')

    def inspect(item):
        role, position, row = item; family = row['byte_family']; path = Path(row['image'])
        if row['split'] != role or file_hash(path) != row['stored_image_sha256']:
            raise ValueError('Original complete word source role/image bytes changed')
        with Image.open(path) as loaded: image = loaded.convert('RGB')
        if family == 'rendered':
            if row['font_sha256'] != render_config['fonts'][row['font']]: raise ValueError('Original font source differs')
            expected, style = render(row['response'], fonts[row['font']], render_config['seed']+row['attempt'])
            if json.loads(json.dumps(style)) != row['style']: raise ValueError('Original rendering seed/style must reconstruct')
            origin = dict(kind='exact_source_string_rendered_with_original_font_and_seed', font_sha256=row['font_sha256'])
        else:
            key = row['source_image_id']; original = annotations[role]; annotation = original['anns'][row['original_word_id']]
            photo = original['imgs'][key]; official = metadata[key]
            if (str(annotation['id']) != row['original_word_id'] or str(annotation['image_id']) != key
                    or annotation.get('utf8_string', '').strip() != row['response'] or annotation['bbox'] != row['original_bbox']
                    or row['original_annotation_split'] != role or photo['set'] != ('train' if role == 'train' else 'val')
                    or official['Subset'] != row['source_subset'] or (role == 'train' and official['Subset'] != 'train')
                    or row['group_id'] != 'openimages:'+key or row['parent_image_id'] != key):
                raise ValueError('Whole original human word/box/photo/source split must reconstruct: '+row['id'])
            if row['license'] != 'CC-BY-2.0' or row['annotation_license'] != 'CC-BY-4.0':
                raise ValueError('Exact retained photograph and original annotation declarations required')
            for name, source in [('author','Author'), ('source_url','OriginalLandingURL'), ('source_title','Title'),
                                 ('author_profile_url','AuthorProfileURL'), ('original_url','OriginalURL'),
                                 ('original_md5','OriginalMD5'), ('license_url','License')]:
                if row[name] != official[source]: raise ValueError('Original publisher attribution field differs: '+name)
            expected = original_crop(parent_image(key, row['source_subset'], row['download_sha256']), photo, annotation, family)
            origin = dict(kind='original_format_human_annotation_and_publisher_photo_crop',
                annotation_source_sha256=original_receipts[role]['sha256'], parent_photo_sha256=row['download_sha256'],
                publisher_metadata_subset=official['Subset'], annotation_mirror_equivalence_independently_verified=False)
        if image.size != expected.size or image.tobytes() != expected.tobytes():
            raise ValueError('Every stored word crop must match its actual original render/photo pixels: '+row['id'])
        word, native = image_arrays(image); target, minimum = native_bytes(row['response'])
        ids, start = encoder.sequence(row, 'response')
        return dict(**row, original_selected_position=position,
            original_runtime_row_sha256=hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',',':')).encode()).hexdigest(),
            original_pixel_and_label_proof=origin, word_input_pixels_sha256=array_digest(word), native_input_pixels_sha256=array_digest(native),
            source_RGB_pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest(), source_dimensions=list(image.size),
            native_byte_targets=target, minimum_CTC_steps=minimum, _native_ids=ids, _target_start=start)

    counts = {}; fingerprints = {}; all_files = {}; role_groups = {}; rendered_groups = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for role in ('valid', 'train'):
            statistics = Counter(); hashes = dict(word=set(), native=set(), stored=set()); groups = set(); rendered_texts = set()
            stream_path = OUTPUT/(role+'.jsonl'); token_path = OUTPUT/(role+'.tokens.u16'); cursor = 0; targets = 0
            with stream_path.open('x') as stream, token_path.open('xb') as packed:
                for position, row in enumerate(pool.map(inspect, ((role,p,r) for p,r in enumerate(rows[role])))):
                    ids = row.pop('_native_ids'); start = row.pop('_target_start')
                    row['native'] = dict(offset=cursor, length=len(ids), target_start=start)
                    packed.write(np.asarray(ids, dtype='<u2').tobytes()); cursor += len(ids); targets += len(ids)-start
                    stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                    statistics[row['byte_family']] += 1; groups.add(row['group_id'])
                    if row['byte_family'] == 'rendered': rendered_texts.add(text_group(row['response']))
                    for name, field in [('word','word_input_pixels_sha256'), ('native','native_input_pixels_sha256'), ('stored','stored_image_sha256')]:
                        hashes[name].add(row[field])
                    if (position+1)%1000 == 0:
                        print(role, position+1, '/', len(rows[role]), flush=True)
                        atomic_json(OUTPUT/'progress.json', dict(role=role, reconstructed_original_words=position+1, planned=len(rows[role]), model_updates=0, goal_achieved=False))
            counts[role] = dict(examples=sum(statistics.values()), by_family=dict(statistics), native_context_tokens=cursor, native_response_EOS_targets=targets)
            fingerprints[role] = hashes; role_groups[role] = groups; rendered_groups[role] = rendered_texts
            for path in (stream_path, token_path): all_files[path.name] = file_hash(path)
    if (role_groups['train'] & role_groups['valid'] or rendered_groups['train'] & rendered_groups['valid']
            or any(fingerprints['train'][key] & fingerprints['valid'][key] for key in ('word','native','stored'))):
        raise ValueError('Original image/parent/rendered-text or actual new input pixels cross training and held roles')
    guard.check(full=True)
    atomic_json(OUTPUT/'manifest.json', dict(format=FORMAT, completed_at=datetime.now(timezone.utc).isoformat(),
        plan_sha256=file_hash(OUTPUT/'plan.json'), source_files_sha256=files, files_sha256=all_files, splits=counts,
        input_geometry=dict(word=[3,64,448], native=[3,448,448]), ctc_blank=256, ctc_frames=140,
        tokenizer_sha256=file_hash(ROOT/'checkpoints/whole-human-SFT-v1/initial/tokenizer.json'),
        all_original_rendered_font_seed_and_human_photo_box_transcripts_reconstructed=True,
        all_source_responses_and_EOS_preserved=True, original_source_parent_and_pixel_cross_role_overlap=0,
        annotation_mirror_equivalence_independently_verified=False, word_crops_not_generic_full_photo_reading=True,
        independent_readback_and_training_admission_pending=True, training_admitted=False, preparation_model_updates=0, goal_achieved=False))


if __name__ == '__main__': main()
