"""Independent full original word/box/render/pixel/BPE readback and admission."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageOps
from tokenizers import Tokenizer

from slm_lab.byte_word_reader import load_rows
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, word_crops
from slm_lab.rendered_text import render
from slm_lab.shared_word_data_v1 import SharedWordData
from slm_lab.whole_human_alignment_v2 import WholeResponseEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/shared-byte-words-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('Source admission reader must explicitly hide GPUs')
    if (DATA/'independent-source-readback.json').exists() or (DATA/'admission.json').exists():
        raise ValueError('Preserve previous completed full-source word reviews')
    manifest = json.loads((DATA/'manifest.json').read_text()); plan = json.loads((DATA/'plan.json').read_text())
    files = {**manifest['source_files_sha256'], str(DATA/'manifest.json'):file_hash(DATA/'manifest.json'),
             str(DATA/'plan.json'):file_hash(DATA/'plan.json'), str(Path(__file__).resolve()):file_hash(__file__)}
    guard = FileBindings(files); guard.check(full=True)
    rows, original_manifest = load_rows(ROOT/'data/byte-word-reader-v1')
    originals = {role:SharedWordData(DATA, role, 'audit') for role in ('train','valid')}
    if any(len(originals[role]) != len(rows[role]) for role in rows):
        raise ValueError('Every original selected source word must remain in its original role')
    annotations = {role:json.loads((ROOT/'data/image-source-cache-v1'/f'TextOCR_0.1_{part}.json').read_text())
                   for role,part in [('train','train'),('valid','val')]}
    config = SimpleNamespace(**json.loads((ROOT/'configs/model.json').read_text()))
    tokenizer = Tokenizer.from_file(str(ROOT/'checkpoints/whole-human-SFT-v1/initial/tokenizer.json'))
    encoder = WholeResponseEncoder(tokenizer, config)
    render_config = json.loads((ROOT/'data/rendered-text-v1/configuration.json').read_text())
    font_paths = {name:next(p for p in Path('/usr/share/fonts').rglob(name) if file_hash(p) == digest)
                  for name,digest in render_config['fonts'].items()}

    @lru_cache(maxsize=16)
    def parent(key, subset, digest):
        path = ROOT/'data/image-source-cache-v1/source-images'/subset/(key+'.jpg')
        if file_hash(path) != digest: raise ValueError('Pinned original source photograph changed during independent readback')
        with Image.open(path) as image: return image.convert('RGB')

    def inspect(item):
        role, position = item; source = rows[role][position]; row = originals[role].rows[position]
        digest = hashlib.sha256(json.dumps(source,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if (any(row[key] != value for key,value in source.items()) or row['original_selected_position'] != position
                or row['original_runtime_row_sha256'] != digest):
            raise ValueError('Every original retained word source field and position must reconstruct')
        path = Path(source['image'])
        if file_hash(path) != source['stored_image_sha256']: raise ValueError('Original word crop file changed')
        with Image.open(path) as image: image = image.convert('RGB')
        if source['byte_family'] == 'rendered':
            expected, _ = render(source['response'],font_paths[source['font']],render_config['seed']+source['attempt'])
        else:
            key = source['source_image_id']; original = annotations[role]; annotation = original['anns'][source['original_word_id']]
            metadata = original['imgs'][key]
            if (str(annotation['image_id']) != key or annotation.get('utf8_string','').strip() != source['response']
                    or annotation['bbox'] != source['original_bbox'] or source['split'] != role):
                raise ValueError('Every original human word/box/source role must reconstruct')
            crop = word_crops(parent(key,source['source_subset'],source['download_sha256']),metadata['width'],metadata['height'],[annotation],1)
            if len(crop) != 1 or crop[0][0] != source['original_word_id']: raise ValueError('Whole original word crop cannot be silently omitted')
            side = 448 if source['byte_family'] == 'additional_real_word' else 768
            expected = ImageOps.contain(crop[0][2].convert('RGB'),(side,side),Image.Resampling.LANCZOS)
        if expected.size != image.size or expected.tobytes() != image.tobytes():
            raise ValueError('Independent original font/photo pixels differ from the stored source crop')
        # Reconstruct transforms directly, independently of image_arrays().
        word = np.ascontiguousarray((np.asarray(image.resize((448,64),Image.Resampling.BICUBIC),dtype=np.float32)/127.5-1).transpose(2,0,1))
        native = ImageOps.pad(image,(448,448),method=Image.Resampling.BICUBIC,color=(128,128,128))
        native = np.ascontiguousarray((np.asarray(native,dtype=np.float32)/127.5-1).transpose(2,0,1))
        if (hashlib.sha256(word.tobytes()).hexdigest() != row['word_input_pixels_sha256']
                or hashlib.sha256(native.tobytes()).hexdigest() != row['native_input_pixels_sha256']
                or hashlib.sha256(image.tobytes()).hexdigest() != row['source_RGB_pixels_sha256']):
            raise ValueError('All actual source RGB/shared-word/native pixels must independently reconstruct')
        ids, start = encoder.sequence(source,'response'); item = row['native']
        packed = originals[role].tokens[item['offset']:item['offset']+item['length']].astype(np.int64).tolist()
        target = list(source['response'].encode('utf-8')); required = len(target)+sum(a == b for a,b in zip(target,target[1:]))
        if (ids != packed or item['target_start'] != start or row['native_byte_targets'] != target
                or row['minimum_CTC_steps'] != required or required > 140 or ids[-1] != 2):
            raise ValueError('Every whole native BPE/BOS/prompt/response/EOS target and actual byte alignment must reconstruct')
        return source['byte_family'],len(ids),len(ids)-start,len(target),required

    results = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for role in ('valid','train'):
            families = Counter(); contexts = targets = byte_targets = maximum = 0
            for position,(family,length,response,byte_count,steps) in enumerate(pool.map(inspect,((role,p) for p in range(len(rows[role]))))):
                families[family] += 1; contexts += length; targets += response; byte_targets += byte_count; maximum = max(maximum,steps)
                if (position+1)%1000 == 0:
                    print('independent_whole_word_readback',role,position+1,'/',len(rows[role]),flush=True)
                    atomic_json(DATA/'review-progress.json',dict(role=role,original_rows_checked=position+1,model_updates=0,goal_achieved=False))
            if manifest['splits'][role] != dict(examples=sum(families.values()),by_family=dict(families),native_context_tokens=contexts,native_response_EOS_targets=targets):
                raise ValueError('Every original source-family/context/response exposure total must reconstruct')
            results[role] = dict(original_words=sum(families.values()),source_families=dict(families),native_context_tokens=contexts,
                native_response_EOS_targets=targets,whole_UTF8_byte_targets=byte_targets,maximum_required_CTC_steps=maximum)
    train = originals['train'].rows; valid = originals['valid'].rows
    for field in ('group_id','stored_image_sha256','word_input_pixels_sha256','native_input_pixels_sha256'):
        if {r[field] for r in train} & {r[field] for r in valid}: raise ValueError('Original role/pixel leakage in independently read word source')
    guard.check(full=True)
    atomic_json(DATA/'independent-source-readback.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
        reader_sha256=file_hash(__file__),manifest_sha256=file_hash(DATA/'manifest.json'),full_original_source_roles=results,
        all_original_source_rendered_and_human_photo_crop_pixels_reconstructed=True,
        all_full_new_pixel_transforms_and_native_BPE_BOS_response_EOS_masks_reconstructed=True,
        original_parent_and_byte_pixel_cross_role_overlap=0,annotation_mirror_equivalence_independently_verified=False,
        semantic_generalization_or_human_transcription_accuracy_not_certified=True,word_crops_not_full_photo_reading=True,
        model_updates=0,goal_achieved=False))
    atomic_json(DATA/'admission.json',dict(training_admitted=True,admitted_role='train',
        manifest_sha256=file_hash(DATA/'manifest.json'),independent_readback_sha256=file_hash(DATA/'independent-source-readback.json'),
        held_original_gradients_forbidden=True,native_CTC_bytes='0..255; blank256; complete transcripts without truncation',
        annotation_mirror_equivalence_independently_verified=False,word_crops_not_full_photo_reading=True,model_updates=0,goal_achieved=False))


if __name__ == '__main__': main()
