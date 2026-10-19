"""Fresh whole-source/native/pixel qualification of four machine corrections.

Admits only these source-bound data examples. It does not admit or start the
full visual epoch, change a current recipe or establish improved model quality.
"""
import ast
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image,ImageOps
from tokenizers import Tokenizer

from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
from slm_lab.codenet_replay_v1 import training_role
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.training import letterbox_box

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT/'data/public-document-visible-corrections-candidate-v1'
OUT = ROOT/'data/public-document-visible-corrections-admitted-v1'
REPORT = ROOT/'reports/public-document-visible-corrections-v1-readback-admission'
TOKENIZER = ROOT/'data/tokenizer-corpus/tokenizer.json'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def atomic(path,value):
    pending = path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    pending.replace(path)


def main():
    if (OUT.exists() or REPORT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh one-thread low-priority CPU correction qualification required')
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0))
    proof = read(CANDIDATE/'complete.json')
    if (not proof.get('complete') or proof['training_admitted'] or proof['exact_original_prepared_views'] != 4
            or proof['actual_positive_response_EOS_targets'] != 20 or proof['actual_complete_original_negative_targets'] != 80):
        raise ValueError('All four complete machine correction candidates and intact original negatives required')
    source = CANDIDATE/'all4-complete-correction-preference-candidates.jsonl'
    if sha(source) != proof['candidate_file_sha256']:raise ValueError('Every candidate byte must remain exact')
    files = dict(proof['source_files_sha256'])
    files.update({str(p):sha(p) for p in [CANDIDATE/'complete.json',source,TOKENIZER,*python_sources(ROOT,[Path(__file__)])]})
    if any(sha(p) != h for p,h in files.items()):raise ValueError('Every original source, source audit, independent pixel readback and candidate byte must match')
    wait_for_reserve(ROOT,minimum_bytes=40*1024**3)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    if tokenizer.truncation is not None or tokenizer.padding is not None:raise ValueError('Exact full native own tokenizer with no hidden clipping/padding required')
    REPORT.mkdir();atomic(REPORT/'plan.json',dict(source_files_sha256=files,candidates=4,
        qualification_scope='data_only_machine_SFT_and_preference_examples',new_neural_training_or_epoch_admission=False,goal_achieved=False))
    try:
        examples = [json.loads(line) for line in source.open()];admitted = [];observations = []
        for item in examples:
            view = item['original_prepared_native_view'];reference = view['original_source_reference']
            if item['source_role'] != 'train' or item['training_admitted'] or view['source_role'] != 'train' or view['family'] != 'replay:image_reading':
                raise ValueError('Only original prepared train views may receive source-derived machine corrections')
            with Path(reference['path']).open('rb') as stream:stream.seek(reference['offset']);raw = stream.read(reference['bytes'])
            original = json.loads(raw);training_role(original)
            image_path = Path(original['image']);image_path = (image_path if image_path.is_absolute() else Path(reference['path']).parent/image_path).resolve(strict=True)
            original = {**original,'image':str(image_path)}
            corrected = item['corrected_row'];training_role(corrected)
            expected = {**original,'response':'No visible text.',
                'annotation_origin':'Programmatic visible-region adjudication: every requested original RGB pixel is white; independently checked, not human transcription',
                'label_origin':'Machine source-derived correction; no human label, neural teacher, or sampled student answer'}
            if (hashlib.sha256(raw).hexdigest() != reference['row_sha256'] or original['id'] != item['source_id']
                    or original['response'] != item['original_response'] or corrected != expected
                    or str(image_path) != view['image_path'] or sha(image_path) != view['original_stored_image_sha256']):
                raise ValueError('Only the explicit response and its honest annotation-origin fields may change; every source/input must stay exact')
            with Image.open(image_path) as opened:image = opened.convert('RGB')
            padded = ImageOps.pad(image,(448,448),method=Image.Resampling.BICUBIC,color=(128,128,128))
            if hashlib.sha256(padded.tobytes()).hexdigest() != view['original_RGB448_metadata_sha256']:
                raise ValueError('Every actual native RGB448 input must exactly match its original committed protocol')
            if original['task'] == 'document_ocr':
                if image.getextrema() != ((255,255),)*3:raise ValueError('Every original whole crop pixel must independently be white')
                native_region = dict(kind='whole_source_crop_with_only_gray_native_padding',source_white=True)
            elif original['task'] == 'document_region_ocr':
                l,t,r,b = original['normalized_box'];source_box = [math.floor(l*image.width),math.floor(t*image.height),math.ceil(r*image.width),math.ceil(b*image.height)]
                if image.crop(source_box).getextrema() != ((255,255),)*3:raise ValueError('The complete original requested source region must independently contain no visible ink')
                match = re.search(r'box (\[[0-9., ]+\])',item['native_prompt'])
                if match is None:raise ValueError('The actual native prompt must retain its original padded-image coordinates')
                native_box = ast.literal_eval(match.group(1))
                if native_box != letterbox_box(original['normalized_box'],image.size,448):raise ValueError('Actual native prompt coordinates must reproduce the original source-to-pad transform')
                native_pixels = [math.floor(native_box[0]*448),math.floor(native_box[1]*448),math.ceil(native_box[2]*448),math.ceil(native_box[3]*448)]
                if padded.crop(native_pixels).getextrema() != ((255,255),)*3:raise ValueError('Every actual requested native padded pixel must also independently be white')
                native_region = dict(kind='boxed_native_padded_region',source_box=source_box,native_box=native_box,native_pixel_box=native_pixels,source_and_native_regions_white=True)
            else:raise ValueError('No unrelated source/task may enter the corrective overlay')
            for field,response in [('positive_native','No visible text.'),('complete_original_negative_native',original['response'])]:
                native = item[field];ids = native['input_ids'];labels = native['labels'];start = native['target_start']
                prefix = [1]+tokenizer.encode(item['native_prompt'],add_special_tokens=False).ids
                expected_ids = prefix+tokenizer.encode(response,add_special_tokens=False).ids+[2]
                if (ids != expected_ids or ids[0] != 1 or ids[-1] != 2 or not 3 <= len(ids) <= 4096
                        or start != len(prefix) or start != view['target_start'] or labels != [-100]*start+ids[start:]
                        or any(not 4 <= v < 32768 for v in ids[1:-1]) or native['response_EOS_targets'] != len(ids)-start
                        or hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest() != native['input_ids_sha256']
                        or tokenizer.decode(ids[1:start],skip_special_tokens=False) != item['native_prompt']
                        or tokenizer.decode(ids[start:-1],skip_special_tokens=False) != response):
                    raise ValueError('Both full native prompt/response/EOS branches and exact loss masks must independently rebuild')
            if item['complete_original_negative_native']['input_ids_sha256'] != view['input_ids_sha256']:
                raise ValueError('The complete original erroneous positive must be retained intact as the negative branch')
            admitted.append({**item,'training_admitted':True,'data_admission_scope':'machine_source_derived_correction_and_preference_only'})
            observations.append(dict(source_id=item['source_id'],original_family_position=view['original_local_position'],native_region=native_region,
                positive_targets=item['positive_native']['response_EOS_targets'],whole_original_negative_targets=item['complete_original_negative_native']['response_EOS_targets'],
                all_original_prompts_images_positions_roles_and_whole_negative_retained=True,whole_native_masks_and_genuine_EOS_verified=True))
        if len(admitted) != 4 or len({r['source_id'] for r in admitted}) != 4 or sum(r['positive_targets'] for r in observations) != 20 or sum(r['whole_original_negative_targets'] for r in observations) != 80:
            raise ValueError('All four prepared original occurrences,20corrective and80complete original negative targets must remain')
        if any(sha(p) != h for p,h in files.items()):raise ValueError('Every original source/readback/candidate/runtime byte must remain exact before data admission')
        OUT.mkdir();destination = OUT/'all4-complete-corrections-and-preferences.jsonl'
        with destination.open('x') as stream:
            for row in admitted:stream.write(json.dumps(row,ensure_ascii=True)+'\n')
            stream.flush();os.fsync(stream.fileno())
        result = dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,actual_observations=observations,
            exact_admitted_original_views=4,positive_response_EOS_targets=20,complete_original_negative_targets=80,
            original_unprepared_or_held_views_not_readmitted=True,all_genuine_native_BOS_prompt_response_EOS_masks_rebuilt=True,
            actual_padded448_source_regions_rechecked=True,admitted_file=str(destination),admitted_file_sha256=sha(destination),
            source_training_roles_unchanged=True,machine_supervision_only=True,not_human_feedback_or_on_policy_RL=True,
            full_302627_view_4733_update_epoch_not_admitted_by_this_data_receipt=True,
            no_existing_source_or_neural_worker_or_recipe_changed=True,
            training_admitted=True,model_has_not_consumed_these_new_targets=True,new_production_optimizer_updates=0,goal_achieved=False)
        atomic(REPORT/'complete.json',result)
        atomic(OUT/'admission.json',dict(training_admitted=True,admitted_role='train',qualification_scope='data_only_machine_correction_and_preference_examples',
            independent_readback=str(REPORT/'complete.json'),independent_readback_sha256=sha(REPORT/'complete.json'),
            exact_original_prepared_views=4,positive_response_EOS_targets=20,complete_original_negative_targets=80,
            data_file_sha256=sha(destination),full_vision_epoch_and_new_neural_job_not_admitted=True,goal_achieved=False))
        print(json.dumps(dict(complete=True,actual_admitted_views=4,positive_targets=20,complete_original_negative_targets=80,new_neural_training=False)),flush=True)
    except BaseException as error:
        atomic(REPORT/'failure.json',dict(error=repr(error),automatic_retry=False,training_admitted=False,goal_achieved=False));raise


if __name__ == '__main__':main()
