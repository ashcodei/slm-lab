"""One-to-one source-derived positive corrections with intact original negatives.

These are machine-supervised data candidates, never human feedback, on-policy
samples or an open-ended OCR/RL oracle. No existing source/recipe is rewritten.
"""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
from slm_lab.codenet_replay_v1 import FullReplayEncoder,training_role
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.model import ModelConfig

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'reports/full-public-document-visible-labels-CPU-v1'
READER = ROOT/'reports/full-public-document-visible-labels-v1-readback/complete.json'
OUT = ROOT/'data/public-document-visible-corrections-candidate-v1'
POSITIVE = 'No visible text.'
TOKENIZER = ROOT/'data/tokenizer-corpus/tokenizer.json'
CONFIG = ROOT/'data/public-reasoning-curriculum-v1/model-config.json'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):value.update(block)
    return value.hexdigest()


def token_sha(ids):return hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()


def atomic(path,value):
    pending = path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    pending.replace(path)


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh bounded low-priority source-only candidate builder required')
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0))
    proof,reader,vetoes = read(SOURCE/'complete.json'),read(READER),read(SOURCE/'semantic-vetoes.json')
    if (not proof.get('complete') or not reader.get('complete') or proof['training_admitted'] or reader['training_admitted']
            or proof['actual_counts']['original_rows'] != 1433 or reader['actual_counts']['original_rows'] != 1433
            or reader['actual_counts']['confirmed_white_requested_region_vetoes'] != 8
            or proof['actual_confirmed_veto_prepared_native_views'] != 4
            or not reader['actual_veto_list_matches_source_audit'] or vetoes['evidence_sha256'] != sha(SOURCE/'complete.json')):
        raise ValueError('Complete original source/PDF and independent requested-pixel readback required')
    files = {**proof['source_files_sha256'],**reader['source_files_sha256']}
    paths = [SOURCE/'complete.json',SOURCE/'semantic-vetoes.json',READER,TOKENIZER,CONFIG,*python_sources(ROOT,[Path(__file__)])]
    files.update({str(p):sha(p) for p in paths})
    if any(sha(p) != digest for p,digest in files.items()):raise ValueError('All original source and independent visibility evidence must remain exact')
    wait_for_reserve(ROOT,minimum_bytes=40*1024**3)
    tokenizer = Tokenizer.from_file(str(TOKENIZER));config = ModelConfig(**read(CONFIG))
    if tokenizer.truncation is not None or tokenizer.padding is not None:raise ValueError('No hidden tokenizer clipping or padding permitted')
    encoder = FullReplayEncoder(tokenizer,config,ROOT,448);OUT.mkdir()
    atomic(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        original_source_audit_rows=1433,confirmed_original_source_vetoes=8,exact_prepared_views_to_correct=4,
        fixed_machine_positive_response=POSITIVE,original_wrong_response_retained_whole_as_negative=True,
        every_original_prepared_source_position_retained=True,excluded_original_views_not_readmitted=True,
        source_pixel_and_prompt_and_role_unchanged=True,training_admitted=False,goal_achieved=False))
    try:
        examples = [];positive_targets = negative_targets = 0
        for entry in vetoes['unchanged_positive_source_vetoes']:
            for native in entry['prepared_native_views']:
                if native['family'] != 'replay:image_reading' or native['source_role'] != 'train':raise ValueError('Only exact originally prepared train views may receive a candidate correction')
                reference = native['original_source_reference']
                with Path(reference['path']).open('rb') as stream:
                    stream.seek(reference['offset']);raw = stream.read(reference['bytes'])
                row = json.loads(raw);training_role(row)
                image = Path(row['image']);image = (image if image.is_absolute() else Path(reference['path']).parent/image).resolve(strict=True)
                if (hashlib.sha256(raw).hexdigest() != reference['row_sha256'] or row['id'] != entry['source_id']
                        or str(image) != native['image_path'] or sha(image) != native['original_stored_image_sha256']
                        or row['response'] == POSITIVE or hashlib.sha256(row['response'].encode()).hexdigest() != native['response_utf8_sha256']):
                    raise ValueError('Complete exact source/image/old negative target bindings required')
                original = {**row,'image':str(image)}
                corrected = {**original,'response':POSITIVE,
                    'annotation_origin':'Programmatic visible-region adjudication: every requested original RGB pixel is white; independently checked, not human transcription',
                    'label_origin':'Machine source-derived correction; no human label, neural teacher, or sampled student answer'}
                old_ids,old_labels = encoder.sequence(original,'response');new_ids,new_labels = encoder.sequence(corrected,'response')
                start = next(i for i,t in enumerate(old_labels) if t != -100);new_start = next(i for i,t in enumerate(new_labels) if t != -100)
                if (token_sha(old_ids) != native['input_ids_sha256'] or len(old_ids) != native['complete_context_tokens']
                        or start != native['target_start'] or len(old_ids)-start != native['response_EOS_targets']
                        or start != new_start or old_ids[:start] != new_ids[:start]
                        or any(ids[0] != 1 or ids[-1] != 2 or len(ids) > 4096 or any(v < 4 or v >= config.vocab_size for v in ids[1:-1]) for ids in [old_ids,new_ids])
                        or old_labels[:start] != [-100]*start or new_labels[:start] != [-100]*start
                        or old_labels[start:] != old_ids[start:] or new_labels[start:] != new_ids[start:]
                        or tokenizer.decode(new_ids[start:-1],skip_special_tokens=False) != POSITIVE
                        or tokenizer.decode(old_ids[start:-1],skip_special_tokens=False) != row['response']):
                    raise ValueError('Both whole own native prompt/response/EOS branches must reconstruct without clipping or wrong masking')
                positive_targets += len(new_ids)-start;negative_targets += len(old_ids)-start
                examples.append(dict(id=row['id']+':visible-region-correction-v1',source_id=row['id'],source_role='train',
                    original_prepared_native_view=native,original_document_JSONL_position=entry['original_JSONL_position'],
                    original_document_line_sha256=entry['original_line_sha256'],image_path=str(image),
                    native_prompt=encoder.prompt(original),original_response=row['response'],corrected_response=POSITIVE,
                    original_annotation_origin=row.get('annotation_origin'),corrected_row=corrected,
                    positive_native=dict(input_ids=new_ids,labels=new_labels,target_start=start,input_ids_sha256=token_sha(new_ids),response_EOS_targets=len(new_ids)-start),
                    complete_original_negative_native=dict(input_ids=old_ids,labels=old_labels,target_start=start,input_ids_sha256=token_sha(old_ids),response_EOS_targets=len(old_ids)-start),
                    source_evidence=str(SOURCE/'complete.json'),independent_pixel_readback=str(READER),
                    supervision='machine-derived visible-region correction and preference candidate; not human feedback or on-policy RL',training_admitted=False))
        if len(examples) != 4 or len({r['source_id'] for r in examples}) != 4 or negative_targets != 80:
            raise ValueError('Every exact originally prepared defective view and its80complete old targets must be retained')
        output = OUT/'all4-complete-correction-preference-candidates.jsonl'
        with output.open('x') as stream:
            for row in examples:stream.write(json.dumps(row,ensure_ascii=True)+'\n')
            stream.flush();os.fsync(stream.fileno())
        if any(sha(p) != digest for p,digest in files.items()):raise ValueError('All original source/readback/native code bytes must remain unchanged')
        atomic(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
            exact_original_prepared_views=4,actual_positive_response_EOS_targets=positive_targets,actual_complete_original_negative_targets=negative_targets,
            actual_original_family_positions=[r['original_prepared_native_view']['original_local_position'] for r in examples],
            candidate_file_sha256=sha(output),every_original_prompt_image_role_position_and_whole_negative_preserved=True,
            no_excluded_original_view_readmitted=True,original_inputs_and_training_recipes_unmodified=True,
            no_neural_teacher_or_human_label_or_student_sample=True,fresh_full_readback_and_qualification_still_required=True,
            full_vision_epoch_pixel_capacity_and_training_and_quality_pending=True,
            training_admitted=False,new_production_optimizer_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=True,actual_candidates=4,positive_targets=positive_targets,complete_original_negative_targets=negative_targets,training_admitted=False)),flush=True)
    except BaseException as error:
        atomic(OUT/'failure.json',dict(error=repr(error),automatic_retry=False,training_admitted=False,goal_achieved=False));raise


if __name__ == '__main__':main()
