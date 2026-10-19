"""Reproduce a nonempty OCR label on actual constant-white public PDF pixels.

This source-only diagnostic never rewrites existing data or live training. It
certifies one specific defective positive pair, not every document or caption.
"""
from collections import Counter
import ctypes
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path

from PIL import Image,ImageOps
import pypdfium2 as pdfium

from slm_lab.document_images import rectangle_text
from slm_lab.diverse_images import pixel_hash
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.multimodal_pixel_protocols_v1 import observe_pixels

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/public-document-blank-supervision-CPU-v1'
DATA = ROOT/'data/document-images-v1'
IDENTIFIER = 'PLAW-117publ169-0-2'
PDF = ROOT/'data/image-source-cache-v1/PLAW-117publ169.pdf'
IMAGE = DATA/'images'/f'{IDENTIFIER}.png'
JOINT = ROOT/'data/extended-curriculum-v1/joint.train.jsonl'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def atomic(path,value):
    temporary = path.with_suffix(path.suffix+'.pending')
    with temporary.open('x') as f:
        f.write(json.dumps(value,indent=2)+'\n');f.flush();os.fsync(f.fileno())
    temporary.replace(path)


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh source-only bounded one-thread low-priority CPU diagnostic required')
    manifest = read(DATA/'manifest.json');proof = read(ROOT/'reports/full-multimodal-source-epoch-CPU-v1/complete.json')
    if sha(DATA/'train.jsonl') != manifest['files_sha256']['train.jsonl']:
        raise ValueError('Original complete document training rows must remain exact')
    with (DATA/'train.jsonl').open('rb') as f:
        found = [(i,raw,json.loads(raw)) for i,raw in enumerate(f) if json.loads(raw)['id'] == IDENTIFIER]
    if len(found) != 1:raise ValueError('Exactly one original document train row required')
    position,raw,row = found[0]
    with JOINT.open('rb') as f:f.seek(133670168);joint_raw = f.read(1309)
    joint = json.loads(joint_raw)
    if (row['split'] != 'train' or row['task'] != 'document_ocr' or row['response'] == ''
            or row['id'] != joint['id'] or row['response'] != joint['response']
            or sha(PDF) != row['download_sha256'] or sha(IMAGE) != row['stored_image_sha256']
            or hashlib.sha256(joint_raw).hexdigest() != '49ff6ab37993d0594a7fa0eb5524c45d73a807ce8fee6a9da5fbb59b03ea9064'):
        raise ValueError('Original public PDF/image/train target and committed joint-row bytes must remain exact')
    files = {str(p):sha(p) for p in [DATA/'manifest.json',DATA/'train.jsonl',PDF,IMAGE,JOINT,
        ROOT/'reports/full-multimodal-source-epoch-CPU-v1/complete.json',
        *python_sources(ROOT,[Path(__file__)])]}
    if files[str(JOINT)] != proof['source_files_sha256'][str(JOINT)]:
        raise ValueError('The current source epoch must bind this exact original joint source')
    OUT.mkdir();atomic(OUT/'plan.json',dict(source_files_sha256=files,original_source_id=IDENTIFIER,
        source_role='train',PDF_download_sha256=row['download_sha256'],original_document_JSONL_position=position,
        no_source_or_training_or_held_role_changes=True,training_admitted=False,goal_achieved=False))
    try:
        actual = observe_pixels(IMAGE)
        if (not actual['source_is_spatially_constant_RGB'] or actual['source_channel_extrema'] != [[255,255]]*3
                or actual['stored_image_sha256'] != row['stored_image_sha256']):
            raise ValueError('The actual stored OCR crop must independently be all-white RGB')
        with pdfium.PdfDocument(PDF) as document:
            page = document[row['page_index']];text = page.get_textpage();text.count_rects()
            bitmap = page.render(scale=row['render_scale']);parent = bitmap.to_pil().convert('RGB')
            try:
                rectangle_index = int(IDENTIFIER.rsplit('-',1)[1]);rectangle = text.get_rect(rectangle_index)
                characters = [(chr(pdfium.raw.FPDFText_GetUnicode(text.raw,i)),text.get_charbox(i)) for i in range(text.count_chars())]
                if rectangle_text(rectangle,characters) != row['response']:
                    raise ValueError('The entire original OCR target must reproduce from the same PDF rectangle and characters')
                width,height = page.get_size();left,bottom,right,top = rectangle
                box = [max(0,math.floor((left-.25)*parent.width/width)),max(0,math.floor((height-top-.25)*parent.height/height)),
                    min(parent.width,math.ceil((right+.25)*parent.width/width)),min(parent.height,math.ceil((height-bottom+.25)*parent.height/height))]
                crop = parent.crop(box);stored_shape = ImageOps.contain(crop,(768,768),Image.Resampling.LANCZOS)
                with Image.open(IMAGE) as original:original_RGB = original.convert('RGB')
                if (box != row['page_pixel_box'] or pixel_hash(parent) != row['parent_pixel_sha256']
                        or crop.getextrema() != ((255,255),)*3 or stored_shape.size != original_RGB.size
                        or stored_shape.tobytes() != original_RGB.tobytes()
                        or pixel_hash(stored_shape) != row['model_pixel_sha256']):
                    raise ValueError('The original PDF renderer, crop coordinates, source shape and every stored RGB pixel must reconstruct')
                glyphs = []
                for index,(char,bounds) in enumerate(characters):
                    if not (left-.1 <= bounds[0] and bottom-.1 <= bounds[1] and bounds[2] <= right+.1 and bounds[3] <= top+.1):continue
                    color = [ctypes.c_uint() for _ in range(4)]
                    success = pdfium.raw.FPDFText_GetFillColor(text.raw,index,*[ctypes.byref(v) for v in color])
                    obj = pdfium.raw.FPDFText_GetTextObject(text.raw,index)
                    mode = pdfium.raw.FPDFTextObj_GetTextRenderMode(obj)
                    glyphs.append(dict(original_character_position=index,unicode=ord(char),PDF_character_box=list(bounds),
                        fill_color_query_succeeded=bool(success),RGBA=[v.value for v in color],text_render_mode=int(mode)))
                if not glyphs or any(not g['fill_color_query_succeeded'] or g['RGBA'] != [255,255,255,255]
                                     or g['text_render_mode'] != pdfium.raw.FPDF_TEXTRENDERMODE_FILL for g in glyphs):
                    raise ValueError('Every actual target character must independently use opaque white PDF fill')
                source = dict(actual_page_size=list(page.get_size()),actual_rotation=page.get_rotation(),
                    original_PDF_rectangle_index=rectangle_index,actual_PDF_rectangle=list(rectangle),
                    actual_rendered_page_dimensions=list(parent.size),actual_original_crop_box=box,
                    actual_raw_crop_dimensions=list(crop.size),actual_saved_crop_dimensions=list(stored_shape.size),
                    entire_saved_crop_RGB_matches_original=True,original_parent_normalized_RGB224_sha256=pixel_hash(parent),
                    actual_target_characters=len(glyphs),actual_original_target=row['response'],glyphs=glyphs)
            finally:bitmap.close();text.close();page.close()
        if any(sha(p) != h for p,h in files.items()):raise ValueError('All original public source and runtime bytes must remain unchanged')
        result = dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
            original_source_id=IDENTIFIER,original_document_JSONL_position=position,original_document_line_sha256=hashlib.sha256(raw).hexdigest(),
            original_joint_row=dict(path=str(JOINT),offset=133670168,bytes=1309,sha256=hashlib.sha256(joint_raw).hexdigest()),
            original_source_role='train',actual_source_pixels=actual,actual_source_PDF=source,
            actual_independent_nonempty_OCR_target_on_constant_white_image=True,
            observed_cause='Opaque white PDF text was copied from the text layer as a target despite having no visible ink on the white rendered region',
            unchanged_positive_training_pair_is_semantically_vetoed_for_future_admission=True,
            exact_pixel_checks_do_not_certify_visible_target_alignment=True,
            no_all_document_quality_or_model_accuracy_claim=True,existing_training_and_source_files_unmodified=True,
            proposed_full_source_visual_epoch_requires_explicit_new_qualification_of_this_pair_before_admission=True,
            training_admitted=False,new_production_optimizer_updates=0,goal_achieved=False)
        atomic(OUT/'complete.json',result)
        atomic(OUT/'semantic-veto.json',dict(source_id=IDENTIFIER,original_source_role='train',
            original_stored_image_sha256=row['stored_image_sha256'],original_response_UTF8_sha256=hashlib.sha256(row['response'].encode()).hexdigest(),
            original_joint_row_sha256=hashlib.sha256(joint_raw).hexdigest(),evidence=str(OUT/'complete.json'),
            evidence_sha256=sha(OUT/'complete.json'),veto_unchanged_positive_for_future_admission=True,
            existing_data_or_live_producer_not_modified=True,automatic_source_exclusion_or_relabeling=False))
        print(json.dumps(dict(complete=True,original_source_id=IDENTIFIER,all_white_target_characters=len(glyphs),
            semantic_veto_for_future_positive_admission=True,training_admitted=False)),flush=True)
    except BaseException as error:
        atomic(OUT/'failure.json',dict(error=repr(error),automatic_retry=False,training_admitted=False,goal_achieved=False));raise


if __name__ == '__main__':main()
