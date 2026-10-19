"""Reconstruct all original train PDF labels/pixels and inspect requested ink.

No held source file is opened and no existing data/recipe is rewritten. The
all-white rule is a narrow positive-label veto; other labels remain semantically
uncertified. Original1433 train rows include51 not in the prepared visual epoch;
this audit does not re-admit those excluded rows.
"""
from collections import Counter,defaultdict
from contextlib import ExitStack
import ctypes
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil

from PIL import Image,ImageOps
import pypdfium2 as pdfium

from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
from slm_lab.document_images import rectangle_text,SOURCES
from slm_lab.document_requested_pixel_visibility_v1 import requested_pixels
from slm_lab.diverse_images import pixel_hash
from slm_lab.local_source_closure_v2 import python_sources

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/document-images-v1'
METADATA = ROOT/'reports/full-multimodal-source-epoch-CPU-v1/all302627-original-native-views.jsonl'
OUT = ROOT/'reports/full-public-document-visible-labels-CPU-v1'
TESTS = ROOT/'reports/document-requested-pixel-visibility-v1-tests.log'
KNOWN = ROOT/'reports/public-document-blank-region-CPU-v1/complete.json'
RESERVE,MAX_OUTPUT = 40*1024**3,64*1024**2
WORKING,OTHER_HEADROOM = 2*1024**3,64*1024**3


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def state(path):
    value = Path(path).stat()
    return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)


def atomic(path,value):
    pending = path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as f:
        f.write(json.dumps(value,indent=2)+'\n');f.flush();os.fsync(f.fileno())
    pending.replace(path)


def memory():
    values = {k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
    return values['anon']+values['shmem']+values['kernel'],int(Path('/sys/fs/cgroup/memory.max').read_text())


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
            or '5 passed' not in TESTS.read_text()):
        raise ValueError('Fresh tested one-thread CUDA-hidden low-priority original-source audit required')
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0))
    active,limit = memory()
    if limit-active < WORKING+OTHER_HEADROOM:raise ValueError('Original nonneural working allowance and other-process memory headroom required')
    manifest = read(DATA/'manifest.json');known = read(KNOWN)
    epoch = read(ROOT/'reports/full-multimodal-source-epoch-CPU-v1/complete.json')
    if (not known.get('complete') or len(known['actual_bad_original_positive_views']) != 2
            or not epoch.get('complete') or epoch['training_admitted']
            or sha(DATA/'train.jsonl') != manifest['files_sha256']['train.jsonl']):
        raise ValueError('Original whole train rows, prepared native inventory and actual known source defect required')
    with (DATA/'train.jsonl').open('rb') as f:original = [(raw,json.loads(raw)) for raw in f]
    rows = [row for _,row in original];groups = defaultdict(list);images = {};PDFs = {}
    for position,row in enumerate(rows):
        path = (DATA/row['image']).resolve(strict=True)
        if (row['split'] != 'train' or row['parent_document_id'] not in SOURCES['train'] or not path.is_relative_to(DATA)
                or row['task'] not in ['document_ocr','document_region_ocr']):
            raise ValueError('Only the complete original document training rows/pages may be inspected')
        if str(path) in images and images[str(path)] != row['stored_image_sha256']:raise ValueError('Conflicting original physical image expectations')
        images[str(path)] = row['stored_image_sha256']
        pdf = ROOT/'data/image-source-cache-v1'/(row['parent_document_id']+'.pdf')
        if str(pdf) in PDFs and PDFs[str(pdf)] != row['download_sha256']:raise ValueError('Conflicting original public PDF expectations')
        PDFs[str(pdf)] = row['download_sha256'];groups[(row['parent_document_id'],row['page_index'])].append(position)
    if len(rows) != 1433 or len(groups) != 161 or len(PDFs) != 6 or len({row['id'] for row in rows}) != 1433:
        raise ValueError('Every original1433 row,161 page and6 training PDFs required')
    files = {**images,**PDFs,str(DATA/'train.jsonl'):manifest['files_sha256']['train.jsonl'],str(METADATA):epoch['source_metadata_rows_sha256']}
    paths = [DATA/'manifest.json',KNOWN,TESTS,ROOT/'reports/full-multimodal-source-epoch-CPU-v1/complete.json',
        *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_document_requested_pixel_visibility_v1.py'])]
    files.update({str(p):sha(p) for p in paths});states = {p:state(p) for p in files};waits = Counter()
    def guard():
        if os.sched_getscheduler(0) != os.SCHED_IDLE or any(state(p) != s for p,s in states.items()):
            raise ValueError('Only this new idle-priority worker and unchanged original source files are allowed')
    def checked():
        result = wait_for_reserve(ROOT,minimum_bytes=RESERVE,guard=guard)
        waits['seconds'] += result['waited_seconds'];waits['failed_space_observations'] += result['failed_space_observations']
        current,maximum = memory()
        resident = next(int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
        if maximum-current < OTHER_HEADROOM or resident > WORKING:raise ValueError('The original bounded source-audit memory allowance must remain available')
    checked();OUT.mkdir();owner = dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
        original_train_rows=1433,original_train_PDF_pages=161,original_train_PDF_files=6,prepared_native_views_expected=1382,
        initial_original_source_binding_verification_pending=True,scheduler='SCHED_IDLE',nice=os.nice(0),CUDA_hidden=True,
        CPU_working_allowance=WORKING,other_process_memory_headroom=OTHER_HEADROOM,parent_address_space_ceiling=16*1024**3,
        minimum_free_bytes=RESERVE,maximum_output_bytes=MAX_OUTPUT,bounded_disk_wait_maximum_seconds=1800,
        PDF_fill_color_alone_is_not_a_rejection_rule=True,all_other_transcript_quality_remains_uncertified=True,
        no_existing_source_or_recipe_or_neural_worker_change=True,training_admitted=False,goal_achieved=False))
    try:
        resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
        if any(sha(p) != h for p,h in files.items()):raise ValueError('Every original cached PDF/image/native metadata and runtime byte must match before inspection')
        identifiers = {row['id'] for row in rows};native = defaultdict(list)
        with METADATA.open('rb') as f:
            for line in f:
                if b'"source_group": "govinfo:' not in line:continue
                value = json.loads(line)
                if value['source_id'] not in identifiers:raise ValueError('No unknown GovInfo source may silently enter this original population')
                if value['family'] != 'replay:image_reading' or value['source_role'] != 'train' or value['training_admitted']:
                    raise ValueError('Original prepared native document views must retain their exact train role and unadmitted status')
                native[value['source_id']].append(value)
        if len(native) != 1382 or any(len(v) != 1 for v in native.values()):raise ValueError('Exactly all1382 prepared native document views must be mapped without readmitting excluded originals')
        output = OUT/'all1433-original-train-label-observations.jsonl';counts = Counter();vetoes = [];completed = pages = 0
        with ExitStack() as stack:
            docs = {package:stack.enter_context(pdfium.PdfDocument(ROOT/'data/image-source-cache-v1'/(package+'.pdf'))) for package in SOURCES['train']}
            destination = stack.enter_context(output.open('x'))
            for (package,page_index),positions in groups.items():
                checked();page = docs[package][page_index];text = page.get_textpage();text.count_rects();bitmap = page.render(scale=1.5)
                try:
                    parent = bitmap.to_pil().convert('RGB');parent_hash = pixel_hash(parent);width,height = page.get_size()
                    characters = [(chr(pdfium.raw.FPDFText_GetUnicode(text.raw,j)),text.get_charbox(j)) for j in range(text.count_chars())]
                    glyph_cache = {}
                    for position in positions:
                        raw,row = original[position]
                        if position != completed:raise ValueError('Every original disposition must retain its original JSONL position/order')
                        rectangle_index = int(row['id'].rsplit('-',2)[1] if row['task'] == 'document_region_ocr' else row['id'].rsplit('-',1)[1])
                        rectangle = text.get_rect(rectangle_index);left,bottom,right,top = rectangle
                        actual_text = rectangle_text(rectangle,characters)
                        box = [max(0,math.floor((left-.25)*parent.width/width)),max(0,math.floor((height-top-.25)*parent.height/height)),
                            min(parent.width,math.ceil((right+.25)*parent.width/width)),min(parent.height,math.ceil((height-bottom+.25)*parent.height/height))]
                        with Image.open(DATA/row['image']) as source:actual = source.convert('RGB')
                        expected = parent.crop(box) if row['task'] == 'document_ocr' else parent
                        expected = ImageOps.contain(expected,(768,768),Image.Resampling.LANCZOS)
                        expected_box = box if row['task'] == 'document_ocr' else [round(v,4) for v in [box[0]/parent.width,box[1]/parent.height,box[2]/parent.width,box[3]/parent.height]]
                        original_box = row['page_pixel_box'] if row['task'] == 'document_ocr' else row['normalized_box']
                        matches = dict(original_full_PDF_target_matches=actual_text == row['response'],original_source_box_matches=expected_box == original_box,
                            original_parent_RGB224_hash_matches=parent_hash == row['parent_pixel_sha256'],
                            every_saved_source_RGB_pixel_matches=actual.size == expected.size and actual.tobytes() == expected.tobytes(),
                            original_saved_RGB224_hash_matches=pixel_hash(actual) == row['model_pixel_sha256'])
                        visibility = requested_pixels(actual,row);fill_colors = Counter();modes = Counter();queries_failed = 0;nonspace = 0
                        for index,(char,bounds) in enumerate(characters):
                            if char.isspace() or not (left-.1 <= bounds[0] and bottom-.1 <= bounds[1] and bounds[2] <= right+.1 and bounds[3] <= top+.1):continue
                            nonspace += 1
                            if index not in glyph_cache:
                                color = [ctypes.c_uint() for _ in range(4)]
                                success = pdfium.raw.FPDFText_GetFillColor(text.raw,index,*[ctypes.byref(v) for v in color])
                                obj = pdfium.raw.FPDFText_GetTextObject(text.raw,index)
                                mode = pdfium.raw.FPDFTextObj_GetTextRenderMode(obj) if obj else -1
                                glyph_cache[index] = (bool(success),tuple(v.value for v in color),int(mode))
                            success,color,mode = glyph_cache[index];modes[mode] += 1
                            if success:fill_colors[color] += 1
                            else:queries_failed += 1
                        reconstruction = all(matches.values());veto = reconstruction and visibility['nonempty_target_on_all_white_requested_pixels']
                        status = ('confirmed_nonempty_target_on_all_white_requested_pixels' if veto else
                                  'not_vetoed_by_this_narrow_pixel_rule' if reconstruction else 'original_PDF_label_or_pixel_reconstruction_mismatch')
                        counts[status] += 1;counts['original_rows'] += 1;counts['prepared_native_views_covered'] += len(native[row['id']])
                        counts['source_reconstructions_match'] += reconstruction
                        record = dict(original_JSONL_position=position,original_line_sha256=hashlib.sha256(raw).hexdigest(),source_id=row['id'],
                            original_source_role='train',task=row['task'],public_PDF=package,original_page_index=page_index,original_rectangle_index=rectangle_index,
                            original_response=row['response'],fresh_original_PDF_response=actual_text,source_reconstruction=matches,
                            actual_source_dimensions=list(actual.size),requested_source_pixels=visibility,nonspace_original_characters_inspected=nonspace,
                            PDF_fill_color_observations=[dict(RGBA=list(color),characters=n) for color,n in sorted(fill_colors.items())],
                            PDF_render_mode_character_counts=dict(modes),PDF_fill_query_failures=queries_failed,status=status,
                            veto_unchanged_positive_for_future_admission=veto,prepared_native_views=native[row['id']],
                            nonwhite_and_nonvetoed_does_not_certify_semantic_label_correctness=True,training_admitted=False)
                        encoded = json.dumps(record,ensure_ascii=True)+'\n'
                        if destination.tell()+len(encoded.encode()) > MAX_OUTPUT:raise ValueError('Full original source observations exceeded the declared storage ceiling')
                        destination.write(encoded);completed += 1
                        if veto:vetoes.append(record)
                    pages += 1;destination.flush()
                    atomic(OUT/'progress.json',dict(owner=owner,original_rows_complete=completed,total_original_rows=1433,
                        original_PDF_pages_complete=pages,total_PDF_pages=161,actual_counts=dict(counts),bounded_disk_waits=dict(waits),training_admitted=False))
                finally:bitmap.close();text.close();page.close()
            destination.flush();os.fsync(destination.fileno())
        checked()
        if (completed != 1433 or pages != 161 or counts['prepared_native_views_covered'] != 1382
                or not {r['source_id'] for r in known['actual_bad_original_positive_views']} <= {r['source_id'] for r in vetoes}):
            raise ValueError('Every original row/page/prepared view and both independently established defects must be retained')
        if any(sha(p) != h for p,h in files.items()):raise ValueError('Every original source and runtime byte must remain unchanged at completion')
        atomic(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
            actual_counts=dict(counts),actual_original_PDF_pages=pages,actual_original_PDF_files=6,
            all_original_source_reconstructions_match=counts['source_reconstructions_match'] == 1433,
            actual_confirmed_semantic_veto_source_ids=[r['source_id'] for r in vetoes],
            actual_confirmed_veto_prepared_native_views=sum(len(r['prepared_native_views']) for r in vetoes),
            all_original_dispositions_sha256=sha(output),bounded_disk_waits=dict(waits),
            original51_rows_absent_from_prepared_epoch_not_readmitted=True,
            all_nonvetoed_transcript_semantics_still_uncertified=True,PDF_fill_color_alone_not_a_rejection_rule=True,
            fresh_readback_and_future_source_requalification_pending=True,
            existing_sources_and_recipes_and_neural_workers_unchanged=True,neural_forwards=0,
            training_admitted=False,new_production_optimizer_updates=0,goal_achieved=False))
        atomic(OUT/'semantic-vetoes.json',dict(evidence=str(OUT/'complete.json'),evidence_sha256=sha(OUT/'complete.json'),
            unchanged_positive_source_vetoes=[dict(source_id=r['source_id'],original_JSONL_position=r['original_JSONL_position'],
                original_line_sha256=r['original_line_sha256'],prepared_native_views=r['prepared_native_views']) for r in vetoes],
            automatic_source_removal_or_relabeling=False))
        print(json.dumps(dict(complete=True,actual_counts=dict(counts),training_admitted=False)),flush=True)
    except BaseException as error:
        atomic(OUT/'failure.json',dict(owner=owner,error=repr(error),automatic_retry=False,training_admitted=False,goal_achieved=False));raise


if __name__ == '__main__':main()
