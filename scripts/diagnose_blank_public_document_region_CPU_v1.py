"""Extend the exact PDF defect to its paired full-page region-reading view."""
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path

from PIL import Image,ImageOps
import pypdfium2 as pdfium

from diagnose_blank_public_document_supervision_CPU_v1 import ROOT,DATA,PDF,sha,read,atomic
from slm_lab.diverse_images import pixel_hash

OUT = ROOT/'reports/public-document-blank-region-CPU-v1'
PREVIOUS = ROOT/'reports/public-document-blank-supervision-CPU-v1/complete.json'
METADATA = ROOT/'reports/full-multimodal-source-epoch-CPU-v1/all302627-original-native-views.jsonl'


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh bounded low-priority one-thread source-only region diagnostic required')
    previous = read(PREVIOUS)
    if not previous.get('complete') or not previous['actual_independent_nonempty_OCR_target_on_constant_white_image']:
        raise ValueError('Exact original public PDF, all-white crop and 39 white-filled character observations required')
    rows = [json.loads(line) for line in (DATA/'train.jsonl').open()]
    cropped = next(row for row in rows if row['id'] == previous['original_source_id'])
    page_row = next(row for row in rows if row['id'] == previous['original_source_id']+'-page')
    page_image = DATA/page_row['image']
    files = dict(previous['source_files_sha256'])
    files.update({str(p):sha(p) for p in [Path(__file__),PREVIOUS,METADATA,page_image]})
    if (page_row['split'] != 'train' or page_row['task'] != 'document_region_ocr'
            or page_row['response'] != cropped['response'] or page_row['download_sha256'] != cropped['download_sha256']
            or page_row['group_id'] != cropped['group_id'] or sha(page_image) != page_row['stored_image_sha256']
            or any(sha(p) != digest for p,digest in files.items())):
        raise ValueError('The original paired page label and every original immutable source must remain exact')
    OUT.mkdir();atomic(OUT/'plan.json',dict(source_files_sha256=files,original_source_ids=[cropped['id'],page_row['id']],
        training_admitted=False,original_sources_and_jobs_unchanged=True,goal_achieved=False))
    try:
        with pdfium.PdfDocument(PDF) as document:
            page = document[page_row['page_index']];bitmap = page.render(scale=cropped['render_scale'])
            try:
                parent = bitmap.to_pil().convert('RGB');box = cropped['page_pixel_box']
                normalized = [round(v,4) for v in [box[0]/parent.width,box[1]/parent.height,box[2]/parent.width,box[3]/parent.height]]
                with Image.open(page_image) as opened:actual = opened.convert('RGB')
                reconstructed = ImageOps.contain(parent,(768,768),Image.Resampling.LANCZOS)
                selected = [math.floor(normalized[0]*actual.width),math.floor(normalized[1]*actual.height),
                            math.ceil(normalized[2]*actual.width),math.ceil(normalized[3]*actual.height)]
                region = actual.crop(selected)
                if (normalized != page_row['normalized_box'] or actual.size != reconstructed.size
                        or actual.tobytes() != reconstructed.tobytes() or pixel_hash(actual) != page_row['model_pixel_sha256']
                        or region.getextrema() != ((255,255),)*3):
                    raise ValueError('The original full-page pixels and explicitly requested normalized region must reconstruct as all-white')
            finally:bitmap.close();page.close()
        found = []
        with METADATA.open('rb') as stream:
            for line in stream:
                if any(('"source_id": "'+row['id']+'"').encode() in line for row in [cropped,page_row]):
                    record = json.loads(line)
                    if record['family'] != 'replay:image_reading' or record['source_role'] != 'train' or record['training_admitted']:
                        raise ValueError('Both original defective views must remain train-only unadmitted full-epoch records')
                    reference = record['original_source_reference']
                    with Path(reference['path']).open('rb') as original:
                        original.seek(reference['offset']);raw = original.read(reference['bytes'])
                    joint = json.loads(raw)
                    if hashlib.sha256(raw).hexdigest() != reference['row_sha256'] or joint['id'] != record['source_id'] or joint['response'] != cropped['response']:
                        raise ValueError('Every proposed native view must bind the exact unchanged defective original positive target')
                    found.append(record)
        if len(found) != 2 or {r['source_id'] for r in found} != {cropped['id'],page_row['id']}:
            raise ValueError('Exactly both original native positive views must be accounted for')
        raw = (ROOT/'reports/public-pretraining-recovery-v1/training-metrics.jsonl').read_bytes();prefix = raw[:raw.rfind(b'\n')+1]
        updates = [json.loads(line) for line in prefix.split(b'\n') if line]
        if [row['step'] for row in updates] != list(range(1,len(updates)+1)):raise ValueError('Contiguous committed current-pass exposure observations required')
        exposures = {row['source_id']:sum(update['global_original_positions'].count(row['original_local_position']) for update in updates
            if update['family'] == row['family']) for row in found}
        if any(sha(p) != digest for p,digest in files.items()):raise ValueError('Every original source must remain unchanged')
        result = dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
            actual_bad_original_positive_views=found,actual_bad_native_response_EOS_targets=sum(r['response_EOS_targets'] for r in found),
            actual_saved_page_dimensions=list(actual.size),actual_normalized_prompt_box=normalized,
            actual_requested_saved_page_pixel_box=selected,actual_requested_region_dimensions=list(region.size),
            actual_requested_region_channel_extrema=[list(v) for v in region.getextrema()],
            actual_current_pass_committed_prefix_updates=len(updates),actual_current_pass_prefix_sha256=hashlib.sha256(prefix).hexdigest(),
            actual_current_pass_defective_view_exposures=exposures,total_historical_exposures_not_established=True,
            both_requested_visible_regions_have_no_nonwhite_pixels=True,
            unchanged_positive_views_semantically_vetoed_for_future_admission=True,
            full_source_4733_update_plan_requires_explicit_requalification_without_silent_source_removal=True,
            existing_data_and_main_and_queued_jobs_unchanged=True,model_quality_impact_not_measured=True,
            training_admitted=False,new_production_optimizer_updates=0,goal_achieved=False)
        atomic(OUT/'complete.json',result)
        atomic(OUT/'semantic-vetoes.json',dict(evidence=str(OUT/'complete.json'),evidence_sha256=sha(OUT/'complete.json'),
            veto_unchanged_positive_views=[dict(source_id=r['source_id'],source_role='train',original_joint_row_sha256=r['original_source_reference']['row_sha256'],
                original_stored_image_sha256=r['original_stored_image_sha256'],original_response_UTF8_sha256=r['response_utf8_sha256']) for r in found],
            automatic_source_removal_or_relabeling=False))
        print(json.dumps(dict(complete=True,defective_original_positive_views=len(found),bad_native_response_EOS_targets=result['actual_bad_native_response_EOS_targets'],training_admitted=False)),flush=True)
    except BaseException as error:
        atomic(OUT/'failure.json',dict(error=repr(error),automatic_retry=False,training_admitted=False,goal_achieved=False));raise


if __name__ == '__main__':main()
