"""Conservative all-white requested-region evidence for original OCR sources.

This is a narrow source-admission diagnostic, not an OCR evaluator or RL reward.
Nonwhite pixels do not establish that a particular transcript is visible or
correct. PDF fill colors alone are deliberately insufficient for rejection.
"""
import hashlib
import math


def requested_pixels(image,row):
    rgb = image.convert('RGB')
    if row['task'] == 'document_ocr':
        box = [0,0,rgb.width,rgb.height]
    elif row['task'] == 'document_region_ocr':
        values = row['normalized_box']
        if (not isinstance(values,list) or len(values) != 4
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in values)
                or not 0 <= values[0] < values[2] <= 1 or not 0 <= values[1] < values[3] <= 1):
            raise ValueError('The entire original finite normalized source box must be retained inside the image')
        box = [math.floor(values[0]*rgb.width),math.floor(values[1]*rgb.height),
               math.ceil(values[2]*rgb.width),math.ceil(values[3]*rgb.height)]
    else:raise ValueError('Only explicit original whole-crop or boxed-document OCR sources are supported')
    if min(box[2]-box[0],box[3]-box[1]) < 1:raise ValueError('The complete requested region must contain actual pixels')
    region = rgb.crop(box);extrema = region.getextrema()
    all_white = extrema == ((255,255),)*3
    return dict(original_requested_pixel_box=box,actual_requested_dimensions=list(region.size),
        actual_requested_RGB_sha256=hashlib.sha256(region.tobytes()).hexdigest(),
        actual_requested_channel_extrema=[list(pair) for pair in extrema],
        actual_requested_region_all_white=all_white,
        nonempty_target_on_all_white_requested_pixels=all_white and isinstance(row['response'],str) and bool(row['response'].strip()),
        nonwhite_does_not_certify_transcript_visibility_or_correctness=True,
        PDF_fill_color_alone_not_a_rejection_rule=True)
