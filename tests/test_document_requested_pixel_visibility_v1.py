import pytest
from PIL import Image,ImageDraw

from slm_lab.document_requested_pixel_visibility_v1 import requested_pixels


def test_full_white_crop_with_nonempty_target_has_no_visible_marks():
    image = Image.new('RGB',(39,768),'white')
    result = requested_pixels(image,dict(task='document_ocr',response='hidden source target'))
    assert result['nonempty_target_on_all_white_requested_pixels'] and result['actual_requested_dimensions'] == [39,768]


def test_full_page_ink_elsewhere_does_not_validate_the_requested_white_region():
    image = Image.new('RGB',(100,100),'white');ImageDraw.Draw(image).rectangle((0,0,20,20),fill='black')
    row = dict(task='document_region_ocr',response='hidden',normalized_box=[.6,.6,.9,.9])
    assert requested_pixels(image,row)['nonempty_target_on_all_white_requested_pixels']
    row['normalized_box'] = [0,0,.3,.3]
    assert not requested_pixels(image,row)['nonempty_target_on_all_white_requested_pixels']


def test_white_foreground_on_dark_background_is_not_rejected_by_white_fill_metadata():
    image = Image.new('RGB',(100,30),'black');ImageDraw.Draw(image).text((3,3),'VISIBLE',fill='white')
    row = dict(task='document_ocr',response='VISIBLE',PDF_fill_color=[255,255,255,255])
    result = requested_pixels(image,row)
    assert not result['nonempty_target_on_all_white_requested_pixels']
    assert result['nonwhite_does_not_certify_transcript_visibility_or_correctness']


def test_no_source_pixel_mutation_and_no_nonempty_label_veto_for_empty_transcript():
    image = Image.new('RGB',(11,17),'white');before = image.tobytes()
    result = requested_pixels(image,dict(task='document_ocr',response=''))
    assert not result['nonempty_target_on_all_white_requested_pixels'] and image.tobytes() == before


def test_invalid_or_off_image_boxes_are_not_silently_clipped_or_replaced():
    image = Image.new('RGB',(10,10),'white')
    for box in [[0,0,0,1],[-.1,0,1,1],[0,0,1.1,1],[0,float('nan'),1,1],[0,0,1]]:
        with pytest.raises(ValueError):requested_pixels(image,dict(task='document_region_ocr',response='text',normalized_box=box))
