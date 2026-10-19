from email.message import Message
import json

import google_crc32c
from PIL import Image, ImageDraw
import pytest

from slm_lab.object_qa import FLAGS, checked_asset, extract_subset, questions_for, rotate_box


@pytest.mark.parametrize('angle',[0,90,180,270])
def test_annotation_rotations_follow_the_existing_photo_pixels(angle):
    original=Image.new('RGB',(200,100),'white')
    ImageDraw.Draw(original).rectangle((20,20,60,40),fill='blue')
    image=original.rotate(angle,expand=True)
    left,top,right,bottom=rotate_box([.1,.2,.3,.4],angle)
    center=(round((left+right)*image.width/2),round((top+bottom)*image.height/2))
    assert image.getpixel(center)==(0,0,255)


def box(label,coordinates,**flags):
    return {'label':label,'label_id':label,'box':coordinates,'source':'xclick','row_number':1,
            'flags':{**{name:0 for name in FLAGS},**flags}}


def test_counting_never_treats_a_crowd_as_one_object_or_counts_a_filtered_subset():
    photo={'group_id':'photo'}
    boxes=[box('Person',[.1,.1,.3,.5]),box('Person',[.4,.2,.8,.6],IsGroupOf=1)]
    assert not any(row['task']=='object_count' for row in questions_for(photo,boxes))
    boxes[1]['flags']['IsGroupOf']=0
    counts=[row for row in questions_for(photo,boxes) if row['task']=='object_count']
    assert len(counts)==1 and counts[0]['response']=='2'


def test_letterbox_resolution_can_exclude_a_whole_count_question():
    photo={'group_id':'photo','stored_image_dimensions':[768,384]}
    boxes=[box('Dog',[.1,.1,.3,.4]),box('Dog',[.8,.8,.82,.82])]
    assert not any(row['task']=='object_count' for row in questions_for(photo,boxes))
    boxes[1]['box']=[.6,.1,.8,.4]
    counts=[row for row in questions_for(photo,boxes) if row['task']=='object_count']
    assert len(counts)==1 and counts[0]['response']=='2'


def test_spatial_questions_require_images_and_region_targets_are_unambiguous():
    boxes=[box('Cat',[.05,.25,.3,.6]),box('Dog',[.6,.25,.9,.6])]
    rows=questions_for({'group_id':'photo'},boxes)
    spatial=next(row for row in rows if row['task']=='object_spatial')
    assert spatial['response']==('yes' if spatial['spatial_relation']=='to the left of' else 'no')
    assert '0.05' not in spatial['prompt'] and '[0.' not in spatial['prompt']
    # An identical region with two class labels cannot supervise two conflicting
    # answers to the same question.
    rows=questions_for({'group_id':'photo'},[box('Man',[.1,.2,.6,.9]),box('Person',[.1,.2,.6,.9])])
    assert not any(row['task']=='object_region' for row in rows)


def test_conflicting_duplicate_annotations_invalidate_the_whole_class(tmp_path):
    path=tmp_path/'boxes.csv'
    path.write_text('ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,'+','.join(FLAGS)+'\n'
                    'photo,xclick,Person,1,0.1,0.6,0.2,0.9,0,0,0,0,0\n'
                    'photo,xclick,Person,1,0.1,0.6,0.2,0.9,1,0,0,0,0\n')
    grouped,rejected=extract_subset(path,{'photo':{'applied_source_rotation':0}},{'Person':'Person'},lambda _:None)
    assert not grouped['photo'] and rejected['duplicate_box']==1


def test_publisher_checksum_rejects_an_incomplete_or_corrupt_annotation_file(tmp_path,monkeypatch):
    path=tmp_path/'boxes.csv';path.write_bytes(b'corrupt')
    headers=Message();headers['ETag']='version';headers['x-goog-hash']='crc32c=4waSgw==' # CRC32C("123456789")
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
    response=Response();response.headers=headers
    monkeypatch.setattr('slm_lab.object_qa.asset',lambda *args:(path,{'etag':'version','sha256':'recorded'}))
    monkeypatch.setattr('slm_lab.object_qa.urllib.request.urlopen',lambda *args,**kwargs:response)
    with pytest.raises(ValueError,match='CRC32C mismatch'):
        checked_asset(tmp_path,('boxes.csv','https://example.org/boxes',100))
    path.write_bytes(b'123456789')
    _,record=checked_asset(tmp_path,('boxes.csv','https://example.org/boxes',100))
    assert record['verified_publisher_crc32c']=='4waSgw=='
