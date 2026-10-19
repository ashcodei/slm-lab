import json
from pathlib import Path
import pytest
from PIL import Image
from slm_lab.diverse_images import word_crops
from slm_lab.image_curriculum import combine
from slm_lab.document_images import rectangle_text


def source(tmp_path,name,train,valid,protected=()):
    root=tmp_path/name;root.mkdir()
    for split,rows in [('train',train),('valid',valid)]:
        (root/(split+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (root/'protected-image-ids.txt').write_text('\n'.join(protected))
    return name,root


def row(image,group,key):
    return {'id':key,'image':str(image),'group_id':group,'prompt':'Read this.',
            'response':key,'model_pixel_sha256':'untrusted-source-hash','task':'word_ocr'}


def test_original_photo_protection_and_pixel_duplicate_protection_apply_to_all_crops(tmp_path):
    images=[]
    for n,color in enumerate(['red','blue','green','yellow','black']):
        path=tmp_path/(str(n)+'.png');Image.new('RGB',(32,32),color).save(path);images.append(path)
    # A photo's ID is reserved even though the selected validation crop has different pixels.
    # Another training photo has one exact held-out crop: its unique sibling crop must go too.
    train=[row(images[0],'photo:A','a'),row(images[1],'photo:B','b1'),
           row(images[2],'photo:B','b2'),row(images[3],'photo:C','c')]
    valid=[row(images[1],'photo:D','d')]
    first=source(tmp_path,'photos',train,[])
    second=source(tmp_path,'ocr',[],valid,protected=['photo:A'])
    report=combine([first,second],tmp_path/'combined')
    records=[json.loads(line) for line in (tmp_path/'combined/images.train.jsonl').read_text().splitlines()]
    assert [r['id'] for r in records]==['c']
    assert report['excluded_pixel_duplicate_groups']==1
    assert report['train_valid_exact_model_pixel_overlap']==0
    assert report['train_valid_image_group_overlap']==0
    assert records[0]['model_pixel_sha256']!='untrusted-source-hash'


def test_word_crop_coordinates_scale_and_illegible_labels_are_excluded():
    image=Image.new('RGB',(200,100),'white')
    annotations=[{'id':'one','utf8_string':'Hello','bbox':[20,10,50,20]},
                 {'id':'bad','utf8_string':'.','bbox':[20,10,50,20]},
                 {'id':'outside','utf8_string':'No','bbox':[500,500,20,20]}]
    crops=word_crops(image,100,50,annotations,4)
    assert len(crops)==1 and crops[0][1]=='Hello'
    assert 100<=crops[0][2].width<=112
    assert 40<=crops[0][2].height<=50
    with pytest.raises(ValueError,match='aspect ratio'):
        word_crops(image,50,100,annotations,4)


def test_pdf_reading_target_excludes_intersecting_neighboring_lines():
    characters=[('x',(0,8,4,12)),('H',(1,1,3,7)),('i',(4,1,5,7)),
                ('!',(6,1,7,7)),('y',(0,-2,4,2))]
    assert rectangle_text((0,0,8,8),characters)=='Hi!'
