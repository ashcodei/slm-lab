import copy
import io

from PIL import Image,ImageOps
import pytest

from slm_lab.localized_pointer_v1 import geometry,histogram,jpeg_header,spans


def original():
    return dict(caption='In this image a red car and a blue car.',timed_caption=[
        dict(utterance='In this image',start_time=0.,end_time=1.),
        dict(utterance='a red car',start_time=1.,end_time=2.),
        dict(utterance='and',start_time=2.,end_time=2.2),
        dict(utterance='a blue car.',start_time=2.2,end_time=3.2)],
        traces=[[dict(x=x,y=.5,t=t) for x,t in [(.25,1.),(.25,1.2),(.25,1.4),(.25,1.6),(.25,1.8),
            (.75,2.3),(.75,2.5),(.75,2.7),(.75,2.9),(.75,3.1)]]])


def test_exact_human_utterances_trace_indices_and_real_padding_cells():
    row=original();selected=spans(row)
    assert [s['query'] for s in selected]==['a red car','a blue car.']
    assert selected[0]['source_point_refs']==[[0,i] for i in range(5)]
    assert histogram(row,selected[0],(400,200))==[[101,5]]
    assert histogram(row,selected[1],(400,200))==[[108,5]]
    # The actual PIL padding puts this original-image center on row7/column4.
    image=Image.new('RGB',(375,500));padded=ImageOps.pad(image,(448,448))
    assert padded.size==(448,448)
    assert geometry((.25,.5),image.size)==((.3125,.5),102)
    source=io.BytesIO();Image.new('RGB',(400,200)).save(source,format='JPEG')
    assert jpeg_header(source.getvalue())==(400,200)


def test_misaligned_transcript_bad_trace_off_image_and_rotated_source_reject():
    row=original();row['caption']='A changed transcript.'
    with pytest.raises(ValueError,match='manual transcript'):spans(row)
    row=original();row['traces'][0][2]['t']=-1
    with pytest.raises(ValueError,match='ordered mouse'):spans(row)
    row=original()
    for p in row['traces'][0]:p['x']=1.02
    assert spans(row)==[]
    with pytest.raises(ValueError,match='in-frame point'):geometry((1.,.5),(400,200))
    exif=Image.Exif();exif[274]=6
    source=io.BytesIO();Image.new('RGB',(400,200)).save(source,format='JPEG',exif=exif)
    with pytest.raises(ValueError,match='identity EXIF'):jpeg_header(source.getvalue())
