import pytest
from datetime import datetime,timezone
import io
import urllib.error

from slm_lab.vqa_pairs import canonical_question, flickr_photo_id, known_photo_aliases
from slm_lab import vqa_pairs


@pytest.mark.parametrize('url', ['https://www.flickr.com/photos/example/6415223357/',
                               'http://farm8.staticflickr.com/7156/6415223357_5ef0955050_z.jpg',
                               'https://www.flickr.com/photo.gne?id=6415223357'])
def test_original_flickr_landing_and_image_urls_identify_the_same_photo(url):
    assert flickr_photo_id(url) == '6415223357'


@pytest.mark.parametrize('url', [None, '', 'https://images.cocodataset.org/train2014/6415223357_image.jpg',
                                'https://www.flickr.com.evil.example/photos/example/6415223357'])
def test_unknown_or_unrelated_origins_are_never_photo_aliases(url):
    assert flickr_photo_id(url) is None
    assert known_photo_aliases({'flickr_url': url}) == set()


def test_multiple_source_fields_deduplicate_the_same_actual_photo_without_a_null_alias():
    row = {'flickr_url': 'https://farm8.staticflickr.com/7156/123_signature.jpg',
           'source_url': 'https://www.flickr.com/photos/example/123', 'original_url': None}
    assert known_photo_aliases(row) == {'123'}


def pair():
    return {'first':{'question':{'question':'What COLOR is this?', 'image_id':1},'answer':'red'},
            'second':{'question':{'question':'What color  is this?', 'image_id':2},'answer':'blue'}}


def test_both_images_have_exactly_the_same_model_question_without_filename_or_answer_hints():
    assert canonical_question(pair())=='what color is this?'


@pytest.mark.parametrize('change',['question','image','answer'])
def test_pair_closure_requires_a_matching_question_and_actual_image_answer_contrast(change):
    row=pair()
    if change=='question':row['second']['question']['question']='How many objects?'
    elif change=='image':row['second']['question']['image_id']=1
    else:row['second']['answer']='red'
    with pytest.raises(ValueError):canonical_question(row)


@pytest.mark.parametrize('value,expected',[('600',600),('0',60),(None,300),('invalid',300)])
def test_publisher_retry_after_seconds_and_missing_values_never_trigger_immediate_retries(value,expected):
    assert vqa_pairs.retry_delay(value)==expected


def test_publisher_retry_after_http_date_is_honored():
    now=datetime(2026,10,13,20,tzinfo=timezone.utc)
    assert vqa_pairs.retry_delay('Tue, 13 Oct 2026 20:05:00 GMT',now)==300


def test_rate_limit_retries_the_same_source_after_cooldown_instead_of_dropping_the_image(monkeypatch):
    calls=[]
    class Pacing:
        def acquire(self):calls.append('acquire')
        def cooldown(self,seconds,reason):calls.append(('cooldown',seconds))
    class Response:
        status=200;headers={'Content-Length':'3'};url='https://www.flickr.com/photos/source/123'
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def read(self,limit):return b'abc'
    count=0
    def open_source(request,timeout):
        nonlocal count
        count+=1
        if count==1:raise urllib.error.HTTPError(request.full_url,429,'Too Many Requests',{'Retry-After':'600'},io.BytesIO())
        return Response()
    monkeypatch.setattr(vqa_pairs,'FLICKR',Pacing());monkeypatch.setattr(vqa_pairs.urllib.request,'urlopen',open_source)
    assert vqa_pairs.fetch('https://www.flickr.com/photo.gne?id=123',10)[0]==b'abc'
    assert calls==['acquire',('cooldown',600),'acquire'] and count==2
