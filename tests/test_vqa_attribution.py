import copy
import json

import pytest

from slm_lab.vqa_attribution import attribution


def declaration():
    return {'@type': 'ImageObject', 'acquireLicensePage': 'https://www.flickr.com/photos/example/123',
            'license': 'https://creativecommons.org/licenses/by/2.0/',
            'author': {'name': 'Original photographer', 'url': 'https://www.flickr.com/photos/example'},
            'contentUrl': 'https://live.staticflickr.com/1/123_signature_b.jpg'}


def html(node):
    return ('<script type="application/ld+json">' + json.dumps({'@graph': [node]}) + '</script>').encode()


def test_original_jsonld_retains_creator_and_license_without_executing_page_scripts():
    raw = html(declaration()) + b'<script>throw new Error("must not execute")</script>'
    result = attribution(raw, '123', 'https://www.flickr.com/photos/example/123/')
    assert result['author'] == 'Original photographer' and result['license'] == 'CC-BY-2.0'


@pytest.mark.parametrize('change', ['license', 'no_author', 'wrong_profile', 'wrong_image', 'conflicting_creator'])
def test_original_photo_credit_and_current_license_must_match(change):
    node = declaration()
    if change == 'license': node['license'] = 'https://creativecommons.org/licenses/by-nc/2.0/'
    elif change == 'no_author': del node['author']
    elif change == 'wrong_profile': node['author']['url'] = 'https://www.flickr.com/photos/another'
    elif change == 'wrong_image': node['contentUrl'] = 'https://live.staticflickr.com/1/456_signature_b.jpg'
    else: node['creator'] = {'name': 'Another person', 'url': node['author']['url']}
    with pytest.raises(ValueError): attribution(html(node), '123', 'https://www.flickr.com/photos/example/123')


def test_another_original_page_cannot_supply_the_requested_photo_attribution():
    with pytest.raises(ValueError): attribution(html(declaration()), '456', 'https://www.flickr.com/photos/example/123')


def test_conflicting_photo_declarations_are_rejected():
    first = declaration(); second = copy.deepcopy(first); second['author']['name'] = 'Another person'
    raw = ('<script type="application/ld+json">' + json.dumps([first, second]) + '</script>').encode()
    with pytest.raises(ValueError): attribution(raw, '123', 'https://www.flickr.com/photos/example/123')
