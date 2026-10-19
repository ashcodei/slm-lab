import io
import json

import pytest

from slm_lab.clevr_programs import answer, array_items
from slm_lab.remote_zip import checked_range


class Response:
    def __init__(self, status=206, etag='pinned', content_range='bytes 2-4/10', raw=b'abc'):
        self.status = status; self.headers = {'ETag': etag, 'Content-Range': content_range}; self.raw = raw; self.read_called = False
    def read(self, count): self.read_called = True; return self.raw[:count]


def test_range_ignoring_server_is_rejected_before_reading_the_full_archive():
    response = Response(status=200)
    with pytest.raises(ValueError): checked_range(response, 2, 4, 10, 'pinned')
    assert not response.read_called


@pytest.mark.parametrize('change', ['identity', 'range', 'length'])
def test_changed_source_identity_or_bytes_cannot_be_admitted(change):
    response = Response(etag='other' if change == 'identity' else 'pinned',
                        content_range='bytes 0-2/10' if change == 'range' else 'bytes 2-4/10',
                        raw=b'ab' if change == 'length' else b'abc')
    with pytest.raises(ValueError): checked_range(response, 2, 4, 10, 'pinned')


def test_symbolic_questions_verify_relations_counts_and_exclude_same_object():
    scene = {'objects': [{'color': 'red', 'shape': 'cube'}, {'color': 'blue', 'shape': 'cube'},
                         {'color': 'blue', 'shape': 'sphere'}], 'relationships': {'left': [[], [0], [0, 1]]}}
    node = lambda op, inputs, values=[]: {'function': op, 'inputs': inputs, 'value_inputs': values}
    assert answer(scene, [node('scene', []), node('filter_color', [0], ['red']), node('unique', [1]),
                          node('same_shape', [2]), node('count', [3])]) == '1'
    assert answer(scene, [node('scene', []), node('filter_shape', [0], ['sphere']), node('unique', [1]),
                          node('relate', [2], ['left']), node('count', [3])]) == '2'
    with pytest.raises(ValueError): answer(scene, [node('scene', []), node('unique', [0])])


def test_original_array_reader_consumes_complete_metadata_without_loading_all_records():
    raw = json.dumps({'info': {'license': 'fixture'}, 'questions': [{'question': 'one'}, {'question': 'two'}]}).encode()
    assert list(array_items(io.BytesIO(raw), 'questions')) == [{'question': 'one'}, {'question': 'two'}]
