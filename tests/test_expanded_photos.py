import io
import json

import numpy as np
from PIL import Image

from slm_lab.diverse_images import pixel_hash
from slm_lab.expanded_photos import encode_storage, select_candidates, difference_hash, near_protected


def test_additional_photo_selection_excludes_old_parents_and_unsafe_ids(tmp_path):
    ids = ['0000000000000001', '0000000000000002', '../outside', '0000000000000003']
    rows = [{'image_id': identifier, 'caption': 'A human caption describing an ordinary photo.'} for identifier in ids]
    rows.append(rows[1])
    path = tmp_path / 'captions.jsonl'; path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    selected = select_candidates(path, {'openimages:' + ids[0]}, 10)
    assert {row['image_id'] for row in selected} == {ids[1], ids[3]}
    path.write_text(''.join(json.dumps(row) + '\n' for row in reversed(rows)))
    assert select_candidates(path, {'openimages:' + ids[0]}, 10) == selected


def test_jpeg_input_hashes_describe_actual_stored_pixels_and_perceptual_screen():
    rng = np.random.default_rng(42)
    image = Image.fromarray(rng.integers(0, 256, (800, 1000, 3), dtype=np.uint8))
    raw, hashes, before, fingerprint = encode_storage(image)
    with Image.open(io.BytesIO(raw)) as stored:
        assert max(stored.size) == 768
        assert all(hashes[size] == pixel_hash(stored, size) for size in [224, 448])
        assert fingerprint == difference_hash(stored)
    assert hashes != before
    assert near_protected(fingerprint, np.array([fingerprint], dtype=np.uint64))
    assert not near_protected(fingerprint, np.array([fingerprint ^ ((1 << 64) - 1)], dtype=np.uint64))
