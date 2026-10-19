from PIL import Image
from slm_lab.expanded_scene_words import additional_crops


def test_additional_crops_keep_unused_human_labels_and_filter_tiny_illegible_boxes():
    image = Image.new('RGB', (100, 100), 'white')
    annotations = [{'id': 'old', 'utf8_string': 'USED', 'bbox': [10, 10, 30, 20]},
                   {'id': 'new', 'utf8_string': 'NEW', 'bbox': [40, 10, 30, 20]},
                   {'id': 'illegible', 'utf8_string': '.', 'bbox': [10, 40, 30, 20]},
                   {'id': 'tiny', 'utf8_string': 'SMALL', 'bbox': [10, 70, 4, 4]}]
    crops = additional_crops(image, 100, 100, annotations, {'old'}, 16)
    assert [(identifier, text) for identifier, text, _, _ in crops] == [('new', 'NEW')]
    assert min(crops[0][2].size) >= 8
