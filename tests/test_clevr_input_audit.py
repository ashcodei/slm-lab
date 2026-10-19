import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import zlib

from PIL import Image, ImageOps
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from slm_lab.model import ModelConfig
from slm_lab.training import Encoder

spec = importlib.util.spec_from_file_location('clevr_input_audit', Path(__file__).resolve().parents[1] / 'scripts/audit_clevr_inputs.py')
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


@pytest.fixture
def original(tmp_path):
    scene = {'image_filename': 'CLEVR_train_000000.png', 'image_index': 0, 'split': 'train',
             'objects': [{'color': 'red'}], 'relationships': {}}
    question = {'image_filename': scene['image_filename'], 'image_index': 0, 'split': 'train',
                'question_index': 1, 'question_family_index': 1, 'question': 'How many objects are there?', 'answer': '1',
                'program': [{'function': 'scene', 'inputs': [], 'value_inputs': []},
                            {'function': 'count', 'inputs': [0], 'value_inputs': []}]}
    tokenizer = Tokenizer(models.WordLevel({'<bos>': 0, '<eos>': 1, '<pad>': 2, '<unk>': 3, '1': 4}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    encoder = Encoder(tokenizer, ModelConfig(vocab_size=5), tmp_path, 448)
    row = audit.original_fields(question, scene); ids, labels = encoder.sequence(row, 'response')
    row.update(context_tokens=len(ids), target_tokens=sum(value != -100 for value in labels[1:]))
    return row, question, scene, encoder


def test_complete_original_question_retains_only_original_prompt_and_target(original):
    row, question, scene, encoder = original
    assert audit.validate_question(row, question, scene, encoder)['response'] == '1'
    assert scene['image_filename'] not in encoder.prompt(row)


@pytest.mark.parametrize('field,value', [('response', '2'), ('group_id', 'clevr:val:another.png'),
                                        ('image', 'images/another.png'), ('question_program', []),
                                        ('symbolic_answer_verified', 1), ('context_tokens', 1)])
def test_changed_labels_identity_hidden_program_or_full_tokens_are_rejected(original, field, value):
    row, question, scene, encoder = original
    altered = copy.deepcopy(row); altered[field] = value
    with pytest.raises(ValueError): audit.validate_question(altered, question, scene, encoder)


def test_original_question_cannot_be_attached_to_a_different_scene(original):
    row, question, scene, encoder = original
    changed = copy.deepcopy(scene); changed['image_index'] = 7
    with pytest.raises(ValueError): audit.validate_question(row, question, changed, encoder)


@pytest.mark.parametrize('change', ['pixels', 'crc', 'path'])
def test_image_audit_rejects_changed_pixels_archive_entry_or_escaping_path(tmp_path, change):
    image = Image.new('RGBA', (480, 320), (10, 20, 30, 255)); path = tmp_path / 'original.png'; image.save(path)
    raw = path.read_bytes(); crc = zlib.crc32(raw)
    digests = {size: hashlib.sha256(ImageOps.pad(image.convert('RGB'), (size, size), method=Image.Resampling.BICUBIC,
                 color=(128, 128, 128)).tobytes()).hexdigest() for size in [224, 448]}
    row = {'image': path.name, 'stored_image_sha256': hashlib.sha256(raw).hexdigest(), 'original_member_bytes': len(raw),
           'original_member_crc32': crc, 'model_pixel_sha256': digests[224], 'model_pixel_sha256_448': digests[448]}
    entry = SimpleNamespace(file_size=len(raw), CRC=crc)
    assert audit.validate_image(tmp_path, row, entry) == (digests, len(raw))
    if change == 'pixels': row['model_pixel_sha256_448'] = 'other'
    elif change == 'crc': entry.CRC += 1
    else: row['image'] = '../original.png'
    with pytest.raises(ValueError): audit.validate_image(tmp_path, row, entry)
