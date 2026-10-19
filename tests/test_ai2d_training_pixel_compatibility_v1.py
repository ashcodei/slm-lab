"""Actual held pixels are used only for preprocessing controls, with no NN.

These controls are not training admission or an image/semantic independence
certificate. They check the exact comparison protocol used by the live audit.
"""
import json
from pathlib import Path

from PIL import Image
import torch
from tokenizers import Tokenizer

from slm_lab.ai2d_full_native_evaluation_v1 import tensor_sha
from slm_lab.codenet_runner_v1 import VerifiedReplayEncoder
from slm_lab.corpus import file_hash
from slm_lab.model import ModelConfig
from slm_lab.shared_word_data_v1 import array_digest, image_arrays


def test_actual_original_diagram_matches_both_training_native_preprocessors():
    root = Path(__file__).resolve().parents[1]; folder = root / 'data/ai2d-all-original-test-eval-v2'
    first = json.loads(next((folder / 'image-inventory.jsonl').open()))
    variant = first['native_variants']['original']; path = Path(first['original_image'])
    assert file_hash(path) == first['original_image_sha256']
    tokenizer = Tokenizer.from_file(str(root / 'data/tokenizer-corpus/tokenizer.json'))
    encoder = VerifiedReplayEncoder(tokenizer, ModelConfig(), root, 448)
    # Pixel preprocessing does not optimize or use responses. Keep the held role.
    row = dict(image=str(path), split='test', purpose='evaluation_only',
        model_pixel_sha256_448=variant['model_pixel_sha256_448'])
    actual = encoder.pixels(row)
    assert list(actual.shape) == [3, 448, 448] and actual.dtype == torch.float32
    assert tensor_sha(actual) == variant['model_input_tensor_sha256_448']
    with Image.open(path) as image: word, native = image_arrays(image)
    assert array_digest(native) == tensor_sha(actual)
    assert list(word.shape) == [3, 64, 448] and array_digest(word) != array_digest(native)
    assert not torch.cuda.is_initialized()


def test_actual_ABC_input_survives_training_native_preprocessing_without_hash_domain_change():
    root = Path(__file__).resolve().parents[1]; folder = root / 'data/ai2d-all-original-test-eval-v2'
    first = json.loads(next((folder / 'image-inventory.jsonl').open()))
    variant = first['native_variants']['abc_opaque']; tokenizer = Tokenizer.from_file(str(root / 'data/tokenizer-corpus/tokenizer.json'))
    encoder = VerifiedReplayEncoder(tokenizer, ModelConfig(), root, 448)
    actual = encoder.pixels(dict(image=variant['path'], split='test', purpose='evaluation_only',
        model_pixel_sha256_448=variant['model_pixel_sha256_448']))
    assert tensor_sha(actual) == variant['model_input_tensor_sha256_448']
    assert tensor_sha(actual) != first['native_variants']['original']['model_input_tensor_sha256_448']
