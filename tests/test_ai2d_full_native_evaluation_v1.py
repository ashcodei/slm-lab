import copy
from pathlib import Path

from PIL import Image
import pytest
import torch
from tokenizers import Tokenizer

from slm_lab.ai2d_native_eval_inputs_v1 import PROTOCOL, native_pixels, native_prompt
from slm_lab.ai2d_full_native_evaluation_v1 import (FullAI2D, dispositions, evaluate_case, gray_pixels,
    load_pixels, native_inputs, verdict, verify_record, wrong_images)
from slm_lab.dense_foundation_inference_v1 import DenseNativeView
from slm_lab.deterministic_positions_v1 import install
from slm_lab.model import ModelConfig
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone


def case(tmp_path, tokenizer, identity, color):
    image, rgb_sha, tensor_sha = native_pixels(Image.new('RGB', (93, 117), color))
    path = tmp_path / (str(identity) + '.png'); image.save(path)
    options = ['same', '{}', 'same', 'other']; prompt = native_prompt('What does this unit image show?', options)
    return dict(id='unit:' + str(identity), original_question_id='unit:' + str(identity), image_id=identity,
        image=str(path), protocol=PROTOCOL, split='test', purpose='evaluation_only', admitted_to_training=False,
        question='What does this unit image show?', options=options, answer_index=0, original_ABC_labels=False,
        model_pixel_sha256_448=rgb_sha, model_input_tensor_sha256_448=tensor_sha, source_category='unit',
        whole_native_prompt=prompt, complete_native_prompt_ids=[1, *tokenizer.encode(prompt).ids],
        native_generation_budget=32, original_options_not_pruned=True,
        input_gold_labels_or_annotations_exposed_as_text=False)


@pytest.fixture(scope='module')
def native():
    torch.set_num_threads(1)
    tokenizer = Tokenizer.from_file(str(Path(__file__).resolve().parents[1] / 'data/tokenizer-corpus/tokenizer.json'))
    config = ModelConfig(vocab_size=32768, hidden_size=32, num_layers=1, num_heads=4, num_kv_heads=2,
        intermediate_size=64, max_text_length=512, vision_hidden_size=32, vision_intermediate_size=64,
        vision_layers=1, vision_heads=4, image_size=224, patch_size=32, pool_size=1)
    assistant = OwnDenseAssistant(new_backbone(config, 972117), pointing_rank=16, reward_hidden=8,
        router_hidden=8, detector_hidden=8, word_temporal=8, word_expansion=1).eval()
    install(assistant)
    return DenseNativeView(assistant).eval(), tokenizer


def test_real_own_native_image_forward_gold_invariance_and_cached_greedy_agree(tmp_path, native):
    model, tokenizer = native; first = case(tmp_path, tokenizer, 1, 'blue'); wrong = case(tmp_path, tokenizer, 2, 'red')
    original_prefill = model.prefill; delivered = []
    def observed(*args, **kwargs):
        delivered.append(kwargs['pixel_values'].detach().clone())
        return original_prefill(*args, **kwargs)
    model.prefill = observed
    before = [(p._version, p.data_ptr()) for p in model.parameters()]
    try:
        a = evaluate_case(model, tokenizer, first, wrong)
        changed = copy.deepcopy(first); changed['answer_index'] = 3
        b = evaluate_case(model, tokenizer, changed, wrong)
    finally: del model.prefill
    assert len(delivered) == 6
    assert torch.equal(delivered[0], load_pixels(first)) and torch.equal(delivered[1], gray_pixels())
    assert torch.equal(delivered[2], load_pixels(wrong))
    for condition in a['conditions']:
        assert a['conditions'][condition]['actual_generated_ids'] == b['conditions'][condition]['actual_generated_ids']
        assert a['conditions'][condition]['choice_scores'] == b['conditions'][condition]['choice_scores']
        assert a['conditions'][condition]['candidate_token_evidence'] == b['conditions'][condition]['candidate_token_evidence']
    assert a['conditions']['proper_image']['choice_scores'] != a['conditions']['wrong_image']['choice_scores']
    ids = torch.tensor([first['complete_native_prompt_ids']])
    actual = model.generate(ids, pixel_values=load_pixels(first), max_new_tokens=32, eos_token_id=2, use_cache=True)
    assert actual[0, ids.shape[1]:].tolist() == a['conditions']['proper_image']['actual_generated_ids']
    assert before == [(p._version, p.data_ptr()) for p in model.parameters()] and all(p.grad is None for p in model.parameters())
    verify_record(tokenizer, first, wrong, a)
    for field in ['image', 'scores', 'text', 'short_actions']:
        bad = copy.deepcopy(a); row = bad['conditions']['proper_image']
        if field == 'image': row['actual_model_input_tensor_sha256_448'] = '0' * 64
        elif field == 'scores': row['choice_scores']['direct_letter'][0] += 1
        elif field == 'text': row['actual_generated_response'] = 'fabricated'
        else: row['actual_generated_ids'] = [7]; row['actual_generated_response'] = tokenizer.decode([7]); row['actual_greedy_token_logprobs'] = [-1.]
        with pytest.raises(ValueError): verify_record(tokenizer, first, wrong, bad)


def test_exact_duplicate_equivalence_stays_separate_and_source_replacement_is_gold_free(tmp_path, native):
    _, tokenizer = native
    a = case(tmp_path, tokenizer, 1, 'blue'); b = case(tmp_path, tokenizer, 2, 'red')
    scores = {'spaced_letter': [-2., -3., -1., -4.], 'direct_letter': [-2., -3., -1., -4.]}
    result = verdict(a, 'C', scores)
    assert not result['strict_correct'] and result['exact_duplicate_equivalent_correct']
    assert not result['direct_letter_strict_correct'] and result['direct_letter_exact_duplicate_equivalent_correct']
    assert result['strict_chance'] == .25 and result['duplicate_equivalent_chance'] == .5
    selected = wrong_images([a, b]); assert selected[a['id']]['id'] == b['id']
    c = copy.deepcopy(a); c['answer_index'] = 3
    assert wrong_images([c, b])[c['id']]['id'] == b['id']
    with pytest.raises(ValueError): wrong_images([a, a])
    bad = copy.deepcopy(a); bad['split'] = 'train'
    with pytest.raises(ValueError): native_inputs(tokenizer, bad)
    bad = copy.deepcopy(a); bad['complete_native_prompt_ids'] = [1, 7]
    with pytest.raises(ValueError): native_inputs(tokenizer, bad)
    bad = copy.deepcopy(a); bad['model_input_tensor_sha256_448'] = '0' * 64
    with pytest.raises(ValueError): load_pixels(bad)


def test_empty_publisher_question_files_are_distinct_from_nonempty_or_absent(tmp_path):
    base = dict(original_image_available=True, image_id=1, original_question_count=1, original_question_file_sha256='a', native_variants={})
    rows = [base, dict(base, original_question_count=0), dict(base, original_question_count=0, original_question_file_sha256=None),
        dict(base, image_id=4325, original_image_available=False, original_question_count=0)]
    assert dispositions(rows) == dict(present_with_nonempty_questions=1, present_with_empty_question_file=1,
        present_without_question_file=1, absent_image=1)
    with pytest.raises(FileNotFoundError): FullAI2D(tmp_path)
