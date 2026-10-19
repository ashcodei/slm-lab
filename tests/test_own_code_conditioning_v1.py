from contextlib import nullcontext
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from slm_lab.own_code_conditioning_v1 import score, sequences, summarize
from slm_lab.own_code_delta_v1 import OwnCodeDelta
from test_own_code_delta_v1 import own_models, components


def test_actual_own_routes_match_independent_shifted_complete_target_logits_and_keep_weights_rng():
    general, code = own_models(); model = OwnCodeDelta(general, components(general, code, 2), reserved_parameters=0).eval()
    ids = [1, 4, 7, 8, 9, 2]; labels = [-100, -100, -100, 8, 9, 2]
    seq = dict(input_ids=ids, labels=labels, prefix_tokens=3, target_tokens=3,
        sequence_sha256=hashlib.sha256(json.dumps([ids, labels], separators=(',', ':')).encode()).hexdigest(),
        target_id='a', question_id='a', target_mode='function', condition='proper')
    before = {n: p.clone() for n, p in model.named_parameters()}; rng = torch.get_rng_state().clone()
    for route in ['general', 'code']:
        result = score(model.route(route), seq, device='cpu')
        with torch.no_grad():
            logits = model.route(route)(torch.tensor([ids]))['logits'][0]
        expected = F.cross_entropy(logits[[2, 3, 4]].float(), torch.tensor([8, 9, 2]), reduction='none')
        assert result['per_target_NLL'] == expected.tolist()
        assert result['target_ids'] == [8, 9, 2]
        assert result['teacher_argmax_ids'] == logits[[2, 3, 4]].argmax(-1).tolist()
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())
    with pytest.raises(ValueError, match='Whole source-bound'):
        score(model.code, dict(seq, labels=[-100]*4+[9, 2]), device='cpu')


def test_held_sources_reject_before_encoding_and_whole_response_matrices_include_every_control():
    class Tokenizer:
        def encode(self, value):
            return SimpleNamespace(ids=[4, 5])
    with pytest.raises(ValueError, match='publisher-train'):
        sequences([dict(split='validation', publisher_split='train')], None)
    rows = [dict(id=str(i), split='train', publisher_split='train', mode='function',
        prefix_ids=[1, 7+i], original_human_program='whole source', target_tokens=3) for i in range(2)]
    matrix = sequences(rows, Tokenizer(), max_text_length=64)
    assert len(matrix) == 6
    assert {(r['target_id'], r['question_id']) for r in matrix} == {('0', '0'), ('0', '1'), ('0', None), ('1', '0'), ('1', '1'), ('1', None)}
    assert all(r['input_ids'][-3:] == [4, 5, 2] and r['labels'][-3:] == [4, 5, 2] for r in matrix)
    with pytest.raises(ValueError, match='without clipping'):
        sequences(rows, Tokenizer(), max_text_length=4)


def test_matrix_reader_distinguishes_program_and_token_means_and_rejects_missing_or_corrupt_cells():
    rows = [dict(id='a', split='train', publisher_split='train', mode='function', target_tokens=1),
            dict(id='b', split='train', publisher_split='train', mode='function', target_tokens=3)]
    records = []
    for target, n in [('a', 1), ('b', 3)]:
        for question in ['a', 'b', None]:
            value = (1. if target == 'a' else 3.) if target == question else 5.
            records.append(dict(route='code', target_id=target, question_id=question, target_tokens=n,
                per_target_NLL=[value]*n, target_ids=[2]*n, teacher_argmax_ids=[2]*n,
                whole_response_EOS_sum_NLL=value*n, whole_response_EOS_mean_NLL=value,
                teacher_argmax_correct=n, first_target_NLL=value))
    result = summarize(records, rows, routes=['code'])['code']
    assert result['proper_equal_program_mean_NLL'] == 2.
    assert result['proper_token_mean_NLL'] == 2.5
    assert result['matching_question_best_likelihood_in_same_interface'] == 2
    assert result['mean_other_question_minus_proper_NLL'] == 3.
    with pytest.raises(ValueError, match='exactly once'):
        summarize(records[:-1], rows, routes=['code'])
    corrupted = [dict(r) for r in records]; corrupted[0]['teacher_argmax_correct'] = 0
    with pytest.raises(ValueError, match='reconstruct'):
        summarize(corrupted, rows, routes=['code'])
