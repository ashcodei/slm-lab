from copy import deepcopy
import gzip
import hashlib
import json
import struct

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from slm_lab.codenet_token_pack_v1 import CodePackWriter, CompleteCodePack
from slm_lab.public_reasoned_code_native_readback_v1 import verify


def physical_pack(tmp_path, role='train'):
    source = dict(id='source-fixture', split='train', solution='def square(x):\n    return x*x\n',
        r1_generation='Square the integer. λ\n```python\ndef square(x):\n    return x*x\n```\n\n')
    prompt = 'Return a Python solution for squaring an integer.'
    response = source['r1_generation']
    tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator([prompt, response], trainers.BpeTrainer(vocab_size=512,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=['<pad>', '<bos>', '<eos>', '<unk>'], show_progress=False))
    text = 'User: ' + prompt + '\nAssistant:'
    head, target = tokenizer.encode(text, add_special_tokens=False).ids, tokenizer.encode(response, add_special_tokens=False).ids
    ids = [1] + head + target + [2]
    start = len(head) + 1
    labels = [-100]*start + ids[start:]
    fixed = dict(source_role=role, role_fixed_before_fresh_replay=True, optimization_forbidden=role=='reserved',
        declared_duplicate_component_sha256='c'*64)
    native = dict(complete_context_tokens=len(ids), response_EOS_targets=len(target)+1, target_start=start,
        native_input_u16le_sha256=hashlib.sha256(struct.pack('<'+'H'*len(ids), *ids)).hexdigest(),
        native_target_mask_u8_sha256=hashlib.sha256(bytes([0]*start+[1]*(len(target)+1))).hexdigest())
    row = dict(id='fixture:reasoning', view='reasoning', original_source_id=source['id'], original_fixed_role=fixed,
        split='train' if role=='train' else 'validation', group_id='c'*64, publisher_split='train',
        prompt=prompt, response=response, original_response_utf8_sha256=hashlib.sha256(response.encode()).hexdigest(),
        admitted_use='weak_supervised_coding_only', model_generated_examples=True, human_preferences=False,
        own_model_on_policy=False, use_as_RL_reward_oracle_forbidden=True, external_teacher_weights_loaded_by_this_pipeline=False,
        live_teacher_API_used=False, original_native_metadata=native)
    joined = dict(views=dict(reasoning=dict(prompt=prompt, entire_context_tokens=len(ids),
        prompt_tokens_including_BOS=start, entire_response_tokens_including_EOS=len(target)+1)))
    for name in ['model-config.json','DATASET-LICENSE.txt','publisher-README.md']:
        (tmp_path/name).write_text('{}\n')
    tokenizer.save(str(tmp_path/'tokenizer.json'))
    writer = CodePackWriter(tmp_path, maximum_bytes=1024**2, minimum_free_gib=0)
    writer.add(row, ids, labels)
    writer.finish(dict(control_fixture=True))
    pack = CompleteCodePack(tmp_path, row['split'], purpose='audit')
    with gzip.open(tmp_path/'original-rows.jsonl.gz','rt') as stream:
        physical_row = json.loads(next(stream))
    actual_ids, actual_labels = pack.sequence(0)
    return source, joined, fixed, physical_row, actual_ids, actual_labels, tokenizer


def test_actual_physical_pack_retains_full_reasoning_and_native_eos(tmp_path):
    parts = physical_pack(tmp_path)
    result = verify(*parts)
    assert result['original_text_and_native_arrays_exact'] and not result['training_admission_granted']
    assert result['response_EOS_targets'] > 1
    assert parts[3]['response'].endswith('\n\n') and 'λ' in parts[3]['response']


def test_self_consistent_truncation_wrong_eos_or_loss_mask_cannot_pass(tmp_path):
    original = physical_pack(tmp_path)
    for mutation in ('truncate', 'EOS', 'mask', 'response'):
        parts = deepcopy(original)
        if mutation == 'truncate':
            parts[4].pop(-2)
            parts[5].pop(-2)
        elif mutation == 'EOS':
            parts[4][-1] = parts[5][-1] = 4
        elif mutation == 'mask':
            parts[5][0] = 1
        else:
            parts[3]['response'] = parts[3]['response'].rstrip()
        with pytest.raises(ValueError):
            verify(*parts)


def test_reserved_physical_pack_and_supervision_cannot_be_relabelled(tmp_path):
    parts = physical_pack(tmp_path, role='reserved')
    assert verify(*parts)['sealed_role'] == 'reserved'
    for mutation in ('train', 'RL'):
        changed = deepcopy(parts)
        if mutation == 'train':
            changed[3]['split'] = 'train'
        else:
            changed[3]['use_as_RL_reward_oracle_forbidden'] = False
        with pytest.raises(ValueError):
            verify(*changed)
