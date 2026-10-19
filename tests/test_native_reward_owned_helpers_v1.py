"""Real small complete reward fitting/attachment and strict full-run admission."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.corpus import file_hash
from slm_lab.native_human_reward_v1 import NativeRewardHeadStage, signature
from slm_lab.native_human_reward_v1 import FrozenSharedNativeReward
from slm_lab.native_reward_bundle_v1 import attach
from slm_lab.native_reward_capacity_v1 import cases, prove
from slm_lab.native_reward_fitting_v1 import fit
from slm_lab.native_reward_owned_recipe_v1 import completed_joint
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_native_human_reward_v1 import setup
from test_own_aligned_assistant_v1 import original_signature

ROOT = Path(__file__).resolve().parents[1]


def fitted():
    base, train, shards, features, setting, _ = setup()
    _, held, held_shards, _, _, _ = setup('valid')
    result = fit(base, train, held, shards, held_shards, features.sources, setting, allow_mechanics=True)
    return base, result, features.sources, setting


def test_complete_real_head_fit_replays_capacity_then_resets_to_original_before_full_unrepeated_epoch():
    actual, result, _, _ = fitted(); expected, _, _, features, setting, optimizer = setup()
    stage = NativeRewardHeadStage(expected, optimizer, features, setting)
    for _ in range(setting['steps']): stage.update()
    assert tensor_digest(actual.heads['reward'].named_parameters()) == tensor_digest(expected.heads['reward'].named_parameters())
    assert result['actual_head_recovery']['actual_physical_head_Adam_calls'] == 3
    assert result['actual_head_recovery']['exact_next_update_weights_Adam_RNG_and_metric_replay']
    assert result['actual_head_recovery']['corrupt_Adam_rejected_before_mutation']
    assert result['bundle']['head']['actual_train_pairs'] == 5
    assert result['initial_held']['overall']['pairs'] == result['final_held']['overall']['pairs'] == 5
    assert result['bundle']['training_state']['metrics'][-1]['pairs'] == 1


def test_actual_head_only_bundle_attachment_requires_no_training_features_and_preserves_RNG_and_other_native_values(tmp_path):
    actual, result, sources, options = fitted(); cold = setup()[0]; path = tmp_path/'head.pt'
    torch.save(result['bundle'], path); fixed = [(n, p) for n, p in cold.named_parameters() if not n.startswith('heads.reward.')]
    before, rng = signature(p for _, p in fixed), value_digest(capture_rng('cpu'))
    scorer = attach(cold, path, expected_sha256=file_hash(path), expected_sources=sources, expected_options=options, allow_mechanics=True)
    assert signature(p for _, p in fixed) == before and value_digest(capture_rng('cpu')) == rng
    ids = torch.tensor([[1, 4, 8, 9, 2]])
    assert torch.equal(scorer.score(ids), actual.preference_score(ids).reshape(()))
    assert scorer.counts()['additional_values_over_same_native_policy'] == 0
    with pytest.raises(ValueError): attach(setup()[0], path, expected_sha256=file_hash(path), expected_sources=sources, expected_options=options)


def test_partial_wrong_source_head_or_Adam_bundle_rejects_before_native_or_RNG_mutation(tmp_path):
    _, result, sources, options = fitted()
    changes = [lambda x: x['head'].update(actual_train_pairs=4),
        lambda x: x['sources'].update(native_checkpoint_sha256='0'*64),
        lambda x: x['training_state']['metrics'][0]['source_positions'].reverse(),
        lambda x: x['head']['learned_head']['0.weight'].add_(1),
        lambda x: x['training_state']['optimizer']['state'][0]['exp_avg'].fill_(float('nan'))]
    for i, change in enumerate(changes):
        cold = setup()[0]; altered = copy.deepcopy(result['bundle']); change(altered); path = tmp_path/(str(i)+'.pt'); torch.save(altered, path)
        before, rng = original_signature(cold), value_digest(capture_rng('cpu'))
        with pytest.raises(ValueError): attach(cold, path, expected_sha256=file_hash(path), expected_sources=sources, expected_options=options, allow_mechanics=True)
        assert original_signature(cold) == before and value_digest(capture_rng('cpu')) == rng


def test_capacity_selects_actual_maximum_padding_and_distinct_longest_whole_train_batch():
    data = SimpleNamespace(family='preferences', role='train', purpose='optimization',
        manifest=dict(config=dict(max_text_length=24)))
    lengths = [(3, 16), (8, 8), (8, 8)]
    data.rows = [dict(native={f:dict(length=n) for f,n in zip(['chosen','rejected'],pair)}) for pair in lengths]
    class Rows:
        def __len__(self): return 3
    shaped = Rows(); shaped.__dict__.update(data.__dict__)
    selected = cases(shaped, 0, 1, dict(token_budget=24, maximum_sequences=3))
    assert len(selected) == 2
    assert [(len(b), max(r[2] for r in b)) for b in selected] == [(3, 8), (1, 16)]
    shaped.role = 'valid'; shaped.purpose = 'evaluation'
    with pytest.raises(ValueError, match='train-only'): cases(shaped, 0, 1, dict(token_budget=24, maximum_sequences=3))


def test_actual_small_native_capacity_forward_is_not_full_GPU_admission():
    base, data, shards, _, _, _ = setup(); options = dict(token_budget=base.config.max_text_length, maximum_sequences=3)
    before = original_signature(base); rng = value_digest(capture_rng('cpu'))
    for rank in range(2):
        result = prove(base, data, rank, 2, shards[0]['sources'], options, physical_GPU=rank, allow_mechanics=True)
        assert not result['full_size_GPU_feature_capacity_passed'] and result['actual_device_type'] == 'cpu'
        assert result['actual_complete_capacity_cases'] and result['all_values_metadata_and_private_RNG_unchanged']
    assert original_signature(base) == before and value_digest(capture_rng('cpu')) == rng


def test_live_exact_predecessor_stays_waiting_and_missing_owner_cannot_replace_full_joint_evidence(tmp_path):
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    (tmp_path/'waiter-plan.json').write_text(json.dumps(dict(owner=owner)))
    assert completed_joint(tmp_path, owner) is None
    with pytest.raises(ValueError, match='exact original full joint owner'):
        completed_joint(tmp_path, dict(pid=owner['pid'], ticks='1'))
    absent = dict(pid=99999971, ticks='1'); (tmp_path/'waiter-plan.json').write_text(json.dumps(dict(owner=absent)))
    with pytest.raises(ValueError, match='without all complete'): completed_joint(tmp_path, absent)


def test_recipe_import_remains_non_neural_and_cannot_acquire_CUDA():
    result = subprocess.run([sys.executable, '-c', "import sys;import slm_lab.native_reward_owned_recipe_v1;assert 'torch' not in sys.modules;print('non-neural')"],
        cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and result.stdout.strip() == 'non-neural'


def test_real_trained_head_and_complete_native_roundtrip_keep_one_shared_encoder_and_exact_text_scores(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from slm_lab.own_dense_standalone_v1 import export as export_native, load as load_native
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    base, result, sources, options = fitted()
    vocab = {'<pad>':0,'<bos>':1,'<eos>':2, **{str(i):i for i in range(3,base.config.vocab_size)}}
    tokenizer = tmp_path/'tokenizer.json'; Tokenizer(WordLevel(vocab=vocab,unk_token='<pad>')).save(str(tokenizer))
    path = tmp_path/'model.pt'; torch.save(export_native(base,sources=sources,provenance=dict(kind='small CPU learned reward export control'),
        completed_updates=options['steps'],tokenizer_sha256=file_hash(tokenizer)),path)
    cold,_ = load_native(path,expected_checkpoint_sha256=file_hash(path),expected_sources=sources)
    scorer = FrozenSharedNativeReward(cold,result['bundle']['head'],allow_mechanics=True)
    policy = OwnDenseActor(cold,rank=2).eval(); ids = torch.tensor([[1,4,8,9,2]])
    union = {id(p):p for m in [policy,scorer] for p in m.parameters()}
    assert sum(p.numel() for p in union.values()) == policy.counts()['actual_unique']
    with torch.no_grad():
        assert torch.equal(base(ids)['logits'],cold(ids)['logits'])
        assert torch.equal(base.preference_score(ids).reshape(()),scorer.score(ids))
