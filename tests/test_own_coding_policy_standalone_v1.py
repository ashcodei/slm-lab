import copy
import json
import os
import subprocess
import sys

import pytest
import torch
from tokenizers import Tokenizer, models

from slm_lab.corpus import file_hash
from slm_lab.deterministic_positions_v1 import install
from slm_lab.own_coding_policy_standalone_v1 import export, load
from slm_lab.serial_public_code_bf16_step_v1 import update
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_serial_public_code_bf16_step_v1 import setup
from test_parallel_public_reasoned_code_SFT_stage_v1 import SOURCES


def trained(tmp_path):
    model, adam, data = setup(tmp_path)
    install(model.base)
    for positions in [[0,1,2], [2]]:
        update(model, adam, data, positions, 'plain_tokens', allow_mechanics=True)
    model.eval()
    folder = tmp_path/'standalone'
    folder.mkdir()
    vocab = {t:i for i,t in enumerate(['<pad>','<bos>','<eos>','<unk>']+[f't{i}' for i in range(4,32)])}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token='<unk>'))
    tokenizer.save(str(folder/'tokenizer.json'))
    payload = export(model, sources=SOURCES, tokenizer_sha256=file_hash(folder/'tokenizer.json'),
        completed_updates=2, provenance={'controlled_small_two_update_coding_policy':True})
    path = folder/'model.pt'
    torch.save(payload, path)
    return model, payload, path


def test_trained_complete_unfused_roundtrip_preserves_all_values_modalities_actions_and_RNG(tmp_path):
    model, payload, path = trained(tmp_path)
    ids = torch.tensor([[1,4,5]])
    pixels = torch.linspace(-1,1,3*16*16).reshape(1,3,16,16)
    words = torch.linspace(-1,1,3*16*16).reshape(1,3,16,16)
    before = torch.get_rng_state().clone()
    cold, metadata = load(path, expected_checkpoint_sha256=file_hash(path), expected_sources=SOURCES)
    assert torch.equal(before, torch.get_rng_state())
    assert tensor_digest(model.named_parameters()) == tensor_digest(cold.named_parameters())
    assert metadata['actual_counts']['actual_unique'] == sum(p.numel() for p in cold.parameters())
    assert metadata['no_ancestor_checkpoint_optimizer_or_training_reference_loaded']
    assert model.base.network.embedding.weight is model.base.network.lm_head.weight
    assert cold.base.network.embedding.weight is cold.base.network.lm_head.weight
    assert not ({p.data_ptr() for p in model.parameters()} & {p.data_ptr() for p in cold.parameters()})
    with torch.no_grad():
        for image in [None,pixels]:
            a, b = model(ids, pixel_values=image), cold(ids, pixel_values=image)
            assert all(torch.equal(a[k],b[k]) for k in ['logits','values'])
            assert torch.equal(model.generate(ids,image,max_new_tokens=4,eos_token_id=2), cold.generate(ids,image,max_new_tokens=4,eos_token_id=2))
        assert torch.equal(model.base.word_logits(words), cold.base.word_logits(words))


def test_missing_head_critic_geometry_counts_or_reward_claim_cannot_cold_load(tmp_path):
    model, payload, path = trained(tmp_path)
    mutations = [lambda r:r['policy']['parts'].pop('critic.weight'),
        lambda r:r['complete_dense']['heads'].pop(next(iter(r['complete_dense']['heads']))),
        lambda r:r['actual_counts'].__setitem__('actual_unique',10**9),
        lambda r:r['policy']['config'].__setitem__('max_text_length',64),
        lambda r:r.__setitem__('compatible_trained_reward_model_or_critic_quality_certified',True)]
    for change in mutations:
        raw = copy.deepcopy(payload)
        change(raw)
        torch.save(raw,path)
        before = torch.get_rng_state().clone()
        with pytest.raises(ValueError):load(path, expected_checkpoint_sha256=file_hash(path), expected_sources=SOURCES)
        assert torch.equal(before,torch.get_rng_state())


def test_fresh_documented_query_process_needs_only_complete_model_and_tokenizer(tmp_path):
    model, payload, path = trained(tmp_path)
    result = subprocess.run([sys.executable,'-m','slm_lab.own_coding_policy_standalone_v1',
        '--checkpoint',str(path),'--prompt','t4','--raw-prompt','--max-new-tokens','4',
        '--precision','fp32','--threads','1','--json'],capture_output=True,text=True,check=True,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    report = json.loads(result.stdout)
    assert report['actual_registered_parameters'] == model.counts()['actual_unique']
    assert report['completed_checkpoint_updates'] == 2 and report['no_ancestor_checkpoint_needed']
    with torch.no_grad():expected=model.generate(torch.tensor([[1,4]]),max_new_tokens=4,eos_token_id=2)
    assert report['actual_generated_token_ids'] == expected[0,2:].tolist()
