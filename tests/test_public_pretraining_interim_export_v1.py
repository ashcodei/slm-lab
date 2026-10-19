"""Actual tiny neural partial export and immutable live-path replacement."""
import copy
import json
import os
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from slm_lab.async_durable_checkpoint_v1 import freeze_cpu
from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.corpus import file_hash
from slm_lab.deterministic_positions_v1 import install
from slm_lab.public_pretraining_checkpoint_v1 import make_state
from slm_lab.public_pretraining_interim_export_v1 import pin,resources,validate_source,inference_package,exact_values
from slm_lab.public_pretraining_native_v1 import load
from slm_lab.public_pretraining_update_v1 import update
from test_public_pretraining_checkpoint_v1 import fixture


def state_fixture(tmp_path):
    model,opt,data,schedule,plan,settings=fixture();install(model)
    vocab={'<pad>':0,'<bos>':1,'<eos>':2,'<unk>':3}
    vocab.update({'t'+str(i):i for i in range(4,model.config.vocab_size)})
    tok=Tokenizer(WordLevel(vocab,unk_token='<unk>'));tok.save(str(tmp_path/'tokenizer.json'))
    plan.update(complete_code_first_inputs=dict(checkpoint='deliberately-missing-ancestor.pt',checkpoint_sha256='a'*64,plan='missing-plan',plan_sha256='a'*64),
        tokenizer_sha256=file_hash(tmp_path/'tokenizer.json'),source_schedule_sha256=schedule.identity_sha256,
        public_raw_admission_sha256='c'*64,public_function_admission_sha256='d'*64)
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));digest=file_hash(path);records=[]
    for i in range(2):records.append(update(model,opt,data,schedule,i,settings))
    state=freeze_cpu(make_state(model,opt,plan,digest,2,records,[capture_rng('cpu')],'train',schedule=schedule,data=data))
    model.eval();return model,state,data,schedule,plan,digest,path


def test_actual_partial_model_cold_load_all_values_text_image_and_word_exact_without_ancestor_or_source_mutation(tmp_path):
    model,state,data,schedule,plan,digest,path=state_fixture(tmp_path)
    validate_source(state,plan,digest,schedule,data,state['core']['metrics'],world=1,device_profile='cpu')
    old=state['core']['model']['model']['network'];assert old['embedding.weight'] is not old['lm_head.weight']
    package=inference_package(state,plan,digest);assert package['complete_dense']['network']['embedding.weight'] is package['complete_dense']['network']['lm_head.weight']
    assert old['embedding.weight'] is not old['lm_head.weight'] and old['embedding.weight'].data_ptr()!=old['lm_head.weight'].data_ptr()
    target=tmp_path/'model.pt';torch.save(package,target);before=torch.get_rng_state().clone()
    cold,metadata=load(target,path,'step-2',expected_checkpoint_sha256=file_hash(target),expected_plan_sha256=digest)
    assert metadata['completed_updates']==2 and torch.equal(before,torch.get_rng_state())
    assert exact_values(cold,package)['every_registered_network_head_and_tied_table_value_bit_exact']
    ids=torch.tensor([[1,4,7,8,2]]);pixels=torch.linspace(-1,1,3*32*32).reshape(1,3,32,32);word=torch.linspace(-1,1,3*8*32).reshape(1,3,8,32)
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
        for image in [None,pixels]:assert torch.equal(model(ids,pixel_values=image)['logits'],cold(ids,pixel_values=image)['logits'])
        assert torch.equal(model.word_logits(word),cold.word_logits(word))
    assert not torch.cuda.is_initialized()


def test_atomic_writer_path_replacement_preserves_actual_pinned_old_tensor_file(tmp_path):
    path=tmp_path/'resume.pt';torch.save({'value':torch.tensor([17])},path)
    with pin(path) as pinned:
        old=pinned.load();before=pinned.identity();replacement=tmp_path/'replacement.pt';torch.save({'value':torch.tensor([23])},replacement)
        os.replace(replacement,path);pinned.check()
        assert old['value'].item()==17 and pinned.load()['value'].item()==17
        assert pinned.identity()==before and path.stat().st_ino!=before['inode']
        assert torch.load(path,weights_only=True)['value'].item()==23
    assert old['value'].item()==17 # mmap remains consistent after descriptor close too


def test_in_place_change_and_symbolic_checkpoint_alias_rejected(tmp_path):
    path=tmp_path/'resume.pt';torch.save({'value':torch.tensor([17])},path)
    with pytest.raises(ValueError,match='modified in place'):
        with pin(path):
            with path.open('ab') as stream:stream.write(b'changed')
    alias=tmp_path/'alias.pt';alias.symlink_to(path)
    with pytest.raises(ValueError,match='actual regular'):
        with pin(alias):pass


def test_corrupt_source_cursor_target_mask_clock_private_population_and_resource_shortfall_rejected(tmp_path):
    model,state,data,schedule,plan,digest,path=state_fixture(tmp_path);journal=state['core']['metrics']
    changes=[lambda s:s.update(schedule_identity_sha256='0'*64),lambda s:s['core'].update(world_size=2),
        lambda s:s['core']['model']['model']['sources'].update(joint_run_plan_sha256='0'*64),
        lambda s:s['core']['metrics'][0].update(response_EOS_targets=0),
        lambda s:next(iter(s['core']['optimizer']['state'].values()))['step'].fill_(999),lambda s:s['core'].update(rngs=[])]
    for change in changes:
        bad=copy.deepcopy(state);change(bad)
        with pytest.raises(ValueError):validate_source(bad,plan,digest,schedule,data,bad['core']['metrics'],world=1,device_profile='cpu')
    GiB=1024**3;stat=dict(anon=70*GiB,shmem=32*GiB,kernel=GiB)
    assert resources(60*GiB,12*GiB,stat,256*GiB)['required_free_bytes']==57*GiB
    with pytest.raises(ValueError):resources(56*GiB,12*GiB,stat,256*GiB)
    with pytest.raises(ValueError):resources(60*GiB,12*GiB,{**stat,'anon':170*GiB},256*GiB)
