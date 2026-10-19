from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest

from slm_lab.codenet_complete_context_v1 import CompleteCodeEncoder
from slm_lab.codenet_sft_admission_v1 import audit_source_rows
from slm_lab.codenet_token_pack_v1 import CodePackWriter,CompleteCodePack


class Tokenizer:
    def token_to_id(self,s):return {'<pad>':0,'<bos>':1,'<eos>':2}[s]
    def get_vocab_size(self):return 32768
    def encode(self,s):return SimpleNamespace(ids=list(map(ord,s)))


def build(folder,mutate=None):
    rows=[]
    for i,role in enumerate(['train','validation','test']):
        code=f'print({i+1})\n'
        rows.append(dict(id=f's{i}',original_archive_member=f'original-{i}.py',
            original_source_body_sha256=hashlib.sha256(code.encode()).hexdigest(),
            source_problem_id=f'p{i}',group_id=f'p{i}',split=role,language='Python',
            license='CDLA-Permissive-2.0',original_judge_status='Accepted',
            original_submission_metadata={'status':'Accepted'},prompt=f'Prompt {i}',response=code))
    if mutate:mutate(rows)
    for name in ['tokenizer.json','model-config.json','DATASET-LICENSE.txt','publisher-README.md']:(folder/name).write_text('{}')
    enc=CompleteCodeEncoder(Tokenizer(),SimpleNamespace(max_text_length=128,image_size=224,vocab_size=32768,image_token_count=lambda _:0),folder)
    writer=CodePackWriter(folder,minimum_free_gib=0)
    for row in rows:writer.add(row,*enc.sequence(row,'response'))
    writer.finish({'fixture':True});pack=CompleteCodePack(folder,'train')
    # Use the actual writer's exported row fields.
    import gzip,json
    with gzip.open(folder/'original-rows.jsonl.gz','rt') as f:rows=[json.loads(s) for s in f]
    return rows,pack,enc


def inspect(rows,pack,enc,held=(),native=None):return audit_source_rows(pack,rows,enc,{0,1,2},native or {},held)


def test_authentic_complete_source_reconstruction_admits_only_training_role(tmp_path):
    rows,pack,enc=build(tmp_path);r=inspect(rows,pack,enc)
    assert r['training_examples']==1 and r['original_examples']==3
    assert r['every_original_prompt_programme_and_full_target_mask_reencoded']
    assert r['cross_role_exact_programme_bodies']==0 and not r['actual_native_preparation_executed']


def test_identical_code_across_roles_is_rejected_even_with_different_prompts(tmp_path):
    def mutate(rows):
        rows[1]['response']=rows[0]['response'];rows[1]['original_source_body_sha256']=rows[0]['original_source_body_sha256']
    rows,pack,enc=build(tmp_path,mutate)
    with pytest.raises(ValueError,match='crosses sealed roles'):inspect(rows,pack,enc)


def test_held_reference_or_component_cannot_become_a_training_source(tmp_path):
    rows,pack,enc=build(tmp_path)
    with pytest.raises(ValueError,match='Held algorithm'):inspect(rows,pack,enc,held=[rows[0]])


def test_self_consistent_source_digest_cannot_hide_changed_token_targets(tmp_path):
    rows,pack,enc=build(tmp_path);rows[0]['response']='print(9)\n';rows[0]['original_source_body_sha256']=hashlib.sha256(rows[0]['response'].encode()).hexdigest()
    with pytest.raises(ValueError,match='must reconstruct'):inspect(rows,pack,enc)


def test_unverified_execution_contract_cannot_be_silently_omitted(tmp_path):
    rows,pack,enc=build(tmp_path)
    with pytest.raises(ValueError,match='native contract'):inspect(rows,pack,enc,native={rows[0]['original_archive_member']:rows[0]})


def test_internal_eos_literal_does_not_become_an_ordinary_training_target(tmp_path):
    def mutate(rows):
        rows[0]['response']='print("\x02")\n';rows[0]['original_source_body_sha256']=hashlib.sha256(rows[0]['response'].encode()).hexdigest()
    rows,pack,enc=build(tmp_path,mutate)
    with pytest.raises(ValueError,match='reserved model tokens'):inspect(rows,pack,enc)


def test_historical_nonaccepted_sources_are_not_admitted_by_metadata_claims(tmp_path):
    rows,pack,enc=build(tmp_path);rows[0]['original_submission_metadata']['status']='Wrong Answer'
    with pytest.raises(ValueError,match='Accepted text source'):inspect(rows,pack,enc)
