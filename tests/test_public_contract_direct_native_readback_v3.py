"""Actual candidate native file readback, including inherited pack metadata."""
import copy
import gzip
import json
from pathlib import Path
import sys

from tokenizers import Tokenizer
from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.contract_source_contrasts_v2 import direct

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from read_admit_public_contract_census_CPU_v3 import direct_export_exact


def fixture():
    original=ROOT/'data/direct-algorithm-function-contracts-admitted-v1';folder=ROOT/'data/contract-qualified-public-direct-candidate-v2'
    by={r['id']:r for r in map(json.loads,gzip.open(original/'original-rows.jsonl.gz','rt'))}
    rows=list(map(json.loads,gzip.open(folder/'original-rows.jsonl.gz','rt')))
    return by,rows,CompleteCodePack(folder,'train','audit'),Tokenizer.from_file(str(original/'tokenizer.json'))


def test_all_actual_qualified_direct_native_files_keep_source_bytes_and_renew_pack_metadata_exactly():
    by,rows,pack,tokenizer=fixture();changed_original_ordinal=0
    for i,row in enumerate(rows):
        original=by[row['original_source_function_id']];rebuilt,ids,labels=direct(original,tokenizer)
        assert pack.sequence(i)==(ids,labels) and direct_export_exact(rebuilt,ids,labels,i,row)
        changed_original_ordinal+=original['token_pack_index']!=i
    assert len(rows)==len(pack)==57 and changed_original_ordinal>0


def test_actual_native_metadata_mask_target_and_rights_corruption_never_accept():
    by,rows,pack,tokenizer=fixture();stored=rows[0];rebuilt,ids,labels=direct(by[stored['original_source_function_id']],tokenizer)
    for field,value in [('token_pack_index',1),('target_tokens',stored['target_tokens']-1),('complete_input_ids_sha256','0'*64),('rights',{})]:
        wrong=copy.deepcopy(stored);wrong[field]=value
        assert not direct_export_exact(rebuilt,ids,labels,0,wrong)
    wrong=list(labels);wrong[-1]=-100
    assert not direct_export_exact(rebuilt,ids,wrong,0,stored)
    wrong=list(ids);wrong[-2]=(wrong[-2]+1)%32768
    assert not direct_export_exact(rebuilt,wrong,labels,0,stored)
