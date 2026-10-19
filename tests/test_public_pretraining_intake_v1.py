from copy import deepcopy
import hashlib
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from slm_lab.held_text_overlap_v1 import HeldTextOverlap
from slm_lab.public_pretraining_intake_v1 import candidate


def fixture():
    root=Path(__file__).resolve().parents[1];tokenizer=Tokenizer.from_file(str(root/'data/tokenizer-corpus/tokenizer.json'))
    scan=HeldTextOverlap([dict(audit_id='original-held',question='Find the blue crystalline mineral behind the copper statue.')])
    source=dict(family='code',repo='unit/public-source',revision='a'*40,path='unit.parquet',dataset_license='index-only')
    row=dict(code='def multiply_by_seven(value):\n    return value * 7\n',license='mit',language='Python',repo_name='unit/library',path='src/arithmetic.py')
    return tokenizer,scan,source,row


def test_whole_original_code_native_BPE_EOS_and_lossless_original_source_are_preserved():
    tokenizer,scan,source,row=fixture();original=deepcopy(row);result,why=candidate(row,source,tokenizer,scan)
    assert row==original and result['whole_native_ids'][0]==1 and result['whole_native_ids'][-1]==2
    assert result['whole_native_labels']==[-100,*result['whole_native_ids'][1:]]
    assert tokenizer.decode(result['whole_native_ids'][1:-1],skip_special_tokens=False)==row['code']
    assert result['original_UTF8_sha256']==hashlib.sha256(row['code'].encode()).hexdigest()
    assert result['optimization_admitted'] is False and result['human_preferences'] is False and result['execution_reward'] is False


def test_catalog_license_does_not_admit_unlicensed_or_disallowed_source_code():
    t,q,s,r=fixture()
    for license in [None,'gpl-3.0','unknown','']:
        row=dict(r,license=license);result,why=candidate(row,s,t,q)
        assert result is None and why=='per_file_license_not_in_declared_allowlist'


def test_original_held_stems_benchmark_paths_and_whole_invalid_programs_are_qualified():
    t,q,s,r=fixture()
    result,why=candidate(dict(r,code=r['code']+'# Find the blue crystalline mineral behind the copper statue.\n'),s,t,q)
    assert result is None and why=='declared_held_stem_or_lexical_copy_signal'
    result,why=candidate(dict(r,path='benchmarks/mbpp/original.py'),s,t,q)
    assert result is None and why=='recognizable_held_benchmark_path'
    result,why=candidate(dict(r,code='def wrong_function(:\n    return value * 7\n'),s,t,q)
    assert result is None and why=='whole_Python3_syntax_qualification'


def test_web_and_math_keep_original_text_URL_and_underlying_rights_without_human_labels():
    t,q,s,r=fixture();text='A number can be divided by a nonzero denominator to produce a rational value.\n'
    for family in ['web','math']:
        source=dict(s,family=family,dataset_license='odc-by');row=dict(text=text,url='https://example.org/number-theory')
        result,why=candidate(row,source,t,q)
        assert result['original_public_text']==text and result['original_public_URL']==row['url']
        assert result['rights']['underlying_page_rights_not_relicensed']
        assert result['human_preferences'] is False and result['optimization_admitted'] is False
