from copy import deepcopy
import ast
import hashlib
import io
import tarfile

import pytest

from slm_lab.git_function_readback_v1 import original_files,qualify_file,qualify_function,verify_candidate,observe,observations_pass
from slm_lab.public_git_function_sources_v1 import blob_sha,source_entry,repository_function
from test_public_function_contracts_v1 import PROGRAM
from test_public_pretraining_intake_v1 import fixture


def example():
    tokenizer,overlap,_,_=fixture();raw=PROGRAM.encode()
    git=dict(path='algorithms/sample.py',size=len(raw),sha=blob_sha(raw),type='blob',mode='100644')
    original,reason=qualify_file(git['path'],raw,git,'author/project','a'*40,'b'*64,tokenizer,overlap)
    assert reason is None
    other,reason=source_entry(git['path'],raw,git,repository='author/project',revision='a'*40,
        repository_license_sha256='b'*64,tokenizer=tokenizer,overlap=overlap)
    assert reason is None and original==other
    expected,reason=qualify_function(original,ast.parse(PROGRAM).body[0],tokenizer,overlap)
    produced,reason=repository_function(original,ast.parse(PROGRAM).body[0],tokenizer,overlap)
    produced.pop('source_original_file');produced['original_source_document_id']=original['id'];produced['original_source_git_path']=git['path']
    verify_candidate(produced,expected)
    return expected,original


def test_independent_original_source_signature_docstring_native_and_publisher_partition_rebuild():
    candidate,original=example()
    assert candidate['original_publisher_partition'] is None
    assert candidate['role']==original['role']
    assert candidate['whole_native_labels'][-1]==2 and candidate['response']==ast.get_source_segment(PROGRAM,ast.parse(PROGRAM).body[0])
    assert original['original_UTF8_sha256']==hashlib.sha256(PROGRAM.encode()).hexdigest()


def test_changed_native_EOS_original_field_or_invented_publisher_split_is_rejected():
    candidate,_=example()
    for field,value in [('original_publisher_partition','train'),('role','invalid'),('original_signature','def wrong():')]:
        bad=dict(candidate,**{field:value})
        with pytest.raises(ValueError,match='must rebuild'):verify_candidate(bad,candidate)
    bad=deepcopy(candidate);bad['whole_native_labels'][-1]=-100
    with pytest.raises(ValueError,match='must rebuild'):verify_candidate(bad,candidate)


def test_two_actual_fresh_public_example_runs_and_entry_calls_rebuild_without_hiding_docstring_answers():
    candidate,_=example();actual=[observe(candidate),observe(candidate)]
    assert observations_pass(candidate,actual)
    wrong=deepcopy(actual);wrong[0]['entry_body_calls']+=1
    with pytest.raises(ValueError,match='actual raw observation'):observations_pass(candidate,wrong)
    wrong=deepcopy(actual);wrong[1]['raw_results'][0]['stdout']='incorrect\n'
    with pytest.raises(ValueError,match='actual raw observation'):observations_pass(candidate,wrong)
    wrong=deepcopy(actual);wrong[0]['status']='runtime_qualification';wrong[1]['source_id']='wrong'
    with pytest.raises(ValueError,match='source/program/example identity'):observations_pass(candidate,wrong)


def test_independent_archive_reader_checks_all_original_files_and_git_blob_bytes(tmp_path):
    raw=PROGRAM.encode();revision='c'*40;path=tmp_path/'source.tar.gz'
    with tarfile.open(path,'w:gz') as archive:
        info=tarfile.TarInfo('Python-'+revision+'/source.py');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    tree=dict(truncated=False,tree=[dict(path='source.py',size=len(raw),sha=blob_sha(raw),type='blob',mode='100644')])
    assert original_files(path,tree,revision)['source.py'][0]==raw
    tree['tree'][0]['sha']='b'*40
    with pytest.raises(ValueError,match='differ from pinned Git blob'):original_files(path,tree,revision)
