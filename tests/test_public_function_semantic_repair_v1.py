import ast
from copy import deepcopy
from types import SimpleNamespace

import pytest

from slm_lab.public_function_semantic_repair_v1 import candidates,counterexample,view
from slm_lab.public_function_rollout_observation_v1 import observe


def source():
    doc='Return x plus one.\n>>> increment(0)\n1\n>>> increment(-2)\n-1\n'
    program='def increment(x):\n    """'+doc+'"""\n    return x + 1\n'
    return dict(id='unit-public',group_id='unit-group',split='train',response=program,original_public_URL='https://example.test/source',rights={'synthetic_unit_fixture':True},
        source_candidate=dict(id='unit-public',role='train',function='increment',declared_prelude='',original_function_program=program,original_literal_docstring=doc,whole_original_example_count=2))


class BytesTokenizer:
    def encode(self,text,**kwargs):return SimpleNamespace(ids=[v+4 for v in text.encode()])
    def decode(self,ids,**kwargs):return bytes(v-4 for v in ids).decode()


def test_mutations_are_deterministic_single_operator_changes_preserving_signature_and_public_examples():
    row=source();values=candidates(row)
    assert values==candidates(row) and values
    original=ast.parse(row['response']).body[0]
    for value in values:
        changed=ast.parse(value['program']).body[0]
        assert changed.name==original.name and ast.dump(changed.args)==ast.dump(original.args)
        assert ast.get_docstring(changed,clean=False)==ast.get_docstring(original,clean=False)
    assert values[0]['kind']=='Sub' and 'x - 1' in values[0]['program']


def test_actual_logical_counterexample_and_full_original_target_masks_are_verified_in_real_confined_runtime():
    row=source();mutation=candidates(row)[0]
    assert observe(row,row['response'])['all_original_examples_passed']
    bad=observe(row,mutation['program']);feedback=counterexample(row,bad)
    assert feedback['first_failing_original_public_example_index']==0 and feedback['actual_stdout']=='-1\n'
    (candidate,ids,labels),reason=view(row,mutation,feedback,BytesTokenizer())
    start=next(i for i,v in enumerate(labels) if v!=-100)
    assert candidate['response']==row['response'] and ids[0]==1 and ids[-1]==2
    assert labels[:start]==[-100]*start and labels[start:]==ids[start:]
    assert reason['disposition']=='complete_semantic_repair_candidate'
    assert counterexample(row,observe(row,'def increment(:')) is None


def test_held_sources_and_corrupted_mutation_bytes_reject():
    row=source();bad=deepcopy(row);bad['split']='validation'
    with pytest.raises(ValueError):candidates(bad)
    mutation=candidates(row)[0];mutation['program']+='\n# changed'
    with pytest.raises(ValueError):view(row,mutation,{},BytesTokenizer())


def test_oversized_repair_is_classified_without_truncating_original_response():
    row=source();row['response']+='\n#'+('z'*5000)+'\n';row['source_candidate']['original_function_program']=row['response']
    mutation=candidates(row)[0];bad=observe(row,mutation['program']);feedback=counterexample(row,bad)
    candidate,reason=view(row,mutation,feedback,BytesTokenizer())
    assert candidate is None and reason['disposition']=='complete_native_context_exceeds4096'
    assert reason['context_tokens']>4096 and row['response'].endswith('z'*5000+'\n')
