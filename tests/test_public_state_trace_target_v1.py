from pathlib import Path

import pytest
from tokenizers import Tokenizer

from slm_lab.public_python_state_trace_v1 import capture
from slm_lab.public_state_trace_target_v1 import compact,expand,target


def test_real_observed_program_target_uses_own_tokens_full_states_and_response_EOS_mask():
    root=Path(__file__).resolve().parents[1];tokenizer=Tokenizer.from_file(str(root/'data/tokenizer-corpus/tokenizer.json'))
    code='n = int(input())\ntotal = 0\nfor i in range(n):\n    total += i\nprint(total)\n'
    observed=capture(code,'3\n');assert observed['eligible_complete_trace']
    raw=observed['actual_observations'];assert expand(compact(raw['events']))==raw['events']
    result,reason=target(code,'3\n',raw,tokenizer);assert result is not None and reason is None
    ids,labels=result['ids'],result['labels'];prefix=[1]+tokenizer.encode('User: '+result['prompt']+'\nAssistant:').ids
    assert ids==prefix+tokenizer.encode(result['response']).ids+[2] and len(ids)<=4096
    assert labels==[-100]*len(prefix)+ids[len(prefix):] and labels[-1]==2
    assert '"stdout":"3\\n"' in result['response']
    assert len([v for v in labels if v!=-100])>len(tokenizer.encode('3\n').ids)+1


def test_incomplete_trace_is_rejected_and_whole_context_is_never_clipped():
    root=Path(__file__).resolve().parents[1];tokenizer=Tokenizer.from_file(str(root/'data/tokenizer-corpus/tokenizer.json'))
    code='print(input())\n';observed=capture(code,'public-input\n')['actual_observations']
    with pytest.raises(ValueError):target(code,'public-input\n',dict(observed,whole_trace_complete=False),tokenizer)
    result,reason=target(code,'public-input\n'+'distinctlongword '*9000,observed,tokenizer)
    assert result is None and reason is not None
