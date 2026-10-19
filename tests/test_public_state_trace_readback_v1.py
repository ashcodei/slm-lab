from copy import deepcopy
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from slm_lab.public_python_state_trace_v1 import capture
from slm_lab.public_state_trace_target_v1 import target
from slm_lab.public_state_trace_readback_v1 import rebuild


def test_actual_public_source_states_full_native_target_and_corruption_reconstruct():
    root=Path(__file__).resolve().parents[1];tokenizer=Tokenizer.from_file(str(root/'data/tokenizer-corpus/tokenizer.json'))
    program='n=int(input())\ndef double(x):\n    return 2*x\nprint(double(n))\n';observed=capture(program,'7\n')
    expected,_=target(program,'7\n',observed['actual_observations'],tokenizer)
    actual=rebuild(program,'7\n',observed,tokenizer)
    assert all(actual[k]==expected[k] for k in ['prompt','response','ids','labels'])
    assert actual['actual_stdout']=='14\n' and actual['events']==len(observed['actual_observations']['events'])
    wrong=deepcopy(observed);wrong['actual_observations']['events'][2]['line']=99
    with pytest.raises(ValueError):rebuild(program,'7\n',wrong,tokenizer)
    with pytest.raises(ValueError):rebuild(program,'8\n',observed,tokenizer)
    with pytest.raises(ValueError):rebuild(program+'#changed\n','7\n',observed,tokenizer)
