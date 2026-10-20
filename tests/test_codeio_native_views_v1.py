import json

import pytest
from tokenizers import Tokenizer,decoders,models,pre_tokenizers,trainers

from slm_lab.codeio_native_views_v1 import code_view,io_views,native


def tokenizer():
    t=Tokenizer(models.BPE(unk_token='<unk>'));t.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder=decoders.ByteLevel()
    t.train_from_iterator(['an original example'],trainers.BpeTrainer(vocab_size=300,
        special_tokens=['<pad>','<bos>','<eos>','<unk>'],initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return t


def row():
    return dict(problem_description='Return the original Unicode text.',io_requirements='text: str; return str',
        refcode='def main_solution(text):\n    return text\n',funcname='main_solution',
        ios=[dict(input={'text':s},output=s) for s in ['café','λ\n\tend']])


def test_every_complete_original_case_has_two_directions_and_original_code_is_untouched():
    source=row();views=[code_view(source),*io_views(source)]
    assert len(views)==5 and views[0]['response']==source['refcode']
    assert [(v['view'],v['original_case_position']) for v in views[1:]]==[
        (name,i) for i in range(2) for name in ['output_prediction','input_prediction']]
    for i,case in enumerate(source['ios']):
        forward,reverse=views[1+2*i:3+2*i]
        assert json.loads(forward['response'])==case['output'] and json.loads(reverse['response'])==case['input']
        assert source['refcode'] in forward['prompt'] and source['refcode'] in reverse['prompt']
    bad=row();bad['ios'][1]['input']=[]
    with pytest.raises(ValueError):list(io_views(bad))


def test_native_whole_text_masks_EOS_and_oversize_are_checked_without_cropping():
    t=tokenizer();view=code_view(row());ids,labels,g=native(t,view)
    start=g['target_start']
    assert ids[0]==1 and ids[-1]==2 and labels[:start]==[-100]*start and labels[start:]==ids[start:]
    assert t.decode(ids[start:-1],skip_special_tokens=False)==view['response']
    same_ids,same_labels,small=native(t,view,limit=len(ids)-1)
    assert same_ids==ids and same_labels==labels and not small['fits_complete_native_context']
    for changed in [dict(view,response='<eos>'),dict(view,response='')]:
        with pytest.raises(ValueError):native(t,changed)
