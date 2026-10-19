import torch

from slm_lab.dense_foundation_inference_v1 import DenseNativeView
from slm_lab.deterministic_positions_v1 import install
from slm_lab.model import ModelConfig
from slm_lab.native_cache_comparison_v1 import compare
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone


def native():
    torch.set_num_threads(1)
    c=ModelConfig(vocab_size=96,hidden_size=32,intermediate_size=64,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=64,vision_hidden_size=8,vision_intermediate_size=16,vision_layers=1,vision_heads=2,
        image_size=16,patch_size=4,pool_size=2)
    return DenseNativeView(install(OwnDenseAssistant(new_backbone(c,974117),reward_hidden=8,word_temporal=4,
        pointing_rank=8,router_hidden=8,detector_hidden=4))).eval()


def test_actual_dense_text_image_chunk_and_single_decode_full_vocabulary_alignment():
    m=native();ids=torch.tensor([[1]+list(range(5,36))])
    paths=[dict(prefix=1,chunks=[7,8,16],rotary_tables=False),dict(prefix=1,chunks=[7,8,16],rotary_tables=True),
        dict(prefix=28,chunks=[1,1,1,1],rotary_tables=True)]
    for pixels in [None,torch.linspace(-1,1,3*16*16).reshape(1,3,16,16)]:
        result=compare(m,ids,pixels=pixels,paths=paths)
        assert result['all_paths_within_bound'] and result['all_compared_position_greedy_argmaxes_equal']
        assert [r['actual_compared_text_positions'] for r in result['actual_paths']]==[32,32,5]
        assert result['actual_paths'][0]['actual_vocabulary_values_compared']==32*96


def test_actual_bad_decoder_output_is_detected_without_rewriting_parameters_or_dropping_positions():
    m=native();ids=torch.tensor([[1,5,6,7,8,9]])
    original=m.decode
    def bad(*args,**kwargs):
        state=original(*args,**kwargs);logits=state['logits'].clone();logits[...,9]+=10
        return dict(state,logits=logits)
    m.decode=bad
    result=compare(m,ids,paths=[dict(prefix=1,chunks=[1]*5,rotary_tables=True)])
    assert not result['all_paths_within_bound']
    assert result['actual_paths'][0]['actual_compared_text_positions']==6
    assert result['actual_paths'][0]['actual_greedy_argmax_disagreements']>0
