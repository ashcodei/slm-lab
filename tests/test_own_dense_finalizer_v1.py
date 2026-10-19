import torch

from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_dense_finalizer_v1 import finalize
from test_own_dense_standalone_v1 import fixture


def test_actual_learned_actor_fold_and_independent_native_text_photo_word_export(tmp_path):
    source=tmp_path/'source';source.mkdir();base,ids,_,_=fixture(source)
    actor=OwnDenseActor(base,rank=2,seed=975217).eval()
    parameters=[p for _,p in actor.named_selected(('actor',))]
    optimizer=torch.optim.AdamW(parameters,lr=.001,foreach=False)
    actor(ids,labels=ids)['loss'].backward(inputs=parameters);optimizer.step();optimizer.zero_grad(set_to_none=True)
    pixels=torch.linspace(-1,1,3*16*16).reshape(1,3,16,16);word=torch.linspace(-1,1,3*8*32).reshape(1,3,8,32)
    result=finalize(actor,tmp_path/'standalone',source/'tokenizer.json',sources=dict(own_completed_actor_sha256='c'*64),
        completed_updates=1,probes=[('whole_text',dict(input_ids=ids)),('actual_image',dict(input_ids=ids,pixel_values=pixels))],
        word_pixels=word,minimum_free_bytes=0,maximum_checkpoint_bytes=32*1024**2)
    assert result['all_actual_exported_parameters_and_complete_native_probes_cold_load_bit_exact']
    assert result['source_and_all_private_CPU_RNG_unchanged']
    assert result['actual_fold_receipt']['actual_neural_values_removed']>0
