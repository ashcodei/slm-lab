"""Fold a completed own actor and prove its self-contained dense export.

Real source probes measure FP32 folding drift. Cold native probes must then be
bit-exact to the actually exported dense student; actor quality is not assumed.
"""
import hashlib
from pathlib import Path
import shutil

import torch

from .codenet_checkpoint_v1 import atomic_checkpoint,capture_rng
from .corpus import file_hash
from .own_actor_dense_fold_v1 import fold
from .own_dense_standalone_v1 import export,load
from .whole_human_sft_evidence_v1 import tensor_digest


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def finalize(actor,folder,tokenizer_path,*,sources,completed_updates,probes,word_pixels,
             maximum_logit_error=.002,minimum_free_bytes=26*1024**3,maximum_checkpoint_bytes=5*1024**3):
    if next(actor.parameters()).device.type!='cpu' or not probes:raise ValueError('Actual own CPU actor and complete native source probes required')
    folder=Path(folder)
    if folder.exists():raise ValueError('Preserve every earlier finalizer attempt')
    signature=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in actor.parameters()];rng=capture_rng('cpu')
    dense,receipt=fold(actor);observations=[]
    for name,batch in probes:
        native={k:batch[k] for k in ('input_ids','attention_mask','pixel_values') if k in batch}
        with torch.no_grad():
            a=actor(**native)['logits'];b=dense(**native)['logits'];maximum=float((a-b).abs().max())
            differences=int((a.argmax(-1)!=b.argmax(-1)).sum())
        if maximum>maximum_logit_error:raise ValueError('Actual whole-source actor-to-dense FP32 error exceeds the declared numerical bound')
        observations.append(dict(input=name,complete_text_positions=batch['input_ids'].shape[1],
            maximum_absolute_logit_error=maximum,actual_greedy_argmax_differences=differences))
        del a,b
    with torch.no_grad():
        if not torch.equal(actor.base.word_logits(word_pixels),dense.word_logits(word_pixels)):
            raise ValueError('Actual complete native word path must remain exact through the language fold')
    folder.mkdir();shutil.copyfile(tokenizer_path,folder/'tokenizer.json');path=folder/'model.pt'
    provenance=dict(kind='fold_complete_own_human_and_execution_actor',actual_fold_receipt=receipt,
        completed_code_updates=completed_updates,generated_quality_and_GPU_BF16_parity_not_assumed=True)
    atomic_checkpoint(path,export(dense,sources=sources,provenance=provenance,completed_updates=completed_updates,
        tokenizer_sha256=file_hash(tokenizer_path)),maximum_checkpoint_bytes,minimum_free_bytes)
    cold,metadata=load(path,expected_checkpoint_sha256=file_hash(path),expected_sources=sources)
    if tensor_digest(cold.named_parameters())!=tensor_digest(dense.named_parameters()):
        raise ValueError('Every exported dense parameter must cold-load exactly')
    for name,batch in probes:
        native={k:batch[k] for k in ('input_ids','attention_mask','pixel_values') if k in batch}
        with torch.no_grad():
            if not torch.equal(dense(**native)['logits'],cold(**native)['logits']):
                raise ValueError('Complete real text/visual logits must cold-load bit-exactly')
    with torch.no_grad():
        if not torch.equal(dense.word_logits(word_pixels),cold.word_logits(word_pixels)):
            raise ValueError('Complete actual real-word logits must cold-load bit-exactly')
    if signature!=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in actor.parameters()] or not equal(rng,capture_rng('cpu')):
        raise ValueError('Finalization must preserve source parameters and every private CPU RNG')
    return dict(actual_fold_receipt=receipt,actual_full_native_probes=observations,
        checkpoint=str(path),checkpoint_sha256=file_hash(path),tokenizer_sha256=file_hash(folder/'tokenizer.json'),
        standalone_sources=sources,actual_counts=metadata['actual_counts'],completed_updates=completed_updates,
        all_actual_exported_parameters_and_complete_native_probes_cold_load_bit_exact=True,
        real_word_logits_exact_through_fold_and_cold_native=True,source_and_all_private_CPU_RNG_unchanged=True,
        old_checkpoint_actor_optimizer_or_reference_not_needed_for_inference=True,
        final_benchmarks_and_model_quality_still_unproved=True,model_updates=0,candidate_promoted=False,goal_achieved=False)
