"""Actual full-size learned-word attachment to an already updated dense model.

Both input training artifacts are explicitly disposable CPU controls. This
verifies integration/cold mechanics, not full word training or held quality.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/dense-word-folding-full-CPU-v1'
RESERVE=26*1024**3


def signature(model):return [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in model.parameters()]


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Full offline integration must explicitly hide GPUs')
    if REPORTS.exists():raise ValueError('Preserve each complete or partial integration attempt')
    import torch
    from torch import nn
    from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint
    from slm_lab.corpus import file_hash
    from slm_lab.dense_foundation_inference_v1 import fresh
    from slm_lab.dense_word_folding_v1 import fold_words,FORMAT
    from slm_lab.diverse_images import atomic_json
    from slm_lab.model import TinyMultimodalLM
    from slm_lab.own_aligned_assistant_v3 import OwnAlignedAssistantV3
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.point_grounding_data_v1 import PointerData
    from slm_lab.shared_word_data_v1 import SharedWordData
    from slm_lab.whole_human_sft_inference_v1 import fresh as old_factory
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    tests=ROOT/'reports/dense-word-folding-and-evaluation-v1-tests.log'
    if '6 passed' not in tests.read_text():raise ValueError('Actual folding/guards/all-held dense and actor controls must pass')
    dense_root=ROOT/'reports/dense-foundation-full-CPU-v1';word_root=ROOT/'reports/shared-words-full-CPU-v1'
    dense_plan=json.loads((dense_root/'plan.json').read_text());dense_proof=json.loads((dense_root/'complete.json').read_text())
    word_plan=json.loads((word_root/'plan.json').read_text());word_proof=json.loads((word_root/'complete.json').read_text())
    dense_path=dense_root/'resume-step2.pt';word_path=word_root/'trained-disposable-extensions.pt'
    if file_hash(dense_path)!=dense_proof['retained_step2_sha256'] or file_hash(word_path)!=word_proof['trained_disposable_export_sha256']:
        raise ValueError('Actual complete disposable dense and learned shared-word artifacts required')
    files=dict(dense_plan['source_files_sha256'])
    for path,sha in word_plan['source_files_sha256'].items():
        if path in files and files[path]!=sha:raise ValueError('Never reseal a changed bound original source')
        files[path]=sha
    names=['scripts/probe_dense_word_folding_full_CPU_v1.py','src/slm_lab/dense_word_folding_v1.py',
        'src/slm_lab/dense_word_evaluation_v1.py','src/slm_lab/own_dense_actor_v1.py',
        'tests/test_dense_word_folding_v1.py','tests/test_dense_word_evaluation_v1.py',
        'reports/dense-word-folding-and-evaluation-v1-tests.log']
    for name in names:files[str(ROOT/name)]=file_hash(ROOT/name)
    for folder in (dense_root,word_root):
        for name in ('plan.json','complete.json','independent-complete-readback.json'):
            path=folder/name;files[str(path)]=file_hash(path)
    for path in (dense_path,word_path):files[str(path)]=file_hash(path)
    guard=FileBindings(files);guard.check(full=True);started=time.monotonic()
    settings=dense_plan['initial_dense_settings'];target,_=fresh(settings,'cpu')
    trained_dense=torch.load(dense_path,map_location='cpu',weights_only=True,mmap=True)
    target.import_export(trained_dense['model'],expected_sources=trained_dense['model']['sources']);target.eval();del trained_dense
    old=json.loads((ROOT/'reports/whole-human-SFT-v1/plan.json').read_text())
    prior,_=old_factory(old,'cpu');original=prior.original;del prior
    source=OwnAlignedAssistantV3(original,**old['model_initialization']).eval()
    word_payload=torch.load(word_path,map_location='cpu',weights_only=True,mmap=True)
    expected_word_sources=dict(full_CPU_plan_sha256=file_hash(word_root/'plan.json'),
        word_admission_sha256=file_hash(ROOT/'data/shared-byte-words-v1/admission.json'))
    source.import_export(word_payload,expected_sources=expected_word_sources)
    data=SharedWordData(ROOT/'data/shared-byte-words-v1','train','optimization')
    positions=word_plan['original_word_positions'];batches=[data.batch([p],'cpu') for p in positions]
    pointer=PointerData(ROOT/'data/localized-pointer-candidate-v1','train','audit')
    pp=json.loads((ROOT/'reports/frozen-base-pointer-full-CPU-v1/complete.json').read_text())['original_source_position']
    photo=pointer.batch([pp],'cpu');ids,pixels=photo['input_ids'],photo['pixel_values']
    REPORTS.mkdir()
    atomic_json(REPORTS/'plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        initial_dense_settings=settings,dense_language_checkpoint=str(dense_path),dense_language_sha256=file_hash(dense_path),
        source_word_checkpoint=str(word_path),source_word_sha256=file_hash(word_path),expected_word_sources=expected_word_sources,
        original_word_positions=positions,original_word_ids=[data.rows[p]['id'] for p in positions],
        original_word_transcripts=[data.rows[p]['response'] for p in positions],original_photo_train_position=pp,
        actual_counts=target.counts(),CPU_FP32_threads=2,parity_atol=.001,parity_rtol=.0001,
        greedy_probe_budget=8,both_input_training_artifacts_are_disposable=True,new_optimizer_updates=0,goal_achieved=False))
    sources=dict(full_integration_CPU_plan_sha256=file_hash(REPORTS/'plan.json'),
        dense_language_source_sha256=file_hash(dense_path),word_source_sha256=file_hash(word_path))
    class FactorVisionReference(TinyMultimodalLM):
        """Offline comparator only: the same updated decoder, unfused vision."""
        def __init__(self):
            nn.Module.__init__(self);network=target.network;self.config=target.config
            self.embedding,self.lm_head,self.blocks=network.embedding,network.lm_head,network.blocks
            self.norm,self.value_head=network.norm,network.value_head;self.vision=source.aligned.vision
            self.gradient_checkpointing=False
    reference=FactorVisionReference().eval()
    selected=lambda n:n.startswith(('network.vision.','heads.word.'))
    fixed_before=tensor_digest((n,p) for n,p in target.named_parameters() if not selected(n))
    source_before=tensor_digest(source.named_parameters());source_signature=signature(source)
    fixed_signature=[(id(p),p._version,p.data_ptr(),p.requires_grad) for n,p in target.named_parameters() if not selected(n)]
    before_actor=OwnDenseActor(target).eval();pre_merge_actor=before_actor.export(sources);del before_actor
    with torch.no_grad():
        text_before=[target(b['input_ids'])['logits'].clone() for b in batches]
        old_image=target(ids,pixels)['logits'].clone();expected_image=reference(ids,pixels)['logits'].clone()
        expected_words=[source.word_logits(b['word_pixels']).clone() for b in batches]
        expected_native=[reference(b['input_ids'],b['pixel_values'],attention_mask=b['attention_mask'])['logits'].clone() for b in batches]
        old_code_cache=source.original.language.code.prefill(ids,pixels)['cache']
        old_cached=source.original.language.code.decode(torch.tensor([[4]]),old_code_cache)['logits'].clone()
    receipt=fold_words(target,source,word_payload,expected_sources=expected_word_sources)
    parity={}
    def compare(name,a,b):
        error=float((a-b).abs().max())
        if not torch.allclose(a,b,atol=.001,rtol=.0001):raise ValueError('Actual FP32 factor/folded vision parity exceeded predeclared tolerance: '+name)
        parity[name]=dict(max_absolute_logit_error=error,exact_sampled_outputs=torch.equal(a,b),shape=list(a.shape))
    with torch.no_grad():
        trained_image=target(ids,pixels)['logits'].clone();compare('whole_photo_native',expected_image,trained_image)
        if torch.equal(old_image,trained_image):raise ValueError('The actual learned source vision prefix must be attached to the updated dense language model')
        trained_words=[];trained_native=[];raw_paths=[]
        for p,b,expected_word,expected_text,text in zip(positions,batches,expected_words,expected_native,text_before):
            actual_word=target.word_logits(b['word_pixels']).clone()
            actual_text=target(b['input_ids'],b['pixel_values'],attention_mask=b['attention_mask'])['logits'].clone()
            compare('word_CTC_'+str(p),expected_word,actual_word);compare('word_native_'+str(p),expected_text,actual_text)
            if not torch.equal(text,target(b['input_ids'])['logits']):raise ValueError('Each complete original text-only native output must remain bitwise exact')
            raw_paths.append(dict(original_source_position=p,source_CTC_path=expected_word.argmax(-1).tolist(),
                folded_CTC_path=actual_word.argmax(-1).tolist(),raw_CTC_paths_equal=torch.equal(expected_word.argmax(-1),actual_word.argmax(-1))))
            trained_words.append(actual_word);trained_native.append(actual_text)
        if not torch.equal(old_cached,source.original.language.code.decode(torch.tensor([[4]]),old_code_cache)['logits']):
            raise ValueError('The source original code cache must stay exact')
        greedy_reference=reference.generate(ids,pixels,max_new_tokens=8,eos_token_id=2)
        greedy_folded=target.generate(ids,pixels,max_new_tokens=8,eos_token_id=2)
    if (source_before!=tensor_digest(source.named_parameters()) or source_signature!=signature(source)
            or fixed_before!=tensor_digest((n,p) for n,p in target.named_parameters() if not selected(n))
            or fixed_signature!=[(id(p),p._version,p.data_ptr(),p.requires_grad) for n,p in target.named_parameters() if not selected(n)]):
        raise ValueError('Every complete source and target language/unused head value/metadata must remain exact')
    actor=OwnDenseActor(target).eval();actor_signature=signature(actor)
    try:actor.import_export(pre_merge_actor,expected_sources=sources)
    except ValueError:pass
    else:raise ValueError('A pre-merge policy artifact must reject the changed frozen vision/word representation')
    if signature(actor)!=actor_signature:raise ValueError('Invalid old policy attachment mutated the live combined model')
    if actor.counts()['actual_unique']!=999706442:raise ValueError('Actual combined dense word/vision/actor/critic graph must remain strictly under1B')
    with torch.no_grad():
        if not torch.equal(trained_image,actor(ids,pixels)['logits']):raise ValueError('Fresh bounded actor must preserve the newly merged native image path')
    artifact=dict(format=FORMAT,model=target.export(sources),folding=receipt,
        both_input_training_artifacts_are_disposable=True,candidate_promoted=False,goal_achieved=False)
    atomic_checkpoint(REPORTS/'combined-disposable-model.pt',artifact,5*1024**3,RESERVE);del artifact
    cold,_=fresh(settings,'cpu');payload=torch.load(REPORTS/'combined-disposable-model.pt',map_location='cpu',weights_only=True,mmap=True)
    cold.import_export(payload['model'],expected_sources=sources)
    with torch.no_grad():
        if (not torch.equal(trained_image,cold(ids,pixels)['logits'])
                or any(not torch.equal(a,cold.word_logits(b['word_pixels'])) for a,b in zip(trained_words,batches))
                or any(not torch.equal(a,cold(b['input_ids'],b['pixel_values'],attention_mask=b['attention_mask'])['logits']) for a,b in zip(trained_native,batches))):
            raise ValueError('Actual combined complete native image/word/CTC trained cold reload must be exact')
    guard.check(full=True)
    atomic_json(REPORTS/'complete.json',dict(actual_dense_counts=target.counts(),actual_policy_counts=actor.counts(),
        folding_receipt=receipt,actual_FP32_neural_parity=parity,raw_CTC_paths=raw_paths,
        original_word_positions=positions,source_transcripts=[data.rows[p]['response'] for p in positions],
        dense_language_values_and_metadata_and_all_text_only_outputs_exact=True,
        every_source_weight_metadata_and_original_cached_code_output_exact=True,
        actual_native_image_prefix_changed=True,old_policy_representation_rejected_before_mutation=True,
        fresh_actual999706442_policy_initial_image_logits_exact=True,
        raw_whole_photo_probe_prompt_ids=ids.tolist(),raw_greedy_factor_vision_tokens=greedy_reference.tolist(),
        raw_greedy_folded_vision_tokens=greedy_folded.tolist(),actual_sampled_greedy_probe_tokens_equal=torch.equal(greedy_reference,greedy_folded),
        actual_combined_native_image_word_CTC_cold_reload_exact=True,
        source_word_export_sha256=file_hash(word_path),source_dense_language_sha256=file_hash(dense_path),
        combined_checkpoint_sha256=file_hash(REPORTS/'combined-disposable-model.pt'),
        seconds=time.monotonic()-started,FP32_atol=.001,FP32_rtol=.0001,greedy_probe_generation_budget=8,
        new_optimizer_updates=0,all_input_probe_training_weights_disposable=True,
        full_production_training_GPU_merge_parity_or_held_reading_quality_not_claimed=True,
        candidate_promoted=False,goal_achieved=False))
    print(REPORTS/'complete.json',flush=True)


if __name__=='__main__':main()
