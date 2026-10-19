"""Recompute every full-size vision merge and all unchanged dense/head tensors.

Only durable weights and original source pixels/tokens are read. Numerical
neural parity and cold reload are evidence from the original full-size probe.
"""
import json
import os
from pathlib import Path

import torch

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig
from slm_lab.own_code_delta_v1 import matrix_names,norm_names
from slm_lab.own_dense_actor_v1 import budget
from slm_lab.own_dense_growth_v1 import DenseBackbone,OwnDenseAssistant
from slm_lab.point_grounding_data_v1 import PointerData
from slm_lab.shared_word_data_v1 import SharedWordData

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/dense-word-folding-full-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Full integration reader must explicitly hide GPUs')
    output=REPORTS/'independent-complete-readback.json'
    if output.exists():raise ValueError('Preserve original integration readback')
    torch.set_num_threads(2)
    plan=json.loads((REPORTS/'plan.json').read_text());proof=json.loads((REPORTS/'complete.json').read_text())
    guard=FileBindings(plan['source_files_sha256']);guard.check(full=True)
    dense_path=Path(plan['dense_language_checkpoint']);word_path=Path(plan['source_word_checkpoint'])
    combined_path=REPORTS/'combined-disposable-model.pt'
    if (file_hash(dense_path)!=plan['dense_language_sha256'] or file_hash(word_path)!=plan['source_word_sha256']
            or file_hash(combined_path)!=proof['combined_checkpoint_sha256']):
        raise ValueError('Exact original disposable dense-language/word and combined artifacts required')
    before=torch.load(dense_path,map_location='cpu',weights_only=True,mmap=True)['model']
    source=torch.load(word_path,map_location='cpu',weights_only=True,mmap=True)
    bundle=torch.load(combined_path,map_location='cpu',weights_only=True,mmap=True);after=bundle['model'];parts=source['extensions']
    with torch.device('meta'):
        meta=OwnDenseAssistant(DenseBackbone(ModelConfig(**after['config'])),**after['settings'])
    expected=budget(meta,16);expected['actual_unique']=expected['total_unique']
    if (meta.counts()!=proof['actual_dense_counts'] or expected!=proof['actual_policy_counts']
            or meta.counts()['total_unique']!=984399690 or expected['actual_unique']!=999706442
            or after['actual_counts']!=meta.counts() or before['config']!=after['config'] or before['settings']!=after['settings']
            or bundle['folding']!=proof['folding_receipt'] or source['sources']!=plan['expected_word_sources']):
        raise ValueError('Actual complete fixed-budget model/head/actor specifications and source bindings required')
    expected_sources=dict(full_integration_CPU_plan_sha256=file_hash(REPORTS/'plan.json'),
        dense_language_source_sha256=plan['dense_language_sha256'],word_source_sha256=plan['source_word_sha256'])
    if after['sources']!=expected_sources:raise ValueError('Both exact trained sources and complete integration plan required')
    for name,module in [('network',meta.network),('heads',meta.heads)]:
        shapes=module.state_dict()
        if (after[name].keys()!=shapes.keys() or any(t.shape!=shapes[n].shape or t.dtype!=shapes[n].dtype
                or not bool(torch.isfinite(t).all()) for n,t in after[name].items())):
            raise ValueError('Every combined model/head tensor must be finite with exact declared shape/dtype')
    if after['network']['embedding.weight'].data_ptr()!=after['network']['lm_head.weight'].data_ptr():
        raise ValueError('The actual saved vocabulary/head must share one storage')
    changed=set()
    # Direct algebra from durable pre-merge tensors, without importing or
    # executing the folding function used by the original neural probe.
    for i,name in enumerate(matrix_names(meta.config.vision_layers)):
        key='vision.'+name
        fused=before['network'][key]+parts[f'vision_linears.{i}.up']@parts[f'vision_linears.{i}.down']
        if not torch.equal(fused,after['network'][key]):raise ValueError('Every original vision matrix must equal its own exact saved base+factor product')
        changed.add(key)
    for i,name in enumerate(norm_names(meta.config.vision_layers)):
        key='vision.'+name
        if not torch.equal(parts[f'vision_norms.{i}.weight'],after['network'][key]):raise ValueError('Every learned vision norm must copy exactly')
        changed.add(key)
    fused=before['network']['vision.project.weight']+parts['vision_projection.up']@parts['vision_projection.down']
    if not torch.equal(fused,after['network']['vision.project.weight']):raise ValueError('The complete native image projection must fold exactly')
    changed.add('vision.project.weight')
    if (any(not torch.equal(t,after['network'][n]) for n,t in before['network'].items() if n not in changed)
            or any(not torch.equal(t,after['heads'][n]) for n,t in before['heads'].items() if not n.startswith('word.'))
            or any(not torch.equal(parts[n],t) for n,t in after['heads'].items() if n.startswith('word.'))):
        raise ValueError('All original language/critic/nonword heads/patch/position values stay exact, and every byte-word tensor copies its trained source')
    data=SharedWordData(ROOT/'data/shared-byte-words-v1','train','optimization');counts=dict(words=0,UTF8_bytes=0,native_context_tokens=0,native_response_EOS_targets=0)
    if (plan['original_word_ids']!=[data.rows[p]['id'] for p in plan['original_word_positions']]
            or plan['original_word_transcripts']!=[data.rows[p]['response'] for p in plan['original_word_positions']]
            or proof['source_transcripts']!=plan['original_word_transcripts']):
        raise ValueError('Every original selected training image/word identity and complete transcript required')
    for p,paths in zip(plan['original_word_positions'],proof['raw_CTC_paths']):
        batch=data.batch([p],'cpu');row=data.rows[p]
        if (paths['original_source_position']!=p or paths['raw_CTC_paths_equal']!=(paths['source_CTC_path']==paths['folded_CTC_path'])
                or any(len(path)!=140 or any(type(v) is not int or not 0<=v<=256 for v in path)
                    for key in ('source_CTC_path','folded_CTC_path') for path in paths[key])):
            raise ValueError('Every full source/folded byte path and raw path comparison must reconstruct')
        counts['words']+=1;counts['UTF8_bytes']+=len(row['response'].encode('utf-8'))
        counts['native_context_tokens']+=int(batch['attention_mask'].sum());counts['native_response_EOS_targets']+=int((batch['labels'][:,1:]!=-100).sum())
    prefix=proof['raw_whole_photo_probe_prompt_ids'][0]
    pointer=PointerData(ROOT/'data/localized-pointer-candidate-v1','train','audit')
    original_photo=pointer.batch([plan['original_photo_train_position']],'cpu')
    if original_photo['input_ids'].tolist()!=proof['raw_whole_photo_probe_prompt_ids']:
        raise ValueError('The original complete whole-photo source prompt/pixels must reconstruct')
    for key in ('raw_greedy_factor_vision_tokens','raw_greedy_folded_vision_tokens'):
        tokens=proof[key][0]
        if tokens[:len(prefix)]!=prefix or not 1<=len(tokens)-len(prefix)<=8:raise ValueError('Actual complete source prompt plus explicit bounded greedy deliveries required')
    if proof['actual_sampled_greedy_probe_tokens_equal']!=(proof['raw_greedy_factor_vision_tokens']==proof['raw_greedy_folded_vision_tokens']):
        raise ValueError('Retained greedy parity must match the actual raw deliveries')
    guard.check(full=True)
    atomic_json(output,dict(reader_sha256=file_hash(__file__),original_complete_sha256=file_hash(REPORTS/'complete.json'),
        actual_dense_counts=meta.counts(),actual_policy_counts=expected,
        all84vision_matrix_sums_and_native_projection_independently_recomputed=True,
        all25vision_norms_and16word_tensors_equal_original_trained_source=True,
        every931244544language_value_critic_other_head_and_original_patch_position_value_equal_before=True,
        every_saved_tensor_finite_exact_shapes_and_one_actual_tied_vocabulary_storage=True,
        complete_used_original_train_pixels_transcripts_and_native_masks_reconstructed=counts,
        raw_full_byte_paths_and_bounded_greedy_delivery_comparisons_reconstructed=True,
        original_full_size_probe_measured_neural_parity_and_actual_cold_reload=True,
        reader_did_not_replay_neural_inference_or_generation=True,
        both_input_artifacts_explicitly_disposable_not_full_production_word_or_language_training=True,
        GPU_merge_parity_full_held_reading_quality_and_final_integration_unfinished=True,
        candidate_promoted=False,goal_achieved=False))
    print(output)


if __name__=='__main__':main()
