"""Read retained full-size word updates, both Adam states and complete sources.

This structural/source reader does not replay neural forwards or infer held
reading quality from the disposable original probe's next-update comparison.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from tokenizers import Tokenizer

from slm_lab.byte_word_reader import load_rows
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.code_residual_checkpoint_v1 import _validate_adam, _validate_rng
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.own_aligned_assistant_v3 import FORMAT

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports/shared-words-full-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('Structural word reader must explicitly hide all GPUs')
    output = REPORTS/'independent-complete-readback.json'
    if output.exists(): raise ValueError('Preserve the completed independent word readback')
    torch.set_num_threads(2)
    plan = json.loads((REPORTS/'plan.json').read_text()); proof = json.loads((REPORTS/'complete.json').read_text())
    guard = FileBindings(plan['source_files_sha256']); guard.check(full=True)
    paths = [REPORTS/n for n in ('resume-step1.pt','resume-step2.pt','trained-disposable-extensions.pt')]
    hashes = [proof[k] for k in ('retained_step1_sha256','retained_step2_sha256','trained_disposable_export_sha256')]
    if any(file_hash(p)!=digest for p,digest in zip(paths,hashes)): raise ValueError('Both actual Adam states and trained export bytes must match')
    step1,step2,export = [torch.load(p,map_location='cpu',weights_only=True,mmap=True) for p in paths]
    prefixes = ('vision_linears.','vision_norms.','vision_projection.','word.')
    names = [n for n in export['extensions'] if n.startswith(prefixes)]
    fixed = [n for n in export['extensions'] if not n.startswith(prefixes)]
    expected_sources = dict(full_CPU_plan_sha256=file_hash(REPORTS/'plan.json'),
                            word_admission_sha256=file_hash(ROOT/'data/shared-byte-words-v1/admission.json'))
    if (len(names)!=211 or len(fixed)!=389 or sum(export['extensions'][n].numel() for n in names)!=2714178
            or export['format']!=FORMAT or export['actual_counts']!=proof['actual_counts']
            or export['actual_counts']['total_unique']!=999860418 or export['sources']!=expected_sources
            or len(plan['original_word_positions'])!=len(plan['source_identifiers'])
            or len(plan['original_word_positions'])!=len(plan['full_transcripts'])):
        raise ValueError('Every actual full-size shared vision/word and unused tensor must be counted')
    for step,state in [(1,step1),(2,step2)]:
        model = state['model']; parts = model['extensions']
        if (state['steps']!=step or state['planned_steps']!=2 or state['groups']!=['vision','word']
                or state['optimized_names']!=names or len(state['metrics'])!=step or state['metrics']!=proof['source_metrics'][:step]
                or model['format']!=FORMAT or model['settings']!=export['settings'] or model['sources']!=export['sources']
                or model['actual_counts']!=proof['actual_counts'] or set(parts)!=set(export['extensions'])
                or state['fixed_original_state_sha256']!=proof['fixed_original_digest']
                or state['sampler_state']!=dict(original_word_positions=plan['original_word_positions'],complete_targets=True,completed_updates=step)
                or any(t.shape!=export['extensions'][n].shape or t.dtype!=export['extensions'][n].dtype
                       or not bool(torch.isfinite(t).all()) for n,t in parts.items())
                or any(not torch.equal(parts[n],step1['model']['extensions'][n]) for n in fixed)):
            raise ValueError('Every update/source/options/unused tensor must reconcile with actual retained states')
        _validate_adam(state['optimizer'],[parts[n] for n in names],step); _validate_rng(state['rng'],'cpu')
    if (any(not torch.equal(t,export['extensions'][n]) for n,t in step2['model']['extensions'].items())
            or not any(not torch.equal(step1['model']['extensions'][n],export['extensions'][n]) for n in names)):
        raise ValueError('Actual second-step trained weights must match the export and differ from the first step')
    original_rows,_ = load_rows(ROOT/'data/byte-word-reader-v1')
    packed_rows = [json.loads(line) for line in (ROOT/'data/shared-byte-words-v1/train.jsonl').open()]
    packed = np.memmap(ROOT/'data/shared-byte-words-v1/train.tokens.u16',dtype='<u2',mode='r')
    tokenizer = Tokenizer.from_file(str(ROOT/'checkpoints/whole-human-SFT-v1/initial/tokenizer.json'))
    counts = dict(words=0,UTF8_byte_targets=0,complete_context_tokens=0,response_EOS_targets=0)
    for position,source_id,transcript in zip(plan['original_word_positions'],plan['source_identifiers'],plan['full_transcripts']):
        source = original_rows['train'][position]; row = packed_rows[position]
        digest = hashlib.sha256(json.dumps(source,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if (row['id']!=source_id or row['response']!=source['response'] or row['response']!=transcript
                or any(row[k]!=v for k,v in source.items()) or row['original_runtime_row_sha256']!=digest
                or file_hash(row['image'])!=row['stored_image_sha256']):
            raise ValueError('Every actual probe word must remain its complete original train-only source')
        with Image.open(row['image']) as image:
            rgb = image.convert('RGB')
            word = np.asarray(rgb.resize((448,64),Image.Resampling.BICUBIC),dtype=np.float32)/127.5-1
            native = np.asarray(ImageOps.pad(rgb,(448,448),method=Image.Resampling.BICUBIC,color=(128,128,128)),dtype=np.float32)/127.5-1
        for pixels,key in [(word,'word_input_pixels_sha256'),(native,'native_input_pixels_sha256')]:
            if hashlib.sha256(np.ascontiguousarray(pixels.transpose(2,0,1)).tobytes()).hexdigest()!=row[key]:
                raise ValueError('Every actual probe image must reproduce its full fixed-geometry source pixels')
        prompt = row['prompt'] if row['prompt'].rstrip().endswith('Assistant:') else 'User: '+row['prompt']+'\nAssistant:'
        prefix = [1,*tokenizer.encode(prompt).ids]; ids = [*prefix,*tokenizer.encode(transcript).ids,2]
        item = row['native']; raw = list(transcript.encode('utf-8'))
        required = len(raw)+sum(a==b for a,b in zip(raw,raw[1:]))
        if (ids!=packed[item['offset']:item['offset']+item['length']].astype(np.int64).tolist()
                or item['target_start']!=len(prefix) or row['native_byte_targets']!=raw
                or row['minimum_CTC_steps']!=required or required>140):
            raise ValueError('Every full original byte and BPE/BOS/prompt/response/EOS target must reconstruct')
        counts['words']+=1;counts['UTF8_byte_targets']+=len(raw)
        counts['complete_context_tokens']+=len(ids);counts['response_EOS_targets']+=len(ids)-len(prefix)
    for metric in proof['source_metrics']:
        if (any(metric[k]!=v for k,v in counts.items())
                or metric['summed_objective']!=metric['word_mean_byte_normalized_CTC']+metric['token_mean_native_CE']):
            raise ValueError('Every actual source count and joint loss normalizer must reconstruct')
    if (proof['original_source_positions']!=plan['original_word_positions']
            or proof['original_source_ids']!=plan['source_identifiers']
            or proof['complete_source_transcripts']!=plan['full_transcripts']):
        raise ValueError('Actual source identities must reconcile across plan and completion evidence')
    guard.check(full=True)
    atomic_json(output,dict(reader_sha256=file_hash(__file__),original_complete_sha256=file_hash(REPORTS/'complete.json'),
        actual_counts=proof['actual_counts'],retained_updates=[1,2],actual_Adam_states_per_update=211,
        actual_optimized_values=2714178,all389unused_new_tensors_exact_across_both_updates=True,
        all_final_export_weights_exact_with_retained_step2=True,
        complete_original_probe_sources_pixels_UTF8_and_native_BPE_targets_reconstructed=counts,
        all_saved_Adam_and_CPU_RNG_payloads_validated=True,actual_next_update_replayed_by_original_probe=True,
        reader_did_not_replay_neural_forward_or_optimizer_update=True,
        capacity_only_weights_not_production=True,full_word_training_and_held_reading_quality_unfinished=True,
        candidate_promoted=False,goal_achieved=False))


if __name__=='__main__': main()
