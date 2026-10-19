"""Verify every current-curriculum question match using actual native batches.

This bypasses the audit projection and substring automaton. Current-curriculum
membership is potential exposure; this does not say which questions have
already been consumed by the live partial checkpoint.
"""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import unicodedata

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/current-held-question-matches-v1-readback'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only actual-batch question match readback required')
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.expanded_joint_sources_v1 import ExpandedJointSources
    from slm_lab.model import ModelConfig
    torch.set_num_threads(1)
    source=ROOT/'reports/full-held-question-audit-v2/current_native.json';audit=json.loads(source.read_text())
    registry_path=ROOT/'data/full-held-question-pool-v2/question-registry.jsonl'
    questions={r['audit_id']:r for r in map(json.loads,registry_path.open())}
    plan_path=ROOT/'reports/phase-joint-v3/plan.json';plan=json.loads(plan_path.read_text())
    initial=torch.load(plan['initial_dense_settings']['initializer'],map_location='cpu',weights_only=True,mmap=True)
    config=ModelConfig(**initial['config']);del initial
    data=ExpandedJointSources(ROOT,config);tokenizer_path=ROOT/'data/tokenizer-corpus/tokenizer.json'
    if file_hash(tokenizer_path)!=plan['tokenizer_sha256']:raise ValueError('Exact own tokenizer required')
    tokenizer=Tokenizer.from_file(str(tokenizer_path));bindings=[];contexts={};domains=Counter()
    def normalized(value):return ' '.join(unicodedata.normalize('NFKC',value).casefold().split())
    for identity,match in audit['matches'].items():
        question=questions[identity];native=match['source'];family=native['family'];position=native['position'];key=(family,position)
        if native['scope']!='current_native':raise ValueError('Only original current-curriculum native matches required')
        if key not in contexts:
            batch=data.batch(family,[position],'cpu');length=int(batch['attention_mask'][0].sum())
            ids=batch['input_ids'][0,:length].tolist();labels=batch['labels'][0,:length].tolist()
            if length!=data.lengths[family][position] or sum(v!=-100 for v in labels[1:])!=data.targets[family][position]:
                raise ValueError('Actual complete native input and target budget must match')
            text=tokenizer.decode(ids,skip_special_tokens=False)
            for marker in ['<bos>','<eos>','<pad>']:text=text.replace(marker,'\0')
            contexts[key]=(normalized(text),hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest(),length)
        text,digest,length=contexts[key];needle=normalized(question['question']);start=text.find(needle)
        if start<0 or start!=match['normalized_character_start']:raise ValueError('Claimed exact question substring does not occur at the actual native batch position')
        domains[question['benchmark']]+=1
        bindings.append(dict(audit_id=identity,benchmark=question['benchmark'],id=question['id'],family=family,position=position,
            actual_native_input_ids_sha256=digest,actual_native_context_positions=length,normalized_character_start=start,
            previously_sampled_for_local_evaluation=question['previously_sampled_for_local_evaluation'],
            exact_substring_verified_from_actual_native_batch=True))
    if len(bindings)!=audit['matched_original_question_ids'] or dict(domains)!=audit['question_counts_by_benchmark']:
        raise ValueError('Every recorded matched question identity required')
    pulse_path=Path(plan['pulse_sources']['knowledge']);pulse={str(r['id']) for r in map(json.loads,pulse_path.open())}
    pulse_matches=[r['id'] for r in bindings if r['benchmark']=='MMLU-Pro' and r['id'] in pulse]
    OUT.mkdir();atomic_json(OUT/'actual-native-match-bindings.json',bindings)
    files=[Path(__file__),source,registry_path,plan_path,pulse_path,OUT/'actual-native-match-bindings.json',
        ROOT/'src/slm_lab/expanded_joint_sources_v1.py',ROOT/'src/slm_lab/fast_joint_data_v1.py']
    result=dict(at=datetime.now(timezone.utc).isoformat(),complete=True,all_recorded_current_native_matches_verified=True,
        original_matched_questions=len(bindings),distinct_actual_native_contexts=len(contexts),by_benchmark=dict(domains),
        current_56_question_development_pulse_matches=pulse_matches,
        potential_curriculum_exposure_not_exact_already_consumed_checkpoint_ancestry=True,
        current_or_old_semantic_independence_and_image_text_not_certified=True,
        original_audit_projection_and_substring_automaton_not_used=True,
        source_files_sha256={str(p):file_hash(p) for p in files},
        full_six_scope_source_audit_still_required=True,neural_generations=0,model_updates=0,goal_achieved=False)
    atomic_json(OUT/'complete.json',result);print(json.dumps(dict(verified_current_native_question_matches=len(bindings),
        by_benchmark=dict(domains),actual_native_contexts=len(contexts),current_56_question_pulse_matches=pulse_matches)),flush=True)


if __name__=='__main__':main()
