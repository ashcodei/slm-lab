"""Reconstruct code/docstring regions for every complete admitted train target.

The optional code4/docstring1 weighting retains every original response/EOS
byte. It is a candidate objective, not a demonstrated model improvement or a
change to any running training schedule.
"""
from datetime import datetime,timezone
import gzip,json,math,os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CompleteCodePack,FILES
from slm_lab.code_target_regions_v1 import regions
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources

ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'data/direct-algorithm-function-contracts-admitted-v1'
OUT=ROOT/'reports/full-function-code-target-regions-CPU-v1'

def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only whole-source target analysis required')
    tests=ROOT/'reports/code-target-regions-v1-final-tests.log'
    if '3 passed' not in tests.read_text():raise ValueError('Actual Unicode/CRLF/whole-target/weighted-gradient controls required')
    pack=CompleteCodePack(SOURCE,'train','training');all_rows=list(map(json.loads,gzip.open(SOURCE/'original-rows.jsonl.gz','rt')))
    if len(pack)!=826:raise ValueError('All826complete original train functions required')
    tokenizer=Tokenizer.from_file(str(SOURCE/'tokenizer.json'))
    files={str(p):file_hash(p) for p in [*[SOURCE/name for name in [*FILES,'manifest.json','admission.json']],tests,
        *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_code_target_regions_v1.py'])]};guard=FileBindings(files)
    OUT.mkdir();atomic_json(OUT/'plan.json',dict(owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),
        source_files_sha256=files,source_functions=826,code_weight=4.,docstring_comment_whitespace_and_EOS_base_weight=1.,
        all_positive_target_bytes_and_true_EOS_retained=True,existing_training_and_benchmarks_unchanged=True,neural_updates=0,goal_achieved=False))
    total=0;code=docs=weighted=0.;entries=[];weights=[];offset=0
    with (OUT/'every-original-function-region-summary.jsonl').open('x') as journal:
        for position,index in enumerate(pack.positions):
            row=all_rows[int(index)];ids,labels=pack.sequence(position);target=[v for v in labels if v!=-100]
            actual=regions(row['response'],tokenizer,code_weight=4.)
            if target!=actual['whole_original_response_EOS_ids'] or actual['response_EOS_targets']!=len(target):raise ValueError('Every actual original target/EOS ID must reconstruct before weighting')
            record={k:v for k,v in actual.items() if k not in {'whole_original_response_EOS_ids','positive_target_weights'}}
            record.update(source_function_id=row['id'],original_train_position=position,original_global_index=int(index),target_weight_offset=offset)
            journal.write(json.dumps(record)+'\n');total+=len(target);code+=actual['executable_equivalent_token_mass'];docs+=actual['docstring_equivalent_token_mass'];weighted+=actual['weighted_target_denominator']
            entries.append((position,int(index),offset,len(target)));offset+=len(target);weights.extend(actual['positive_target_weights'])
    if total!=pack.manifest['counts']['train']['target_tokens'] or not all(1<=w<=4 for w in weights):raise ValueError('Every original positive target must retain a bounded nonzero weight')
    np.savez_compressed(OUT/'complete-original-response-weight-candidate.npz',
        index=np.asarray(entries,dtype=[('train_position','<u4'),('original_global_index','<u4'),('offset','<u8'),('targets','<u4')]),weights=np.asarray(weights,dtype='<f4'))
    guard.check(full=True)
    atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,original_train_functions=826,
        original_response_EOS_targets=total,executable_equivalent_token_mass=code,docstring_equivalent_token_mass=docs,
        other_and_EOS_equivalent_token_mass=total-code-docs,executable_target_fraction=code/total,
        docstring_target_fraction=docs/total,candidate_weighted_denominator=weighted,
        candidate_executable_weight_fraction=4*code/weighted,all_original_target_and_EOS_IDs_exact=True,
        every_original_target_weight_at_least1=True,true_EOS_weight=1.,
        fraction_uses_actual_token_offset_overlap_not_a_count_of_instructions_or_runtime_importance=True,
        weighted_objective_still_requires_matched_full_size_training_and_generated_evaluation=True,
        source_files_sha256={**files,**{str(OUT/name):file_hash(OUT/name) for name in ['plan.json','every-original-function-region-summary.jsonl','complete-original-response-weight-candidate.npz']}},
        new_neural_forwards=0,new_neural_updates=0,goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT/'complete.json'),executable_target_fraction=code/total,docstring_target_fraction=docs/total,candidate_executable_weight_fraction=4*code/weighted)),flush=True)

if __name__=='__main__':main()
