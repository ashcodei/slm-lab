"""Reconcile all pairing metadata against original source bytes and token spans."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from slm_lab.code_prompt_pair_corpus_v1 import SourcePairs
from slm_lab.code_prompt_pilot_v1 import validate_selection
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'data/code-prompt-pairs-v1'
    sources={role:SourcePairs(folder,role) for role in ['train','validation']}
    checked={role:0 for role in sources}
    with gzip.open(Path(sources['train'].manifest['code_pack'])/'original-rows.jsonl.gz','rt') as stream:
        for line in stream:
            original=json.loads(line)
            role=original['split']
            if role not in sources:continue
            source=sources[role];i=checked[role];row=source.records[i]
            ids,labels=source.pack.sequence(i)
            raw=np.asarray(ids,dtype='<u2')
            start=row['prefix_tokens']
            if (row['id']!=original['id'] or row['group_id']!=original['group_id']
                    or row['language']!=original['language'] or row['target_tokens']!=original['target_tokens']
                    or row['global_position']!=original['token_pack_index']
                    or hashlib.sha256(raw.tobytes()).hexdigest()!=original['complete_input_ids_sha256']
                    or row['full_input_sha256']!=original['complete_input_ids_sha256']
                    or row['body_sha256']!=original['original_source_body_sha256']
                    or hashlib.sha256(raw[:start].tobytes()).hexdigest()!=row['prompt_sha256']
                    or labels!=[-100]*start+ids[start:] or ids[0]!=1 or ids[-1]!=2):
                raise ValueError('Full original programme/source/prompt/mask mismatch')
            j=int(source.pairs[i])
            if j>=0:
                other=source.records[j]
                if not (other['role']==role and other['language']==row['language']
                        and other['group_id']!=row['group_id'] and other['body_sha256']!=row['body_sha256']
                        and other['prompt_sha256']!=row['prompt_sha256']
                        and other['prefix_tokens']+row['target_tokens']<=4096):
                    raise ValueError('Original whole-prompt source pairing mismatch')
            checked[role]+=1
    if any(checked[role]!=len(sources[role]) for role in sources):
        raise ValueError('All original role programmes must be independently reconciled')
    if ({r['group_id'] for r in sources['train'].records}
            & {r['group_id'] for r in sources['validation'].records}):
        raise ValueError('No train/validation source group overlap permitted')
    validate_selection(json.loads((ROOT/'reports/code-residual-lr-pilot-v1/selected-original-rows.json').read_text()),sources)
    for source in sources.values():source.guard.check(full=True)
    result=dict(passed=True,original_full_programmes_read_back=checked,
        manifest_sha256=file_hash(folder/'manifest.json'), all_original_positive_inputs_and_masks_match=True,
        same_role_language_distinct_group_body_prompt_pairs=True, full4096_fit_without_clipping=True,
        original512_train_128_validation_selection_and_extremes_match=True,
        all_original_role_rows_retained=True, neural_updates=0, source_association_not_correctness_or_RLHF=True,
        source_files_sha256={str(Path(__file__).resolve()):file_hash(__file__)},goal_achieved=False)
    atomic_json(ROOT/'reports/code-prompt-pairs-v1-independent-audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
