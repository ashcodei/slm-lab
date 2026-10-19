"""Read every complete native human prompt for later own-response PPO.

Stored original answers are used only to reconstruct the unchanged prompt and
source fingerprint. They are not own actions or fresh human judgments. Exact
train/held prompt reuse is reported without changing any source role.
"""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
DESTINATION=ROOT/'reports/dense-ppo-full-prompt-source-v1.json'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Source-only CPU prompt audit must hide GPUs')
    if DESTINATION.exists():raise ValueError('Preserve prior source readback')
    from slm_lab.corpus import file_hash
    from slm_lab.dense_ppo_v1 import fingerprint,prompt_source
    from slm_lab.diverse_images import atomic_json
    from slm_lab.whole_human_data_v2 import WholeHumanData
    started=time.monotonic();summaries={};train_prompts=set();overlap=[]
    for role,purpose in (('train','optimization'),('valid','evaluation')):
        data=WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences',role,purpose)
        ledger=hashlib.sha256();seen=set();counts={str(n):0 for n in (4,32,64,128,256)};tokens=0;maximum=0
        for position,row in enumerate(data.rows):
            a,al=data.sequence(position,'chosen');b,bl=data.sequence(position,'rejected')
            start=row['native']['chosen']['target_start']
            original_split='train' if role=='train' else 'test'
            if (row['split']!=role or row['original_split']!=original_split or row['feedback_source']!='human'
                    or start!=row['native']['rejected']['target_start'] or a[:start]!=b[:start] or start<1
                    or al[:start]!=[-100]*start or bl[:start]!=[-100]*start
                    or al[start:]!=a[start:] or bl[start:]!=b[start:] or a[-1]!=2 or b[-1]!=2):
                raise ValueError('Every complete original same-role prompt and both native response masks required')
            prompt=a[:start];sha=fingerprint(prompt)
            if role=='train':
                source=prompt_source(data,position)
                if source['prompt_ids']!=prompt or source['prompt_tokens_sha256']!=sha:
                    raise ValueError('The actual PPO train interface must preserve every whole source prompt')
                train_prompts.add(sha)
            elif sha in train_prompts:overlap.append(row['id'])
            seen.add(sha);tokens+=start;maximum=max(maximum,start)
            for budget in counts:counts[budget]+=int(start+int(budget)+1<=data.manifest['config']['max_text_length'])
            ledger.update(json.dumps(dict(position=position,id=row['id'],source_record_sha256=row['source_record_sha256'],
                whole_prompt_ids=prompt,prompt_tokens_sha256=sha),sort_keys=True,separators=(',',':')).encode()+b'\n')
        summaries[role]=dict(original_pairs=len(data),whole_prompt_tokens=tokens,maximum_prompt_tokens=maximum,
            original_publisher_split='train' if role=='train' else 'test',
            unique_exact_prompt_token_sequences=len(seen),exact_prompt_duplicates_within_role=len(data)-len(seen),
            complete_prompt_and_one_scoring_EOS_fit_counts_by_generation_horizon=counts,
            ordered_complete_original_prompt_ledger_sha256=ledger.hexdigest(),
            every_original_prompt_read_without_truncation=True,source_role_unchanged=True)
        print(role,'complete_original_prompts',len(data),'maximum_prompt_tokens',maximum,flush=True)
    files={}
    for name in ('manifest.json','admission.json','preferences.train.jsonl','preferences.train.tokens.u16','preferences.valid.jsonl','preferences.valid.tokens.u16'):
        p=ROOT/'data/whole-human-alignment-v2'/name;files[str(p)]=file_hash(p)
    for name in ('src/slm_lab/dense_ppo_v1.py','scripts/audit_dense_ppo_prompts_v1.py'):
        p=ROOT/name;files[str(p)]=file_hash(p)
    atomic_json(DESTINATION,dict(observed_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        complete_original_prompt_summaries=summaries,held_rows_with_an_exact_train_prompt=len(overlap),
        held_exact_train_prompt_row_ids=overlap,source_role_reuse_and_development_selection_not_certified_independence=True,
        original_human_rankings_on_other_model_answers=True,own_output_human_judgments_not_collected=True,
        source_readback_is_not_neural_exposure=True,production_PPO_updates=0,goal_achieved=False,seconds=time.monotonic()-started))
    print(DESTINATION,flush=True)


if __name__=='__main__':main()
