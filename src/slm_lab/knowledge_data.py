"""Reserve a revision-pinned, balanced MMLU-Pro sample for knowledge evaluation."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import random
from .corpus import file_hash


def main():
    from datasets import load_dataset
    from huggingface_hub import HfApi
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--per-domain',type=int,default=40);a=p.parse_args();root=Path(a.output)
    if root.exists() and any(root.iterdir()) and not a.resume:p.error('Use a new output directory or --resume.')
    root.mkdir(parents=True,exist_ok=True);dataset='TIGER-Lab/MMLU-Pro';info=HfApi().dataset_info(dataset,token=False)
    revision=info.sha;source=load_dataset(dataset,split='test',revision=revision,token=False)
    groups=defaultdict(list)
    for row in source:
        groups[row['category']].append({'id':str(row['question_id']),'domain':row['category'],
            'question':row['question'],'options':row['options'],'answer_index':row['answer_index'],
            'source':dataset,'revision':revision,'original_split':'test'})
    selected=[]
    for category in sorted(groups):
        random.Random('slm-lab-knowledge-'+category).shuffle(groups[category])
        selected.extend(groups[category][:a.per_domain])
    output=root/'questions.raw.jsonl';output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    manifest={'collected_at':datetime.now(timezone.utc).isoformat(),'dataset':dataset,'revision':revision,
        'license_metadata':info.card_data.to_dict().get('license') if info.card_data else None,
        'test_examples':len(selected),'per_domain':{k:min(a.per_domain,len(v)) for k,v in groups.items()},
        'questions_sha256':file_hash(output),'scoring':'Zero-shot generated answer and candidate-letter likelihood; report both.',
        'scope':'Balanced held-out sample; not the complete standardized MMLU-Pro protocol.',
        'held_out_from_training':True,'contamination_audit':'Pending exact normalized text audit of the actual packed training corpus.',
        'uncovered_requested_topics':['Astrology and astronomy are not separate MMLU-Pro categories; supplemental assessments pending.']}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':main()
