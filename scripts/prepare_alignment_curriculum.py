"""Prepare instruction and image curricula without exposing reserved benchmarks."""
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import time
import ahocorasick
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.image_curriculum import combine,commons_snapshot

ROOT=Path(__file__).resolve().parents[1]

def read(path):
    with path.open() as source:
        for line in source:
            if line.strip():yield json.loads(line)


def write(path,rows):path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-images',action='store_true')
    args=parser.parse_args()
    root=ROOT/'data/alignment-curriculum-v1';root.mkdir(exist_ok=True)
    image_sources=[('everyday_photos',ROOT/'data/everyday-images-v1'),
                   ('scene_text',ROOT/'data/scene-text-images-v1'),
                   ('documents',ROOT/'data/document-images-v1')]
    while missing:=[str(path) for _,path in image_sources if not (path/'manifest.json').exists()]:
        atomic_json(ROOT/'reports/alignment-curriculum-preparation.json',
                    {'status':'waiting for diverse image sources','missing':missing,
                     'updated_at':datetime.now(timezone.utc).isoformat()})
        if not args.wait_images:raise ValueError('Diverse image sources are incomplete: '+', '.join(missing))
        time.sleep(30)
    questions=list(read(ROOT/'data/knowledge-eval-v1/questions.jsonl'));matcher=ahocorasick.Automaton()
    for row in questions:matcher.add_word(' '.join(row['question'].casefold().split()),row['id'])
    matcher.make_automaton();excluded=[];inputs=[];splits={}
    for split in ['train','valid']:
        rows=[]
        for source in [ROOT/'data/alignment-public-v1'/('sft.'+split+'.jsonl'),ROOT/'data/code-functions-v2'/(split+'.jsonl')]:
            inputs.append({'path':str(source),'sha256':file_hash(source)})
            for row in read(source):
                text=' '.join((row['prompt']+' '+row['response']).casefold().split())
                matches=[key for _,key in matcher.iter(text)]
                if matches:
                    excluded.append({'id':row['id'],'benchmark_question_ids':matches,'split':split});continue
                rows.append(row)
        write(root/('instruction.'+split+'.jsonl'),rows);splits['instruction.'+split]=len(rows)
    snapshot=commons_snapshot(ROOT/'data/commons-images-v1',root/'commons-source-snapshot')
    image_sources.extend([('commons',root/'commons-source-snapshot'),('fashion',ROOT/'data/fashion-images-v1')])
    image_result=combine(image_sources,root,matcher=matcher,fashion_limit=500)
    for split,count in image_result['splits'].items():splits['images.'+split]=count['examples']
    result={'created_at':datetime.now(timezone.utc).isoformat(),'splits':splits,'source_files':inputs,
        'image_curriculum':image_result,'commons_snapshot':snapshot,
        'status':'prepared; not yet trained',
        'benchmark_questions_sha256':file_hash(ROOT/'data/knowledge-eval-v1/questions.jsonl'),
        'excluded_benchmark_matches':excluded,
        'files_sha256':{name:file_hash(root/name) for name in
            ['instruction.train.jsonl','instruction.valid.jsonl','images.train.jsonl','images.valid.jsonl',
             'contrastive.train.jsonl','contrastive.valid.jsonl']},
        'purpose':'First instruction/vision pilots; not a replacement for broader training or independent final assessment.',
        'limitations':['Exact benchmark question matching only; near-duplicate/semantic leakage is not ruled out.']}
    atomic_json(root/'manifest.json',result)
    atomic_json(ROOT/'reports/alignment-curriculum-preparation.json',
                {'status':'prepared; not yet trained','splits':splits,'updated_at':datetime.now(timezone.utc).isoformat(),
                 'manifest_sha256':file_hash(root/'manifest.json')})
    print(json.dumps({k:v for k,v in result.items() if k not in {'source_files','image_curriculum'}},indent=2))

if __name__=='__main__':main()
