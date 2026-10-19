"""Prepare a licensed public product-image warmup with original test images held out."""
import argparse
import base64
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
from PIL import Image
from .corpus import file_hash
from .public_sources import api,fetch

CLASSES=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']


def idx(raw):
    payload=gzip.decompress(raw);magic=struct.unpack('>I',payload[:4])[0]
    if magic==2051:
        count,height,width=struct.unpack('>III',payload[4:16])
        if len(payload)!=16+count*height*width:raise ValueError('Invalid image IDX size')
        return np.frombuffer(payload,dtype=np.uint8,offset=16).reshape(count,height,width)
    if magic==2049:
        count=struct.unpack('>I',payload[4:8])[0]
        if len(payload)!=8+count:raise ValueError('Invalid label IDX size')
        return np.frombuffer(payload,dtype=np.uint8,offset=8)
    raise ValueError('Unsupported IDX format')


def collect(output,max_train,max_valid,revision=None):
    root=Path(output)
    if root.exists() and any(root.rglob('*.jsonl')):raise ValueError('Use a new dataset output directory')
    root.mkdir(parents=True,exist_ok=True);(root/'images').mkdir(exist_ok=True)
    repo='zalandoresearch/fashion-mnist'
    revision=revision or api(f'https://api.github.com/repos/{repo}/commits/master')['sha']
    license_text=fetch(f'https://raw.githubusercontent.com/{repo}/{revision}/LICENSE',100_000)
    if b'The MIT License (MIT)' not in license_text[:100]:raise ValueError('Expected source MIT license')
    (root/'LICENSE.txt').write_bytes(license_text)
    arrays={};sources=[]
    for split,prefix in [('train','train'),('test','t10k')]:
        for kind,suffix in [('images','images-idx3-ubyte.gz'),('labels','labels-idx1-ubyte.gz')]:
            path=f'data/fashion/{prefix}-{suffix}'
            url=f'https://raw.githubusercontent.com/{repo}/{revision}/{path}'
            raw=fetch(url,50_000_000);arrays[(split,kind)]=idx(raw)
            sources.append({'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
    # Protect the whole official test set, not just the evaluation sample.
    heldout={hashlib.sha256(image.tobytes()).hexdigest() for image in arrays[('test','images')]}
    seen=set();splits={};duplicates=0
    prompt='What product category is shown? Choose one category: '+', '.join(CLASSES)+'.'
    for split in ['test','train']:
        target='valid' if split=='test' else 'train';budget=max_valid if split=='test' else max_train
        counts={c:0 for c in CLASSES};rows=[]
        for index,(image,label) in enumerate(zip(arrays[(split,'images')],arrays[(split,'labels')])):
            category=CLASSES[int(label)];key=hashlib.sha256(image.tobytes()).hexdigest()
            if key in seen or split=='train' and key in heldout:duplicates+=1;continue
            seen.add(key)
            if counts[category]>=budget//len(CLASSES):continue
            path=root/'images'/f'{key}.png';Image.fromarray(image).save(path)
            rows.append({'id':key,'group_id':key,'domain':'product_images','image':str(path.relative_to(root)),
                'prompt':prompt,'response':category,'class_id':int(label),'classes':CLASSES,
                'source':'GitHub/'+repo,'source_url':f'https://github.com/{repo}/tree/{revision}/data/fashion',
                'revision':revision,'original_split':split,'source_index':index,'license':'MIT',
                'annotation_origin':'Original Fashion-MNIST category labels; not human preference feedback.'})
            counts[category]+=1
        (root/(target+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in rows))
        splits[target]={'images':len(rows),'class_counts':counts};print(target,splits[target],flush=True)
    manifest={'collected_at':datetime.now(timezone.utc).isoformat(),'repo':repo,'revision':revision,
        'license':'MIT','sources':sources,'splits':splits,'duplicates_or_test_overlaps_removed':duplicates,
        'split_rule':'Original train/test split; every official test-image pixel hash protected from training.',
        'files_sha256':{s+'.jsonl':file_hash(root/(s+'.jsonl')) for s in ['train','valid']},
        'limitations':['28px grayscale clothing images; useful for a product-vision warmup, not general image understanding.',
                        'Exact pixel duplicates excluded; perceptual near-duplicates remain possible.']}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--revision')
    p.add_argument('--max-train',type=int,default=30000);p.add_argument('--max-valid',type=int,default=2000)
    a=p.parse_args()
    if min(a.max_train,a.max_valid)<10:p.error('Need at least ten examples for balanced class sampling.')
    collect(a.output,a.max_train,a.max_valid,a.revision)

if __name__=='__main__':main()
