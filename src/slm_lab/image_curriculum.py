"""Combine image tasks with shared photo groups and conservative held-out protection."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from PIL import Image
from .corpus import file_hash
from .diverse_images import jsonl, pixel_hash


def balanced_fashion(rows, limit):
    classes=defaultdict(list)
    for row in rows:classes[row['class_id']].append(row)
    quota=limit//max(1,len(classes));selected=[]
    for category in sorted(classes):
        selected.extend(sorted(classes[category],key=lambda r:r['id'])[:quota])
    return selected


def combine(sources, output, matcher=None, fashion_limit=500):
    """Every supplied source is immutable, or copied to an explicit journal snapshot."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    candidates={'train':[],'valid':[]};protected=set();inputs=[]
    for name,root in sources:
        root=Path(root)
        protection=root/'protected-image-ids.txt'
        if protection.exists():
            protected.update(protection.read_text().splitlines())
            inputs.append({'path':str(protection),'sha256':file_hash(protection)})
        for split in ['train','valid']:
            path=root/(split+'.jsonl')
            rows=list(jsonl(path));inputs.append({'path':str(path),'sha256':file_hash(path)})
            if name=='fashion':rows=balanced_fashion(rows,fashion_limit if split=='train' else 200)
            for row in rows:
                image=(root/row['image']).resolve()
                # Recompute from actual images, rather than trusting a source-supplied duplicate key.
                with Image.open(image) as pixels:
                    digest=pixel_hash(pixels);high_resolution_digest=pixel_hash(pixels,448)
                candidates[split].append({**row,'image':str(image),'model_pixel_sha256':digest,
                    'model_pixel_sha256_448':high_resolution_digest,
                    'curriculum_source':name,'task':row.get('task','category' if name=='fashion' else 'caption')})
    protected.update(row['group_id'] for row in candidates['valid'])
    heldout_pixels={r['model_pixel_sha256'] for r in candidates['valid']}
    heldout_pixels_448={r['model_pixel_sha256_448'] for r in candidates['valid']}
    # If any crop of a training photo matches a held-out crop, exclude the entire photo group.
    overlapping_groups={r['group_id'] for r in candidates['train'] if r['model_pixel_sha256'] in heldout_pixels
                        or r['model_pixel_sha256_448'] in heldout_pixels_448}
    excluded=[];counts={};accepted={}
    for split in ['valid','train']:
        rows=[];seen=set()
        for row in candidates[split]:
            reason=None
            if split=='train' and row['group_id'] in protected:reason='protected original image'
            elif split=='train' and row['group_id'] in overlapping_groups:reason='exact held-out pixel duplicate in photo group'
            text=' '.join((row['prompt']+' '+row['response']).casefold().split())
            matches=[key for _,key in matcher.iter(text)] if matcher is not None else []
            if matches:reason='reserved benchmark question'
            identity=(row['model_pixel_sha256'],row['prompt'],row['response'])
            if identity in seen:reason='duplicate image/task/answer'
            if reason:
                excluded.append({'id':row['id'],'group_id':row['group_id'],'source':row['curriculum_source'],
                                 'split':split,'reason':reason,'benchmark_question_ids':matches})
                continue
            seen.add(identity);rows.append(row)
        accepted[split]=rows
        path=output/('images.'+split+'.jsonl')
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        captions=[r for r in rows if r['task']=='caption']
        (output/('contrastive.'+split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in captions))
        counts[split]={'examples':len(rows),'original_images':len({r['group_id'] for r in rows}),
                      'by_source':dict(Counter(r['curriculum_source'] for r in rows)),
                      'by_task':dict(Counter(r['task'] for r in rows)),
                      'contrastive_captions':len(captions)}
    train_groups={r['group_id'] for r in accepted['train']};valid_groups={r['group_id'] for r in accepted['valid']}
    if train_groups & valid_groups:raise ValueError('Image group leakage after curriculum assembly')
    train_pixels={r['model_pixel_sha256'] for r in accepted['train']};valid_pixels={r['model_pixel_sha256'] for r in accepted['valid']}
    if train_pixels & valid_pixels:raise ValueError('Image pixel leakage after curriculum assembly')
    if {r['model_pixel_sha256_448'] for r in accepted['train']} & {r['model_pixel_sha256_448'] for r in accepted['valid']}:
        raise ValueError('448px image pixel leakage after curriculum assembly')
    return {'splits':counts,'source_files':inputs,'excluded':excluded,
            'protected_image_groups':len(protected),'excluded_pixel_duplicate_groups':len(overlapping_groups),
            'train_valid_image_group_overlap':0,'train_valid_exact_model_pixel_overlap':0,
            'train_valid_exact_model_pixel_overlap_by_resolution':{'224':0,'448':0},
            'fashion_training_example_cap':fashion_limit,
            'limitations':['Exact input-pixel and canonical photo IDs checked; perceptual duplicates remain possible.',
                           'Inputs are audited at 224px and 448px; dense small text can still exceed resolution. OCR crops are included.',
                           'Photo captions and word transcriptions are original public annotations, not preference feedback.']}


def commons_snapshot(source, output):
    """Use only complete journal lines whose image files have already been committed."""
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    journal=source/'collected.jsonl'
    raw=journal.read_bytes() if journal.exists() else b''
    raw=raw[:raw.rfind(b'\n')+1]
    (output/'source-snapshot.jsonl').write_bytes(raw)
    splits={'train':[],'valid':[]}
    for line in raw.splitlines():
        row=json.loads(line);row['image']=str((source/row['image']).resolve())
        split='valid' if int(row['group_id'][:8],16)%10==0 else 'train'
        splits[split].append(row)
    for split,rows in splits.items():
        (output/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    return {'source':str(source),'snapshot_sha256':hashlib.sha256(raw).hexdigest(),
            'images':sum(map(len,splits.values())),'collection_finished':(source/'manifest.json').exists()}
