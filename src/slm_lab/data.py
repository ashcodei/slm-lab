"""Fetch small public-data slices and train a new tokenizer, without model weights."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import urllib.parse
import urllib.request
from PIL import Image


def get_json(url):
    with urllib.request.urlopen(url, timeout=45) as r:
        return json.load(r)


def viewer_rows(dataset, config, split, count):
    if count < 1:
        raise ValueError('Row counts must be positive.')
    result = []
    for offset in range(0,count,100):
        q = urllib.parse.urlencode(dict(dataset=dataset,config=config,split=split,
                                      offset=offset,length=min(100,count-offset)))
        obj = get_json('https://datasets-server.huggingface.co/rows?'+q)
        if not obj.get('rows'):
            break
        result.extend(obj['rows'])
    return result


def stable_id(text):
    normalized = ' '.join(text.casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def read_jsonl(path):
    rows = []
    for i,line in enumerate(Path(path).read_text().splitlines(),1):
        if line.strip():
            try:
                row=json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f'{path}:{i}: invalid JSON') from e
            if not isinstance(row,dict):
                raise ValueError(f'{path}:{i}: expected an object')
            rows.append(row)
    if not rows:
        raise ValueError(f'{path}: empty dataset')
    return rows


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))


def hh_pair(row):
    chosen,rejected = row['chosen'],row['rejected']
    marker='\n\nAssistant:'
    i,j = chosen.rfind(marker),rejected.rfind(marker)
    if i<0 or j<0 or chosen[:i+len(marker)] != rejected[:j+len(marker)]:
        raise ValueError('Human preference pair lacks a common final assistant prompt.')
    prompt=chosen[:i+len(marker)]
    good,bad=chosen[i+len(marker):].strip(),rejected[j+len(marker):].strip()
    if not good or not bad or good==bad:
        raise ValueError('Preference responses must be nonempty and distinct.')
    return {'id':stable_id(prompt),'prompt':prompt,'chosen':good,'rejected':bad,
            'feedback_source':'human','source':'Anthropic/hh-rlhf'}


def fetch_samples(output, text_rows=128, image_rows=24, preference_rows=32):
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output directory; existing data is not overwritten.')
    output.mkdir(parents=True,exist_ok=True)
    (output/'images').mkdir()
    sources=[]
    for dataset,config,split,n in [
        ('HuggingFaceFW/fineweb-edu','sample-10BT','train',text_rows),
        ('HuggingFaceM4/the_cauldron','ai2d','train',image_rows),
        ('Anthropic/hh-rlhf','default','train',preference_rows)]:
        rows=viewer_rows(dataset,config,split,n)
        # The viewer is a cached service. Record its actual response hash; the
        # current repository SHA is context, not a claim that the cache is pinned.
        revision=get_json('https://huggingface.co/api/datasets/'+dataset).get('sha')
        sources.append({'dataset':dataset,'config':config,'split':split,
            'requested_rows':n,'received_rows':len(rows),'repository_sha_at_collection':revision,
            'viewer_payload_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
            'viewer_revision_pinned':False})
        if dataset.endswith('fineweb-edu'):
            clean={stable_id(item['row']['text']):item['row'] for item in rows}
            train,valid=[],[]
            for key,row in clean.items():
                item={'id':key,'text':row['text'],'source':dataset,'source_url':row.get('url')}
                (valid if int(key[:8],16)%10==0 else train).append(item)
            if not train or not valid:
                raise ValueError('Need enough unique text documents for a train/validation split.')
            write_jsonl(output/'pretrain.train.jsonl',train)
            write_jsonl(output/'pretrain.valid.jsonl',valid)
        elif dataset.endswith('the_cauldron'):
            train,valid=[],[]
            for item in rows:
                row=item['row']
                if len(row['images'])!=1:
                    raise ValueError('The current model supports one image per example.')
                with urllib.request.urlopen(row['images'][0]['src'],timeout=45) as r:
                    raw=r.read()
                image=Image.open(io.BytesIO(raw)).convert('RGB')
                # Hash decoded pixels so different encodings of the same image
                # cannot put its questions in both train and validation.
                key=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
                image_path=output/'images'/f'{key}.png'
                image.save(image_path)
                for pair in row['texts']:
                    example={'id':stable_id(key+pair['user']),'group_id':key,
                        'prompt':pair['user'],'response':pair['assistant'],
                        'image':str(image_path.relative_to(output)),'source':dataset+'/ai2d'}
                    (valid if int(key[:8],16)%5==0 else train).append(example)
            if not train or not valid:
                raise ValueError('Need enough images for separate image-group splits.')
            write_jsonl(output/'sft.train.jsonl',train)
            write_jsonl(output/'sft.valid.jsonl',valid)
        else:
            pairs=[]
            seen=set()
            for item in rows:
                try:pair=hh_pair(item['row'])
                except ValueError:continue
                if pair['id'] not in seen:
                    pairs.append(pair);seen.add(pair['id'])
            train=[p for p in pairs if int(p['id'][:8],16)%5]
            valid=[p for p in pairs if not int(p['id'][:8],16)%5]
            if not train or not valid:
                raise ValueError('Need enough unique human preference prompts for separate splits.')
            write_jsonl(output/'preferences.train.jsonl',train)
            write_jsonl(output/'preferences.valid.jsonl',valid)
    files={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(output.rglob('*')) if p.is_file()}
    manifest={'collected_at':datetime.now(timezone.utc).isoformat(),'sources':sources,
        'files_sha256':files,'purpose':'Small pipeline validation sample, not a general-capability corpus.',
        'licenses':{'FineWeb-Edu':'ODC-By 1.0; underlying web content retains original rights.',
                    'AI2D':'Consult AI2D source terms linked by The Cauldron dataset card.',
                    'HH-RLHF':'MIT dataset license; used for preferences, not SFT.'}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def train_tokenizer(data, output, vocab_size):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    if vocab_size<260:
        raise ValueError('Byte-level BPE requires at least 256 bytes plus four special tokens.')
    def texts():
        count=0
        with open(data) as source:
            for line in source:
                if line.strip():
                    row=json.loads(line)
                    if not isinstance(row.get('text'),str):raise ValueError('Tokenizer data requires text strings.')
                    count+=1;yield row['text']
        if not count:raise ValueError('Tokenizer training corpus is empty.')
    tokenizer=Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder=decoders.ByteLevel()
    trainer=trainers.BpeTrainer(vocab_size=vocab_size,min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=['<pad>','<bos>','<eos>','<unk>'])
    tokenizer.train_from_iterator(texts(),trainer=trainer)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    tokenizer.save(str(output))
    return tokenizer.get_vocab_size()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('fetch')
    f.add_argument('--output',required=True)
    f.add_argument('--text-rows',type=int,default=128)
    f.add_argument('--image-rows',type=int,default=24)
    f.add_argument('--preference-rows',type=int,default=32)
    t=sub.add_parser('tokenizer')
    t.add_argument('--data',required=True);t.add_argument('--output',required=True)
    t.add_argument('--vocab-size',type=int,default=32768)
    a=p.parse_args()
    if a.command=='fetch':
        manifest=fetch_samples(a.output,a.text_rows,a.image_rows,a.preference_rows)
        print(json.dumps({'sources':manifest['sources'],'files':len(manifest['files_sha256'])},indent=2))
    else:
        print(json.dumps({'actual_vocab_size':train_tokenizer(a.data,a.output,a.vocab_size)}))


if __name__=='__main__':main()
