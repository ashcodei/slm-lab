"""Pack a dated, checksum-verified Wikimedia article dump without retaining all XML.

Every article retains revision/attribution and exact token offsets. Original article
groups use the existing project split rule. Only complete source documents enter a
token-budgeted corpus; finished source parts are independently resumable.
"""
import argparse
import bz2
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import urllib.parse
import xml.etree.ElementTree as ET
import ahocorasick
import mwparserfromhell
from tokenizers import Tokenizer
from .corpus import ShardWriter,file_hash
from .data import stable_id
from .diverse_images import asset,atomic_json,jsonl
from .public_sources import fetch

DOMAIN_WORDS={
    'medical':['medicine','medical','clinical','health','anatomy','pharmacology','pathology'],
    'law':['law','legal','legislation','constitution','court'],
    'mathematics':['mathematics','algebra','geometry','calculus','statistics','number theory'],
    'physics':['physics','quantum','mechanics','thermodynamics'],
    'chemistry':['chemistry','chemical','molecules','compound'],
    'biology':['biology','genetics','ecology','organisms','species','botany'],
    'computers':['computer','computing','hardware','data structure'],
    'software_engineering':['software','programming','algorithm','compiler'],
    'networking':['network','internet','protocol','telecommunication'],
    'psychology':['psychology','psychological','cognitive','behaviour'],
    'astronomy':['astronomy','astronomical','astrophysics','galaxy','planet','star'],
    'astrology':['astrology','astrological','horoscope'],
    'economics':['economics','economic','finance','trade'],
    'engineering':['engineering','electrical','mechanical','manufacturing'],
}


def clean_article(markup,title):
    if not 600<=len(markup)<=300_000:return None
    if re.search(r'\{\{\s*(?:disambiguation|disambig|hndis|geodis|numberdis)',markup,re.I):return None
    categories=re.findall(r'\[\[Category:([^\]|]+)',markup,re.I)
    # Cut reference sections before stripping heading syntax; plain strip_code
    # removes the == markers, making a later markup-heading regex ineffective.
    markup=re.split(r'(?m)^==+\s*(?:References|External links|Further reading|Bibliography|Sources|Notes)\s*==+',markup)[0]
    code=mwparserfromhell.parse(markup)
    for tag in code.filter_tags(recursive=True):
        if str(tag.tag).strip().lower() in {'ref','references','gallery','script','style'}:
            try:code.remove(tag)
            except ValueError:pass
    for template in code.filter_templates(recursive=True):
        if str(template.name).strip().lower() in {'math','mvar','nowrap','mono'} and template.has(1):
            try:code.replace(template,str(template.get(1).value))
            except ValueError:pass
    text=code.strip_code(normalize=True,collapse=True)
    text=re.split(r'(?m)^==+\s*(?:References|External links|Further reading|Bibliography|Sources)\s*==+',text)[0].strip()
    if len(text)<500:return None
    cues=(title+' '+ ' '.join(categories)).casefold()
    scores={domain:sum(word in cues for word in words) for domain,words in DOMAIN_WORDS.items()}
    domain=max(scores,key=scores.get) if max(scores.values()) else 'encyclopedia_other'
    return title+'\n\n'+text,domain,categories


def articles(path):
    with bz2.open(path,'rb') as source:
        context=ET.iterparse(source,events=('start','end'));_,root=next(context)
        namespace=root.tag.partition('}')[0]+'}' if root.tag.startswith('{') else ''
        for event,page in context:
            if event!='end' or page.tag!=namespace+'page':continue
            if page.findtext(namespace+'ns')!='0' or page.find(namespace+'redirect') is not None:
                page.clear();root.clear();continue
            revision=page.find(namespace+'revision')
            if revision is not None:
                yield {'page_id':page.findtext(namespace+'id'),'title':page.findtext(namespace+'title'),
                       'revision':revision.findtext(namespace+'id'),
                       'timestamp':revision.findtext(namespace+'timestamp'),
                       'markup':revision.findtext(namespace+'text')}
            page.clear();root.clear()


def protected_questions(path):
    matcher=ahocorasick.Automaton()
    for row in jsonl(path):matcher.add_word(' '.join(row['question'].casefold().split()),row['id'])
    matcher.make_automaton();return matcher


def official_hash(path):
    digest=hashlib.sha1()
    with path.open('rb') as source:
        while block:=source.read(1<<20):digest.update(block)
    return digest.hexdigest()


def snapshot_files(cache,date):
    url=f'https://dumps.wikimedia.org/enwiki/{date}/dumpstatus.json'
    path=Path(cache)/'wikipedia-dumpstatus.json'
    if path.exists():raw=path.read_bytes()
    else:
        raw=fetch(url,8_000_000,timeout=30);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    status=json.loads(raw);job=status['jobs']['articlesdump']
    if job['status']!='done':raise ValueError('Choose a completed dated Wikimedia snapshot')
    pattern=re.compile(r'enwiki-'+re.escape(date)+r'-pages-articles(\d+)\.xml-p(\d+)p(\d+)\.bz2')
    selected=[]
    for name,metadata in job['files'].items():
        match=pattern.fullmatch(name)
        if match:selected.append((int(match[1]),int(match[2]),name,metadata))
    if not selected:raise ValueError('Snapshot has no article XML parts')
    return sorted(selected),{'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'date':date}


def collect(output,cache,tokenizer_path,questions,date='20261001',max_tokens=2_000_000_000):
    root=Path(output);cache=Path(cache);root.mkdir(parents=True,exist_ok=True);(root/'parts').mkdir(exist_ok=True)
    if (root/'corpus.json').exists():raise ValueError('Completed corpus exists; use a new output for a new budget')
    token_hash=file_hash(tokenizer_path);question_hash=file_hash(questions)
    tokenizer=Tokenizer.from_file(str(tokenizer_path));eos=tokenizer.token_to_id('<eos>')
    if eos is None:raise ValueError('Tokenizer lacks EOS')
    dtype='<u2' if tokenizer.get_vocab_size()<=65536 else '<u4';matcher=protected_questions(questions)
    configuration={'date':date,'max_tokens':max_tokens,'tokenizer_sha256':token_hash,'questions_sha256':question_hash,
                   'extraction_version':'wiki-clean-v2'}
    configuration_path=root/'collection-config.json'
    if configuration_path.exists() and json.loads(configuration_path.read_text())!=configuration:
        raise ValueError('Resume requires the original snapshot, budget, tokenizer and protected questions')
    atomic_json(configuration_path,configuration)
    files,snapshot=snapshot_files(cache,date);db=sqlite3.connect(root/'dedup.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS documents (hash BLOB PRIMARY KEY, part TEXT)');db.commit()
    finished=[];total=0;stop=False
    for _,_,name,metadata in files:
        if total>=max_tokens:break
        directory=root/'parts'/name.removesuffix('.xml.bz2');manifest_path=directory/'corpus.json'
        if manifest_path.exists():
            manifest=json.loads(manifest_path.read_text())
            if manifest['tokenizer_sha256']!=token_hash:raise ValueError('Part tokenizer mismatch')
            for split in manifest['splits'].values():
                for shard in split['shards']:
                    if file_hash(directory/shard['path'])!=shard['sha256']:raise ValueError('Completed part shard changed')
            finished.append((directory,manifest));total+=manifest['total_tokens']
            if manifest.get('stopped_at_budget'):break
            continue
        # Remove only interrupted, generated part outputs. Committed parts remain immutable.
        db.execute('DELETE FROM documents WHERE part=?',(name,));db.commit()
        if directory.exists():shutil.rmtree(directory)
        directory.mkdir();writers={s:ShardWriter(directory,s,dtype,1_000_000) for s in ['train','valid']}
        cached,download=asset(cache,name,'https://dumps.wikimedia.org'+metadata['url'],metadata['size']+1)
        if official_hash(cached)!=metadata['sha1']:raise ValueError('Official Wikipedia source SHA1 mismatch')
        source={**download,'official_sha1':metadata['sha1'],'snapshot':snapshot}
        documents=0;part_tokens=0;excluded=Counter();coverage=defaultdict(lambda:Counter());questions_seen=[]
        db.execute('BEGIN')
        with (directory/'documents.jsonl').open('w') as journal:
            for row in articles(cached):
                if not row['page_id'] or not row['revision'] or not isinstance(row['markup'],str):continue
                result=clean_article(row['markup'],row['title'])
                if result is None:excluded['quality_or_length']+=1;continue
                text,domain,categories=result
                matches=[qid for _,qid in matcher.iter(' '.join(text.casefold().split()))]
                if matches:
                    excluded['reserved_question']+=1;questions_seen.append({'page_id':row['page_id'],'question_ids':matches});continue
                digest=hashlib.sha256(' '.join(text.partition('\n\n')[2].split()).encode()).digest()
                if db.execute('SELECT 1 FROM documents WHERE hash=?',(digest,)).fetchone():excluded['exact_body_duplicate']+=1;continue
                ids=tokenizer.encode(text).ids+[eos]
                if total+part_tokens+len(ids)>max_tokens:stop=True;break
                group='wikipedia:'+row['page_id'];split='valid' if int(stable_id(group)[:8],16)%10==0 else 'train'
                writer=writers[split];offset=writer.tokens+len(writer.buffer);writer.append(ids)
                db.execute('INSERT INTO documents VALUES (?,?)',(digest,name))
                url='https://en.wikipedia.org/wiki/'+urllib.parse.quote(row['title'].replace(' ','_'))
                journal.write(json.dumps({'group_id':group,'page_id':row['page_id'],'title':row['title'],
                    'revision':row['revision'],'timestamp':row['timestamp'],'source':'Wikipedia',
                    'source_url':url+'?oldid='+row['revision'],'attribution_history_url':url+'?action=history',
                    'license':'CC-BY-SA-4.0','license_url':'https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use',
                    'split':split,'token_start':offset,'token_count':len(ids),'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                    'domain_hint':domain,'source_categories':categories[:40]},ensure_ascii=False)+'\n')
                part_tokens+=len(ids);documents+=1;coverage[domain][split]+=len(ids)
                if documents%500==0:
                    journal.flush();atomic_json(root/'progress.json',{'status':'processing verified Wikipedia part',
                        'part':name,'part_documents':documents,'processed_tokens':total+part_tokens,'target_tokens':max_tokens})
                    print('wikipedia',name,'documents',documents,'tokens',total+part_tokens,flush=True)
        for writer in writers.values():writer.flush()
        journal_hash=file_hash(directory/'documents.jsonl')
        manifest={'source':source,'dtype':dtype,'tokenizer_sha256':token_hash,'total_tokens':part_tokens,'documents':documents,
            'extraction_version':configuration['extraction_version'],
            'domain_tokens':{k:dict(v) for k,v in coverage.items()},'excluded':dict(excluded),'reserved_matches':questions_seen,
            'document_provenance_sha256':journal_hash,'stopped_at_budget':stop,
            'splits':{s:{'tokens':w.tokens,'shards':w.files} for s,w in writers.items()},
            'split_rule':'Same Wikipedia page-ID SHA256 modulo 10 as the earlier project corpus.'}
        db.commit();atomic_json(manifest_path,manifest);finished.append((directory,manifest));total+=part_tokens
        cached.unlink();print('completed Wikipedia part',name,part_tokens,flush=True)
        if stop:break
    db.close()
    result={'source':{'dataset':'Wikimedia English article dump','snapshot':snapshot},'dtype':dtype,'tokenizer_sha256':token_hash,
        'total_tokens':total,'documents':sum(m['documents'] for _,m in finished),'requested_max_tokens':max_tokens,
        'splits':{s:{'tokens':0,'shards':[]} for s in ['train','valid']},'domain_tokens':{},'parts':[],
        'protected_questions_sha256':question_hash,'split_rule':'Canonical Wikipedia page ID hashed modulo 10; previous held-out pages remain held out.',
        'limitations':['Category-keyword domain hints are approximate, not a verified subject taxonomy.',
                       'Exact protected question text filtered; semantic and near-duplicate overlap are not ruled out.',
                       'Cleaned article text is licensed public data, not public-domain text or guaranteed factual truth.',
                       'Bounded snapshot prefix; it is not all Wikipedia or all domain knowledge.']}
    domain_totals=defaultdict(lambda:Counter())
    for directory,manifest in finished:
        result['parts'].append({'path':str(directory),'manifest_sha256':file_hash(directory/'corpus.json')})
        for domain,tokens in manifest['domain_tokens'].items():domain_totals[domain].update(tokens)
        for split,values in manifest['splits'].items():
            result['splits'][split]['tokens']+=values['tokens']
            for shard in values['shards']:
                result['splits'][split]['shards'].append({**shard,'path':str((directory/shard['path']).relative_to(root))})
    result['domain_tokens']={k:dict(v) for k,v in domain_totals.items()}
    if not all(v['tokens']>4096 for v in result['splits'].values()):raise ValueError('Insufficient train/validation tokens')
    atomic_json(root/'corpus.json',result);atomic_json(root/'progress.json',{'status':'completed public Wikipedia corpus','tokens':total,'documents':result['documents']})
    print('completed Wikipedia corpus',total,result['documents'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--tokenizer',required=True);p.add_argument('--questions',required=True);p.add_argument('--date',default='20261001')
    p.add_argument('--max-tokens',type=int,default=2_000_000_000);a=p.parse_args()
    if a.max_tokens<40960 or not re.fullmatch(r'\d{8}',a.date):p.error('Need a positive corpus budget and a dated snapshot')
    collect(a.output,a.cache,a.tokenizer,a.questions,a.date,a.max_tokens)

if __name__=='__main__':main()
