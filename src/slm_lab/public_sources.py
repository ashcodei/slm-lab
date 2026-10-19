"""Bounded, resumable collection of licensed public reference, code and essays.
Downloaded repository contents are read as data and never imported or executed.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path, PurePosixPath
import random
import re
import time
import urllib.parse
import urllib.request
import zipfile
from .data import stable_id

AGENT='SLMLabResearch/0.1 (bounded educational corpus collection)'


def fetch(url, limit=180_000_000, timeout=90, attempts=4):
    for attempt in range(attempts):
        try:
            request=urllib.request.Request(url,headers={'User-Agent':AGENT})
            with urllib.request.urlopen(request,timeout=timeout) as response:
                data=response.read(limit+1)
            if len(data)>limit:raise ValueError('Download exceeds bounded size limit')
            return data
        except Exception as error:
            if attempt==attempts-1 or isinstance(error,ValueError) or getattr(error,'code',None) in {400,401,403,404}:raise
            delay=2**attempt
            if getattr(error,'code',None)==429:
                delay=max(30,int(error.headers.get('Retry-After','30')))
            time.sleep(min(delay,120))


def api(url):return json.loads(fetch(url))


def wiki(**params):
    time.sleep(3.0)
    return api('https://en.wikipedia.org/w/api.php?'+urllib.parse.urlencode(
        {'format':'json','formatversion':2,'action':'query','maxlag':5,**params}))


def chunks(text, limit=24000):
    # Keep all chunks of an original document in the same split via group_id.
    lines=text.splitlines(keepends=True);buffer=''
    for line in lines:
        if len(buffer)+len(line)>limit and buffer:
            yield buffer;buffer=''
        buffer+=line
    if buffer:yield buffer


def emit(path, documents):
    count=characters=0
    with path.open('w') as output:
        for document in documents:
            text=document.pop('text')
            for n,part in enumerate(chunks(text)):
                if len(part.strip())<200:continue
                row={**document,'id':stable_id(part),'chunk':n,'text':part}
                output.write(json.dumps(row,ensure_ascii=False)+'\n')
                count+=1;characters+=len(part)
    return {'documents':count,'characters':characters,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def collect_wikipedia(domain, seeds, count, target):
    import mwparserfromhell
    candidates=set(seeds)
    for offset in range(0,len(seeds),10):
        result=wiki(titles='|'.join(seeds[offset:offset+10]),prop='links',plnamespace=0,pllimit=500,redirects=1)
        if result.get('error'):raise ValueError(str(result['error']))
        for page in result.get('query',{}).get('pages',[]):
            candidates.update(link['title'] for link in page.get('links',[]))
    extras=sorted(candidates-set(seeds));random.Random(42).shuffle(extras)
    titles=seeds+extras[:max(0,count-len(seeds))];documents=[]
    for offset in range(0,len(titles),20):
        # Revisions supports full wikitext for multiple pages, unlike TextExtracts.
        result=wiki(titles='|'.join(titles[offset:offset+20]),prop='revisions',
                    rvprop='ids|timestamp|content',rvslots='main',redirects=1)
        if result.get('error'):raise ValueError(str(result['error']))
        for page in result.get('query',{}).get('pages',[]):
            revisions=page.get('revisions',[])
            if not revisions:continue
            revision=revisions[0]
            markup=revision.get('slots',{}).get('main',{}).get('content','')
            if '{{disambiguation' in markup.lower() or '{{disambig' in markup.lower():continue
            text=mwparserfromhell.parse(markup).strip_code(normalize=True,collapse=True)
            # Strip bibliography sections; external works are not included as a corpus.
            text=re.split(r'(?m)^==+\s*(?:References|External links|Further reading|Bibliography)\s*==+',text)[0]
            if len(text)<500:continue
            title=page['title'];url='https://en.wikipedia.org/wiki/'+urllib.parse.quote(title.replace(' ','_'))
            documents.append({'text':title+'\n\n'+text,'domain':domain,'kind':'reference',
                'source':'Wikipedia','source_url':url,
                'revision':{k:v for k,v in revision.items() if k!='slots'},'license':'CC-BY-SA-4.0',
                'attribution_history_url':url+'?action=history','group_id':'wikipedia:'+str(page['pageid']),
                'extraction':'full wikitext stripped of markup, templates and bibliography'})
        print(f'wikipedia {domain}: {min(offset+20,len(titles))}/{len(titles)} pages requested',flush=True)
    if not documents:raise ValueError('No reference articles collected')
    return {**emit(target,documents),'requested_articles':len(titles),'received_articles':len(documents),
            'license_url':'https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use',
            'domain_assignment':'Seed articles plus a bounded sample of outgoing links; not a verified subject taxonomy.'}


EXTENSIONS={'.py','.c','.h','.cpp','.hpp','.cc','.go','.rs','.java','.js','.ts','.tsx','.jsx',
            '.cs','.swift','.rb','.php','.sh','.sql','.md','.rst','.txt','.f','.f90','.jl','.lua','.kt','.scala'}
SKIP={'vendor','vendored','third_party','third-party','thirdparty','external','extern','node_modules',
      'dist','build','generated','fixtures','testdata','submodules','.git','translations'}


def collect_repository(item,target,licenses):
    repo=item['repo'];info=api('https://api.github.com/repos/'+repo)
    commit=api('https://api.github.com/repos/'+repo+'/commits/'+info['default_branch'])['sha']
    try:
        license_info=api('https://api.github.com/repos/'+repo+'/license?ref='+commit)
    except Exception:
        if repo!='JetBrains/kotlin':raise
        path='license/LICENSE.txt'
        raw_license=fetch(f'https://raw.githubusercontent.com/{repo}/{commit}/{path}',100_000)
        license_info={'content':base64.b64encode(raw_license).decode(),'path':path,
                      'license':{'spdx_id':'NOASSERTION'}}
    license_text=base64.b64decode(license_info['content']).decode()
    spdx=license_info.get('license',{}).get('spdx_id')
    expected=item['expected_license']
    if spdx!=expected:
        # Some dual-licensed books are not identified automatically by GitHub.
        manual_apache=repo in {'rust-lang/book','JetBrains/kotlin'} and 'Apache License' in license_text[:200]
        manual_mit=expected=='MIT' and 'MIT License' in license_text[:100] and 'Permission is hereby granted, free of charge' in license_text
        if not (manual_apache or manual_mit):
            raise ValueError(f'License mismatch: expected {expected}, API reported {spdx}')
    (licenses/(repo.replace('/','--')+'.txt')).write_text(license_text)
    raw=fetch('https://codeload.github.com/'+repo+'/zip/'+commit)
    documents=[];skipped=0;total=0
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for member in sorted(archive.infolist(),key=lambda m:m.filename):
            path=PurePosixPath(member.filename);relative=PurePosixPath(*path.parts[1:])
            if member.is_dir() or path.suffix.lower() not in EXTENSIONS or member.file_size>200_000:
                continue
            if set(p.lower() for p in relative.parts)&SKIP:continue
            if any(x in str(relative).lower() for x in ['min.js','lockfile','package-lock','license','copying']):continue
            # Never extract archive paths or execute downloaded programs.
            try:text=archive.read(member).decode('utf-8')
            except UnicodeDecodeError:continue
            if '\x00' in text or len(text.strip())<200:continue
            if re.search(r'generated (?:file|code)|do not edit',text[:2000],re.I):
                skipped+=1;continue
            markers=re.findall(r'SPDX-License-Identifier:\s*([^\n]+)',text[:4000])
            if markers and any(expected not in marker for marker in markers):
                skipped+=1;continue
            total+=len(text)
            if total>100_000_000:break
            documents.append({'text':f'File: {relative}\nLanguage: {path.suffix}\n\n'+text,
                'domain':item['domain'],'kind':'code' if path.suffix not in {'.md','.rst','.txt'} else 'documentation',
                'language_extension':path.suffix,'source':'GitHub/'+repo,
                'source_url':f'https://github.com/{repo}/blob/{commit}/{relative}',
                'revision':commit,'license':expected,'group_id':f'github:{repo}:{relative}',
                'license_url':f'https://github.com/{repo}/blob/{commit}/{license_info["path"]}'})
    return {**emit(target,documents),'repo':repo,'revision':commit,'license':expected,
            'archive_sha256':hashlib.sha256(raw).hexdigest(),'excluded_files':skipped,
            'github_license_id':spdx,'manual_license_text_check':spdx!=expected}


class ArticleText(HTMLParser):
    def __init__(self):super().__init__();self.depth=0;self.skip=0;self.parts=[]
    def handle_starttag(self,tag,attrs):
        if tag in {'d-article','article'}:self.depth+=1
        if tag in {'script','style','svg'}:self.skip+=1
        if self.depth and not self.skip and tag in {'p','div','h1','h2','h3','li','pre','br'}:self.parts.append('\n')
    def handle_endtag(self,tag):
        if tag in {'script','style','svg'}:self.skip=max(0,self.skip-1)
        if tag in {'d-article','article'}:self.depth=max(0,self.depth-1)
    def handle_data(self,data):
        if self.depth and not self.skip:self.parts.append(data)


def collect_blog(item,target):
    raw=fetch(item['url'],5_000_000);html=raw.decode()
    if 'https://creativecommons.org/licenses/by/4.0/' not in html:
        raise ValueError('Expected explicit CC BY 4.0 license link absent')
    parser=ArticleText();parser.feed(html)
    text=''.join(parser.parts)
    return {**emit(target,[{'text':text,**item,'source':'Distill','source_url':item['url'],
        'kind':'blog','group_id':item['url'],'download_sha256':hashlib.sha256(raw).hexdigest()}]),
        'url':item['url'],'download_sha256':hashlib.sha256(raw).hexdigest()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/public-sources.json');p.add_argument('--output',required=True)
    args=p.parse_args();config=json.loads(Path(args.config).read_text());root=Path(args.output)
    root.mkdir(parents=True,exist_ok=True);(root/'licenses').mkdir(exist_ok=True)
    tasks=[('wiki-'+d,lambda out,d=d,s=s:collect_wikipedia(d,s,config['wikipedia_articles_per_domain'],out))
           for d,s in config['domains'].items()]
    tasks += [('repo-'+r['repo'].replace('/','--'),lambda out,r=r:collect_repository(r,out,root/'licenses'))
              for r in config['repositories']]
    tasks += [('blog-'+str(i),lambda out,b=b:collect_blog(b,out)) for i,b in enumerate(config['blogs'])]
    successes=[];failures=[]
    for name,collect in tasks:
        target=root/(name+'.jsonl');report=root/(name+'.json')
        try:
            if report.exists() and target.exists():result=json.loads(report.read_text())
            else:
                result=collect(target);result['collected_at']=datetime.now(timezone.utc).isoformat()
                report.write_text(json.dumps(result,indent=2)+'\n')
            successes.append({'name':name,**result});print(name,result,flush=True)
        except Exception as error:
            failures.append({'name':name,'error':str(error)});print(name,'FAILED',str(error),flush=True)
    seen=set();coverage={};languages={};count=0
    with (root/'all.jsonl').open('w') as output:
        for source in successes:
            with (root/(source['name']+'.jsonl')).open() as stream:
                for line in stream:
                    row=json.loads(line)
                    if row['id'] in seen:continue
                    seen.add(row['id']);count+=1;output.write(line)
                    coverage[row['domain']]=coverage.get(row['domain'],0)+len(row['text'])
                    if row.get('language_extension'):
                        ext=row['language_extension'];languages[ext]=languages.get(ext,0)+len(row['text'])
    manifest={'collected_at':datetime.now(timezone.utc).isoformat(),'sources':successes,'failures':failures,
        'unique_documents':count,'domain_characters':coverage,'language_characters':languages,
        'all_sha256':hashlib.sha256((root/'all.jsonl').read_bytes()).hexdigest(),
        'limitations':['Bounded first collection, not all domains or all open-source code.',
                        'Public reference articles are not clinical or legal authority.',
                        'Repository license checks are conservative filters, not an exhaustive per-file audit.']}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':main()
