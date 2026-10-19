"""Collect a bounded, resumable Commons image-caption curriculum with licenses."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import time
import urllib.parse
from PIL import Image,ImageOps
from .data import stable_id
from .public_sources import fetch,api

QUERIES={
 'biology':'biology diagram','chemistry':'chemistry apparatus','physics':'physics experiment',
 'mathematics':'mathematics graph','medicine':'anatomy illustration','astronomy':'astronomy telescope',
 'computers':'computer hardware','networking':'network router','engineering':'mechanical engineering machine',
 'animals':'animal mammal photograph','plants':'plant flower photograph','landscapes':'landscape mountain photograph',
 'food':'food fruit vegetable','products':'hand tool product','transport':'vehicle car bicycle'}


class PlainText(HTMLParser):
    def __init__(self):super().__init__();self.parts=[]
    def handle_data(self,data):self.parts.append(data)
    def handle_starttag(self,tag,attrs):
        if tag in {'p','li','br','div'}:self.parts.append(' ')
    def handle_endtag(self,tag):
        if tag in {'p','li','div'}:self.parts.append(' ')


def plain(html):
    p=PlainText();p.feed(html);return ' '.join(''.join(p.parts).split())


def license_allowed(metadata):
    short=metadata.get('LicenseShortName',{}).get('value','')
    url=metadata.get('LicenseUrl',{}).get('value','')
    if short in {'Public domain','CC0','CC0 1.0'}:return True
    parsed=urllib.parse.urlparse(url)
    return parsed.hostname=='creativecommons.org' and (
        parsed.path.startswith('/licenses/by/') or parsed.path.startswith('/licenses/by-sa/')
        or parsed.path.startswith('/publicdomain/zero/'))


def caption(metadata):
    raw=metadata.get('ImageDescription',{}).get('value','');text=plain(raw)
    if len(text)<25 or len(text)>900 or len(text.split())<5:return None
    if any(marker in text for marker in ['View Book Page','Identifier:','Text Appearing Before Image','Contributing Library']):return None
    return re.sub(r'^English:\s*','',text).strip()


def prepare_image(page,domain):
    title=page['title'];info=page.get('imageinfo',[{}])[0];metadata=info.get('extmetadata',{})
    if not license_allowed(metadata):return {'rejected':'license'}
    description=caption(metadata)
    if description is None:return {'rejected':'caption'}
    url=info.get('thumburl') or info.get('url','');parsed=urllib.parse.urlparse(url)
    if parsed.scheme!='https' or parsed.hostname not in {'upload.wikimedia.org','thumb.wikimedia.org'}:
        return {'rejected':'image'}
    try:
        raw=fetch(url,8_000_000,timeout=20,attempts=2)
        with Image.open(io.BytesIO(raw)) as image:
            if min(image.size)<96 or image.width*image.height>40_000_000:return {'rejected':'image'}
            image=ImageOps.exif_transpose(image).convert('RGBA')
            background=Image.new('RGBA',image.size,(255,255,255,255));background.alpha_composite(image)
            image=ImageOps.contain(background.convert('RGB'),(512,512),Image.Resampling.LANCZOS)
            pixels=ImageOps.pad(image,(224,224),method=Image.Resampling.BICUBIC,color=(128,128,128))
            group=hashlib.sha256(pixels.tobytes()).hexdigest();png=io.BytesIO();image.save(png,format='PNG')
        row={'id':stable_id(group+'Describe this image.'),'group_id':group,'domain':domain,
            'prompt':'Describe this image.','response':description,'image':'images/'+group+'.png',
            'source':'Wikimedia Commons','source_title':title,'source_url':info.get('descriptionurl'),
            'download_url':url,'source_sha1':info.get('sha1'),'source_timestamp':info.get('timestamp'),
            'license':metadata.get('LicenseShortName',{}).get('value'),
            'caption_license':'CC-BY-SA-4.0 for Commons description-page text; quoted third-party material may differ.',
            'license_url':metadata.get('LicenseUrl',{}).get('value'),
            'artist':plain(metadata.get('Artist',{}).get('value','')),
            'credit':plain(metadata.get('Credit',{}).get('value','')),
            'download_sha256':hashlib.sha256(raw).hexdigest(),
            'stored_image_sha256':hashlib.sha256(png.getvalue()).hexdigest(),
            'metadata_sha256':hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest(),
            'caption_origin':'Uploader metadata requested in English; language and grounding unverified.',
            'collected_at':datetime.now(timezone.utc).isoformat()}
        return {'row':row,'png':png.getvalue()}
    except Exception as error:return {'error':str(error),'title':title}


def collect(root,per_domain=100,pages=4):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);(root/'images').mkdir(exist_ok=True)
    # One JSONL journal is appended only after the image is saved successfully.
    journal=root/'collected.jsonl';rows=[]
    if journal.exists():rows=[json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    seen_pixels={r['group_id'] for r in rows};seen_titles={r['source_title'] for r in rows}
    coverage={d:sum(r['domain']==d for r in rows) for d in QUERIES}
    failures=[];rejected={'license':0,'caption':0,'image':0,'duplicate':0}
    with journal.open('a') as output:
        for domain,query in QUERIES.items():
            if coverage[domain]>=per_domain:continue
            for page_num in range(pages):
                q={'action':'query','format':'json','formatversion':2,'generator':'search',
                   'gsrsearch':query+' filetype:bitmap','gsrnamespace':6,'gsrlimit':50,
                   'gsroffset':page_num*50,'prop':'imageinfo','iiprop':'url|extmetadata|sha1|size|timestamp',
                   'iiurlwidth':512,'iiextmetadatalanguage':'en'}
                time.sleep(3)
                try:result=json.loads(fetch('https://commons.wikimedia.org/w/api.php?'+urllib.parse.urlencode(q),timeout=30,attempts=2))
                except Exception as error:
                    failures.append({'domain':domain,'page':page_num,'error':str(error)});continue
                if result.get('error'):
                    failures.append({'domain':domain,'page':page_num,'error':result['error']});continue
                candidates=[page for page in result.get('query',{}).get('pages',[]) if page['title'] not in seen_titles]
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures=[pool.submit(prepare_image,page,domain) for page in candidates]
                    for future in as_completed(futures):
                        prepared=future.result()
                        if prepared.get('rejected'):
                            rejected[prepared['rejected']]+=1;continue
                        if prepared.get('error'):
                            failures.append({'domain':domain,**prepared});continue
                        row=prepared['row'];group=row['group_id']
                        if group in seen_pixels:rejected['duplicate']+=1;continue
                        if coverage[domain]>=per_domain:continue
                        target=root/row['image'];temporary=target.with_suffix('.tmp')
                        temporary.write_bytes(prepared['png']);temporary.replace(target)
                        output.write(json.dumps(row,ensure_ascii=False)+'\n');output.flush()
                        rows.append(row);seen_titles.add(row['source_title']);seen_pixels.add(group);coverage[domain]+=1
                        if coverage[domain]%10==0:print(domain,coverage[domain],flush=True)
                print('page result',domain,page_num,'accepted',coverage[domain],'failed downloads',len(failures),flush=True)
                if coverage[domain]>=per_domain:break
            print('completed domain',domain,coverage[domain],flush=True)
    splits={'train':[],'valid':[]}
    for row in rows:splits['valid' if int(row['group_id'][:8],16)%10==0 else 'train'].append(row)
    for split,records in splits.items():
        (root/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    manifest={'collected_at':datetime.now(timezone.utc).isoformat(),'coverage':coverage,
        'images':len(rows),'splits':{s:len(r) for s,r in splits.items()},'failures':failures,'rejected':rejected,
        'files_sha256':{s+'.jsonl':hashlib.sha256((root/(s+'.jsonl')).read_bytes()).hexdigest() for s in splits},
        'split_rule':'SHA256 of decoded model-input pixels modulo 10; a repeated image cannot cross splits.',
        'limitations':['Uploader captions are weak supervision and can contain facts not visible in pixels.',
                        'This is a bounded first visual curriculum, not a broad image-understanding benchmark.',
                        'Near-duplicate crops/resolutions require further auditing.']}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--per-domain',type=int,default=100);p.add_argument('--pages',type=int,default=4)
    a=p.parse_args()
    if a.per_domain<1 or a.pages<1:p.error('Collection limits must be positive.')
    collect(a.output,a.per_domain,a.pages)

if __name__=='__main__':main()
