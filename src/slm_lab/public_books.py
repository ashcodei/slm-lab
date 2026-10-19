"""Collect bounded US-public-domain books from an official Gutenberg mirror."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET
from .public_sources import fetch,emit

BOOKS=[1342,11,84,345,2701,1661,2600,98,1400,174,35,36,1232,1228,3300,4363]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    successes=[];failures=[]
    for book in BOOKS:
        path=root/f'book-{book}.jsonl';report=root/f'book-{book}.json'
        try:
            if path.exists() and report.exists():result=json.loads(report.read_text())
            else:
                time.sleep(1)
                metadata_url=f'https://www.gutenberg.org/ebooks/{book}.rdf'
                metadata=fetch(metadata_url,500_000);xml=ET.fromstring(metadata)
                ns={'d':'http://purl.org/dc/terms/'}
                rights=xml.find('.//d:rights',ns)
                if rights is None or rights.text!='Public domain in the USA.':
                    raise ValueError('Primary catalog does not confirm US public-domain status')
                title=xml.find('.//d:title',ns).text
                url=f'https://gutenberg.pglaf.org/cache/epub/{book}/pg{book}.txt'
                raw=fetch(url,20_000_000);text=raw.decode('utf-8-sig')
                start=re.search(r'\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*',text,re.I)
                end=re.search(r'\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*',text,re.I)
                if not start or not end or end.start()<=start.end():raise ValueError('Cannot safely strip Gutenberg wrapper')
                body=text[start.end():end.start()].strip()
                result=emit(path,[{'text':body,'title':title,'source':'Project Gutenberg','source_url':url,
                    'metadata_url':metadata_url,'license':'Public-domain-US','kind':'book','domain':'humanities',
                    'group_id':f'gutenberg:{book}','download_sha256':hashlib.sha256(raw).hexdigest()}])
                result.update({'book':book,'title':title,'source_url':url,'rights':rights.text,
                    'download_sha256':hashlib.sha256(raw).hexdigest(),
                    'metadata_sha256':hashlib.sha256(metadata).hexdigest(),
                    'collected_at':datetime.now(timezone.utc).isoformat()})
                report.write_text(json.dumps(result,indent=2)+'\n')
            successes.append(result);print(result,flush=True)
        except Exception as error:failures.append({'book':book,'error':str(error)});print(book,str(error),flush=True)
    with (root/'all.jsonl').open('w') as output:
        for book in successes:output.write((root/f'book-{book["book"]}.jsonl').read_text())
    (root/'manifest.json').write_text(json.dumps({'sources':successes,'failures':failures,
        'license_scope':'Public domain in the United States, per primary Gutenberg RDF catalog.',
        'all_sha256':hashlib.sha256((root/'all.jsonl').read_bytes()).hexdigest()},indent=2)+'\n')

if __name__=='__main__':main()
