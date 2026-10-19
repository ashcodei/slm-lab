"""Actual public US legislation page images and reading regions from their PDF text layers."""
import argparse
from datetime import datetime,timezone
import hashlib
import heapq
import math
from pathlib import Path
import pypdfium2 as pdfium
from .diverse_images import asset,persist,pixel_hash,save_image

SOURCES={
    'train':['BILLS-119hr1enr','PLAW-117publ58','PLAW-117publ167',
             'PLAW-117publ169','PLAW-118publ27','PLAW-116publ260'],
    'valid':['PLAW-118publ158','PLAW-118publ94'],
}


def rectangle_text(rectangle,characters,tolerance=.1):
    """Ignore adjacent glyphs which merely intersect a text rectangle."""
    left,bottom,right,top=rectangle
    return ' '.join(''.join(char for char,box in characters
        if left-tolerance<=box[0] and bottom-tolerance<=box[1]
        and box[2]<=right+tolerance and box[3]<=top+tolerance).split())


def collect(output,cache,train_pages=32,valid_pages=20,regions=8):
    root=Path(output);root.mkdir(parents=True,exist_ok=True);(root/'images').mkdir(exist_ok=True)
    if (root/'manifest.json').exists():raise ValueError('Completed document dataset already exists')
    rows=[];sources=[];protected=set();errors=[]
    for split,packages in SOURCES.items():
        for package in packages:
            url=f'https://www.govinfo.gov/content/pkg/{package}/pdf/{package}.pdf'
            try:
                path,record=asset(cache,package+'.pdf',url,40_000_000);sources.append(record)
                with pdfium.PdfDocument(path) as document:
                    count=min(len(document),train_pages if split=='train' else valid_pages)
                    indices=sorted({int(i*len(document)/count) for i in range(count)})
                    if split=='valid':protected.update('govinfo:'+package+':'+str(i) for i in range(len(document)))
                    for page_index in indices:
                        page=document[page_index];textpage=page.get_textpage();width,height=page.get_size()
                        bitmap=page.render(scale=1.5);image=bitmap.to_pil().convert('RGB')
                        characters=[(chr(pdfium.raw.FPDFText_GetUnicode(textpage.raw,j)),textpage.get_charbox(j))
                                    for j in range(textpage.count_chars())]
                        candidates=[]
                        for i in range(textpage.count_rects()):
                            left,bottom,right,top=textpage.get_rect(i)
                            text=rectangle_text((left,bottom,right,top),characters)
                            if not 4<=len(text)<=100 or not any(c.isalpha() for c in text):continue
                            if right<=left or top<=bottom:continue
                            box=(max(0,math.floor((left-.25)*image.width/width)),
                                 max(0,math.floor((height-top-.25)*image.height/height)),
                                 min(image.width,math.ceil((right+.25)*image.width/width)),
                                 min(image.height,math.ceil((height-bottom+.25)*image.height/height)))
                            if min(box[2]-box[0],box[3]-box[1])<8:continue
                            key=f'{package}-{page_index}-{i}'
                            candidates.append((key,text,box))
                        selected=heapq.nsmallest(regions,candidates,key=lambda r:hashlib.sha256(r[0].encode()).digest())
                        group='govinfo:'+package+':'+str(page_index)
                        common={'group_id':group,'parent_document_id':package,'page_index':page_index,
                            'source':'US Government Publishing Office / GovInfo','source_url':url,
                            'download_sha256':record['sha256'],'license':'US-public-domain government legislation',
                            'license_url':'https://www.govinfo.gov/about/policies',
                            'annotation_origin':'Text and reading boxes extracted from original PDF text layer; not OCR-model output.',
                            'split':split,'domain':'document_pages','parent_pixel_sha256':pixel_hash(image)}
                        for key,text,box in selected:
                            crop=image.crop(box);relative,digest,pixels=save_image(root,key,crop)
                            rows.append({**common,'id':key,'task':'document_ocr','image':relative,
                                'prompt':'Read the text in this document image.','response':text,
                                'stored_image_sha256':digest,'model_pixel_sha256':pixels,
                                'page_pixel_box':list(box),'render_scale':1.5})
                        if selected:
                            key,text,box=selected[0];relative,digest,pixels=save_image(root,f'{package}-{page_index}-page',image)
                            normalized=[round(v,4) for v in (box[0]/image.width,box[1]/image.height,box[2]/image.width,box[3]/image.height)]
                            rows.append({**common,'id':key+'-page','task':'document_region_ocr','image':relative,
                                'prompt':'Read the text inside the box '+str(normalized)+'. Coordinates are normalized [left, top, right, bottom].',
                                'response':text,'normalized_box':normalized,
                                'stored_image_sha256':digest,'model_pixel_sha256':pixels})
                        bitmap.close();textpage.close();page.close()
                print('document',split,package,'selected pages',len(indices),'examples total',len(rows),flush=True)
            except Exception as error:
                errors.append({'package':package,'error':str(error)});print('failed document',package,str(error),flush=True)
    persist(root,rows,sources,protected,[
        'Small legislation-layout supplement; photos, scanned handwriting, tables and charts need broader data.',
        'Page images rendered faithfully from actual government PDFs; targets extracted from text layers can have segmentation errors.',
        'Entire held-out source documents are excluded from document training; cross-source pixel checks run in the combined curriculum.',
        'US legislation selected; copyright-free status of unrelated third-party material in government publications is not assumed.',
        'Whole-page reading can exceed the current 224px encoder resolution; legible region crops are included.'
    ],errors)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--cache',required=True);args=parser.parse_args();collect(args.output,args.cache)

if __name__=='__main__':main()
