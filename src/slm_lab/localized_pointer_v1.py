"""Original transcript/attention spans and exact padded-image token targets.

Mouse traces are approximate annotator attention, not object boundaries or OCR.
No external model predictions or rewritten human transcript targets are used.
"""
from bisect import bisect_left,bisect_right
from collections import Counter
import io
import math
import re

from PIL import Image

STOP=set('a an the in this image picture we can see there is are was were some and or of at on to with as it that these those also has have had side left right top bottom background front'.split())


def canonical(text):return ' '.join(text.split())


def spans(record):
    timed=record['timed_caption'];traces=record['traces']
    if (not isinstance(record.get('caption'),str) or not isinstance(timed,list) or not isinstance(traces,list)
            or canonical(record['caption'])!=canonical(' '.join(x['utterance'] for x in timed))):
        raise ValueError('Complete original manual transcript must match its original timed utterances')
    segment_times=[]
    for segment in traces:
        times=[p['t'] for p in segment]
        if (any(not all(type(p[k]) in {int,float} and math.isfinite(p[k]) for k in ['x','y','t']) for p in segment)
                or any(t<0 for t in times) or times!=sorted(times)):
            raise ValueError('Original finite, ordered mouse trace samples required')
        segment_times.append(times)
    selected=[];previous_start=-1.
    for index,utterance in enumerate(timed):
        start,end=utterance['start_time'],utterance['end_time'];text=utterance['utterance']
        if (type(start) not in {int,float} or type(end) not in {int,float} or not math.isfinite(start)
                or not math.isfinite(end) or not 0<=start<=end or start<previous_start or not isinstance(text,str)):
            raise ValueError('Original finite nondecreasing word times required')
        previous_start=start
        words=re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",text.casefold())
        if not .15<=end-start<=3. or not 1<=len(words)<=12 or not any(w not in STOP and len(w)>=3 for w in words):continue
        options=[]
        for segment_index,(segment,times) in enumerate(zip(traces,segment_times)):
            first,last=bisect_left(times,start),bisect_right(times,end)
            indices=[i for i in range(first,last) if 0<=segment[i]['x']<1 and 0<=segment[i]['y']<1]
            if len(indices)<4 or len(indices)<.8*(last-first):continue
            points=[segment[i] for i in indices]
            if max(p['x'] for p in points)-min(p['x'] for p in points)>.3 or max(p['y'] for p in points)-min(p['y'] for p in points)>.3:continue
            options.append((len(indices),segment_index,indices))
        if not options:continue
        count,segment_index,indices=max(options,key=lambda x:(x[0],-x[1]))
        selected.append(dict(timed_utterance_index=index,query=text,start_time=start,end_time=end,
            source_point_refs=[[segment_index,i] for i in indices],source_point_count=count))
    return selected


def geometry(point,dimensions,size=448,grid=14):
    width,height=dimensions;x,y=point
    if (any(type(v) is not int or v<1 for v in [width,height,size,grid]) or size%grid
            or not all(math.isfinite(v) and 0<=v<1 for v in [x,y])):
        raise ValueError('Actual image dimensions and an original in-frame point required')
    resized=(size,round(height/width*size)) if width>=height else (round(width/height*size),size)
    offset=(round((size-resized[0])*.5),round((size-resized[1])*.5))
    padded=((offset[0]+x*resized[0])/size,(offset[1]+y*resized[1])/size)
    cell=int(padded[1]*grid)*grid+int(padded[0]*grid)
    if not 0<=cell<grid*grid:raise ValueError('Original in-frame point mapped outside real model grid')
    return padded,cell


def histogram(record,span,dimensions,size=448,grid=14):
    counts=Counter()
    for segment,index in span['source_point_refs']:
        point=record['traces'][segment][index]
        _,cell=geometry((point['x'],point['y']),dimensions,size,grid);counts[cell]+=1
    if sum(counts.values())!=span['source_point_count']:raise ValueError('Every retained original point sample required')
    return [[cell,count] for cell,count in sorted(counts.items())]


def jpeg_header(raw):
    if not isinstance(raw,bytes) or not raw or len(raw)>131072:raise ValueError('Bounded original JPEG header required')
    with Image.open(io.BytesIO(raw)) as image:
        if image.format!='JPEG' or image.getexif().get(274,1)!=1:
            raise ValueError('Only original JPEGs with identity EXIF orientation are eligible')
        if min(image.size)<96 or image.width*image.height>10_000_000:
            raise ValueError('Original JPEG dimensions exceed declared image contract')
        return image.size
