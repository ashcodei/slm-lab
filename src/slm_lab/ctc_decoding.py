"""Bounded CTC prefix search, without a lexicon, reference words or text repair."""
import math

from .character_ctc import ALPHABET


def logadd(first,second):
    if first==-math.inf:return second
    if second==-math.inf:return first
    high,low=max(first,second),min(first,second)
    return high+math.log1p(math.exp(low-high))


def prefix_search(log_probabilities,beam_width=32,token_prune=16):
    if beam_width<1 or token_prune<1:raise ValueError('Positive decoding budgets required')
    frames=log_probabilities.tolist()
    if not frames or any(len(frame)!=len(ALPHABET)+1 for frame in frames):raise ValueError('CTC class/frame shape differs')
    if any(math.isnan(value) or value==math.inf for frame in frames for value in frame):raise ValueError('Invalid CTC log probabilities')
    beams={'':(0.,-math.inf)}
    for frame in frames:
        candidates=sorted(range(1,len(frame)),key=lambda value:(-frame[value],value))[:token_prune]
        candidates=[value for value in candidates if frame[value]!=-math.inf]
        following={}
        def add(prefix,blank=None,nonblank=None):
            a,b=following.get(prefix,(-math.inf,-math.inf))
            following[prefix]=(logadd(a,blank) if blank is not None else a,
                               logadd(b,nonblank) if nonblank is not None else b)
        for prefix,(blank,nonblank) in beams.items():
            total=logadd(blank,nonblank);add(prefix,blank=total+frame[0])
            for value in candidates:
                character=ALPHABET[value-1]
                if prefix and prefix[-1]==character:
                    add(prefix,nonblank=nonblank+frame[value])
                    add(prefix+character,nonblank=blank+frame[value])
                else:add(prefix+character,nonblank=total+frame[value])
        beams=dict(sorted(following.items(),key=lambda item:(-logadd(*item[1]),item[0]))[:beam_width])
    if not beams:raise ValueError('No finite CTC prefix path')
    return min(beams,key=lambda prefix:(-logadd(*beams[prefix]),prefix))
