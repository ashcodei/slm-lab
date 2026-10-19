"""Bounded data-only transport preserving Python container/value equality."""
import array


def encode(value,depth=0,budget=None):
    if budget is None:budget=[100000]
    budget[0]-=1
    if depth>64 or budget[0]<0:raise ValueError('Complete returned value exceeds declared structural bound')
    t=type(value)
    if value is None:return ['none']
    if t is bool:return ['bool',value]
    if t is int:return ['int',str(value)]
    if t is float:return ['float',value.hex()]
    if t is complex:return ['complex',value.real.hex(),value.imag.hex()]
    if t is str:return ['str',value]
    if t is bytes:return ['bytes',value.hex()]
    if t in (list,tuple,set,frozenset):
        return [{list:'list',tuple:'tuple',set:'set',frozenset:'frozenset'}[t],
            [encode(v,depth+1,budget) for v in value]]
    if t is dict:return ['dict',[[encode(k,depth+1,budget),encode(v,depth+1,budget)] for k,v in value.items()]]
    if t is array.array:return ['array',value.typecode,encode(value.tolist(),depth+1,budget)]
    raise ValueError('Returned type outside explicit data transport: '+t.__name__)


def decode(value,depth=0,budget=None):
    if budget is None:budget=[100000]
    budget[0]-=1
    if depth>64 or budget[0]<0 or not isinstance(value,list) or not value or type(value[0]) is not str:
        raise ValueError('Bounded tagged data required')
    tag=value[0]
    if tag=='none' and len(value)==1:return None
    if tag=='bool' and len(value)==2 and type(value[1]) is bool:return value[1]
    if tag=='int' and len(value)==2 and type(value[1]) is str:return int(value[1])
    if tag=='float' and len(value)==2 and type(value[1]) is str:return float.fromhex(value[1])
    if tag=='complex' and len(value)==3 and all(type(v) is str for v in value[1:]):
        return complex(float.fromhex(value[1]),float.fromhex(value[2]))
    if tag=='str' and len(value)==2 and type(value[1]) is str:return value[1]
    if tag=='bytes' and len(value)==2 and type(value[1]) is str:return bytes.fromhex(value[1])
    if tag in ('list','tuple','set','frozenset') and len(value)==2 and isinstance(value[1],list):
        rows=[decode(v,depth+1,budget) for v in value[1]]
        return {'list':list,'tuple':tuple,'set':set,'frozenset':frozenset}[tag](rows)
    if tag=='dict' and len(value)==2 and isinstance(value[1],list):
        pairs=[]
        for pair in value[1]:
            if not isinstance(pair,list) or len(pair)!=2:raise ValueError('Exact key/value pair required')
            pairs.append((decode(pair[0],depth+1,budget),decode(pair[1],depth+1,budget)))
        result=dict(pairs)
        if len(result)!=len(pairs):raise ValueError('Duplicate transported dictionary key')
        return result
    if tag=='array' and len(value)==3 and type(value[1]) is str and value[1] in 'bBuhHiIlLqQfd':
        return array.array(value[1],decode(value[2],depth+1,budget))
    raise ValueError('Unknown or malformed data-only value tag')
