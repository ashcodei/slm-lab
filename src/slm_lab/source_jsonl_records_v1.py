"""Read physical UTF-8 JSONL records without splitting Unicode text values."""
import json
from pathlib import Path


def records(path, maximum_record_characters=16*1024**2):
    if type(maximum_record_characters) is not int or maximum_record_characters < 1:
        raise ValueError('A positive physical record bound is required')
    def pairs(values):
        result={}
        for key,value in values:
            if key in result: raise ValueError('Duplicate source JSON key requires review')
            result[key]=value
        return result
    def constant(value): raise ValueError('Nonfinite source JSON number requires review')
    with Path(path).open('r',encoding='utf-8',newline='\n') as stream:
        position=0
        while True:
            line=stream.readline(maximum_record_characters+1)
            if not line: break
            position+=1
            if len(line)>maximum_record_characters or not line.strip():
                raise ValueError(f'Physical source record {position} is oversized or empty')
            value=json.loads(line,object_pairs_hook=pairs,parse_constant=constant)
            if not isinstance(value,dict): raise ValueError('Every physical source record must be an object')
            yield value
