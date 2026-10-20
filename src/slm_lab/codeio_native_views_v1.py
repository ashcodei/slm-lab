"""Complete direct code and I/O supervision candidates, without admission.

These targets contain original refactored public code or cached JSON values,
not generated rationales, human preferences or independent reward labels.
Every cached case receives both prediction directions. Reverse targets are
one recorded input candidate, not a unique inverse. Validity awaits checks.
"""
import hashlib
import json
import struct

from .codeio_published_cases_v1 import contract


def source_fields(row):
    values=[row.get(k) for k in ['problem_description','io_requirements','refcode','funcname']]
    if any(not isinstance(v,str) for v in values) or not values[2] or not values[3].isidentifier():
        raise ValueError('Complete original text fields and explicit Python entrypoint required')
    if any('\0' in v for v in values):raise ValueError('Original embedded null requires qualification')
    return values


def code_view(row):
    description,requirements,code,name=source_fields(row)
    prompt=('Write a complete Python solution matching the description and input/output contract below. '
        'Use the entrypoint '+name+'. Return only the complete code.\n\nDescription:\n'+description
        +'\n\nInput/output contract:\n'+requirements)
    return dict(view='code_generation',original_case_position=None,prompt=prompt,response=code)


def io_views(row):
    description,requirements,code,name=source_fields(row);cases=contract(row)
    shared=('\n\nDescription:\n'+description+'\n\nInput/output contract:\n'+requirements
        +'\n\nComplete Python code:\n```python\n'+code+'\n```')
    for position,case in enumerate(cases['cases']):
        inputs=json.dumps(case['input'],ensure_ascii=False,allow_nan=False,separators=(',',':'))
        output=json.dumps(case['output'],ensure_ascii=False,allow_nan=False,separators=(',',':'))
        prompt=('Determine the JSON return value when '+name+' is called with the keyword arguments below. '
            'Return only JSON.'+shared+'\n\nKeyword arguments:\n'+inputs)
        yield dict(view='output_prediction',original_case_position=position,prompt=prompt,response=output)
        prompt=('Find a JSON object of keyword arguments for '+name+' that produces the return value below. '
            'Return only JSON.'+shared+'\n\nReturn value:\n'+output)
        yield dict(view='input_prediction',original_case_position=position,prompt=prompt,response=inputs)


def native(tokenizer,view,*,limit=4096):
    if type(limit) is not int or limit<3:raise ValueError('Explicit native context bound required')
    for text,expected in [('<pad>',0),('<bos>',1),('<eos>',2),('<unk>',3)]:
        if tokenizer.token_to_id(text)!=expected:raise ValueError('Exact native special-token identities required')
    prompt='User: '+view['prompt']+'\nAssistant:';response=view['response']
    head=tokenizer.encode(prompt,add_special_tokens=False).ids
    target=tokenizer.encode(response,add_special_tokens=False).ids
    if (not target or any(not 4<=v<32768 for v in head+target)
            or tokenizer.decode(head,skip_special_tokens=False)!=prompt
            or tokenizer.decode(target,skip_special_tokens=False)!=response):
        raise ValueError('Complete native texts must round-trip without reserved source IDs or substitution')
    ids=[1]+head+target+[2];start=len(head)+1
    raw=struct.pack('<'+'H'*len(ids),*ids)
    geometry=dict(complete_context_tokens=len(ids),response_EOS_targets=len(target)+1,target_start=start,
        native_input_u16le_sha256=hashlib.sha256(raw).hexdigest(),
        native_target_mask_u8_sha256=hashlib.sha256(bytes([0]*start+[1]*(len(target)+1))).hexdigest(),
        fits_complete_native_context=len(ids)<=limit,whole_original_prompt_and_response_preserved=True)
    return ids,[-100]*start+ids[start:],geometry
