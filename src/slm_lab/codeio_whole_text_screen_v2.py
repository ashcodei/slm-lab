"""Screen all source code and cached JSON text, keeping fixed components.

This extends the earlier description/contract screen. Literal/lexical hits
are exposure signals, not semantic equivalence or historical independence.
An uninspectable whole source cannot silently pass the admission screen.
"""
import hashlib
import json
import math


def whole_text(row):
    fields=[row.get(k) for k in ['problem_description','io_requirements','refcode','funcname']]
    if any(not isinstance(v,str) for v in fields):raise ValueError('Every complete original source text field is required')
    ios=row.get('ios')
    if not isinstance(ios,list):raise ValueError('Original cached-case container requires qualification')
    # Check the complete original JSON population without accepting NaN/Inf or
    # replacing unsupported objects. Then visit every original key and value.
    json.dumps(ios,ensure_ascii=False,allow_nan=False)
    parts=list(fields);stack=[ios]
    while stack:
        value=stack.pop()
        if isinstance(value,dict):
            for key,item in reversed(list(value.items())):
                if not isinstance(key,str):raise ValueError('Original JSON object keys must be strings')
                stack.extend([item,key])
        elif isinstance(value,list):stack.extend(reversed(value))
        elif isinstance(value,str):parts.append(value)
        elif value is None or type(value) in [bool,int,float]:
            if type(value) is float and not math.isfinite(value):raise ValueError('Nonfinite cached value requires qualification')
            parts.append(json.dumps(value,allow_nan=False))
        else:raise ValueError('Every complete original cached JSON value must be inspectable')
    text='\n'.join(parts)
    if '\0' in text:raise ValueError('An original embedded boundary sentinel requires whole-source qualification')
    return text


def inspect(row,index):
    text=whole_text(row)
    return dict(full_source_text_utf8_sha256=hashlib.sha256(text.encode()).hexdigest(),
        complete_source_text_characters=len(text),literal_held_stem_ids=sorted(index.literal(text)),
        lexical_held_stem_ids=sorted(index.lexical_copies(text)),
        includes_entire_code_comments_and_all_cached_JSON_keys_values=True,
        source_text_cropped=False,semantic_or_historical_independence_certified=False)


def blocked_components(records):
    return {r['fixed_role']['component_sha256'] for r in records
        if r['prior_component_quarantined'] or r['whole_source_screen_qualification']
        or r['screen']['literal_held_stem_ids'] or r['screen']['lexical_held_stem_ids']}
