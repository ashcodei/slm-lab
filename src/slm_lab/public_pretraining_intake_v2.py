"""Normalize the publisher's GO spelling while retaining every original field."""
from .public_pretraining_intake_v1 import candidate as original_candidate


def language(value):
    return 'Go' if value=='GO' else value


def candidate(row,source,tokenizer,overlap):
    raw=row.get('language');canonical=language(raw)
    normalized=dict(row,language=canonical) if source['family']=='code' and canonical!=raw else row
    item,reason=original_candidate(normalized,source,tokenizer,overlap)
    if item is not None and source['family']=='code':
        item['original_publisher_row']=row
        item['original_publisher_language']=raw
        item['publisher_language_normalization']=dict(original=raw,canonical=canonical,
            original_source_fields_and_program_bytes_unchanged=True)
    return item,reason
