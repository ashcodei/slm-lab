"""Whole public documents for a source preflight, never optimization admission.

This applies explicit per-file code-license filters and scans the actual own
native text for the declared held stems. It does not certify semantics,
algorithmic correctness, all-history novelty or legal rights in every page.
"""
import ast
import hashlib
from urllib.parse import quote,urlsplit

PERMISSIVE={'mit','apache-2.0','bsd-2-clause','bsd-3-clause','isc','cc0-1.0','unlicense','zlib','0bsd'}
LANGUAGES={'Python','C','C++','Java','JavaScript','TypeScript','Go','Rust'}


def candidate(row,source,tokenizer,overlap):
    family=source['family'];field='code' if family=='code' else 'text'
    if family not in {'web','math','code'}:raise ValueError('An explicit pinned public source family is required')
    text=row.get(field)
    if not isinstance(text,str) or not text.strip() or '\0' in text:return None,'empty_nontext_or_NUL'
    raw=text.encode('utf-8')
    if not 32<=len(raw)<=1024**2:return None,'whole_document_byte_bound'
    if family=='code':
        license=row.get('license');language=row.get('language');repo=row.get('repo_name');path=row.get('path')
        if not isinstance(license,str) or license.casefold() not in PERMISSIVE:return None,'per_file_license_not_in_declared_allowlist'
        if language not in LANGUAGES:return None,'outside_declared_programming_languages'
        if not isinstance(repo,str) or repo.count('/')!=1 or not isinstance(path,str) or not path:return None,'missing_original_repository_or_path'
        if any(marker in (repo+'/'+path).casefold() for marker in ['humaneval','human_eval','mbpp','gsm8k','mmlu']):
            return None,'recognizable_held_benchmark_path'
        if language=='Python':
            try:ast.parse(text)
            except (SyntaxError,ValueError,MemoryError,RecursionError):return None,'whole_Python3_syntax_qualification'
        origin='https://github.com/'+repo+'/blob/HEAD/'+quote(path.lstrip('/'),safe='/')
        rights=dict(publisher_per_file_license=license,repository=repo,path=path,
            repository_HEAD_URL_is_attribution_not_verified_current_file_bytes=True,
            dataset_catalog_license_does_not_replace_per_file_license=True,license_text_not_independently_retrieved=True)
    else:
        origin=row.get('url')
        if not isinstance(origin,str) or urlsplit(origin).scheme not in {'http','https'} or not urlsplit(origin).netloc:
            return None,'missing_original_public_page_URL'
        rights=dict(dataset_license=source['dataset_license'],underlying_page_rights_not_relicensed=True,
            CommonCrawl_terms_apply=True)
    values=tokenizer.encode(text,add_special_tokens=False).ids
    if not values or any(type(v) is not int or not 4<=v<32768 for v in values):return None,'reserved_boundary_inside_original_text'
    if tokenizer.decode(values,skip_special_tokens=False)!=text:return None,'own_BPE_not_lossless_for_whole_original_text'
    # Compare the actual decoded model body, not an unconsumed metadata field.
    decoded=tokenizer.decode(values,skip_special_tokens=False)
    literal=overlap.literal(decoded);lexical=overlap.lexical_copies(decoded)
    if literal or lexical:return None,'declared_held_stem_or_lexical_copy_signal'
    digest=hashlib.sha256(raw).hexdigest();ids=[1,*values,2]
    return dict(id='public-source:'+digest,original_public_text=text,original_UTF8_sha256=digest,
        original_public_URL=origin,rights=rights,family=family,language=row.get('language') if family=='code' else 'English',
        source_repository=source['repo'],source_revision=source['revision'],source_parquet=source['path'],
        original_publisher_row=row,role='valid' if int(digest[-8:],16)%100==0 else 'train',
        whole_native_ids=ids,whole_native_labels=[-100,*ids[1:]],complete_context_tokens=len(ids),
        complete_causal_targets=len(ids)-1,whole_document_preserved_without_clipping=True,
        own_BPE_decodes_entire_original_UTF8_text_exactly=True,
        source_quality_filter_is_not_human_or_execution_correctness_label=True,
        human_preferences=False,execution_reward=False,optimization_admitted=False,
        source_semantic_and_all_history_independence_not_certified=True), 'whole_public_native_document_ready_for_independent_readback'
