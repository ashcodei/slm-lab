"""Pinned public Git blobs and honest repository-function contract adapters."""
import ast
import hashlib
from pathlib import PurePosixPath
import tarfile
from urllib.parse import quote

from .public_function_contracts_v1 import from_function,verified_training_row


def blob_sha(raw):return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()


def archive_files(path,tree,root_name):
    if tree.get('truncated') is not False:raise ValueError('A complete pinned Git tree is required')
    expected = {r['path']:r for r in tree['tree'] if r['type']=='blob' and r['mode'] in {'100644','100755'}}
    seen = set()
    with tarfile.open(path,'r:gz') as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != root_name or member.name.startswith('/') or '..' in parts:
                raise ValueError('Every archive path must remain inside the exact pinned repository root')
            if member.isdir():continue
            relative = '/'.join(parts[1:])
            if member.issym() or member.islnk():
                # Never dereference or execute a repository link.
                continue
            if not member.isfile() or relative not in expected or relative in seen:
                raise ValueError('Every ordinary archive file must have one exact pinned Git blob')
            entry = expected[relative]
            if member.size != entry['size'] or member.size > 128*1024**2:
                raise ValueError('The complete original blob must match its declared bounded byte size')
            raw = archive.extractfile(member).read()
            if len(raw) != entry['size'] or blob_sha(raw) != entry['sha']:
                raise ValueError('Actual complete archive bytes differ from the original pinned Git blob')
            seen.add(relative)
            yield relative,raw,entry
    if seen != set(expected):raise ValueError('Every original ordinary repository file must remain present')


def source_entry(path,raw,git_entry,*,repository,revision,repository_license_sha256,tokenizer,overlap):
    if blob_sha(raw) != git_entry['sha'] or len(raw) != git_entry['size']:
        raise ValueError('Exact original public Git blob bytes required')
    try:text = raw.decode('utf-8')
    except UnicodeDecodeError:return None,'original_file_not_UTF8'
    if not 32 <= len(raw) <= 1024**2 or '\0' in text:return None,'original_whole_Python_byte_bound'
    lower = text[:8192].casefold()
    if 'gnu general public license' in lower or 'gnu lesser general' in lower or 'spdx-license-identifier: gpl' in lower:
        return None,'original_file_notice_conflicts_with_repository_MIT_scope'
    try:ast.parse(text)
    except (SyntaxError,ValueError,RecursionError):return None,'whole_original_Python3_syntax_qualification'
    ids = tokenizer.encode(text,add_special_tokens=False).ids
    if any(not 4 <= n < 32768 for n in ids) or tokenizer.decode(ids,skip_special_tokens=False) != text:
        return None,'whole_original_Python_not_lossless_in_own_BPE'
    if overlap.literal(text) or overlap.lexical_copies(text):return None,'original_complete_file_has_declared_held_text_overlap'
    sha = hashlib.sha256(raw).hexdigest()
    return dict(id='public-git-file:'+sha,family='code',language='Python',original_UTF8_sha256=sha,
        original_public_text=text,role='valid' if int(sha[-8:],16)%100==0 else 'train',publisher_partition=None,
        source_kind='pinned_public_Git_file_not_HuggingFace_Parquet',source_repository=repository,source_revision=revision,
        source_git_path=path,original_git_tree_entry=git_entry,
        original_public_URL='https://github.com/'+repository+'/blob/'+revision+'/'+quote(path,safe='/'),
        original_publisher_row=dict(code=text,language='Python',path=path,repo_name=repository,repository_license='MIT'),
        original_publisher_row_is_derived_git_file_view_not_publisher_dataset_row=True,
        rights=dict(publisher_repository_license='MIT',full_repository_license_sha256=repository_license_sha256,
            original_complete_archive_and_source_file_notices_retained=True,per_file_license_text_not_separately_retrieved=True),
        no_publisher_train_or_test_partition_exists=True,deterministic_document_role_is_our_source_split=True,
        semantic_and_all_history_independence_not_certified=True),None


def repository_function(original,node,tokenizer,overlap):
    program = ast.get_source_segment(original['original_public_text'],node)
    if program is not None and '```' in program:
        return None,'embedded_markdown_fence_not_supported_by_declared_legacy_runtime_contract'
    core,reason = from_function(original,node,tokenizer,overlap)
    if core is None:return None,reason
    core.pop('original_publisher_role_is_train_derived_source_role_stays_sealed')
    core.update(original_publisher_partition=None,no_publisher_train_or_test_partition_exists=True,
        legacy_runtime_original_partition_train_is_adapter_field_not_publisher_metadata=True,
        original_deterministic_document_role_stays_sealed=True)
    return core,None


def verified_repository_row(candidate,observations,tokenizer,overlap):
    original = candidate['source_original_file']
    node = next((n for n in ast.parse(original['original_public_text']).body if isinstance(n,ast.FunctionDef)
        and n.lineno==candidate['original_function_span']['lineno']),None)
    if node is None:raise ValueError('The exact original ordinary repository function is required')
    rebuilt,reason = repository_function(original,node,tokenizer,overlap)
    if rebuilt != candidate or reason is not None:raise ValueError('Every Git-source field and sealed derived role must rebuild')
    core,reason = from_function(original,node,tokenizer,overlap)
    row,ids,labels = verified_training_row(core,observations,tokenizer,overlap)
    row.update(source_candidate=candidate,publisher_split=None,original_publisher_partition=None,
        no_publisher_train_or_test_partition_exists=True,
        legacy_runtime_partition_field_not_a_claim_of_publisher_training_labels=True)
    return row,ids,labels
