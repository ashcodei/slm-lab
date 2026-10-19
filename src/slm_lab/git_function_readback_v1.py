"""Independent original-Git/function/native reconstruction without the producer."""
import ast
import doctest
import hashlib
import io
from pathlib import PurePosixPath
import tarfile
import tokenize
from urllib.parse import quote

from .documented_doctest_v1 import PRELUDE,compare,execute,unresolved_globals


def sha(text):return hashlib.sha256(text.encode()).hexdigest()


def original_files(archive_path,tree,revision):
    if tree.get('truncated') is not False:raise ValueError('Every original Git tree entry is required')
    ordinary={r['path']:r for r in tree['tree'] if r['type']=='blob' and r['mode'] in {'100644','100755'}}
    result={}
    with tarfile.open(archive_path,'r:gz') as archive:
        for member in archive:
            parts=PurePosixPath(member.name).parts
            if not parts or parts[0]!='Python-'+revision or member.name.startswith('/') or '..' in parts:
                raise ValueError('Every original archive path must retain its exact pinned Git root')
            if member.isdir() or member.issym() or member.islnk():continue
            path='/'.join(parts[1:]);entry=ordinary.get(path)
            if not member.isfile() or entry is None or path in result or member.size!=entry['size'] or member.size>128*1024**2:
                raise ValueError('Every ordinary source file needs one exact original bounded Git blob')
            raw=archive.extractfile(member).read()
            digest=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            if digest!=entry['sha'] or len(raw)!=entry['size']:raise ValueError('Actual original archive bytes differ from pinned Git blob')
            result[path]=(raw,entry)
    if result.keys()!=ordinary.keys():raise ValueError('No original ordinary Git file may be missing')
    return result


def qualify_file(path,raw,entry,repository,revision,license_sha,tokenizer,overlap):
    try:text=raw.decode('utf-8')
    except UnicodeDecodeError:return None,'original_file_not_UTF8'
    if not 32<=len(raw)<=1024**2 or '\0' in text:return None,'original_whole_Python_byte_bound'
    header=text[:8192].casefold()
    if any(s in header for s in ['gnu general public license','gnu lesser general','spdx-license-identifier: gpl']):
        return None,'original_file_notice_conflicts_with_repository_MIT_scope'
    try:ast.parse(text)
    except (SyntaxError,ValueError,RecursionError):return None,'whole_original_Python3_syntax_qualification'
    values=tokenizer.encode(text,add_special_tokens=False).ids
    if any(not 4<=v<32768 for v in values) or tokenizer.decode(values,skip_special_tokens=False)!=text:
        return None,'whole_original_Python_not_lossless_in_own_BPE'
    if overlap.literal(text) or overlap.lexical_copies(text):return None,'original_complete_file_has_declared_held_text_overlap'
    digest=hashlib.sha256(raw).hexdigest()
    return dict(id='public-git-file:'+digest,family='code',language='Python',original_UTF8_sha256=digest,
        original_public_text=text,role='valid' if int(digest[-8:],16)%100==0 else 'train',publisher_partition=None,
        source_kind='pinned_public_Git_file_not_HuggingFace_Parquet',source_repository=repository,source_revision=revision,
        source_git_path=path,original_git_tree_entry=entry,
        original_public_URL='https://github.com/'+repository+'/blob/'+revision+'/'+quote(path,safe='/'),
        original_publisher_row=dict(code=text,language='Python',path=path,repo_name=repository,repository_license='MIT'),
        original_publisher_row_is_derived_git_file_view_not_publisher_dataset_row=True,
        rights=dict(publisher_repository_license='MIT',full_repository_license_sha256=license_sha,
            original_complete_archive_and_source_file_notices_retained=True,per_file_license_text_not_separately_retrieved=True),
        no_publisher_train_or_test_partition_exists=True,deterministic_document_role_is_our_source_split=True,
        semantic_and_all_history_independence_not_certified=True),None


def original_signature(program):
    lines=program.splitlines(keepends=True);stack=[]
    for token in tokenize.generate_tokens(io.StringIO(program).readline):
        if token.type!=tokenize.OP:continue
        if token.string in '([{':stack.append(token.string)
        elif token.string in ')]}':stack.pop()
        elif token.string==':' and not stack:
            line,column=token.end
            return ''.join(lines[:line-1])+lines[line-1][:column]
    raise ValueError('A complete original function signature must rebuild')


def qualify_function(original,node,tokenizer,overlap):
    program=ast.get_source_segment(original['original_public_text'],node)
    if program is not None and '```' in program:return None,'embedded_markdown_fence_not_supported_by_declared_legacy_runtime_contract'
    if not isinstance(node,ast.FunctionDef) or node.decorator_list:return None,'decorated_async_or_nonordinary_function'
    description=ast.get_docstring(node,clean=False)
    if not program or not isinstance(description,str) or not 24<=len(description.encode())<=20000:
        return None,'missing_or_unbounded_original_literal_docstring'
    try:
        tests=doctest.DocTestParser().get_examples(description);missing=unresolved_globals(program)
    except (SyntaxError,ValueError,RecursionError):return None,'unsupported_original_docstring_or_function_context'
    if not 2<=len(tests)<=128 or any(t.options.get(doctest.SKIP,False) for t in tests):
        return None,'requires_all_original_examples_at_least_two_and_no_SKIP'
    if missing:return None,'unresolved_function_context_not_in_declared_stdlib'
    signature=original_signature(program)
    prompt=('Implement this Python function according to its original public documentation. '
        'Return the complete function only. The declared standard-library context is available.\n\n'
        'Declared standard-library context:\n'+PRELUDE+'\nOriginal function signature:\n'+signature+
        '\n\nOriginal literal docstring, including every published example:\n'+description)
    prefix=[1,*tokenizer.encode('User: '+prompt+'\nAssistant:',add_special_tokens=False).ids]
    response=[*tokenizer.encode(program,add_special_tokens=False).ids,2];ids=prefix+response
    if (len(ids)>4096 or len(response)<2 or any(not 4<=v<32768 for v in ids[1:-1])
            or tokenizer.decode(prefix[1:],skip_special_tokens=False)!='User: '+prompt+'\nAssistant:'
            or tokenizer.decode(response[:-1],skip_special_tokens=False)!=program):
        return None,'whole_original_function_contract_not_lossless_within4096'
    native=tokenizer.decode(ids[1:-1],skip_special_tokens=False)
    if overlap.literal(native) or overlap.lexical_copies(native):return None,'actual_function_native_input_or_response_has_declared_held_overlap'
    identity='public-function:'+sha(original['original_UTF8_sha256']+':'+str(node.lineno)+':'+sha(program))
    runtime=dict(source=dict(id=identity,original_partition='train',language='Python',prompt=description,response=program),
        function=node.name,program=program,program_sha256=sha(program),doctest_text=description,
        doctest_text_sha256=sha(description),existing_formatted_source_target_sha256=sha(program),whole_original_example_count=len(tests))
    compact=dict(id=identity,function=node.name,
        original_function_span=dict(lineno=node.lineno,end_lineno=node.end_lineno,col_offset=node.col_offset,end_col_offset=node.end_col_offset),
        original_function_program=program,original_function_program_sha256=sha(program),original_literal_docstring=description,
        original_literal_docstring_sha256=sha(description),original_signature=signature,prompt=prompt,response=program,
        runtime_candidate=runtime,role=original['role'],whole_original_example_count=len(tests),whole_native_ids=ids,
        whole_native_labels=[-100]*len(prefix)+response,native_target_start=len(prefix),complete_native_context_tokens=len(ids),
        response_EOS_targets=len(response),declared_prelude=PRELUDE,declared_prelude_sha256=sha(PRELUDE),
        prompt_derived_from_original_signature_and_literal_docstring_not_publisher_authored_instruction=True,
        complete_original_function_AST_segment_is_unchanged_response=True,
        worker_adapter_legacy_formatted_target_field_points_to_original_function_not_a_prior_training_row=True,
        model_generated_target=False,external_neural_teacher=False,human_preferences=False,optimization_admitted=False,
        all_original_published_examples_must_still_pass=True,source_semantic_and_all_history_independence_not_certified=True,
        original_publisher_partition=None,no_publisher_train_or_test_partition_exists=True,
        legacy_runtime_original_partition_train_is_adapter_field_not_publisher_metadata=True,
        original_deterministic_document_role_stays_sealed=True,
        original_source_document_id=original['id'],original_source_git_path=original['source_git_path'])
    return compact,None


def verify_candidate(actual,expected):
    if actual!=expected:raise ValueError('Every original source/function/docstring field, composed native token, role and EOS must rebuild')


def observe(candidate):
    result=execute(candidate['runtime_candidate'],seconds=3)
    result.update(expected_text_not_written_to_worker=False,original_program_docstring_includes_public_example_answers=True,
        expected_outputs_not_sent_as_separate_assertions_or_comparison_artifacts=True)
    return result


def observations_pass(candidate,observations):
    tests=doctest.DocTestParser().get_examples(candidate['original_literal_docstring'])
    if len(observations)!=2:raise ValueError('Two whole actual source-example observations are required')
    passed=True
    for observed in observations:
        if (observed.get('source_id')!=candidate['id'] or observed.get('program_sha256')!=candidate['original_function_program_sha256']
                or observed.get('doctest_text_sha256')!=candidate['original_literal_docstring_sha256']
                or observed.get('declared_prelude_sha256')!=candidate['declared_prelude_sha256']
                or observed.get('original_example_count')!=len(tests)
                or observed.get('expected_text_not_written_to_worker') is not False
                or observed.get('original_program_docstring_includes_public_example_answers') is not True
                or observed.get('expected_outputs_not_sent_as_separate_assertions_or_comparison_artifacts') is not True):
            raise ValueError('Every actual original source/program/example identity and visible-answer scope must remain exact')
        if observed.get('status')!='all_original_examples_passed':
            passed=False
            continue
        raw=observed.get('raw_results',[])
        if (len(raw)!=len(tests) or not observed.get('all_original_examples_observed') or not observed.get('all_original_examples_passed')
                or observed.get('entry_body_calls',0)<=0 or sum(r['entry_body_calls'] for r in raw)!=observed['entry_body_calls']
                or observed.get('passed_examples')!=len(tests) or observed.get('example_passes')!=[True]*len(tests)
                or not all(compare(test,row) for test,row in zip(tests,raw))):
            raise ValueError('Every originally claimed passing public case must match its actual raw observation')
    return passed and observations[0]['raw_results']==observations[1]['raw_results']
