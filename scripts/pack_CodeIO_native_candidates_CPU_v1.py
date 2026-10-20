"""Pack all whole 4K code/I/O candidates while source execution continues.

Every original row/case is accounted for. Quarantined components remain in
the disposition ledger and never enter the candidate token store. All other
packs retain fixed roles and remain forbidden to training until a separate
complete execution/readback/license/held-source admission is implemented.
"""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
RAW=Path('/tmp/slm-CodeIO-raw-candidate-v1')
INDEX=Path('/tmp/slm-CodeIO-whole-source-index-v1')
OUT=Path('/tmp/slm-CodeIO-native-candidates-v1')
TOKENIZER=ROOT/'checkpoints/public-coding-research-v1/plain_tokens/tokenizer.json'
CONFIG=ROOT/'data/public-reasoned-code-SFT-v1/model-config.json'
GIB=1024**3


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(name,value):
    p=OUT/name;q=p.with_suffix(p.suffix+'.pending')
    with q.open('x') as f:
        f.write(json.dumps(value,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(q,p)


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or 'torch' in sys.modules
            or os.environ.get('TOKENIZERS_PARALLELISM')!='false'):
        raise ValueError('A fresh explicitly CPU-only one-thread source pack is required')
    os.sched_setaffinity(0,{22});os.nice(10)
    from slm_lab.codeio_native_views_v1 import code_view,io_views,native
    from slm_lab.codeio_published_cases_v1 import strict_loads
    from slm_lab.codenet_token_pack_v1 import CodePackWriter
    from slm_lab.local_source_closure_v2 import python_sources
    from tokenizers import Tokenizer
    tests=ROOT/'reports/codeio-native-views-v1-tests.log'
    if '2 passed' not in tests.read_text():raise ValueError('Complete-direction/source-text/EOS/no-cropping checks must pass')
    source=json.loads((RAW/'complete.json').read_text());index=json.loads((INDEX/'complete.json').read_text())
    path=Path(source['original_source_file']);roles=INDEX/'every-original-source-with-fixed-role.jsonl'
    files={str(p):sha(p) for p in [*python_sources(ROOT,[Path(__file__).resolve()]),tests,
        ROOT/'tests/test_codeio_native_views_v1.py',RAW/'complete.json',RAW/'publisher-README.md',INDEX/'complete.json',
        path,roles,TOKENIZER,CONFIG]}
    config=json.loads(CONFIG.read_text())
    if (not source['complete'] or not index['complete'] or index['complete_original_rows']!=227669
            or sha(path)!=source['original_source_sha256'] or sha(roles)!=index['final_role_ledger_sha256']
            or config['max_text_length']!=4096 or config['vocab_size']!=32768
            or sha(TOKENIZER)!='0072bb6de61da87bd9ba5f4e6739b594cc3e60c62c75a59d25d499629a70088f'):
        raise ValueError('Complete unchanged original sources, fixed roles and original native geometry required')
    if 'license: odc-by' not in (RAW/'publisher-README.md').read_text():
        raise ValueError('Original declared dataset attribution must remain explicit')
    stat=dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
    nonfile=sum(int(stat[k]) for k in ['anon','shmem','kernel'])
    if (int(Path('/sys/fs/cgroup/memory.max').read_text())-nonfile<56*GIB
            or shutil.disk_usage(OUT.parent).free<60*GIB):
        raise ValueError('8GiB own/48GiB other-job host funding and full16GiB pack/40GiB scratch reserve required')
    resource.setrlimit(resource.RLIMIT_AS,(8*GIB,8*GIB))
    tokenizer=Tokenizer.from_file(str(TOKENIZER))
    if tokenizer.get_vocab_size()!=32768 or 'torch' in sys.modules:
        raise ValueError('The exact own tokenizer without a model or Torch import is required')
    OUT.mkdir();started=time.monotonic();counts=Counter();by_view={};writer=None
    write('plan.json',dict(at=datetime.now(timezone.utc).isoformat(),
        owner=dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
        source_files_sha256=files,original_rows=227669,original_cached_pairs=1342795,
        views=['code_generation','output_prediction','input_prediction'],every_original_cached_case_has_both_IO_views=True,
        entire_original_prompts_and_responses_required=True,response_cropping_forbidden=True,
        fixed_source_roles_unchanged=True,quarantined_components_forbidden_to_pack=True,
        maximum_complete_context=4096,maximum_durable_pack_bytes=16*GIB,minimum_free_scratch_bytes=40*GIB,
        maximum_own_virtual_address_bytes=8*GIB,own_nice=10,CPU_affinity=[22],CPU_threads=1,
        no_Torch_or_model_or_GPU_loaded=True,original_refactored_code_and_cached_values_not_CoT_or_human_feedback=True,
        complete_source_execution_readback_license_and_admission_pending=True,data_admitted_to_training=False,RL_reward_admitted=False))
    try:
        for name,p in [('tokenizer.json',TOKENIZER),('model-config.json',CONFIG),('publisher-README.md',RAW/'publisher-README.md')]:
            shutil.copyfile(p,OUT/name)
        (OUT/'DATASET-LICENSE.txt').write_text('Candidate source: hkust-nlp/CodeIO-PyEdu-Reasoning-Raw\n'
            'Revision: 413a78caa01e634318ef8aa0e84f5ff242f25807\n'
            'Publisher-declared dataset license: ODC-BY. Original publisher README is retained.\n'
            'Original repository/path attribution remains in every packed row.\n'
            'This does not reinterpret or grant upstream code licenses. Source-license review and training admission are pending.\n'
            'Changes: native prompt framing, complete original code/JSON targets and fixed source roles; no original code edits.\n')
        writer=CodePackWriter(OUT,maximum_bytes=16*GIB,minimum_free_gib=40)
        with path.open('rb') as raw,roles.open() as role_stream,(OUT/'every-original-source-disposition.jsonl').open('x') as ledger:
            for position,line in enumerate(raw):
                r=json.loads(next(role_stream));counts['original_rows']+=1
                if r['original_position']!=position or hashlib.sha256(line).hexdigest()!=r['original_row_sha256']:
                    raise ValueError('Every exact original source row and fixed role must remain aligned')
                result=dict(original_position=position,original_row_sha256=r['original_row_sha256'],fixed_role=r['fixed_role'],
                    whole_component_quarantined=r['entire_component_quarantined_for_protected_signal'],views=[],
                    original_cached_cases=r['cached_case_count'],source_execution_and_license_admission_pending=True,
                    data_admitted_to_training=False)
                try:row=strict_loads(line)
                except (ValueError,TypeError,RecursionError) as error:
                    counts['strict_original_JSON_qualification']+=1;result['qualification']=str(error)
                else:
                    views=[]
                    try:views.append(code_view(row))
                    except (ValueError,TypeError,KeyError,RecursionError) as error:
                        counts['code_view_schema_qualification']+=1;result['code_view_qualification']=str(error)
                    try:views.extend(list(io_views(row)))
                    except (ValueError,TypeError,KeyError,RecursionError) as error:
                        counts['whole_IO_population_schema_qualification']+=1;result['whole_IO_qualification']=str(error)
                    role=r['fixed_role']['source_role'];split='train' if role=='train' else 'validation'
                    for view in views:
                        geometry=None;entry=dict(view=view['view'],original_case_position=view['original_case_position'])
                        try:ids,labels,geometry=native(tokenizer,view,limit=4096)
                        except (ValueError,TypeError,KeyError,RecursionError) as error:
                            entry.update(status='native_token_or_roundtrip_qualification',diagnostic=str(error))
                        else:
                            entry['native_geometry']=geometry
                            if result['whole_component_quarantined']:entry['status']='quarantined_entire_source_component'
                            elif not geometry['fits_complete_native_context']:entry['status']='whole_context_exceeds_4096'
                            else:
                                packed=dict(id='CodeIO:'+str(position)+':'+view['view']+':'+str(view['original_case_position']),
                                    group_id=r['fixed_role']['component_sha256'],split=split,publisher_split='original_raw_release',
                                    original_position=position,original_row_sha256=r['original_row_sha256'],
                                    original_fixed_role=r['fixed_role'],original_meta=row.get('meta'),publisher_source=row.get('source'),
                                    source_revision='413a78caa01e634318ef8aa0e84f5ff242f25807',
                                    dataset='hkust-nlp/CodeIO-PyEdu-Reasoning-Raw',
                                    original_cached_case_count=r['cached_case_count'],**view,
                                    source_execution_and_license_admission_pending=True,human_preferences=False,
                                    independent_RL_reward_oracle=False,admitted_to_training=False)
                                entry['candidate_pack_position']=len(writer.entries);writer.add(packed,ids,labels)
                                entry['status']='complete_native_candidate_only'
                        counter=by_view.setdefault(role,{}).setdefault(view['view'],Counter())
                        counter[entry['status']]+=1
                        if entry['status']=='complete_native_candidate_only':
                            counter['complete_context_tokens']+=geometry['complete_context_tokens']
                            counter['response_EOS_targets']+=geometry['response_EOS_targets']
                        result['views'].append(entry)
                ledger.write(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n')
                if (position+1)%100==0:
                    ledger.flush();writer.guard()
                    write('progress.json',dict(phase='complete_native_candidate_packing',actual_original_rows=position+1,
                        planned_original_rows=227669,actual_candidate_views=len(writer.entries),counts=dict(counts),
                        view_counts=by_view,elapsed_seconds=time.monotonic()-started,data_admitted_to_training=False))
            if position+1!=227669 or role_stream.read(1):raise ValueError('Every original row and role must be accounted for')
            ledger.flush();os.fsync(ledger.fileno())
        if any(sha(p)!=h for p,h in files.items()):raise ValueError('An original source/index/tokenizer/runtime byte changed')
        manifest=writer.finish(dict(dataset='hkust-nlp/CodeIO-PyEdu-Reasoning-Raw',
            revision='413a78caa01e634318ef8aa0e84f5ff242f25807',source_files_sha256=files,
            source_execution_readback_license_and_training_admission_pending=True,only_original_cached_JSON_or_code_targets=True))
        write('complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
            actual_original_rows=counts['original_rows'],actual_candidate_views=manifest['source_examples'],
            candidate_native_counts=manifest['counts'],view_counts=by_view,counts=dict(counts),
            manifest_sha256=sha(OUT/'manifest.json'),dispositions_sha256=sha(OUT/'every-original-source-disposition.jsonl'),
            stored_pack_bytes=manifest['stored_bytes'],elapsed_seconds=time.monotonic()-started,
            all_original_sources_and_cached_IO_accounted_for=True,complete_candidate_pack_not_training_admission=True,
            source_roles_and_original_bytes_unchanged=True,actual_model_updates=0,human_feedback=False,
            independent_RL_reward_admitted=False,data_admitted_to_training=False,model_quality_gain_proved=False,goal_achieved=False))
    except BaseException as error:
        if writer is not None:writer.close()
        write('failure.json',dict(error=repr(error),traceback=traceback.format_exc(),partial_outputs_retained=True,
            original_sources_unchanged=True,data_admitted_to_training=False));raise


if __name__=='__main__':main()
