"""Separate matched CPU BF16 coding evaluation with explicit derived casts.

All 96 original programs remain. The original FP32 job is not modified or used
as this protocol's baseline; both endpoints are generated again in CPU BF16.
"""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def run(args):
    import torch
    from slm_lab.CPU_frozen_cast_cache_v1 import CPUFrozenCastCache
    from slm_lab import code_quality
    from slm_lab.broad_anchored_policy_v3 import fresh,sources
    from slm_lab.broad_interim_snapshot_reader_v1 import materialize
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.code_tasks import TASKS,request
    from slm_lab.decoded_native_code_reexecution_v1 import reconstruct
    from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    from slm_lab.function_repair_evaluation_CPU_v1 import legacy_runtime
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.ordinary_python_function_cases_v1 import execute as ordinary,implementation_sources
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest

    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    snapshot=args.snapshot.resolve();out=args.output.absolute();receipt=json.loads((snapshot/'complete.json').read_text())
    probe=args.probe.resolve();probe_result=json.loads((probe/'complete.json').read_text())
    plan=json.loads((snapshot/'plan.json').read_text());completed=receipt['completed']
    if (out.exists() or out.resolve()!=out or receipt.get('complete') is not True or completed!=512
            or receipt.get('source_checkpoint_is_interim_not_completed_epoch') is not True
            or receipt.get('original_mutable_files_not_bound_or_changed') is not True
            or plan['planned_steps']!=3721 or len(TASKS)!=12 or torch.cuda.is_initialized()):
        raise ValueError('Fresh output and the exact committed512/3721snapshot/full original suite required')
    files=dict(receipt['source_files_sha256']);files[str(snapshot/'complete.json')]=sha(snapshot/'complete.json')
    for path in python_sources(ROOT,[Path(__file__)]):
        digest=sha(path)
        if str(path) in files and files[str(path)]!=digest:
            raise ValueError('Original immutable runtime source changed before BF16 evaluation')
        files[str(path)]=digest
    for path in [probe/'complete.json',probe/'owner-plan.json',probe/'full-source-parity.json']:
        files[str(path)]=sha(path)
    for path,digest in probe_result['source_files_sha256'].items():
        if path in files and files[path]!=digest:raise ValueError('Probe and assessment source bytes disagree')
        files[path]=digest
    files.update(implementation_sources());files[str(code_quality.WORKER)]=sha(code_quality.WORKER)
    guard=FileBindings(files);cast_cache=CPUFrozenCastCache()
    def checked(full=False, before_load=False):
        if (shutil.disk_usage(out.parent).free<41*1024**3 or torch.cuda.is_initialized()
                or int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')))*1024<48*1024**3):
            raise ValueError('CPU-only evaluation must retain40GiBscratch reserve, output headroom and48GiBhost headroom')
        stat=dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
        live=sum(int(stat[k]) for k in ('anon','shmem','kernel'))
        required = (60 if before_load else 48) * 1024**3
        if int(Path('/sys/fs/cgroup/memory.max').read_text())-live<required:
            raise ValueError('CPU BF16 inference needs12GiB before loading and48GiB remaining other-job headroom')
        cast_cache.assert_current()
        guard.check(full=full)
    checked(True, before_load=True);out.mkdir();owner=dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    write(out/'owner-plan.json',dict(owner=owner,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        snapshot=str(snapshot),baseline='complete own human_DPO starting actor',interim_arm='anchor_CE',interim_updates=completed,
        full_original_epoch_updates=3721,device='CPU',precision='BF16',CPU_threads=1,CPU_affinity=sorted(os.sched_getaffinity(0)),
        priority='nice19 SCHED_IDLE',generation_budget=512,prompt_frames=['completion','instruction'],
        working_allowance_bytes=12*1024**3,minimum_other_job_memory_bytes=48*1024**3,
        original_tasks_per_language=12,languages=['Python','JavaScript'],programs_per_endpoint=48,total_programs=96,
        original_GPU_BF16_results_are_not_the_matched_baseline=True,held_references_only_used_after_generation=True,
        original_CPU_FP32_job_is_not_changed_or_used_as_this_protocol_baseline=True,
        full_size_bit_exact_BF16_cast_and_native_generation_probe=str(probe/'complete.json'),
        derived_precision_copies_are_counted_as_resident_bytes_not_additional_learned_parameters=True,
        one_autocast_context_per_frame_prevents_stale_selected_actor_casts_across_endpoints=True,
        no_training_optimizer_or_RNG_resume=True,original_workers_not_changed=True,goal_achieved=False))
    try:
        print(json.dumps(dict(phase='load_exact_own_starting_policy',owner=owner)),flush=True)
        load_plan=copy.deepcopy(plan);load_plan['starting_policy_inputs']['checkpoint']=str(snapshot/'starting-human-DPO-actor.pt')
        model,meta=fresh(load_plan,'cpu');model.eval();expected_sources=sources(plan,sha(snapshot/'plan.json'))
        initial=model.export(expected_sources);corpus=ExpandedExecutionCorpus(ROOT)
        print(json.dumps(dict(phase='validate_all_four_private_states_and_every_original_journal_record')),flush=True)
        proof=materialize(model,corpus,plan,sha(snapshot/'plan.json'),[snapshot/f'rank-{r}.pt' for r in range(4)],
            snapshot/'committed-journal-prefix.jsonl',expected_completed=completed,checked=checked)
        corpus.guard.check(full=True);checked(True);write(out/'interim-source-state-readback.json',proof)
        trained=model.export(expected_sources);model.import_export(initial,expected_sources=expected_sources)
        cast_cache.attach(model.policy)
        print(json.dumps(dict(phase='start_actual_matched_code_generation',interim_actor_changed=proof['initial_actor_parts_sha256']!=proof['interim_actor_parts_sha256'])),flush=True)
        original_load=code_quality.load_checkpoint;original_generate=model.generate;reports={};actions=[]
        try:
            code_quality.load_checkpoint=lambda checkpoint,device,training_state=False:(model,meta)
            with inference_cache():
                for endpoint,payload,checkpoint in [('initial',initial,snapshot/'starting-human-DPO-actor.pt'),
                                                     ('step512',trained,snapshot/'rank-0.pt')]:
                    model.import_export(payload,expected_sources=expected_sources);model.eval()
                    expected_parts=tensor_digest(model.parts.named_parameters());reports[endpoint]={}
                    for frame in ('completion','instruction'):
                        counter=[0];frame_actions=[]
                        def generate(ids,*positional,**kwargs):
                            checked()
                            if not torch.is_autocast_enabled('cpu') or torch.get_autocast_dtype('cpu')!=torch.bfloat16:
                                raise ValueError('Every actual generation must execute in CPU BF16 autocast')
                            if positional or kwargs!={'max_new_tokens':512,'eos_token_id':2,'use_cache':True}:
                                raise ValueError('Every original cached greedy512-action case must keep the exact decoding contract')
                            index=counter[0];language=('Python','JavaScript')[index//12];task=TASKS[index%12][0]
                            print(json.dumps(dict(endpoint=endpoint,frame=frame,case=index+1,language=language,task=task,phase='generating')),flush=True)
                            begin=time.monotonic();generated=original_generate(ids,**kwargs)
                            if not torch.equal(ids,generated[:,:ids.shape[1]]):raise ValueError('Every generated delivery must retain its entire original prompt')
                            response=generated[0,ids.shape[1]:].tolist()
                            if not response or len(response)>512 or 2 in response[:-1]:raise ValueError('Actual response actions/EOS boundary differ')
                            record=dict(endpoint=endpoint,frame=frame,case=index+1,language=language,task=task,
                                inference_device_type='cpu',inference_precision='bf16',CPU_threads=torch.get_num_threads(),
                                complete_prompt_ids=ids[0].tolist(),actual_response_ids=response,
                                stopped_on_EOS=response[-1]==2,ended_on_budget=len(response)==512 and response[-1]!=2,
                                seconds=time.monotonic()-begin)
                            with (out/'actual-native-actions.jsonl').open('a') as stream:
                                stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                            frame_actions.append(record);actions.append(record);counter[0]+=1;checked();return generated
                        model.generate=generate
                        report_path=out/f'{endpoint}-{frame}.json'
                        with legacy_runtime(model,checkpoint,meta):
                            code_quality.evaluate(str(checkpoint),str(report_path),device='cpu',max_new_tokens=512,prompt_mode=frame)
                        model.generate=original_generate;report=json.loads(report_path.read_text())
                        if report.get('inference_precision')!='bf16' or report.get('actual_CPU_threads')!=1:
                            raise ValueError('Every sealed frame must truthfully report actual CPU BF16 execution')
                        if counter[0]!=24 or report['task_count']!=24:raise ValueError('Every original program must be generated and assessed')
                        ordinary_rows=[]
                        for case,action in zip(report['results'],frame_actions):
                            task,code=reconstruct(case,frame)
                            if (case['task']!=action['task'] or case['language']!=action['language']
                                    or corpus.tokenizer.decode(action['actual_response_ids'])!=case['raw_response']):
                                raise ValueError('Every actual action sequence must reconstruct its unchanged decoded delivery')
                            if case['language']=='Python':
                                checked();ordinary_rows.append(dict(task=task[0],actual_unchanged_code_sha256=hashlib.sha256(code.encode()).hexdigest(),
                                    execution=ordinary(code,task[0],request(task,'Python',code)['cases'])))
                        if tensor_digest(model.parts.named_parameters())!=expected_parts:raise ValueError('Evaluation cannot alter its source policy')
                        write(out/f'{endpoint}-{frame}-ordinary-Python.json',dict(all_original_programs=12,results=ordinary_rows,
                            functional_passed=sum(r['execution']['functional_passed'] for r in ordinary_rows)))
                        reports[endpoint][frame]=report;checked(True)
        finally:
            model.generate=original_generate;code_quality.load_checkpoint=original_load
            resident_cast_inventory=cast_cache.inventory();cast_cache.close()
        if len(actions)!=96:raise ValueError('Every original program in both matched endpoints remains required')
        paired=[];summary={}
        for endpoint in ('initial','step512'):
            rows=[r for frame in ('completion','instruction') for r in reports[endpoint][frame]['results']]
            summary[endpoint]=dict(programs=len(rows),syntax=sum(bool(r['candidate'].get('syntax_valid')) for r in rows),
                functional=sum(bool(r['candidate'].get('functional_passed')) for r in rows),
                empirical_efficiency=sum(bool(r['efficiency_stress_passed']) for r in rows),
                ordinary_Python=sum(json.loads((out/f'{endpoint}-{frame}-ordinary-Python.json').read_text())['functional_passed'] for frame in ('completion','instruction')))
        for frame in ('completion','instruction'):
            for before,after in zip(reports['initial'][frame]['results'],reports['step512'][frame]['results']):
                if (before['language'],before['task'])!=(after['language'],after['task']):raise ValueError('Both endpoints must use the same complete task order')
                paired.append(dict(frame=frame,language=before['language'],task=before['task'],
                    initial_functional=bool(before['candidate'].get('functional_passed')),step512_functional=bool(after['candidate'].get('functional_passed'))))
        checked(True);corpus.guard.check(full=True)
        write(out/'complete.json',dict(complete=True,owner=owner,summary=summary,paired=paired,actual_generated_programs=96,
            source_readback_sha256=sha(out/'interim-source-state-readback.json'),actual_action_journal_sha256=sha(out/'actual-native-actions.jsonl'),
            source_files_sha256=files,precision='matched CPU BF16',original_full_GPU_assessments_not_replaced=True,
            original_CPU_FP32_assessment_not_changed_or_replaced=True,
            extra_resident_derived_cast_inventory=resident_cast_inventory,
            additional_learned_parameters_from_cast_cache=0,
            performance_probe_sha256=files[str(probe/'complete.json')],
            repeated_small_development_suite_not_independent_final_benchmark=True,source_epoch_incomplete=True,
            shared_machine_empirical_timing_not_asymptotic_proof=True,goal_achieved=False))
        print(json.dumps(dict(complete=True,summary=summary)),flush=True)
    except BaseException as exc:
        cast_cache.close()
        write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),partial_outputs_retained=True,
            original_training_inputs_and_workers_not_changed=True,goal_achieved=False));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',type=Path,default=Path('/tmp/slm-broad-code-step512-snapshot-CPU-v1'))
    parser.add_argument('--output',type=Path,default=Path('/tmp/slm-broad-code-step512-code-evaluation-CPU-BF16-v1'))
    parser.add_argument('--probe',type=Path,default=Path('/tmp/slm-broad-CPU-BF16-generation-probe-v1'));args=parser.parse_args()
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)!=19
            or any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))):
        raise ValueError('CUDA-hidden nice19 one-thread environment required')
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    proof=json.loads((args.probe/'complete.json').read_text())
    probe_owner=dict(pid=proof['owner']['pid'],ticks=proof['owner']['birth'])
    if (not terminal(probe_owner) or not proof['complete']
            or not proof['full_source_BF16_cast_parity'] or not proof['all_three_complete_native_deliveries_bit_exact']
            or not proof['every_original_parameter_and_CPU_RNG_unchanged']
            or proof['separate_BF16_assessment_timing_admitted'] is not True):
        raise ValueError('Actual full-size exact-source BF16 parity and timing probe must complete successfully first')
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0));run(args)


if __name__=='__main__':main()
