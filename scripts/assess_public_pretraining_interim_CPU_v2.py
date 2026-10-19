"""Matched full local CPU assessment of initial and one pinned partial model.

Two low-priority CPU threads, one resident inference model at a time, no CUDA
context or GPU lease. The full original public training, optional GPU research,
human/RL lineage and complete final benchmarks remain independent and intact.
"""
from datetime import datetime,timezone
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback

from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.local_source_closure_v2 import python_sources

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/public-pretraining-interim-CPU-v2'
DEST=ROOT/'checkpoints/public-pretraining-interim-CPU-v2'
MATH=ROOT/'reports/public-pretraining-v1/plan.json'
RUNTIME=ROOT/'reports/public-pretraining-recovery-v1/plan.json'
RESUME=ROOT/'checkpoints/public-pretraining-v1/resume.pt'
INITIAL=ROOT/'checkpoints/public-pretraining-v1/initial/model.pt'
MAX_MODEL=INITIAL.stat().st_size+64*1024**2;RESERVE=26*1024**3


def read(path):return json.loads(Path(path).read_text())
def birth():return dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])


def available():
    memory={k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
    return shutil.disk_usage(ROOT).free,memory,int(Path('/sys/fs/cgroup/memory.max').read_text())


def ordinary_python(reports,destination):
    from slm_lab.code_tasks import TASKS,request
    from slm_lab.ordinary_python_function_cases_v1 import execute
    tasks={task[0]:task for task in TASKS};results=[];references={}
    for frame in ['completion','instruction']:
        report=reports['code-'+frame]
        for value in report['results']:
            if value['language']!='Python':continue
            task=tasks[value['task']];cases=request(task,'Python',task[4])['cases']
            if task[0] not in references:
                reference=execute(task[4],task[0],cases)
                if not reference['functional_passed']:raise ValueError('Every unchanged local reference must pass the ordinary stdlib protocol')
                references[task[0]]=reference
            program=value['code']
            if not program or len(program.encode())>50000 or '\0' in program:
                result=dict(status='empty_or_unbounded_native_program',functional_passed=False,
                    all_cases_observed=False,total_cases=len(cases),passed_cases=0,source_bytes_not_repaired=True)
            else:result=execute(program,task[0],cases)
            results.append(dict(frame=frame,task=task[0],program=value['code'],ordinary_stdlib_execution=result))
    if len(results)!=24 or len(references)!=12:raise ValueError('Every original generated Python function and unchanged case population required')
    report=dict(complete=True,programs=24,functional_passed=sum(r['ordinary_stdlib_execution']['functional_passed'] for r in results),
        results=results,original_reference_executions=references,generated_programs_unmodified=True,neural_forwards=0,
        efficiency_not_assessed_by_ordinary_stdlib_protocol=True,goal_achieved=False)
    atomic_json(destination,report);return report


def scores(reports,ordinary):
    code=[r for frame in ['completion','instruction'] for r in reports['code-'+frame]['results']]
    return dict(code_functional=dict(correct=sum(r['candidate'].get('functional_passed',False) for r in code),total=48),
        code_syntax=dict(correct=sum(r['candidate'].get('syntax_valid',False) for r in code),total=48),
        code_efficiency_stress=dict(correct=sum(r['efficiency_stress_passed'] for r in code),total=48),
        ordinary_Python_functional=dict(correct=ordinary['functional_passed'],total=24),
        knowledge_generated=dict(correct=sum(r['generated_correct'] for r in reports['knowledge']['predictions']),total=56),
        knowledge_likelihood=dict(correct=sum(r['likelihood_correct'] for r in reports['knowledge']['predictions']),total=56),
        math=dict(correct=reports['math']['correct'],total=16),
        OCR=dict(correct=sum(r['exact_match'] for r in reports['ocr']['results']),total=20),
        objects=dict(correct=sum(r['proper_exact'] for r in reports['objects']['results']),total=15),
        photos={k:dict(correct=round(reports['vision']['caption_retrieval'][k]*32),total=32)
            for k in ['image_to_text_accuracy','text_to_image_accuracy']})


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or any(p.exists() for p in [OUT,DEST])
            or len(os.sched_getaffinity(0))<8):raise ValueError('Fresh CUDA-hidden low-priority CPU assessment on the original eight-core affinity required')
    lock=(ROOT/'reports/public-pretraining-interim-CPU.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import torch
    from slm_lab.model import ModelConfig
    from slm_lab.public_pretraining_data_v2 import PublicPretrainingData
    from slm_lab.public_pretraining_run_contract_v1 import schedule_for
    from slm_lab.public_pretraining_interim_export_v1 import pin,resources,validate_source,inference_package,exact_values
    from slm_lab.public_pretraining_native_v1 import load
    from slm_lab.dense_native_view_v2 import CompleteDenseNativeView
    from slm_lab.function_repair_evaluation_CPU_v1 import local_suites
    from slm_lab.parallel_function_repair_stage_v1 import value_digest
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    if torch.cuda.is_initialized():raise ValueError('CPU-only assessment must not initialize any GPU')
    previous=ROOT/'reports/public-pretraining-interim-CPU-v1'
    if (not (previous/'failure.json').exists() or not terminal(read(previous/'capture-plan.json')['owner'])
            or 'Every actual full native cold output must be finite CPU BF16' not in read(previous/'failure.json')['error']):
        raise ValueError('The prior precision-guard failure must be verified terminal; no live reader may be replaced')
    tests=ROOT/'reports/public-pretraining-interim-export-v1-tests.log'
    evaluator_tests=ROOT/'reports/function-repair-evaluation-CPU-v1-tests.log'
    if '4 passed' not in tests.read_text() or '2 passed' not in evaluator_tests.read_text():raise ValueError('Actual pinned-source/cold/precision/native-evaluation controls required')
    precision_tests=ROOT/'reports/public-native-cold-inference-v1-tests.log'
    if '1 passed' not in precision_tests.read_text():raise ValueError('Actual CPU BF16 text/photo and original FP32 word-readout regression must pass')
    plan=read(MATH);runtime=read(RUNTIME);me=birth();math_sha=file_hash(MATH)
    if terminal(runtime['owner']):raise ValueError('This interim reader requires the actual original public producer to be live at capture')
    free,stat,limit=available();capacity=resources(free,RESUME.stat().st_size,stat,limit,export_maximum=MAX_MODEL)
    pulses=plan['pulse_sources'];paths=[MATH,RUNTIME,INITIAL,INITIAL.parent/'tokenizer.json',tests,evaluator_tests,precision_tests,
        previous/'capture-plan.json',previous/'failure.json',ROOT/'src/slm_lab/public_native_cold_inference_v1.py',
        ROOT/'data/math-reasoning-v2/test.jsonl',ROOT/'data/math-reasoning-v2/manifest.json',
        Path(pulses['knowledge']).with_suffix('.audit.json'),ROOT/'scripts/python_function_observations_v1.py',
        ROOT/'scripts/code_worker.py',ROOT/'scripts/native_sandbox.py']
    for name in pulses.values():
        path=Path(name);paths.append(path)
        for row in map(json.loads,path.read_text().splitlines()):
            if row.get('image'):paths.append((path.parent/row['image']).resolve())
    paths += python_sources(ROOT,[Path(__file__),ROOT/'tests/test_public_pretraining_interim_export_v1.py'])
    files={str(p):file_hash(p) for p in set(paths)}
    if files[str(INITIAL)]!=runtime['initial_checkpoint_sha256']:raise ValueError('Exact immutable original public-pretraining initial endpoint required')
    guard=FileBindings(files);OUT.mkdir();DEST.mkdir();initial_header=torch.load(INITIAL,map_location='cpu',weights_only=True,mmap=True)
    config=ModelConfig(**initial_header['complete_dense']['config']);del initial_header
    atomic_json(OUT/'capture-plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=me,source_files_sha256=files,
        actual_live_public_producer=runtime['owner'],resource_preflight=capacity,
        source_selection='first complete atomically published resume inode available after full data/source preparation',
        fixed_endpoints=['initial','interim'],CPU_threads=2,CPU_nice=os.nice(0),precision='cpu_bf16',
        original_native_word_readout_precision='fp32_inside_original_model',
        maximum_model_bytes_derived_from_actual_complete_initial_serialization_plus64MiB=True,
        prior_failed_attempt_preserved=str(previous),no_original_science_or_evaluation_scope_changed=True,
        full_fixed_populations=dict(local_code=48,code_native_budget=512,ordinary_Python=24,knowledge=56,math=16,OCR=20,objects=15,photos=32,image_size=448),
        source_validation15_and_full_final_benchmarks_remain_in_original_GPU_studies=True,
        one_resident_inference_model_at_a_time=True,maximum_single_student_values=984399690,
        native_outputs_never_repaired=True,new_neural_updates=0,candidate_promoted=False,goal_achieved=False))
    def interrupted(signum,frame):raise KeyboardInterrupt('Owned CPU assessment interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        atomic_json(OUT/'progress.json',dict(phase='preparing exact original public source schedule',goal_achieved=False))
        data=PublicPretrainingData(ROOT,config);schedule=schedule_for(data,plan,'train')
        provider_guard=FileBindings(data.source_files_sha256);provider_guard.check(full=True)
        files.update(data.source_files_sha256);guard=FileBindings(files)
        while True:
            if terminal(runtime['owner']):raise ValueError('Exact public producer ended before its interim capture')
            free,stat,limit=available()
            try:capacity=resources(free,RESUME.stat().st_size,stat,limit,export_maximum=MAX_MODEL)
            except ValueError as exc:
                atomic_json(OUT/'progress.json',dict(phase='waiting for original main writer resources',reason=str(exc),goal_achieved=False));time.sleep(5);continue
            break
        before_gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        with pin(RESUME) as pinned:
            state=pinned.load();raw=(ROOT/'reports/public-pretraining-recovery-v1/training-metrics.jsonl').read_bytes()
            complete=raw[:raw.rfind(b'\n')+1];journal=list(map(json.loads,complete.splitlines()))
            source_proof=validate_source(state,plan,math_sha,schedule,data,journal);completed=source_proof['actual_completed_updates']
            prefix=b''.join(complete.splitlines(keepends=True)[:completed]);prefix_path=OUT/'captured-complete-source-prefix.jsonl';prefix_path.write_bytes(prefix)
            source_identity=pinned.identity();package=inference_package(state,plan,math_sha)
            shutil.copyfile(INITIAL.parent/'tokenizer.json',DEST/'tokenizer.json')
            atomic_json(OUT/'progress.json',dict(phase='exporting exact pinned inference values',actual_checkpoint_update=completed,goal_achieved=False))
            atomic_checkpoint(DEST/'model.pt',package,MAX_MODEL,RESERVE);captured_sha=file_hash(DEST/'model.pt')
            cold,metadata=load(DEST/'model.pt',MATH,'step-'+str(completed),expected_checkpoint_sha256=captured_sha,expected_plan_sha256=math_sha,device='cpu')
            parity=exact_values(cold,package)
            if cold.counts()['total_unique']!=984399690:raise ValueError('Every actual984399690dense/vision/word/head value required')
            raw_name,raw_positions=plan['capacity_global_batches'][0];photo_name,photo_positions=plan['capacity_global_batches'][-2]
            word_position=next(i for i,r in enumerate(data.retention.replay.words.rows) if r['byte_family']=='real_word')
            text=data.batch(raw_name,[raw_positions[0]],'cpu');photo=data.batch(photo_name,[photo_positions[0]],'cpu');word=data.batch('words',[word_position],'cpu')
            native_inputs={n:{k:b[k] for k in ['input_ids','attention_mask','pixel_values'] if k in b} for n,b in [('text',text),('photo',photo)]}
            native_inputs['word']={'word_pixels':word['word_pixels']}
            atomic_checkpoint(DEST/'native-cold-inputs.pt',native_inputs,16*1024**2,RESERVE)
            from slm_lab.public_native_cold_inference_v1 import probe
            atomic_json(OUT/'progress.json',dict(phase='cold full-size4096text448photo real-word forwards',actual_checkpoint_update=completed,goal_achieved=False))
            cold_probes=probe(cold,native_inputs)
            if text['input_ids'].shape[1]!=4096 or photo['pixel_values'].shape[-2:]!=(448,448):raise ValueError('Actual full4096text and448photo native contexts required')
            pinned.check();provider_guard.check();guard.check()
            after_gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
            if before_gpu.splitlines()!=after_gpu.splitlines() or terminal(runtime['owner']) or torch.cuda.is_initialized():
                raise ValueError('The same actual main GPU compute births must continue throughout this CUDA-hidden capture')
            atomic_json(OUT/'captured-source.json',dict(complete=True,actual_checkpoint_update=completed,source_identity=source_identity,
                actual_source_prefix_validation=source_proof,complete_source_prefix_sha256=file_hash(prefix_path),
                full_model_values_after_cold_load=parity,actual_native_cold_probes=cold_probes,
                original_actual_GPU_compute_PIDs_unchanged=True,resource_preflight=capacity,
                native_input_bundle_sha256=file_hash(DEST/'native-cold-inputs.pt'),native_input_semantic_sha256=value_digest(native_inputs),
                captured_standalone_sha256=captured_sha,whole_public_pretraining_epoch_complete=False,
                optimizer_and_private_RNG_not_loaded_for_inference_or_resumed=True,new_neural_updates=0,goal_achieved=False))
            del state,package,cold,text,photo,word,native_inputs,metadata
        del data,schedule;gc.collect();torch.clear_autocast_cache()
        for path in [OUT/'captured-source.json',OUT/'captured-complete-source-prefix.jsonl',DEST/'model.pt',DEST/'tokenizer.json',DEST/'native-cold-inputs.pt']:
            files[str(path)]=file_hash(path)
        guard=FileBindings(files)
        endpoints=dict(initial=dict(checkpoint=str(INITIAL),sha256=runtime['initial_checkpoint_sha256'],arm='initial',completed_updates=0),
            interim=dict(checkpoint=str(DEST/'model.pt'),sha256=captured_sha,arm='step-'+str(completed),completed_updates=completed))
        atomic_json(OUT/'evaluation-plan.json',dict(owner=me,source_files_sha256=files,actual_endpoints=endpoints,
            fixed_endpoint_order=['initial','interim'],same_CPU_precision_threads_original_scorers_populations_and_budgets=True,
            all_original_GPU_training_and_full_benchmarks_independent=True,no_new_training_or_candidate_adoption=True,goal_achieved=False))
        results={}
        for name,endpoint in endpoints.items():
            guard.check();atomic_json(OUT/'progress.json',dict(phase='full native local assessment',endpoint=name,
                endpoint_completed_updates=endpoint['completed_updates'],already_complete_endpoints=list(results),goal_achieved=False))
            model,metadata=load(endpoint['checkpoint'],MATH,endpoint['arm'],expected_checkpoint_sha256=endpoint['sha256'],expected_plan_sha256=math_sha,device='cpu')
            view=CompleteDenseNativeView(model).eval()
            if sum(p.numel() for p in view.parameters())!=984399690 or torch.cuda.is_initialized():raise ValueError('One own complete under1B CPU endpoint required')
            reports=local_suites(view,Path(endpoint['checkpoint']),metadata,OUT/name,ROOT,pulses)
            ordinary=ordinary_python(reports,OUT/name/'ordinary-Python.json');result=scores(reports,ordinary)
            atomic_json(OUT/(name+'-complete.json'),dict(complete=True,endpoint=endpoint,actual_matched_local_metrics=result,
                original_native_action_journals_retained=True,all48native_code_programs_scored_and_all24Python_additionally_rerun=True,
                same_CPU_BF16_two_thread_protocol=True,new_neural_updates=0,candidate_promoted=False,goal_achieved=False))
            results[name]=result;del view,model,metadata,reports,ordinary;gc.collect();torch.clear_autocast_cache();guard.check()
        guard.check(full=True)
        atomic_json(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),actual_endpoints=endpoints,
            actual_matched_metrics=results,original_public_pretraining_source_and_all_frozen_GPU_lineage_unmodified=True,
            matched_CPU_BF16_local_development_metrics_not_full_independent_benchmarks=True,
            original_last_completed_GPU_metrics_not_treated_as_same_precision_matched_control=True,
            source_files_sha256={**files,**{str(p):file_hash(p) for p in OUT.glob('*-complete.json')}},
            actual_new_neural_updates=0,candidate_promoted=False,goal_achieved=False))
        atomic_json(OUT/'progress.json',dict(phase='complete',actual_matched_metrics=results,goal_achieved=False))
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,
            original_GPU_work_not_signalled_or_restarted=True,goal_achieved=False));raise


if __name__=='__main__':main()
