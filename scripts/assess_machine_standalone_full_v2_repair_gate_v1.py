"""Full original assessments of the fixed final machine-DPO dense student.

Waits for both the intact machine comparison and its standalone CPU export to
finish. It loads only the standalone weights/tokenizer for neural inference.
All four original L40s score each suite; no training, selection or promotion.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import train_whole_human_SFT_v1 as legacy
import own_training_gpu_lease_v2 as owner
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.public_pretraining_lineage_tests_v1 import bind_tests
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/machine-standalone-full-assessment-v2'
QUEUE=ROOT/'reports/machine-standalone-full-assessment-v2-repair-gate-v1-queue.json'
PREVIOUS=ROOT/'reports/full-machine-preference-v2'
PREVIOUS_QUEUE=ROOT/'reports/full-machine-preference-v2-repair-gate-v1-queue.json'
FINALIZER=ROOT/'reports/machine-preference-final-standalone-CPU-v2'
FINALIZER_QUEUE=ROOT/'reports/machine-preference-final-standalone-CPU-v2-repair-gate-v1-queue.json'
SOURCE_SUITES=('code','broad','preferences','QA','AI2D','MBPP')
SUITES=(*SOURCE_SUITES,'words')
legacy.owner=owner


def configure():
    owner.REPORTS=REPORTS;owner.resources.REPORTS=REPORTS;legacy.REPORTS=REPORTS


def previous_ready(identity):
    if owner.resources.same(identity):return False
    children=PREVIOUS/'children.json'
    if children.exists() and any(owner.resources.same(r) for r in json.loads(children.read_text())):return False
    if (PREVIOUS/'failure.json').exists() or any(not (PREVIOUS/n).is_file() for n in ['plan.json','complete.json','restored.json']):
        raise ValueError('The exact complete machine comparison and all-worker restoration are required')
    plan=json.loads((PREVIOUS/'plan.json').read_text());done=json.loads((PREVIOUS/'complete.json').read_text())
    if (plan.get('owner')!=identity or plan.get('original_machine_pairs')!=1620 or plan.get('planned_steps')!=102
            or plan.get('arms')!=['chosen_CE','machine_DPO'] or plan.get('human_preferences') is not False
            or done.get('all1620original_machine_pairs_once_each_arm') is not True
            or done.get('full_original_QA16961_AI2D3088times3_MBPP600_and_human4642_each_endpoint_complete') is not True
            or done.get('matched_initial_and_both_training_arms_assessed') is not True
            or done.get('original_nine_endpoint_pipeline_preserved') is not True
            or json.loads((PREVIOUS/'restored.json').read_text()).get('all_registered_point_actors_terminal') is not True):
        raise ValueError('Every complete original machine source epoch and assessment endpoint must finish successfully')
    for arm in ['initial','chosen_CE','machine_DPO']:
        for name,count in [('full-QA.json',16961),('full-AI2D.json',3088),('full-original-MBPP.json',600),('held-preferences.json',4642)]:
            path=PREVIOUS/arm/name
            if not path.is_file() or len(json.loads(path.read_text()).get('results',[]))!=count:
                raise ValueError('All complete original question/program/preference populations are required before standalone assessment')
        for suite in SOURCE_SUITES:
            for rank in range(4):
                row=json.loads((PREVIOUS/arm/f'assessment-{suite}-{rank}-complete.json').read_text())
                if (row.get('physical_GPU')!=rank or row.get('all_loaded_values_unchanged') is not True
                        or row.get('model_updates')!=0 or row.get('actual_counts',{}).get('actual_unique')!=999706442):
                    raise ValueError('All original actor assessment receipts must retain actual counts and unchanged values')
    return True


def finalizer_ready(identity):
    if owner.resources.same(identity):return False
    if (FINALIZER/'failure.json').exists() or any(not (FINALIZER/n).is_file() for n in ['plan.json','complete.json']):
        raise ValueError('Exact standalone finalizer must finish successfully')
    plan=json.loads((FINALIZER/'plan.json').read_text());done=json.loads((FINALIZER/'complete.json').read_text())
    if (plan.get('owner')!=identity or done.get('complete') is not True or done.get('completed_updates')!=102
            or done.get('actual_counts',{}).get('total_unique')!=984399690
            or done.get('all_actual_exported_parameters_and_complete_native_probes_cold_load_bit_exact') is not True
            or done.get('real_word_logits_exact_through_fold_and_cold_native') is not True
            or done.get('old_checkpoint_actor_optimizer_or_reference_not_needed_for_inference') is not True
            or done.get('fixed_machine_DPO_export_not_quality_winner') is not True
            or done.get('all_original_machine_source_updates_complete') is not True):
        raise ValueError('Every complete fixed machine export, real text/photo/word cold probe and actual sub1B count is required')
    return True


def load_student(plan,device='cpu'):
    from slm_lab.own_dense_standalone_v1 import load
    from slm_lab.dense_foundation_inference_v1 import DenseNativeView
    spec=plan['standalone'];receipt_path=Path(spec['receipt'])
    if file_hash(receipt_path)!=spec['receipt_sha256']:raise ValueError('Exact fixed standalone export receipt required')
    receipt=json.loads(receipt_path.read_text())
    if (receipt.get('complete') is not True or receipt.get('checkpoint_sha256')!=spec['checkpoint_sha256']
            or receipt.get('completed_updates')!=102
            or receipt.get('all_actual_exported_parameters_and_complete_native_probes_cold_load_bit_exact') is not True):
        raise ValueError('The actual complete source-bound standalone export must match before inference')
    model,metadata=load(spec['checkpoint'],expected_checkpoint_sha256=spec['checkpoint_sha256'],expected_sources=receipt['standalone_sources'])
    if (metadata['completed_updates']!=102 or metadata['actual_counts']['total_unique']!=plan['expected_student_values']
            or model.counts()['total_unique']!=plan['expected_student_values'] or not 0<plan['expected_student_values']<10**9
            or metadata['tokenizer_sha256']!=plan['tokenizer_sha256']):
        raise ValueError('Every actual standalone value, original own tokenizer and complete102update cursor must match')
    native=DenseNativeView(model.to(device)).eval()
    if sum(p.numel() for p in native.parameters())!=plan['expected_student_values'] or file_hash(receipt_path)!=spec['receipt_sha256']:
        raise ValueError('Every loaded standalone parameter and exact completion receipt must remain intact')
    return model,native,metadata


def worker(suite,gpu):
    plan=legacy.registered('standalone-'+suite+'-'+str(gpu),gpu)
    import torch
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85,0)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name(0)!='NVIDIA L40':raise ValueError('Only an original isolated physical L40 is admitted')
    guard=FileBindings(plan['source_files_sha256']);model,native,metadata=load_student(plan,'cuda:0')
    checkpoint=Path(plan['standalone']['checkpoint']);folder=REPORTS/'standalone';folder.mkdir(exist_ok=True)
    signature=[(id(p),p._version,p.data_ptr()) for p in native.parameters()]
    if suite=='preferences':
        from slm_lab.standalone_held_human_v1 import CachedHeldReference,evaluate_shard
        from slm_lab.whole_human_data_v2 import WholeHumanData
        data=WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','valid','evaluation')
        spec=plan['cached_own_reference'];cache=CachedHeldReference(data,spec['report'],
            expected_sha256=spec['report_sha256'],expected_profile=spec['profile'])
        provenance=dict(checkpoint_sha256=file_hash(checkpoint),run_plan_sha256=file_hash(REPORTS/'plan.json'),
            human_manifest_sha256=spec['profile']['human_manifest_sha256'],frozen_dense_sha256=spec['profile']['frozen_dense_sha256'],
            inference_device_type='cuda',inference_precision='float32')
        evaluate_shard(model,data,folder/f'held-preferences-shard-{gpu}.json',cache,rank=gpu,world=4,
            provenance=provenance,beta=spec['profile']['DPO_beta'],checked=lambda:owner.owned(plan))
    elif suite=='words':
        from tokenizers import Tokenizer
        from slm_lab.shared_word_data_v1 import SharedWordData
        from slm_lab.standalone_word_shards_v1 import evaluate_shard
        data=SharedWordData(ROOT/'data/shared-byte-words-v1','valid','evaluation')
        tokenizer=Tokenizer.from_file(str(checkpoint.parent/'tokenizer.json'))
        if len(data)!=3429 or data.manifest['tokenizer_sha256']!=metadata['tokenizer_sha256']:
            raise ValueError('Every original3429held crop and exact own tokenizer is required')
        with inference_cache():
            evaluate_shard(model,data,tokenizer,folder/f'words-shard-{gpu}.json',rank=gpu,world=4,batch_size=16,generation_budget=64,
                amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16),checked=lambda:owner.owned(plan),
                provenance=dict(checkpoint_sha256=file_hash(checkpoint),run_plan_sha256=file_hash(REPORTS/'plan.json'),
                    word_manifest_sha256=file_hash(data.root/'manifest.json'),inference_device_type='cuda',inference_precision='bf16_autocast'))
    elif suite in ['QA','AI2D','MBPP']:
        if suite=='QA':
            from slm_lab.full_held_QA_native_v1 import FullHeldQA,evaluate_shard
            population=FullHeldQA(ROOT)
        elif suite=='AI2D':
            from slm_lab.ai2d_full_native_evaluation_v1 import FullAI2D,evaluate_shard
            population=FullAI2D(ROOT)
        else:
            from slm_lab.MBPP_original_population_v2 import FullOriginalMBPP
            from slm_lab.MBPP_all_native_evaluation_v1 import evaluate_shard
            population=FullOriginalMBPP(ROOT)
        if metadata['tokenizer_sha256']!=population.tokenizer_sha256:raise ValueError('The full original assessment population and own tokenizer must match')
        journal=folder/f'{suite}-shard-{gpu}-actual-deliveries.jsonl'
        with journal.open('x') as stream:
            count=0
            def record(row):
                nonlocal count
                stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n');stream.flush();count+=1
                if count%25==0:atomic_json(folder/f'{suite}-shard-{gpu}-progress.json',dict(actual_original_questions=count,model_updates=0,goal_achieved=False))
            with inference_cache():
                args=(native,population,folder/f'{suite}-shard-{gpu}.json') if suite=='MBPP' else (
                    native,population.tokenizer,population,folder/f'{suite}-shard-{gpu}.json')
                report=evaluate_shard(*args,rank=gpu,world=4,checkpoint_sha256=file_hash(checkpoint),checked=lambda:owner.owned(plan),record=record)
            os.fsync(stream.fileno())
        atomic_json(folder/f'{suite}-shard-{gpu}-complete.json',dict(physical_GPU=gpu,actual_original_questions=len(report['results']),
            actual_delivery_journal_sha256=file_hash(journal),model_updates=0,goal_achieved=False))
    else:
        def evaluate(module,name,data=None,**kwargs):
            original=module.load_checkpoint;module.load_checkpoint=lambda *a,**kw:(native,metadata)
            try:
                owner.owned(plan)
                if data is None:module.evaluate(checkpoint,folder/name,device='cuda:0',**kwargs)
                else:module.evaluate(checkpoint,data,folder/name,device='cuda:0',**kwargs)
            finally:module.load_checkpoint=original
        original_threads=torch.set_num_threads;torch.set_num_threads=lambda count:None
        try:
            with inference_cache():
                if suite=='code':
                    from slm_lab.native_code_shards_v1 import evaluate_shard
                    from slm_lab.native_greedy_observer_v1 import observe
                    for frame in ['completion','instruction']:
                        journal=folder/f'code-{frame}-shard-{gpu}-actual-actions.jsonl'
                        with journal.open('x') as stream:
                            def record(row):stream.write(json.dumps(row)+'\n');stream.flush()
                            with observe(native,record) as delivered:
                                evaluate_shard(native,metadata,checkpoint,folder,rank=gpu,world=4,frame=frame)
                            if len(delivered)!=6:raise ValueError('Every original assigned Python/JavaScript case must retain its actual token actions')
                            os.fsync(stream.fileno())
                elif gpu==0:
                    from slm_lab import knowledge_evaluation
                    evaluate(knowledge_evaluation,'knowledge.json',Path(plan['pulse_sources']['knowledge']),max_examples=56,max_new_tokens=32)
                elif gpu==1:
                    from slm_lab import math_evaluation
                    evaluate(math_evaluation,'math.json',ROOT/'data/math-reasoning-v2/test.jsonl',max_examples=16,max_new_tokens=256)
                elif gpu==2:
                    from slm_lab import ocr_evaluation,object_evaluation
                    evaluate(ocr_evaluation,'ocr.json',Path(plan['pulse_sources']['ocr']),max_examples=32,max_new_tokens=128,image_size=448)
                    evaluate(object_evaluation,'objects.json',Path(plan['pulse_sources']['objects']),max_examples=32,max_new_tokens=32)
                else:
                    from slm_lab import vision_evaluation
                    evaluate(vision_evaluation,'vision.json',Path(plan['pulse_sources']['vision']),max_examples=32,image_size=448)
        finally:torch.set_num_threads=original_threads
    if signature!=[(id(p),p._version,p.data_ptr()) for p in native.parameters()] or any(p.grad is not None for p in native.parameters()):
        raise ValueError('Every actual standalone model value and clean gradient must remain unchanged')
    guard.check(full=True);owner.owned(plan)
    atomic_json(folder/f'assessment-{suite}-{gpu}-complete.json',dict(physical_GPU=gpu,actual_student_values=sum(p.numel() for p in native.parameters()),
        all_loaded_values_unchanged=True,only_standalone_weights_and_tokenizer_for_neural_inference=True,
        checkpoint_sha256=file_hash(checkpoint),model_updates=0,goal_achieved=False))


def reconcile_suite(suite,plan):
    folder=REPORTS/'standalone'
    if suite=='code':
        from slm_lab.native_code_shards_v1 import reconcile
        for frame in ['completion','instruction']:reconcile([folder/f'code-{frame}-shard-{r}.json' for r in range(4)],folder/f'code-{frame}.json')
    elif suite=='preferences':
        from slm_lab.standalone_held_human_v1 import CachedHeldReference,reconcile
        from slm_lab.whole_human_data_v2 import WholeHumanData
        data=WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','valid','evaluation');spec=plan['cached_own_reference']
        cache=CachedHeldReference(data,spec['report'],expected_sha256=spec['report_sha256'],expected_profile=spec['profile'])
        reconcile(data,[folder/f'held-preferences-shard-{r}.json' for r in range(4)],folder/'held-preferences.json',cache)
    elif suite=='words':
        from tokenizers import Tokenizer
        from slm_lab.shared_word_data_v1 import SharedWordData
        from slm_lab.standalone_word_shards_v1 import reconcile
        data=SharedWordData(ROOT/'data/shared-byte-words-v1','valid','evaluation')
        tokenizer=Tokenizer.from_file(str(Path(plan['standalone']['checkpoint']).parent/'tokenizer.json'))
        reconcile(data,tokenizer,[folder/f'words-shard-{r}.json' for r in range(4)],folder/'all-held-words.json')
    elif suite=='QA':
        from slm_lab.full_held_QA_native_v1 import FullHeldQA,reconcile
        reconcile(FullHeldQA(ROOT),[folder/f'QA-shard-{r}.json' for r in range(4)],folder/'full-QA.json')
    elif suite=='AI2D':
        from slm_lab.ai2d_full_native_evaluation_v1 import FullAI2D,reconcile
        reconcile(FullAI2D(ROOT),[folder/f'AI2D-shard-{r}.json' for r in range(4)],folder/'full-AI2D.json')
    elif suite=='MBPP':
        from slm_lab.MBPP_original_population_v2 import FullOriginalMBPP
        from slm_lab.MBPP_all_native_evaluation_v1 import reconcile
        reconcile(FullOriginalMBPP(ROOT),[folder/f'MBPP-shard-{r}.json' for r in range(4)],folder/'full-original-MBPP.json')
    elif suite!='broad':raise ValueError('Only the fixed complete assessment suites are allowed')


def compare_completed():
    from assess_full_held_QA_pipeline_v5 import paired as paired_QA
    from assess_full_AI2D_pipeline_v5 import paired as paired_AI2D
    read=lambda p:json.loads(p.read_text())
    a=PREVIOUS/'machine_DPO';b=REPORTS/'standalone'
    QA=paired_QA(read(a/'full-QA.json'),read(b/'full-QA.json'))
    visual=paired_AI2D(read(a/'full-AI2D.json'),read(b/'full-AI2D.json'))
    before,after=[read(p/'full-original-MBPP.json') for p in [a,b]]
    if any(before[k]!=after[k] for k in ['original_population_sha256','own_tokenizer_sha256','generation_budget']):
        raise ValueError('Every original code population and fixed native generation budget must match')
    gained=[];lost=[];same_actions=0
    for x,y in zip(before['results'],after['results']):
        if any(x[k]!=y[k] for k in ['original_task_id','publisher_role','original_native_prompt_ids','generation_budget','inference_device_type','inference_precision']):
            raise ValueError('Every paired original code input, task role and native runtime must match')
        p,q=bool(x['execution']['functional_passed']),bool(y['execution']['functional_passed'])
        if not p and q:gained.append(x['original_task_id'])
        if p and not q:lost.append(x['original_task_id'])
        same_actions+=x['actual_generated_ids']==y['actual_generated_ids']
    report=dict(QA=QA,visual=visual,MBPP=dict(before=before['summary'],after=after['summary'],gained_original_ids=gained,
        lost_original_ids=lost,identical_generated_action_sequences=same_actions),
        held_human_before=read(a/'held-preferences.json')['overall'],held_human_after=read(b/'held-preferences.json')['overall'],
        final_standalone_word_reading=read(b/'all-held-words.json')['overall'],
        word_reading_is_absolute_final_score_not_a_matched_before_after_gain=True,
        folding_does_not_assume_greedy_or_quality_parity=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(REPORTS/'actor-to-standalone-comparisons.json',report)


def parent():
    if '16 passed' not in (ROOT/'reports/public-pretraining-lineage-drivers-v1-final2-tests.log').read_text():
        raise ValueError('Actual complete revised public-pretraining lineage and every original benchmark endpoint must pass')

    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or REPORTS.exists() or QUEUE.exists():raise ValueError('Fresh GPU-hidden standalone assessment queue required')
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.full_held_QA_native_v1 import FullHeldQA
    from slm_lab.ai2d_full_native_evaluation_v1 import FullAI2D
    from slm_lab.MBPP_original_population_v2 import FullOriginalMBPP
    previous=json.loads(PREVIOUS_QUEUE.read_text());finalizer=json.loads(FINALIZER_QUEUE.read_text())
    me=owner.resources.identity(os.getpid());files={}
    tests={'standalone-held-human-v1-tests':4,'standalone-word-shards-v1-tests':3,
           'machine-standalone-full-v1-driver-tests':11,'terminal-waiter-gpu-lease-v2-tests':3}
    for name,count in tests.items():
        path=ROOT/'reports'/(name+'.log')
        if str(count)+' passed' not in path.read_text():raise ValueError('Actual standalone neural scoring, full-source ownership and unchanged complete dispatch tests required')
        files[str(path)]=file_hash(path)
    for population in [FullHeldQA(ROOT),FullAI2D(ROOT),FullOriginalMBPP(ROOT)]:files.update(population.bindings)
    from slm_lab.shared_word_data_v1 import SharedWordData
    words=SharedWordData(ROOT/'data/shared-byte-words-v1','valid','evaluation')
    if len(words)!=3429:raise ValueError('Every original held word crop must remain in the assessment')
    for row in words.rows:files[str(Path(row['image']))]=row['stored_image_sha256']
    for path in [words.root/'manifest.json',words.root/'valid.jsonl',words.root/'valid.tokens.u16']:files[str(path)]=file_hash(path)
    extra=[PREVIOUS_QUEUE,FINALIZER_QUEUE,*python_sources(ROOT,[Path(__file__),ROOT/'tests/test_machine_standalone_full_v1_driver.py',ROOT/'tests/test_standalone_held_human_v1.py'])]
    files.update({str(p):file_hash(p) for p in extra});bind_tests(files,ROOT);guard=FileBindings(files);expires=time.time()+23*86400
    atomic_json(QUEUE,dict(owner=me,exact_machine_owner=previous['owner'],exact_finalizer_owner=finalizer['owner'],source_files_sha256=files,
        status='waiting for complete machine comparison and actual standalone export, with all original owners left intact',
        fixed_endpoint='complete_machine_DPO_standalone',expected_student_values=984399690,suites=list(SUITES),
        full_QA_questions=16961,full_AI2D_questions=3088,AI2D_image_conditions=3,full_MBPP_problems=600,held_human_pairs=4642,
        held_word_crops=3429,word_image_conditions=['proper','blank','wrong'],native_word_generation_budget=64,
        all_original_case_populations_budgets_and_source_exposure_disclosures_preserved=True,
        old_actor_or_reference_weights_not_loaded_for_neural_inference=True,
        human_reference_scores_reused_from_matching_completed_original_assessment=True,
        expires_at_unix=expires,automatic_retry=False,model_updates=0,candidate_promoted=False,goal_achieved=False))
    while not (previous_ready(previous['owner']) and finalizer_ready(finalizer['owner'])):
        if time.time()>expires:raise ValueError('Queue observation lifetime ended without preempting original work')
        guard.check();time.sleep(5)
    source_plan=PREVIOUS/'plan.json';source=json.loads(source_plan.read_text());receipt_path=FINALIZER/'complete.json';receipt=json.loads(receipt_path.read_text())
    checkpoint=Path(receipt['checkpoint']);tokenizer=checkpoint.parent/'tokenizer.json';reference_path=PREVIOUS/'machine_DPO/held-preferences.json'
    reference=json.loads(reference_path.read_text());files.update(owner.lease_sources())
    for path in [source_plan,PREVIOUS/'complete.json',PREVIOUS/'restored.json',receipt_path,FINALIZER/'plan.json',checkpoint,tokenizer,
        reference_path,reference_path.with_suffix('.jsonl')]:files[str(path)]=file_hash(path)
    for arm in ['initial','chosen_CE','machine_DPO']:
        for name in ['full-QA.json','full-AI2D.json','full-original-MBPP.json','held-preferences.json']:
            path=PREVIOUS/arm/name;files[str(path)]=file_hash(path)
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);owner.idle_original()
        used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
        if len(used)!=4 or any(int(v)>128 for v in used):raise ValueError('All four original L40s must be idle before evaluation ownership')
        if shutil.disk_usage(ROOT).free<28*1024**3:raise ValueError('Complete assessment outputs and persistent reserve must fit')
        bind_tests(files,ROOT);guard=FileBindings(files);guard.check(full=True);REPORTS.mkdir()
        plan=dict(owner=me,waiter=owner.WAITER,terminal_waiter_receipt_sha256=owner.terminal_receipt_sha256(),source_files_sha256=files,
            standalone=dict(checkpoint=str(checkpoint),checkpoint_sha256=receipt['checkpoint_sha256'],receipt=str(receipt_path),receipt_sha256=file_hash(receipt_path)),
            cached_own_reference=dict(report=str(reference_path),report_sha256=file_hash(reference_path),profile=reference['profile']),
            expected_student_values=984399690,tokenizer_sha256=receipt['tokenizer_sha256'],pulse_sources=source['pulse_sources'],
            suites=list(SUITES),expires_at_unix=time.time()+7*86400,all_four_L40s_each_suite_in_sequence=True,
            only_standalone_weights_and_tokenizer_for_neural_inference=True,automatic_retry=False,model_updates=0,candidate_promoted=False,goal_achieved=False)
        atomic_json(REPORTS/'plan.json',plan);atomic_json(REPORTS/'children.json',[]);driver=str(Path(__file__).resolve());watcher=None
        def interrupted(signum,frame):raise KeyboardInterrupt('Owned standalone assessment interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        try:
            owner.hold_original(plan)
            watcher=subprocess.Popen(['.venv/bin/python','-u',driver,'--watchdog'],cwd=ROOT,stdout=(REPORTS/'watchdog.log').open('x'),
                stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
            for suite in SUITES:
                children=[owner.launch(plan,['-u',driver,'--worker',suite,'--gpu',str(gpu)],'standalone-'+suite+'-'+str(gpu),str(gpu)) for gpu in range(4)]
                owner.wait_children(plan,children);reconcile_suite(suite,plan)
            compare_completed();guard.check(full=True)
            atomic_json(REPORTS/'complete.json',dict(complete=True,actual_unique_student_values=984399690,
                all16961QA_3088visual_times3conditions_600MBPP_and4642held_human_pairs_complete=True,
                all_original_native_code_and_small_broad_checks_complete=True,actor_to_standalone_comparisons_complete=True,
                all3429held_word_crops_CTC_and_native_text_with_three_image_controls_complete=True,
                no_old_actor_or_reference_weights_loaded_for_neural_inference=True,
                source_exposure_and_nonleaderboard_protocol_disclosures_preserved=True,
                independent_neural_score_and_generated_program_readbacks_pending=True,
                final_high_end_capability_not_established_by_completion=True,model_updates=0,candidate_promoted=False,goal_achieved=False))
        except BaseException as error:
            atomic_json(REPORTS/'failure.json',dict(error=repr(error),automatic_retry=False,model_updates=0,goal_achieved=False));raise
        finally:
            owner.restore_owned(plan,'Standalone full assessment ended; owned workers released')
            if watcher is not None:watcher.wait(timeout=20)


if __name__=='__main__':
    configure();p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watchdog',action='store_true')
    p.add_argument('--worker',choices=SUITES);p.add_argument('--gpu',type=int,choices=range(4));args=p.parse_args()
    if args.watchdog:owner.watchdog()
    elif args.worker:worker(args.worker,args.gpu)
    else:parent()
