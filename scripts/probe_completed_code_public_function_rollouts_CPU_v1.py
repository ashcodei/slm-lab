"""Own complete-model rollouts on all14short admitted source-train functions.

No optimization, source-valid functions, external teacher or benchmark score.
Original public example answers are visible in the unchanged model prompt.
"""
from datetime import datetime,timezone
import gzip
import json
import os
from pathlib import Path
import traceback

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.public_function_rollout_observation_v1 import observe as execute

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/completed-code-public-function-rollouts-CPU-v1'
SOURCE=ROOT/'reports/code-first-continuation-v3'
CHECKPOINT=ROOT/'checkpoints/code-first-continuation-v3/standalone/model.pt'
DATA=ROOT/'data/direct-algorithm-function-contracts-admitted-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh GPU-hidden source-train-only rollout diagnostic required')
    tests=ROOT/'reports/public-function-rollout-observation-v1-tests.log'
    observer_tests=ROOT/'reports/native-greedy-observer-v1-final-tests.log'
    if '2 passed' not in tests.read_text() or '2 passed' not in observer_tests.read_text():
        raise ValueError('Actual finite-case runtime and transparent native-token observation tests must pass')
    receipt_path=ROOT/'reports/direct-algorithm-function-contracts-v1-readback-admission/complete.json'
    receipt=json.loads(receipt_path.read_text());admission=json.loads((DATA/'admission.json').read_text())
    if (not receipt.get('dataset_admitted_to_function_CE_training') or not admission.get('admitted_to_training')
            or admission.get('execution_reward_training_admitted') is not False or admission.get('human_preference_training_admitted') is not False):
        raise ValueError('Source CE admission and nonadmission of new execution/human training must remain explicit')
    pack=CompleteCodePack(DATA,'train','audit')
    rows=list(map(json.loads,gzip.open(DATA/'original-rows.jsonl.gz','rt')));selected=[]
    for position,index in enumerate(pack.positions):
        row=rows[int(index)];ids,labels=pack.sequence(position);start=next(i for i,v in enumerate(labels) if v!=-100)
        if len(ids)-start<=128 and start<=1024:
            if row['split']!='train' or row['source_candidate']['role']!='train':raise ValueError('Only sealed source-train records are eligible')
            selected.append(dict(position=position,global_native_index=int(index),row=row,prompt_ids=ids[:start],complete_gold_ids=ids[start:],original_prefix_length=start))
    if len(pack)!=826 or len(selected)!=14:raise ValueError('Every actual qualifying source-train function in the declared short-response population must remain')
    source_plan=SOURCE/'plan.json';source=json.loads(source_plan.read_text());trained=json.loads((SOURCE/'training-complete.json').read_text())
    if trained['actual_updates']!=7558 or source['planned_steps']!=7558:raise ValueError('Only the actual complete original code epoch is eligible')
    sources=dict(code_first_continuation_plan_sha256=file_hash(source_plan),own_complete_main_initializer_sha256=source['initial_dense_settings']['initializer_sha256'])
    paths=[CHECKPOINT,CHECKPOINT.parent/'tokenizer.json',source_plan,SOURCE/'training-complete.json',receipt_path,DATA/'manifest.json',DATA/'admission.json',tests,observer_tests,
        *[DATA/name for name in pack.manifest['files_sha256']],*python_sources(ROOT,[Path(__file__),ROOT/'tests/test_public_function_rollout_observation_v1.py'])]
    files={str(p):file_hash(p) for p in paths};guard=FileBindings(files)
    me=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]);OUT.mkdir()
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=me,source_files_sha256=files,
        exact_complete_code_checkpoint_sha256=file_hash(CHECKPOINT),expected_standalone_sources=sources,source_role='train',
        admitted_source_train_population=826,actual_complete_qualifying_diagnostic_population=14,
        source_selection_before_any_neural_forward=dict(maximum_complete_original_response_EOS_tokens=128,maximum_original_prompt_tokens=1024,all_qualifying_functions=True),
        selected_original_train_positions=[r['position'] for r in selected],native_greedy_action_budget=256,
        source_gold_programs_only_for_separate_execution_controls_not_model_input=True,
        original_literal_docstring_public_answers_are_visible_in_model_input=True,
        final_benchmark_or_all_history_unseen_population=False,neural_optimizer_updates=0,execution_reward_or_human_preference_training_admitted=False,
        no_GPU_or_external_model_or_teacher=True,goal_achieved=False))
    atomic_json(OUT/'all14-original-source-train-records.json',selected)
    try:
        positives=[]
        for item in selected:
            controls=[execute(item['row'],item['row']['response']) for _ in range(2)]
            if any(not r['all_original_examples_passed'] for r in controls):raise ValueError('Every unchanged complete admitted reference must still pass every original public example twice')
            positives.append(dict(source_id=item['row']['id'],actual_controls=controls))
        atomic_json(OUT/'all14-original-reference-positive-controls.json',positives)
        import torch
        from tokenizers import Tokenizer
        from slm_lab.own_dense_standalone_v1 import load
        from slm_lab.public_pretraining_native_v1 import validate_start
        from slm_lab.dense_native_view_v2 import CompleteDenseNativeView
        from slm_lab.inference_rotary_cache_v1 import inference_cache
        from slm_lab.native_greedy_observer_v1 import observe
        from slm_lab.code_quality import instruction_code
        from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        raw=torch.load(CHECKPOINT,map_location='cpu',weights_only=True,mmap=True);validate_start(source,raw);del raw
        model,metadata=load(CHECKPOINT,expected_checkpoint_sha256=files[str(CHECKPOINT)],expected_sources=sources)
        if model.counts()['total_unique']!=984399690 or metadata['completed_updates']!=7558 or torch.cuda.is_initialized():
            raise ValueError('Only the actual complete own984399690-value source model on CPU may generate')
        native=CompleteDenseNativeView(model).eval();tokenizer=Tokenizer.from_file(str(DATA/'tokenizer.json'))
        frozen=tensor_digest(model.named_parameters());observations=[];generated=[]
        with (OUT/'all14-actual-native-token-actions.jsonl').open('x') as actions,(OUT/'all14-generated-programs-and-public-executions.jsonl').open('x') as journal:
            def record(event):actions.write(json.dumps(event)+'\n');actions.flush();observations.append(event)
            with inference_cache(),observe(native,record),torch.no_grad():
                for item in selected:
                    row=item['row'];prompt=torch.tensor([item['prompt_ids']],dtype=torch.long)
                    output=native.generate(prompt,max_new_tokens=256,eos_token_id=2,use_cache=True)
                    ids=output[0,len(item['prompt_ids']):].tolist();response=tokenizer.decode(ids);program=instruction_code(response)
                    actual=execute(row,program)
                    result=dict(original_source_id=row['id'],original_train_position=item['position'],function=row['source_candidate']['function'],
                        actual_generated_token_ids=ids,raw_decoded_response=response,actual_generated_program=program,actual_public_example_observations=actual,
                        generated_code_fence_extraction_only_no_program_repair=True,model_updates=0)
                    generated.append(result);journal.write(json.dumps(result,ensure_ascii=False)+'\n');journal.flush()
                    atomic_json(OUT/'progress.json',dict(actual_complete_original_source_train_rollouts=len(generated),planned_rollouts=14,
                        syntax_valid=sum(r['actual_public_example_observations']['syntax_valid'] for r in generated),
                        all_original_public_examples_passed=sum(r['actual_public_example_observations']['all_original_examples_passed'] for r in generated),
                        benchmark=False,model_updates=0,goal_achieved=False))
        if len(generated)!=14 or len(observations)!=14 or frozen!=tensor_digest(model.named_parameters()) or torch.cuda.is_initialized():
            raise ValueError('Every complete predeclared source-train rollout and all unchanged own model values are required')
        guard.check(full=True)
        result=dict(tasks=14,syntax_valid=sum(r['actual_public_example_observations']['syntax_valid'] for r in generated),
            functions_passing_every_original_public_example=sum(r['actual_public_example_observations']['all_original_examples_passed'] for r in generated),
            original_public_examples=sum(r['actual_public_example_observations']['original_public_example_count'] for r in generated),
            EOS_stopped=sum(r['actual_stopped_on_EOS'] for r in observations),budget_exhaustions=sum(r['actual_native_action_count']==256 and not r['actual_stopped_on_EOS'] for r in observations))
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,actual_source_train_diagnostic=result,
            actual_unique_student_values=model.counts()['total_unique'],all_parameters_unchanged=True,all14actual_original_prompt_and_generated_token_actions_retained=True,
            all14unchanged_original_reference_controls_passed_twice=True,every_original_public_case_retained=True,
            not_a_held_benchmark_or_all_history_independence_or_general_correctness_claim=True,
            original_public_example_answers_visible_in_model_prompt=True,execution_reward_training_admitted=False,human_preferences=False,
            source_files_sha256={**files,**{str(p):file_hash(p) for p in OUT.iterdir() if p.is_file() and p.name not in {'complete.json','progress.json'}}},
            neural_optimizer_updates=0,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_source_train_diagnostic=result,neural_updates=0)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),every_original_training_job_preserved=True,goal_achieved=False));raise


if __name__=='__main__':main()
