"""Seal greedy own-model completions before executing any held assessment cases."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import shutil

import torch
from tokenizers import Tokenizer

from slm_lab.algorithm_program_v2 import PROFILE, assess_program
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_model_assessment_v1 import generate_record, score_record, summarize, validate_question
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.recurrent_ablation import state_digest
from slm_lab.standalone_gnu_v1 import gnu_toolchain
from slm_lab.training import Encoder,load_checkpoint
from slm_lab.exclusive_inference_resources_v1 import inference_reservation,validate_active_lease

ROOT=Path(__file__).resolve().parents[1]


def evaluate(checkpoint,reports,device='cuda:3',threads=2,memory_fraction=.14,expected_metadata=None,label='own_checkpoint',expected_checkpoint_sha256=None):
    checkpoint=Path(checkpoint).resolve()
    reports=Path(reports).resolve()
    if not checkpoint.is_relative_to(ROOT/'checkpoints') or not reports.is_relative_to(ROOT/'reports'):
        raise ValueError('Own local checkpoint and fresh workspace report directory required')
    if reports.exists():raise ValueError('Preserve earlier generations/results; use a fresh report directory')
    if type(threads) is not int or threads<1 or not 0<memory_fraction<=.25:
        raise ValueError('Explicit bounded CPU/CUDA evaluation resources required')
    occupied=inference_reservation(ROOT)
    remaining_writer_reserve=occupied['minimum_reserve_bytes']
    if shutil.disk_usage(ROOT).free<remaining_writer_reserve+64*1024**2:
        raise ValueError('Reserve remaining writer capacity and the full64MiB assessment budget')
    if str(device).startswith('cuda'):
        dev=torch.device(device)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError('Explicit BF16-capable CUDA device required')
        torch.cuda.set_device(dev)
        if torch.cuda.mem_get_info(dev)[0]<12*1024**3:
            raise ValueError('Keep at least12GiB initially free on the shared evaluation device')
        torch.cuda.set_per_process_memory_fraction(memory_fraction,dev)
    torch.set_num_threads(threads)
    reports.mkdir(parents=True)
    maximum_report_bytes=64*1024**2

    def storage_guard():
        validate_active_lease(ROOT)
        if (sum(p.stat().st_size for p in reports.iterdir() if p.is_file())>maximum_report_bytes
            or shutil.disk_usage(ROOT).free<512*1024**2):
            raise ValueError('Reserved assessment report budget exceeded')

    def status(message,**fields):
        atomic_json(reports/'status.json',{'status':message,'updated_at':datetime.now(timezone.utc).isoformat(),
            'optimizer_updates':0,'candidate_promoted':False,'goal_achieved':False,**fields})

    try:
        data=ROOT/'data/codenet-algorithm-assessment-v1'
        manifest_path=data/'manifest.json';manifest=json.loads(manifest_path.read_text())
        ref_folder=ROOT/'reports/codenet-algorithm-reference-readiness-v2'
        ref_path=ref_folder/'summary.json';reference=json.loads(ref_path.read_text())
        ref_plan=json.loads((ref_folder/'plan.json').read_text())
        if (reference['assessment_manifest_sha256']!=file_hash(manifest_path)
            or not reference['source_reference_coverage_complete'] or ref_plan['runtime_profile']!=PROFILE
            or not manifest['test_only'] or manifest['training_or_preference_use']):
            raise ValueError('Exact predeclared held-case dataset and independently checked runtime required')
        token_path=checkpoint.parent/'tokenizer.json'
        paths=[Path(__file__),ROOT/'src/slm_lab/exclusive_inference_resources_v1.py',ROOT/'scripts/assess_codenet_model_v2.py',ROOT/'scripts/prepare_codenet_native_contract_pack_v2.py',
            ROOT/'src/slm_lab/codenet_native_contracts_v1.py',ROOT/'src/slm_lab/codenet_model_assessment_v1.py',
            ROOT/'src/slm_lab/algorithm_program_v2.py',ROOT/'src/slm_lab/codenet_algorithm_oracles_v2.py',
            ROOT/'src/slm_lab/codenet_algorithm_assessment_v1.py',ROOT/'src/slm_lab/codenet_sample_oracles_v1.py',
            ROOT/'src/slm_lab/codenet_checkpoint_v1.py',ROOT/'src/slm_lab/training.py',ROOT/'src/slm_lab/model.py',
            ROOT/'src/slm_lab/corpus.py',ROOT/'src/slm_lab/diverse_images.py',ROOT/'src/slm_lab/evaluation_provenance.py',
            ROOT/'src/slm_lab/recurrent_ablation.py',ROOT/'src/slm_lab/standalone_gnu_v1.py',
            ROOT/'src/slm_lab/standalone_numpy_v1.py',ROOT/'src/slm_lab/standalone_program_v1.py',
            ROOT/'src/slm_lab/native_code.py',ROOT/'scripts/native_sandbox.py',ROOT/'scripts/python_program_sandbox_v1.py',
            ROOT/'.tools/gnu-toolchain-v1.json']
        runtime=capture(paths)
        bound={str(p):file_hash(p) for p in paths+[checkpoint,token_path,manifest_path,ref_path,ref_folder/'plan.json']}
        if expected_checkpoint_sha256 is not None and bound[str(checkpoint)]!=expected_checkpoint_sha256:
            raise ValueError('Exact intended own checkpoint bytes differ from launch binding')
        bound.update({str(data/name):digest for name,digest in manifest['files_sha256'].items()})
        guard=FileBindings(bound)
        if file_hash(token_path)!=manifest['tokenizer_sha256']:
            raise ValueError('Own trained tokenizer must match the whole-prompt assessment budget')
        questions=list(map(json.loads,(data/'questions.jsonl').open()))
        if (len(questions)!=manifest['language_questions'] or len({r['id'] for r in questions})!=len(questions)
            or {r['problem_id'] for r in questions}!=set(manifest['task_ids'])):
            raise ValueError('Every fixed held language-question must be generated and scored')
        for row in questions:validate_question(row)
        status('loading exact own model for complete held-question generation',questions=len(questions),device=str(device))
        model,metadata=load_checkpoint(checkpoint,device,training_state=False)
        if metadata['tokenizer_sha256']!=bound[str(token_path)] or sum(p.numel() for p in model.parameters())>=10**9:
            raise ValueError('Exact own sub1B model/tokenizer required')
        if expected_metadata and any(metadata.get(key)!=value for key,value in expected_metadata.items()):
            raise ValueError('Own checkpoint differs from expected stage/completed-update/planned-budget metadata')
        model.eval()
        before=state_digest(model.state_dict())
        tokenizer=Tokenizer.from_file(str(token_path))
        encoder=Encoder(tokenizer,model.config,data,image_size=448)
        budget=manifest['generation_budget']
        atomic_json(reports/'plan.json',{'created_at':datetime.now(timezone.utc).isoformat(),'label':label,
            'checkpoint':str(checkpoint),'checkpoint_sha256':bound[str(checkpoint)],'checkpoint_stage':metadata.get('stage'),
            'checkpoint_steps':metadata.get('steps'),'checkpoint_planned_steps':metadata.get('planned_steps'),
            'storage_reservation':{'occupied_verified_writer':occupied,'remaining_writer_reserve_bytes':remaining_writer_reserve},
            'checkpoint_data_sha256':metadata.get('data_sha256'),'checkpoint_world_size':metadata.get('world_size'),
            'own_model_parameters':sum(p.numel() for p in model.parameters()),
            'source_files_sha256':bound,'runtime':runtime,'questions':len(questions),'tasks':manifest['tasks'],
            'sealed_components':manifest['independent_sealed_components'],'generation_budget':budget,
            'inference_device':str(device),'inference_precision':'BF16' if str(device).startswith('cuda') else 'FP32',
            'temperature':0.,'num_samples_per_question':1,'complete_prompts_required':True,
            'primary_program':'Raw model output, removing only its actual final EOS token; no fence stripping, source repairs, reference hints or feedback loops.',
            'model_input_policy':'Only complete public question prompts/target language reach the model. Original public examples remain in the unchanged statements; reference programs, authored cases and scoring feedback are excluded.',
            'runtime_profile':PROFILE,'optimizer_updates':0,'candidate_promoted':False,'goal_achieved':False})
        def amp():
            return torch.autocast(device_type='cuda',dtype=torch.bfloat16) if str(device).startswith('cuda') else nullcontext()
        generated=[]
        generations_path=reports/'generations.jsonl'
        with generations_path.open('x') as stream:
            for question in questions:
                guard.check();validate_active_lease(ROOT)
                record=generate_record(model,tokenizer,encoder,question,budget,device,amp)
                generated.append(record)
                serialized=json.dumps(record,ensure_ascii=False)+'\n'
                if stream.tell()+len(serialized.encode())>maximum_report_bytes-4*1024**2:
                    raise ValueError('Generated outputs exceed the reserved report capacity')
                stream.write(serialized);stream.flush();storage_guard()
                status('generating complete fixed held-question population',generated=len(generated),questions=len(questions),
                       question=question['id'],last_stop_reason=record['stop_reason'])
                print('generated_held_program',len(generated),question['id'],record['stop_reason'],flush=True)
            os.fsync(stream.fileno())
        after=state_digest(model.state_dict())
        if before!=after:raise ValueError('Own model state changed during evaluation')
        guard.check(full=True)
        if capture(paths)!=runtime:raise ValueError('Generation implementation/runtime changed')
        atomic_json(reports/'generation-seal.json',{'completed_at':datetime.now(timezone.utc).isoformat(),
            'generations_sha256':file_hash(generations_path),'checkpoint_sha256':bound[str(checkpoint)],
            'assessment_manifest_sha256':bound[str(manifest_path)],'questions_sha256':manifest['files_sha256']['questions.jsonl'],
            'questions':len(generated),'generation_budget':budget,'before_model_state_sha256':before,
            'after_model_state_sha256':after,'runtime':runtime,'test_feedback_seen_by_model':False,
            'optimizer_updates':0,'goal_achieved':False})
        # Release the model before parsing case values or running generated programs.
        del model,encoder;gc.collect()
        if str(device).startswith('cuda'):torch.cuda.empty_cache()
        if file_hash(generations_path)!=json.loads((reports/'generation-seal.json').read_text())['generations_sha256']:
            raise ValueError('Sealed generated outputs changed')
        generated_guard=FileBindings({str(generations_path):file_hash(generations_path),
            str(reports/'generation-seal.json'):file_hash(reports/'generation-seal.json')})
        saved=list(map(json.loads,generations_path.open()))
        cases=defaultdict(list)
        for row in map(json.loads,(data/'cases.jsonl').open()):cases[row['problem_id']].append(row)
        if sum(len(c) for c in cases.values())!=sum(manifest['case_counts'].values()):
            raise ValueError('Complete original scoring-case population required')
        tools=gnu_toolchain()
        numpy_guard=FileBindings(ref_plan['numpy_runtime_files_sha256'])
        results=[]
        with (reports/'execution.jsonl').open('x') as stream:
            for question,generation in zip(questions,saved):
                guard.check();validate_active_lease(ROOT)
                generated_guard.check()
                if shutil.disk_usage('/tmp').free<12*1024**3+256*1024**2:
                    raise ValueError('Preserve compiler/runtime temporary workspace reserve')
                result=score_record(question,generation,cases[question['problem_id']],tools,assess_program)
                results.append(result)
                stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush()
                storage_guard()
                status('scoring sealed raw model programs under the checked profile',scored=len(results),questions=len(questions))
                print('scored_held_program',question['id'],result['functional_passed'],result['bounded_stress_passed'],flush=True)
        summary=summarize(questions,saved,results)
        guard.check(full=True)
        generated_guard.check(full=True)
        if capture(paths)!=runtime:raise ValueError('Assessment implementation/runtime changed')
        gnu_toolchain()
        numpy_guard.check(full=True)
        report={**summary,'completed_at':datetime.now(timezone.utc).isoformat(),'label':label,
            'checkpoint':str(checkpoint),'checkpoint_sha256':bound[str(checkpoint)],'tokenizer_sha256':bound[str(token_path)],
            'assessment_manifest_sha256':bound[str(manifest_path)],'generation_budget':budget,
            'generation_seal_sha256':file_hash(reports/'generation-seal.json'),'generations_sha256':file_hash(generations_path),
            'execution_records_sha256':file_hash(reports/'execution.jsonl'),'runtime':runtime,
            'inference_device':str(device),'runtime_profile':PROFILE,'all_model_state_bytes_unchanged':True,
            'shared_with_live_training':False,'optimizer_updates':0,
            'automatic_candidate_promotion':False,'goal_achieved':False,
            'scope':'One greedy raw program per predeclared language-question. Targeted14-task/13-component assessment, not universal code capability, broad independent benchmarks, original hidden tests or certified absence of prior-corpus exposure. Local stress completion is a bounded profile result, not asymptotic efficiency proof or cross-language speed ranking.'}
        atomic_json(reports/'results.json',report)
        status('complete held-model assessment finished; broader capability unverified',summary=summary)
        print(json.dumps(report),flush=True)
        return report
    except Exception as error:
        status('held-model assessment stopped; preserve partial generations and inspect',error=str(error))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--reports',required=True)
    parser.add_argument('--device',default='cuda:3')
    parser.add_argument('--expected-stage')
    parser.add_argument('--expected-steps',type=int)
    parser.add_argument('--expected-planned-steps',type=int)
    parser.add_argument('--expected-data-sha256')
    parser.add_argument('--label',default='own_checkpoint')
    parser.add_argument('--expected-checkpoint-sha256')
    args=parser.parse_args()
    if Path(args.reports).name!=args.reports:raise ValueError('Fresh simple report name required')
    expected={}
    if args.expected_stage is not None:expected['stage']=args.expected_stage
    if args.expected_steps is not None:expected['steps']=args.expected_steps
    if args.expected_planned_steps is not None:expected['planned_steps']=args.expected_planned_steps
    if args.expected_data_sha256 is not None:expected['data_sha256']=args.expected_data_sha256
    evaluate(args.checkpoint,ROOT/'reports'/args.reports,args.device,expected_metadata=expected,label=args.label,
             expected_checkpoint_sha256=args.expected_checkpoint_sha256)


if __name__=='__main__':main()
