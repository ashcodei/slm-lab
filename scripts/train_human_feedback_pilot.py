"""Train original human-feedback reward/DPO branches, with a held-out gate for PPO."""
from datetime import datetime,timezone
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json,jsonl
from slm_lab.reward_gate import gate
from slm_lab.alignment_selection import REPORT_FILES,compare,validate_collection
from slm_lab.evaluation_provenance import capture
from slm_lab.feedback_reading_guard import enforce as enforce_reading_retention, validate_coverage as validate_reading_coverage
from slm_lab.textvqa_evaluation import compare_reports as compare_reading_reports
from slm_lab.feedback_own_context_guard import enforce as enforce_own_context_retention, validate_coverage as validate_own_context_coverage
from slm_lab.feedback_nq_guard import enforce as enforce_fact_retention, validate_coverage as validate_fact_coverage
from slm_lab.nq_input_audit import admit as admit_fact_inputs
from slm_lab.feedback_exercise_js_guard import enforce as enforce_js_retention, validate_coverage as validate_js_coverage
from slm_lab.exercise_js_runner import toolchain as verify_js_toolchain
from slm_lab.feedback_exercise_go_guard import enforce as enforce_go_retention, validate_coverage as validate_go_coverage
from slm_lab.exercise_go_runner import toolchain as verify_go_toolchain
from slm_lab.feedback_exercise_rust_guard import enforce as enforce_rust_retention, validate_coverage as validate_rust_coverage
from slm_lab.exercise_rust_runner import toolchain as verify_rust_toolchain
from slm_lab.vqa_pair_admission import admit as admit_paired_photos
from slm_lab.vqa_pair_guard import enforce as enforce_paired_photo_retention, validate_coverage as validate_paired_photo_coverage
from slm_lab.completed_state_cleanup import retire
from train_multimodal_pilot import complete_stage

ROOT=Path(__file__).resolve().parents[1]
GUARD_IMPLEMENTATION_SHA256=file_hash(ROOT/'src/slm_lab/alignment_selection.py')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preceding-status',default='reports/multimodal-pilot-v1/status.json')
    parser.add_argument('--preceding-completed-state',default='visual pilot training and evaluation completed; broad capability unverified')
    parser.add_argument('--base-checkpoint',default='checkpoints/multimodal-pilot-v1/joint-instruction-images/model.pt')
    parser.add_argument('--reports',default='reports/human-feedback-pilot-v1')
    parser.add_argument('--checkpoints',default='checkpoints/human-feedback-pilot-v1')
    parser.add_argument('--capability-guards',action='store_true',help='Compare all declared capabilities before selecting a feedback branch')
    parser.add_argument('--capability-baseline',default='reports/extended-curriculum-v1')
    parser.add_argument('--real-photo-reading-guard',action='store_true',help='Require matched128-photo native reading retention when selecting a feedback branch')
    parser.add_argument('--own-context-reading-guard',action='store_true',help='Also retain compact own-reader conditioned answers on the same128 original photos')
    parser.add_argument('--human-fact-question-guard',action='store_true',help='Also retain generated answers on all3610 original NQ development questions')
    parser.add_argument('--javascript-module-guard',action='store_true',help='Also retain full named modules on all37 original JavaScript publisher suites')
    parser.add_argument('--go-module-guard',action='store_true',help='Also retain full native Go modules on all42 original publisher suites')
    parser.add_argument('--rust-module-guard',action='store_true',help='Also retain complete native Rust modules on all24 original publisher suites')
    parser.add_argument('--paired-photo-guard',action='store_true',help='Also retain all finalized selected real-photo pairs with blank/counterpart controls')
    args=parser.parse_args()
    if args.real_photo_reading_guard and not args.capability_guards:
        raise ValueError('Real-photo reading retention requires the existing full capability guards')
    if args.own_context_reading_guard and not args.real_photo_reading_guard:
        raise ValueError('Own-context retention requires existing plain-reading and full capability guards')
    if args.human_fact_question_guard and not args.capability_guards:
        raise ValueError('Human factual QA retention requires the existing full capability guards')
    if args.javascript_module_guard and not args.capability_guards:
        raise ValueError('JavaScript module retention requires the existing full capability guards')
    if args.go_module_guard and not args.capability_guards:
        raise ValueError('Go module retention requires the existing full capability guards')
    if args.rust_module_guard and not args.capability_guards:
        raise ValueError('Rust module retention requires the existing full capability guards')
    if args.paired_photo_guard and not args.capability_guards:
        raise ValueError('Paired-photo retention requires the existing full capability guards')
    os.chdir(ROOT);reports=ROOT/args.reports;reports.mkdir(parents=True,exist_ok=True)
    reading_source_files=([ROOT/name for name in capture()['files_sha256']]+[Path(__file__),
        ROOT/'src/slm_lab/feedback_reading_guard.py',ROOT/'src/slm_lab/textvqa_evaluation.py',
        ROOT/'src/slm_lab/textvqa_data.py',ROOT/'src/slm_lab/vqa_pairs.py',
        ROOT/'src/slm_lab/vqa_pair_evaluation.py',ROOT/'src/slm_lab/clevr_evaluation.py',
        ROOT/'src/slm_lab/objectives.py']) if (args.real_photo_reading_guard or args.human_fact_question_guard or args.javascript_module_guard or args.go_module_guard or args.rust_module_guard or args.paired_photo_guard) else []
    if args.own_context_reading_guard:
        reading_source_files.extend(ROOT/'src/slm_lab'/(name+'.py') for name in
            ['feedback_own_context_guard','own_context_transfer_evaluation','own_ocr_question_admission',
             'own_ocr_training_data','own_ocr_context','own_ocr_question_evaluation','photo_word_reading',
             'word_localizer','byte_word_reader','text_localization_metrics','textvqa_admission','textvqa_input_audit'])
    if args.human_fact_question_guard:
        reading_source_files.extend(ROOT/'src/slm_lab'/(name+'.py') for name in
            ['feedback_nq_guard','nq_input_audit','nq_open_data','nq_open_evaluation','exposure_audit'])
        reading_source_files.extend([ROOT/'scripts/fetch_nq_open_source.py',ROOT/'scripts/audit_nq_exposure.py'])
    if args.javascript_module_guard:
        reading_source_files.extend(ROOT/'src/slm_lab'/(name+'.py') for name in
            ['feedback_exercise_js_guard','exercise_js_evaluation','exercise_js_runner'])
        reading_source_files.extend([ROOT/'scripts/javascript_sandbox.py',ROOT/'scripts/native_sandbox.py',
            ROOT/'scripts/verify_exercise_js_references.py',ROOT/'.tools/javascript-toolchain-v1.json',
            ROOT/'.tools/javascript-package-lock-v1.json',ROOT/'reports/exercism-javascript-publisher-references-v1.json'])
    if args.go_module_guard:
        reading_source_files.extend(ROOT/'src/slm_lab'/(name+'.py') for name in
            ['feedback_exercise_go_guard','exercise_go_evaluation','exercise_go_runner'])
        reading_source_files.extend([ROOT/'scripts/go_sandbox.py',ROOT/'scripts/native_sandbox.py',
            ROOT/'scripts/verify_exercise_go_references.py',ROOT/'.tools/go-toolchain-v1.json',
            ROOT/'reports/exercism-go-publisher-references-v1.json'])
    if args.rust_module_guard:
        reading_source_files.extend(ROOT/'src/slm_lab'/(name+'.py') for name in
            ['feedback_exercise_rust_guard','exercise_rust_evaluation_v2','exercise_rust_runner'])
        reading_source_files.extend(ROOT/name for name in ['scripts/rust_sandbox.py','scripts/javascript_sandbox.py',
            'scripts/native_sandbox.py','scripts/verify_exercise_rust_references.py','scripts/prepare_rust_test_libraries.py',
            'scripts/verify_rust_confinement.py','.tools/rust-toolchain-v1.json',
            'reports/exercism-rust-publisher-references-v2.json','reports/rust-kernel-and-report-verification-v1.json'])
    if args.paired_photo_guard:
        reading_source_files.extend(ROOT/'src/slm_lab'/(name+'.py') for name in
            ['vqa_attribution','vqa_pair_admission','vqa_pair_guard','expanded_photos','masked_visual_pretraining'])
        reading_source_files.extend([ROOT/'configs/model.json',ROOT/'scripts/inspect_vqa_sources.py'])
    reading_sources_loaded=capture(reading_source_files) if reading_source_files else None
    def verify_reading_sources():
        if reading_sources_loaded is not None and capture(reading_source_files)!=reading_sources_loaded:
            raise ValueError('Loaded feedback/real-reading implementation changed; inspect and reload an idle scheduler')
    def status(state,**fields):
        atomic_json(reports/'status.json',{'status':state,'updated_at':datetime.now(timezone.utc).isoformat(),**fields})
    preceding=ROOT/args.preceding_status
    while not preceding.exists() or json.loads(preceding.read_text()).get('status')!=args.preceding_completed_state:
        status('waiting for completed multimodal training and evaluation',
               own_context_reading_guard=args.own_context_reading_guard,
               human_fact_question_guard=args.human_fact_question_guard,
               javascript_module_guard=args.javascript_module_guard,
               go_module_guard=args.go_module_guard,
               rust_module_guard=args.rust_module_guard,
               paired_photo_guard=args.paired_photo_guard);time.sleep(30)
    verify_reading_sources()
    data=ROOT/'data/human-preference-curriculum-v1';manifest=json.loads((data/'manifest.json').read_text())
    for name,digest in manifest['files_sha256'].items():
        if file_hash(data/name)!=digest:raise ValueError('Preference data changed: '+name)
    questions=ROOT/'data/knowledge-eval-v1/questions.jsonl'
    if file_hash(questions)!=manifest['benchmark_questions_sha256']:raise ValueError('Reserved questions changed')
    base=ROOT/args.base_checkpoint
    env={**os.environ,'OMP_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
    train_count=sum(1 for _ in jsonl(data/'preferences.train.jsonl'))
    plan={'base_checkpoint':str(base),'base_checkpoint_sha256':file_hash(base),'human_feedback_manifest_sha256':file_hash(data/'manifest.json'),
          'reward_steps':math.ceil(train_count/8),'dpo_steps':math.ceil(train_count/4),
          'ppo_steps_if_reward_gate_passes':128,'ppo_batch':4,'ppo_rollout_tokens':256,
          'branch_design':'DPO and gated PPO each start from the same multimodal SFT policy. Reward learning is a separate model.',
          'capability_guards':args.capability_guards,
          'capability_baseline':args.capability_baseline if args.capability_guards else None,
          'capability_families':list(REPORT_FILES) if args.capability_guards else [],
          'capability_guard_implementation_sha256':GUARD_IMPLEMENTATION_SHA256 if args.capability_guards else None,
          'real_photo_reading_guard':args.real_photo_reading_guard,
          'own_context_reading_guard':args.own_context_reading_guard,
          'human_fact_question_guard':args.human_fact_question_guard,
          'javascript_module_guard':args.javascript_module_guard,
          'go_module_guard':args.go_module_guard,
          'rust_module_guard':args.rust_module_guard,
          'paired_photo_guard':args.paired_photo_guard,
          'real_reading_source_runtime':reading_sources_loaded,
          'completed_optimizer_retirement':'After reward ranking and each policy preference assessment; exact retained tensors verified, weights/reports preserved.',
          'goal_achieved':False,'scope':'A bounded offline-human-feedback pilot; production usefulness and broad capability remain unverified.'}
    atomic_json(reports/'plan.json',plan)
    def train_stage(name,kind,source,checkpoint,steps,batch,lr,world=4,extra=()):
        verify_reading_sources()
        directory=ROOT/args.checkpoints/name
        stage={'stage':kind,'steps':steps,'trainable':'language','image_size':448,'data_sha256':file_hash(source)}
        if not complete_stage(directory,stage):
            if directory.exists() and any(directory.iterdir()):raise ValueError('Incomplete human-feedback stage requires inspection/resume: '+name)
            if shutil.disk_usage(ROOT).free<25*1024**3:raise ValueError('Insufficient disk for human-feedback checkpoint/reference states')
            prefix=([str(ROOT/'.venv/bin/torchrun'),'--standalone','--nproc_per_node='+str(world)] if world>1 else [sys.executable])
            command=[*prefix,'-m','slm_lab.training',kind,'--checkpoint',str(checkpoint),
                '--tokenizer',str(base.parent/'tokenizer.json'),'--data',str(source),'--output',str(directory),
                '--steps',str(steps),'--batch-size',str(batch),'--learning-rate',str(lr),
                '--trainable','language','--device','cuda','--threads','1','--gradient-checkpointing',
                '--checkpoint-every','100','--keep-checkpoints','1','--log-every','10',*extra]
            status('training human-feedback pilot',stage=name,planned_steps=steps)
            subprocess.run(command,check=True,cwd=ROOT,env=env)
            if not complete_stage(directory,stage):raise ValueError('Human-feedback stage did not complete its plan')
        for path in directory.glob('step-*.pt'):path.unlink(missing_ok=True)
        return directory/'model.pt'
    def evaluate(module,checkpoint,output,extra=()):
        verify_reading_sources()
        status('evaluating completed human-feedback stage',task=module,checkpoint=str(checkpoint))
        subprocess.run([sys.executable,'-u','-m',module,'--checkpoint',str(checkpoint),'--output',str(output),
                        '--device','cuda',*extra],check=True,cwd=ROOT,env=env)
    old=ROOT/'data/alignment-curriculum-v1'
    capability_specs=[
        ('slm_lab.code_quality','code-quality.json',[]),
        ('slm_lab.knowledge_evaluation','knowledge.json',['--data',str(questions)]),
        ('slm_lab.vision_evaluation','vision.json',['--data',str(old/'contrastive.valid.jsonl')]),
        ('slm_lab.ocr_evaluation','ocr.json',['--data',str(old/'images.valid.jsonl')]),
    ]
    if args.capability_guards:
        capability_specs.extend([
            ('slm_lab.math_evaluation','math.json',['--data',str(ROOT/'data/math-reasoning-v2/test.jsonl')]),
            ('slm_lab.object_evaluation','objects.json',['--data',str(ROOT/'data/object-qa-v1/valid.jsonl')]),
            ('slm_lab.public_code_benchmark','public-code.json',['--data',str(ROOT/'data/public-code-assessment-v1/questions.jsonl')]),
            ('slm_lab.cpp_evaluation','cpp-code.json',[]),
        ])
    capability_data={
        'knowledge':(questions,'questions_sha256'),
        'math':(ROOT/'data/math-reasoning-v2/test.jsonl','data_sha256'),
        'vision':(old/'contrastive.valid.jsonl','data_sha256'),
        'ocr':(old/'images.valid.jsonl','data_sha256'),
        'objects':(ROOT/'data/object-qa-v1/valid.jsonl','data_sha256'),
        'public_code':(ROOT/'data/public-code-assessment-v1/questions.jsonl','data_sha256'),
    }
    def read_capabilities(directory):
        return {name:json.loads((directory/file).read_text()) for name,file in REPORT_FILES.items()}
    try:
        baseline=None
        if args.capability_guards:
            if file_hash(ROOT/'src/slm_lab/alignment_selection.py')!=GUARD_IMPLEMENTATION_SHA256:raise ValueError('Loaded capability guard changed while this scheduler waited; reload the waiting scheduler')
            base_sha=file_hash(base);directory=ROOT/args.capability_baseline
            # Reuse the completed extended-policy assessment only when it belongs
            # to this exact model and current evaluator. Otherwise produce fresh
            # evidence in a separate base directory before feedback training.
            try:
                baseline=read_capabilities(directory)
                if (validate_collection(baseline,list(REPORT_FILES))!=base_sha or baseline['code']['evaluation_provenance']!=capture()
                    or any(baseline[name][field]!=file_hash(path) for name,(path,field) in capability_data.items())):baseline=None
            except (FileNotFoundError,KeyError,ValueError,json.JSONDecodeError):baseline=None
            if baseline is None:
                directory=reports/'base';directory.mkdir(exist_ok=True)
                for module,name,extra in capability_specs:evaluate(module,base,directory/name,extra)
                baseline=read_capabilities(directory)
                if validate_collection(baseline,list(REPORT_FILES))!=base_sha:raise ValueError('Base capability evidence differs from the pre-feedback policy')
            atomic_json(reports/'capability-baseline.json',{'checkpoint':str(base),'checkpoint_sha256':base_sha,
                        'directory':str(directory),'reports_sha256':{name:file_hash(directory/file) for name,file in REPORT_FILES.items()},
                        'scope':'Pre-feedback local tuning assessment; not high-end capability proof.'})
        reading_baseline=None
        reading_data=ROOT/'data/textvqa-human-reading-v1'
        if args.real_photo_reading_guard:
            verify_reading_sources()
            reading_manifest=json.loads((reading_data/'manifest.json').read_text())
            reading_digest=file_hash(reading_data/'valid.jsonl')
            if reading_digest!=reading_manifest['files_sha256']['valid.jsonl']:
                raise ValueError('Finalized real-reading validation records changed')
            reading_path=ROOT/args.capability_baseline/'textvqa.json'
            try:
                reading_baseline=json.loads(reading_path.read_text())
                validate_reading_coverage(reading_baseline)
                reading_files=[ROOT/name for name in reading_baseline['evaluation_provenance']['files_sha256']]
                if (reading_baseline['checkpoint_sha256']!=base_sha or reading_baseline['data_sha256']!=reading_digest
                        or reading_baseline['manifest_sha256']!=file_hash(reading_data/'manifest.json')
                        or capture(reading_files)!=reading_baseline['evaluation_provenance']):
                    reading_baseline=None
                elif compare_reading_reports(reading_baseline,reading_baseline)['passed'] is not True:
                    reading_baseline=None
            except (FileNotFoundError,KeyError,ValueError,json.JSONDecodeError):
                reading_baseline=None
            if reading_baseline is None:
                reading_path=reports/'base-textvqa.json'
                evaluate('slm_lab.textvqa_evaluation',base,reading_path,['--data',str(reading_data)])
                reading_baseline=json.loads(reading_path.read_text())
            validate_reading_coverage(reading_baseline)
            if reading_baseline['checkpoint_sha256']!=base_sha:
                raise ValueError('Native real-reading baseline belongs to another own policy')
            atomic_json(reports/'textvqa-baseline-receipt.json',{'checkpoint_sha256':base_sha,
                'report':str(reading_path),'report_sha256':file_hash(reading_path),'data_sha256':reading_digest,
                'manifest_sha256':file_hash(reading_data/'manifest.json'),'goal_achieved':False,
                'scope':'Selected own policy/native real-reading baseline before any feedback training.'})
        own_context_baseline=None
        if args.own_context_reading_guard:
            verify_reading_sources()
            context_manifest_path=ROOT/'data/own-ocr-human-questions-v1/manifest.json'
            context_manifest=json.loads(context_manifest_path.read_text())
            context_path=ROOT/args.capability_baseline/'own-context.json'
            try:
                own_context_baseline=json.loads(context_path.read_text())
                validate_own_context_coverage(own_context_baseline)
                context_files=[ROOT/name for name in own_context_baseline['evaluation_provenance']['files_sha256']]
                if (own_context_baseline['checkpoint_sha256']!=base_sha
                        or own_context_baseline['data_sha256']!=reading_digest
                        or own_context_baseline['manifest_sha256']!=file_hash(reading_data/'manifest.json')
                        or own_context_baseline['own_context_manifest_sha256']!=file_hash(context_manifest_path)
                        or own_context_baseline['detector_sha256']!=context_manifest['own_component_sha256'][str(ROOT/'checkpoints/word-localizer-v1/model.pt')]
                        or own_context_baseline['reader_sha256']!=context_manifest['own_component_sha256'][str(ROOT/'checkpoints/native-word-refinement-v1/model.pt')]
                        or capture(context_files)!=own_context_baseline['evaluation_provenance']):
                    own_context_baseline=None
            except (FileNotFoundError,KeyError,ValueError,json.JSONDecodeError):
                own_context_baseline=None
            if own_context_baseline is None:
                context_path=reports/'base-own-context.json'
                evaluate('slm_lab.own_context_transfer_evaluation',base,context_path)
                own_context_baseline=json.loads(context_path.read_text())
            validate_own_context_coverage(own_context_baseline)
            if own_context_baseline['checkpoint_sha256']!=base_sha:
                raise ValueError('Own-context native baseline belongs to another selected policy')
            atomic_json(reports/'own-context-baseline-receipt.json',{'checkpoint_sha256':base_sha,
                'report':str(context_path),'report_sha256':file_hash(context_path),
                'own_context_manifest_sha256':file_hash(context_manifest_path),'goal_achieved':False,
                'scope':'Selected own policy/compact own-reader native answers before feedback; all128 original development photos.'})
        fact_baseline=None
        if args.human_fact_question_guard:
            verify_reading_sources()
            _,fact_admission=admit_fact_inputs(ROOT)
            atomic_json(reports/'human-fact-input-admission.json',fact_admission)
            fact_data=ROOT/'data/nq-human-questions-v1'
            fact_path=ROOT/args.capability_baseline/'nq-open.json'
            try:
                fact_baseline=json.loads(fact_path.read_text())
                validate_fact_coverage(fact_baseline)
                fact_files=[ROOT/name for name in fact_baseline['evaluation_provenance']['files_sha256']]
                if (fact_baseline['checkpoint_sha256']!=base_sha
                        or fact_baseline['data_sha256']!=file_hash(fact_data/'valid.jsonl')
                        or fact_baseline['manifest_sha256']!=fact_admission['manifest_sha256']
                        or fact_baseline['input_audit_sha256']!=fact_admission['input_audit_sha256']
                        or fact_baseline['exposure_audit_sha256']!=fact_admission['exposure_audit_sha256']
                        or capture(fact_files)!=fact_baseline['evaluation_provenance']):
                    fact_baseline=None
            except (FileNotFoundError,KeyError,ValueError,json.JSONDecodeError):
                fact_baseline=None
            if fact_baseline is None:
                fact_path=reports/'base-nq-open.json'
                evaluate('slm_lab.nq_open_evaluation',base,fact_path)
                fact_baseline=json.loads(fact_path.read_text())
            validate_fact_coverage(fact_baseline)
            if fact_baseline['checkpoint_sha256']!=base_sha:
                raise ValueError('Original factual QA baseline belongs to another selected own policy')
            atomic_json(reports/'human-fact-baseline-receipt.json',{'checkpoint_sha256':base_sha,
                'report':str(fact_path),'report_sha256':file_hash(fact_path),
                'input_audit_sha256':fact_admission['input_audit_sha256'],
                'exposure_audit_sha256':fact_admission['exposure_audit_sha256'],
                'original_development_questions':3610,'goal_achieved':False,
                'scope':'Selected own policy/generated factual answers before feedback; all original cases and exposure categories retained.'})
        js_baseline=None
        if args.javascript_module_guard:
            verify_reading_sources()
            verify_js_toolchain()
            js_data=ROOT/'data/exercism-code-v1/valid.jsonl'
            js_references_path=ROOT/'reports/exercism-javascript-publisher-references-v1.json'
            js_references=json.loads(js_references_path.read_text())
            js_manifest_path=js_data.parent/'manifest.json'
            if (js_references['complete_all37_reference_suites_passed'] is not True
                    or js_references['examples']!=37 or js_references['passed']!=37
                    or js_references['data_sha256']!=file_hash(js_data)
                    or js_references['manifest_sha256']!=file_hash(js_manifest_path)):
                raise ValueError('Complete original JavaScript publisher reference evidence changed')
            js_path=ROOT/args.capability_baseline/'exercise-js.json'
            try:
                js_baseline=json.loads(js_path.read_text())
                validate_js_coverage(js_baseline,js_references['results'])
                js_files=[ROOT/name for name in js_baseline['evaluation_provenance']['files_sha256']]
                if (js_baseline['checkpoint_sha256']!=base_sha
                        or js_baseline['data_sha256']!=file_hash(js_data)
                        or js_baseline['manifest_sha256']!=file_hash(js_manifest_path)
                        or js_baseline['reference_report_sha256']!=file_hash(js_references_path)
                        or js_baseline['toolchain_manifest_sha256']!=file_hash(ROOT/'.tools/javascript-toolchain-v1.json')
                        or capture(js_files)!=js_baseline['evaluation_provenance']):
                    js_baseline=None
            except (FileNotFoundError,KeyError,ValueError,json.JSONDecodeError):
                js_baseline=None
            if js_baseline is None:
                js_path=reports/'base-exercise-js.json'
                evaluate('slm_lab.exercise_js_evaluation',base,js_path,['--data',str(js_data)])
                js_baseline=json.loads(js_path.read_text())
            validate_js_coverage(js_baseline,js_references['results'])
            if js_baseline['checkpoint_sha256']!=base_sha:
                raise ValueError('JavaScript module baseline belongs to another selected own policy')
            atomic_json(reports/'javascript-module-baseline-receipt.json',{'checkpoint_sha256':base_sha,
                'report':str(js_path),'report_sha256':file_hash(js_path),'data_sha256':file_hash(js_data),
                'reference_report_sha256':file_hash(js_references_path),'original_modules':37,'goal_achieved':False,
                'scope':'Selected own policy/full named JavaScript modules before feedback; complete original publisher suites.'})
        go_baseline=None
        if args.go_module_guard:
            verify_reading_sources()
            verify_go_toolchain()
            go_data=ROOT/'data/exercism-code-v1/valid.jsonl'
            go_references_path=ROOT/'reports/exercism-go-publisher-references-v1.json'
            go_references=json.loads(go_references_path.read_text())
            go_manifest_path=go_data.parent/'manifest.json'
            if (go_references['complete_all42_reference_suites_passed'] is not True
                    or go_references['examples']!=42 or go_references['passed']!=42
                    or go_references['data_sha256']!=file_hash(go_data)
                    or go_references['manifest_sha256']!=file_hash(go_manifest_path)):
                raise ValueError('Complete original Go publisher reference evidence changed')
            go_path=ROOT/args.capability_baseline/'exercise-go.json'
            try:
                go_baseline=json.loads(go_path.read_text())
                validate_go_coverage(go_baseline,go_references['results'])
                go_files=[ROOT/name for name in go_baseline['evaluation_provenance']['files_sha256']]
                if (go_baseline['checkpoint_sha256']!=base_sha
                        or go_baseline['data_sha256']!=file_hash(go_data)
                        or go_baseline['manifest_sha256']!=file_hash(go_manifest_path)
                        or go_baseline['reference_report_sha256']!=file_hash(go_references_path)
                        or go_baseline['toolchain_manifest_sha256']!=file_hash(ROOT/'.tools/go-toolchain-v1.json')
                        or capture(go_files)!=go_baseline['evaluation_provenance']):
                    go_baseline=None
            except (FileNotFoundError,KeyError,ValueError,json.JSONDecodeError):
                go_baseline=None
            if go_baseline is None:
                go_path=reports/'base-exercise-go.json'
                evaluate('slm_lab.exercise_go_evaluation',base,go_path,['--data',str(go_data)])
                go_baseline=json.loads(go_path.read_text())
            validate_go_coverage(go_baseline,go_references['results'])
            if go_baseline['checkpoint_sha256']!=base_sha:
                raise ValueError('Go native module baseline belongs to another selected own policy')
            atomic_json(reports/'go-module-baseline-receipt.json',{'checkpoint_sha256':base_sha,
                'report':str(go_path),'report_sha256':file_hash(go_path),'data_sha256':file_hash(go_data),
                'reference_report_sha256':file_hash(go_references_path),'original_modules':42,'goal_achieved':False,
                'scope':'Selected own policy/full native Go modules before feedback; complete original publisher suites.'})
        rust_baseline=None
        if args.rust_module_guard:
            verify_reading_sources()
            verify_rust_toolchain()
            rust_data=ROOT/'data/exercism-code-v1/valid.jsonl'
            rust_references_path=ROOT/'reports/exercism-rust-publisher-references-v2.json'
            rust_references=json.loads(rust_references_path.read_text())
            rust_manifest_path=rust_data.parent/'manifest.json'
            if (rust_references['complete_all24_reference_suites_passed'] is not True
                    or rust_references['examples']!=24 or rust_references['passed']!=24
                    or rust_references['data_sha256']!=file_hash(rust_data)
                    or rust_references['manifest_sha256']!=file_hash(rust_manifest_path)):
                raise ValueError('Complete original Rust publisher reference evidence changed')
            rust_path=ROOT/args.capability_baseline/'exercise-rust.json'
            try:
                rust_baseline=json.loads(rust_path.read_text())
                validate_rust_coverage(rust_baseline,rust_references['results'])
                rust_files=[ROOT/name for name in rust_baseline['evaluation_provenance']['files_sha256']]
                if (rust_baseline['checkpoint_sha256']!=base_sha
                        or rust_baseline['data_sha256']!=file_hash(rust_data)
                        or rust_baseline['manifest_sha256']!=file_hash(rust_manifest_path)
                        or rust_baseline['reference_report_sha256']!=file_hash(rust_references_path)
                        or rust_baseline['toolchain_manifest_sha256']!=file_hash(ROOT/'.tools/rust-toolchain-v1.json')
                        or rust_baseline['kernel_report_sha256']!=file_hash(ROOT/'reports/rust-kernel-and-report-verification-v1.json')
                        or 'src/slm_lab/exercise_rust_evaluation_v2.py' not in rust_baseline['evaluation_provenance']['files_sha256']
                        or capture(rust_files)!=rust_baseline['evaluation_provenance']):
                    rust_baseline=None
            except (FileNotFoundError,KeyError,ValueError,json.JSONDecodeError):
                rust_baseline=None
            if rust_baseline is None:
                rust_path=reports/'base-exercise-rust.json'
                evaluate('slm_lab.exercise_rust_evaluation_v2',base,rust_path,['--data',str(rust_data)])
                rust_baseline=json.loads(rust_path.read_text())
            validate_rust_coverage(rust_baseline,rust_references['results'])
            if rust_baseline['checkpoint_sha256']!=base_sha:
                raise ValueError('Rust native module baseline belongs to another selected own policy')
            atomic_json(reports/'rust-module-baseline-receipt.json',{'checkpoint_sha256':base_sha,
                'report':str(rust_path),'report_sha256':file_hash(rust_path),'data_sha256':file_hash(rust_data),
                'reference_report_sha256':file_hash(rust_references_path),'original_modules':24,'goal_achieved':False,
                'scope':'Selected own policy/full native Rust modules before feedback; complete original publisher suites.'})
        if args.paired_photo_guard:
            verify_reading_sources()
            paired_photo_data=ROOT/'data/vqa-real-pairs-v1'
            paired_photo_rows,paired_photo_receipt=admit_paired_photos(ROOT)
            del paired_photo_rows
            atomic_json(reports/'vqa-pair-input-admission.json',paired_photo_receipt)
            paired_photo_path=ROOT/args.capability_baseline/'vqa-pairs.json'
            try:
                paired_photo_baseline=json.loads(paired_photo_path.read_text())
                validate_paired_photo_coverage(paired_photo_baseline,paired_photo_data)
                paired_photo_files=[ROOT/name for name in paired_photo_baseline['evaluation_provenance']['files_sha256']]
                if (paired_photo_baseline['checkpoint_sha256']!=base_sha
                        or capture(paired_photo_files)!=paired_photo_baseline['evaluation_provenance']):
                    paired_photo_baseline=None
            except (FileNotFoundError,KeyError,ValueError,json.JSONDecodeError):
                paired_photo_baseline=None
            if paired_photo_baseline is None:
                paired_photo_path=reports/'base-vqa-pairs.json'
                evaluate('slm_lab.vqa_pair_evaluation',base,paired_photo_path,['--data',str(paired_photo_data)])
                paired_photo_baseline=json.loads(paired_photo_path.read_text())
            paired_photo_coverage=validate_paired_photo_coverage(paired_photo_baseline,paired_photo_data)
            if paired_photo_baseline['checkpoint_sha256']!=base_sha:
                raise ValueError('Paired-photo baseline belongs to another selected own policy')
            atomic_json(reports/'vqa-pair-baseline-receipt.json',{'checkpoint_sha256':base_sha,
                'report':str(paired_photo_path),'report_sha256':file_hash(paired_photo_path),
                'manifest_sha256':paired_photo_receipt['manifest_sha256'],**paired_photo_coverage,'goal_achieved':False,
                'scope':'Selected own policy/original paired photos before feedback; proper/blank/counterpart controls.'})
        reward=train_stage('reward','reward',data/'preferences.train.jsonl',base,plan['reward_steps'],2,1e-5)
        ranking=reports/'reward-ranking.json'
        evaluate('slm_lab.evaluation',reward,ranking,['--data',str(data/'preferences.valid.jsonl'),
                                                   '--task','preference','--max-examples','1000','--threads','1'])
        quality=gate(json.loads(ranking.read_text()));atomic_json(reports/'reward-gate.json',quality)
        if (reward.parent/'training-state.pt').exists():
            retire(reward.parent,reports/'reward-completed-state-cleanup.json')
        dpo=train_stage('dpo','dpo',data/'preferences.train.jsonl',base,plan['dpo_steps'],1,1e-5)
        evaluate('slm_lab.evaluation',dpo,reports/'dpo-preferences.json',[
            '--data',str(data/'preferences.valid.jsonl'),'--task','policy_preference',
            '--reference-checkpoint',str(base),'--max-examples','1000','--threads','1'])
        if (dpo.parent/'training-state.pt').exists():
            retire(dpo.parent,reports/'dpo-completed-state-cleanup.json')
        branches={'dpo':dpo}
        if quality['passed']:
            ppo=train_stage('ppo','ppo',data/'ppo.train.jsonl',base,128,4,1e-6,world=1,
                extra=['--reward-checkpoint',str(reward),'--max-new-tokens','256','--ppo-epochs','2','--kl-coef','0.05'])
            branches['ppo']=ppo
            evaluate('slm_lab.evaluation',ppo,reports/'ppo-preferences.json',[
                '--data',str(data/'preferences.valid.jsonl'),'--task','policy_preference',
                '--reference-checkpoint',str(base),'--max-examples','1000','--threads','1'])
            if (ppo.parent/'training-state.pt').exists():
                retire(ppo.parent,reports/'ppo-completed-state-cleanup.json')
        else:status('PPO withheld because the held-out reward gate failed',failures=quality['failures'])
        for label,checkpoint in branches.items():
            branch=reports/label;branch.mkdir(exist_ok=True)
            for module,name,extra in capability_specs:
                evaluate(module,checkpoint,branch/name,extra)
            if args.real_photo_reading_guard:
                evaluate('slm_lab.textvqa_evaluation',checkpoint,branch/'textvqa.json',['--data',str(reading_data)])
            if args.own_context_reading_guard:
                evaluate('slm_lab.own_context_transfer_evaluation',checkpoint,branch/'own-context.json')
            if args.human_fact_question_guard:
                evaluate('slm_lab.nq_open_evaluation',checkpoint,branch/'nq-open.json')
            if args.javascript_module_guard:
                evaluate('slm_lab.exercise_js_evaluation',checkpoint,branch/'exercise-js.json',['--data',str(js_data)])
            if args.go_module_guard:
                evaluate('slm_lab.exercise_go_evaluation',checkpoint,branch/'exercise-go.json',['--data',str(go_data)])
            if args.rust_module_guard:
                evaluate('slm_lab.exercise_rust_evaluation_v2',checkpoint,branch/'exercise-rust.json',['--data',str(rust_data)])
            if args.paired_photo_guard:
                evaluate('slm_lab.vqa_pair_evaluation',checkpoint,branch/'vqa-pairs.json',['--data',str(paired_photo_data)])
        if args.capability_guards:
            admitted=[];guards={}
            for label,checkpoint in branches.items():
                preference=json.loads((reports/(label+'-preferences.json')).read_text())
                if preference['data_sha256']!=file_hash(data/'preferences.valid.jsonl'):raise ValueError('Policy preference validation data changed')
                result=compare(baseline,read_capabilities(reports/label),preference)
                if result['candidate_checkpoint_sha256']!=file_hash(checkpoint):raise ValueError('Feedback checkpoint differs from its assessments')
                if args.real_photo_reading_guard:
                    result=enforce_reading_retention(result,reading_baseline,
                        json.loads((reports/label/'textvqa.json').read_text()))
                if args.own_context_reading_guard:
                    result=enforce_own_context_retention(result,own_context_baseline,
                        json.loads((reports/label/'own-context.json').read_text()))
                if args.human_fact_question_guard:
                    result=enforce_fact_retention(result,fact_baseline,
                        json.loads((reports/label/'nq-open.json').read_text()))
                if args.javascript_module_guard:
                    result=enforce_js_retention(result,js_baseline,
                        json.loads((reports/label/'exercise-js.json').read_text()),js_references['results'])
                if args.go_module_guard:
                    result=enforce_go_retention(result,go_baseline,
                        json.loads((reports/label/'exercise-go.json').read_text()),go_references['results'])
                if args.rust_module_guard:
                    result=enforce_rust_retention(result,rust_baseline,
                        json.loads((reports/label/'exercise-rust.json').read_text()),rust_references['results'])
                if args.paired_photo_guard:
                    result=enforce_paired_photo_retention(result,paired_photo_baseline,
                        json.loads((reports/label/'vqa-pairs.json').read_text()),paired_photo_data)
                atomic_json(reports/label/'capability-guard.json',result);guards[label]=result
                if result['eligible_for_selection']:admitted.append((result['preference_positive_margin_95ci'][0],label,checkpoint))
            chosen=max(admitted,key=lambda item:item[0]) if admitted else (None,'base',base)
            atomic_json(reports/'selection.json',{'selected_branch':chosen[1],'selected_checkpoint':str(chosen[2]),
                        'selected_checkpoint_sha256':file_hash(chosen[2]),'guard_reports':{label:str(reports/label/'capability-guard.json') for label in guards},
                        'selection_rule':'Among branches passing all declared capability guards and requested plain/own-reader/factual/paired-photo native answer and full JavaScript/Go/Rust module retention checks, maximize the lower offline positive-preference-margin interval; retain the base if none pass.',
                        'goal_achieved':False,'limitations':['Local tuning evidence, not independent final benchmarks.',
                            'Offline human-pair likelihood improvement is not fresh human judgement of this policy\u2019s generated responses.']})
        atomic_json(reports/'summary.json',{'branches':{k:str(v) for k,v in branches.items()},
            'capability_selection':str(reports/'selection.json') if args.capability_guards else None,
            'plain_native_reading_guard':args.real_photo_reading_guard,
            'own_context_native_reading_guard':args.own_context_reading_guard,
            'human_fact_question_guard':args.human_fact_question_guard,
            'javascript_module_guard':args.javascript_module_guard,
            'go_module_guard':args.go_module_guard,
            'rust_module_guard':args.rust_module_guard,
            'paired_photo_guard':args.paired_photo_guard,
            'reward_gate':quality,'preference_data_manifest_sha256':plan['human_feedback_manifest_sha256'],
            'goal_achieved':False,'high_end_llm_parity':'Not established',
            'limitations':['Offline original human rankings transfer from other-model responses; fresh human policy evaluation is still needed.',
                           'Reward validation and sample benchmarks do not establish broad multimodal or code competence.',
                           'The PPO branch is a small experiment, not a production alignment guarantee.']})
        status('human-feedback pilot training/evaluation completed; broader goal unverified',branches={k:str(v) for k,v in branches.items()})
    except Exception as error:
        status('requires inspection; human-feedback pipeline stopped',error=str(error));raise

if __name__=='__main__':main()
