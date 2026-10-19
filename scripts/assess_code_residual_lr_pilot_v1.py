"""Wait for the exact pilot, then assess its frozen candidate with existing protocols."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path
import re
import time

import torch

import assess_codenet_model_v2 as code_assessment
from slm_lab import knowledge_evaluation
from slm_lab.code_residual_inference_v1 import load_residual_policy
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_model_assessment_v1 import score_record,summarize
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json


ROOT=Path(__file__).resolve().parents[1]


def original_process_live(launch):
    process=Path('/proc')/str(launch['pid'])
    try:stat=(process/'stat').read_text()
    except FileNotFoundError:return False
    fields=stat[stat.rfind(')')+2:].split()
    return fields[19]==launch['start_ticks'] and fields[0]!='Z'


def compare_fixed_results(candidate_code_folder,candidate_knowledge,output):
    data=ROOT/'data/codenet-algorithm-assessment-v1'
    questions=list(map(json.loads,(data/'questions.jsonl').open()))
    cases=defaultdict(list)
    for case in map(json.loads,(data/'cases.jsonl').open()):cases[case['problem_id']].append(case)
    folders=[ROOT/'reports/codenet-language-completed-baseline-v1',candidate_code_folder]
    populations=[];summaries=[];core_sources=[];artifacts={}
    for folder in folders:
        plan=json.loads((folder/'plan.json').read_text());report=json.loads((folder/'results.json').read_text())
        seal=json.loads((folder/'generation-seal.json').read_text())
        generations=list(map(json.loads,(folder/'generations.jsonl').open()))
        executions=list(map(json.loads,(folder/'execution.jsonl').open()))
        if (file_hash(folder/'generations.jsonl')!=seal['generations_sha256']
                or report['generations_sha256']!=seal['generations_sha256']
                or file_hash(folder/'execution.jsonl')!=report['execution_records_sha256']
                or file_hash(folder/'generation-seal.json')!=report['generation_seal_sha256']
                or file_hash(data/'manifest.json')!=report['assessment_manifest_sha256']
                or len(generations)!=28 or len(executions)!=28 or report['generation_budget']!=2048):
            raise ValueError('Every original sealed fixed-budget code completion/execution required')
        recomputed=[]
        for q,g,e in zip(questions,generations,executions):
            if (g['id']!=q['id'] or e['id']!=q['id'] or g['generation_budget']!=2048
                    or hashlib.sha256(g['program_text'].encode()).hexdigest()!=g['program_sha256']
                    or g['program_sha256']!=e['program_sha256']):
                raise ValueError('Original fixed question/raw program identity differs')
            row=score_record(q,g,cases[q['problem_id']],None,
                lambda code,language,source_cases,tools,stored=e['execution']:stored)
            if row!=e:raise ValueError('Every sealed code score must rederive from unchanged execution evidence')
            recomputed.append(row)
        summary=summarize(questions,generations,recomputed)
        if any(report[k]!=v for k,v in summary.items()):raise ValueError('Complete code aggregate differs')
        populations.append([(g['id'],g['prompt_input_ids_sha256'],g['prompt_tokens'],g['generation_budget']) for g in generations])
        summaries.append(summary);core_sources.append(plan['source_files_sha256'])
        for name in ['plan.json','results.json','generation-seal.json','generations.jsonl','execution.jsonl']:
            artifacts[str(folder/name)]=file_hash(folder/name)
    if populations[0]!=populations[1]:raise ValueError('All original code prompt IDs/budgets must match')
    common={name:h for name,h in core_sources[0].items() if name in core_sources[1] and '/src/slm_lab/' in name}
    if not common or any(core_sources[1][n]!=h or file_hash(Path(n))!=h for n,h in common.items()):
        raise ValueError('Common primary generation/scoring/model runtime source changed')
    qa_paths=[ROOT/'reports/language-completed-knowledge-baseline-v1/knowledge.json',candidate_knowledge]
    original_questions=list(map(json.loads,(ROOT/'data/knowledge-eval-v1/questions.jsonl').open()))
    qa=[]
    for path in qa_paths:
        report=json.loads(path.read_text());predictions=report['predictions']
        if (len(predictions)!=559 or report['skipped'] or report['generation_budget']!=32
                or report['questions_sha256']!=file_hash(ROOT/'data/knowledge-eval-v1/questions.jsonl')):
            raise ValueError('Every original full-budget broad-domain question required')
        for q,p in zip(original_questions,predictions):
            generated=knowledge_evaluation.letter(p['generated_response'],len(q['options']))
            likely=max(range(len(p['candidate_scores'])),key=lambda i:p['candidate_scores'][i])
            if (p['id']!=q['id'] or p['answer_index']!=q['answer_index'] or p['domain']!=q['domain']
                    or p['generated_index']!=generated or p['generated_correct']!=(generated==q['answer_index'])
                    or p['likelihood_index']!=likely or p['likelihood_correct']!=(likely==q['answer_index'])):
                raise ValueError('Every original broad-domain gold and raw score must rederive')
        qa.append(dict(questions=559,generated_correct=sum(p['generated_correct'] for p in predictions),
            letter_likelihood_correct=sum(p['likelihood_correct'] for p in predictions),
            strict_single_letter_responses=sum(bool(re.fullmatch(r'\s*[A-J]\s*',p['generated_response'],re.I)) for p in predictions),
            by_domain={d:dict(questions=sum(p['domain']==d for p in predictions),
                generated_correct=sum(p['domain']==d and p['generated_correct'] for p in predictions),
                letter_likelihood_correct=sum(p['domain']==d and p['likelihood_correct'] for p in predictions)) for d in sorted({p['domain'] for p in predictions})}))
        artifacts[str(path)]=file_hash(path)
    comparison=dict(completed_at=datetime.now(timezone.utc).isoformat(),baseline_code=summaries[0],candidate_code=summaries[1],
        baseline_raw_knowledge=qa[0],candidate_code_route_raw_knowledge=qa[1],
        every_original_code_prompt_budget_sealed_program_and_execution_recomputed=True,
        every_original559_question_gold_and_score_recomputed=True,common_primary_runtime_files_sha256=common,
        artifacts_sha256=artifacts,automatic_routing_quality_assessed=False,candidate_promoted=False,goal_achieved=False,
        scope='One internally validation-selected128-update own857M code-route pilot versus its own706M completed-language base; fixed28 code questions/2048 tokens and559 raw QA questions/32 tokens. Intermediate development samples with retained earlier-corpus/semantic-exposure qualifications, not final independent all-domain or high-end benchmarks. The explicit frozen general route remains a separate original route.')
    atomic_json(output,comparison);return comparison


def main():
    reports=ROOT/'reports/code-residual-pilot-assessment-controller-v1'
    if reports.exists():raise ValueError('Preserve earlier assessment orchestration evidence')
    launch_path=ROOT/'reports/code-residual-lr-pilot-v1-launch.json'
    launch=json.loads(launch_path.read_text())
    files=[Path(__file__),launch_path,ROOT/'scripts/assess_codenet_model_v2.py',
        ROOT/'src/slm_lab/code_residual_inference_v1.py',ROOT/'src/slm_lab/code_residual_branch_v1.py',
        ROOT/'src/slm_lab/code_residual_checkpoint_v1.py',ROOT/'src/slm_lab/code_residual_pilot_v1.py',
        ROOT/'src/slm_lab/knowledge_evaluation.py',ROOT/'src/slm_lab/codenet_model_assessment_v1.py',
        ROOT/'tests/test_code_residual_inference_v1.py',ROOT/'reports/code-residual-inference-v1-focused-tests.log',
        ROOT/'tests/test_code_residual_pilot_comparison_v1.py',ROOT/'reports/code-residual-pilot-comparison-v1-focused-tests.log']
    bound={str(p):file_hash(p) for p in files};guard=FileBindings(bound);reports.mkdir()
    def status(message,**fields):
        atomic_json(reports/'status.json',dict(status=message,updated_at=datetime.now(timezone.utc).isoformat(),
            candidate_promoted=False,goal_achieved=False,**fields))
    atomic_json(reports/'plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,
        preceding_pilot_pid=launch['pid'],preceding_pilot_start_ticks=launch['start_ticks'],
        code_driver='scripts/assess_codenet_model_v2.py',code_question_count=28,code_generation_budget=2048,
        primary_code_parsing_or_scoring_changes=False,architecture_loading_only='src/slm_lab/code_residual_inference_v1.py',
        raw_knowledge_driver='src/slm_lab/knowledge_evaluation.py',raw_knowledge_question_count=559,raw_knowledge_generation_budget=32,
        evaluation_GPU='cuda:0',maximum_memory_fraction=.14,selection_uses_functional_or_QA_test_scores=False,
        assessment_optimizer_updates=0,automatic_promotion=False,goal_achieved=False))
    while original_process_live(launch):
        guard.check();status('waiting for exact live train/validation learning-rate pilot to finish',
            preceding_pid=launch['pid'],preceding_start_ticks=launch['start_ticks'])
        time.sleep(5)
    result_path=ROOT/'reports/code-residual-lr-pilot-v1/results.json';trained=json.loads(result_path.read_text())
    if trained['actual_Adam_updates_total']!=384 or trained['held_functional_test_feedback_used']:
        raise ValueError('Completed predeclared train/validation-only learning-rate comparison required')
    checkpoint=Path(trained['best_complete_checkpoint']);base=ROOT/'checkpoints/extended-curriculum-v1/language-pretraining/model.pt'
    token=checkpoint.parent/'tokenizer.json';plan_path=ROOT/'reports/code-residual-lr-pilot-v1/plan.json'
    plan=json.loads(plan_path.read_text())
    if file_hash(plan_path)!=trained['run_plan_sha256'] or file_hash(checkpoint)!=trained['best_complete_checkpoint_sha256']:
        raise ValueError('Exact frozen internally selected checkpoint/plan required')
    bound.update({str(result_path):file_hash(result_path),str(checkpoint):trained['best_complete_checkpoint_sha256'],
        str(base):plan['source_checkpoint_sha256'],str(plan_path):trained['run_plan_sha256'],str(token):plan['tokenizer_sha256'],
        str(ROOT/'data/knowledge-eval-v1/questions.jsonl'):file_hash(ROOT/'data/knowledge-eval-v1/questions.jsonl'),
        str(ROOT/'data/knowledge-eval-v1/questions.audit.json'):file_hash(ROOT/'data/knowledge-eval-v1/questions.audit.json')})
    guard=FileBindings(bound)
    atomic_json(reports/'frozen-candidate-binding.json',dict(recorded_at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=bound,selected_profile=trained['best_profile'],pilot_terminal=True,
        selected_without_any_functional_or_QA_test_feedback=True,checkpoint= str(checkpoint),goal_achieved=False))
    def loader(path,device='cpu',training_state=True):
        if Path(path).resolve()!=checkpoint.resolve() or training_state is not False:
            raise ValueError('Only the frozen own residual inference policy may be loaded')
        policy,metadata=load_residual_policy(checkpoint,base,token,expected_checkpoint_sha256=trained['best_complete_checkpoint_sha256'],
            expected_base_sha256=plan['source_checkpoint_sha256'],expected_plan_sha256=trained['run_plan_sha256'],device=device)
        if metadata['pilot_profile']!=trained['best_profile']:raise ValueError('Frozen selected profile differs')
        return policy,metadata
    # Process-local architecture loading adapters; original scorer/source files remain unchanged.
    code_assessment.load_checkpoint=loader
    code_folder=ROOT/'reports/code-residual-lr-pilot-v1-held-code-v1'
    status('assessing frozen pilot on every original fixed raw code question',selected_profile=trained['best_profile'])
    code_assessment.evaluate(checkpoint,code_folder,'cuda:0',threads=1,memory_fraction=.14,
        expected_metadata=dict(stage='code_residual_lr_pilot_v1',steps=128,planned_steps=128,
            data_sha256=file_hash(ROOT/'data/codenet-durable-pack-v2/manifest.json')),
        label='own857M_code_residual_512_original_train_programmes_internal_validation_selected_pilot',
        expected_checkpoint_sha256=trained['best_complete_checkpoint_sha256'])
    guard.check(full=True);gc.collect();torch.cuda.empty_cache()
    knowledge_evaluation.load_checkpoint=loader
    knowledge_path=reports/'code-route-raw-knowledge.json'
    status('assessing frozen code route on all559 original raw broad-domain questions')
    knowledge_evaluation.evaluate(checkpoint,ROOT/'data/knowledge-eval-v1/questions.jsonl',knowledge_path,
        device='cuda:0',max_examples=559,max_new_tokens=32)
    guard.check(full=True)
    comparison=compare_fixed_results(code_folder,knowledge_path,reports/'matched-comparison.json')
    status('fixed held code and raw broad-domain assessment complete; inspect matched capability evidence',
        code=comparison['candidate_code'],code_route_knowledge=comparison['candidate_code_route_raw_knowledge'])
    print('own857M_code_residual_pilot_complete_matched_assessment',json.dumps(comparison),flush=True)


if __name__=='__main__':main()
