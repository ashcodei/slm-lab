"""Compare complete unchanged assessment populations before/after4000 SFT updates."""
from datetime import datetime, timezone
import json
from pathlib import Path

from slm_lab.codenet_model_assessment_v1 import summarize
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(s) for s in path.open()]


def common_runtime(a, b, required):
    if any(a[k] != b[k] for k in ['version', 'python', 'packages']):
        raise ValueError('Matched interpreter/package versions required')
    if any(a['files_sha256'].get(k) != b['files_sha256'].get(k) or k not in a['files_sha256'] for k in required):
        raise ValueError('Matched model/encoding/generation/scoring source bytes required')
    for k in a['files_sha256'].keys() & b['files_sha256'].keys():
        if a['files_sha256'][k] != b['files_sha256'][k]:
            raise ValueError('Shared assessment source changed: ' + k)


def knowledge_summary(predictions):
    n = len(predictions)
    result = dict(questions=n, generated_correct=sum(r['generated_correct'] for r in predictions),
        likelihood_correct=sum(r['likelihood_correct'] for r in predictions),
        generated_accuracy=sum(r['generated_correct'] for r in predictions)/n,
        likelihood_accuracy=sum(r['likelihood_correct'] for r in predictions)/n,
        leading_answer_letter_parseable=sum(r['generated_index'] is not None for r in predictions),
        strict_single_letter_responses=sum(len(r['generated_response'].strip()) == 1
            and r['generated_response'].strip() in 'ABCDEFGHIJ'[:len(r['candidate_scores'])] for r in predictions),
        nominal_uniform_chance=sum(r['chance_accuracy'] for r in predictions)/n)
    if all('native_letter_likelihood_correct' in r for r in predictions):
        result['direct_letter_likelihood_correct'] = sum(r['native_letter_likelihood_correct'] for r in predictions)
        result['direct_letter_likelihood_accuracy'] = result['direct_letter_likelihood_correct'] / n
    return result


def main():
    output = ROOT / 'reports/code-instruction-4000-matched-comparison-v1'
    if output.exists():
        raise ValueError('Preserve every earlier comparison or partial result')
    before_code = ROOT / 'reports/codenet-language-completed-baseline-v1'
    after_code = ROOT / 'reports/code-instruction-4000-codenet-v1'
    before_knowledge = ROOT / 'reports/language-completed-knowledge-baseline-v1'
    after_knowledge = ROOT / 'reports/code-instruction-4000-knowledge-v1'
    before_native = ROOT / 'reports/language-completed-native-knowledge-v2'
    after_native = ROOT / 'reports/code-instruction-4000-native-knowledge-v1'
    b, a = read(before_code / 'results.json'), read(after_code / 'results.json')
    kb, ka = read(before_knowledge / 'knowledge.json'), read(after_knowledge / 'knowledge.json')
    nb, na = read(before_native / 'knowledge.json'), read(after_native / 'knowledge.json')
    extraction_path = ROOT / 'checkpoints/code-instruction-4000-assessment-v1/extraction.json'
    extraction = read(extraction_path)
    if (a['checkpoint_sha256'] != extraction['inference_checkpoint_sha256']
            or ka['checkpoint_sha256'] != a['checkpoint_sha256']
            or kb['checkpoint_sha256'] != b['checkpoint_sha256']
            or nb['checkpoint_sha256'] != b['checkpoint_sha256']
            or na['checkpoint_sha256'] != a['checkpoint_sha256']):
        raise ValueError('Exact intended baseline and4000-update policy required in both assessments')
    if any(b[k] != a[k] for k in ['tokenizer_sha256','assessment_manifest_sha256','generation_budget','runtime_profile']):
        raise ValueError('Unchanged code data/tokenizer/budget/runtime profile required')
    for old, new in [(before_code, after_code), (before_knowledge, after_knowledge), (before_native, after_native)]:
        for folder in [old, new]:
            plan = read(folder / 'plan.json')
            for name, digest in plan['source_files_sha256'].items():
                if file_hash(name) != digest:
                    raise ValueError('Assessment input/source file changed: ' + name)
    common_runtime(b['runtime'], a['runtime'], ['src/slm_lab/model.py', 'src/slm_lab/training.py',
        'src/slm_lab/codenet_model_assessment_v1.py', 'src/slm_lab/algorithm_program_v2.py',
        'src/slm_lab/codenet_algorithm_oracles_v2.py'])
    common_runtime(kb['evaluation_provenance'], ka['evaluation_provenance'],
                   ['src/slm_lab/model.py', 'src/slm_lab/training.py', 'src/slm_lab/knowledge_evaluation.py'])
    if any(kb[k] != ka[k] for k in ['questions_sha256','generation_budget','tokenizer_sha256',
                                  'inference_precision','inference_device_type','protocol']):
        raise ValueError('Unchanged complete knowledge protocol required')
    if kb['skipped'] or ka['skipped'] or len(kb['predictions']) != 559 or len(ka['predictions']) != 559:
        raise ValueError('Every original knowledge question must retain its denominator')
    for x, y in zip(kb['predictions'], ka['predictions']):
        if (any(x[k] != y[k] for k in ['id','domain','answer_index','chance_accuracy','context_tokens'])
                or len(x['candidate_scores']) != len(y['candidate_scores'])):
            raise ValueError('Original question/gold/order/prompt-token length differs')
    common_runtime(nb['evaluation_provenance'], na['evaluation_provenance'],
                   ['src/slm_lab/model.py', 'src/slm_lab/training.py', 'src/slm_lab/knowledge_native_evaluation_v1.py'])
    if (nb['skipped'] or na['skipped'] or len(nb['predictions']) != 559 or len(na['predictions']) != 559
            or any(nb[k] != na[k] for k in ['questions_sha256','generation_budget','tokenizer_sha256',
                                          'inference_precision','inference_device_type','protocol'])):
        raise ValueError('Unchanged full native knowledge diagnostic required')
    for x, y, z in zip(nb['predictions'], na['predictions'], kb['predictions']):
        if (any(x[k] != y[k] for k in ['id','domain','answer_index','chance_accuracy','context_tokens'])
                or any(x[k] != z[k] for k in ['id','domain','answer_index','chance_accuracy'])):
            raise ValueError('Native diagnostic questions/golds/order differ')
    questions = lines(ROOT / 'data/codenet-algorithm-assessment-v1/questions.jsonl')
    generations = [lines(folder / 'generations.jsonl') for folder in [before_code, after_code]]
    execution = [lines(folder / 'execution.jsonl') for folder in [before_code, after_code]]
    for folder, report, g, e in zip([before_code, after_code], [b, a], generations, execution):
        if file_hash(folder / 'generations.jsonl') != report['generations_sha256'] or file_hash(folder / 'execution.jsonl') != report['execution_records_sha256']:
            raise ValueError('Original sealed generation/execution bytes differ')
        seal = read(folder / 'generation-seal.json')
        if (file_hash(folder / 'generation-seal.json') != report['generation_seal_sha256']
                or seal['before_model_state_sha256'] != seal['after_model_state_sha256']):
            raise ValueError('Unchanged sealed model state required')
        for k, v in summarize(questions, g, e).items():
            if report[k] != v:
                raise ValueError('Recomputed complete code denominator or score differs')
    for x, y in zip(*generations):
        if any(x[k] != y[k] for k in ['id','language','prompt_tokens','prompt_input_ids_sha256','generation_budget','temperature','sampling_or_repairs']):
            raise ValueError('Exact original complete code prompts and greedy sampling required')
    domains = sorted({r['domain'] for r in kb['predictions']})
    primary = dict(before=knowledge_summary(kb['predictions']), after=knowledge_summary(ka['predictions']))
    code_keys = ['questions','functional_correct','overall_correct_including_defined_stress',
        'syntax_or_compilation_passed','stress_eligible_questions','bounded_stress_passed',
        'generation_budget_exhausted_questions','per_language']
    report = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        baseline_checkpoint_sha256=b['checkpoint_sha256'], after_checkpoint_sha256=a['checkpoint_sha256'],
        own_code_instruction_updates=4000, total_planned_code_instruction_updates=28343,
        all_original_questions_prompts_gold_labels_and_common_sources_match=True,
        primary_raw_knowledge=primary,
        fixed_native_knowledge_diagnostic=dict(before=knowledge_summary(nb['predictions']),
                                               after=knowledge_summary(na['predictions'])),
        paired_knowledge_changes={name: dict(gained=sum(not x[name] and y[name] for x,y in zip(kb['predictions'],ka['predictions'])),
            lost=sum(x[name] and not y[name] for x,y in zip(kb['predictions'],ka['predictions']))) for name in ['generated_correct','likelihood_correct']},
        by_domain={domain: {label: knowledge_summary([r for r in data['predictions'] if r['domain']==domain])
            for label,data in [('before',kb),('after',ka)]} for domain in domains},
        code=dict(before={k:b[k] for k in code_keys}, after={k:a[k] for k in code_keys}),
        source_reports_sha256={str(folder / filename):file_hash(folder / filename)
            for folder, filename in [(before_code,'results.json'),(after_code,'results.json'),
                                     (before_knowledge,'knowledge.json'),(after_knowledge,'knowledge.json'),
                                     (before_native,'knowledge.json'),(after_native,'knowledge.json')]},
        same_model_no_training_control_rerun=False, earlier_corpus_contamination_completely_excluded=False,
        assessment_optimizer_updates=0, automatic_candidate_promotion=False, goal_achieved=False,
        scope='Descriptive matched-task intermediate before/after comparison under unchanged generation/scoring protocols. Not a controlled causal trial, final policy, broad independent benchmark or frontier capability proof.')
    output.mkdir(); atomic_json(output / 'comparison.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['by_domain','source_reports_sha256']}), flush=True)


if __name__ == '__main__':
    main()
