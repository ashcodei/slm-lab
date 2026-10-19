"""Reconcile every fixed-endpoint code/knowledge result; never select a model."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.knowledge_evaluation import letter

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/attention-allocation-pilot-v1'
ARMS = ['FFN_only_control', 'FFN_plus_attention']


def main():
    target = REPORTS / 'independent-comparison.json'
    if target.exists(): raise ValueError('Preserve earlier complete endpoint comparison')
    plan = json.loads((REPORTS / 'plan.json').read_text())
    needed = [REPORTS / name / file for name in ARMS for file in ['evaluation-binding.json', 'code-completion.json', 'code-instruction.json', 'knowledge.json']]
    while not all(p.exists() for p in needed):
        proc = Path('/proc') / str(plan['owner']['pid'])
        try: fields = (proc / 'stat').read_text().split(') ', 1)[1].split()
        except FileNotFoundError: fields = None
        if not fields or fields[0] in {'Z', 'X'} or fields[19] != plan['owner']['ticks']:
            raise ValueError('Exact experiment owner ended before complete endpoint evidence; inspect rather than restart')
        status = json.loads((REPORTS / 'status.json').read_text())
        if any(c['exit_code'] not in [None, 0] for c in status.get('children', [])):
            raise ValueError('A registered experiment worker failed before assessment completion')
        time.sleep(2)
    data = ROOT / 'data/knowledge-eval-v1/questions.jsonl'
    questions = list(map(json.loads, data.read_text().splitlines()))
    if len(questions) != 559: raise ValueError('Every original fixed knowledge question required')
    baseline_code = {mode: json.loads((ROOT / 'reports/extended-curriculum-v1' / file).read_text())
        for mode, file in [('completion', 'code-quality.json'), ('instruction', 'code-quality-instruction.json')]}
    baseline_knowledge = json.loads((ROOT / 'reports/extended-curriculum-v1/knowledge.json').read_text())
    bindings = {str(REPORTS / 'plan.json'): file_hash(REPORTS / 'plan.json'), str(data): file_hash(data), str(Path(__file__).resolve()): file_hash(__file__)}
    policies = {}
    for name in ARMS:
        checkpoint = ROOT / 'checkpoints/attention-allocation-pilot-v1' / name / 'latest.pt'
        receipt = json.loads((REPORTS / name / 'frozen-final-policy.json').read_text())
        saved_sha = file_hash(checkpoint)
        if saved_sha != receipt['checkpoint_sha256'] or receipt['steps'] != 128 or receipt['sealed_before_benchmark_evaluation'] is not True:
            raise ValueError('Exact fixed128-update policy sealed before assessment required')
        eval_binding = json.loads((REPORTS / name / 'evaluation-binding.json').read_text())
        if (eval_binding['checkpoint']['checkpoint_sha256'] != saved_sha or eval_binding['loader_calls'] != 3
                or eval_binding['unchanged_native_prompts_and_scoring'] is not True
                or eval_binding['held_gold_used_for_gradients'] is not False):
            raise ValueError('Actual frozen explicit-code-route native evaluation binding required')
        modes = {}
        for mode in ['completion', 'instruction']:
            path = REPORTS / name / ('code-' + mode + '.json'); report = json.loads(path.read_text()); base = baseline_code[mode]
            for key in ['suite_sha256', 'generation_budget', 'decoding', 'prompt_mode', 'task_count', 'tokenizer_sha256', 'evaluation_provenance', 'inference_device_type', 'inference_precision']:
                if report[key] != base[key]: raise ValueError('Original local code protocol differs: ' + key)
            cases = lambda r: [(x['language'], x['task'], x['specification']) for x in r['results']]
            if (cases(report) != cases(base) or len(set(cases(report))) != 24 or report['checkpoint_sha256'] != saved_sha):
                raise ValueError('Every original local code case and exact learned policy required')
            counts = dict(tasks=24, syntax=sum(x['candidate']['syntax_valid'] for x in report['results']),
                functional=sum(x['candidate']['functional_passed'] for x in report['results']), efficiency=sum(x['efficiency_stress_passed'] for x in report['results']))
            if any(counts[k] / 24 != report[field] for k, field in [('syntax', 'syntax_pass_rate'), ('functional', 'functional_pass_at_1'), ('efficiency', 'efficiency_stress_pass_rate')]):
                raise ValueError('Raw code cases do not reproduce complete aggregates')
            modes[mode] = counts; bindings[str(path)] = file_hash(path)
        path = REPORTS / name / 'knowledge.json'; report = json.loads(path.read_text())
        for key in ['questions_sha256', 'generation_budget', 'contamination_audit_sha256', 'protocol', 'tokenizer_sha256', 'evaluation_provenance', 'inference_device_type', 'inference_precision']:
            if report[key] != baseline_knowledge[key]: raise ValueError('Original knowledge protocol differs: ' + key)
        if report['checkpoint_sha256'] != saved_sha or report['skipped'] or len(report['predictions']) != 559:
            raise ValueError('Exact learned policy and every fixed knowledge item required')
        generated = likelihood = 0; domains = {}
        for actual, expected in zip(report['predictions'], questions):
            scores = actual['candidate_scores']
            if len(scores) != len(expected['options']) or not all(math.isfinite(v) for v in scores):
                raise ValueError('Complete finite answer-choice scores required')
            g = letter(actual['generated_response'], len(scores)); l = max(range(len(scores)), key=lambda i: scores[i])
            if (actual['id'] != expected['id'] or actual['domain'] != expected['domain'] or actual['answer_index'] != expected['answer_index']
                    or actual['generated_index'] != g or actual['likelihood_index'] != l
                    or actual['generated_correct'] != (g == expected['answer_index']) or actual['likelihood_correct'] != (l == expected['answer_index'])):
                raise ValueError('Raw generated/likelihood answers do not reproduce original targets')
            generated += int(actual['generated_correct']); likelihood += int(actual['likelihood_correct'])
            d = domains.setdefault(actual['domain'], dict(questions=0, generated_correct=0, likelihood_correct=0))
            d['questions'] += 1; d['generated_correct'] += int(actual['generated_correct']); d['likelihood_correct'] += int(actual['likelihood_correct'])
        if generated / 559 != report['overall']['generated_accuracy'] or likelihood / 559 != report['overall']['likelihood_accuracy']:
            raise ValueError('Full native knowledge aggregates differ from raw predictions')
        for domain, count in domains.items():
            if (count['questions'] != report['by_domain'][domain]['questions']
                    or count['generated_correct'] / count['questions'] != report['by_domain'][domain]['generated_accuracy']
                    or count['likelihood_correct'] / count['questions'] != report['by_domain'][domain]['likelihood_accuracy']):
                raise ValueError('Every complete native domain aggregate required')
        policies[name] = dict(checkpoint_sha256=saved_sha, architecture=receipt['architecture'], code=modes,
            knowledge=dict(questions=559, generated_correct=generated, likelihood_correct=likelihood, by_domain=domains),
            explicit_route='code', general_route_original_joint_weights_unchanged=True)
        bindings[str(checkpoint)] = saved_sha
        for file in ['knowledge.json', 'evaluation-binding.json', 'frozen-final-policy.json', 'reference.json', 'initial-policy.json', 'source-conditioning-before.json', 'source-conditioning-after.json']:
            p = REPORTS / name / file; bindings[str(p)] = file_hash(p)
    selection = json.loads((ROOT / 'reports/code-residual-lr-pilot-v1/selected-original-rows.json').read_text())['validation']
    for name in ARMS:
        phases = {}
        for phase in ['before', 'after']:
            d = json.loads((REPORTS / name / ('source-conditioning-' + phase + '.json')).read_text())
            if (d['phase'] != phase or d['source_ids'] != [r['id'] for r in selection]
                    or len(d['measurements']) != 128 or d['used_for_gradients_or_endpoint_selection'] is not False):
                raise ValueError('All128 original full held source diagnostics, never used for updates/selection, required')
            v = d['measurements']; paired = [m for m in v if m['mismatched'] is not None]
            if any(not math.isfinite(m[k]) for m in v for k in ['matched', 'mismatched'] if m[k] is not None):
                raise ValueError('Finite complete source-pair NLLs required')
            phases[phase] = dict(examples=128, paired_examples=len(paired),
                proper_source_token_mean_NLL=sum(m['matched'] * r['target_tokens'] for m, r in zip(v, selection)) / sum(r['target_tokens'] for r in selection),
                wrong_minus_proper_sequence_mean_NLL=sum(m['mismatched'] - m['matched'] for m in paired) / len(paired),
                paired_proper_lower_NLL=sum(m['matched'] < m['mismatched'] for m in paired))
        policies[name]['full128_source_conditioning'] = phases
    result = dict(completed_at=datetime.now(timezone.utc).isoformat(), policies=policies,
        original_joint_baseline=dict(code={m: dict(syntax=sum(x['candidate']['syntax_valid'] for x in r['results']),
            functional=sum(x['candidate']['functional_passed'] for x in r['results']), efficiency=sum(x['efficiency_stress_passed'] for x in r['results'])) for m, r in baseline_code.items()},
            knowledge=dict(generated_correct=sum(x['generated_correct'] for x in baseline_knowledge['predictions']), likelihood_correct=sum(x['likelihood_correct'] for x in baseline_knowledge['predictions']))),
        completed_training_steps_per_arm=128, machine_execution_not_human_RLHF=True, compute_matched=False,
        total_unique_parameters_and_data_exposure_budget_matched=True,
        LoRA_prior_work="https://arxiv.org/abs/2106.09685",
        full112609_program_all30_source_production=False, candidate_promoted=False, goal_achieved=False,
        scope='Parameter-matched fixed-budget actual857M explicit code-route architecture ablation on complete local24-task/two-format and559-question/14-domain reused development protocols. General route retains own joint weights; automatic routing, independent final benchmarks, multimodal quality and high-end capability are not established.',
        files_sha256=bindings)
    for path, digest in bindings.items():
        if file_hash(path) != digest: raise ValueError('Completed result bytes changed during readback')
    atomic_json(target, result)
    print(json.dumps(policies, indent=2))


if __name__ == '__main__': main()
