"""Compare two own policies on the unchanged 175-problem public Python subset."""
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import tempfile

import torch

from slm_lab.checkpoint_xor import restore
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.public_code_benchmark import evaluate

ROOT = Path(__file__).resolve().parents[1]


def summarize(report):
    rows = report['results']
    if len(rows) != 175 or len({r['id'] for r in rows}) != 175 or report['supported_problems'] != 175:
        raise ValueError('Complete original175-problem population required')
    syntax = sum(bool(r['execution'].get('syntax_valid')) for r in rows)
    correct = sum(bool(r['execution'].get('functional_passed')) for r in rows)
    if report['syntax_rate_on_supported_subset'] != syntax / 175 or report['pass_at_1_on_supported_subset'] != correct / 175:
        raise ValueError('Published subset rates differ from actual cases')
    return {'problems': 175, 'syntax_correct': syntax, 'functional_correct': correct}


def main():
    torch.set_num_threads(1)
    reports = ROOT / 'reports/public-code-generalization-v1'
    if reports.exists(): raise ValueError('Preserve the earlier public-code assessment')
    reports.mkdir()
    data = ROOT / 'data/public-code-assessment-v1/questions.jsonl'
    manifest_path = data.parent / 'manifest.json'; manifest = json.loads(manifest_path.read_text())
    if manifest['supported_problems'] != 175 or file_hash(data) != manifest['questions_sha256']:
        raise ValueError('Original prepared public problem population differs')
    base = ROOT / 'checkpoints/instruction-generalization-probe-v3/candidate/model.pt'
    token = base.parent / 'tokenizer.json'
    archive = ROOT / 'checkpoints/binding-retention-v1/model.pt.xor'
    preceding = ROOT / 'reports/binding-retention-v1'
    if json.loads((preceding / 'status.json').read_text())['status'] != 'binding/rehearsal training and matched assessments completed; broader capability unverified':
        raise ValueError('Completed preceding intervention required')
    before_path = ROOT / 'reports/instruction-generalization-probe-v3/after-code-instruction.json'
    after_path = preceding / 'after-code-instruction.json'
    before_report = json.loads(before_path.read_text())
    after_report = json.loads(after_path.read_text())
    if file_hash(base) != before_report['checkpoint_sha256']:
        raise ValueError('Exact own original instruction policy differs')
    files = list(dict.fromkeys([ROOT / name for name in capture()['files_sha256']] + [Path(__file__)] +
        [ROOT / 'src/slm_lab' / (n + '.py') for n in ['public_code_benchmark', 'verified_code', 'checkpoint_xor', 'checkpoint_archive']]))
    runtime = capture(files)
    bound = {p: file_hash(p) for p in [data, manifest_path, base, token, before_path, after_path,
                                     archive / 'manifest.json', archive / 'payload.bin']}
    def guard():
        if capture(files) != runtime or any(file_hash(p) != digest for p, digest in bound.items()):
            raise ValueError('Bound public-code policy/data/runtime changed')
    def status(message, **fields):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
            'optimizer_updates': 0, 'goal_achieved': False, **fields})
    atomic_json(reports / 'plan.json', {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'bound_files_sha256': {str(p.relative_to(ROOT)): digest for p, digest in bound.items()},
        'runtime': runtime, 'supported_problems': 175,
        'official_sanitized_test_problems': manifest['official_sanitized_test_problems'],
        'excluded_problems': manifest['excluded_problems'],
        'base_checkpoint_sha256': before_report['checkpoint_sha256'],
        'candidate_checkpoint_sha256': after_report['checkpoint_sha256'],
        'generation_budget': 512, 'device': 'cuda:3', 'new_model_training': False,
        'automatic_policy_promotion': False, 'goal_achieved': False})
    temporary = Path(tempfile.mkdtemp(prefix='slm-public-code-generalization-v1-', dir='/tmp'))
    try:
        status('restoring exact own completed policy for public-code comparison')
        candidate = temporary / 'model.pt'; restore(archive, candidate); shutil.copy2(token, temporary / 'tokenizer.json')
        if file_hash(candidate) != after_report['checkpoint_sha256']:
            raise ValueError('Exact own completed binding/rehearsal policy differs')
        torch.cuda.set_per_process_memory_fraction(.3, 3)
        collections = {}
        for label, policy in [('original_instruction', base), ('binding_rehearsal', candidate)]:
            status('assessing original public Python problems', policy=label, problems=175)
            evaluate(policy, data, reports / (label + '.json'), 'cuda:3', max_new_tokens=512)
            gc.collect(); torch.cuda.empty_cache(); guard()
            collections[label] = json.loads((reports / (label + '.json')).read_text())
        before, after = [collections[name] for name in ['original_instruction', 'binding_rehearsal']]
        fields = ['data_sha256', 'official_sanitized_test_problems', 'supported_problems', 'excluded_problems',
                  'protocol', 'generation_budget', 'evaluation_provenance', 'tokenizer_sha256',
                  'inference_device_type', 'inference_precision']
        if any(before[k] != after[k] for k in fields): raise ValueError('Public subset protocol/population/runtime differs')
        identities = lambda report: [(r['id'], r['task_id']) for r in report['results']]
        if identities(before) != identities(after): raise ValueError('Original task identities differ')
        result = {'completed_at': datetime.now(timezone.utc).isoformat(),
            'before': summarize(before), 'after': summarize(after),
            'base_checkpoint_sha256': before['checkpoint_sha256'], 'candidate_checkpoint_sha256': after['checkpoint_sha256'],
            'functional_correct_gained': [b['id'] for a, b in zip(before['results'], after['results'])
                if not a['execution'].get('functional_passed') and b['execution'].get('functional_passed')],
            'functional_correct_lost': [a['id'] for a, b in zip(before['results'], after['results'])
                if a['execution'].get('functional_passed') and not b['execution'].get('functional_passed')],
            'scope': 'Original supported sanitized MBPP subset, zero-shot specifications/signatures without scoring cases. Restricted175/257 problems; prior pretraining contamination is not fully certified. Published correctness only, no efficiency or frontier comparison.',
            'new_model_training': False, 'automatic_policy_promotion': False, 'goal_achieved': False}
        atomic_json(reports / 'comparison.json', result)
        shutil.rmtree(temporary)
        atomic_json(reports / 'temporary-cleanup.json', {
            'owned_temporary': str(temporary), 'removed': not temporary.exists(),
            'durable_candidate_archive_retained': archive.exists(),
            'required_own_base_retained': base.exists()})
        status('public-code policy comparison completed; broader capability unverified')
    except Exception as error:
        status('public-code assessment stopped; requires inspection', error=str(error), temporary=str(temporary)); raise


if __name__ == '__main__': main()
