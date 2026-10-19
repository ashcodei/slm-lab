"""Separate known-program recall from example/name cue dependence."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

import torch
from tokenizers import Tokenizer

from slm_lab.checkpoint_xor import restore
from slm_lab.code_prompt_controls_v1 import without_examples, new_function_name
from slm_lab.code_quality import execute, instruction_code
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.evaluation_provenance import capture
from slm_lab.instruction_generalization_trial_v3 import complete_encoding
from slm_lab.parameter_binding_curriculum_v1 import check
from slm_lab.recurrent_ablation import state_digest
from slm_lab.training import Encoder, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    folder = ROOT / 'reports/code-prompt-controls-v1'
    if folder.exists(): raise ValueError('Preserve earlier prompt diagnosis')
    folder.mkdir()
    preceding = ROOT / 'reports/binding-retention-v1'
    if json.loads((preceding / 'status.json').read_text())['status'] != 'binding/rehearsal training and matched assessments completed; broader capability unverified':
        raise ValueError('Inspect the completed preceding intervention first')
    archive = ROOT / 'checkpoints/binding-retention-v1/model.pt.xor'
    source = ROOT / 'checkpoints/instruction-generalization-probe-v3/data/verified_python.jsonl'
    tokenizer = ROOT / 'checkpoints/instruction-generalization-probe-v3/candidate/tokenizer.json'
    baseline_path = preceding / 'after-seen.json'; baseline = json.loads(baseline_path.read_text())
    original = {row['id']: row for row in jsonl(source)}
    rows = [original[row['id']] for row in baseline['results']]
    if len(rows) != 128 or len({r['original_program_sha256'] for r in rows}) != 128:
        raise ValueError('Expected the exact original128 known training algorithms')
    if file_hash(source) != baseline['data_sha256'] or file_hash(tokenizer) != baseline['tokenizer_sha256']:
        raise ValueError('Original diagnosed public source/tokenizer differs')
    files = list(dict.fromkeys([ROOT / name for name in capture()['files_sha256']] + [Path(__file__)] +
        [ROOT / 'src/slm_lab' / (name + '.py') for name in ['code_prompt_controls_v1', 'verified_code',
          'parameter_binding_curriculum_v1', 'instruction_generalization_trial_v3', 'recurrent_ablation',
          'checkpoint_xor', 'checkpoint_archive']]))
    runtime = capture(files)
    bound = {p: file_hash(p) for p in [source, source.parent / 'manifest.json', tokenizer, baseline_path,
                                     archive / 'manifest.json', archive / 'payload.bin']}
    def guard():
        if capture(files) != runtime or any(file_hash(p) != digest for p, digest in bound.items()):
            raise ValueError('Bound prompt diagnosis inputs/runtime changed')
    def status(message, **fields):
        atomic_json(folder / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
            'weights_changed': False, 'goal_achieved': False, **fields})
    location = Path(tempfile.mkdtemp(prefix='slm-code-prompt-controls-v1-', dir='/tmp'))
    try:
        status('restoring exact own completed policy for prompt controls')
        checkpoint = location / 'model.pt'; restore(archive, checkpoint); shutil.copy2(tokenizer, location / 'tokenizer.json')
        if file_hash(checkpoint) != baseline['checkpoint_sha256']: raise ValueError('Exact completed diagnosis policy differs')
        torch.cuda.set_per_process_memory_fraction(.3, 3)
        model, metadata = load_checkpoint(checkpoint, 'cuda:3', training_state=False); model.eval()
        encoder = Encoder(Tokenizer.from_file(str(tokenizer)), model.config, source.parent)
        initial = state_digest(model.state_dict()); reports = {}
        for name, transform in [('without_examples', without_examples), ('new_function_name', new_function_name)]:
            results = []
            for original_row in rows:
                row = transform(original_row); counts = complete_encoding(row, encoder)
                gold = check(row)
                if not gold['functional_passed']: raise ValueError('Transformed public reference fails original examples: ' + row['id'])
                batch = encoder.batch([row], 'response', 'cuda:3')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out = model(**batch); nll = float(out['loss'])
                    prompt = [encoder.bos] + encoder.tokens(encoder.prompt(row))
                    if len(prompt) + 512 > model.config.max_text_length: raise ValueError('Complete controlled generation exceeds context')
                    generated = model.generate(torch.tensor([prompt], device='cuda:3'), max_new_tokens=512, eos_token_id=encoder.eos)
                raw = encoder.tokenizer.decode(generated[0, len(prompt):].tolist()); code = instruction_code(raw)
                outcome = execute({'language': 'Python', 'code': code, 'function': row['function'],
                                   'cases': row['verification_cases'], 'stress': []})
                results.append({'id': row['id'], 'original_program_sha256': row['original_program_sha256'],
                    'original_function': original_row['function'], 'requested_function': row['function'],
                    'prompt': row['prompt'], 'target_response': row['response'], 'published_cases': len(row['verification_cases']),
                    **counts, 'teacher_nll': nll, 'gold_reference': gold, 'raw_response': raw, 'code': code, 'candidate': outcome})
                print(json.dumps({'condition': name, 'examples': len(results), 'functional': outcome['functional_passed']}), flush=True)
                status('assessing known programs with prompt controls', condition=name, completed_examples=len(results), planned_examples=128)
            tokens = sum(r['target_tokens'] for r in results)
            report = {'condition': name, 'checkpoint_sha256': baseline['checkpoint_sha256'], 'source_sha256': bound[source],
                'examples': len(results), 'generation_budget': 512, 'functional_correct': sum(r['candidate']['functional_passed'] for r in results),
                'syntax_correct': sum(r['candidate']['syntax_valid'] for r in results),
                'teacher_nll': sum(r['teacher_nll'] * r['target_tokens'] for r in results) / tokens,
                'baseline_original_prompt_correct': sum(r['candidate']['functional_passed'] for r in baseline['results']),
                'results': results, 'source_runtime': runtime, 'inference_device': 'cuda:3', 'inference_precision': 'bf16',
                'weights_changed': False, 'goal_achieved': False,
                'scope': 'Known training algorithms only. Example removal/name novelty controls; not fresh algorithm generalization or efficiency evidence.',
                'limitations': ['Reference/public source examples still used for scoring but removed from model prompts.',
                    'New function identifiers introduce tokenization novelty and must be delivered exactly.',
                    'Original descriptions may retain semantic algorithm names; ambiguous descriptions are not independently relabeled.',
                    'Prompt lengths/target identifiers differ; not an isolated attention-position or causal semantic-understanding proof.']}
            atomic_json(folder / (name + '.json'), report); reports[name] = report; guard()
        if state_digest(model.state_dict()) != initial or encoder.truncated: raise ValueError('Policy changed or controlled source was clipped')
        atomic_json(folder / 'summary.json', {'completed_at': datetime.now(timezone.utc).isoformat(),
            'original_full_prompt_correct': sum(r['candidate']['functional_passed'] for r in baseline['results']),
            'conditions': {k: {field: v[field] for field in ['functional_correct', 'syntax_correct', 'teacher_nll']} for k, v in reports.items()},
            'exact_all_model_tensors_unchanged': True, 'known_training_algorithms': 128, 'goal_achieved': False,
            'scope': 'Prompt/name dependence of known-program generation; no new policy learning, independent final or frontier comparison.'})
        del model; shutil.rmtree(location)
        status('known-program prompt controls completed; broader capability unverified')
    except Exception as error:
        status('prompt controls stopped; requires inspection', error=str(error), temporary_directory=str(location)); raise


if __name__ == '__main__': main()
