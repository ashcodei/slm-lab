"""Bounded execution-derived preference pilot; isolated from production selection."""
from argparse import Namespace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import torch
from tokenizers import Tokenizer
from slm_lab.code_quality import evaluate as code_evaluate
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.evaluation_provenance import capture
from slm_lab.model import ModelConfig
from slm_lab.training import train, Encoder
from slm_lab.verified_code_diagnostics import evaluate as seen_evaluate

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    reports = ROOT / 'reports/code-execution-dpo-v1'; reports.mkdir(exist_ok=True)
    checkpoints = ROOT / 'checkpoints/code-execution-dpo-v1'
    if any(reports.iterdir()) or (checkpoints.exists() and any(checkpoints.iterdir())):
        raise ValueError('Existing execution pilot requires inspection; no automatic restart')
    def status(message, **fields):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
                                             'goal_achieved': False, 'production_promoted': False, **fields})
    try:
        if shutil.disk_usage(ROOT).free < 30 * 1024**3:
            raise ValueError('Need 30 GiB free alongside existing training snapshots')
        source = ROOT / 'checkpoints/verified-code-foundation-probe-v1/candidate/model.pt'
        preferences = ROOT / 'data/code-execution-preferences-v1'
        manifest = json.loads((preferences / 'manifest.json').read_text())
        data = preferences / 'train.jsonl'
        if (file_hash(source) != manifest['configuration']['own_checkpoint_sha256']
                or file_hash(data) != manifest['train_sha256'] or manifest['human_feedback']):
            raise ValueError('Execution preference/model provenance differs')
        rows = list(jsonl(data))
        if len(rows) != manifest['train_examples'] or any(row['feedback_source'] != 'synthetic' for row in rows):
            raise ValueError('Expected explicit execution-derived synthetic feedback')
        state = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
        tokenizer = source.parent / 'tokenizer.json'
        if file_hash(tokenizer) != state['tokenizer_sha256']:
            raise ValueError('Own tokenizer changed')
        encoder = Encoder(Tokenizer.from_file(str(tokenizer)), ModelConfig(**state['config']), preferences, 448)
        for row in rows:
            encoder.sequence(row, 'chosen'); encoder.sequence(row, 'rejected')
        if encoder.truncated:
            raise ValueError('Execution preferences would truncate source/model responses')
        del state
        implementations = capture()
        extras = {str(p.relative_to(ROOT)): file_hash(p) for p in
                  [Path(__file__), ROOT / 'src/slm_lab/code_execution_preferences.py',
                   ROOT / 'src/slm_lab/verified_code_diagnostics.py', ROOT / 'src/slm_lab/vision_transfer.py',
                   ROOT / 'src/slm_lab/objectives.py']}
        steps = 4 * len(rows)
        plan = {'base_checkpoint': str(source), 'base_checkpoint_sha256': file_hash(source),
                'preference_manifest_sha256': file_hash(preferences / 'manifest.json'),
                'data_sha256': file_hash(data), 'pairs': len(rows), 'passes': 4, 'steps': steps,
                'learning_rate': 1e-6, 'beta': .1, 'batch_size': 1, 'trainable': 'language',
                'physical_gpu': 1, 'feedback_source': 'synthetic', 'human_feedback': False,
                'runtime_and_evaluator': implementations, 'implementations': extras,
                'goal_achieved': False, 'production_promoted': False,
                'limitations': manifest['limitations'] + ['Short isolated pilot; no full nine-family retention or policy-promotion decision.']}
        atomic_json(reports / 'plan.json', plan)
        training_data = ROOT / 'data/verified-code-curriculum-v1/train.jsonl'

        def assess(checkpoint, position):
            status('assessing execution-preference pilot', position=position, family='seen source programs')
            seen_evaluate(checkpoint, training_data, reports / (position + '-seen.json'), count=32)
            torch.cuda.empty_cache()
            for mode in ['completion', 'instruction']:
                status('assessing execution-preference pilot', position=position, family='held-out local Python', mode=mode)
                code_evaluate(checkpoint, reports / (position + '-' + mode + '.json'),
                              max_new_tokens=512, languages=('Python',), prompt_mode=mode)
                torch.cuda.empty_cache()

        assess(source, 'before')
        status('training execution-derived DPO pilot', steps=steps, pairs=len(rows), feedback_source='synthetic')
        train(Namespace(stage='dpo', config='configs/model.json', tokenizer=str(tokenizer), data=str(data),
                        output=str(checkpoints), checkpoint=str(source), reward_checkpoint=None, device='cuda',
                        steps=steps, batch_size=1, gradient_accumulation=1, learning_rate=1e-6, seed=42,
                        threads=1, beta=.1, kl_coef=.05, ppo_epochs=2, max_new_tokens=16, log_every=10,
                        gradient_checkpointing=True, checkpoint_every=0, keep_checkpoints=1,
                        resume=False, trainable='language', image_size=448, contrastive_temperature=.07))
        torch.cuda.empty_cache()
        candidate = checkpoints / 'model.pt'
        trained = json.loads((checkpoints / 'training.json').read_text())
        if (trained['steps'] != steps or trained['data_sha256'] != plan['data_sha256']
                or trained['truncated_sequences'] or trained['feedback_sources'] != ['synthetic']):
            raise ValueError('Execution pilot completion/provenance differs')
        a = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
        b = torch.load(candidate, map_location='cpu', weights_only=True, mmap=True)
        if a['config'] != b['config'] or any(not torch.equal(value, b['model'][key]) for key, value in a['model'].items() if key.startswith('vision.')):
            raise ValueError('Language-only code pilot changed vision architecture/weights')
        del a, b
        # Audit the completed redundant optimizer/reference state before removal.
        resume_state = checkpoints / 'training-state.pt'
        a = torch.load(resume_state, map_location='cpu', weights_only=True, mmap=True)
        b = torch.load(candidate, map_location='cpu', weights_only=True, mmap=True)
        if a['steps'] != steps or a['planned_steps'] != steps or any(not torch.equal(value, b['model'][key]) for key, value in a['model'].items()):
            raise ValueError('Completed optimizer state and retained model differ')
        cleanup = {'removed_path': str(resume_state), 'removed_sha256': file_hash(resume_state),
                   'removed_bytes': resume_state.stat().st_size, 'retained_model_sha256': file_hash(candidate),
                   'all_model_tensors_verified_equal': True, 'limitation': 'Exact old optimizer history cannot resume.'}
        atomic_json(reports / 'completed-state-cleanup.json', cleanup)
        del a, b
        resume_state.unlink()
        assess(candidate, 'after')
        results = {}
        for family in ['seen', 'completion', 'instruction']:
            first = json.loads((reports / ('before-' + family + '.json')).read_text())
            second = json.loads((reports / ('after-' + family + '.json')).read_text())
            for key in ['evaluation_provenance', 'tokenizer_sha256', 'inference_precision']:
                if first[key] != second[key]:
                    raise ValueError('Pilot evaluation protocol changed: ' + key)
            if family == 'seen':
                for key in ['data_sha256', 'selection', 'generation_budget']:
                    if first[key] != second[key]: raise ValueError('Seen-program protocol differs')
                if [(r['id'], r['prompt_sha256']) for r in first['results']] != [(r['id'], r['prompt_sha256']) for r in second['results']]:
                    raise ValueError('Seen-program cases differ')
                metrics = ['teacher_forced_nll', 'teacher_forced_token_accuracy', 'syntax_pass_rate', 'published_example_functional_rate']
            else:
                for key in ['suite_sha256', 'prompt_mode', 'generation_budget', 'decoding']:
                    if first[key] != second[key]: raise ValueError('Held-out coding protocol differs')
                if [(r['language'], r['task'], r['specification']) for r in first['results']] != [(r['language'], r['task'], r['specification']) for r in second['results']]:
                    raise ValueError('Held-out coding cases differ')
                metrics = ['syntax_pass_rate', 'functional_pass_at_1', 'efficiency_stress_pass_rate']
            results[family] = {key: {'before': first[key], 'after': second[key]} for key in metrics}
        if capture() != implementations or any(file_hash(ROOT / path) != digest for path, digest in extras.items()):
            raise ValueError('Execution-pilot source/runtime changed')
        atomic_json(reports / 'comparison.json', {'results': results, 'goal_achieved': False,
                    'production_promoted': False, 'vision_weights_preserved': True, 'human_feedback': False,
                    'limitations': plan['limitations']})
        status('execution-derived coding preference pilot completed; broader capability unverified')
    except Exception as error:
        status('execution-preference pilot stopped; requires inspection', error=str(error))
        raise


if __name__ == '__main__':
    main()
